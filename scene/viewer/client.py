import math
import time
import threading
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import viser
import viser.transforms as vtf
from scene.cameras import ViewerCam
from utils.graphics_utils import fov2focal, focal2fov


def erp_to_perspective(erp_image, fov_x, fov_y, out_w, out_h):
    """Sample a perspective view from the center of an ERP image.

    The ERP is in camera space where forward=+Z, right=+X, down=+Y (COLMAP).
    The perspective crop looks straight ahead (lon=0, lat=0).

    Args:
        erp_image: (3, H_erp, W_erp) tensor
        fov_x, fov_y: perspective field of view in radians
        out_w, out_h: output size
    Returns:
        (3, out_h, out_w) tensor
    """
    device = erp_image.device
    fx = out_w / (2.0 * math.tan(fov_x / 2.0))
    fy = out_h / (2.0 * math.tan(fov_y / 2.0))

    v, u = torch.meshgrid(
        torch.arange(out_h, device=device, dtype=torch.float32) + 0.5,
        torch.arange(out_w, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    x = (u - out_w / 2.0) / fx
    y = (v - out_h / 2.0) / fy
    z = torch.ones_like(x)
    norm = torch.sqrt(x * x + y * y + z * z)
    x, y, z = x / norm, y / norm, z / norm

    lon = torch.atan2(x, z)
    lat = torch.asin(y.clamp(-1.0, 1.0))

    grid_x = lon / math.pi                # [-1, 1]
    grid_y = lat / (math.pi / 2.0)        # [-1, 1]
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    result = F.grid_sample(
        erp_image.unsqueeze(0), grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )
    return result.squeeze(0)


class ClientThread(threading.Thread):
    def __init__(self, viewer, renderer, client: viser.ClientHandle):
        super().__init__()
        self.viewer = viewer
        self.renderer = renderer
        self.client = client

        self.render_trigger = threading.Event()

        self.last_move_time = 0

        self.last_camera = None  # store camera information

        self.state = "low"  # low or high render resolution

        self.stop_client = False  # whether stop this thread

        client.camera.up_direction = viewer.up_direction

        @client.camera.on_update
        def _(cam: viser.CameraHandle) -> None:
            with self.client.atomic():
                self.last_camera = cam
                self.state = "low"  # switch to low resolution mode when a new camera received
                self.render_trigger.set()

    def render_and_send(self):
        with self.client.atomic():
            cam = self.last_camera

            self.last_move_time = time.time()

            # get camera pose
            R = vtf.SO3(wxyz=self.client.camera.wxyz)
            R = R @ vtf.SO3.from_x_radians(np.pi)
            R = torch.tensor(R.as_matrix())
            pos = torch.tensor(self.client.camera.position, dtype=torch.float64)
            c2w = torch.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = pos

            c2w = torch.matmul(self.viewer.camera_transform, c2w)

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = torch.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            is_pano = getattr(self.viewer, 'pano_mode', None) is not None and self.viewer.pano_mode.value
            is_pano_sphere = getattr(self.viewer, 'pano_sphere_mode', None) is not None and self.viewer.pano_sphere_mode.value

            if is_pano or is_pano_sphere:
                pano_w = self.viewer.pano_width.value
                image_width = int(pano_w)
                image_height = image_width // 2
                _, jpeg_quality = self.get_render_options()
                erp_fov = 2.0 * math.atan(0.5)
                fov_x = erp_fov
                fov_y = erp_fov
                camera_type = 3
            else:
                aspect_ratio = cam.aspect
                max_res, jpeg_quality = self.get_render_options()
                image_height = max_res
                image_width = int(image_height * aspect_ratio)
                if image_width > max_res:
                    image_width = max_res
                    image_height = int(image_width / aspect_ratio)
                fov_y = cam.fov
                f = fov2focal(cam.fov, image_height)
                fov_x = focal2fov(f, image_width)
                camera_type = 1

            camera = ViewerCam(
                R=np.array(R),
                T=np.array(T),
                FoVx=fov_x,
                FoVy=fov_y,
                width=image_width,
                height=image_height,
                data_device=self.viewer.device,
                camera_type=camera_type,
            )

            with torch.no_grad():
                image = self.renderer.get_outputs(camera, scaling_modifier=self.viewer.scaling_modifier.value)
                image = torch.clamp(image, max=1.)

                if is_pano_sphere:
                    persp_fov_y = cam.fov
                    # Max useful resolution: match the ERP's angular density
                    # so we never upsample beyond what the panorama provides.
                    erp_w, erp_h = image_width, image_height
                    max_useful_w = int(persp_fov_y * cam.aspect * erp_h / math.pi)
                    max_useful_h = int(persp_fov_y * erp_h / math.pi)

                    max_res, _ = self.get_render_options()
                    persp_h = min(max_useful_h, max_res)
                    persp_w = int(cam.aspect * persp_h)
                    if persp_w > min(max_useful_w, max_res):
                        persp_w = min(max_useful_w, max_res)
                        persp_h = int(persp_w / cam.aspect)
                    persp_w = max(persp_w, 64)
                    persp_h = max(persp_h, 64)

                    fy = fov2focal(persp_fov_y, persp_h)
                    persp_fov_x = focal2fov(fy, persp_w)
                    image = erp_to_perspective(image, persp_fov_x, persp_fov_y, persp_w, persp_h)

                image = torch.permute(image, (1, 2, 0))
                self.client.set_background_image(
                    image.cpu().numpy(),
                    format=self.viewer.image_format,
                    jpeg_quality=jpeg_quality,
                )

    def run(self):
        while True:
            trigger_wait_return = self.render_trigger.wait(0.2)  # TODO: avoid wasting CPU
            # stop client thread?
            if self.stop_client is True:
                break
            if not trigger_wait_return:
                # skip if camera is none
                if self.last_camera is None:
                    continue

                # if we haven't received a trigger in a while, switch to high resolution
                if self.state == "low":
                    self.state = "high"  # switch to high resolution mode
                else:
                    continue  # skip if already in high resolution mode

            self.render_trigger.clear()

            try:
                self.render_and_send()
            except Exception as err:
                print("error occurred when rendering for client")
                traceback.print_exc()
                break

        self._destroy()

    def get_render_options(self):
        if self.state == "low":
            return self.viewer.max_res_when_moving.value, int(self.viewer.jpeg_quality_when_moving.value)
        return self.viewer.max_res_when_static.value, int(self.viewer.jpeg_quality_when_static.value)

    def stop(self):
        self.stop_client = True
        # self.render_trigger.set()  # TODO: potential thread leakage?

    def _destroy(self):
        print("client thread #{} destroyed".format(self.client.client_id))
        self.viewer = None
        self.renderer = None
        self.client = None
        self.last_camera = None
