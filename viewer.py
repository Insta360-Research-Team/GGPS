import os
from pathlib import Path
import math
import glob
import time
import json
import yaml
import argparse
from typing import Tuple, Literal, List

import numpy as np
import viser
import viser.transforms as vtf
import torch
from gaussian_renderer import render_viewer
from utils.general_utils import parse_cfg
from utils.graphics_utils import fov2focal, focal2fov
from scene.cameras import ViewerCam
from scene.viewer import ClientThread, ViewerRenderer
from scene.viewer.ui import populate_render_tab, TransformPanel, EditPanel

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"


class Viewer:
    def __init__(
            self,
            model_path: str,
            host: str = "0.0.0.0",
            port: int = 8080,
            background_color: Tuple = (0, 0, 0),
            image_format: Literal["jpeg", "png"] = "jpeg",
            reorient: Literal["auto", "enable", "disable"] = "auto",
            sh_degree: int = 3,
            enable_transform: bool = False,
            show_cameras: bool = False,
            cameras_json: str = None,
    ):
        self.device = torch.device("cuda")

        self.model_path = model_path
        self.host = host
        self.port = port
        self.background_color = background_color
        self.image_format = image_format
        self.sh_degree = sh_degree
        self.enable_transform = enable_transform
        self.show_cameras = show_cameras

        self.up_direction = np.asarray([0., 0., 1.])

        load_from = self._search_load_file(model_path)

        self.simplified_model = True
        self.show_edit_panel = True
        self.show_render_panel = True

        # TODO: load multiple models more elegantly
        # load and create models
        model, renderer, training_output_base_dir, dataset_type, self.checkpoint = self._load_model_from_file(load_from)

        def get_load_iteration() -> int:
            return int(os.path.basename(os.path.dirname(load_from)).replace("iteration_", ""))

        # reorient the scene
        cameras_json_path = cameras_json
        if cameras_json_path is None:
            cameras_json_path = os.path.join(training_output_base_dir, "cameras.json")
        self.camera_transform = self._reorient(cameras_json_path, mode=reorient, dataset_type=dataset_type)
        # load camera poses
        self.camera_poses = self.load_camera_poses(cameras_json_path)

        self.available_appearance_options = None

        self.loaded_model_count = 1

        self.gaussian_model = model
        # create renderer
        self.viewer_renderer = ViewerRenderer(
            model,
            render_viewer,
            torch.tensor(background_color, dtype=torch.float, device=self.device),
        )

        self.clients = {}

    @staticmethod
    def _search_load_file(model_path: str) -> str:
        # if a directory path is provided, auto search checkpoint or ply
        if os.path.isdir(model_path) is False:
            return model_path
        # search checkpoint
        checkpoint_dir = os.path.join(model_path, "checkpoints")
        # find checkpoint with max iterations
        load_from = None
        previous_checkpoint_iteration = -1
        for i in glob.glob(os.path.join(checkpoint_dir, "*.ckpt")):
            try:
                checkpoint_iteration = int(i[i.rfind("=") + 1:i.rfind(".")])
            except Exception as err:
                print("error occurred when parsing iteration from {}: {}".format(i, err))
                continue
            if checkpoint_iteration > previous_checkpoint_iteration:
                previous_checkpoint_iteration = checkpoint_iteration
                load_from = i

        # not a checkpoint can be found, search point cloud
        if load_from is None:
            previous_point_cloud_iteration = -1
            for i in glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")):
                try:
                    point_cloud_iteration = int(os.path.basename(i).replace("iteration_", ""))
                except Exception as err:
                    print("error occurred when parsing iteration from {}: {}".format(i, err))
                    continue

                ply_path = os.path.join(i, "point_cloud.ply")
                if not os.path.isfile(ply_path):
                    continue

                if point_cloud_iteration > previous_point_cloud_iteration:
                    previous_point_cloud_iteration = point_cloud_iteration
                    load_from = ply_path

        assert load_from is not None, "not a checkpoint or point cloud can be found"

        return load_from

    def _reorient(self, cameras_json_path: str, mode: str, dataset_type: str = None):
        transform = torch.eye(4, dtype=torch.float)

        if mode == "disable":
            return transform

        # detect whether cameras.json exists
        is_cameras_json_exists = os.path.exists(cameras_json_path)

        if is_cameras_json_exists is False:
            if mode == "enable":
                raise RuntimeError("{} not exists".format(cameras_json_path))
            else:
                return transform

        # skip reorient if dataset type is blender
        if dataset_type in ["blender", "nsvf"] and mode == "auto":
            print("skip reorient for {} dataset".format(dataset_type))
            return transform

        print("load {}".format(cameras_json_path))
        with open(cameras_json_path, "r") as f:
            cameras = json.load(f)
        up = torch.zeros(3)
        for i in cameras:
            up += torch.tensor(i["rotation"])[:3, 1]
        up = -up / torch.linalg.norm(up)

        print("up vector = {}".format(up))
        self.up_direction = up.numpy()

        return transform

        # rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        # transform[:3, :3] = rotation
        # transform = torch.linalg.inv(transform)
        #
        # return transform

    def load_camera_poses(self, cameras_json_path: str):
        if os.path.exists(cameras_json_path) is False:
            return []
        with open(cameras_json_path, "r") as f:
            return json.load(f)

    def _compute_camera_viser_poses(self):
        """Pre-compute viser-space (wxyz, position) for every camera in self.camera_poses."""
        camera_pose_transform = np.linalg.inv(self.camera_transform.cpu().numpy())
        self._viser_cam_poses = []
        for camera in self.camera_poses:
            c2w = np.eye(4)
            c2w[:3, :3] = np.asarray(camera["rotation"])
            c2w[:3, 3] = np.asarray(camera["position"])
            c2w[:3, 1:3] *= -1
            c2w = np.matmul(camera_pose_transform, c2w)
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            self._viser_cam_poses.append((R.wxyz, c2w[:3, 3].copy()))

        self._build_panorama_poses()

    def _build_panorama_poses(self):
        """Group cubemap faces by panorama ID and build unique panoramic poses."""
        from collections import OrderedDict
        groups = OrderedDict()
        cubemap_suffixes = {"px", "py", "pz", "nx", "ny", "nz"}
        for i, cam in enumerate(self.camera_poses):
            name = cam["img_name"]
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in cubemap_suffixes:
                pano_id = parts[0]
            else:
                pano_id = name
            if pano_id not in groups:
                groups[pano_id] = []
            groups[pano_id].append(i)

        # sort panoramas by ID so the trajectory follows capture order
        try:
            sorted_keys = sorted(groups.keys(), key=lambda k: int(k))
        except ValueError:
            sorted_keys = sorted(groups.keys())
        self._pano_names = sorted_keys
        self._pano_face_indices = [groups[k] for k in sorted_keys]
        # use the first face's position (all faces share the same position)
        # and the pz face's rotation as "forward" if available, else first face
        self._pano_viser_poses = []
        for pano_id, face_indices in zip(self._pano_names, self._pano_face_indices):
            pos = self._viser_cam_poses[face_indices[0]][1]
            rot_idx = face_indices[0]
            for fi in face_indices:
                if self.camera_poses[fi]["img_name"].endswith("_pz"):
                    rot_idx = fi
                    break
            wxyz = self._viser_cam_poses[rot_idx][0]
            self._pano_viser_poses.append((wxyz, pos))

        print(f"Found {len(self._pano_names)} panoramic positions "
              f"(from {len(self.camera_poses)} perspective cameras)")

    def _goto_camera(self, idx: int):
        """Move every connected client to camera *idx*."""
        if not self._viser_cam_poses:
            return
        idx = idx % len(self._viser_cam_poses)
        wxyz, position = self._viser_cam_poses[idx]
        self._apply_camera_pose(wxyz, position)

    def _goto_panorama(self, idx: int):
        """Move every connected client to panoramic position *idx*."""
        if not self._pano_viser_poses:
            return
        idx = idx % len(self._pano_viser_poses)
        wxyz, position = self._pano_viser_poses[idx]
        self._apply_camera_pose(wxyz, position)

    def _apply_camera_pose(self, wxyz, position):
        orbit_mode = getattr(self, '_orbit_mode', None)
        is_lookaround = orbit_mode is not None and orbit_mode.value == "Look-around"

        if is_lookaround:
            R = vtf.SO3(wxyz=wxyz)
            forward = R @ np.array([0.0, 0.0, 1.0])
            look_at = position + forward * 0.01
        else:
            look_at = None

        for cid, ct in self.clients.items():
            try:
                with ct.client.atomic():
                    ct.client.camera.position = position
                    ct.client.camera.wxyz = wxyz
                    if look_at is not None:
                        ct.client.camera.look_at = look_at
            except Exception:
                pass

    def add_cameras_to_scene(self, viser_server, max_frustums: int =500):
        if len(self.camera_poses) == 0:
            return

        self.camera_handles = []

        n = len(self.camera_poses)
        if n > max_frustums:
            step = n / max_frustums
            indices = [int(i * step) for i in range(max_frustums)]
            print(f"Too many cameras ({n}), only showing {max_frustums} frustums in the scene")
        else:
            indices = list(range(n))

        for i in indices:
            camera = self.camera_poses[i]
            name = camera["img_name"]
            wxyz, position = self._viser_cam_poses[i]

            cx = camera["width"] // 2
            cy = camera["height"] // 2
            fx = camera["fx"]

            camera_handle = viser_server.add_camera_frustum(
                name="cameras/{}".format(name),
                fov=float(2 * np.arctan(cx / fx)),
                scale=0.1,
                aspect=float(cx / cy),
                wxyz=wxyz,
                position=position,
                color=(205, 25, 0),
            )

            @camera_handle.on_click
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles.append(camera_handle)

        self.camera_visible = self.show_cameras
        for h in self.camera_handles:
            h.visible = self.camera_visible

        def toggle_camera_visibility(_):
            with viser_server.atomic():
                self.camera_visible = not self.camera_visible
                for i in self.camera_handles:
                    i.visible = self.camera_visible

        with viser_server.add_gui_folder("Cameras"):
            self.toggle_camera_button = viser_server.add_gui_button("Toggle Camera Visibility")
        self.toggle_camera_button.on_click(toggle_camera_visibility)

    def add_camera_navigation(self, viser_server):
        """Add prev/next/slider controls for navigating through loaded camera poses."""
        if len(self.camera_poses) == 0:
            return

        n_persp = len(self.camera_poses)
        persp_names = [c["img_name"] for c in self.camera_poses]
        n_pano = len(self._pano_names)
        has_pano = n_pano > 0 and n_pano != n_persp

        with viser_server.add_gui_folder("Camera Navigation"):
            self._orbit_mode = viser_server.add_gui_dropdown(
                "Orbit Mode",
                options=["Look-around", "Scene Orbit"],
                initial_value="Look-around",
                hint="Look-around: rotate in place (panoramic). "
                     "Scene Orbit: orbit around the scene center.",
            )

            if has_pano:
                self._nav_mode = viser_server.add_gui_dropdown(
                    "Navigation Mode",
                    options=["Panorama", "Perspective (All)"],
                    initial_value="Panorama",
                    hint="Panorama: navigate by panoramic positions and auto-enable ERP. "
                         "Perspective (All): navigate all individual perspective cameras.",
                )

            self._cam_idx_slider = viser_server.add_gui_slider(
                "Camera Index",
                min=0,
                max=(n_pano - 1) if has_pano else (n_persp - 1),
                step=1,
                initial_value=0,
            )
            self._cam_name_text = viser_server.add_gui_text(
                "Camera Name",
                initial_value=self._pano_names[0] if has_pano else persp_names[0],
                disabled=True,
            )
            prev_btn = viser_server.add_gui_button("Prev", icon=viser.Icon.ARROW_LEFT)
            next_btn = viser_server.add_gui_button("Next", icon=viser.Icon.ARROW_RIGHT)
            goto_btn = viser_server.add_gui_button("Go To Camera", icon=viser.Icon.CAMERA)

        def _is_pano_nav():
            return has_pano and self._nav_mode.value == "Panorama"

        def _current_max():
            return n_pano if _is_pano_nav() else n_persp

        def _on_nav_mode_change(_):
            m = _current_max()
            self._cam_idx_slider.max = m - 1
            self._cam_idx_slider.value = 0
            self._traj_start.max = m - 1
            self._traj_start.value = 0
            self._traj_end.max = m - 1
            self._traj_end.value = m - 1
            if _is_pano_nav():
                self._cam_name_text.value = self._pano_names[0]
                self.pano_mode.value = True
            else:
                self._cam_name_text.value = persp_names[0]

        def _on_goto(_):
            idx = int(self._cam_idx_slider.value)
            if _is_pano_nav():
                self._cam_name_text.value = self._pano_names[idx]
                self.pano_mode.value = True
                self._goto_panorama(idx)
            else:
                self._cam_name_text.value = persp_names[idx]
                self._goto_camera(idx)

        def _on_prev(_):
            m = _current_max()
            self._cam_idx_slider.value = (int(self._cam_idx_slider.value) - 1) % m
            _on_goto(_)

        def _on_next(_):
            m = _current_max()
            self._cam_idx_slider.value = (int(self._cam_idx_slider.value) + 1) % m
            _on_goto(_)

        def _on_orbit_mode_change(_):
            idx = int(self._cam_idx_slider.value)
            if _is_pano_nav():
                self._goto_panorama(idx)
            else:
                self._goto_camera(idx)

        prev_btn.on_click(_on_prev)
        next_btn.on_click(_on_next)
        goto_btn.on_click(_on_goto)
        self._orbit_mode.on_update(_on_orbit_mode_change)
        if has_pano:
            self._nav_mode.on_update(_on_nav_mode_change)

        # ---- Trajectory Render ----
        traj_max = (n_pano - 1) if has_pano else (n_persp - 1)
        with viser_server.add_gui_folder("Trajectory Render"):
            self._traj_start = viser_server.add_gui_slider(
                "Start Frame", min=0, max=traj_max, step=1, initial_value=0,
                hint="First camera index to include in the rendered trajectory.",
            )
            self._traj_end = viser_server.add_gui_slider(
                "End Frame", min=0, max=traj_max, step=1, initial_value=traj_max,
                hint="Last camera index to include in the rendered trajectory.",
            )
            self._traj_render_mode = viser_server.add_gui_dropdown(
                "Render Mode",
                options=["ERP (Panoramic)", "Perspective", "Panoramic Sphere"],
                initial_value="ERP (Panoramic)",
                hint="ERP: full equirectangular. Perspective: pinhole camera. "
                     "Panoramic Sphere: ERP internally, then sample perspective.",
            )
            self._traj_render_width = viser_server.add_gui_slider(
                "Width", min=512, max=8192, step=128, initial_value=2048,
                hint="Output image width. For ERP, height = width / 2. "
                     "For perspective, height is derived from the camera's aspect ratio.",
            )
            self._traj_interp_frames = viser_server.add_gui_slider(
                "Interp Frames", min=0, max=120, step=1, initial_value=30,
                hint="Number of interpolated frames between consecutive cameras. "
                     "0 = only render at original camera positions (no interpolation).",
            )
            self._traj_render_fps = viser_server.add_gui_slider(
                "FPS", min=1, max=60, step=1, initial_value=30,
            )
            self._traj_render_path = viser_server.add_gui_text(
                "Save Path",
                initial_value=os.path.join(self.model_path, "trajectory.mp4"),
            )
            self._traj_render_status = viser_server.add_gui_text(
                "Status", initial_value="Idle", disabled=True,
            )
            render_traj_btn = viser_server.add_gui_button(
                "Render Trajectory", icon=viser.Icon.VIDEO,
            )

        def _on_render_trajectory(_):
            self._render_trajectory_to_video()

        render_traj_btn.on_click(_on_render_trajectory)
    
    def _build_interpolated_trajectory_viser(self, interp_frames: int, use_pano: bool = False,
                                              start_idx: int = 0, end_idx: int = -1):
        """Build a smooth trajectory by interpolating pre-computed viser poses.

        Args:
            start_idx: first camera index (inclusive) to include.
            end_idx: last camera index (inclusive). -1 means last available.

        Returns a list of (wxyz, position) tuples in viser convention.
        """
        from scipy.spatial.transform import Rotation, Slerp
        from scipy.interpolate import CubicSpline

        if use_pano and self._pano_viser_poses:
            poses = self._pano_viser_poses
        else:
            poses = self._viser_cam_poses

        if end_idx < 0:
            end_idx = len(poses) - 1
        start_idx = max(0, min(start_idx, len(poses) - 1))
        end_idx = max(start_idx, min(end_idx, len(poses) - 1))
        poses = poses[start_idx:end_idx + 1]

        n = len(poses)
        if n == 0:
            return []
        if interp_frames <= 0 or n < 2:
            return list(poses)

        positions = np.array([p[1] for p in poses])
        quats_wxyz = np.array([p[0] for p in poses])
        quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
        rotations = Rotation.from_quat(quats_xyzw)

        key_times = np.arange(n, dtype=np.float64)
        pos_spline = CubicSpline(key_times, positions, bc_type="clamped")
        rot_slerp = Slerp(key_times, rotations)

        total_frames = (n - 1) * interp_frames + n
        t_samples = np.linspace(0, n - 1, total_frames)

        interp_positions = pos_spline(t_samples)
        interp_rotations = rot_slerp(t_samples)
        interp_quats_xyzw = interp_rotations.as_quat()
        interp_quats_wxyz = interp_quats_xyzw[:, [3, 0, 1, 2]]

        return [(q, p) for q, p in zip(interp_quats_wxyz, interp_positions)]

    def _viser_pose_to_w2c(self, wxyz, position):
        """Convert a viser pose to w2c R, T — same transform as render_and_send."""
        R = vtf.SO3(wxyz=wxyz)
        R = R @ vtf.SO3.from_x_radians(np.pi)
        R = torch.tensor(R.as_matrix())
        pos = torch.tensor(position, dtype=torch.float64)
        c2w = torch.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        c2w = torch.matmul(self.camera_transform, c2w)
        c2w[:3, 1:3] *= -1
        w2c = torch.linalg.inv(c2w)
        return np.array(w2c[:3, :3]), np.array(w2c[:3, 3])

    def _render_trajectory_to_video(self):
        """Render a smooth trajectory along all loaded cameras and save as video."""
        import cv2
        from scene.viewer.client import erp_to_perspective

        if len(self.camera_poses) == 0 or len(self._viser_cam_poses) == 0:
            self._traj_render_status.value = "Error: no camera poses loaded"
            return

        mode = self._traj_render_mode.value
        out_w = int(self._traj_render_width.value)
        interp_frames = int(self._traj_interp_frames.value)
        fps = int(self._traj_render_fps.value)
        save_path = self._traj_render_path.value

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        use_pano = mode in ("ERP (Panoramic)", "Panoramic Sphere") \
            and hasattr(self, '_nav_mode') \
            and self._nav_mode.value == "Panorama" \
            and len(self._pano_viser_poses) > 0

        start_idx = int(self._traj_start.value)
        end_idx = int(self._traj_end.value)
        if start_idx > end_idx:
            self._traj_render_status.value = "Error: Start Frame > End Frame"
            return

        self._traj_render_status.value = f"Building trajectory (frames {start_idx}–{end_idx}) ..."
        trajectory = self._build_interpolated_trajectory_viser(
            interp_frames, use_pano=use_pano, start_idx=start_idx, end_idx=end_idx)
        total = len(trajectory)
        self._traj_render_status.value = f"Rendering 0/{total} ..."

        ref_cam = self.camera_poses[0]
        cam_fx = ref_cam.get("fx", ref_cam.get("focal_x", out_w / 2))
        cam_fy = ref_cam.get("fy", ref_cam.get("focal_y", cam_fx))
        cam_w = ref_cam.get("width", out_w)
        cam_h = ref_cam.get("height", out_w // 2)
        cam_aspect = cam_w / cam_h

        writer = None

        for i, (wxyz, position) in enumerate(trajectory):
            R, T = self._viser_pose_to_w2c(wxyz, position)

            if mode == "ERP (Panoramic)":
                image_width = out_w
                image_height = out_w // 2
                erp_fov = 2.0 * math.atan(0.5)
                fov_x, fov_y = erp_fov, erp_fov
                camera_type = 3
            elif mode == "Perspective":
                image_width = out_w
                image_height = int(out_w / cam_aspect)
                fov_y = focal2fov(cam_fy, cam_h)
                fov_x = focal2fov(cam_fx, cam_w)
                camera_type = 1
            else:  # Panoramic Sphere
                image_width = out_w
                image_height = out_w // 2
                erp_fov = 2.0 * math.atan(0.5)
                fov_x, fov_y = erp_fov, erp_fov
                camera_type = 3

            camera = ViewerCam(
                R=R, T=T,
                FoVx=fov_x, FoVy=fov_y,
                width=image_width, height=image_height,
                data_device=self.device,
                camera_type=camera_type,
            )

            with torch.no_grad():
                image = self.viewer_renderer.get_outputs(camera,
                            scaling_modifier=self.scaling_modifier.value)
                image = torch.clamp(image, max=1.0)

                if mode == "Panoramic Sphere":
                    persp_fov_y = focal2fov(cam_fy, cam_h)
                    persp_h = int(out_w / cam_aspect)
                    persp_w = out_w
                    fy = fov2focal(persp_fov_y, persp_h)
                    persp_fov_x = focal2fov(fy, persp_w)
                    image = erp_to_perspective(image, persp_fov_x, persp_fov_y,
                                               persp_w, persp_h)

                frame = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if writer is None:
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                writer.write(frame_bgr)

            self._traj_render_status.value = f"Rendering {i + 1}/{total} ..."

        if writer is not None:
            writer.release()
        self._traj_render_status.value = f"Done! {total} frames saved to {save_path}"
        print(f"[Trajectory Render] {total} frames saved to {save_path}")

    @staticmethod
    def _do_initialize_models_from_vq(point_cloud_path: str, sh_degree, device):
        # if simplified is True:
        #     return GaussianModelLoader.initialize_simplified_model_from_point_cloud(point_cloud_path, sh_degree, device)
        from scene.gaussian_model import GaussianModelLOD
        model = GaussianModelLOD(sh_degree=sh_degree, device=device)
        model.load_vq(point_cloud_path)
        return model, render_viewer

    @staticmethod
    def _do_initialize_models_from_point_cloud(point_cloud_path: str, sh_degree, device):
        # if simplified is True:
        #     return GaussianModelLoader.initialize_simplified_model_from_point_cloud(point_cloud_path, sh_degree, device)
        from scene.gaussian_model import GaussianModel
        model = GaussianModel(sh_degree=sh_degree)
        model.load_ply(point_cloud_path)
        return model, render_viewer

    def _initialize_models_from_point_cloud(self, point_cloud_path: str):
        return self._do_initialize_models_from_point_cloud(point_cloud_path, self.sh_degree, self.device)

    def _load_model_from_file(self, load_from: str):
        print("load model from {}".format(load_from))
        checkpoint = None
        dataset_type = None
        if load_from.endswith(".yaml") is True:
            from scene.gaussian_model import GatheredGaussian, BlockedGaussian
            with open(load_from) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
                config_name = os.path.splitext(os.path.basename(load_from))[0]
                lp, op, pp = parse_cfg(cfg, None)
                lp.model_path = os.path.join("output/", config_name) if lp.model_path == '' else lp.model_path
                if lp.aabb is None:
                    lp.aabb = np.load(os.path.join(lp.source_path, "data_partitions", f"{lp.partition_name}_aabb.npy")).tolist()
                    print(f"Use default AABB of {[round(x, 2) for x in lp.aabb]}")

                training_output_base_dir = lp.model_path
                self.sh_degree = lp.sh_degree
                
            with torch.no_grad():
                lod_gs_list = []
                for i in range(len(lp.lod_configs)):
                    pcd_path = lp.lod_configs[i]                     
                    lod_gs, renderer = self._do_initialize_models_from_vq(pcd_path, self.sh_degree, self.device)
                    lod_gs = BlockedGaussian(lod_gs, lp, compute_cov3D_python=pp.compute_cov3D_python)
                    lod_gs_list.append(lod_gs)
                
                model = lod_gs_list
                
                del lod_gs_list, lod_gs
            
        elif load_from.endswith(".ply") is True:
            model, renderer = self._initialize_models_from_point_cloud(load_from)
            training_output_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(load_from)))
            self.sh_degree = model.max_sh_degree
        else:
            raise ValueError("unsupported file {}".format(load_from))

        return model, renderer, training_output_base_dir, dataset_type, checkpoint

    def start(self):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        server.configure_theme(
            control_layout="collapsible",
            show_logo=False,
        )
        tabs = server.add_gui_tab_group()

        with tabs.add_tab("General"):
            reset_up_button = server.add_gui_button(
                "Reset up direction",
                icon=viser.Icon.ARROW_AUTOFIT_UP,
                hint="Reset the orbit up direction.",
            )

            @reset_up_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.up_direction = vtf.SO3(event.client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

            # pre-compute viser poses for all loaded cameras
            self._compute_camera_viser_poses()

            # add camera frustums to 3D scene (initially hidden unless --show_cameras)
            self.add_cameras_to_scene(server)

            # always add navigation controls (prev/next/slider)
            self.add_camera_navigation(server)

            # add render options
            with server.add_gui_folder("Render"):
                self.max_res_when_static = server.add_gui_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.jpeg_quality_when_static = server.add_gui_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.max_res_when_moving = server.add_gui_slider(
                    "Max Res when Moving",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1280,
                )
                self.jpeg_quality_when_moving = server.add_gui_slider(
                    "JPEG Quality when Moving",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                )

            with server.add_gui_folder("Panorama"):
                self.pano_mode = server.add_gui_checkbox(
                    "Panoramic Mode (ERP)",
                    initial_value=False,
                    hint="Show the full equirectangular panoramic image.",
                )
                self.pano_mode.on_update(self._handle_option_updated)
                self.pano_sphere_mode = server.add_gui_checkbox(
                    "Panoramic Sphere",
                    initial_value=False,
                    hint="Render ERP internally and sample a perspective view from it. "
                         "Gives artifact-free perspective by resampling the clean panorama.",
                )
                self.pano_sphere_mode.on_update(self._handle_option_updated)
                self.pano_width = server.add_gui_slider(
                    "Panorama Width",
                    min=512,
                    max=4096,
                    step=128,
                    initial_value=2048,
                    hint="Width of the ERP output image. Height = Width / 2.",
                )
                self.pano_width.on_update(self._handle_option_updated)

            with server.add_gui_folder("Model"):
                self.scaling_modifier = server.add_gui_slider(
                    "Scaling Modifier",
                    min=0.,
                    max=1.,
                    step=0.1,
                    initial_value=1.,
                )
                self.scaling_modifier.on_update(self._handle_option_updated)

                if self.sh_degree > 0:
                    self.active_sh_degree_slider = server.add_gui_slider(
                        "Active SH Degree",
                        min=0,
                        max=self.sh_degree,
                        step=1,
                        initial_value=self.sh_degree,
                    )
                    self.active_sh_degree_slider.on_update(self._handle_activate_sh_degree_slider_updated)

                if self.available_appearance_options is not None:
                    # find max appearance id
                    max_input_id = 0
                    available_option_values = list(self.available_appearance_options.values())
                    if isinstance(available_option_values[0], list) or isinstance(available_option_values[0], tuple):
                        for i in available_option_values:
                            if i[0] > max_input_id:
                                max_input_id = i[0]
                    else:
                        # convert to tuple, compatible with previous version
                        for i in self.available_appearance_options:
                            self.available_appearance_options[i] = (0, self.available_appearance_options[i])
                    self.available_appearance_options[DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE] = None

                    self.appearance_id = server.add_gui_slider(
                        "Appearance Direct",
                        min=0,
                        max=max_input_id,
                        step=1,
                        initial_value=0,
                        visible=max_input_id > 0
                    )

                    self.normalized_appearance_id = server.add_gui_slider(
                        "Normalized Appearance Direct",
                        min=0.,
                        max=1.,
                        step=0.01,
                        initial_value=0.,
                    )

                    appearance_options = list(self.available_appearance_options.keys())

                    self.appearance_group_dropdown = server.add_gui_dropdown(
                        "Appearance Group",
                        options=appearance_options,
                        initial_value=appearance_options[0],
                    )
                    self.appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.normalized_appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.appearance_group_dropdown.on_update(self._handel_appearance_group_dropdown_updated)

                self.time_slider = server.add_gui_slider(
                    "Time",
                    min=0.,
                    max=1.,
                    step=0.01,
                    initial_value=0.,
                )
                self.time_slider.on_update(self._handle_option_updated)

        if self.show_edit_panel is True:
            with tabs.add_tab("Edit") as edit_tab:
                self.edit_panel = EditPanel(server, self, edit_tab)

        self.transform_panel: TransformPanel = None
        if self.enable_transform is True:
            with tabs.add_tab("Transform"):
                self.transform_panel = TransformPanel(server, self, self.loaded_model_count)

        if self.show_render_panel is True:
            with tabs.add_tab("Render"):
                populate_render_tab(
                    server,
                    self,
                    self.model_path,
                    Path("./"),
                    orientation_transform=torch.linalg.inv(self.camera_transform).cpu().numpy(),
                    enable_transform=self.enable_transform,
                    background_color=self.background_color,
                    sh_degree=self.sh_degree,
                )

        # register client hooks AFTER all GUI elements are created
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        while True:
            time.sleep(999)

    def _handle_appearance_embedding_slider_updated(self, event: viser.GuiEvent):
        """
        Change appearance group dropdown to "@Direct" on slider updated
        """

        if event.client is None:  # skip if not updated by client
            return
        self.appearance_group_dropdown.value = DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE
        self._handle_option_updated(event)

    def _handle_activate_sh_degree_slider_updated(self, _):
        self.viewer_renderer.gaussian_model.active_sh_degree = self.active_sh_degree_slider.value
        self._handle_option_updated(_)

    def get_appearance_id_value(self):
        """
        Return appearance id according to the slider and dropdown value
        """

        # no available appearance options, simply return zero
        if self.available_appearance_options is None:
            return (0, 0.)
        name = self.appearance_group_dropdown.value
        # if the value of dropdown is "@Direct", or not in available_appearance_options, return the slider's values
        if name == DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE or name not in self.available_appearance_options:
            return (self.appearance_id.value, self.normalized_appearance_id.value)
        # else return the values according to the dropdown
        return self.available_appearance_options[name]

    def _handel_appearance_group_dropdown_updated(self, event: viser.GuiEvent):
        """
        Update slider's values when dropdown updated
        """

        if event.client is None:  # skip if not updated by client
            return

        # get appearance ids according to the dropdown value
        appearance_id, normalized_appearance_id = self.available_appearance_options[self.appearance_group_dropdown.value]
        # update sliders
        self.appearance_id.value = appearance_id
        self.normalized_appearance_id.value = normalized_appearance_id
        # rerender
        self._handle_option_updated(event)

    def _handle_option_updated(self, _):
        """
        Simply push new render to all client
        """
        return self.rerender_for_all_client()

    def handle_option_updated(self, _):
        return self._handle_option_updated(_)

    def rerender_for_client(self, client_id: int):
        """
        Render for specific client
        """
        try:
            # switch to low resolution mode first, then notify the client to render
            self.clients[client_id].state = "low"
            self.clients[client_id].render_trigger.set()
        except:
            # ignore errors
            pass

    def rerender_for_all_client(self):
        for i in self.clients:
            self.rerender_for_client(i)

    def _handle_new_client(self, client: viser.ClientHandle) -> None:
        """
        Create and start a thread for every new client
        """

        # create client thread
        client_thread = ClientThread(self, self.viewer_renderer, client)
        client_thread.start()
        # store this thread
        self.clients[client.client_id] = client_thread

    def _handle_client_disconnect(self, client: viser.ClientHandle):
        """
        Destroy client thread when client disconnected
        """

        try:
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as err:
            print(err)


if __name__ == "__main__":
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--background_color", "--background_color", "--bkg_color", "-b",
                        type=str, nargs="+", default=["black"],
                        help="e.g.: white, black, 0 0 0, 1 1 1")
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", "-r", type=str, default="auto",
                        help="whether reorient the scene, available values: auto, enable, disable")
    parser.add_argument("--sh_degree", "--sh-degree", "--sh",
                        type=int, default=3)
    parser.add_argument("--enable_transform", "--enable-transform",
                        action="store_true", default=False,
                        help="Enable transform options on Web UI. May consume more memory")
    parser.add_argument("--show_cameras", "--show-cameras",
                        action="store_true")
    parser.add_argument("--cameras-json", "--cameras_json", type=str, default=None)
    args = parser.parse_args()

    # arguments post process
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = (1., 1., 1.)
        else:
            args.background_color = (0., 0., 0.)
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args)}
    viewer = Viewer(**viewer_init_args)

    # start viewer server
    viewer.start()
