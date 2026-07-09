#!/usr/bin/env python3
"""
Web-based real-time viewer for 3DGS training (network_gui client).

Connects to the training process TCP socket and provides a browser-based
free-viewpoint rendering interface via viser.

Usage:
    python train_viewer.py [--train_port 6009] [--web_port 8080] [--resolution 800]

Then open http://<host>:8080 in your browser (use SSH port forwarding if remote).
"""

import argparse
import socket
import struct
import json
import math
import time
import threading

import numpy as np

try:
    import viser
    import viser.transforms as vtf
except ImportError:
    raise ImportError("viser is required: pip install viser")


def recvall(sock, n):
    """Receive exactly n bytes from a socket."""
    data = bytearray()
    while len(data) < n:
        pkt = sock.recv(min(n - len(data), 1 << 16))
        if not pkt:
            raise ConnectionError("Connection lost")
        data.extend(pkt)
    return bytes(data)


def get_projection_matrix(znear, zfar, fov_x, fov_y):
    """Replicate utils.graphics_utils.getProjectionMatrix (numpy version)."""
    tan_x = math.tan(fov_x * 0.5)
    tan_y = math.tan(fov_y * 0.5)
    top = tan_y * znear
    bottom = -top
    right = tan_x * znear
    left = -right

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class TrainingViewerClient:
    """Manages the TCP connection to the training network_gui server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6009):
        self.host = host
        self.port = port
        self.sock = None
        self.connected = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        with self._lock:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((self.host, self.port))
                s.settimeout(None)
                self.sock = s
                self.connected = True
                print(f"[viewer] Connected to training at {self.host}:{self.port}")
                return True
            except Exception as e:
                print(f"[viewer] Connection failed: {e}")
                self.connected = False
                return False

    def disconnect(self):
        with self._lock:
            if self.sock:
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
            self.connected = False

    def render(self, width, height, fov_x, fov_y,
               view_matrix_list, proj_matrix_list,
               znear=0.01, zfar=100.0,
               do_training=True, keep_alive=True,
               scaling_modifier=1.0, camera_type=1):
        """Send a render request and receive the image back."""
        with self._lock:
            if not self.connected:
                return None

            msg = {
                "resolution_x": int(width),
                "resolution_y": int(height),
                "fov_y": float(fov_y),
                "fov_x": float(fov_x),
                "z_near": float(znear),
                "z_far": float(zfar),
                "shs_python": False,
                "rot_scale_python": False,
                "keep_alive": keep_alive,
                "scaling_modifier": float(scaling_modifier),
                "view_matrix": view_matrix_list,
                "view_projection_matrix": proj_matrix_list,
                "train": do_training,
                "camera_type": int(camera_type),
            }

            try:
                raw = json.dumps(msg).encode("utf-8")
                self.sock.sendall(struct.pack("<I", len(raw)))
                self.sock.sendall(raw)

                img_bytes = recvall(self.sock, width * height * 3)
                verify_len = struct.unpack("<I", recvall(self.sock, 4))[0]
                recvall(self.sock, verify_len)

                return np.frombuffer(img_bytes, dtype=np.uint8).reshape(height, width, 3)
            except Exception as e:
                print(f"[viewer] Render error: {e}")
                self.connected = False
                return None


def viser_cam_to_network_gui(wxyz, position, fov_y, aspect,
                              znear=0.01, zfar=100.0, camera_type=1):
    """
    Convert a viser camera pose to the flat matrix lists expected by network_gui.

    The conversion follows the same logic as scene/viewer/client.py:
      viser (OpenGL: Y-up, Z-back) -> COLMAP (Y-down, Z-forward) -> 3DGS matrices.
    """
    R = vtf.SO3(wxyz=wxyz) @ vtf.SO3.from_x_radians(np.pi)
    R_mat = np.asarray(R.as_matrix(), dtype=np.float64)
    pos = np.asarray(position, dtype=np.float64)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R_mat
    c2w[:3, 3] = pos
    c2w[:3, 1:3] *= -1  # OpenGL/viser -> COLMAP convention

    w2c = np.linalg.inv(c2w).astype(np.float32)
    w2c_T = w2c.T  # transposed, matching 3DGS Camera internal storage

    if camera_type == 3:
        # ERP cameras use fixed FOV: focal2fov(W, W) = focal2fov(H, H) = 2*atan(0.5)
        fov_x = 2.0 * math.atan(0.5)
        fov_y = 2.0 * math.atan(0.5)
    else:
        fov_x = 2.0 * math.atan(math.tan(fov_y * 0.5) * aspect)

    P = get_projection_matrix(znear, zfar, fov_x, fov_y)
    full_proj = w2c_T @ P.T

    # Pre-negate columns that network_gui.receive() will flip back
    sent_view = w2c_T.copy()
    sent_view[:, 1] *= -1
    sent_view[:, 2] *= -1

    sent_proj = full_proj.copy()
    sent_proj[:, 1] *= -1

    return fov_x, fov_y, sent_view.flatten().tolist(), sent_proj.flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description="Web viewer for 3DGS training")
    parser.add_argument("--train_host", default="127.0.0.1",
                        help="Training server host (default: 127.0.0.1)")
    parser.add_argument("--train_port", type=int, default=6009,
                        help="Training server port (default: 6009)")
    parser.add_argument("--web_port", type=int, default=8080,
                        help="Web viewer port (default: 8080)")
    parser.add_argument("--resolution", type=int, default=800,
                        help="Render resolution width (default: 800)")
    parser.add_argument("--camera_type", type=int, default=1,
                        help="Camera type: 1=pinhole, 3=ERP/panoramic (default: 1)")
    args = parser.parse_args()

    znear, zfar = 0.01, 100.0

    # --- TCP client ---
    client = TrainingViewerClient(args.train_host, args.train_port)
    for attempt in range(10):
        if client.connect():
            break
        wait = min(2 ** attempt, 30)
        print(f"[viewer] Retry in {wait}s ... (attempt {attempt + 1}/10)")
        time.sleep(wait)
    else:
        print("[viewer] Could not connect. Is the training running?")
        return

    # --- Viser web server (compatible with older viser API) ---
    server = viser.ViserServer(host="0.0.0.0", port=args.web_port)
    server.configure_theme(control_layout="collapsible", show_logo=False)

    with server.add_gui_folder("Viewer Settings"):
        pause_cb = server.add_gui_checkbox(
            "Pause training (free-look)", initial_value=False
        )
        res_slider = server.add_gui_slider(
            "Resolution", min=256, max=1920, step=64, initial_value=args.resolution
        )

    do_training = [True]

    @pause_cb.on_update
    def _on_pause(_):
        do_training[0] = not pause_cb.value
        mode = "PAUSED (free-look)" if pause_cb.value else "TRAINING"
        print(f"[viewer] Mode: {mode}")

    print(f"\n{'=' * 60}")
    print(f"  Web viewer ready: http://0.0.0.0:{args.web_port}")
    print(f"  SSH forward: ssh -L {args.web_port}:127.0.0.1:{args.web_port} user@host")
    print(f"{'=' * 60}\n")

    def handle_new_client(viser_client: viser.ClientHandle):
        print(f"[viewer] Browser client connected: {viser_client.client_id}")

        latest_cam = [None]
        render_event = threading.Event()

        @viser_client.camera.on_update
        def _on_cam(_):
            latest_cam[0] = (
                viser_client.camera.wxyz.copy(),
                viser_client.camera.position.copy(),
                float(viser_client.camera.fov),
                float(viser_client.camera.aspect),
            )
            render_event.set()

        def render_loop():
            while True:
                render_event.wait(timeout=0.5)
                render_event.clear()

                cam = latest_cam[0]
                if cam is None:
                    continue

                wxyz, position, fov, aspect = cam
                if aspect < 0.01:
                    continue

                w = res_slider.value
                if args.camera_type == 3:
                    # ERP: force 2:1 aspect ratio
                    h = max(w // 2, 64)
                else:
                    h = max(int(w / aspect), 64)

                if not client.connected:
                    client.connect()
                    if not client.connected:
                        time.sleep(2)
                        continue

                fov_x, fov_y, view_list, proj_list = viser_cam_to_network_gui(
                    wxyz, position, fov, aspect, znear, zfar,
                    camera_type=args.camera_type,
                )
                img = client.render(
                    w, h, fov_x, fov_y, view_list, proj_list,
                    znear=znear, zfar=zfar,
                    do_training=do_training[0],
                    camera_type=args.camera_type,
                )
                if img is not None:
                    viser_client.set_background_image(
                        img, format="jpeg", jpeg_quality=80,
                    )

        t = threading.Thread(target=render_loop, daemon=True)
        t.start()

    server.on_client_connect(handle_new_client)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[viewer] Shutting down ...")
        client.disconnect()


if __name__ == "__main__":
    main()
