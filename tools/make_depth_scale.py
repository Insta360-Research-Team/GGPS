import argparse
import json
import os
import sys

import cv2
import numpy as np
from joblib import Parallel, delayed

# 直接导入 colmap_loader 模块，绕过 scene/__init__.py 的重依赖链
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scene"))
from colmap_loader import (  # noqa: E402
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_next_bytes,
)


def _read_colmap(base_dir: str, model_type: str):
    sparse_dir = os.path.join(base_dir, "sparse", "0")
    if model_type == "bin":
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
        image_metas = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    elif model_type == "txt":
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
        image_metas = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
    else:
        raise ValueError(f"Unknown model_type={model_type}, expected bin or txt")
    return cam_intrinsics, image_metas


def _read_points3d_ordered(base_dir: str):
    sparse_dir = os.path.join(base_dir, "sparse", "0")
    pts_path = os.path.join(sparse_dir, "points3D.bin")
    if not os.path.exists(pts_path):
        pts_path = os.path.join(sparse_dir, "points3D.txt")
    if not os.path.exists(pts_path):
        raise FileNotFoundError("Cannot find points3D.bin or points3D.txt")

    if pts_path.endswith(".bin"):
        with open(pts_path, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            ids = np.empty((num_points,), dtype=np.int64)
            xyzs = np.empty((num_points, 3), dtype=np.float64)
            for i in range(num_points):
                row = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
                ids[i] = row[0]
                xyzs[i] = np.array(row[1:4], dtype=np.float64)
                track_len = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
                fid.seek(8 * track_len, 1)  # skip (image_id, point2D_idx)
        points3d_ordered = np.zeros((int(ids.max()) + 1, 3), dtype=np.float64)
        points3d_ordered[ids] = xyzs
    else:
        ids = []
        xyzs = []
        with open(pts_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                elems = line.split()
                ids.append(int(elems[0]))
                xyzs.append([float(elems[1]), float(elems[2]), float(elems[3])])
        ids = np.array(ids, dtype=np.int64)
        xyzs = np.array(xyzs, dtype=np.float64)
        points3d_ordered = np.zeros((int(ids.max()) + 1, 3), dtype=np.float64)
        points3d_ordered[ids] = xyzs
    return points3d_ordered


def _get_depth_for_image(depths_dir: str, image_name: str):
    stem = os.path.splitext(image_name)[0]
    depth_path = os.path.join(depths_dir, f"{stem}.png")
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.ndim != 2:
        depth = depth[..., 0]
    return depth.astype(np.float32) / float(2**16)


def _get_scale_offset(image_key, cam_intrinsics, image_metas, points3d_ordered, depths_dir, camera_type=1):
    image_meta = image_metas[image_key]
    cam_intrinsic = cam_intrinsics[image_meta.camera_id]

    pts_idx = image_meta.point3D_ids
    valid_mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
    pts_idx = pts_idx[valid_mask]
    valid_xys = image_meta.xys[valid_mask]

    if len(pts_idx) <= 10:
        return None

    pts = points3d_ordered[pts_idx]
    R = qvec2rotmat(image_meta.qvec)
    pts_cam = np.dot(pts, R.T) + image_meta.tvec
    if camera_type == 3:
        # ERP/LonLat: rasterizer uses radial distance r = sqrt(x²+y²+z²)
        r = np.sqrt(np.sum(pts_cam ** 2, axis=-1))
        inv_colmap = 1.0 / np.clip(r, 1e-8, None)
    else:
        inv_colmap = 1.0 / np.clip(pts_cam[..., 2], 1e-8, None)

    inv_mono_map = _get_depth_for_image(depths_dir, image_meta.name)
    if inv_mono_map is None:
        return None

    s = inv_mono_map.shape[0] / float(cam_intrinsic.height)
    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0)
        & (maps[..., 1] >= 0)
        & (maps[..., 0] < cam_intrinsic.width * s)
        & (maps[..., 1] < cam_intrinsic.height * s)
        & (inv_colmap > 0)
    )
    if valid.sum() <= 10 or (inv_colmap.max() - inv_colmap.min()) <= 1e-3:
        return {"image_name": os.path.splitext(image_meta.name)[0], "scale": 0.0, "offset": 0.0}

    maps = maps[valid]
    inv_colmap = inv_colmap[valid]
    inv_mono = cv2.remap(
        inv_mono_map,
        maps[..., 0],
        maps[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )[..., 0]

    t_colmap = np.median(inv_colmap)
    s_colmap = np.mean(np.abs(inv_colmap - t_colmap))
    t_mono = np.median(inv_mono)
    s_mono = np.mean(np.abs(inv_mono - t_mono))
    if s_mono <= 1e-8:
        scale, offset = 0.0, 0.0
    else:
        scale = float(s_colmap / s_mono)
        offset = float(t_colmap - t_mono * scale)

    return {"image_name": os.path.splitext(image_meta.name)[0], "scale": scale, "offset": offset}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="scene root containing sparse/0")
    parser.add_argument("--depths_dir", required=True, help="directory containing inverse depth png")
    parser.add_argument("--model_type", default="bin", choices=["bin", "txt"])
    parser.add_argument("--camera_type", type=int, default=1, choices=[1, 3],
                        help="1=pinhole (z-depth), 3=ERP/LonLat (radial distance)")
    args = parser.parse_args()

    cam_intrinsics, image_metas = _read_colmap(args.base_dir, args.model_type)
    points3d_ordered = _read_points3d_ordered(args.base_dir)

    if args.camera_type == 3:
        print("[make_depth_scale] camera_type=3 (ERP): using radial distance for alignment")

    keys = list(image_metas.keys())
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_get_scale_offset)(
            key,
            cam_intrinsics,
            image_metas,
            points3d_ordered,
            args.depths_dir,
            camera_type=args.camera_type,
        )
        for key in keys
    )

    depth_params = {}
    scales = []
    for item in results:
        if item is None:
            continue
        depth_params[item["image_name"]] = {
            "scale": item["scale"],
            "offset": item["offset"],
        }
        if item["scale"] > 0:
            scales.append(item["scale"])

    med_scale = float(np.median(scales)) if scales else 0.0
    for v in depth_params.values():
        v["med_scale"] = med_scale

    n_filtered = sum(1 for v in depth_params.values()
                     if v["scale"] > 0 and (v["scale"] < 0.2 * med_scale or v["scale"] > 5.0 * med_scale))
    print(f"[make_depth_scale] med_scale={med_scale:.4f}, "
          f"filtered={n_filtered}/{len(depth_params)} frames")

    out_path = os.path.join(args.base_dir, "sparse", "0", "depth_params.json")
    with open(out_path, "w") as f:
        json.dump(depth_params, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
