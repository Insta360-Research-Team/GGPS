#!/usr/bin/env python3
"""把 openMVG 输出的场景目录转成 panorama_large 能直接读的 COLMAP 文本格式。

支持的输入目录结构（openMVG 标准重建输出）：
    <scene>/
    ├── images/                         ERP 全景 jpg/png
    ├── matches/                        openMVG 特征与匹配（不使用）
    └── reconstruction/
        ├── sfm_data.bin                openMVG 二进制（不直接读）
        ├── sfm_data_full.json          openMVG JSON 导出（主要输入）
        └── colorized.ply              着色稀疏点云

转换后追加：
    <scene>/imgs                        软链 -> images
    <scene>/sparse/0/cameras.txt        dummy PINHOLE（ERP 通道忽略内参）
    <scene>/sparse/0/images.txt         每帧两行（含 openMVG structure 的 2D 观测）
    <scene>/sparse/0/points3D.txt       来自 openMVG structure（含 track，深度对齐用）
    <scene>/sparse/0/points3D.ply       来自 reconstruction/colorized.ply（3DGS 初始化用）
    <scene>/train.txt / test.txt        按间隔采样自动划分

openMVG sfm_data_full.json 格式要点：
- intrinsics[0].value.polymorphic_name == "spherical" → ERP 全景
- extrinsics[i].value.rotation : 3×3 list (Rcw，world→cam)
- extrinsics[i].value.center   : 3-list   (相机在世界坐标的位置 = -Rcw^T @ tcw)
- views[i].value.ptr_wrapper.data.id_pose → 对应 extrinsics 的 key

用法：
    python tools/convert_openmvg_to_colmap.py --scene /path/to/VID_SYSU1
    python tools/convert_openmvg_to_colmap.py --scene /path/to/VID_SYSU1 --llffhold 8
    python tools/convert_openmvg_to_colmap.py --scene /path/to/VID_SYSU1 --sfm-json reconstruction/sfm_data_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pano_colmap_utils import (
    ensure_imgs_symlink,
    normalize_ply_for_3dgs,
    random_pcd_from_camera_centers,
    rotmat_to_qvec,
    store_ply,
    write_cameras_txt,
    write_images_txt,
    write_points3d_txt,
    write_split_txt,
)


def parse_openmvg_sfm_data(sfm_json_path: str):
    """解析 openMVG sfm_data_full.json，返回 (width, height, frames_by_stem, cam_centers)。

    frames_by_stem: dict[stem] -> {name, qvec, tvec, stem, pose_key}
    cam_centers: np.ndarray [N, 3]
    """
    print(f"  Loading {sfm_json_path} ...")
    with open(sfm_json_path, "r") as f:
        data = json.load(f)

    views = data.get("views", [])
    intrinsics = data.get("intrinsics", [])
    extrinsics = data.get("extrinsics", [])

    if not views or not extrinsics:
        raise RuntimeError("sfm_data 中 views 或 extrinsics 为空")

    # intrinsic → 分辨率
    intr = intrinsics[0]["value"]["ptr_wrapper"]["data"]
    if "value0" in intr:
        width = int(intr["value0"]["width"])
        height = int(intr["value0"]["height"])
    else:
        width = int(intr.get("width", 0))
        height = int(intr.get("height", 0))
    if width == 0 or height == 0:
        raise RuntimeError(f"无法从 intrinsics 读取分辨率: {intr}")

    intr_type = intrinsics[0]["value"].get("polymorphic_name", "unknown")
    print(f"  Intrinsic type: {intr_type}, resolution: {width}x{height}")

    # extrinsics 按 key 建索引
    extr_by_key = {}
    for e in extrinsics:
        extr_by_key[e["key"]] = e["value"]

    # 遍历 views，匹配 extrinsic
    frames_by_stem: dict[str, dict] = {}
    cam_centers = []
    n_skip = 0

    for v in views:
        view_id = v["key"]
        vdata = v["value"]["ptr_wrapper"]["data"]
        filename = vdata["filename"]
        pose_key = vdata["id_pose"]

        if pose_key not in extr_by_key:
            n_skip += 1
            continue

        ext = extr_by_key[pose_key]
        Rcw = np.array(ext["rotation"], dtype=np.float64)  # 3×3, world→cam
        center = np.array(ext["center"], dtype=np.float64)  # cam position in world

        # COLMAP 的 tvec = -R @ center
        tvec = -Rcw @ center
        qvec = rotmat_to_qvec(Rcw)

        stem = Path(filename).stem
        frames_by_stem[stem] = {
            "name": filename,
            "qvec": qvec,
            "tvec": tvec,
            "stem": stem,
            "pose_key": pose_key,
            "view_id": view_id,
        }
        cam_centers.append(center)

    if n_skip:
        print(f"  {n_skip} views 无对应 extrinsic，已跳过")

    cam_centers = np.array(cam_centers, dtype=np.float64)
    print(f"  有效帧数: {len(frames_by_stem)}, 3D structure: {len(data.get('structure', []))}")

    return width, height, frames_by_stem, cam_centers, data


def extract_openmvg_structure(
    sfm_data: dict,
    view_id_to_image_id: dict[int, int],
) -> tuple[np.ndarray, np.ndarray | None,
           dict[int, list[tuple[float, float, int]]],
           dict[int, list[tuple[int, int]]]]:
    """从 openMVG sfm_data 的 structure 中提取 3D 点和 2D-3D 对应关系。

    Returns:
        pts_xyz:      (N, 3) 3D 坐标
        pts_rgb:      None（structure 中一般不含颜色）
        observations: {image_id: [(px, py, point3d_id), ...]}  给 images.txt
        tracks:       {point3d_id: [(image_id, point2d_idx), ...]}  给 points3D.txt
    """
    structure = sfm_data.get("structure", [])
    if not structure:
        return np.empty((0, 3)), None, {}, {}

    pts_xyz = np.empty((len(structure), 3), dtype=np.float64)
    observations: dict[int, list[tuple[float, float, int]]] = {}
    tracks: dict[int, list[tuple[int, int]]] = {}

    for new_id_0based, s in enumerate(structure):
        pid = new_id_0based + 1  # COLMAP POINT3D_ID 从 1 开始
        pts_xyz[new_id_0based] = s["value"]["X"]

        for obs in s["value"]["observations"]:
            view_id = obs["key"]
            if view_id not in view_id_to_image_id:
                continue
            image_id = view_id_to_image_id[view_id]
            px, py = obs["value"]["x"]

            if image_id not in observations:
                observations[image_id] = []
            p2d_idx = len(observations[image_id])
            observations[image_id].append((px, py, pid))
            tracks.setdefault(pid, []).append((image_id, p2d_idx))

    total_obs = sum(len(v) for v in observations.values())
    n_images_with_obs = sum(1 for v in observations.values() if v)
    print(f"  openMVG structure: {len(structure)} 个 3D 点, "
          f"{total_obs} 条观测, 覆盖 {n_images_with_obs} 张图像")
    return pts_xyz, None, observations, tracks


def convert_scene(
    scene_dir: str,
    sfm_json_rel: str = "reconstruction/sfm_data_full.json",
    llffhold: int = 8,
    images_subdir: str = "images",
    ply_source: str = "reconstruction/colorized.ply",
    num_fallback_points: int = 200_000,
) -> dict:
    """转换单个 openMVG 场景。"""
    scene_name = os.path.basename(os.path.normpath(scene_dir))
    sfm_json_path = os.path.join(scene_dir, sfm_json_rel)

    if not os.path.isfile(sfm_json_path):
        raise FileNotFoundError(f"找不到 sfm_data JSON: {sfm_json_path}")

    width, height, frames_by_stem, cam_centers, sfm_data = parse_openmvg_sfm_data(sfm_json_path)

    # 按 stem 排序（数字字符串自然排序）
    sorted_stems = sorted(frames_by_stem.keys(), key=lambda x: (len(x), x))
    ordered_frames = [frames_by_stem[s] for s in sorted_stems]

    # train/test 划分：每 llffhold 帧取最后 1 帧做 test
    # 例如 llffhold=8 → 帧 1-7 train, 帧 8 test, 帧 9-15 train, 帧 16 test ...
    train_stems = []
    test_stems = []
    for i, stem in enumerate(sorted_stems):
        if (i + 1) % llffhold == 0:
            test_stems.append(stem)
        else:
            train_stems.append(stem)

    # 写 sparse/0
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    write_cameras_txt(os.path.join(sparse_dir, "cameras.txt"), width=width, height=height, camera_id=1)

    # 从 openMVG structure 中提取真实的 2D-3D 特征匹配对应关系
    view_id_to_image_id = {}
    stem_to_image_id = {stem: i + 1 for i, stem in enumerate(sorted_stems)}
    for stem, fr in frames_by_stem.items():
        view_id_to_image_id[fr["view_id"]] = stem_to_image_id[stem]

    struct_xyz, _, observations, tracks = extract_openmvg_structure(
        sfm_data, view_id_to_image_id,
    )

    # points3D.ply（3DGS 初始化用）：优先用 colorized.ply，否则用 structure 点
    dst_ply = os.path.join(sparse_dir, "points3D.ply")
    src_ply = os.path.join(scene_dir, ply_source)
    if os.path.isfile(src_ply):
        n_pts = normalize_ply_for_3dgs(src_ply, dst_ply)
        print(f"  点云 PLY: {src_ply} -> {dst_ply} ({n_pts} pts)")
    elif len(struct_xyz) > 0:
        store_ply(dst_ply, struct_xyz.astype(np.float32),
                  np.full((len(struct_xyz), 3), 200, dtype=np.uint8))
        print(f"  点云 PLY: 从 structure 生成 ({len(struct_xyz)} pts)")
    else:
        print(f"  [warn] 无 PLY 且无 structure，生成随机点云 ({num_fallback_points} pts)")
        xyz, rgb = random_pcd_from_camera_centers(cam_centers, num_points=num_fallback_points)
        store_ply(dst_ply, xyz, rgb)

    # points3D.txt（深度对齐用）：来自 structure 的真实特征匹配点
    if len(struct_xyz) > 0:
        write_points3d_txt(
            os.path.join(sparse_dir, "points3D.txt"),
            struct_xyz, tracks=tracks,
        )

    # images.txt：包含 structure 的 2D 观测
    write_images_txt(
        os.path.join(sparse_dir, "images.txt"),
        ordered_frames, camera_id=1, observations=observations,
    )

    # train/test
    write_split_txt(os.path.join(scene_dir, "train.txt"), train_stems)
    write_split_txt(os.path.join(scene_dir, "test.txt"), test_stems)

    # imgs 软链
    ensure_imgs_symlink(scene_dir, images_subdir=images_subdir, target="imgs")

    return {
        "scene": scene_name,
        "width": width,
        "height": height,
        "train": len(train_stems),
        "test": len(test_stems),
        "total_frames": len(ordered_frames),
    }


def main():
    ap = argparse.ArgumentParser(description="openMVG sfm_data → COLMAP text for panorama_large")
    ap.add_argument("--scene", required=True, help="场景根目录（含 images/ 和 reconstruction/）")
    ap.add_argument("--sfm-json", default="reconstruction/sfm_data_full.json",
                    help="sfm_data JSON 相对场景目录的路径")
    ap.add_argument("--llffhold", type=int, default=8,
                    help="每 N 帧取 1 帧做 test（默认 8）")
    ap.add_argument("--images-subdir", default="images",
                    help="图像子目录名（默认 images）")
    ap.add_argument("--ply-source", default="reconstruction/colorized.ply",
                    help="稀疏点云 ply 相对路径")
    ap.add_argument("--num-fallback-points", type=int, default=200_000,
                    help="无 ply 时随机点数")
    args = ap.parse_args()

    scene_dir = os.path.abspath(args.scene)
    if not os.path.isdir(scene_dir):
        print(f"错误：目录不存在 {scene_dir}")
        sys.exit(1)

    print(f"[openMVG → COLMAP] scene={scene_dir}")
    result = convert_scene(
        scene_dir,
        sfm_json_rel=args.sfm_json,
        llffhold=args.llffhold,
        images_subdir=args.images_subdir,
        ply_source=args.ply_source,
        num_fallback_points=args.num_fallback_points,
    )
    print(f"\n完成: {result['scene']}  {result['width']}x{result['height']}  "
          f"train={result['train']}  test={result['test']}  total={result['total_frames']}")


if __name__ == "__main__":
    main()
