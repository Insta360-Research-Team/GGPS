#!/usr/bin/env python3
"""把 360Roam 数据集原地转成 real2sim 能直接读的 COLMAP 文本格式。

输入目录约定（每个场景下）：
    <scene>/
    ├── images/                        ERP jpg（典型 6080x3040）
    ├── pose_c2w.json                  {"train": [...], "test": [...]}, 每帧 {rgb_file, transform_matrix(4x4 c2w)}
    └── scene.ply | <scene>.ply        点云（带 RGB）

转换后追加：
    <scene>/imgs                       软链 -> images
    <scene>/train.txt / test.txt       image stem
    <scene>/sparse/0/cameras.txt       dummy PINHOLE
    <scene>/sparse/0/images.txt        每帧两行（COLMAP txt 规范）
    <scene>/sparse/0/points3D.ply      软链已有 ply（带 RGB）

用法：
    python tools/convert_360roam_to_colmap.py --root /path/to/datasets/360Roam
    python tools/convert_360roam_to_colmap.py --root <path> --convention blender   # 第一次出图反了再切
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pano_colmap_utils import (
    c2w_to_qt,
    ensure_imgs_symlink,
    normalize_ply_for_3dgs,
    write_cameras_txt,
    write_images_txt,
    write_split_txt,
)

DEFAULT_SCENES = [
    "bar", "base", "cafe", "canteen", "center", "center1",
    "corridor", "innovation", "lab", "library", "office",
]


def find_scene_ply(scene_dir: str, scene_name: str) -> str:
    """360Roam 大多是 scene.ply，少数命名为 <场景名>.ply（如 library/library.ply）。"""
    candidates = [
        os.path.join(scene_dir, "scene.ply"),
        os.path.join(scene_dir, f"{scene_name}.ply"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"在 {scene_dir} 未找到 scene.ply 或 {scene_name}.ply")


def convert_scene(scene_dir: str, convention: str) -> tuple[int, int, int, int]:
    """返回 (train_n, test_n, width, height)。"""
    scene_name = os.path.basename(os.path.normpath(scene_dir))
    pose_json = os.path.join(scene_dir, "pose_c2w.json")
    if not os.path.isfile(pose_json):
        raise FileNotFoundError(f"缺 pose_c2w.json: {pose_json}")
    with open(pose_json, "r") as f:
        pose = json.load(f)

    train_list = pose.get("train", [])
    test_list = pose.get("test", [])
    if not train_list:
        raise RuntimeError(f"{pose_json} 的 train 列表为空")

    # 用第一张图 probe 分辨率（同一场景内分辨率一致）
    images_dir = os.path.join(scene_dir, "images")
    first_img_path = os.path.join(images_dir, train_list[0]["rgb_file"])
    with Image.open(first_img_path) as im:
        width, height = im.size

    # 解析 frames
    frames = []
    train_stems, test_stems = [], []
    for kind, lst, stems in [("train", train_list, train_stems), ("test", test_list, test_stems)]:
        for fr in lst:
            name = fr["rgb_file"]
            stem = Path(name).stem
            c2w = np.array(fr["transform_matrix"], dtype=np.float64)
            qvec, tvec = c2w_to_qt(c2w, convention=convention)
            frames.append({"name": name, "qvec": qvec, "tvec": tvec, "stem": stem})
            stems.append(stem)

    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    write_cameras_txt(os.path.join(sparse_dir, "cameras.txt"), width=width, height=height, camera_id=1)
    write_images_txt(os.path.join(sparse_dir, "images.txt"), frames, camera_id=1)

    # 点云：重新写一份 ply，强制带 nx/ny/nz（real2sim 的 fetchPly 要求）
    src_ply = find_scene_ply(scene_dir, scene_name)
    n_pts = normalize_ply_for_3dgs(src_ply, os.path.join(sparse_dir, "points3D.ply"))

    # train/test
    write_split_txt(os.path.join(scene_dir, "train.txt"), train_stems)
    write_split_txt(os.path.join(scene_dir, "test.txt"), test_stems)

    # imgs 软链
    ensure_imgs_symlink(scene_dir, images_subdir="images", target="imgs")

    return len(train_stems), len(test_stems), width, height, n_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="360Roam 数据集根目录")
    ap.add_argument(
        "--scenes", nargs="*", default=None,
        help="可选：只处理这些场景；默认处理全部 11 个",
    )
    ap.add_argument(
        "--convention", choices=["opencv", "blender"], default="opencv",
        help="c2w 坐标系约定。先用 opencv，若渲染上下倒/左右翻则改 blender 重跑（10秒）",
    )
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    scenes = args.scenes if args.scenes else DEFAULT_SCENES

    print(f"[360Roam→COLMAP] root={root}  convention={args.convention}")
    print(f"[360Roam→COLMAP] scenes={scenes}")
    n_ok, n_fail = 0, 0
    for s in scenes:
        scene_dir = os.path.join(root, s)
        if not os.path.isdir(scene_dir):
            print(f"  [skip] {s}: 目录不存在 {scene_dir}")
            n_fail += 1
            continue
        try:
            tr, te, w, h, npts = convert_scene(scene_dir, convention=args.convention)
            print(f"  [ok]   {s:12s}  train={tr:3d} test={te:3d}  W={w} H={h}  pts={npts}")
            n_ok += 1
        except Exception as e:
            print(f"  [fail] {s}: {e}")
            n_fail += 1

    print(f"\n完成：成功 {n_ok}，失败 {n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
