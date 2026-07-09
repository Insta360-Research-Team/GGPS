#!/usr/bin/env python3
"""把 OmniBlender 数据集原地转成 real2sim 能直接读的 COLMAP 文本格式。

输入目录约定（每个场景下）：
    <scene>/
    ├── images/                        ERP png（典型 2000x1000）
    ├── train.txt / test.txt           内容是数字索引，例如 0,4,8,...,96
    └── transform.json                 OmniBlender 自定义格式：
                                       - width / height
                                       - euler_angles (xyz, 度) 与 euler_type
                                       - frames: [{file_path, position, transform_matrix(4x4 c2w)}]

转换后追加：
    <scene>/imgs                       软链 -> images
    <scene>/sparse/0/cameras.txt       dummy PINHOLE
    <scene>/sparse/0/images.txt        每帧两行（COLMAP txt 规范）
    <scene>/sparse/0/points3D.ply      随机点云（基于相机包围盒外扩；OmniBlender 无原生点云）
    <scene>/train.txt / test.txt       重写为 image stem（OmniBlender 原文件就是数字 stem，等价）

注意：
- transform.json 里 frames 的 c2w 已经是世界坐标里的 c2w，但 顶层 euler_angles 表示
  Blender Z-up → 渲染坐标系（Y-up）的全局旋转。我们需要把它应用到每个 c2w 的 translation 与
  rotation 上，等价于把世界坐标整体旋转一下。
- 没有点云，直接基于相机包围盒撒随机点（默认 200k）。

用法：
    python tools/convert_omniblender_to_colmap.py --root /path/to/datasets/OmniBlender
    python tools/convert_omniblender_to_colmap.py --root <path> --convention blender
    python tools/convert_omniblender_to_colmap.py --root <path> --num-init-points 500000
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
    c2w_to_qt,
    ensure_imgs_symlink,
    euler_xyz_to_matrix,
    random_pcd_from_camera_centers,
    store_ply,
    write_cameras_txt,
    write_images_txt,
    write_split_txt,
)

DEFAULT_SCENES = [
    "LOU", "archiviz-flat", "barbershop", "bistro_bike", "bistro_square",
    "classroom", "fisher-hut", "lone_monk",
    "pavilion_midday_chair", "pavilion_midday_pond", "restroom",
]


def read_split_indices(path: str) -> list[str]:
    """train.txt/test.txt 一行一个数字字符串。返回 stem 列表（即 '0','4',...）。"""
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out


def convert_scene(
    scene_dir: str,
    convention: str,
    num_init_points: int,
    bbox_expand: float,
) -> tuple[int, int, int, int]:
    scene_name = os.path.basename(os.path.normpath(scene_dir))
    tjson_path = os.path.join(scene_dir, "transform.json")
    if not os.path.isfile(tjson_path):
        raise FileNotFoundError(f"缺 transform.json: {tjson_path}")
    with open(tjson_path, "r") as f:
        meta = json.load(f)

    width = int(meta["width"])
    height = int(meta["height"])

    # 顶层全局旋转：Blender Z-up -> 渲染坐标。仅当 euler_type == 'xyz' 时按 xyz 应用。
    # 这是把世界坐标整体旋转，需要 left-multiply 到每个 c2w 上：
    #   X_render = R_global @ X_blender
    #   c2w_new = T_global @ c2w  其中 T_global = [[R_global, 0],[0, 1]]
    eu = meta.get("euler_angles", [0, 0, 0])
    eu_type = meta.get("euler_type", "xyz")
    if eu_type.lower() != "xyz":
        print(f"  [warn] {scene_name} euler_type={eu_type}，当前只实现 'xyz'，按 xyz 处理")
    R_global = euler_xyz_to_matrix(eu)
    T_global = np.eye(4, dtype=np.float64)
    T_global[:3, :3] = R_global

    frames_meta = meta["frames"]
    if not frames_meta:
        raise RuntimeError(f"{tjson_path} frames 为空")

    # stem -> frame
    by_stem: dict[str, dict] = {}
    cam_centers = []
    for fr in frames_meta:
        fname = fr["file_path"]
        # OmniBlender 的 file_path 多数已是 'X.png'，少数情况可能含子路径
        name = os.path.basename(fname)
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            name = name + ".png"
        stem = Path(name).stem
        c2w = np.array(fr["transform_matrix"], dtype=np.float64)
        c2w = T_global @ c2w
        qvec, tvec = c2w_to_qt(c2w, convention=convention)
        # cam center in world = c2w[:3, 3]
        cam_centers.append(c2w[:3, 3])
        by_stem[stem] = {"name": name, "qvec": qvec, "tvec": tvec, "stem": stem}

    cam_centers = np.array(cam_centers, dtype=np.float64)

    # train/test
    src_train = os.path.join(scene_dir, "train.txt")
    src_test = os.path.join(scene_dir, "test.txt")
    if not (os.path.isfile(src_train) and os.path.isfile(src_test)):
        raise FileNotFoundError(f"{scene_dir} 下需要 train.txt 和 test.txt（OmniBlender 自带）")
    train_stems = read_split_indices(src_train)
    test_stems = read_split_indices(src_test)

    # 必须保证 stems 都在 transform.json 里
    have = set(by_stem.keys())
    miss_tr = [s for s in train_stems if s not in have]
    miss_te = [s for s in test_stems if s not in have]
    if miss_tr or miss_te:
        raise RuntimeError(
            f"{scene_name}: train/test 索引在 transform.json 中找不到对应 frame: "
            f"miss_train={miss_tr[:5]}{'...' if len(miss_tr)>5 else ''}, "
            f"miss_test={miss_te[:5]}{'...' if len(miss_te)>5 else ''}"
        )

    # 写 sparse/0
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    write_cameras_txt(os.path.join(sparse_dir, "cameras.txt"), width=width, height=height, camera_id=1)

    # images.txt 按 stem 排序写，跟 reader 内 sorted(... key=image_name) 一致便于阅读
    ordered = [by_stem[s] for s in sorted(by_stem.keys(), key=lambda x: (len(x), x))]
    write_images_txt(os.path.join(sparse_dir, "images.txt"), ordered, camera_id=1)

    # 随机点云
    xyz, rgb = random_pcd_from_camera_centers(
        cam_centers,
        num_points=num_init_points,
        expand=bbox_expand,
        seed=0,
    )
    store_ply(os.path.join(sparse_dir, "points3D.ply"), xyz, rgb)

    # 重写 train/test 为 stem（OmniBlender 原本就是数字 stem，这里等价于复制并保证换行符干净）
    write_split_txt(os.path.join(scene_dir, "train.txt"), train_stems)
    write_split_txt(os.path.join(scene_dir, "test.txt"), test_stems)

    # imgs 软链
    ensure_imgs_symlink(scene_dir, images_subdir="images", target="imgs")

    return len(train_stems), len(test_stems), width, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OmniBlender 数据集根目录")
    ap.add_argument("--scenes", nargs="*", default=None, help="可选：只处理这些场景；默认全部 11 个")
    ap.add_argument("--convention", choices=["opencv", "blender"], default="opencv",
                    help="c2w 坐标系约定。先用 opencv，渲染反了再切 blender")
    ap.add_argument("--num-init-points", type=int, default=200_000, help="随机点云点数")
    ap.add_argument("--bbox-expand", type=float, default=3.0, help="相机包围盒外扩倍数")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    scenes = args.scenes if args.scenes else DEFAULT_SCENES

    print(f"[OmniBlender→COLMAP] root={root}  convention={args.convention}  "
          f"init_pts={args.num_init_points}  bbox_expand={args.bbox_expand}")
    print(f"[OmniBlender→COLMAP] scenes={scenes}")
    n_ok, n_fail = 0, 0
    for s in scenes:
        scene_dir = os.path.join(root, s)
        if not os.path.isdir(scene_dir):
            print(f"  [skip] {s}: 目录不存在 {scene_dir}")
            n_fail += 1
            continue
        try:
            tr, te, w, h = convert_scene(
                scene_dir,
                convention=args.convention,
                num_init_points=args.num_init_points,
                bbox_expand=args.bbox_expand,
            )
            print(f"  [ok]   {s:24s}  train={tr:3d} test={te:3d}  W={w} H={h}")
            n_ok += 1
        except Exception as e:
            print(f"  [fail] {s}: {e}")
            n_fail += 1

    print(f"\n完成：成功 {n_ok}，失败 {n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
