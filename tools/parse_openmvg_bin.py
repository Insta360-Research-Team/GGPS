#!/usr/bin/env python3
"""解析 openMVG sfm_data.bin (Cereal binary) 并导出 sfm_data_full.json。

当系统 GLIBC 版本不够无法运行 openMVG_main_ConvertSfM_DataFormat 时，
可用此脚本替代。

用法：
    python tools/parse_openmvg_bin.py /path/to/scene
    python tools/parse_openmvg_bin.py /path/to/scene --bin reconstruction/sfm_data.bin --out reconstruction/sfm_data_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from io import BytesIO
from typing import BinaryIO


def read_u8(f: BinaryIO) -> int:
    return struct.unpack("B", f.read(1))[0]


def read_u32(f: BinaryIO) -> int:
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f: BinaryIO) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def read_f64(f: BinaryIO) -> float:
    return struct.unpack("<d", f.read(8))[0]


def read_str(f: BinaryIO) -> str:
    n = read_u64(f)
    return f.read(n).decode("utf-8")


def read_vec_double(f: BinaryIO) -> list[float]:
    n = read_u64(f)
    return [read_f64(f) for _ in range(n)]


def read_mat_double(f: BinaryIO) -> list[list[float]]:
    n_rows = read_u64(f)
    rows = []
    for _ in range(n_rows):
        n_cols = read_u64(f)
        rows.append([read_f64(f) for _ in range(n_cols)])
    return rows


def parse_sfm_data_bin(bin_path: str) -> dict:
    """解析 openMVG sfm_data.bin，返回与 sfm_data_full.json 兼容的 dict。"""

    with open(bin_path, "rb") as fp:
        raw = fp.read()

    f = BytesIO(raw)

    _tracking = read_u8(f)

    version = read_str(f)
    root_path = read_str(f)
    print(f"  version={version}, root_path={root_path}")

    # ── Views ──
    n_views = read_u64(f)
    print(f"  parsing {n_views} views ...")
    views = []
    for i in range(n_views):
        key = read_u32(f)
        poly_id = read_u32(f)
        ptr_id = read_u32(f)
        local_path = read_str(f)
        filename = read_str(f)
        width = read_u32(f)
        height = read_u32(f)
        id_view = read_u32(f)
        id_intrinsic = read_u32(f)
        id_pose = read_u32(f)

        views.append({
            "key": key,
            "value": {
                "polymorphic_id": 1073741824,
                "ptr_wrapper": {
                    "id": ptr_id,
                    "data": {
                        "local_path": local_path,
                        "filename": filename,
                        "width": width,
                        "height": height,
                        "id_view": id_view,
                        "id_intrinsic": id_intrinsic,
                        "id_pose": id_pose,
                    }
                }
            }
        })

    # ── Intrinsics ──
    n_intrinsics = read_u64(f)
    print(f"  parsing {n_intrinsics} intrinsics ...")
    intrinsics = []
    for i in range(n_intrinsics):
        key = read_u32(f)
        poly_id = read_u32(f)

        type_name = read_str(f)
        ptr_id = read_u32(f)

        width = read_u32(f)
        height = read_u32(f)

        intrinsics.append({
            "key": key,
            "value": {
                "polymorphic_id": 2147483649,
                "polymorphic_name": type_name,
                "ptr_wrapper": {
                    "id": ptr_id,
                    "data": {
                        "value0": {
                            "width": width,
                            "height": height,
                        }
                    }
                }
            }
        })

    # ── Extrinsics ──
    n_extrinsics = read_u64(f)
    print(f"  parsing {n_extrinsics} extrinsics ...")
    extrinsics = []
    for i in range(n_extrinsics):
        key = read_u32(f)
        rotation = read_mat_double(f)
        center = read_vec_double(f)
        extrinsics.append({
            "key": key,
            "value": {
                "rotation": rotation,
                "center": center,
            }
        })

    # ── Structure ──
    n_structure = read_u64(f)
    print(f"  parsing {n_structure} structure points ...")
    structure = []
    for i in range(n_structure):
        key = read_u32(f)
        X = read_vec_double(f)

        n_obs = read_u64(f)
        observations = []
        for _ in range(n_obs):
            view_id = read_u32(f)
            id_feat = read_u32(f)
            x = read_vec_double(f)
            observations.append({
                "key": view_id,
                "value": {"id_feat": id_feat, "x": x}
            })

        structure.append({
            "key": key,
            "value": {
                "X": X,
                "observations": observations,
            }
        })
        if (i + 1) % 50000 == 0:
            print(f"    ... {i + 1}/{n_structure}")

    # ── Control Points ──
    remaining = len(raw) - f.tell()
    control_points: list = []
    if remaining > 8:
        try:
            n_cp = read_u64(f)
            for _ in range(n_cp):
                key = read_u32(f)
                X = read_vec_double(f)
                n_obs = read_u64(f)
                observations = []
                for __ in range(n_obs):
                    id_feat = read_u32(f)
                    x = read_vec_double(f)
                    observations.append({"key": id_feat, "value": {"x": x}})
                control_points.append({
                    "key": key,
                    "value": {"X": X, "observations": observations}
                })
        except Exception:
            pass

    result = {
        "sfm_data_version": version,
        "root_path": root_path,
        "views": views,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "structure": structure,
        "control_points": control_points,
    }

    print(f"  完成: {len(views)} views, {len(intrinsics)} intrinsics, "
          f"{len(extrinsics)} extrinsics, {len(structure)} structure, "
          f"{len(control_points)} control_points")

    return result


def main():
    ap = argparse.ArgumentParser(description="Parse openMVG sfm_data.bin → JSON")
    ap.add_argument("scene", help="场景根目录")
    ap.add_argument("--bin", default="reconstruction/sfm_data.bin",
                    help="sfm_data.bin 相对路径")
    ap.add_argument("--out", default="reconstruction/sfm_data_full.json",
                    help="输出 JSON 相对路径")
    args = ap.parse_args()

    scene = os.path.abspath(args.scene)
    bin_path = os.path.join(scene, args.bin)
    out_path = os.path.join(scene, args.out)

    if not os.path.isfile(bin_path):
        print(f"错误: 找不到 {bin_path}")
        sys.exit(1)

    print(f"[parse_openmvg_bin] {bin_path}")
    data = parse_sfm_data_bin(bin_path)

    print(f"  写入 {out_path} ...")
    with open(out_path, "w") as fp:
        json.dump(data, fp, indent=2)
    fsize_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  完成 ({fsize_mb:.1f} MB)")


if __name__ == "__main__":
    main()
