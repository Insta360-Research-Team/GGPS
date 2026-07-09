#!/usr/bin/env python3
"""根据模板批量生成 config_360/<dataset>_<scene>.yaml。

模板文件中：@ROOT@ 被替换为 --root，@SCENE@ 被替换为各场景目录名。

支持的数据集：
    - 360roam      模板: config_360/360roam_template.yaml
    - omniblender  模板: config_360/omniblender_template.yaml

用法：
    python tools/gen_360_yamls.py --dataset 360roam      --root /path/to/datasets/360Roam
    python tools/gen_360_yamls.py --dataset omniblender  --root /path/to/datasets/OmniBlender
    python tools/gen_360_yamls.py --dataset all                                                # 用默认根
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "360roam": {
        "template": REPO_ROOT / "config_360" / "360roam_template.yaml",
        "default_root": "/path/to/datasets/360Roam",
        "scenes": [
            "bar", "base", "cafe", "canteen", "center", "center1",
            "corridor", "innovation", "lab", "library", "office",
        ],
        "out_prefix": "360roam",
    },
    "omniblender": {
        "template": REPO_ROOT / "config_360" / "omniblender_template.yaml",
        "default_root": "/path/to/datasets/OmniBlender",
        "scenes": [
            "LOU", "archiviz-flat", "barbershop", "bistro_bike", "bistro_square",
            "classroom", "fisher-hut", "lone_monk",
            "pavilion_midday_chair", "pavilion_midday_pond", "restroom",
        ],
        "out_prefix": "omniblender",
    },
}


def gen_one(dataset_key: str, root: str, scenes: list[str] | None) -> list[Path]:
    info = DATASETS[dataset_key]
    template_path = info["template"]
    out_prefix = info["out_prefix"]
    if not template_path.is_file():
        raise FileNotFoundError(f"模板不存在: {template_path}")
    tmpl = template_path.read_text()
    out_dir = REPO_ROOT / "config_360"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_scenes = scenes if scenes else info["scenes"]
    written = []
    for s in target_scenes:
        text = tmpl.replace("@ROOT@", root).replace("@SCENE@", s)
        out_path = out_dir / f"{out_prefix}_{s}.yaml"
        out_path.write_text(text)
        written.append(out_path)
        print(f"  [ok] {out_path.relative_to(REPO_ROOT)}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"], required=True)
    ap.add_argument("--root", default=None,
                    help="数据集根目录；不传则使用脚本里的 default_root（仅对单数据集有效）")
    ap.add_argument("--scenes", nargs="*", default=None, help="可选：只生成这些场景")
    args = ap.parse_args()

    if args.dataset == "all":
        if args.root or args.scenes:
            print("[err] --dataset all 时不接受 --root / --scenes（每个数据集用各自默认根，全部场景）")
            sys.exit(1)
        for k in DATASETS:
            print(f"--- {k} ---")
            gen_one(k, DATASETS[k]["default_root"], None)
    else:
        root = args.root if args.root else DATASETS[args.dataset]["default_root"]
        print(f"--- {args.dataset} ({root}) ---")
        gen_one(args.dataset, root, args.scenes)


if __name__ == "__main__":
    main()
