#!/usr/bin/env python3
"""用 DAP 模型为全景数据集生成深度图和天空 mask。

输出：
    <scene>/depths/<name>.png      16-bit PNG 逆深度（invdepth），天空=0
    <scene>/skymasks/<name>.png    8-bit PNG（1=前景，0=天空）
    <scene>/depths_vis/<name>.png  彩色可视化（调试用）

用法：
    python tools/generate_depth_and_skymask.py --scene data/a1nanshan
    python tools/generate_depth_and_skymask.py --scene data/a1nanshan --images imgs --gpu 0
    # 默认续跑：已存在 depths/ 与 skymasks/ 同名 png 的帧会跳过，便于断点续跑
    python tools/generate_depth_and_skymask.py --scene ... --force  # 全部重算
"""

from __future__ import annotations

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from tqdm import tqdm

DAP_ROOT = os.path.expanduser("~/DAP")
sys.path.insert(0, DAP_ROOT)

from networks.models import make as dap_make


def load_dap_model(config_path: str, device: str):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_path = os.path.join(DAP_ROOT, "model", "model.pth")
    state = torch.load(model_path, map_location=device, weights_only=False)

    m = dap_make(config["model"])
    if any(k.startswith("module") for k in state.keys()):
        m = nn.DataParallel(m)

    m = m.to(device)
    m_state = m.state_dict()
    m.load_state_dict({k: v for k, v in state.items() if k in m_state}, strict=False)
    m.eval()
    return m


@torch.inference_mode()
def infer_depth_and_mask(model, device, img_rgb_u8: np.ndarray):
    """返回 (depth_f32, sky_mask_bool)。

    depth_f32: [H,W] float32，metric depth（DAP 归一化范围）
    sky_mask_bool: [H,W] bool，True=天空
    """
    img = img_rgb_u8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    outputs = model(tensor)

    pred_depth = outputs["pred_depth"][0].detach().cpu().squeeze().numpy()

    sky_mask = np.zeros_like(pred_depth, dtype=bool)
    if "pred_mask" in outputs:
        pm = outputs["pred_mask"][0].detach().cpu().squeeze().numpy()
        sky_mask = pm > 0.5

    return pred_depth.astype(np.float32), sky_mask


def depth_to_invdepth_u16(depth: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """将 metric depth 转为 16-bit 逆深度 PNG。

    invdepth = 1/depth（天空=0），归一化到 [0, 65535]。
    """
    eps = 1e-6
    invdepth = np.zeros_like(depth, dtype=np.float32)
    fg = ~sky_mask & (depth > eps)
    invdepth[fg] = 1.0 / depth[fg]

    max_val = invdepth.max()
    if max_val > 0:
        invdepth_norm = invdepth / max_val
    else:
        invdepth_norm = invdepth

    return (invdepth_norm * 65535).astype(np.uint16), max_val


def colorize_depth(depth: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
    """生成彩色深度可视化图（BGR）。"""
    import matplotlib
    vis = depth.copy()
    vis[sky_mask] = 0
    if vis.max() > 0:
        vis = vis / vis.max()
    vis_u8 = (vis * 255).astype(np.uint8)
    colored = matplotlib.colormaps["Spectral"](vis_u8)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    colored[sky_mask] = [0, 0, 0]
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="用 DAP 生成深度图和天空 mask")
    parser.add_argument("--scene", required=True, help="场景目录路径，如 data/a1nanshan")
    parser.add_argument("--images", default="images", help="图片子目录名")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dap-config", default=os.path.join(DAP_ROOT, "config", "infer.yaml"))
    parser.add_argument("--vis", action="store_true", default=True, help="生成彩色深度可视化")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="不跳过已存在输出，每帧都重新推理（与 --force 类似但可不加载已有文件检查）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略已有输出，整场景重新推理（覆盖 --resume）",
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    scene_dir = os.path.abspath(args.scene)
    images_dir = os.path.join(scene_dir, args.images)
    depths_dir = os.path.join(scene_dir, "depths")
    skymasks_dir = os.path.join(scene_dir, "skymasks")
    vis_dir = os.path.join(scene_dir, "depths_vis")

    os.makedirs(depths_dir, exist_ok=True)
    os.makedirs(skymasks_dir, exist_ok=True)
    if args.vis:
        os.makedirs(vis_dir, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    img_files = sorted([
        f for f in os.listdir(images_dir)
        if Path(f).suffix.lower() in exts and not f.startswith("._")
    ])
    print(f"场景: {scene_dir}")
    print(f"图片目录: {images_dir} ({len(img_files)} 帧)")
    do_resume = args.resume and not args.force
    if do_resume:
        print("续跑模式: 已存在 depths + skymasks 的帧将跳过（加 --force 可全部重算）")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载 DAP 模型 (device={device}) ...")
    model = load_dap_model(args.dap_config, device)
    print("模型加载完成\n")

    max_invdepths = []

    skipped = 0
    for fname in tqdm(img_files, desc="DAP 推理"):
        stem = Path(fname).stem
        depth_out = os.path.join(depths_dir, stem + ".png")
        sky_out = os.path.join(skymasks_dir, stem + ".png")
        if do_resume and os.path.isfile(depth_out) and os.path.isfile(sky_out):
            skipped += 1
            continue

        img_path = os.path.join(images_dir, fname)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"警告: 无法读取 {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth, sky_mask = infer_depth_and_mask(model, device, img_rgb)

        invdepth_u16, max_invdepth = depth_to_invdepth_u16(depth, sky_mask)
        max_invdepths.append(max_invdepth)

        cv2.imwrite(depth_out, invdepth_u16)

        sky_png = np.where(sky_mask, 0, 255).astype(np.uint8)
        cv2.imwrite(sky_out, sky_png)

        if args.vis:
            vis_img = colorize_depth(depth, sky_mask)
            cv2.imwrite(os.path.join(vis_dir, stem + ".png"), vis_img)

    print(f"\n完成!")
    if do_resume and skipped:
        print(f"  跳过（已有输出）: {skipped} 帧")
    print(f"  depths:   {depths_dir} ({len(img_files)} 帧, 16-bit PNG invdepth)")
    print(f"  skymasks: {skymasks_dir} ({len(img_files)} 帧, 1=前景 0=天空)")
    if args.vis:
        print(f"  depths_vis: {vis_dir} (彩色可视化)")
    if max_invdepths:
        print(f"  本次推理 max invdepth 范围: [{min(max_invdepths):.4f}, {max(max_invdepths):.4f}]")
    else:
        print("  本次推理: 无新帧（全部为跳过）；max invdepth 未更新")


if __name__ == "__main__":
    main()
