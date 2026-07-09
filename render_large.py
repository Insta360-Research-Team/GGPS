#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import yaml
import json
import torch
import torchvision
import time
import numpy as np
from tqdm import tqdm
from arguments import GroupParams
from scene import LargeScene
from scene.datasets import GSDataset, CacheDataLoader
from os import makedirs
from gaussian_renderer import render_large
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.general_utils import parse_cfg


def _get_camera_type(cam_info):
    """兼容拿 camera_type：GSDataset 返回的是 dict，Camera 对象用 attr。"""
    if isinstance(cam_info, dict):
        return int(cam_info.get("camera_type", 1))
    return int(getattr(cam_info, "camera_type", 1))


def _crop_pano_bottom(tensor, ratio, camera_type):
    """与 train_large._crop_pano_bottom 保持一致：仅 ERP/LonLat (camera_type==3) 且 ratio>0 时
    沿 H 维度裁掉底部 round(H*ratio) 行。其它相机类型不变。
    """
    if tensor is None or ratio is None or ratio <= 0:
        return tensor
    if int(camera_type) != 3:
        return tensor
    H = tensor.shape[-2]
    n = int(round(H * float(ratio)))
    if n <= 0 or n >= H:
        return tensor
    return tensor[..., : H - n, :]


def render_set(model_path, name, iteration, gs_dataset, gaussians, pipeline, background, max_cache_num: int = 512, skip_bottom_ratio: float = 0.0):
    avg_render_time = 0
    max_render_time = 0
    avg_memory = 0
    max_memory = 0

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 与 train_large 一致：CacheDataLoader 直接 yield GSDataset 元组，无 default_collate，可含 None
    # 渲染需顺序与 cameras 列表一致，shuffle=False；预缓存勿开多线程（loadCam 内 cuda 与注释同 train_large）
    data_loader = CacheDataLoader(
        gs_dataset,
        max_cache_num=max_cache_num,
        seed=42,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    for idx, batch in enumerate(tqdm(data_loader, desc="Rendering progress")):
        cam_info, gt_image = batch[0], batch[1]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        rendering = render_large(cam_info, gaussians, pipeline, background)["render"]
        torch.cuda.synchronize()
        end = time.time()
        
        gt = gt_image[0:3, :, :]
        avg_render_time += end-start
        max_render_time = max(max_render_time, end-start)

        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)

        # 与 train_large 评估口径一致：ERP 全景 (camera_type=3) 时裁掉底部，
        # 离线 metrics_large 在裁后图上计算 PSNR/SSIM/LPIPS，与 OmniGS 对齐。
        # 注意：cam_info 是 GSDataset 返回的 dict，不能用 getattr。
        _ctype = _get_camera_type(cam_info)
        rendering = _crop_pano_bottom(rendering, skip_bottom_ratio, _ctype)
        gt = _crop_pano_bottom(gt, skip_bottom_ratio, _ctype)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    with open(model_path + "/costs.json", 'w') as fp:
        json.dump({
            "Average FPS": len(data_loader)/avg_render_time,
            "Min FPS": 1/max_render_time,
            "Average Memory(M)": avg_memory/len(data_loader),
            "Max Memory(M)": max_memory,
            "Number of Gaussians": gaussians.get_xyz.shape[0]
        }, fp, indent=True)
    
    print(f'Average FPS: {len(data_loader)/avg_render_time:.4f}')
    print(f'Min FPS: {1/max_render_time:.4f}')
    print(f'Average Memory: {avg_memory/len(data_loader):.4f} M')
    print(f'Max Memory: {max_memory:.4f} M')
    print(f'Number of Gaussians: {gaussians.get_xyz.shape[0]}')


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    load_vq: bool,
    skip_train: bool,
    skip_test: bool,
    custom_test: bool,
    max_cache_num: int = 512,
    skip_bottom_ratio: float = 0.0,
):

    with torch.no_grad():
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_config['kwargs'])

        if custom_test:
            dataset.source_path = custom_test
            filename = os.path.basename(dataset.source_path)
        scene = LargeScene(dataset, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)
        print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if custom_test:
            views = scene.getTrainCameras() + scene.getTestCameras()
            gs_dataset = GSDataset(views, scene, dataset, pipeline)
            render_set(
                dataset.model_path, filename, scene.loaded_iter, gs_dataset, gaussians, pipeline, background,
                max_cache_num=max_cache_num, skip_bottom_ratio=skip_bottom_ratio,
            )
            print("Skip both train and test, render all views")
        else:
            if not skip_train:
                gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipeline)
                render_set(
                    dataset.model_path, "train", scene.loaded_iter, gs_dataset, gaussians, pipeline, background,
                    max_cache_num=max_cache_num, skip_bottom_ratio=skip_bottom_ratio,
                )

            if not skip_test:
                gs_dataset = GSDataset(scene.getTestCameras(), scene, dataset, pipeline)
                render_set(
                    dataset.model_path, "test", scene.loaded_iter, gs_dataset, gaussians, pipeline, background,
                    max_cache_num=max_cache_num, skip_bottom_ratio=skip_bottom_ratio,
                )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--config', type=str, help='train config file path of fused model')
    parser.add_argument('--model_path', type=str, help='model path of fused model')
    parser.add_argument("--custom_test", type=str, help="appointed test path")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument('--block_id', type=int, default=-1)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.model_path is None:
        args.model_path = os.path.join('output', os.path.basename(args.config).split('.')[0])
    if args.load_vq:
        args.iteration = 30000  # apply a default value

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)

    render_sets(
        lp,
        args.iteration,
        pp,
        args.load_vq,
        args.skip_train,
        args.skip_test,
        args.custom_test,
        max_cache_num=getattr(op, "max_cache_num", 512),
        skip_bottom_ratio=float(getattr(op, "skip_bottom_ratio", 0.0)),
    )