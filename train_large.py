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
import time
import yaml
import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_large, network_gui
import sys
from lightning.pytorch.loggers import (
    TensorBoardLogger,
    WandbLogger,
)
from scene import LargeScene
from scene.datasets import GSDataset, CacheDataLoader
from utils.camera_utils import loadCam
from utils.general_utils import safe_state, parse_cfg, get_expon_lr_func
from utils.fast_utils import sampling_cameras, sampling_cameras_with_coverage, compute_gaussian_score_dual_filter, compute_gaussian_score_with_ssim
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.log_utils import tensorboard_log_image, wandb_log_image
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace, SUPPRESS
from arguments import GroupParams


def _get_camera_type(cam_info):
    """兼容拿 camera_type：GSDataset 返回的是 dict，Camera 对象用 attr。"""
    if isinstance(cam_info, dict):
        return int(cam_info.get("camera_type", 1))
    return int(getattr(cam_info, "camera_type", 1))


def _crop_pano_bottom(tensor, ratio, camera_type):
    """对 ERP/LonLat 全景张量裁掉底部 round(H*ratio) 行。

    与 OmniGS Optimization::skip_bottom_ratio 行为一致：仅当 camera_type==3
    (ERP / LonLat 全景) 且 ratio>0 时生效，沿倒数第二维 (H) 裁尾，
    用于规避南极投影失真 / 三脚架 / 地面噪声。
    其他相机类型 (pinhole / 立方体面 等) 不做任何修改。
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


def training(dataset, opt, pipe, testing_iterations, saving_iterations, refilter_iterations, checkpoint_iterations, checkpoint, max_cache_num, debug_from, use_weight_sensitive=False, skybox_num=0, skybox_locked=False, train_config_path=None):
    first_iter = 0
    log_writer, image_logger = prepare_output_and_logger(dataset, train_config_path=train_config_path)

    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_config['kwargs'])
    
    # 设置天空球参数
    dataset.skybox_num = skybox_num
    dataset.skybox_locked = skybox_locked
    
    scene = LargeScene(dataset, gaussians)
    gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipe)
    if len(gs_dataset) > 0:
        cache_workers = int(getattr(dataset, "cache_num_workers", 8))
        print(f"Using maximum cache size of {max_cache_num} for {len(gs_dataset)} training images (cache_workers={cache_workers})")
        data_loader = CacheDataLoader(gs_dataset, max_cache_num=max_cache_num, seed=42, batch_size=1, shuffle=True, num_workers=cache_workers)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # [覆盖优化] 初始化相机采样计数器
    cam_sample_counts = {}

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init,
        opt.depth_l1_weight_final,
        max_steps=opt.iterations,
    )
    if opt.depth_l1_weight_init > 0 or opt.depth_l1_weight_final > 0:
        pipe.output_depth = True
    depth_warned = False

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_time_render = 0.0
    ema_time_loss = 0.0
    ema_time_densify = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    while iteration <= opt.iterations:
        if len(gs_dataset) == 0:
            print("No training data found")
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration, dataset)
            break    
        
        for dataset_index, batch in enumerate(data_loader):
            if len(batch) >= 6:
                cam_info, gt_image, mask, mono_invdepth, depth_mask, depth_reliable = batch[:6]
            elif len(batch) == 3:
                cam_info, gt_image, mask = batch
                mono_invdepth, depth_mask, depth_reliable = None, None, False
            else:
                cam_info, gt_image = batch[0], batch[1]
                mask = None
                mono_invdepth, depth_mask, depth_reliable = None, None, False
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Render
            start = time.time()
            if (iteration - 1) == debug_from:
                pipe.debug = True
            render_pkg = render_large(cam_info, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            end = time.time()
            ema_time_render = 0.4 * (end - start) + 0.6 * ema_time_render

            # Loss
            start = time.time()
            gt_image = gt_image.cuda()
            if mask is not None:
                mask = mask.cuda().float()
                image = image * mask
                gt_image = gt_image * mask

            # OmniGS skip_bottom_ratio：ERP 全景 (camera_type=3) 时裁掉底部一段，
            # 不影响 pinhole（默认 0.0 也无副作用）。
            # 注意：cam_info 是 GSDataset 返回的 dict，必须用 dict 取键，否则 getattr 永远拿默认值
            _sbr = getattr(opt, "skip_bottom_ratio", 0.0)
            _ctype = _get_camera_type(cam_info)
            image = _crop_pano_bottom(image, _sbr, _ctype)
            gt_image = _crop_pano_bottom(gt_image, _sbr, _ctype)

            Ll1 = l1_loss(image, gt_image)
            if opt.lambda_dssim > 0:
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            else:
                loss = Ll1
            loss_depth_log = 0.0
            depth_reliable_flag = bool(depth_reliable) if not torch.is_tensor(depth_reliable) else bool(depth_reliable.item())
            depth_w = depth_l1_weight(iteration)
            if depth_w > 0 and depth_reliable_flag and mono_invdepth is not None and depth_mask is not None:
                pred_invdepth = render_pkg.get("depth", None)
                if pred_invdepth is not None:
                    mono_invdepth = mono_invdepth.cuda(non_blocking=True)
                    depth_mask = depth_mask.cuda(non_blocking=True).float()
                    pred_invdepth = _crop_pano_bottom(pred_invdepth, _sbr, _ctype)
                    mono_invdepth = _crop_pano_bottom(mono_invdepth, _sbr, _ctype)
                    depth_mask = _crop_pano_bottom(depth_mask, _sbr, _ctype)
                    depth_residual = torch.abs((pred_invdepth - mono_invdepth) * depth_mask)
                    loss_depth = depth_w * depth_residual.mean()
                    loss = loss + loss_depth
                    loss_depth_log = loss_depth.detach().item()
                elif not depth_warned:
                    print("[Depth] depth supervision requested but renderer has no depth output, skipping depth loss.")
                    depth_warned = True

            loss.backward()
            end = time.time()
            ema_time_loss = 0.4 * (end - start) + 0.6 * ema_time_loss

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    num_gaussians = gaussians.get_xyz.shape[0]
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "depth_Loss": f"{loss_depth_log:.{7}f}", "Gaussians": f"{num_gaussians:,}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()
                
                grads = gaussians.xyz_gradient_accum / gaussians.denom
                grads[grads.isnan()] = 0.0
                ema_time = {
                    "render": ema_time_render,
                    "loss": ema_time_loss,
                    "densify": ema_time_densify,
                    "num_points": radii.shape[0],
                    "mean_grad": grads.mean().item(),
                }

                lr = {}
                for param_group in gaussians.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                # Log and save
                training_report(dataset, opt, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr,
                                iter_start.elapsed_time(iter_end), testing_iterations, scene, render_large, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, dataset)
                if (iteration in refilter_iterations):
                    print("\n[ITER {}] Refiltering Training Data".format(iteration))
                    gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipe)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        start = time.time()
                        torch.cuda.empty_cache()
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        
                        if use_weight_sensitive:
                            # [感知一致性] 使用带覆盖保证的采样
                            num_cams = 10 
                            
                            # 调用改进的采样函数
                            cam_info_list, cam_sample_counts = sampling_cameras_with_coverage(
                                scene.getTrainCameras(), 
                                num_cams=num_cams, 
                                sample_counts=cam_sample_counts,
                                coverage_priority=0.9 
                            )
                            # 将 CameraInfo 转换为真正的 Camera 对象（加载图片和矩阵）
                            camlist = [loadCam(dataset, 0, cam_info, 1.0) for cam_info in cam_info_list]
                            
                            importance_score, _ = compute_gaussian_score_dual_filter(
                                camlist, gaussians, pipe, background, render, DENSIFY=True, norm_loss_thresh=0.1
                            )
                            # 使用新增的 SSIM 综合评分函数
                            # importance_score, _ = compute_gaussian_score_with_ssim(
                            #     camlist, gaussians, pipe, background, render, DENSIFY=True, 
                            #     norm_loss_thresh=0.08, lambda_dssim=opt.lambda_dssim
                            # )
                            gaussians.densify_and_prune_fastgs(
                                max_screen_size=size_threshold,
                                min_opacity=0.005,
                                extent=scene.cameras_extent,
                                radii=radii,
                                grad_thresh=opt.densify_grad_threshold,
                                grad_thresh_abs=opt.densify_grad_abs_threshold,
                                importance_score=importance_score,
                                prune_by_extent=getattr(opt, "prune_by_extent", True),
                            )
                        else:
                            # 原始密集化
                            gaussians.densify_and_prune(
                                opt.densify_grad_threshold,
                                opt.densify_grad_abs_threshold,
                                0.005,
                                scene.cameras_extent,
                                size_threshold,
                                getattr(opt, "prune_by_extent", True),
                            )
                        
                        end = time.time()
                        ema_time_densify = 0.4 * (end - start) + 0.6 * ema_time_densify

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    # 天空球梯度处理
                    if gaussians._xyz.grad is not None and gaussians.skybox_points > 0:
                        # 天空球的 xyz 和 scaling 始终锁定（位置和尺度保持初始化）
                        gaussians._xyz.grad[:gaussians.skybox_points, :] = 0
                        gaussians._scaling.grad[:gaussians.skybox_points, :] = 0
                        
                        # 如果 skybox_locked=True（Block 训练），额外锁定 rotation, features, opacity
                        if gaussians.skybox_locked:
                            gaussians._rotation.grad[:gaussians.skybox_points, :] = 0
                            gaussians._features_dc.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._features_rest.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._opacity.grad[:gaussians.skybox_points, :] = 0
                    
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            iteration += 1
            if iteration >= opt.iterations:
                break

def prepare_output_and_logger(args, train_config_path=None):
    if not args.model_path:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        # time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        args.model_path = os.path.join("./output/", config_name)
        if args.block_id >= 0:
            if args.block_id < args.block_dim[0] * args.block_dim[1] * args.block_dim[2]:
                args.model_path = f"{args.model_path}/cells/cell{args.block_id}"
                if args.logger_config is not None:
                    args.logger_config['name'] = f"{args.logger_config['name']}_cell{args.block_id}"
            else:
                raise ValueError("Invalid block_id: {}".format(args.block_id))
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # cfg_args 会被 arguments.get_combined_args 整文件 eval，须保持单行合法 Namespace(...)
    cfg_vars = dict(vars(args))
    if train_config_path:
        try:
            with open(train_config_path, "r", encoding="utf-8", errors="replace") as yf:
                cfg_vars["config_yaml_snapshot"] = yf.read()
        except OSError:
            pass
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**cfg_vars)))
    
    # build logger
    log_writer = None
    image_logger = None
    logger_args = {
        "save_dir": args.model_path
    }
    if args.logger_config is None or args.logger_config['logger'] == "tensorboard":
        log_writer = TensorBoardLogger(**logger_args)
        image_logger = tensorboard_log_image
    elif args.logger_config['logger'] == "wandb":
        logger_args.update(name=args.logger_config['name'])
        logger_args.update(project=args.logger_config['project'])
        log_writer = WandbLogger(**logger_args)
        image_logger = wandb_log_image
    else:
        raise ValueError("Unknown logger: {}".format(args.logger_config['logger']))
    
    return log_writer, image_logger

def training_report(dataset, opt, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr, elapsed, testing_iterations, scene : LargeScene, renderFunc, renderArgs):
    if log_writer:
        metrics_to_log = {
            "train_loss_patches/l1_loss": Ll1.item(),
            "train_loss_patches/total_loss": loss.item(),
            "train_time/render": ema_time["render"],
            "train_time/loss": ema_time["loss"],
            "train_time/densify": ema_time["densify"],
            "train_time/num_points": ema_time["num_points"],
            "train_time/mean_grad": ema_time["mean_grad"],
            "iter_time": elapsed,
        }
        for key, value in lr.items():
            metrics_to_log["trainer/" + key] = value
        log_writer.log_metrics(metrics_to_log, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, camera in enumerate(config['cameras']):
                    viewpoint_cam = loadCam(dataset, idx, camera, 1)
                    viewpoint = {
                        "FoVx": viewpoint_cam.FoVx,
                        "FoVy": viewpoint_cam.FoVy,
                        "image_name": viewpoint_cam.image_name,
                        "image_height": viewpoint_cam.image_height,
                        "image_width": viewpoint_cam.image_width,
                        "camera_center": viewpoint_cam.camera_center,
                        "world_view_transform": viewpoint_cam.world_view_transform,
                        "full_proj_transform": viewpoint_cam.full_proj_transform,
                        "camera_type": int(getattr(viewpoint_cam, "camera_type", 1)),
                    }
                    org_img = viewpoint_cam.original_image
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(org_img.to("cuda"), 0.0, 1.0)
                    eval_mask = getattr(viewpoint_cam, "gt_alpha_mask", None)
                    if eval_mask is not None:
                        eval_mask = eval_mask.to("cuda")
                    # 与训练 loss 口径一致：ERP 全景按 skip_bottom_ratio 裁底部
                    _eval_sbr = getattr(opt, "skip_bottom_ratio", 0.0)
                    _eval_ctype = int(viewpoint["camera_type"])
                    image = _crop_pano_bottom(image, _eval_sbr, _eval_ctype)
                    gt_image = _crop_pano_bottom(gt_image, _eval_sbr, _eval_ctype)
                    eval_mask = _crop_pano_bottom(eval_mask, _eval_sbr, _eval_ctype)
                    if log_writer and (idx < 5):
                        grid = torchvision.utils.make_grid(torch.concat([image, gt_image], dim=-1))
                        image_logger(
                            log_writer=log_writer,
                            tag=config['name'] + "_view_{}".format(viewpoint["image_name"]),
                            image_tensor=grid,
                            step=iteration,
                        )
                    l1_test += l1_loss(image, gt_image, eval_mask).mean().double()
                    psnr_test += psnr(image, gt_image, eval_mask).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if log_writer:
                    metrics_to_log = {
                        config['name'] + '/loss_viewpoint/l1_loss': l1_test,
                        config['name'] + '/loss_viewpoint/psnr': psnr_test,
                    }
                    log_writer.log_metrics(metrics_to_log, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--block_id', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--refilter_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # 默认以 YAML optim_params.max_cache_num 为准；仅在命令行显式传入时覆盖
    parser.add_argument("--max_cache_num", type=int, default=SUPPRESS)
    parser.add_argument("--weight_sensitive", action="store_true", help="Enable weight-sensitive densification")

    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)
        args.save_iterations.append(op.iterations)
    
    # 直接从配置文件读取参数（不再通过命令行覆盖）
    skybox_num = getattr(lp, 'skybox_num', 0)
    skybox_locked = getattr(lp, 'skybox_locked', False)
    
    print("Optimizing " + lp.model_path)
    print(f"Skybox settings: num={skybox_num}, locked={skybox_locked}")
    print(f"max_cache_num (optim_params / YAML): {op.max_cache_num}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, args.test_iterations, args.save_iterations, args.refilter_iterations, args.checkpoint_iterations, args.start_checkpoint, op.max_cache_num, args.debug_from, args.weight_sensitive, skybox_num, skybox_locked, train_config_path=args.config)

    # All done
    print("\nTraining complete.")
