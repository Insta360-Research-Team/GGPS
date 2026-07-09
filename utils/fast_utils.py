#
# 感知一致性 (Weight-Sensitive Densification) 工具函数
#

import torch
import random
import numpy as np
from utils.loss_utils import ssim, ssim_per_pixel


def sampling_cameras(viewpoint_stack, num_cams=10, current_cam=None, mode='random'):
    """
    从视点栈中采样相机
    
    Args:
        viewpoint_stack: 相机列表
        num_cams: 采样数量（默认10）
        current_cam: 当前训练相机对象
        mode: 采样模式 ('random', 'spatial')
    
    Returns:
        camlist: 采样的相机列表
    """
    stack = list(viewpoint_stack)
    n_total = len(stack)
    num_to_sample = min(num_cams, n_total)
    
    if current_cam is None or mode == 'random':
        indices = random.sample(range(n_total), num_to_sample)
        return [stack[i] for i in indices]
    
    if mode == 'spatial':
        current_pos = current_cam.camera_center
        if current_pos.device != torch.device('cpu'):
            current_pos = current_pos.cpu()
        
        distances = []
        for i, cam in enumerate(stack):
            cam_pos = cam.camera_center
            if cam_pos.device != torch.device('cpu'):
                cam_pos = cam_pos.cpu()
            dist = torch.norm(current_pos - cam_pos).item()
            distances.append((i, dist))
        
        distances.sort(key=lambda x: x[1])
        indices = [d[0] for d in distances[:num_to_sample]]
        return [stack[i] for i in indices]
    
    return random.sample(stack, num_to_sample)


def sampling_cameras_with_coverage(cameras, num_cams=10, sample_counts=None, coverage_priority=0.7):
    """
    带覆盖保证的相机采样
    
    Args:
        cameras: 当前候选相机列表
        num_cams: 采样数量
        sample_counts: 全局采样次数记录 {camera_id: count}
        coverage_priority: 覆盖优先级（0-1）
    
    Returns:
        camlist: 采样的相机列表
        updated_sample_counts: 更新后的采样次数记录
    """
    if len(cameras) == 0:
        return [], sample_counts or {}
    
    num_to_sample = min(num_cams, len(cameras))
    
    if sample_counts is None:
        sample_counts = {}
    
    camera_ids = [cam.image_name for cam in cameras]
    sample_counts_array = np.array([sample_counts.get(cam_id, 0) for cam_id in camera_ids], dtype=np.float32)
    
    # 计算覆盖权重
    if sample_counts_array.max() > 0:
        decay_factor = 2.0
        coverage_weights = np.exp(-sample_counts_array * decay_factor)
        max_coverage_weight = coverage_weights.max()
        if max_coverage_weight > 0:
            coverage_weights = coverage_weights / max_coverage_weight
    else:
        coverage_weights = np.ones(len(cameras), dtype=np.float32)
    
    random_weights = np.ones(len(cameras), dtype=np.float32)
    combined_weights = coverage_priority * coverage_weights + (1.0 - coverage_priority) * random_weights
    combined_weights = combined_weights / (combined_weights.sum() + 1e-10)
    
    random_tiebreaker = np.random.rand(len(cameras)) * 1e-6
    weighted_scores = combined_weights + random_tiebreaker
    sorted_indices = np.argsort(weighted_scores)[::-1]
    indices = sorted_indices[:num_to_sample]
    
    camlist = [cameras[i] for i in indices]
    
    # 更新采样次数
    for idx in indices:
        cam_id = camera_ids[idx]
        sample_counts[cam_id] = sample_counts.get(cam_id, 0) + 1
    
    return camlist, sample_counts


def get_loss(reconstructed_image, original_image):
    """
    计算归一化的L1损失图
    
    Args:
        reconstructed_image: 重建图像 (C, H, W)
        original_image: 原始图像 (C, H, W)
    
    Returns:
        l1_loss_norm: 归一化的L1损失图 (H, W)，范围[0, 1]
    """
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_min = torch.min(l1_loss)
    l1_max = torch.max(l1_loss)
    if l1_max - l1_min > 0:
        l1_loss_norm = (l1_loss - l1_min) / (l1_max - l1_min)
    else:
        l1_loss_norm = torch.zeros_like(l1_loss)
    
    return l1_loss_norm


def compute_gaussian_score_dual_filter(camlist, gaussians, pipe, background, 
                                        render_func, DENSIFY=False,
                                        norm_loss_thresh=0.05):
    """
    [改进版] 计算视图分数：逐像素误差加权
    
    核心改进：
    CUDA 层直接累加 alpha * T * L1_error[pixel]，精确反映每个高斯对误差的贡献。
    """
    full_error_contribution = None  # 累加每个高斯的误差贡献
    full_metric_counts = None       # 累加每个高斯的误差贡献（用于 DENSIFY）
    # 统计每个高斯被观测到的视图数量
    visible_view_counts = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
    
    for viewpoint_cam in camlist:
        with torch.no_grad():
            # 确保相机参数在 CUDA 上
            viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
            viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
            viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
            viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
            
            # 第一次渲染：获取图像计算误差
            render_pkg = render_func(viewpoint_cam, gaussians, pipe, background)
            render_image = render_pkg["render"]
            radii = render_pkg["radii"]
            
            # 统计该视图中可见的高斯
            visible_mask = radii > 0
            visible_view_counts += visible_mask.float()
            
            gt_image = viewpoint_cam.original_image.cuda()
            
            # 计算原始L1误差图
            l1_error_map = torch.mean(torch.abs(render_image - gt_image), dim=0).detach()  # (H, W)
            # 计算归一化误差图
            l1_loss_norm = get_loss(render_image, gt_image)  # (H, W)
            
            # 用归一化误差筛选，保证跨视图一致性
            high_error_mask = l1_loss_norm > norm_loss_thresh
            
            # 保存原始误差值用于 CUDA 加权计算
            filtered_error_map = torch.where(high_error_mask, l1_error_map, torch.zeros_like(l1_error_map))
            metric_map = filtered_error_map.float().contiguous()
            
            # 第二次渲染：CUDA 层累积误差
            render_pkg2 = render_func(
                viewpoint_cam, gaussians, pipe, background,
                metric_map=metric_map
            )
            error_contribution = render_pkg2.get("metric_count", None)
            
            if error_contribution is None:
                continue
            
            if full_error_contribution is None:
                full_error_contribution = error_contribution.clone()
            else:
                full_error_contribution += error_contribution
            
            if DENSIFY:
                if full_metric_counts is None:
                    full_metric_counts = error_contribution.clone()
                else:
                    full_metric_counts += error_contribution
    
    # 计算剪枝分数（归一化到 0-1）
    if full_error_contribution is not None:
        score_min = torch.min(full_error_contribution)
        score_max = torch.max(full_error_contribution)
        if score_max - score_min > 0:
            pruning_score = (full_error_contribution - score_min) / (score_max - score_min)
        else:
            pruning_score = torch.zeros_like(full_error_contribution)
    else:
        pruning_score = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
    
    # 计算重要性分数（用于 DENSIFY）
    if DENSIFY and full_metric_counts is not None:
        safe_view_counts = torch.clamp(visible_view_counts, min=1.0)
        importance_score = full_metric_counts / safe_view_counts
    else:
        importance_score = None
    
    return importance_score, pruning_score

def compute_gaussian_score_with_ssim(camlist, gaussians, pipe, background, 
                                     render_func, DENSIFY=False,
                                     norm_loss_thresh=0.05, lambda_dssim=0.2):
    """
    [新函数] 结合 L1 和 SSIM 的感知一致性评分函数
    """
    full_metric_counts = None
    visible_view_counts = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
    
    for viewpoint_cam in camlist:
        with torch.no_grad():
            viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
            viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
            viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
            viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
            
            # 渲染图像
            render_pkg = render_func(viewpoint_cam, gaussians, pipe, background)
            render_image = render_pkg["render"]
            radii = render_pkg["radii"]
            
            # 统计可见性
            visible_mask = radii > 0
            visible_view_counts += visible_mask.float()
            
            gt_image = viewpoint_cam.original_image.cuda()
            
            # 1. 计算 L1 误差图 (H, W)
            l1_error_map = torch.mean(torch.abs(render_image - gt_image), dim=0).detach()
            
            # 2. 计算 SSIM 误差图 (H, W)
            ssim_map = ssim_per_pixel(render_image.unsqueeze(0), gt_image.unsqueeze(0)).squeeze(0).detach()
            dssim_map = 1.0 - ssim_map
            
            # 3. 综合物理误差 (用于 CUDA 累加)
            combined_error_map = (1.0 - lambda_dssim) * l1_error_map + lambda_dssim * dssim_map
            
            # 4. 综合归一化误差 (用于筛选阈值)
            # 对 combined_error_map 进行局部归一化作为筛选判据
            l_min, l_max = combined_error_map.min(), combined_error_map.max()
            norm_map = (combined_error_map - l_min) / (l_max - l_min + 1e-7)
            
            high_error_mask = norm_map > norm_loss_thresh
            metric_map = torch.where(high_error_mask, combined_error_map, torch.zeros_like(combined_error_map))
            metric_map = metric_map.float().contiguous()
            
            # 误差反向映射到高斯点
            render_pkg2 = render_func(
                viewpoint_cam, gaussians, pipe, background,
                metric_map=metric_map
            )
            error_contribution = render_pkg2.get("metric_count", None)
            
            if error_contribution is None:
                continue
            
            if full_metric_counts is None:
                full_metric_counts = error_contribution.clone()
            else:
                full_metric_counts += error_contribution
    
    if DENSIFY and full_metric_counts is not None:
        importance_score = full_metric_counts / torch.clamp(visible_view_counts, min=1.0)
    else:
        importance_score = None
        
    return importance_score, None

