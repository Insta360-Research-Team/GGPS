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
import torch
import traceback
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from typing import NamedTuple
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.large_utils import block_filtering
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_symmetric
from utils.vq_utils import load_vqgaussian

class GatheredGaussian(NamedTuple):
    gs_xyz: torch.Tensor
    gs_feats: torch.Tensor
    gs_ids: torch.Tensor
    block_scalings: torch.Tensor
    cell_corners: torch.Tensor
    aabb: list
    block_dim: list
    max_sh_degree: int

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)  # [感知一致性] 绝对值梯度累积器
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        # 天空球相关属性
        self.skybox_points = 0       # 天空球点数量
        self.skybox_locked = True    # 天空球锁定标志（锁定后不参与优化）
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, skybox_num : int = 0, skybox_locked : bool = False, scene_center=None):
        """
        从点云创建高斯模型（保持CityGS原始逻辑，仅添加天空球功能）
        
        Args:
            pcd: 点云数据
            spatial_lr_scale: 空间学习率缩放因子
            skybox_num: 天空球点数量（0表示不添加天空球，保持CityGS原始行为）
            skybox_locked: 天空球锁定标志（True=不参与优化，用于Block训练）
            scene_center: 场景中心 (numpy array [3])，用于天空球圆心
        """
        self.spatial_lr_scale = spatial_lr_scale
        self.skybox_locked = skybox_locked
        
        # 从点云提取 xyz 和颜色（H3DGS 方式：保持 RGB 格式，最后统一转换为 SH）
        original_xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()  # RGB 格式
        
        if skybox_num > 0:
            # ---------- H3DGS 天空球：先拼成整份点云，再统一用 KNN 算 scale ----------
            self.skybox_points = skybox_num
            mean = torch.tensor(scene_center, dtype=torch.float32, device="cuda")
            radius = torch.linalg.norm(original_xyz - mean, dim=1).max()
            skybox_radius_mult = 5.0
            theta = (2.0 * torch.pi * torch.rand(skybox_num, device="cuda")).float()
            phi = (torch.arccos(1.0 - 2.0 * torch.rand(skybox_num, device="cuda"))).float()
            skybox_xyz = torch.zeros((skybox_num, 3), device="cuda")
            skybox_xyz[:, 0] = radius * skybox_radius_mult * torch.cos(theta) * torch.sin(phi)
            skybox_xyz[:, 1] = radius * skybox_radius_mult * torch.sin(theta) * torch.sin(phi)
            skybox_xyz[:, 2] = radius * skybox_radius_mult * torch.cos(phi)
            skybox_xyz += mean
            # H3DGS 方式：先拼接颜色（RGB格式），再修改天空球点的颜色
            fused_point_cloud = torch.concat((skybox_xyz, original_xyz))
            fused_color = torch.concat((torch.ones((skybox_num, 3), device="cuda"), fused_color))
            fused_color[:skybox_num, 0] *= 0.7
            fused_color[:skybox_num, 1] *= 0.8
            fused_color[:skybox_num, 2] *= 0.95
            # H3DGS：对整份点云做 KNN，天空球段 dist2 *= 10，场景段 clamp_max(10)
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
            dist2[:skybox_num] *= 10
            dist2[skybox_num:] = torch.clamp_max(dist2[skybox_num:], 10.0)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            radius_val = radius.item() if torch.is_tensor(radius) else radius
            print(f"Added {skybox_num} skybox points (H3DGS scale), scene radius: {radius_val:.2f}")
        else:
            dist2 = torch.clamp_min(distCUDA2(original_xyz), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            fused_point_cloud = original_xyz
            # fused_color 已经是 RGB 格式，保持不变
        
        # CityGS原始逻辑：初始化SH特征
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = RGB2SH(fused_color)
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # CityGS原始逻辑：初始化不透明度
        if skybox_num > 0:
            opacities = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities[:skybox_num] = 0.7
        else:
            opacities = inverse_sigmoid(0.01 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # CityGS原始逻辑：注册参数
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # [感知一致性]
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        """重置不透明度，保护天空球点"""
        if self.skybox_points > 0:
            # 天空球点保持原不透明度，其他点重置
            opacities_new = torch.cat(
                (
                    self._opacity[:self.skybox_points],  # 天空球点保持不变
                    inverse_sigmoid(
                        torch.min(
                            self.get_opacity[self.skybox_points:],
                            torch.ones_like(self.get_opacity[self.skybox_points:]) * 0.01
                        )
                    )
                ),
                0
            )
        else:
            opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]  # [感知一致性]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # [感知一致性]
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        # 过滤低不透明度的点，避免对无效的“浮漂”点进行致密化
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)
        
        # 保护天空球点不参与密集化
        if self.skybox_points > 0:
            selected_pts_mask[:self.skybox_points] = False

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 过滤低不透明度的点
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)
        
        # 保护天空球点不参与密集化
        if self.skybox_points > 0:
            selected_pts_mask[:self.skybox_points] = False
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, max_grad_abs, min_opacity, extent, max_screen_size, prune_by_extent=True):


        n_init = self.get_xyz.shape[0]
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        # 获取 Clone 和 Split 数量
        clone_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        clone_mask = torch.logical_and(clone_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense*extent)
        if self.skybox_points > 0: clone_mask[:self.skybox_points] = False
        n_clone = clone_mask.sum().item()

        # Split: abs grad
        split_mask = torch.where(torch.norm(grads_abs, dim=-1) >= max_grad_abs, True, False)
        split_mask = torch.logical_and(split_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense*extent)
        if self.skybox_points > 0: split_mask[:self.skybox_points] = False
        n_split = split_mask.sum().item()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads_abs, max_grad_abs, extent)
        
        n_after_densify = self.get_xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        n_low_opacity = prune_mask.sum().item()
        n_big_points = 0
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            # OmniGS：prune_by_extent=0 时仅按屏幕半径剪「大点」，不按 world scale>0.1*extent（LonLat 下易误杀）
            if prune_by_extent:
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            else:
                big_points_ws = torch.zeros_like(big_points_vs, dtype=torch.bool, device=big_points_vs.device)
            n_big_points = torch.logical_or(big_points_vs, big_points_ws).sum().item()
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 保护天空球点不被剪枝
        if self.skybox_points > 0:
            prune_mask[:self.skybox_points] = False
        
        n_prune = prune_mask.sum().item()
        self.prune_points(prune_mask)
        
        n_final = self.get_xyz.shape[0]
        # print(f"[Densify] Total: {n_init} -> {n_after_densify} -> {n_final}")
        # print(f"[Densify] Clone: {n_clone}, Split: {n_split}")
        # print(f"[Densify] Prune: {n_prune} (Low Opacity: {n_low_opacity}, Big Points: {n_big_points})")

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """更新密集化统计信息

        viewspace_point_tensor.grad 形状 [P, 4] 
        """
        grad = viewspace_point_tensor.grad
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        if grad.shape[-1] >= 4:  
            self.xyz_gradient_accum_abs[update_filter] += torch.norm(grad[update_filter, 2:4], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
    
    # ==================== 感知一致性 Weight Sensitivity 函数 ====================
    
    def densify_and_split_fastgs(self, metric_mask, filter, N=2):
        """
        [感知一致性] 基于权重感知的高斯分裂
        
        Args:
            metric_mask: 贡献权重掩码（importance_score > 阈值）
            filter: 基于梯度和尺寸的候选掩码
            N: 分裂数量（默认2）
        """
        n_init_points = self.get_xyz.shape[0]
        
        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        mask = torch.logical_and(metric_mask, filter)
        selected_pts_mask[:mask.shape[0]] = mask
        
        # 保护天空球点
        if self.skybox_points > 0:
            selected_pts_mask[:self.skybox_points] = False
        
        if selected_pts_mask.sum() == 0:
            return
        
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def densify_and_clone_fastgs(self, metric_mask, filter):
        """
        [感知一致性] 高斯克隆
        
        Args:
            metric_mask: 贡献权重掩码（importance_score > 阈值）
            filter: 基于梯度和尺寸的候选掩码
        """
        selected_pts_mask = torch.logical_and(metric_mask, filter)
        
        # 保护天空球点
        if self.skybox_points > 0:
            selected_pts_mask[:self.skybox_points] = False
        
        if selected_pts_mask.sum() == 0:
            return
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
    
    def densify_and_prune_fastgs(self, max_screen_size, min_opacity, extent, radii, 
                                  grad_thresh=0.0002, importance_score=None, prune_by_extent=True):
        """
        [感知一致性] 基于权重感知的密集化和剪枝
        
        Args:
            max_screen_size: 最大屏幕尺寸阈值
            min_opacity: 最小不透明度阈值
            extent: 场景范围
            radii: 当前视图下的高斯半径
            grad_thresh: 梯度阈值
            importance_score: 重要性分数
        """
        # CityGS 原始模式：使用平均梯度 (grads / denom)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        # CityGS 原始分裂条件
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # 分裂候选点筛选（CityGS原始条件）
        split_pts_mask = torch.where(padded_grad >= grad_thresh, True, False)
        split_pts_mask = torch.logical_and(
            split_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * extent
        )
        split_pts_mask = torch.logical_and(split_pts_mask, self.get_opacity.flatten() > 0.15)
        
        # 克隆候选点筛选（CityGS原始条件）
        clone_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_thresh, True, False)
        clone_pts_mask = torch.logical_and(
            clone_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent
        )
        clone_pts_mask = torch.logical_and(clone_pts_mask, self.get_opacity.flatten() > 0.15)
        
        # 保护天空球点
        if self.skybox_points > 0:
            clone_pts_mask[:self.skybox_points] = False
            split_pts_mask[:self.skybox_points] = False
        
        # 动态阈值过滤
        if importance_score is not None:
            valid_scores = importance_score[importance_score > 0]
            if len(valid_scores) > 0:
                dynamic_thresh = valid_scores.quantile(0.5).item()
                final_thresh = max(0.1, dynamic_thresh)
            else:
                final_thresh = 0.1
            metric_mask = importance_score > final_thresh
        else:
            metric_mask = torch.ones(self.get_xyz.shape[0], dtype=bool, device="cuda")
        
        n_total = self.get_xyz.shape[0]
        n_clone_candidates = clone_pts_mask.sum().item()
        n_split_candidates = split_pts_mask.sum().item()
        
        final_clone_mask = torch.logical_and(metric_mask, clone_pts_mask)
        final_split_mask = torch.logical_and(metric_mask, split_pts_mask)
        n_final_clone = final_clone_mask.sum().item()
        n_final_split = final_split_mask.sum().item()
        
        self.densify_and_clone_fastgs(metric_mask, clone_pts_mask)
        self.densify_and_split_fastgs(metric_mask, split_pts_mask)
        
        n_after_densify = self.get_xyz.shape[0]

        # 剪枝（与CityGS原版保持一致）
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        n_low_opacity = prune_mask.sum().item()
        n_big_points = 0
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            if prune_by_extent:
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            else:
                big_points_ws = torch.zeros_like(big_points_vs, dtype=torch.bool, device=big_points_vs.device)
            n_big_points = torch.logical_or(big_points_vs, big_points_ws).sum().item()
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 保护天空球点不被剪枝
        if self.skybox_points > 0:
            prune_mask[:self.skybox_points] = False
        
        n_prune = prune_mask.sum().item()
        self.prune_points(prune_mask)
        
        n_final = self.get_xyz.shape[0]
        print(f"[Weight Sensitivity] Total: {n_total} -> {n_after_densify} -> {n_final}")
        print(f"[Weight Sensitivity] Clone: {n_clone_candidates}->{n_final_clone}, Split: {n_split_candidates}->{n_final_split}")
        print(f"[Weight Sensitivity] Prune: {n_prune} (Low Opacity: {n_low_opacity}, Big Points: {n_big_points})")

        torch.cuda.empty_cache()

class BlockedGaussian:

    gaussians : GaussianModel

    def __init__(self, gaussians, lp, range=[0, 1], scale=1.0, compute_cov3D_python=False):
        self.cell_corners = []
        self.avg_scalings = []
        self.feats = None
        self.max_sh_degree = lp.sh_degree
        self.device = gaussians.get_xyz.device
        self.compute_cov3D_python = compute_cov3D_python
        self.cell_idxs = [0]
        self.mask = torch.zeros(gaussians.get_opacity.shape[0], dtype=torch.bool, device=self.device)

        self.block_dim = lp.block_dim
        self.num_cell = lp.block_dim[0] * lp.block_dim[1] * lp.block_dim[2]
        self.aabb = lp.aabb
        self.block_splits = getattr(lp, "block_splits", None)
        self.scale = scale
        self.range = range

        self.cell_divider(gaussians)
        self.cell_corners = torch.stack(self.cell_corners, dim=0)

    def cell_divider(self, gaussians, n=4):
        with torch.no_grad():
            if self.compute_cov3D_python:
                geometry = gaussians.get_covariance(self.scale).to(self.device)
            else:
                geometry = torch.cat([gaussians.get_scaling,
                                      gaussians.get_rotation], dim=1)
            self.feats = torch.cat([gaussians.get_xyz,
                                    gaussians.get_opacity,  
                                    gaussians.get_features.reshape(geometry.shape[0], -1),
                                    geometry], dim=1)

            xyz = gaussians.get_xyz
            scaling = gaussians.get_scaling
            feat_list = []
            for cell_idx in range(self.num_cell):
                cell_mask = block_filtering(cell_idx, self.feats[:, :3], self.aabb, self.block_dim, self.scale,
                                           block_splits=self.block_splits)
                self.cell_idxs.append(self.cell_idxs[-1] + cell_mask.sum())
                feat_list.append(self.feats[cell_mask])
                # MAD to eliminate influence of outsiders
                xyz_median = torch.median(xyz[cell_mask], dim=0)[0]
                delta_median = torch.median(torch.abs(xyz[cell_mask] - xyz_median), dim=0)[0]
                xyz_min = xyz_median - n * delta_median
                xyz_min = torch.max(xyz_min, torch.min(xyz[cell_mask], dim=0)[0])
                xyz_max = xyz_median + n * delta_median
                xyz_max = torch.min(xyz_max, torch.max(xyz[cell_mask], dim=0)[0])
                corners = torch.tensor([[xyz_min[0], xyz_min[1], xyz_min[2]],
                                       [xyz_min[0], xyz_min[1], xyz_max[2]],
                                       [xyz_min[0], xyz_max[1], xyz_min[2]],
                                       [xyz_min[0], xyz_max[1], xyz_max[2]],
                                       [xyz_max[0], xyz_min[1], xyz_min[2]],
                                       [xyz_max[0], xyz_min[1], xyz_max[2]],
                                       [xyz_max[0], xyz_max[1], xyz_min[2]],
                                       [xyz_max[0], xyz_max[1], xyz_max[2]]], device=xyz.device)
                self.cell_corners.append(corners)
                self.avg_scalings.append(torch.mean(scaling[cell_mask], dim=0))
            
            self.feats = torch.cat(feat_list, dim=0)
            self.avg_scalings = torch.max(torch.stack(self.avg_scalings, dim=0), dim=-1).values
    
    def get_feats(self, indices):
        out = []
        if len(indices) > 0:
            for idx in indices:
                out.append(self.feats[self.cell_idxs[idx]:self.cell_idxs[idx+1]])
        return out

class GaussianModelLOD(GaussianModel):
    def __init__(self, 
                 sh_degree : int,
                 device='cuda'):
        super().__init__(sh_degree)
        self.device = device
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda(self.device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda(self.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda(self.device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        self.active_sh_degree = self.max_sh_degree
    
    def load_vq(self, path):
        # can't load from zip folder
        dequantized_feats = load_vqgaussian(os.path.join(path,'extreme_saving')).cpu().numpy()
        sh_dim = 3*(self.max_sh_degree + 1) ** 2 - 3 
        self.active_sh_degree = self.max_sh_degree
        # ic("in load_vq")
        # 24 for degree 2, and 45 for degree 3
        # abc = dequantized_feats[:, 0:3]
        
        xyz = dequantized_feats[:, 0:3]
        features_dc = dequantized_feats[:, 6:9]
        features_dc = features_dc.reshape((features_dc.shape[0],3,1))
        
        extra_f_names = dequantized_feats[:, 9:9+sh_dim]
        extra_f_names = extra_f_names.reshape((features_dc.shape[0],3,sh_dim//3))
        
        self._xyz = nn.Parameter(
            torch.tensor(dequantized_feats[:, 0:3], dtype=torch.float, device=self.device).requires_grad_(True)
        ) 
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(extra_f_names, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(dequantized_feats[:,-8:-7], dtype=torch.float, device=self.device).requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(dequantized_feats[:,-7:-4], dtype=torch.float, device=self.device).requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(dequantized_feats[:,-4:], dtype=torch.float, device=self.device).requires_grad_(True)
        )

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)