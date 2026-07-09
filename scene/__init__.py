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
import tqdm
import random
import json
import yaml
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import BasicPointCloud
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly, SceneInfo
from scene.gaussian_model import GaussianModel, GaussianModelLOD, GatheredGaussian
from arguments import ModelParams, GroupParams
from plyfile import PlyData, PlyElement
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.alpha_masks,
                args.eval,
                use_alpha_masks=getattr(args, "use_alpha_masks", True),
                use_sky_masks=getattr(args, "use_sky_masks", True),
                use_depth=getattr(args, "use_depth", True),
                default_camera_type=int(getattr(args, "default_camera_type", 1)),
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self._write_cameras_json(scene_info)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffcameras_extentling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def _write_cameras_json(self, scene_info):
        """将 train+test 相机按 image_name 排序后写入 cameras.json（始终重新生成）。"""
        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
            dest_file.write(src_file.read())
        camlist = list(scene_info.test_cameras or []) + list(scene_info.train_cameras or [])
        camlist = sorted(camlist, key=lambda c: c.image_name)
        json_cams = [camera_to_JSON(i, cam) for i, cam in enumerate(camlist)]
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
class LargeScene(Scene):
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, load_vq=False, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.load_vq = load_vq
        self.gaussians = gaussians
        self.pretrain_path = args.pretrain_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if args.block_id >= 0:
            partition = np.load(os.path.join(args.source_path, "data_partitions", f"{args.partition_name}.npy"))[:, args.block_id]
            if args.aabb is None:
                args.aabb = np.load(os.path.join(args.source_path, "data_partitions", f"{args.partition_name}_aabb.npy")).tolist()
            splits_path = os.path.join(args.source_path, "data_partitions", f"{args.partition_name}_block_splits.json")
            if os.path.isfile(splits_path):
                with open(splits_path, "r") as f:
                    args.block_splits = json.load(f)
                print(f"Using adaptive block_splits from {splits_path}")
            print(f"Using Partition File {args.partition_name}.npy")
        else:
            partition = None

        kf_a = os.path.join(args.source_path, "keyframes.json")
        kf_b = os.path.join(args.source_path, "openmvg_keyframes.json")
        if os.path.isfile(kf_a) or os.path.isfile(kf_b):
            # OmniGS / openMVG keyframes JSON (LonLat); see dataset_readers.readOpenMVGSceneInfo
            scene_info = sceneLoadTypeCallbacks["OpenMVG"](
                args.source_path,
                args.images,
                args.alpha_masks,
                args.eval,
                args.llffhold,
                partition=partition,
                use_alpha_masks=getattr(args, "use_alpha_masks", True),
                use_sky_masks=getattr(args, "use_sky_masks", True),
                use_depth=getattr(args, "use_depth", True),
            )
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.alpha_masks,
                args.eval,
                args.llffhold,
                partition=partition,
                use_alpha_masks=getattr(args, "use_alpha_masks", True),
                use_sky_masks=getattr(args, "use_sky_masks", True),
                use_depth=getattr(args, "use_depth", True),
                default_camera_type=int(getattr(args, "default_camera_type", 1)),
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        self._write_cameras_json(scene_info)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.train_cameras = scene_info.train_cameras
        self.test_cameras = scene_info.test_cameras

        # 按几何分区过滤 test 相机：仅保留中心落在当前 block 的测试视角
        if args.block_id >= 0 and len(self.test_cameras) > 0 and args.aabb is not None:
            from utils.large_utils import contract_to_unisphere
            from utils.graphics_utils import getWorld2View2
            aabb = args.aabb
            if not isinstance(aabb, torch.Tensor):
                aabb = torch.tensor(aabb, dtype=torch.float32)
            dx, dy, dz = args.block_dim
            bid = args.block_id
            bz_t = bid // (dx * dy)
            by_t = (bid % (dx * dy)) // dx
            bx_t = (bid % (dx * dy)) % dx
            filtered_test = []
            for c in self.test_cameras:
                W2C = getWorld2View2(c.R, c.T)
                C2W = np.linalg.inv(W2C)
                cc = contract_to_unisphere(
                    torch.tensor(C2W[:3, 3], dtype=torch.float32),
                    aabb, ord=torch.inf,
                )
                bx = int(torch.floor((cc[0] * dx).clamp(0, dx - 1)).item())
                by = int(torch.floor((cc[1] * dy).clamp(0, dy - 1)).item())
                bz = int(torch.floor((cc[2] * dz).clamp(0, dz - 1)).item())
                if bx == bx_t and by == by_t and bz == bz_t:
                    filtered_test.append(c)
            print(f"Filtered Test Cameras by block {bid}: {len(filtered_test)}/{len(self.test_cameras)}")
            self.test_cameras = filtered_test

        if self.load_vq:
            self.gaussians.load_vq(self.model_path)
        elif self.loaded_iter:
            ply_path = os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter))
            self.gaussians.load_ply(os.path.join(ply_path, "point_cloud.ply"))
            # 读取天空球信息
            pc_info_path = os.path.join(ply_path, "pc_info.txt")
            if os.path.exists(pc_info_path):
                with open(pc_info_path, "r") as f:
                    self.gaussians.skybox_points = int(f.readline().strip())
                print(f"Loaded skybox_points: {self.gaussians.skybox_points}")
        elif self.pretrain_path:
            self.gaussians.load_ply(os.path.join(self.pretrain_path, "point_cloud.ply"))
            self.gaussians.spatial_lr_scale = self.cameras_extent
            # 读取天空球信息
            pc_info_path = os.path.join(self.pretrain_path, "pc_info.txt")
            if os.path.exists(pc_info_path):
                with open(pc_info_path, "r") as f:
                    self.gaussians.skybox_points = int(f.readline().strip())
                # Block 训练时设置 skybox_locked（从命令行参数读取）
                self.gaussians.skybox_locked = getattr(args, 'skybox_locked', True)  # Block 默认锁定
                print(f"Loaded skybox_points from pretrain: {self.gaussians.skybox_points}, locked={self.gaussians.skybox_locked}")
        else:
            if args.add_background_sphere:
                import math
                scene_center = -scene_info.nerf_normalization['translate']
                scene_radius = scene_info.nerf_normalization['radius']
                # build unit sphere points
                n_points = args.background_sphere_points
                samples = np.arange(n_points)
                y = 1 - (samples / float(n_points - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y
                phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
                theta = phi * samples  # golden angle increment
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                unit_sphere_points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
                # build background sphere
                background_sphere_point_xyz = (unit_sphere_points * scene_radius * args.background_sphere_radius) + scene_center
                background_sphere_point_rgb = np.asarray(np.random.random(background_sphere_point_xyz.shape), dtype=np.float64)
                # add background sphere to scene
                scene_info = SceneInfo(
                    point_cloud=BasicPointCloud(
                                points=np.concatenate([scene_info.point_cloud.points, background_sphere_point_xyz], axis=0),
                                colors=np.concatenate([scene_info.point_cloud.colors, background_sphere_point_rgb], axis=0),
                                normals=np.zeros_like(background_sphere_point_xyz)),
                    train_cameras=scene_info.train_cameras,
                    test_cameras=scene_info.test_cameras,
                    nerf_normalization=scene_info.nerf_normalization,
                    ply_path=scene_info.ply_path)
                # increase prune extent
                # TODO: resize scene_extent without changing lr
                self.cameras_extent = scene_radius * args.background_sphere_radius * 1.0001

                print("added {} background sphere points, rescale prune extent from {} to {}".format(n_points, scene_radius, self.cameras_extent))

            # 支持天空球参数
            skybox_num = getattr(args, 'skybox_num', 0)
            skybox_locked = getattr(args, 'skybox_locked', False)
            scene_center = -scene_info.nerf_normalization['translate']
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, skybox_num, skybox_locked, scene_center=scene_center)
    
    def save(self, iteration, args=None):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))

        if args.block_id >= 0:
            xyz_org = self.gaussians.get_xyz
            if len(args.aabb) == 4:
                aabb = [args.aabb[0], args.aabb[1], xyz_org[:, -1].min(), 
                        args.aabb[2], args.aabb[3], xyz_org[:, -1].max()]
            elif len(args.aabb) == 6:
                aabb = args.aabb
            else:
                assert False, "Unknown aabb format!"
            aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_org.device)
            xyz_contracted = self.contract_to_unisphere(xyz_org, aabb, ord=torch.inf)
            block_id_z = args.block_id // (args.block_dim[0] * args.block_dim[1])
            block_id_y = (args.block_id % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[0]
            block_id_x = (args.block_id % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[0]

            min_x, max_x = float(block_id_x) / args.block_dim[0], float(block_id_x + 1) / args.block_dim[0]
            min_y, max_y = float(block_id_y) / args.block_dim[1], float(block_id_y + 1) / args.block_dim[1]
            min_z, max_z = float(block_id_z) / args.block_dim[2], float(block_id_z + 1) / args.block_dim[2]

            block_mask = (xyz_contracted[:, 0] >= min_x) & (xyz_contracted[:, 0] < max_x)  \
                        & (xyz_contracted[:, 1] >= min_y) & (xyz_contracted[:, 1] < max_y) \
                        & (xyz_contracted[:, 2] >= min_z) & (xyz_contracted[:, 2] < max_z)
            
            # 排除天空球点（天空球属于全局，不应被分割到单个 block）
            if self.gaussians.skybox_points > 0:
                skybox_exclude_mask = torch.ones(xyz_org.shape[0], dtype=torch.bool, device=xyz_org.device)
                skybox_exclude_mask[:self.gaussians.skybox_points] = False
                block_mask = block_mask & skybox_exclude_mask
            
            sh_degree = self.gaussians.max_sh_degree
            masked_gaussians = GaussianModel(sh_degree)
            masked_gaussians._xyz = self.gaussians.get_xyz[block_mask]
            masked_gaussians._scaling = self.gaussians._scaling[block_mask]
            masked_gaussians._rotation = self.gaussians._rotation[block_mask]
            masked_gaussians._features_dc = self.gaussians._features_dc[block_mask]
            masked_gaussians._features_rest = self.gaussians._features_rest[block_mask]
            masked_gaussians._opacity = self.gaussians._opacity[block_mask]
            masked_gaussians.max_radii2D = self.gaussians.max_radii2D[block_mask]

            block_point_cloud_path = os.path.join(self.model_path, "point_cloud_blocks/scale_1.0/iteration_{}".format(iteration))
            masked_gaussians.save_ply(os.path.join(block_point_cloud_path, "point_cloud.ply"))

            if args.save_block_only:
                return
        
        # 保存完整点云（天空球点在最前面）
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        # 保存天空球点数量（参考 LDGS）
        # 合并时从 coarse 的 point_cloud.ply 取前 skybox_points 个点即可
        with open(os.path.join(point_cloud_path, "pc_info.txt"), "w") as f:
            f.write(str(self.gaussians.skybox_points))
        
        if self.gaussians.skybox_points > 0:
            print(f"Saved with {self.gaussians.skybox_points} skybox points (indices 0:{self.gaussians.skybox_points})")
    
    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: float = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x
