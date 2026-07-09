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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 sky_mask=None, invdepthmap=None, depth_params=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 camera_type=1,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        # 1 = pinhole (COLMAP), 3 = OmniGS LonLat / equirectangular (GPL raster path)
        self.camera_type = int(camera_type)
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
            self.original_image *= self.gt_alpha_mask
        else:
            self.gt_alpha_mask = None
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        if sky_mask is not None:
            self.sky_mask = sky_mask.to(self.data_device)
        else:
            self.sky_mask = None

        self.invdepthmap = None
        self.depth_mask = None
        self.depth_reliable = False
        if invdepthmap is not None:
            invdepth_raw = invdepthmap.to(self.data_device)
            depth_eps = 1e-6
            depth_scale = depth_params.get("scale", 0) if depth_params is not None else 0
            if depth_scale > 0:
                off = depth_params.get("offset", 0.0)
                aligned = invdepth_raw * depth_scale + off
                invdepth = torch.where(invdepth_raw > depth_eps, aligned, torch.zeros_like(invdepth_raw))
            else:
                invdepth = invdepth_raw
            invdepth = torch.clamp_min(invdepth, 0.0)
            self.invdepthmap = invdepth

            if self.gt_alpha_mask is not None:
                self.depth_mask = self.gt_alpha_mask.clone()
            else:
                self.depth_mask = torch.ones_like(self.invdepthmap, dtype=torch.float32)
            if self.sky_mask is not None:
                self.depth_mask *= self.sky_mask
            else:
                self.depth_mask *= (self.invdepthmap > 1e-6).float()

            med_scale = depth_params.get("med_scale", 0) if depth_params is not None else 0
            if med_scale > 0 and depth_scale > 0:
                if depth_scale < 0.2 * med_scale or depth_scale > 5.0 * med_scale:
                    self.depth_mask.fill_(0)
                else:
                    self.depth_reliable = True
            else:
                self.depth_reliable = True

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, camera_type=1):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_type = int(camera_type)
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class LightCam(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, data_device = "cuda",
                 camera_type=1,
                 ):
        super(LightCam, self).__init__()

        # 1 = pinhole (COLMAP), 3 = OmniGS LonLat / equirectangular (panorama path)
        # 没有 camera_type 时默认针孔，分区/渲染走 pinhole 公式（向后兼容）
        self.camera_type = int(camera_type)
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class ViewerCam(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, data_device = "cuda",
                 camera_type=1,
                 ):
        super(ViewerCam, self).__init__()

        self.camera_type = int(camera_type)
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = width
        self.image_height = height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R.transpose(), T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
