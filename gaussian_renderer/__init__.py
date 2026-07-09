import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel, GatheredGaussian
from utils.sh_utils import eval_sh
from utils.large_utils import in_frustum

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, metric_map = None):
    """
    Render the scene. 
    """
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        metric_map=metric_map,
        skybox_num=getattr(pc, 'skybox_points', 0),
        output_invdepth=getattr(pipe, 'output_depth', False),
        camera_type=getattr(viewpoint_camera, 'camera_type', 1),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, metric_count, invdepth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "metric_count": metric_count}
    if invdepth is not None and invdepth.numel() > 0:
        out["depth"] = invdepth
    return out

def _frustum_mask_pinhole(means3D, world_view_transform, tanfovx, tanfovy, margin=1.3):
    """Pre-filter Gaussians outside the perspective frustum on the Python side.

    The CUDA rasterizer's in_frustum only checks z > 0.2 (the NDC xy check
    is commented out), so Gaussians far outside the FOV still pass and their
    covariance tails bleed into the visible area, causing fog-like artifacts
    for models trained with panoramic (ERP) cameras.
    """
    # world_view_transform is stored as W2V.T (transposed in ViewerCam/Camera).
    # CUDA transformPoint4x3 treats the flat array as column-major, effectively
    # computing W2V @ p.  In batched Python: p_view = xyz_h @ W2V.T = xyz_h @ wvt.
    xyz_h = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    p_view = (xyz_h @ world_view_transform)[:, :3]  # (N, 3)  camera-space
    z = p_view[:, 2]
    valid_z = z > 0.2
    x_over_z = p_view[:, 0] / z.clamp(min=1e-6)
    y_over_z = p_view[:, 1] / z.clamp(min=1e-6)
    valid_x = x_over_z.abs() < margin * tanfovx
    valid_y = y_over_z.abs() < margin * tanfovy
    return valid_z & valid_x & valid_y


def render_large(cam_info, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, metric_map = None, lod_threshold = 0):
    """
    Render the scene for large scale.
    """
    # 兼容处理：cam_info 可能是字典也可能是 Camera 对象
    if isinstance(cam_info, dict):
        FoVx = cam_info["FoVx"]
        FoVy = cam_info["FoVy"]
        image_height = int(cam_info["image_height"])
        image_width = int(cam_info["image_width"])
        world_view_transform = cam_info["world_view_transform"]
        full_proj_transform = cam_info["full_proj_transform"]
        camera_center = cam_info["camera_center"]
        ct = int(cam_info.get("camera_type", 1))
    else:
        FoVx = cam_info.FoVx
        FoVy = cam_info.FoVy
        image_height = int(cam_info.image_height)
        image_width = int(cam_info.image_width)
        world_view_transform = cam_info.world_view_transform
        full_proj_transform = cam_info.full_proj_transform
        camera_center = cam_info.camera_center
        ct = int(getattr(cam_info, "camera_type", 1))

    # Set up rasterization configuration
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    means3D = pc.get_xyz
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # For pinhole cameras, pre-filter Gaussians outside the frustum to avoid
    # fog artifacts from ERP-trained models (CUDA in_frustum lacks NDC check).
    if ct == 1:
        with torch.no_grad():
            mask = _frustum_mask_pinhole(means3D, world_view_transform, tanfovx, tanfovy)
        means3D = means3D[mask]
        opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]

        # Mip-Splatting style 3D anti-aliasing: inflate thin Gaussian scales
        # so every axis projects to at least ~1 pixel. Prevents elongated
        # streak artifacts from ERP-trained Gaussians viewed in perspective.
        if scales is not None and cov3D_precomp is None:
            with torch.no_grad():
                focal = 0.5 * image_width / tanfovx
                dist = (means3D - camera_center).norm(dim=1, keepdim=True).clamp(min=0.1)
                min_scale = dist / focal  # 1-pixel footprint at this distance
                scales = torch.max(scales, min_scale)

    # Latitude-adaptive LoD for panoramic cameras: skip Gaussians whose
    # ERP pixel coverage falls below ``lod_threshold``.  Near the equator
    # (cos≈1) the threshold is strict (more Gaussians kept); near the poles
    # (cos→0) it is lenient (more Gaussians skipped).
    if ct == 3 and lod_threshold > 0 and scales is not None:
        with torch.no_grad():
            _dist = (means3D - camera_center).norm(dim=1).clamp(min=0.1)
            _max_scale = scales.max(dim=1).values
            _dir = means3D - camera_center
            # Transform to camera space so Y corresponds to ERP latitude,
            # independent of world coordinate conventions.
            _dir_cam = _dir @ world_view_transform[:3, :3]
            _lat = torch.asin((_dir_cam[:, 1] / _dist).clamp(-1, 1))
            _pix_cov = (_max_scale / _dist
                        * image_width * torch.cos(_lat).clamp(min=0.05)
                        / (2.0 * math.pi))
            lod_mask = _pix_cov >= lod_threshold
            skybox_num = getattr(pc, 'skybox_points', 0)
            if skybox_num > 0:
                lod_mask[:skybox_num] = True
            means3D = means3D[lod_mask]
            opacity = opacity[lod_mask]
            scales = scales[lod_mask]
            if rotations is not None:
                rotations = rotations[lod_mask]
            if cov3D_precomp is not None:
                cov3D_precomp = cov3D_precomp[lod_mask]
            if shs is not None:
                shs = shs[lod_mask]
            if colors_precomp is not None:
                colors_precomp = colors_precomp[lod_mask]

    raster_settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=pipe.debug,
        metric_map=metric_map,
        skybox_num=getattr(pc, 'skybox_points', 0),
        output_invdepth=getattr(pipe, 'output_depth', False),
        camera_type=ct,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # [AbsGS] [P, 4] placeholder; cols 2/3 receive component-wise absolute gradients
    screenspace_points = torch.zeros((means3D.shape[0], 4), dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    means2D = screenspace_points

    rendered_image, radii, metric_count, invdepth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    out = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "metric_count": metric_count}
    if invdepth is not None and invdepth.numel() > 0:
        out["depth"] = invdepth
    return out

def render_lod(viewpoint_cam, lod_list : list, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    # ... lod 渲染逻辑保持简洁承接 ...
    in_frustum_mask, distance3D = in_frustum(viewpoint_cam, lod_list[-1].cell_corners, lod_list[-1].aabb, lod_list[-1].block_dim)
    in_frustum_indices = in_frustum_mask.nonzero().squeeze(0)
    
    focal_length = 0.5 * viewpoint_cam.image_width / math.tan(viewpoint_cam.FoVx * 0.5)
    nyquist_scalings = 2 * distance3D / focal_length
    avg_scalings = torch.stack([lod_list[i].avg_scalings for i in range(len(lod_list))], dim=0)[:, in_frustum_mask]
    
    # compare avg_scalings with nyquist_scalings to decide which lod to use
    values, lod_indices = torch.max((avg_scalings > nyquist_scalings.unsqueeze(0)).to(torch.uint8), dim=0)
    lod_indices[values==0] = len(lod_list) - 1
    
    # used for BlockedGaussianV3
    out_list = []
    main_device = lod_list[-1].feats.device
    max_sh_degree = lod_list[-1].max_sh_degree
    feat_end_dim = 3 * (max_sh_degree + 1) ** 2 + 4
    
    for lod_idx, lod_gs in enumerate(lod_list):
        out_i = lod_gs.get_feats(in_frustum_indices[lod_indices==lod_idx])
        out_list += out_i

    feats = torch.cat(out_list, dim=0)

    means3D = feats[:, :3].float()
    # [AbsGS] [P, 4] placeholder for consistency with the [P, 4] backward gradient layout
    screenspace_points = torch.zeros((means3D.shape[0], 4), dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    means2D = screenspace_points
    opacity = feats[:, 3].float()
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = feats[:, feat_end_dim:].float()
    else:
        scales = feats[:, feat_end_dim:feat_end_dim+3].float()
        rotations = feats[:, (feat_end_dim+3):].float()
        
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        features = feats[:, 4:feat_end_dim].reshape(-1, (max_sh_degree+1)**2, 3).float()
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_cam.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(max_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color  # check if requires masking
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=max_sh_degree,
        campos=viewpoint_cam.camera_center, 
        prefiltered=False,
        debug=pipe.debug,
        metric_map=None,
        skybox_num=0,
        output_invdepth=False,
        camera_type=int(getattr(viewpoint_cam, "camera_type", 1)),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii, metric_count, _invdepth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_viewer(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if isinstance(pc, GaussianModel):
        return render_large(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    else:
        return render_lod(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
