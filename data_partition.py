import os
import sys
import json
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser, Namespace
from transforms3d.quaternions import mat2quat
from scene import LargeScene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state, parse_cfg
from utils.large_utils import (
    contract_to_unisphere,
    get_default_aabb,
    get_aabb_from_cameras,
    get_aabb_from_cameras_pano,
    _get_block_bounds,
)
from utils.loss_utils import ssim, ssim_erp_weighted, l1_loss
from utils.camera_utils import loadCam_woImage, loadCam
from utils.graphics_utils import getWorld2View2
from arguments import GroupParams

def block_partitioning(cameras, gaussians, args, pp, scale=1.0, quiet=False, disable_inblock=False, simple_selection=False):

        xyz_org = gaussians.get_xyz
        num_threshold = args.num_threshold
        block_num = args.block_dim[0] * args.block_dim[1] * args.block_dim[2]

        if args.aabb is None:
            torch.cuda.empty_cache()
            mode = getattr(args, "aabb_autofit_mode", None)
            cam_type = int(getattr(args, "default_camera_type", 1))
            # 全景 (ERP/LonLat) 数据集且 yaml 未显式指定 → 自动走全景版 AABB
            if mode is None or (isinstance(mode, str) and mode.strip() == ""):
                mode = "cameras_pano" if cam_type == 3 else "focus"
            if mode == "cameras_pano":
                args.aabb = get_aabb_from_cameras_pano(args, cameras, xyz_org, scale)
                print("AABB autofit: cameras_pano (mean center + per-axis percentile radius + baseline*factor margin)")
            elif mode == "cameras":
                args.aabb = get_aabb_from_cameras(args, cameras, xyz_org, scale)
                print("AABB autofit: cameras (min/max centers + optional gaussian union + margin)")
            else:
                args.aabb = get_default_aabb(args, cameras, xyz_org, scale)
                print("AABB autofit: focus (original CityGS-style)")
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            np.save(os.path.join(args.source_path, "data_partitions", f"{config_name}_aabb.npy"), np.array(args.aabb.detach().cpu()))
        else:
            assert len(args.aabb) == 6, "Unknown args.aabb format!"
            args.aabb = torch.tensor(args.aabb, dtype=torch.float32, device=xyz_org.device)
        
        print(f"Block number: {block_num}, Gaussian number threshold: {num_threshold}")

        cam_type = int(getattr(args, "default_camera_type", 1))
        is_pano = (cam_type == 3)

        camera_mask = torch.zeros((len(cameras), block_num), dtype=torch.bool, device=xyz_org.device)
        
        with torch.no_grad():

            for block_id in range(block_num):
                block_id_z = block_id // (args.block_dim[0] * args.block_dim[1])
                block_id_y = (block_id % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[0]
                block_id_x = (block_id % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[0]

                xyz = contract_to_unisphere(xyz_org, args.aabb, ord=torch.inf)
                min_x, max_x = float(block_id_x) / args.block_dim[0], float(block_id_x + 1) / args.block_dim[0]
                min_y, max_y = float(block_id_y) / args.block_dim[1], float(block_id_y + 1) / args.block_dim[1]
                min_z, max_z = float(block_id_z) / args.block_dim[2], float(block_id_z + 1) / args.block_dim[2]

                num_gs, org_min_x, org_max_x, org_min_y, org_max_y, org_min_z, org_max_z = 0, min_x, max_x, min_y, max_y, min_z, max_z

                while num_gs < num_threshold:
                    # TODO: select better threshold
                    block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                                & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                                & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)
                    num_gs = block_mask.sum()
                    min_x -= 0.01
                    max_x += 0.01
                    min_y -= 0.01
                    max_y += 0.01
                    min_z -= 0.01
                    max_z += 0.01
                
                block_mask = ~block_mask
                sh_degree = gaussians.max_sh_degree
                masked_gaussians = GaussianModel(sh_degree)
                masked_gaussians._xyz = xyz_org[block_mask]
                masked_gaussians._scaling = gaussians._scaling[block_mask]
                masked_gaussians._rotation = gaussians._rotation[block_mask]
                masked_gaussians._features_dc = gaussians._features_dc[block_mask]
                masked_gaussians._features_rest = gaussians._features_rest[block_mask]
                masked_gaussians._opacity = gaussians._opacity[block_mask]
                masked_gaussians.max_radii2D = gaussians.max_radii2D[block_mask]

                for idx in tqdm(range(len(cameras)), desc=f"Block {block_id} / {block_num}"):
                    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device=xyz_org.device)
                    c = cameras[idx]
                    viewpoint_cam = loadCam_woImage(args, idx, c, scale)
                    contract_cam_center = contract_to_unisphere(viewpoint_cam.camera_center, args.aabb, ord=torch.inf)

                    if simple_selection > 1.0:
                        # enlarge the box to 1.5x
                        rate = (simple_selection - 1.0) / 2
                        min_x -= rate * (org_max_x - org_min_x)
                        max_x += rate * (org_max_x - org_min_x)
                        min_y -= rate * (org_max_y - org_min_y)
                        max_y += rate * (org_max_y - org_min_y)
                        min_z -= rate * (org_max_z - org_min_z)
                        max_z += rate * (org_max_z - org_min_z)
                        if contract_cam_center[0] > min_x and contract_cam_center[0] < max_x \
                            and contract_cam_center[1] > min_y and contract_cam_center[1] < max_y \
                            and contract_cam_center[2] > min_z and contract_cam_center[2] < max_z :
                            camera_mask[idx, block_id] = True
                        continue

                    if (not disable_inblock) and contract_cam_center[0] > org_min_x and contract_cam_center[0] < org_max_x \
                        and contract_cam_center[1] > org_min_y and contract_cam_center[1] < org_max_y \
                        and contract_cam_center[2] > org_min_z and contract_cam_center[2] < org_max_z :
                        camera_mask[idx, block_id] = True
                        continue

                    render_pkg_block = render(viewpoint_cam, gaussians, pp, background)

                    org_image_block = render_pkg_block["render"]
                    render_pkg_block = render(viewpoint_cam, masked_gaussians, pp, background)
                    image_block = render_pkg_block["render"]
                    if is_pano:
                        loss = 1.0 - ssim_erp_weighted(image_block, org_image_block)
                    else:
                        loss = 1.0 - ssim(image_block, org_image_block)
                    if loss > args.ssim_threshold:
                        camera_mask[idx, block_id] = True
        
        if not quiet:
            for block_id in range(block_num):
                print(f"Block {block_id} / {block_num} has {camera_mask[:, block_id].sum()} cameras.")
                    
        return camera_mask


def _compute_adaptive_splits(cam_coords_contracted, block_dim):
    """Compute quantile-based split boundaries from camera centers in contracted
    [0,1]^3 space so that each block gets roughly equal number of cameras.

    Returns a dict ``{"x": [...], "y": [...], "z": [...]}`` where each value is
    a sorted list of length ``dim_size + 1`` starting at 0.0 and ending at 1.0.
    """
    splits = {}
    for axis, (dim_size, label) in enumerate(
        zip(block_dim, ["x", "y", "z"])
    ):
        if dim_size <= 1:
            splits[label] = [0.0, 1.0]
        else:
            vals = cam_coords_contracted[:, axis]
            quantiles = [i / dim_size for i in range(1, dim_size)]
            boundaries = np.quantile(vals, quantiles).tolist()
            splits[label] = [0.0] + boundaries + [1.0]
    return splits


def block_partitioning_geometry(cameras, gaussians, args, scale=1.0, quiet=False):
    """纯几何分区：仅用相机中心在 contracted 空间 [0,1]^3 内的网格坐标分配唯一 block，
    与 utils.large_utils 中 xyz→block_id 的规则一致，不做 SSIM / 梯度渲染。

    当 args.adaptive_partition == True 时使用分位数自适应切割边界，使各 block 相机数
    量大致均衡；否则回退到均匀网格。"""
    xyz_org = gaussians.get_xyz
    block_num = args.block_dim[0] * args.block_dim[1] * args.block_dim[2]
    dx, dy, dz = args.block_dim[0], args.block_dim[1], args.block_dim[2]
    adaptive = bool(getattr(args, "adaptive_partition", False))

    if args.aabb is None:
        torch.cuda.empty_cache()
        mode = getattr(args, "aabb_autofit_mode", None)
        cam_type = int(getattr(args, "default_camera_type", 1))
        if mode is None or (isinstance(mode, str) and mode.strip() == ""):
            mode = "cameras_pano" if cam_type == 3 else "focus"
        if mode == "cameras_pano":
            args.aabb = get_aabb_from_cameras_pano(args, cameras, xyz_org, scale)
            print("AABB autofit: cameras_pano (geometry partition)")
        elif mode == "cameras":
            args.aabb = get_aabb_from_cameras(args, cameras, xyz_org, scale)
            print("AABB autofit: cameras (geometry partition)")
        else:
            args.aabb = get_default_aabb(args, cameras, xyz_org, scale)
            print("AABB autofit: focus (geometry partition)")
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        np.save(
            os.path.join(args.source_path, "data_partitions", f"{config_name}_aabb.npy"),
            np.array(args.aabb.detach().cpu()),
        )
    else:
        assert len(args.aabb) == 6, "Unknown args.aabb format!"
        args.aabb = torch.tensor(args.aabb, dtype=torch.float32, device=xyz_org.device)

    # --- Collect all camera centers in contracted space ---
    cam_cc_list = []
    with torch.no_grad():
        for idx in tqdm(range(len(cameras)), desc="Geometry partition (collect centers)"):
            c = cameras[idx]
            viewpoint_cam = loadCam_woImage(args, idx, c, scale)
            cc = contract_to_unisphere(viewpoint_cam.camera_center, args.aabb, ord=torch.inf)
            cam_cc_list.append(cc.cpu().numpy())
    cam_cc_all = np.stack(cam_cc_list, axis=0)  # (N, 3)

    # --- Compute split boundaries ---
    block_splits = None
    if adaptive:
        block_splits = _compute_adaptive_splits(cam_cc_all, [dx, dy, dz])
        args.block_splits = block_splits
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        splits_path = os.path.join(
            args.source_path, "data_partitions", f"{config_name}_block_splits.json")
        with open(splits_path, "w") as f:
            json.dump(block_splits, f, indent=2)
        print(f"Adaptive partition splits: {block_splits}")
        print(f"Partition mode: geometry (adaptive quantile) | blocks: {block_num}")
    else:
        print(f"Partition mode: geometry (uniform grid) | blocks: {block_num}")

    # --- Assign cameras to blocks ---
    camera_mask = torch.zeros((len(cameras), block_num), dtype=torch.bool, device=xyz_org.device)

    for idx in range(len(cameras)):
        cc = cam_cc_all[idx]
        if block_splits is not None:
            bx = int(np.searchsorted(block_splits["x"][1:-1], cc[0]))
            by = int(np.searchsorted(block_splits["y"][1:-1], cc[1]))
            bz = int(np.searchsorted(block_splits["z"][1:-1], cc[2]))
            bx = min(max(bx, 0), dx - 1)
            by = min(max(by, 0), dy - 1)
            bz = min(max(bz, 0), dz - 1)
        else:
            bx = min(int(cc[0] * dx), dx - 1)
            by = min(int(cc[1] * dy), dy - 1)
            bz = min(int(cc[2] * dz), dz - 1)
            bx = max(bx, 0)
            by = max(by, 0)
            bz = max(bz, 0)
        bid = bz * (dx * dy) + by * dx + bx
        camera_mask[idx, bid] = True

    if not quiet:
        for block_id in range(block_num):
            print(f"Block {block_id} / {block_num} has {camera_mask[:, block_id].sum()} cameras.")

    return camera_mask


def block_partitioning_gradient(cameras, gaussians, args, pp, op=None,
                                scale=1.0, quiet=False):
    """Gradient-aware partitioning for panoramic (ERP) cameras.

    For each camera we run one forward + backward (unless fully covered by geometry)
    and aggregate a per-block signal: mean |grad(xyz)| inside the block, weighted
    by 1 / dist^2 to the block centre in contracted space.

    A camera is assigned to block b if (1) its centre lies in the block's nominal
    AABB, or (2) ``signals[b] / max(signals) > grad_ratio_threshold``.

    Per-block signals must be filled for *all* blocks before dividing by
    ``max(signals)``.  Skipping blocks that already pass the geometric test would
    make ``max_sig`` come from a single block and force ratio 1.0, incorrectly
    assigning every other block.

    This avoids the SSIM remove-and-compare strategy which always triggers for
    360-degree cameras because removing *any* block visibly changes the panorama.
    """
    xyz_org = gaussians.get_xyz
    num_threshold = args.num_threshold
    block_num = args.block_dim[0] * args.block_dim[1] * args.block_dim[2]
    grad_ratio_threshold = float(getattr(args, 'grad_ratio_threshold', 0.1))
    skip_bottom_ratio = float(getattr(op, 'skip_bottom_ratio',
                              getattr(args, 'skip_bottom_ratio', 0.0)))
    cam_type = int(getattr(args, "default_camera_type", 1))
    is_pano = (cam_type == 3)
    lambda_dssim = float(getattr(op, 'lambda_dssim',
                          getattr(args, 'lambda_dssim', 0.2)))

    # ---------- AABB (reuse logic from block_partitioning) ----------
    if args.aabb is None:
        torch.cuda.empty_cache()
        mode = getattr(args, "aabb_autofit_mode", None)
        if mode is None or (isinstance(mode, str) and mode.strip() == ""):
            mode = "cameras_pano" if cam_type == 3 else "focus"
        if mode == "cameras_pano":
            args.aabb = get_aabb_from_cameras_pano(args, cameras, xyz_org, scale)
            print("AABB autofit: cameras_pano")
        elif mode == "cameras":
            args.aabb = get_aabb_from_cameras(args, cameras, xyz_org, scale)
            print("AABB autofit: cameras")
        else:
            args.aabb = get_default_aabb(args, cameras, xyz_org, scale)
            print("AABB autofit: focus")
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        np.save(os.path.join(args.source_path, "data_partitions",
                             f"{config_name}_aabb.npy"),
                np.array(args.aabb.detach().cpu()))
    else:
        assert len(args.aabb) == 6, "Unknown args.aabb format!"
        args.aabb = torch.tensor(args.aabb, dtype=torch.float32,
                                 device=xyz_org.device)

    print(f"Block number: {block_num}, Gaussian number threshold: {num_threshold}")
    print(f"Partition mode: gradient | grad_ratio_threshold: {grad_ratio_threshold}")

    # ---------- Pre-compute block boundaries & Gaussian masks ----------
    xyz_contracted = contract_to_unisphere(xyz_org, args.aabb, ord=torch.inf)

    block_splits = getattr(args, "block_splits", None)

    block_info = []  # list of dicts with boundary & gs mask info
    for block_id in range(block_num):
        org_min_x, org_max_x, org_min_y, org_max_y, org_min_z, org_max_z = \
            _get_block_bounds(block_id, args.block_dim, block_splits)

        # Expand until num_threshold Gaussians are inside (same as original)
        mn_x, mx_x = org_min_x, org_max_x
        mn_y, mx_y = org_min_y, org_max_y
        mn_z, mx_z = org_min_z, org_max_z
        num_gs = 0
        while num_gs < num_threshold:
            gs_mask = ((xyz_contracted[:, 0] >= mn_x) & (xyz_contracted[:, 0] < mx_x)
                       & (xyz_contracted[:, 1] >= mn_y) & (xyz_contracted[:, 1] < mx_y)
                       & (xyz_contracted[:, 2] >= mn_z) & (xyz_contracted[:, 2] < mx_z))
            num_gs = gs_mask.sum()
            mn_x -= 0.01; mx_x += 0.01
            mn_y -= 0.01; mx_y += 0.01
            mn_z -= 0.01; mx_z += 0.01

        block_info.append({
            "org_bounds": (org_min_x, org_max_x, org_min_y, org_max_y,
                           org_min_z, org_max_z),
            "gs_mask": gs_mask,  # BoolTensor [N]
        })

    camera_mask = torch.zeros((len(cameras), block_num), dtype=torch.bool,
                              device=xyz_org.device)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=xyz_org.device)

    for idx in tqdm(range(len(cameras)), desc="Gradient partition"):
        c = cameras[idx]

        # --- Geometric in-block (camera center inside nominal block AABB) ---
        light_cam = loadCam_woImage(args, idx, c, scale)
        contract_cam_center = contract_to_unisphere(
            light_cam.camera_center, args.aabb, ord=torch.inf)

        geom_mask = torch.zeros(block_num, dtype=torch.bool, device=xyz_org.device)
        for block_id in range(block_num):
            mn_x, mx_x, mn_y, mx_y, mn_z, mx_z = block_info[block_id]["org_bounds"]
            if (contract_cam_center[0] > mn_x and contract_cam_center[0] < mx_x
                    and contract_cam_center[1] > mn_y and contract_cam_center[1] < mx_y
                    and contract_cam_center[2] > mn_z and contract_cam_center[2] < mx_z):
                geom_mask[block_id] = True

        if geom_mask.all():
            camera_mask[idx] = geom_mask
            continue  # every block covered geometrically (rare)

        # --- Gradient signal: must compute for *all* blocks so max_sig is global ---
        viewpoint_cam = loadCam(args, idx, c, scale)
        gt_image = viewpoint_cam.original_image.cuda()

        # Crop bottom for ERP (consistent with training loss)
        if is_pano and skip_bottom_ratio > 0:
            crop_rows = int(round(gt_image.shape[-2] * skip_bottom_ratio))
            if crop_rows > 0:
                gt_image = gt_image[:, :-crop_rows, :]

        for attr in ('_xyz', '_features_dc', '_features_rest',
                     '_scaling', '_rotation', '_opacity'):
            p = getattr(gaussians, attr, None)
            if p is not None and p.grad is not None:
                p.grad = None
        render_pkg = render(viewpoint_cam, gaussians, pp, background)
        image = render_pkg["render"]

        if is_pano and skip_bottom_ratio > 0:
            crop_rows = int(round(image.shape[-2] * skip_bottom_ratio))
            if crop_rows > 0:
                image = image[:, :-crop_rows, :]

        if is_pano:
            loss = (1.0 - lambda_dssim) * l1_loss(image, gt_image) \
                   + lambda_dssim * (1.0 - ssim_erp_weighted(image, gt_image))
        else:
            loss = (1.0 - lambda_dssim) * l1_loss(image, gt_image) \
                   + lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        grad_mag = gaussians._xyz.grad.detach().abs().sum(dim=1)  # [N]

        # Aggregate per-block gradient signal.
        # Use mean |grad(xyz)| directly — the rendering gradient already
        # encodes distance (nearby Gaussians project to more pixels and
        # contribute more to the loss, yielding larger gradients).
        signals = torch.zeros(block_num, device=xyz_org.device)
        for block_id in range(block_num):
            mask = block_info[block_id]["gs_mask"]
            if mask.sum() == 0:
                continue
            signals[block_id] = grad_mag[mask].mean()

        max_sig = signals.max().item()
        # geom_mask: always train blocks whose cell contains the camera centre.
        # Gradient ratio uses global max_sig over *all* blocks — otherwise skipping
        # geom-assigned blocks leaves max_sig from a single block and ratio==1.
        if max_sig > 0:
            ratio = signals / max_sig
            camera_mask[idx] = geom_mask | (ratio > grad_ratio_threshold)
        else:
            camera_mask[idx] = geom_mask

        # Free GPU memory from loaded GT image
        del image, gt_image, render_pkg, loss, grad_mag
        torch.cuda.empty_cache()

    if not quiet:
        for block_id in range(block_num):
            print(f"Block {block_id} / {block_num} has "
                  f"{camera_mask[:, block_id].sum()} cameras.")

    return camera_mask


def visualize_partition(cameras, camera_mask, aabb, block_dim, save_path,
                        gaussians=None, block_splits=None):
    """Visualize partition result: block grid + camera positions + point cloud
    in contracted space. Points outside the AABB are also shown (contracted
    coords will fall outside [0,1]) so the user can judge AABB coverage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    block_dim = [int(d) for d in block_dim]
    aabb_t = torch.tensor(aabb, dtype=torch.float32) if not isinstance(aabb, torch.Tensor) else aabb.cpu()
    block_num = block_dim[0] * block_dim[1] * block_dim[2]

    cam_centers_world = []
    for cam in cameras:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers_world.append(C2W[:3, 3])
    cam_centers_world = np.stack(cam_centers_world, axis=0)

    cam_centers_contracted = contract_to_unisphere(
        torch.tensor(cam_centers_world, dtype=torch.float32),
        aabb_t, ord=torch.inf
    ).numpy()

    # ---- point cloud (gaussian centers) ----
    pts_contracted = None
    pts_inside_mask = None
    if gaussians is not None:
        xyz = gaussians.get_xyz.detach().float().cpu()
        aabb_min, aabb_max = aabb_t[:3], aabb_t[3:]
        inside = ((xyz >= aabb_min) & (xyz <= aabb_max)).all(dim=1).numpy()

        MAX_PTS = 50000
        if xyz.shape[0] > MAX_PTS:
            idx = np.random.default_rng(42).choice(xyz.shape[0], MAX_PTS, replace=False)
            xyz = xyz[idx]
            inside = inside[idx]

        pts_contracted = contract_to_unisphere(xyz, aabb_t, ord=torch.inf).numpy()
        pts_inside_mask = inside

    mask_np = camera_mask.cpu().numpy() if isinstance(camera_mask, torch.Tensor) else camera_mask
    primary_block = np.argmax(mask_np, axis=1)
    num_blocks_per_cam = mask_np.sum(axis=1)
    multi_block = num_blocks_per_cam > 1

    cmap = plt.cm.get_cmap("tab20", max(block_num, 2))
    colors = [cmap(primary_block[i]) for i in range(len(cameras))]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    plane_configs = [
        (0, 1, "X (contracted)", "Y (contracted)", "Top-down  (X-Y)"),
        (0, 2, "X (contracted)", "Z (contracted)", "Front     (X-Z)"),
        (1, 2, "Y (contracted)", "Z (contracted)", "Side      (Y-Z)"),
    ]

    for ax, (ax_i, ax_j, xlabel, ylabel, title) in zip(axes, plane_configs):
        dim_i, dim_j = block_dim[ax_i], block_dim[ax_j]

        # point cloud (draw first so cameras are on top)
        if pts_contracted is not None:
            outside = ~pts_inside_mask
            if outside.any():
                ax.scatter(
                    pts_contracted[outside, ax_i],
                    pts_contracted[outside, ax_j],
                    c="salmon", s=0.3, alpha=0.25, edgecolors="none",
                    zorder=0, rasterized=True, label="_out",
                )
            if pts_inside_mask.any():
                ax.scatter(
                    pts_contracted[pts_inside_mask, ax_i],
                    pts_contracted[pts_inside_mask, ax_j],
                    c="dodgerblue", s=0.3, alpha=0.25, edgecolors="none",
                    zorder=1, rasterized=True, label="_in",
                )

        # AABB boundary ([0,1] box)
        aabb_rect = Rectangle((0, 0), 1, 1)
        ax.add_patch(Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor="black",
                                facecolor="none", linestyle="-", zorder=4))

        # block grid (adaptive or uniform boundaries)
        axis_labels = ["x", "y", "z"]
        if block_splits is not None:
            splits_i = block_splits[axis_labels[ax_i]]
            splits_j = block_splits[axis_labels[ax_j]]
        else:
            splits_i = [bi / dim_i for bi in range(dim_i + 1)]
            splits_j = [bj / dim_j for bj in range(dim_j + 1)]
        patches = []
        for bi in range(dim_i):
            for bj in range(dim_j):
                x0 = splits_i[bi]
                y0 = splits_j[bj]
                w = splits_i[bi + 1] - splits_i[bi]
                h = splits_j[bj + 1] - splits_j[bj]
                patches.append(Rectangle((x0, y0), w, h))
        pc = PatchCollection(patches, facecolor="none", edgecolor="gray",
                             linewidth=0.8, linestyle="--")
        ax.add_collection(pc)

        # cameras
        single = ~multi_block
        ax.scatter(
            cam_centers_contracted[single, ax_i],
            cam_centers_contracted[single, ax_j],
            c=[colors[i] for i in np.where(single)[0]],
            s=12, alpha=0.85, edgecolors="none", zorder=5,
        )
        if multi_block.any():
            ax.scatter(
                cam_centers_contracted[multi_block, ax_i],
                cam_centers_contracted[multi_block, ax_j],
                c=[colors[i] for i in np.where(multi_block)[0]],
                s=30, alpha=0.9, edgecolors="red", linewidths=0.8,
                marker="D", zorder=6,
            )

        view_margin = 0.15
        if pts_contracted is not None:
            all_vals_i = np.concatenate([pts_contracted[:, ax_i],
                                         cam_centers_contracted[:, ax_i]])
            all_vals_j = np.concatenate([pts_contracted[:, ax_j],
                                         cam_centers_contracted[:, ax_j]])
            lo_i = min(all_vals_i.min(), 0) - view_margin
            hi_i = max(all_vals_i.max(), 1) + view_margin
            lo_j = min(all_vals_j.min(), 0) - view_margin
            hi_j = max(all_vals_j.max(), 1) + view_margin
            p98_i = np.percentile(np.abs(all_vals_i), 98)
            p98_j = np.percentile(np.abs(all_vals_j), 98)
            lo_i = max(lo_i, -p98_i - view_margin)
            hi_i = min(hi_i,  p98_i + view_margin)
            lo_j = max(lo_j, -p98_j - view_margin)
            hi_j = min(hi_j,  p98_j + view_margin)
            ax.set_xlim(lo_i, hi_i)
            ax.set_ylim(lo_j, hi_j)
        else:
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(False)

    # ---- legend ----
    legend_handles = []
    for bid in range(block_num):
        cnt = int(mask_np[:, bid].sum())
        patch = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cmap(bid), markersize=8,
                           label=f"Block {bid} ({cnt} cams)")
        legend_handles.append(patch)
    n_multi = int(multi_block.sum())
    legend_handles.append(plt.Line2D([0], [0], marker='D', color='w',
                          markerfacecolor='gray', markeredgecolor='red',
                          markersize=8, label=f"Multi-block ({n_multi} cams)"))
    if pts_contracted is not None:
        n_in = int(pts_inside_mask.sum())
        n_out = int((~pts_inside_mask).sum())
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='dodgerblue', markersize=6,
                              label=f"Pts inside AABB ({n_in})"))
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='salmon', markersize=6,
                              label=f"Pts outside AABB ({n_out})"))
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(legend_handles), 8), fontsize=8, frameon=False)

    stats_text = (
        f"Total cameras: {len(cameras)}  |  Blocks: {block_num} "
        f"({block_dim[0]}×{block_dim[1]}×{block_dim[2]})  |  "
        f"Avg cams/block: {mask_np.sum() / block_num:.1f}  |  "
        f"Avg blocks/cam: {num_blocks_per_cam.mean():.2f}  |  "
        f"Multi-block cams: {n_multi} ({100*n_multi/len(cameras):.1f}%)"
    )
    fig.suptitle(stats_text, fontsize=11, y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Partition visualization saved to: {save_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_inblock", action="store_true")
    parser.add_argument("--simple_selection", type=float, default=0)
    parser.add_argument(
        "--aabb_autofit_mode",
        type=str,
        default=None,
        choices=["focus", "cameras", "cameras_pano"],
        help=(
            "When yaml aabb is null: focus=original CityGS focus+median; "
            "cameras=legacy bbox from camera centers (+ optional gaussian union); "
            "cameras_pano=panorama-friendly mean+percentile+baseline*factor margin "
            "(auto-picked when default_camera_type==3 and this flag is not set). "
            "Overrides yaml if set."
        ),
    )
    args = parser.parse_args(sys.argv[1:])
    # 避免 argparse 默认 None 覆盖 yaml / get_default_lp 中的 aabb_autofit_mode
    cfg_cmd = Namespace(**{k: v for k, v in vars(args).items() if not (k == "aabb_autofit_mode" and v is None)})
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, cfg_cmd)
    if args.aabb_autofit_mode is not None:
        lp.aabb_autofit_mode = args.aabb_autofit_mode

    # Initialize system state (RNG)
    safe_state(args.quiet)

    config_name = os.path.splitext(os.path.basename(lp.config))[0]
    if not lp.model_path:
        # time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        lp.model_path = os.path.join("./output/", config_name)
    
    print("Output folder: {}".format(lp.model_path))
    os.makedirs(lp.model_path, exist_ok = True)
    with open(os.path.join(lp.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(lp))))

    modules = __import__('scene')
    model_config = lp.model_config
    gaussians = getattr(modules, model_config['name'])(lp.sh_degree, **model_config['kwargs'])
    scene = LargeScene(lp, gaussians, shuffle=False)
    if not os.path.exists(os.path.join(lp.source_path, "data_partitions")):
        os.makedirs(os.path.join(lp.source_path, "data_partitions"))

    partition_mode = getattr(lp, 'partition_mode', 'ssim')
    if partition_mode == 'gradient':
        camera_mask = block_partitioning_gradient(
            scene.getTrainCameras(), gaussians, lp, pp, op=op,
            scale=1.0, quiet=args.quiet
        )
    elif partition_mode == 'geometry':
        camera_mask = block_partitioning_geometry(
            scene.getTrainCameras(), gaussians, lp,
            scale=1.0, quiet=args.quiet
        )
    else:
        camera_mask = block_partitioning(
            scene.getTrainCameras(), gaussians, lp, pp, 1.0,
            args.quiet, args.disable_inblock, args.simple_selection
        )

    vis_path = os.path.join(lp.source_path, "data_partitions", f"{config_name}_partition_vis.png")
    visualize_partition(
        scene.getTrainCameras(), camera_mask, lp.aabb, lp.block_dim, vis_path,
        gaussians=gaussians,
        block_splits=getattr(lp, "block_splits", None),
    )

    camera_mask = camera_mask.cpu().numpy()
    np.save(os.path.join(lp.source_path, "data_partitions", f"{config_name}.npy"), camera_mask)

    # All done
    print("\nPartition complete.")