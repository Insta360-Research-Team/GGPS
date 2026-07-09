import torch
import numpy as np
from utils.camera_utils import loadCam_woImage

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def contract_to_unisphere(
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

def _get_block_bounds(block_id, block_dim, block_splits=None):
    """Return (min_x, max_x, min_y, max_y, min_z, max_z) for *block_id*.

    *block_splits*: ``None`` → uniform grid; otherwise a dict
    ``{"x": [0.0, ..., 1.0], "y": [...], "z": [...]}`` with per-axis
    boundary lists (length = dim_size + 1) in contracted [0,1] space.
    """
    block_id_z = block_id // (block_dim[0] * block_dim[1])
    block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
    block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0]

    if block_splits is not None:
        min_x = block_splits["x"][block_id_x]
        max_x = block_splits["x"][block_id_x + 1]
        min_y = block_splits["y"][block_id_y]
        max_y = block_splits["y"][block_id_y + 1]
        min_z = block_splits["z"][block_id_z]
        max_z = block_splits["z"][block_id_z + 1]
    else:
        min_x = float(block_id_x) / block_dim[0]
        max_x = float(block_id_x + 1) / block_dim[0]
        min_y = float(block_id_y) / block_dim[1]
        max_y = float(block_id_y + 1) / block_dim[1]
        min_z = float(block_id_z) / block_dim[2]
        max_z = float(block_id_z + 1) / block_dim[2]

    return min_x, max_x, min_y, max_y, min_z, max_z


def block_filtering(block_id, xyz_org, aabb, block_dim, scale=1.0, mask_only=True,
                    block_splits=None):

    if len(aabb) == 4:
        aabb = [aabb[0], aabb[1], xyz_org[:, -1].min(), 
                aabb[2], aabb[3], xyz_org[:, -1].max()]
    elif len(aabb) == 6:
        aabb = aabb
    else:
        assert False, "Unknown aabb format!"

    xyz_tensor = torch.tensor(xyz_org)
    aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_tensor.device)

    xyz = contract_to_unisphere(xyz_tensor, aabb, ord=torch.inf)
    min_x, max_x, min_y, max_y, min_z, max_z = _get_block_bounds(
        block_id, block_dim, block_splits)

    delta_x = (max_x - min_x) * (scale - 1.0)
    delta_y = (max_y - min_y) * (scale - 1.0)
    delta_z = (max_z - min_z) * (scale - 1.0)

    min_x -= delta_x / 2
    max_x += delta_x / 2
    min_y -= delta_y / 2
    max_y += delta_y / 2
    min_z -= delta_z / 2
    
    block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                    & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                    & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)

    if mask_only:
        return block_mask
    else:
        return mask_only, xyz, [min_x, max_x, min_y, max_y, min_z, max_z]

def which_block(xyz_org, aabb, block_dim, block_splits=None):

    if len(aabb) == 4:
        aabb = [aabb[0], aabb[1], xyz_org[:, -1].min(), 
                aabb[2], aabb[3], xyz_org[:, -1].max()]
    elif len(aabb) == 6:
        aabb = aabb
    else:
        assert False, "Unknown aabb format!"

    xyz_tensor = torch.tensor(xyz_org)
    aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_tensor.device)

    xyz = contract_to_unisphere(xyz_tensor, aabb, ord=torch.inf)

    if block_splits is not None:
        inner_x = torch.tensor(block_splits["x"][1:-1], dtype=xyz.dtype, device=xyz.device)
        inner_y = torch.tensor(block_splits["y"][1:-1], dtype=xyz.dtype, device=xyz.device)
        inner_z = torch.tensor(block_splits["z"][1:-1], dtype=xyz.dtype, device=xyz.device)
        block_id_x = torch.bucketize(xyz[:, 0], inner_x).clamp(0, block_dim[0] - 1)
        block_id_y = torch.bucketize(xyz[:, 1], inner_y).clamp(0, block_dim[1] - 1)
        block_id_z = torch.bucketize(xyz[:, 2], inner_z).clamp(0, block_dim[2] - 1)
    else:
        block_id_x = torch.floor((xyz[:, 0] * block_dim[0]).clamp(0, block_dim[0] - 1)).long()
        block_id_y = torch.floor((xyz[:, 1] * block_dim[1]).clamp(0, block_dim[1] - 1)).long()
        block_id_z = torch.floor((xyz[:, 2] * block_dim[2]).clamp(0, block_dim[2] - 1)).long()

    block_id = block_id_z * block_dim[0] * block_dim[1] + block_id_y * block_dim[0] + block_id_x

    return block_id

def in_frustum(viewpoint_cam, cell_corners, aabb, block_dim):
    num_cell = cell_corners.shape[0]
    device = cell_corners.device

    cell_corners = torch.cat([cell_corners, torch.ones_like(cell_corners[..., [0]])], dim=-1)
    full_proj_transform = viewpoint_cam.full_proj_transform.repeat(num_cell, 1, 1)
    viewmatrix = viewpoint_cam.world_view_transform.repeat(num_cell, 1, 1)
    cell_corners_screen = cell_corners.bmm(full_proj_transform)
    cell_corners_screen = cell_corners_screen / cell_corners_screen[..., [-1]]
    cell_corners_screen = cell_corners_screen[..., :-1].reshape(-1, 3)

    cell_corners_cam = cell_corners.bmm(viewmatrix)
    dist = torch.norm(cell_corners_cam[:, :, :3], dim=-1)
    dist_min = torch.min(dist, dim=-1)[0]
    cam_center_id = torch.argmin(dist_min)
    mask = (cell_corners_cam[..., 2] > 0.2)

    mask_ = mask.reshape(-1)
    cell_corners_screen_ = cell_corners_screen.clone().reshape(-1, 3)
    cell_corners_screen_[~mask_] = torch.inf
    cell_corners_screen_min = cell_corners_screen_.reshape(num_cell, -1, 3).min(dim=1).values
    cell_corners_screen_min[cell_corners_screen_min==torch.inf] = 0.0

    cell_corners_screen_ = cell_corners_screen.clone().reshape(-1, 3)
    cell_corners_screen_[~mask_] = -torch.inf
    cell_corners_screen_max = cell_corners_screen_.reshape(num_cell, -1, 3).max(dim=1).values
    cell_corners_screen_max[cell_corners_screen_max==-torch.inf] = 0.0

    box_a = torch.cat([cell_corners_screen_min[:, :2], cell_corners_screen_max[:, :2]], dim=1)
    box_b = torch.tensor([[-1, -1, 1, 1]], dtype=torch.float32, device=device)
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                    box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                    box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    mask = (inter[:, 0, 0] * inter[:, 0, 1]) > 0
    mask[cam_center_id] = True
    
    return mask, dist_min[mask]

def get_default_aabb(args, cameras, xyz_org, scale=1.0):
    
    torch.cuda.empty_cache()
    c2ws = np.array([np.linalg.inv(np.asarray((loadCam_woImage(args, idx, cam, scale).world_view_transform.T).cpu().numpy())) for idx, cam in enumerate(cameras)])
    poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    center = (focus_point_fn(poses))
    radius = torch.tensor(np.median(np.abs(c2ws[:,:3,3] - center), axis=0), device=xyz_org.device)
    center = torch.from_numpy(center).float().to(xyz_org.device)
    if radius.min() / radius.max() < 0.02:
        # If the radius is too small, we don't contract in this dimension
        radius[torch.argmin(radius)] = 0.5 * (xyz_org[:, torch.argmin(radius)].max() - xyz_org[:, torch.argmin(radius)].min())
    aabb = torch.zeros(6, device=xyz_org.device)
    aabb[:3] = center - radius
    aabb[3:] = center + radius

    return aabb


def get_aabb_from_cameras(args, cameras, xyz_org, scale=1.0):
    """
    Axis-aligned box from training camera centers (world), for panorama / ring setups
    where focus-ray AABB is ill-conditioned. Optionally union with gaussian xyz extent
    and add relative margin so contraction stays stable.

    Expects attributes on args (with defaults via getattr):
      aabb_camera_margin (float): fraction of per-axis span to pad on each side, default 0.05
      aabb_cameras_union_gaussians (bool): union min/max with xyz_org, default True
    """
    torch.cuda.empty_cache()
    margin_ratio = float(getattr(args, "aabb_camera_margin", 0.05))
    union_gs = bool(getattr(args, "aabb_cameras_union_gaussians", True))

    centers = []
    for idx, cam in enumerate(cameras):
        c = loadCam_woImage(args, idx, cam, scale)
        centers.append(c.camera_center.detach().cpu().numpy())
    cc = np.stack(centers, axis=0)

    lo = cc.min(axis=0).astype(np.float64)
    hi = cc.max(axis=0).astype(np.float64)

    if union_gs:
        xyz_np = xyz_org.detach().float().cpu().numpy()
        lo = np.minimum(lo, xyz_np.min(axis=0))
        hi = np.maximum(hi, xyz_np.max(axis=0))

    span = np.maximum(hi - lo, 1e-6)
    pad = margin_ratio * span
    lo = lo - pad
    hi = hi + pad

    for d in range(3):
        if hi[d] - lo[d] < 1e-4:
            mid = 0.5 * (hi[d] + lo[d])
            lo[d] = mid - 0.01
            hi[d] = mid + 0.01

    aabb = torch.zeros(6, device=xyz_org.device, dtype=torch.float32)
    aabb[:3] = torch.from_numpy(lo.astype(np.float32)).to(xyz_org.device)
    aabb[3:] = torch.from_numpy(hi.astype(np.float32)).to(xyz_org.device)
    return aabb


def _median_nearest_neighbor_distance(centers_np: np.ndarray) -> float:
    """Median of nearest-neighbor distances in a point cloud, used as a robust
    estimator of the typical baseline between adjacent panoramas. Implemented
    with torch.cdist to avoid an extra scipy dependency.
    """
    n = centers_np.shape[0]
    if n < 2:
        return 0.0
    pts = torch.from_numpy(centers_np.astype(np.float32))
    d = torch.cdist(pts, pts)
    d.fill_diagonal_(float("inf"))
    nn = d.min(dim=1).values.cpu().numpy()
    return float(np.median(nn))


def get_aabb_from_cameras_pano(args, cameras, xyz_org, scale=1.0):
    """
    AABB tailored for 360 panorama (ERP) datasets where the rendering frustum
    is the full sphere and the COLMAP/OpenMVG point cloud is built from
    multi-view triangulation.

    Design choices (vs. the legacy ``get_aabb_from_cameras``):
      * center  : mean of camera centers (panorama sees in all directions, so
                  the camera position itself *is* the local viewpoint center).
      * radius  : per-axis percentile of |C_i - center| (default p=100 i.e. max),
                  so the AABB at minimum encloses the camera trajectory.
      * margin  : absolute meters derived from the median nearest-neighbor
                  baseline times a disparity factor (default 15). The rationale
                  is that triangulation is reliable up to roughly
                  ``baseline * factor`` meters, so this naturally pushes the
                  AABB out to cover the trustworthy near-field content
                  (facades, ground, signage) regardless of how the camera
                  trajectory is shaped (line / loop / grid).
      * union   : do NOT union with gaussian min/max by default (sky / far
                  outliers in the sparse cloud would otherwise blow the AABB).
      * fallback: if some axis is still nearly degenerate after radius+pad,
                  reuse the second-largest axis radius (kept for robustness;
                  rarely triggers thanks to the absolute pad).

    YAML knobs (read via getattr):
      aabb_pano_radius_percentile (float, default 100.0)  -> 100 == max
      aabb_pano_disparity_factor  (float, default 15.0)
      aabb_pano_union_gaussians   (bool , default False)
    """
    torch.cuda.empty_cache()
    p_radius = float(getattr(args, "aabb_pano_radius_percentile", 100.0))
    disparity_factor = float(getattr(args, "aabb_pano_disparity_factor", 15.0))
    union_gs = bool(getattr(args, "aabb_pano_union_gaussians", False))

    centers = []
    for idx, cam in enumerate(cameras):
        c = loadCam_woImage(args, idx, cam, scale)
        centers.append(c.camera_center.detach().cpu().numpy())
    cc = np.stack(centers, axis=0).astype(np.float64)

    center = cc.mean(axis=0)
    deviations = np.abs(cc - center)
    radius = np.percentile(deviations, p_radius, axis=0)

    baseline = _median_nearest_neighbor_distance(cc)
    margin_meters = baseline * disparity_factor
    pad = np.full(3, margin_meters, dtype=np.float64)

    lo = center - radius - pad
    hi = center + radius + pad

    if union_gs:
        xyz_np = xyz_org.detach().float().cpu().numpy()
        lo = np.minimum(lo, xyz_np.min(axis=0))
        hi = np.maximum(hi, xyz_np.max(axis=0))

    half = 0.5 * (hi - lo)
    if half.max() > 0 and (half.min() / half.max()) < 0.02:
        sorted_h = np.sort(half)
        replacement = sorted_h[-2]
        d_min = int(np.argmin(half))
        mid = 0.5 * (hi[d_min] + lo[d_min])
        lo[d_min] = mid - replacement
        hi[d_min] = mid + replacement

    print(
        "[pano AABB] N={n}  center={c}  radius_p{p:g}={r}  "
        "baseline={b:.3f}m  margin={m:.3f}m  AABB lo={lo}  hi={hi}  span={sp}".format(
            n=cc.shape[0],
            c=np.round(center, 3).tolist(),
            p=p_radius,
            r=np.round(radius, 3).tolist(),
            b=baseline,
            m=margin_meters,
            lo=np.round(lo, 3).tolist(),
            hi=np.round(hi, 3).tolist(),
            sp=np.round(hi - lo, 3).tolist(),
        )
    )

    aabb = torch.zeros(6, device=xyz_org.device, dtype=torch.float32)
    aabb[:3] = torch.from_numpy(lo.astype(np.float32)).to(xyz_org.device)
    aabb[3:] = torch.from_numpy(hi.astype(np.float32)).to(xyz_org.device)
    return aabb