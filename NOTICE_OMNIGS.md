# OmniGS-derived components (GPL-3.0)

This repository includes **equirectangular / LonLat** differentiable Gaussian rasterization code adapted from **OmniGS** (Longwei Li et al., WACV 2025), which is a **derivative work** of **3D Gaussian Splatting** and is licensed under the **GNU General Public License v3.0**.

Affected paths (non-exhaustive):

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/` — `LonlatRasterizer`, `preprocessLonlat`, `computeCov2DLonlat`, related backward kernels
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h` — `too_close`, `point3ToLonlatScreen`, etc.
- Python wiring: `camera_type` in `GaussianRasterizationSettings` (`1` = pinhole, `3` = LonLat)

If you distribute binaries including this code, comply with **GPL-3.0** (source offer, license notice, etc.). The rest of panoOGS may remain under its existing license where compatible.

OmniGS reference: https://github.com/liquorleaf/OmniGS  
Paper: https://arxiv.org/abs/2404.03202
