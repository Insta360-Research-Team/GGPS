/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <stdexcept>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& metric_map,
	const int skyboxnum,
	const bool output_invdepth,
	const int camera_type)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;
  if (output_invdepth)
  {
    out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
    out_invdepthptr = out_invdepth.data_ptr<float>();
  }
  
  bool use_metric = metric_map.size(0) > 0 && camera_type == 1;
  torch::Tensor metric_count = torch::zeros({P}, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  if (camera_type == 1)
	  {
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		means3D.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(), 
		opacity.contiguous().data_ptr<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data_ptr<float>(), 
		viewmatrix.contiguous().data_ptr<float>(), 
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data_ptr<float>(),
		skyboxnum,
		radii.contiguous().data_ptr<int>(),
		debug,
		use_metric ? metric_map.contiguous().data_ptr<float>() : nullptr,
		use_metric ? metric_count.contiguous().data_ptr<float>() : nullptr,
		out_invdepthptr);
	  }
	  else if (camera_type == 3)
	  {
	  rendered = CudaRasterizer::LonlatRasterizer::forward(
		geomFunc,
		binningFunc,
		imgFunc,
		P, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		means3D.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(),
		opacity.contiguous().data_ptr<float>(),
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		prefiltered,
		out_color.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>(),
		out_invdepthptr);
	  }
	  else
	  {
		throw std::runtime_error("[diff_gaussian_rasterization] Invalid camera_type (use 1=pinhole, 3=lonlat)");
	  }
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth, metric_count);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const int camera_type) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  // [AbsGS] dL_dmeans2D is shape [P, 4]:
  //   cols 0/1 = signed view-space gradient (used by autograd to flow back into means3D)
  //   cols 2/3 = component-wise absolute view-space gradient (densification criterion only)
  torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  float* dL_dinvdepthsptr = nullptr;
  const float* dL_dout_invdepthptr = nullptr;
  if (dL_dout_invdepth.size(0) > 0)
  {
    dL_dinvdepths = torch::zeros({P, 1}, means3D.options()).contiguous();
    dL_dinvdepthsptr = dL_dinvdepths.data_ptr<float>();
    dL_dout_invdepthptr = dL_dout_invdepth.contiguous().data_ptr<float>();
  }
  
  if(P != 0)
  {
	  if (camera_type == 1)
	  {
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H, 
	  means3D.contiguous().data_ptr<float>(),
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dout_invdepthptr,
	  dL_dinvdepthsptr,
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),  
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>(),
	  debug);
	  }
	  else if (camera_type == 3)
	  {
	  torch::Tensor dpx_dt = torch::zeros({P, 3}, means3D.options());
	  torch::Tensor dpy_dt = torch::zeros({P, 3}, means3D.options());
	  CudaRasterizer::LonlatRasterizer::backward(P, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H,
	  means3D.contiguous().data_ptr<float>(),
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dout_invdepthptr,
	  dL_dinvdepthsptr,
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>(),
	  dpx_dt.contiguous().data_ptr<float>(),
	  dpy_dt.contiguous().data_ptr<float>());
	  }
	  else
	  {
		throw std::runtime_error("[diff_gaussian_rasterization] Invalid camera_type (use 1=pinhole, 3=lonlat)");
	  }
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix,
		const int camera_type)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	if (camera_type == 1)
	{
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
	}
	else if (camera_type == 3)
	{
	CudaRasterizer::LonlatRasterizer::markVisible(P,
		present.contiguous().data_ptr<bool>());
	}
	else
	{
		throw std::runtime_error("[diff_gaussian_rasterization] Invalid camera_type (use 1=pinhole, 3=lonlat)");
	}
  }
  
  return present;
}