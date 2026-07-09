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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

/* OmniGS LonLat helpers (GPL; see NOTICE_OMNIGS.md in repo root) */
#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef M_1_PIf32
#define M_1_PIf32 (1.0f / M_PIf)
#endif
#ifndef M_2_PIf32
#define M_2_PIf32 (2.0f / M_PIf)
#endif

#endif