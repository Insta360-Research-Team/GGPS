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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask_path: str = ""
    skymask_path: str = ""
    depth_path: str = ""
    depth_params: dict = None
    camera_type: int = 1  # 3 = OmniGS LonLat (GPL raster)

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, masks_folder, skymasks_folder="", depths_folder="", depths_params=None,
                      default_camera_type=1):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        w, h = image.size

        ct = int(default_camera_type)
        if ct == 3:
            # LonLat / ERP：与 OmniGS openMVG 一致，用图像尺寸作为焦距，不依赖 COLMAP cameras 模型
            fx, fy = float(w), float(h)
            FovX = focal2fov(fx, w)
            FovY = focal2fov(fy, h)
            width, height = w, h
            if idx == 0:
                print("[COLMAP] camera_type=3 (ERP/LonLat): FOV 由图像宽高按 fx=W, fy=H 计算")
        elif intr.model == "SIMPLE_PINHOLE":
            height = intr.height
            width = intr.width
            f = float(intr.params[0])
            FovY = focal2fov(f, height)
            FovX = focal2fov(f, width)
        elif intr.model == "PINHOLE":
            height = intr.height
            width = intr.width
            fx, fy = float(intr.params[0]), float(intr.params[1])
            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)
        else:
            height = intr.height
            width = intr.width
            m = intr.model
            p = intr.params
            if m in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"):
                f = float(p[0])
                FovY = focal2fov(f, height)
                FovX = focal2fov(f, width)
            elif m in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
                fx, fy = float(p[0]), float(p[1])
                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)
            elif m == "FOV" and len(p) >= 5:
                fx, fy = float(p[3]), float(p[4])
                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)
            elif len(p) >= 4:
                fx, fy = float(p[0]), float(p[1])
                FovY = focal2fov(fy, height)
                FovX = focal2fov(fx, width)
                if idx == 0:
                    print(f"[Colmap] 相机模型 {m}：按 params 前两项为 fx,fy")
            elif len(p) >= 1:
                f = float(p[0])
                FovY = focal2fov(f, height)
                FovX = focal2fov(f, width)
                if idx == 0:
                    print(f"[Colmap] 相机模型 {m}：按 params[0] 为焦距")
            else:
                assert False, f"Colmap 相机模型 {m} 无法解析内参；全景请设 default_camera_type: 3"

        mask_path = ""
        skymask_path = ""
        depth_path = ""
        depth_param = None
        if masks_folder:
            mask_path = os.path.join(masks_folder, image_name + ".png")
            if not os.path.exists(mask_path):
                mask_path = ""
        if skymasks_folder:
            skymask_path = os.path.join(skymasks_folder, image_name + ".png")
            if not os.path.exists(skymask_path):
                skymask_path = ""
        if depths_folder:
            depth_path = os.path.join(depths_folder, image_name + ".png")
            if not os.path.exists(depth_path):
                depth_path = ""
            elif depths_params is not None:
                depth_param = depths_params.get(image_name, None)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, mask_path=mask_path,
                              skymask_path=skymask_path, depth_path=depth_path, depth_params=depth_param,
                              camera_type=int(default_camera_type))
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, masks, eval, llffhold=None, partition=None,
                        use_alpha_masks=True, use_sky_masks=True, use_depth=True, default_camera_type=1):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    if use_alpha_masks:
        if masks == "" or masks is None:
            masks_reading_dir = os.path.join(path, "masks")
        else:
            masks_reading_dir = masks if os.path.isabs(masks) else os.path.join(path, masks)
        if not os.path.isdir(masks_reading_dir):
            masks_reading_dir = ""
    else:
        masks_reading_dir = ""

    if use_sky_masks:
        skymasks_reading_dir = os.path.join(path, "skymasks")
        if not os.path.isdir(skymasks_reading_dir):
            skymasks_reading_dir = ""
    else:
        skymasks_reading_dir = ""

    if use_depth:
        depths_reading_dir = os.path.join(path, "depths")
        if not os.path.isdir(depths_reading_dir):
            depths_reading_dir = ""
    else:
        depths_reading_dir = ""

    depths_params = None
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    if depths_reading_dir and os.path.exists(depth_params_file):
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[k]["scale"] for k in depths_params])
            pos_scales = all_scales[all_scales > 0]
            med_scale = float(np.median(pos_scales)) if len(pos_scales) > 0 else 0.0
            for k in depths_params:
                depths_params[k]["med_scale"] = med_scale
        except Exception as e:
            print(f"[Warning] Failed to read depth params: {e}")
            depths_params = None

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        masks_folder=masks_reading_dir,
        skymasks_folder=skymasks_reading_dir,
        depths_folder=depths_reading_dir,
        depths_params=depths_params,
        default_camera_type=default_camera_type,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    split_train = os.path.join(path, "train.txt")
    split_test = os.path.join(path, "test.txt")
    split_val = os.path.join(path, "val.txt")
    test_list_path = split_test if os.path.isfile(split_test) else (split_val if os.path.isfile(split_val) else None)

    if os.path.isfile(split_train) and test_list_path:
        with open(split_train, "r") as f:
            train_names = {line.strip() for line in f if line.strip()}
        with open(test_list_path, "r") as f:
            test_names = {line.strip() for line in f if line.strip()}
        train_cam_infos = [c for c in cam_infos if c.image_name in train_names]
        test_cam_infos = [c for c in cam_infos if c.image_name in test_names]
        have = {c.image_name for c in cam_infos}
        if train_names - have:
            print(f"[train/test lists] train.txt 中 {len(train_names - have)} 个名字无对应相机，已忽略")
        if test_names - have:
            print(f"[train/test lists] test/val 中 {len(test_names - have)} 个名字无对应相机，已忽略")
        dropped = [c.image_name for c in cam_infos if c.image_name not in train_names and c.image_name not in test_names]
        if dropped:
            print(f"[train/test lists] {len(dropped)} 个相机不在 train 也不在 test/val 列表中，不参与训练与测试")
        if train_names & test_names:
            print("[train/test lists] 警告: train 与 test/val 名单存在交集")
        union_infos = train_cam_infos + test_cam_infos
        if len(union_infos) > 0:
            nerf_normalization = getNerfppNorm(union_infos)
        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []
    elif eval:
        import math
        if llffhold is None:
            eval_image_num = max(math.ceil(0.05 * len(cam_infos)), 1)
            llffhold = max(len(cam_infos) // eval_image_num, 8)
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # partition mask 的索引与训练集（按名称排序）一一对应，
    # 必须在 train/test 分割之后对 train_cam_infos 应用，否则索引错位。
    if partition is not None:
        filtered_train = []
        for i in range(partition.shape[0]):
            if partition[i]:
                filtered_train.append(train_cam_infos[i])
        train_cam_infos = filtered_train if len(filtered_train) >= 50 else []
        print(f"Filtered Train Cameras by partition: {len(train_cam_infos)}. ")

    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # image_path = os.path.join(path, cam_name)
            image_path = cam_name
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], mask_path=""))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readOpenMVGSceneInfo(path, images, masks, eval, llffhold=None, partition=None,
                         use_alpha_masks=True, use_sky_masks=True, use_depth=True):
    """
    OmniGS-compatible keyframes JSON + COLMAP-style sparse/0 point cloud.
    JSON: list of {id, img_name, width, height, fx, fy, position:[3], rotation:[[3x3]]}.
    GPL raster path uses camera_type=3 (LonLat).
    """
    json_path = os.path.join(path, "keyframes.json")
    if not os.path.isfile(json_path):
        json_path = os.path.join(path, "openmvg_keyframes.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError("Expected keyframes.json or openmvg_keyframes.json in " + path)
    with open(json_path, "r") as f:
        keyframes = json.load(f)
    reading_dir = "images" if images is None else images
    if use_alpha_masks:
        if masks == "" or masks is None:
            masks_reading_dir = os.path.join(path, "masks")
        else:
            masks_reading_dir = masks if os.path.isabs(masks) else os.path.join(path, masks)
        if not os.path.isdir(masks_reading_dir):
            masks_reading_dir = ""
    else:
        masks_reading_dir = ""
    if use_sky_masks:
        skymasks_reading_dir = os.path.join(path, "skymasks")
        if not os.path.isdir(skymasks_reading_dir):
            skymasks_reading_dir = ""
    else:
        skymasks_reading_dir = ""
    if use_depth:
        depths_reading_dir = os.path.join(path, "depths")
        if not os.path.isdir(depths_reading_dir):
            depths_reading_dir = ""
    else:
        depths_reading_dir = ""
    depths_params = None
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    if depths_reading_dir and os.path.exists(depth_params_file):
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[k]["scale"] for k in depths_params])
            pos_scales = all_scales[all_scales > 0]
            med_scale = float(np.median(pos_scales)) if len(pos_scales) > 0 else 0.0
            for k in depths_params:
                depths_params[k]["med_scale"] = med_scale
        except Exception as e:
            print(f"[Warning] Failed to read depth params: {e}")
            depths_params = None

    cam_infos = []
    for idx, kf in enumerate(keyframes):
        img_name = kf["img_name"]
        stem = os.path.splitext(os.path.basename(img_name))[0]
        image_path = os.path.join(path, reading_dir, os.path.basename(img_name))
        image = Image.open(image_path)
        w, h = int(kf["width"]), int(kf["height"])
        assert image.size[0] == w and image.size[1] == h, f"{image_path} size mismatch json {w}x{h}"
        fx = float(kf["fx"])
        fy = float(kf["fy"])
        FovX = focal2fov(fx, w)
        FovY = focal2fov(fy, h)
        R_w2c = np.array(kf["rotation"], dtype=np.float64)
        t = np.array(kf["position"], dtype=np.float64).reshape(3)
        R = np.transpose(R_w2c)
        T = t
        mask_path = ""
        skymask_path = ""
        depth_path = ""
        if masks_reading_dir:
            mp = os.path.join(masks_reading_dir, stem + ".png")
            if os.path.exists(mp):
                mask_path = mp
        if skymasks_reading_dir:
            smp = os.path.join(skymasks_reading_dir, stem + ".png")
            if os.path.exists(smp):
                skymask_path = smp
        if depths_reading_dir:
            dp = os.path.join(depths_reading_dir, stem + ".png")
            if os.path.exists(dp):
                depth_path = dp
        cam_infos.append(CameraInfo(
            uid=int(kf.get("id", idx)),
            R=R, T=T, FovY=FovY, FovX=FovX,
            image=image, image_path=image_path, image_name=stem,
            width=w, height=h,
            mask_path=mask_path, skymask_path=skymask_path, depth_path=depth_path,
            depth_params=depths_params,
            camera_type=3,
        ))
    cam_infos = sorted(cam_infos, key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)
    split_train = os.path.join(path, "train.txt")
    split_test = os.path.join(path, "test.txt")
    split_val = os.path.join(path, "val.txt")
    test_list_path = split_test if os.path.isfile(split_test) else (split_val if os.path.isfile(split_val) else None)
    if os.path.isfile(split_train) and test_list_path:
        with open(split_train, "r") as f:
            train_names = {line.strip() for line in f if line.strip()}
        with open(test_list_path, "r") as f:
            test_names = {line.strip() for line in f if line.strip()}
        train_cam_infos = [c for c in cam_infos if c.image_name in train_names]
        test_cam_infos = [c for c in cam_infos if c.image_name in test_names]
        union_infos = train_cam_infos + test_cam_infos
        if len(union_infos) > 0:
            nerf_normalization = getNerfppNorm(union_infos)
        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []
    elif eval:
        import math
        if llffhold is None:
            eval_image_num = max(math.ceil(0.05 * len(cam_infos)), 1)
            llffhold = max(len(cam_infos) // eval_image_num, 8)
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # partition mask 的索引与训练集（按名称排序）一一对应，
    # 必须在 train/test 分割之后对 train_cam_infos 应用，否则索引错位。
    if partition is not None:
        filtered_train = []
        for i in range(partition.shape[0]):
            if partition[i]:
                filtered_train.append(train_cam_infos[i])
        train_cam_infos = filtered_train if len(filtered_train) >= 50 else []
        print(f"Filtered Train Cameras by partition: {len(train_cam_infos)}. ")
    print(f"[OpenMVG/OmniGS keyframes] Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None
    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization, ply_path=ply_path)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "OpenMVG": readOpenMVGSceneInfo,
}