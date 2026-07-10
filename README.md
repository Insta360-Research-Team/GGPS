<div align="center">

# Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction

Weijian Chen<sup>1,2</sup>, Weibo Yao<sup>1,3</sup>, Yuhang Zhang<sup>1</sup>, Xiaolin Tang<sup>1</sup>, Guo Wang<sup>1</sup>, Weijun Zhang<sup>1</sup>, Xitong Gao<sup>4</sup>, Yihao Chen<sup>1</sup>, Hongde Qin<sup>5</sup>, Lu Qi<sup>1,6</sup>

<sup>1</sup>Insta360 Research &nbsp; <sup>2</sup>Sun Yat-sen University &nbsp; <sup>3</sup>South China University of Technology<br>
<sup>4</sup>University of Chinese Academy of Sciences &nbsp; <sup>5</sup>Harbin Engineering University &nbsp; <sup>6</sup>Wuhan University

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.08769)
[![Homepage](https://img.shields.io/badge/Project-Homepage-1a73e8?logo=googlechrome&logoColor=white)](https://insta360-research-team.github.io/GGPS-Website/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffcc00)](#)

**English** | [中文](./README_zh.md)

<img src="assets/panolog_teaser.png" width="80%" alt="PanoLOG teaser">

</div>

## 📖 Overview

Scaling 3DGS to large outdoor scenes is costly in both data acquisition and computation. Panoramic images with equirectangular projection (ERP) reduce capture effort through their full 360° field of view, but the resulting omnipresent visibility invalidates existing partitioning strategies that rely on local camera frustums, causing block-wise optimization to degenerate into global training. PanoLOG addresses this with a **Geometry and Gradient-based Partitioning Strategy (G²PS)**: in the coarse stage it leverages sky-sphere modeling and panoramic monocular depth supervision for reliable geometry; in the refinement stage, G²PS builds adaptive bounding volumes via parallax-driven uncertainty and assigns cameras via gradient-based importance scoring. We further construct **Pano360**, the first large-scale panoramic benchmark for outdoor scene reconstruction.

<div align="center">
<img src="assets/panolog_pipeline.png" width="85%" alt="PanoLOG pipeline (G²PS)">
<p><i>The two-stage coarse-to-fine pipeline with the Geometry and Gradient-based Partitioning Strategy (G²PS).</i></p>
</div>

Datasets must be prepared by the user (COLMAP / openMVG results). Training outputs go to `output/` by default (ignored via `.gitignore`).

## 🗺️ Roadmap

We are actively working towards the open-source release. Progress will be tracked here.

> **Surprise!** On July 15 we will release an **Unreal Engine plugin for 3DGS rendering**, letting you use your own reconstructed models directly inside the engine. A free version with a watermark is provided; a paid version (one-time purchase, no watermark) will be available soon, with the link posted here.

- [ ] **2026-07-09** — Full training code.
- [ ] **2026-07-10** — Two authorized panoramic `.ply` models for reproduction and verification.
- [ ] **2026-07-15** — Pano360 dataset.
- [ ] **2026-07-15** — Free (watermarked) rendering plugin; paid version (one-time purchase) available soon, link posted here.

> Suggestions and issues are welcome — please open an issue.

## ⚙️ Environment

An **NVIDIA driver** is required; compiling the `submodules` depends on a local **CUDA Toolkit** (check with `nvcc --version`; it must match or be compatible with the CUDA version of the PyTorch build below).

**Reference conda environment `PanoLOG`:** Python **3.9**, PyTorch **2.8.0** (**cu128** / CUDA **12.8**); system `nvcc` **12.8** (`/usr/local/cuda-12.8`); GPU **RTX 5090 D** (Blackwell, compute capability **sm_120**), driver **580.x**. For a different CUDA version, pick the matching wheel from the [PyTorch website](https://pytorch.org/get-started/locally/) according to `nvcc --version`.

> **Required reading for RTX 5090 / Blackwell (sm_120):** the minimum support for sm_120 is **CUDA 12.8 + PyTorch ≥ 2.7 (cu128)**. Anything below (e.g. torch 2.5.x on cu124/cu121) fails on the 5090 with `CUDA error: no kernel image is available for execution on the device` and cannot be worked around via PTX/JIT (cuBLAS/cuDNN in older CUDA have no Blackwell code path). The two CUDA submodules must also be recompiled with nvcc 12.8.

### a. Clone the repository

```bash
# clone repository (replace the URL with your remote)
git clone <YOUR_REPO_URL> PanoLOG
cd PanoLOG

# store your dataset here
mkdir -p data

# store your output here
mkdir -p output
```

### b. Create the virtual environment

```bash
conda create -yn PanoLOG python=3.9 pip
conda activate PanoLOG
```

### c. Install PyTorch

- Matching the reference `PanoLOG` environment: **CUDA 12.8 (cu128)** + `PyTorch==2.8.0` (required for RTX 5090 / Blackwell).
- On other machines, install the PyTorch build matching your **`nvcc --version`** (older cards can use cu121/cu124, but sm_120 cannot).

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### d. Install dependencies

First install the Python dependencies (`requirements.txt` contains no torch and no longer auto-compiles the CUDA submodules):

```bash
pip install -r requirements.txt
```

Then **compile** `submodules/diff-gaussian-rasterization` and `submodules/simple-knn` **separately**. We deliberately avoid the auto-compilation triggered by `pip install -r`: it enables build isolation by default, which pulls a mismatched torch into an isolated environment to compile against, producing the wrong ABI/architecture (especially on the 5090). Use the commands below to force compilation against this environment's torch + nvcc 12.8 + sm_120:

```bash
# point to CUDA 12.8 and emit code for Blackwell sm_120
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9;12.0"        # for multi-GPU generality: "8.0;8.6;8.9;9.0;12.0"

# --no-build-isolation: compile against the torch 2.8 already installed in this env
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

If compilation fails, verify that the CUDA versions of the **driver / nvcc / PyTorch** all agree (all three are 12.8 on the reference machine).

> **Common error:** `identifier "FLT_MAX" is undefined` (simple-knn), etc., is caused by newer nvcc no longer implicitly including headers; this repo already fixes it by adding `#include <cfloat>` in the source.

> **Verify the install:** `python -c "import torch; print(torch.cuda.get_arch_list())"` should include `sm_120`; then running `import diff_gaussian_rasterization, simple_knn` without error confirms success.

## 🌐 4K Panoramic Pipeline (ERP / LonLat)

The full pipeline for panoramic (ERP / LonLat) scenes is below. The main differences from the standard pinhole pipeline:

- Input is **equirectangular (ERP) panoramas**, with camera type `camera_type=3`.
- Data comes from **openMVG** (`sfm_data.bin`) and must first be converted to COLMAP txt format.
- **DAP** is used to generate monocular inverse-depth maps and sky masks.
- **Depth-scale alignment** (`make_depth_scale.py`) must be run before training; otherwise depth supervision is harmed by scale mismatch.

### Data directory layout

```text
data/<scene_name>/
├── images -> <storage>/images        # ERP panoramas (symlink)
├── imgs   -> images                  # alias
├── reconstruction/
│   ├── sfm_data.bin                  # openMVG binary (symlink)
│   ├── sfm_data_full.json            # openMVG JSON (exported from bin or symlinked)
│   └── colorized.ply                 # colorized sparse point cloud (symlink)
├── sparse/0/
│   ├── cameras.txt                   # COLMAP format (PINHOLE dummy for ERP)
│   ├── images.txt                    # extrinsics with 2D observations
│   ├── points3D.txt                  # with track info (used for depth alignment)
│   ├── points3D.ply                  # for 3DGS initialization (from colorized.ply)
│   └── depth_params.json             # depth scale/offset alignment (from make_depth_scale)
├── depths/*.png                      # 16-bit monocular inverse-depth maps (from DAP)
├── skymasks/*.png                    # sky masks (foreground=255, sky=0)
├── train.txt                         # training frame names
└── test.txt                          # test frame names (1 in every 8 frames)
```

### Full pipeline (two steps)

Data preparation and training are one command each, wrapped by scripts under `scripts_new/`:

```text
prepare_data.sh   Data prep (one-off): symlink → openMVG→COLMAP → (optional) DAP depth/skymask → depth-scale alignment
      ↓
train.sh          Training (6 stages): coarse → partition → per-block fine-tune → merge → render → metrics
```

Batch over multiple scenes: `scripts_new/run_batch.sh` (edit the `SCENES` list inside).

#### Step 1: Data preparation `scripts_new/prepare_data.sh`

Edit `DATASETS_ROOT` / `DATASET_NAME` / `SCENE_NAME` at the top of the script, then run:

```bash
bash scripts_new/prepare_data.sh
# Pure RGB (no depth/sky masks):  SKIP_DAP=1 bash scripts_new/prepare_data.sh
# Force re-run of all steps:      FORCE=1 bash scripts_new/prepare_data.sh
```

It performs in order: ① symlink images and the openMVG reconstruction from `DATASETS_ROOT/DATASET_NAME` into `data/<SCENE_NAME>/`; ② call `scripts_new/prepare_pano_scene.sh` for the openMVG→COLMAP conversion (plus DAP depth/sky-mask generation when not `SKIP_DAP`); ③ run `tools/make_depth_scale.py` for depth-scale alignment.

If the data is already linked, you can run only the conversion without generating depth:

```bash
bash scripts_new/prepare_pano_scene.sh --scene data/<scene> --skip-dap
```

> **DAP dependency:** generating depth/sky masks requires the DAP model installed at `~/DAP` (`~/DAP/model/model.pth` + `~/DAP/config/infer.yaml`). Skip it with `SKIP_DAP=1` / `--skip-dap` when not using depth supervision; DAP is then not needed.

**Depth-scale alignment (critical):** `prepare_data.sh` internally calls

```bash
python tools/make_depth_scale.py \
    --base_dir data/<scene> --depths_dir data/<scene>/depths \
    --model_type txt --camera_type 3      # camera_type: 1=pinhole z-depth, 3=ERP radial; CPU-only
```

> **If using depth supervision (`use_depth: True`), this step cannot be skipped** — without `depth_params.json`, the unaligned raw depth is used directly as the supervision signal at completely the wrong scale, severely damaging training. For pure-RGB training (`use_depth: False`), the entire depth chain can be omitted (`SKIP_DAP=1`).

#### Step 2: Training `scripts_new/train.sh`

Prepare `config_360/<scene>.yaml` (coarse) and `config_360/<scene>_c4.yaml` (fine) first, then:

```bash
SCENE_NAME=<scene> bash scripts_new/train.sh
# Batch: edit the SCENES list in scripts_new/run_batch.sh, then  bash scripts_new/run_batch.sh
```

It runs the following 6 stages in order (with equivalent commands):

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python train_large.py --config config_360/<scene>.yaml` | Coarse training (30K iterations) |
| 2 | `python data_partition.py --config config_360/<scene>_c4.yaml` | Data partitioning |
| 3 | `python train_large.py --config config_360/<scene>_c4.yaml --block_id N` | Per-block fine-tuning |
| 4 | `python merge.py --config config_360/<scene>_c4.yaml` | Merge |
| 5 | `python render_large.py --config config_360/<scene>_c4.yaml --skip_train` | Render |
| 6 | `python metrics_large.py -m output/<scene>_c4 -t test` | Evaluate |

Existing artifacts are skipped automatically (coarse and each block are skipped if their `point_cloud.ply` exists). Environment variables: `GPU` / `CUDA_VISIBLE_DEVICES`, `BASE_PORT` (default 4070), `MAX_BLOCK_ID` (default auto-derived from `block_dim`).

### Configuration files

`config_360/` is organized into subdirectories by data source. The **basename of the file must be unique**; which subdirectory it lives in does not affect training — every entry point derives the output name via `os.path.basename(--config)`, and `train.sh` searches recursively by filename:

```text
config_360/
├── benchmark/{360roam,omniblender,ricoh360}/   # public, reproducible datasets
├── self_captured/                               # self-captured scenes
└── templates/                                   # config templates per source
```

Each (partitioned) scene has two config files:

- **`<scene>.yaml`** — coarse-training config (`model_path` points to `output/<scene>_coarse`).
- **`<scene>_c4.yaml`** — fine-training config (`pretrain_path` points to the coarse output; `block_dim` defines the number of partitions).

> Small benchmark scenes usually have only a coarse config (single model, no partitioning); the 6-stage divide-and-conquer flow in `scripts_new/train.sh` targets large scenes that have a `_c4`. When calling directly, use the full path `--config config_360/<subdir>/<scene>.yaml`.

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `use_depth: True` | Enable depth supervision (requires depth alignment first, see "Step 1") |
| `use_sky_masks: True/False` | Whether to use sky masks |
| `default_camera_type: 3` | ERP panoramic camera |
| `skybox_num: 100000` | Number of sky-sphere Gaussians (coarse stage) |
| `skybox_locked: True` | Lock the sky sphere during block training |
| `depth_l1_weight_init / final` | Depth-loss weight (decays from init to final) |

### Quick reference: onboarding a new scene

```bash
# 1. Prepare configs: copy and edit config_360/<local_name>.yaml and <local_name>_c4.yaml
#    (source_path → data/<local_name>; set use_depth / use_sky_masks as needed)

# 2. Data prep: edit DATASETS_ROOT / DATASET_NAME / SCENE_NAME=<local_name>
#    at the top of scripts_new/prepare_data.sh, then:
bash scripts_new/prepare_data.sh
#    Pure RGB (no depth/sky masks):  SKIP_DAP=1 bash scripts_new/prepare_data.sh

# 3. Train:
SCENE_NAME=<local_name> bash scripts_new/train.sh
```

## 🙏 Acknowledgements

- [CityGaussian / CityGaussianV2](https://github.com/Linketic/CityGaussian) — large-scale divide-and-conquer training and LoD framework
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — the original 3DGS and differentiable rasterizer
- [OmniGS](https://github.com/liquorleaf/OmniGS) — panoramic (ERP / LonLat) rasterization path, see [`NOTICE_OMNIGS.md`](./NOTICE_OMNIGS.md)
- [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) — anti-aliasing filtering
- [AbsGS](https://github.com/TY424/AbsGS) — absolute-gradient densification

This work was conducted while **Weijian Chen** and **Weibo Yao** were interns at Insta360 Research.

## 📌 Citation

If this project helps your research, please cite it:

```bibtex
@article{panolog2026,
  title     = {Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction},
  author    = {Chen, Weijian and Yao, Weibo and Zhang, Yuhang and Tang, Xiaolin and
               Wang, Guo and Zhang, Weijun and Gao, Xitong and Chen, Yihao and
               Qin, Hongde and Qi, Lu},
  journal   = {arXiv preprint arXiv:2607.08769},
  year      = {2026}
}
```

## 📄 License

This project is released under **CC BY-NC 4.0** (non-commercial use only).



