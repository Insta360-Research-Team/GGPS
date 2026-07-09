<div align="center">

# 面向全景户外重建的几何与梯度分区方法<br><sub>Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction</sub>

Weijian Chen<sup>1,2</sup>, Weibo Yao<sup>1,3</sup>, Yuhang Zhang<sup>1</sup>, Xiaolin Tang<sup>1</sup>, Guo Wang<sup>1</sup>, Weijun Zhang<sup>1</sup>, Xitong Gao<sup>4</sup>, Yihao Chen<sup>1</sup>, Hongde Qin<sup>5</sup>, Lu Qi<sup>1,6</sup>

<sup>1</sup>Insta360 Research &nbsp; <sup>2</sup>中山大学 &nbsp; <sup>3</sup>华南理工大学<br>
<sup>4</sup>中国科学院大学 &nbsp; <sup>5</sup>哈尔滨工程大学 &nbsp; <sup>6</sup>武汉大学

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](#)
[![Homepage](https://img.shields.io/badge/Project-Homepage-1a73e8?logo=googlechrome&logoColor=white)](https://insta360-research-team.github.io/GGPS-Website/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffcc00)](#)

[English](./README.md) | **中文**

<img src="assets/panolog_teaser.png" width="80%" alt="PanoLOG teaser">

</div>

## 📖 简介

将 3DGS 扩展到大规模户外场景，在数据采集与计算上都代价高昂。采用等距柱状投影（ERP）的全景图像凭借完整的 360° 视场可降低采集成本，但由此带来的"处处可见"特性使得依赖局部相机视锥的现有分区策略失效，导致分块优化退化为全局训练。PanoLOG 以**几何与梯度分区策略（G²PS，Geometry and Gradient-based Partitioning Strategy）**应对：在粗训练阶段借助天空球建模与全景单目深度监督获得可靠几何；在精细阶段，G²PS 通过视差驱动的不确定性构建自适应包围体，并以基于梯度的重要性评分分配相机。我们进一步构建了 **Pano360**——首个面向户外场景重建的大规模全景 benchmark。

<div align="center">
<img src="assets/panolog_pipeline.png" width="85%" alt="PanoLOG pipeline (G²PS)">
<p><i>两阶段 coarse-to-fine 流程与几何-梯度分区策略（G²PS）。</i></p>
</div>

数据集需自行准备（COLMAP / openMVG 结果），训练输出默认在 `output/`（已在 `.gitignore` 中忽略）。

## 🗺️ 更新路线图 Roadmap

我们正在推进开源发布，进度将在此持续记录。

> **Surprise！** 我们将于 7 月 15 日对外提供一款基于 **Unreal Engine 的 3DGS 渲染插件**，让你在渲染引擎内直接使用自己重建的模型。提供带水印的免费版本；付费版（一次买断）随后提供链接。

- [ ] **2026-07-09** — 提供完整的训练代码。
- [ ] **2026-07-10** — 提供两个经过授权的全景 `.ply` 模型，供复现验证。
- [ ] **2026-07-15** — 提供 Pano360 数据集。
- [ ] **2026-07-15** — 提供免费版（带水印）渲染插件；付费版（一次买断）随后提供链接。

> 欢迎提出建议与问题，请通过 issue 反馈。

## ⚙️ 环境

需要 **NVIDIA 驱动**；编译 `submodules` 时依赖本机 **CUDA Toolkit**（可用 `nvcc --version` 查看，需与下方 PyTorch 的 CUDA 版本一致或兼容）。

**当前本机 conda 环境 `PanoLOG`（供对齐）：** Python **3.9**，PyTorch **2.8.0**（**cu128** / CUDA **12.8**）；系统 `nvcc` **12.8**（`/usr/local/cuda-12.8`），GPU **RTX 5090 D**（Blackwell，算力 **sm_120**），驱动 **580.x**。若 CUDA 不同，请按 `nvcc --version` 在 [PyTorch 官网](https://pytorch.org/get-started/locally/) 另选轮子。

> **RTX 5090 / Blackwell（sm_120）必读：** sm_120 的支持下限是 **CUDA 12.8 + PyTorch ≥ 2.7（cu128）**。低于此（如 cu124/cu121 的 torch 2.5.x）在 5090 上会直接报 `CUDA error: no kernel image is available for execution on the device`，且无法通过 PTX/JIT 绕过（cuBLAS/cuDNN 在旧 CUDA 里没有 Blackwell 代码路径）。两个 CUDA 子模块也必须用 nvcc 12.8 重新编译。

### a. 克隆存储库

```bash
# clone repository（将 URL 换成你的远程仓库地址）
git clone <YOUR_REPO_URL> PanoLOG
cd PanoLOG

# store your dataset here
mkdir -p data

# store your output here
mkdir -p output
```

### b. 创建虚拟环境

```bash
# create virtual environment
conda create -yn PanoLOG python=3.9 pip
conda activate PanoLOG
```

### c. 安装 PyTorch

- 与当前本机 `PanoLOG` 环境一致：**CUDA 12.8（cu128）** + `PyTorch==2.8.0`（RTX 5090/Blackwell 必需）。
- 其他机器请安装与 **`nvcc --version`** 匹配的 PyTorch（旧卡可用 cu121/cu124，但 sm_120 不行）。

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

### d. 安装依赖

先装 Python 依赖（`requirements.txt` 不含 torch，也不再自动编译 CUDA 子模块）：

```bash
pip install -r requirements.txt
```

再**单独编译** `submodules/diff-gaussian-rasterization` 与 `submodules/simple-knn`。这里特意不走 `pip install -r` 的自动编译：它默认开启 build isolation，会在隔离环境临时拉一个不匹配的 torch 来编译，导致 ABI/架构不对（5090 上尤甚）。用下面的方式强制用本环境的 torch + nvcc 12.8 + sm_120：

```bash
# 指向 CUDA 12.8，并为 Blackwell sm_120 生成代码
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.9;12.0"        # 多卡通用可写 "8.0;8.6;8.9;9.0;12.0"

# --no-build-isolation：用当前环境已装好的 torch 2.8 编译
pip install --no-build-isolation ./submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

若编译失败，请核对 **驱动 / nvcc / PyTorch 的 CUDA 版本** 是否一致（本机三者均为 12.8）。

> **常见报错**：`identifier "FLT_MAX" is undefined`（simple-knn）等，是新 nvcc 不再隐式包含头文件所致；本仓库已在源码补上 `#include <cfloat>` 修复。

> **验证安装**：`python -c "import torch; print(torch.cuda.get_arch_list())"` 应包含 `sm_120`；再跑一遍 `import diff_gaussian_rasterization, simple_knn` 不报错即可。

## 🌐 4K 全景场景流程（ERP / LonLat）

全景（ERP / LonLat）场景的完整流程如下。与常规 pinhole 流程相比，主要区别：

- 输入为 **equirectangular (ERP) 全景图**，相机类型为 `camera_type=3`
- 数据源自 **openMVG**（`sfm_data.bin`），需先转换为 COLMAP txt 格式
- 使用 **DAP** 生成单目逆深度图和天空掩码
- 训练前需运行 **深度尺度对齐**（`make_depth_scale.py`），否则深度监督会因尺度不匹配而损害训练

### 数据目录结构

```text
data/<scene_name>/
├── images -> <盘阵>/images          # ERP 全景图（软链接）
├── imgs   -> images                 # 别名
├── reconstruction/
│   ├── sfm_data.bin                 # openMVG 二进制（软链接）
│   ├── sfm_data_full.json           # openMVG JSON（从 bin 导出或软链接）
│   └── colorized.ply                # 着色稀疏点云（软链接）
├── sparse/0/
│   ├── cameras.txt                  # COLMAP 格式（PINHOLE dummy for ERP）
│   ├── images.txt                   # 含 2D 观测的外参
│   ├── points3D.txt                 # 含 track 信息（深度对齐用）
│   ├── points3D.ply                 # 3DGS 初始化用（来自 colorized.ply）
│   └── depth_params.json            # 深度 scale/offset 对齐参数（make_depth_scale 生成）
├── depths/*.png                     # 16-bit 单目逆深度图（DAP 生成）
├── skymasks/*.png                   # 天空掩码（前景=255, 天空=0）
├── train.txt                        # 训练集帧名
└── test.txt                         # 测试集帧名（每 8 帧取 1 帧）
```

### 完整流程（两步）

数据准备与训练各一条命令，分别由 `scripts_new/` 下的脚本封装：

```text
prepare_data.sh   数据准备（一次性）：软链接 → openMVG→COLMAP → (可选)DAP 深度/天空掩码 → 深度尺度对齐
      ↓
train.sh          训练（6 阶段）：粗训练 → 分区 → 各 block 微调 → 合并 → 渲染 → 指标
```

批量多场景：`scripts_new/run_batch.sh`（编辑其中的 `SCENES` 列表）。

#### 第一步：数据准备 `scripts_new/prepare_data.sh`

编辑脚本顶部的 `DATASETS_ROOT` / `DATASET_NAME` / `SCENE_NAME`，然后运行：

```bash
bash scripts_new/prepare_data.sh
# 纯 RGB（不要深度/天空掩码）：SKIP_DAP=1 bash scripts_new/prepare_data.sh
# 强制重跑所有步骤：           FORCE=1 bash scripts_new/prepare_data.sh
```

它依次完成：① 从 `DATASETS_ROOT/DATASET_NAME` 软链接 images 与 openMVG 重建到 `data/<SCENE_NAME>/`；② 调用 `scripts_new/prepare_pano_scene.sh` 做 openMVG→COLMAP 转换（非 `SKIP_DAP` 时再用 DAP 生成深度/天空掩码）；③ 运行 `tools/make_depth_scale.py` 做深度尺度对齐。

若数据已链接好，也可单独只做转换、不生成深度：

```bash
bash scripts_new/prepare_pano_scene.sh --scene data/<scene> --skip-dap
```

> **DAP 依赖**：生成深度/天空掩码需要 DAP 模型装在 `~/DAP`（`~/DAP/model/model.pth` + `~/DAP/config/infer.yaml`）。不做深度监督时用 `SKIP_DAP=1` / `--skip-dap` 跳过，无需安装 DAP。

**深度尺度对齐（关键）**：`prepare_data.sh` 内部会调用

```bash
python tools/make_depth_scale.py \
    --base_dir data/<scene> --depths_dir data/<scene>/depths \
    --model_type txt --camera_type 3      # camera_type: 1=pinhole z-depth, 3=ERP radial；纯 CPU
```

> **如果使用深度监督（`use_depth: True`），此步不可跳过**——没有 `depth_params.json` 时，未对齐的原始深度会被直接当作监督信号，尺度完全错误，严重破坏训练。纯 RGB 训练（`use_depth: False`）则整条深度链都可省（`SKIP_DAP=1`）。

#### 第二步：训练 `scripts_new/train.sh`

需先备好 `config_360/<scene>.yaml`（粗）与 `config_360/<scene>_c4.yaml`（精），然后：

```bash
SCENE_NAME=<scene> bash scripts_new/train.sh
# 批量：编辑 scripts_new/run_batch.sh 的 SCENES 列表后， bash scripts_new/run_batch.sh
```

依次执行 6 阶段（各步等价命令）：

| 步骤 | 命令 | 说明 |
|------|------|------|
| 1 | `python train_large.py --config config_360/<scene>.yaml` | 粗训练（30K 迭代） |
| 2 | `python data_partition.py --config config_360/<scene>_c4.yaml` | 数据分区 |
| 3 | `python train_large.py --config config_360/<scene>_c4.yaml --block_id N` | 各 block 精细训练 |
| 4 | `python merge.py --config config_360/<scene>_c4.yaml` | 合并 |
| 5 | `python render_large.py --config config_360/<scene>_c4.yaml --skip_train` | 渲染 |
| 6 | `python metrics_large.py -m output/<scene>_c4 -t test` | 评估 |

已有产物会自动跳过（coarse 与各 block 的 `point_cloud.ply` 存在即跳过）。环境变量：`GPU`/`CUDA_VISIBLE_DEVICES`、`BASE_PORT`（默认 4070）、`MAX_BLOCK_ID`（默认从 `block_dim` 自动推导）。

### 配置文件说明

`config_360/` 按数据来源分子目录组织，**文件名（basename）唯一**即可，放在哪个子目录不影响训练——所有入口都用 `os.path.basename(--config)` 推导输出名，`train.sh` 也按文件名递归查找：

```text
config_360/
├── benchmark/{360roam,omniblender,ricoh360}/   # 公开可复现数据集
├── self_captured/                               # 自采场景
└── templates/                                   # 各来源的配置模板
```

每个（需分块的）场景有两个配置文件：

- **`<scene>.yaml`** — 粗训练配置（`model_path` 指向 `output/<scene>_coarse`）
- **`<scene>_c4.yaml`** — 精细训练配置（`pretrain_path` 指向 coarse 的输出，`block_dim` 定义分区数）

> benchmark 里的小场景通常只有粗训练配置（单模型，不分块）；`scripts_new/train.sh` 的 6 阶段分治流程针对有 `_c4` 的大场景。直接调用时用完整路径 `--config config_360/<子目录>/<scene>.yaml`。

关键参数：

| 参数 | 说明 |
|------|------|
| `use_depth: True` | 启用深度监督（需先做深度对齐，见"第一步"） |
| `use_sky_masks: True/False` | 是否使用天空掩码 |
| `default_camera_type: 3` | ERP 全景相机 |
| `skybox_num: 100000` | 天空球 Gaussian 数量（coarse 阶段） |
| `skybox_locked: True` | block 训练时锁定天空球 |
| `depth_l1_weight_init / final` | 深度 loss 权重（从 init 衰减到 final） |

### 快速参考：新场景接入

```bash
# 1. 准备 config：复制并修改 config_360/<local_name>.yaml 与 <local_name>_c4.yaml
#    （source_path 指向 data/<local_name>；use_depth / use_sky_masks 按需）

# 2. 数据准备：编辑 scripts_new/prepare_data.sh 顶部的
#    DATASETS_ROOT / DATASET_NAME / SCENE_NAME=<local_name>，然后：
bash scripts_new/prepare_data.sh
#    纯 RGB（无深度/天空掩码）：SKIP_DAP=1 bash scripts_new/prepare_data.sh

# 3. 训练：
SCENE_NAME=<local_name> bash scripts_new/train.sh
```

## 🙏 致谢与第三方代码 Acknowledgements

- [CityGaussian / CityGaussianV2](https://github.com/Linketic/CityGaussian) — 大规模分治训练与 LoD 框架
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — 原始 3DGS 与可微光栅化
- [OmniGS](https://github.com/liquorleaf/OmniGS) — 全景（ERP / LonLat）光栅化路径，见 [`NOTICE_OMNIGS.md`](./NOTICE_OMNIGS.md)
- [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) — 抗锯齿滤波
- [AbsGS](https://github.com/TY424/AbsGS) — 绝对梯度稠密化

本工作由 **Weijian Chen** 与 **Weibo Yao** 于 Insta360 Research 实习期间完成。

## 📌 引用 Citation

如果本项目对你的研究有帮助，请引用（论文信息待正式发表后更新）：

```bibtex
@inproceedings{panolog,
  title     = {Geometry and Gradient-based Partitioning for Panoramic Outdoor Reconstruction},
  author    = {Chen, Weijian and Yao, Weibo and Zhang, Yuhang and Tang, Xiaolin and
               Wang, Guo and Zhang, Weijun and Gao, Xitong and Chen, Yihao and
               Qin, Hongde and Qi, Lu},
  booktitle = {arXiv preprint},
  year      = {2025}
}
```

## 📄 许可证 License

本项目基于 **CC BY-NC 4.0** 发布（仅限非商业用途）。
