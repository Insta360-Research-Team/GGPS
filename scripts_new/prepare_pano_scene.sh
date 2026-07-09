#!/bin/bash
# 全景场景数据预处理脚本（只生成数据，不做对齐和训练）
#
# 流程：
#   1) openMVG sfm_data → COLMAP 格式（images.txt 含 2D 观测 + points3D.txt 含 track）
#   2) DAP 推理 → 深度图 + 天空掩码
#
# 输入要求：
#   <scene>/
#   ├── images/                    ERP 全景图
#   └── reconstruction/
#       ├── sfm_data.bin           或已有 sfm_data_full.json（仅有 bin 时会自动导出 JSON）
#       ├── sfm_data_full.json     openMVG JSON（含 structure）
#       └── colorized.ply          稀疏点云
#
# 输出：
#   <scene>/sparse/0/cameras.txt
#   <scene>/sparse/0/images.txt      含 openMVG 特征匹配的 POINTS2D
#   <scene>/sparse/0/points3D.txt    含 track 信息（深度对齐用）
#   <scene>/sparse/0/points3D.ply    3DGS 初始化用
#   <scene>/depths/*.png             16-bit 逆深度
#   <scene>/skymasks/*.png           天空掩码（前景=255, 天空=0）
#   <scene>/train.txt / test.txt
#   <scene>/imgs -> images
#
# 用法：
#   bash scripts_new/prepare_pano_scene.sh --scene data/a1nanshan
#   bash scripts_new/prepare_pano_scene.sh --scene data/a1nanshan --gpu 1
#   bash scripts_new/prepare_pano_scene.sh --scene data/a1nanshan --force
#   bash scripts_new/prepare_pano_scene.sh --scene data/x5_insta_2k --skip-dap   # 不需深度/天空掩码时
#
# 自动检测：已有产物的步骤会自动跳过，加 --force 可强制全部重跑

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ====== 默认参数 ======
SCENE=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"
IMAGES="images"
LLFFHOLD=8
SFM_JSON="reconstruction/sfm_data_full.json"
PLY_SOURCE="reconstruction/colorized.ply"
FORCE=0
SKIP_DAP=0

# ====== 解析命令行参数 ======
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)       SCENE="$2"; shift 2 ;;
        --gpu)         GPU="$2"; shift 2 ;;
        --images)      IMAGES="$2"; shift 2 ;;
        --llffhold)    LLFFHOLD="$2"; shift 2 ;;
        --sfm-json)    SFM_JSON="$2"; shift 2 ;;
        --ply-source)  PLY_SOURCE="$2"; shift 2 ;;
        --force)       FORCE=1; shift ;;
        --skip-dap)    SKIP_DAP=1; shift ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash scripts_new/prepare_pano_scene.sh --scene <场景目录> [选项]"
            exit 1 ;;
    esac
done

if [ -z "${SCENE}" ]; then
    echo "错误: 必须指定 --scene 参数"
    echo ""
    echo "用法:"
    echo "  bash scripts_new/prepare_pano_scene.sh --scene data/a1nanshan"
    echo ""
    echo "选项:"
    echo "  --gpu <id>          GPU 编号（默认 0）"
    echo "  --images <dir>      图像子目录名（默认 images）"
    echo "  --llffhold <n>      每 n 帧取 1 帧做 test（默认 8）"
    echo "  --sfm-json <path>   sfm_data JSON 相对路径（默认 reconstruction/sfm_data_full.json）"
    echo "  --ply-source <path> 点云 PLY 相对路径（默认 reconstruction/colorized.ply）"
    echo "  --force             强制重新运行所有步骤（忽略已有产物）"
    echo "  --skip-dap          跳过 DAP 深度/天空掩码（训练 use_depth/use_sky_masks 均为 False 时可用）"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo ""
echo "############################################################"
if [ "${SKIP_DAP}" = "1" ]; then
    echo "# 全景数据预处理（生成 COLMAP；已跳过 DAP）"
else
    echo "# 全景数据预处理（生成 COLMAP + 深度 + 天空掩码）"
fi
echo "# 场景: ${SCENE}"
echo "# GPU:  ${GPU}"
echo "############################################################"
echo ""

# ====== 步骤 1: openMVG → COLMAP ======
STEP1_DONE=0
if [ -f "${SCENE}/sparse/0/points3D.txt" ] && [ -f "${SCENE}/sparse/0/images.txt" ]; then
    HAS_OBS=$(awk 'NR>3 && NR%2==1 && NF>0 {print "yes"; exit}' "${SCENE}/sparse/0/images.txt")
    if [ "${HAS_OBS}" = "yes" ]; then
        STEP1_DONE=1
    fi
fi

if [ "${STEP1_DONE}" = "1" ] && [ "${FORCE}" != "1" ]; then
    echo "========== [1/2] openMVG → COLMAP（已有，跳过） =========="
    echo "  已有 points3D.txt + images.txt（含 2D 观测）"
    echo ""
else
    echo "========== [1/2] openMVG → COLMAP 转换 =========="
    echo "  sfm_data:   ${SCENE}/${SFM_JSON}"
    echo "  ply_source: ${SCENE}/${PLY_SOURCE}"
    echo "  llffhold:   ${LLFFHOLD}"
    echo ""

    # DATASETS_4K 等Release常仅有 sfm_data.bin，无 JSON：先从 bin 导出再转换
    if [ ! -f "${SCENE}/${SFM_JSON}" ] && [ -f "${SCENE}/reconstruction/sfm_data.bin" ]; then
        echo "  未找到 ${SFM_JSON}，从 reconstruction/sfm_data.bin 导出 ..."
        python tools/parse_openmvg_bin.py "${SCENE}" \
            --bin reconstruction/sfm_data.bin \
            --out "${SFM_JSON}"
        echo ""
    fi

    python tools/convert_openmvg_to_colmap.py \
        --scene "${SCENE}" \
        --sfm-json "${SFM_JSON}" \
        --ply-source "${PLY_SOURCE}" \
        --llffhold "${LLFFHOLD}" \
        --images-subdir "${IMAGES}"

    echo ""
    echo "  生成文件:"
    ls -lh "${SCENE}/sparse/0/cameras.txt"   2>/dev/null || true
    ls -lh "${SCENE}/sparse/0/images.txt"    2>/dev/null || true
    ls -lh "${SCENE}/sparse/0/points3D.txt"  2>/dev/null || true
    ls -lh "${SCENE}/sparse/0/points3D.ply"  2>/dev/null || true
    echo ""
fi

# ====== 步骤 2: DAP 深度 + 天空掩码 ======
IMG_COUNT=$(ls "${SCENE}/${IMAGES}/"*.png "${SCENE}/${IMAGES}/"*.jpg "${SCENE}/${IMAGES}/"*.jpeg 2>/dev/null | grep -v '/\._' | wc -l)
DEPTH_COUNT=$(ls "${SCENE}/depths/"*.png 2>/dev/null | grep -v '/\._' | wc -l)
SKY_COUNT=$(ls "${SCENE}/skymasks/"*.png 2>/dev/null | grep -v '/\._' | wc -l)

if [ "${SKIP_DAP}" = "1" ]; then
    echo "========== [2/2] DAP 深度 + 天空掩码（--skip-dap，跳过） =========="
    echo ""
elif [ "${DEPTH_COUNT}" -ge "${IMG_COUNT}" ] && [ "${SKY_COUNT}" -ge "${IMG_COUNT}" ] \
   && [ "${IMG_COUNT}" -gt 0 ] && [ "${FORCE}" != "1" ]; then
    echo "========== [2/2] DAP 深度 + 天空掩码（已有，跳过） =========="
    echo "  depths:   ${DEPTH_COUNT} 帧"
    echo "  skymasks: ${SKY_COUNT} 帧"
    echo ""
else
    echo "========== [2/2] DAP 深度推理 + 天空掩码 =========="
    echo "  图像目录: ${SCENE}/${IMAGES} (${IMG_COUNT} 帧)"
    echo ""

    python tools/generate_depth_and_skymask.py \
        --scene "${SCENE}" \
        --images "${IMAGES}" \
        --gpu "${GPU}"

    echo ""
fi

# ====== 最终检查 ======
echo "============================================================"
echo "数据预处理完成，检查输出："
echo "------------------------------------------------------------"

check_file() {
    if [ -e "$1" ]; then
        echo "  ✓ $1"
    else
        echo "  ✗ $1 (缺失)"
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        local count=$(ls "$1/"*.png 2>/dev/null | wc -l)
        echo "  ✓ $1/ (${count} 帧)"
    else
        echo "  ✗ $1/ (缺失)"
    fi
}

echo ""
echo "COLMAP 数据:"
check_file "${SCENE}/sparse/0/cameras.txt"
check_file "${SCENE}/sparse/0/images.txt"
check_file "${SCENE}/sparse/0/points3D.txt"
check_file "${SCENE}/sparse/0/points3D.ply"

echo ""
echo "深度数据:"
if [ "${SKIP_DAP}" = "1" ]; then
    echo "  （已跳过 DAP，无需 depths/skymasks）"
else
    check_dir  "${SCENE}/depths"
    check_dir  "${SCENE}/skymasks"
fi

echo ""
echo "训练/测试划分:"
check_file "${SCENE}/train.txt"
check_file "${SCENE}/test.txt"
check_file "${SCENE}/imgs"

echo ""
echo "============================================================"
echo "下一步: 运行深度对齐 + 训练"
echo "  bash scripts_new/prepare_data.sh   # 补深度对齐"
echo "  bash scripts_new/train.sh          # 训练"
echo "============================================================"
