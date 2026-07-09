#!/bin/bash
# 数据准备（一次性）：链接 → 预处理(openMVG→COLMAP + 深度 + 天空掩码) → 深度对齐
#
# 用法:
#   bash scripts_new/prepare_data.sh
#   SKIP_DAP=1 bash scripts_new/prepare_data.sh      # 跳过深度/天空掩码推理
#   FORCE=1 bash scripts_new/prepare_data.sh          # 强制重跑所有步骤
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
#  修改这里切换数据
# ============================================================
DATASETS_ROOT="/path/to/your/datasets"
DATASET_NAME="NSK"
SCENE_NAME="nsk"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
LLFFHOLD=8
# ============================================================

SRC_DIR="${DATASETS_ROOT}/${DATASET_NAME}"
SCENE_DIR="data/${SCENE_NAME}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo ""
echo "############################################################"
echo "#  数据准备: ${SCENE_NAME}"
echo "#  源: ${SRC_DIR}"
echo "#  目标: ${SCENE_DIR}"
echo "#  GPU: ${GPU}"
echo "############################################################"
echo ""

# ==================== 1. 数据链接 ====================
echo "========== [1/3] 数据链接 =========="

if [ ! -d "${SRC_DIR}/images" ]; then
    echo "错误: 源目录不存在或缺少 images/: ${SRC_DIR}"
    exit 1
fi

SRC_REC="${SRC_DIR}/reconstruction"
DST_REC="${SCENE_DIR}/reconstruction"

[ -L "${SCENE_DIR}" ] && rm "${SCENE_DIR}"
mkdir -p "${SCENE_DIR}"
[ -L "${DST_REC}" ] && rm "${DST_REC}"
mkdir -p "${DST_REC}"

rm -f "${SCENE_DIR}/images" "${SCENE_DIR}/imgs"
ln -sfn "$(realpath "${SRC_DIR}/images")" "${SCENE_DIR}/images"
ln -sfn "images" "${SCENE_DIR}/imgs"
echo "  [OK] images"

[ -f "${SRC_REC}/sfm_data.bin" ] && \
    ln -sfn "$(realpath "${SRC_REC}/sfm_data.bin")" "${DST_REC}/sfm_data.bin" && \
    echo "  [OK] sfm_data.bin"

[ -f "${SRC_REC}/colorized.ply" ] && \
    ln -sfn "$(realpath "${SRC_REC}/colorized.ply")" "${DST_REC}/colorized.ply" && \
    echo "  [OK] colorized.ply"

if [ -f "${DST_REC}/sfm_data_full.json" ] && [ ! -L "${DST_REC}/sfm_data_full.json" ]; then
    echo "  [OK] sfm_data_full.json（已有本地文件）"
elif [ -f "${SRC_REC}/sfm_data_full.json" ]; then
    ln -sfn "$(realpath "${SRC_REC}/sfm_data_full.json")" "${DST_REC}/sfm_data_full.json"
    echo "  [OK] sfm_data_full.json"
elif [ -f "${SRC_REC}/sfm_data.json" ]; then
    ln -sfn "$(realpath "${SRC_REC}/sfm_data.json")" "${DST_REC}/sfm_data_full.json"
    echo "  [OK] sfm_data_full.json <- sfm_data.json"
else
    echo "  [--] 无 JSON，后续从 bin 导出"
fi
echo ""

# ==================== 2. 预处理 ====================
echo "========== [2/3] 预处理 (openMVG→COLMAP + 深度 + 天空掩码) =========="

PREPARE_ARGS="--scene ${SCENE_DIR} --gpu ${GPU} --llffhold ${LLFFHOLD}"
[ "${SKIP_DAP:-0}" = "1" ] && PREPARE_ARGS="${PREPARE_ARGS} --skip-dap"
[ "${FORCE:-0}" = "1" ]    && PREPARE_ARGS="${PREPARE_ARGS} --force"

bash scripts_new/prepare_pano_scene.sh ${PREPARE_ARGS}
echo ""

# ==================== 3. 深度对齐 ====================
echo "========== [3/3] 深度尺度对齐 =========="

if [ "${SKIP_DAP:-0}" = "1" ]; then
    echo "  已跳过 DAP，无需深度对齐"
elif [ -f "${SCENE_DIR}/sparse/0/depth_params.json" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "  已有 depth_params.json，跳过（FORCE=1 可强制重跑）"
else
    if [ ! -d "${SCENE_DIR}/depths" ]; then
        echo "错误: ${SCENE_DIR}/depths/ 不存在"
        exit 1
    fi

    MODEL_TYPE="txt"
    [ ! -f "${SCENE_DIR}/sparse/0/images.txt" ] && [ -f "${SCENE_DIR}/sparse/0/images.bin" ] && MODEL_TYPE="bin"

    python tools/make_depth_scale.py \
        --base_dir "${SCENE_DIR}" \
        --depths_dir "${SCENE_DIR}/depths" \
        --model_type "${MODEL_TYPE}" \
        --camera_type 3

    echo "  完成: ${SCENE_DIR}/sparse/0/depth_params.json"
fi

echo ""
echo "############################################################"
echo "#  数据准备完成: ${SCENE_DIR}"
echo "#  下一步: bash scripts_new/train.sh"
echo "############################################################"
