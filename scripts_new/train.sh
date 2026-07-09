#!/bin/bash
# 训练 + 评估：粗训练 → 分区 → block 微调 → 合并 → 渲染 → 指标
#
# 用法:
#   bash scripts_new/train.sh
#   SCENE_NAME=nsk bash scripts_new/train.sh    # 指定场景
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
#  修改这里切换场景
# ============================================================
SCENE_NAME="${SCENE_NAME:-a1_bijiashan_4k}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
BASE_PORT="${BASE_PORT:-4070}"
# ============================================================

COARSE_CONFIG="${SCENE_NAME}"
FINE_CONFIG="${SCENE_NAME}_c4"
SCENE_DIR="data/${SCENE_NAME}"
# 配置可放在 config_360/ 的任意子目录下，按文件名递归查找
COARSE_YAML="$(find config_360 -name "${COARSE_CONFIG}.yaml" | head -1)"
FINE_YAML="$(find config_360 -name "${FINE_CONFIG}.yaml" | head -1)"
COARSE_PLY="output/${COARSE_CONFIG}_coarse/point_cloud/iteration_30000/point_cloud.ply"
TIME_LOG="output/${FINE_CONFIG}_time.txt"

# 从 block_dim 自动推导 MAX_BLOCK_ID
if [ -z "${MAX_BLOCK_ID}" ] && [ -f "${FINE_YAML}" ]; then
    MAX_BLOCK_ID=$(python3 -c "
import re, math
with open('${FINE_YAML}') as f:
    m = re.search(r'block_dim:\s*\[([^\]]+)\]', f.read())
print(math.prod(int(x) for x in m.group(1).split(',')) - 1 if m else 1)
" 2>/dev/null)
fi
MAX_BLOCK_ID="${MAX_BLOCK_ID:-1}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# 检查 config
if [ -z "${COARSE_YAML}" ] || [ ! -f "${COARSE_YAML}" ]; then
    echo "错误: 在 config_360/ 下找不到 ${COARSE_CONFIG}.yaml，请先创建配置文件"
    exit 1
fi
if [ -z "${FINE_YAML}" ] || [ ! -f "${FINE_YAML}" ]; then
    echo "错误: 在 config_360/ 下找不到 ${FINE_CONFIG}.yaml，请先创建配置文件"
    exit 1
fi

echo ""
echo "############################################################"
echo "#  训练: ${SCENE_NAME}"
echo "#  粗训练: ${COARSE_YAML}"
echo "#  分块:   ${FINE_YAML}"
echo "#  GPU: ${GPU}   blocks: 0..${MAX_BLOCK_ID}"
echo "############################################################"
echo ""

mkdir -p output
echo "${FINE_CONFIG} 训练时间记录" > "${TIME_LOG}"
echo "开始时间: $(date)" >> "${TIME_LOG}"
echo "==============================" >> "${TIME_LOG}"

TOTAL_START=$(date +%s)
port=${BASE_PORT}

record_time() {
    local name=$1 start=$2
    local dur=$(( $(date +%s) - start ))
    printf "%s: %dh %dm %ds (%ds)\n" "$name" $((dur/3600)) $(((dur%3600)/60)) $((dur%60)) $dur | tee -a "${TIME_LOG}"
}

# --- 1. 粗训练 ---
if [ -f "${COARSE_PLY}" ]; then
    echo "========== [1/6] 粗训练（已有，跳过） =========="
    echo "  ${COARSE_PLY}"
else
    echo "========== [1/6] 粗训练 =========="
    STEP_START=$(date +%s)
    WANDB_MODE=offline python train_large.py --config "${COARSE_YAML}"
    record_time "粗训练" ${STEP_START}
fi
echo ""

# --- 2. 数据分区 ---
echo "========== [2/6] 数据分区 =========="
STEP_START=$(date +%s)
python data_partition.py --config "${FINE_YAML}"
record_time "数据分区" ${STEP_START}
echo ""

# --- 3. block 微调 ---
echo "========== [3/6] block 微调 =========="
STEP_START=$(date +%s)
SKIPPED_BLOCKS=0
for bid in $(seq 0 ${MAX_BLOCK_ID}); do
    BLOCK_PLY="output/${FINE_CONFIG}/cells/cell${bid}/point_cloud_blocks/scale_1.0/iteration_30000/point_cloud.ply"
    if [ -f "${BLOCK_PLY}" ]; then
        echo "  block ${bid}/${MAX_BLOCK_ID} — 已有，跳过"
        SKIPPED_BLOCKS=$((SKIPPED_BLOCKS + 1))
    else
        echo "  block ${bid}/${MAX_BLOCK_ID} — 训练中 ..."
        WANDB_MODE=offline python train_large.py \
            --config "${FINE_YAML}" \
            --block_id ${bid} \
            --port ${port}
    fi
    port=$((port + 1))
done
if [ ${SKIPPED_BLOCKS} -eq $((MAX_BLOCK_ID + 1)) ]; then
    echo "  所有 block 均已完成"
else
    record_time "block微调" ${STEP_START}
fi
echo ""

# --- 4. 合并 ---
echo "========== [4/6] 合并 =========="
STEP_START=$(date +%s)
python merge.py --config "${FINE_YAML}"
record_time "合并" ${STEP_START}

TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))
echo "==============================" >> "${TIME_LOG}"
printf "训练总时间: %dh %dm %ds (%ds)\n" $((TOTAL_DUR/3600)) $(((TOTAL_DUR%3600)/60)) $((TOTAL_DUR%60)) $TOTAL_DUR >> "${TIME_LOG}"
echo "结束时间: $(date)" >> "${TIME_LOG}"
echo ""
cat "${TIME_LOG}"
echo ""

# --- 5. 渲染 ---
echo "========== [5/6] 渲染 =========="
python render_large.py --config "${FINE_YAML}" --skip_train
echo ""

# --- 6. 计算指标 ---
echo "========== [6/6] 计算指标 =========="
python metrics_large.py -m "output/${FINE_CONFIG}" -t test

echo ""
echo "############################################################"
echo "#  完成: output/${FINE_CONFIG}/"
echo "#  时间记录: ${TIME_LOG}"
echo "############################################################"
