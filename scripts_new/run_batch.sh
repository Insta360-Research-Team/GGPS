#!/bin/bash
# 批量训练多个数据集，单个失败不影响后续
#
# 用法:  bash scripts_new/run_batch.sh
set -o pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================
#  要跑的场景列表（本地短名，须已有 data/<名> 和 config）
# ============================================================
SCENES=(
    nsk
    nsc
    ftp
)

GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

TOTAL=${#SCENES[@]}
PASS=0
FAIL=0
declare -a FAIL_LIST=()
declare -a TIME_LIST=()

echo ""
echo "############################################################"
echo "#  批量训练: ${SCENES[*]}"
echo "#  GPU: ${GPU}"
echo "############################################################"

for i in "${!SCENES[@]}"; do
    name="${SCENES[$i]}"
    idx=$((i + 1))

    echo ""
    echo "============================================================"
    echo "  [${idx}/${TOTAL}] ${name}  开始: $(date)"
    echo "============================================================"

    SCENE_START=$(date +%s)

    if SCENE_NAME="${name}" bash scripts_new/train.sh; then
        STATUS="OK"
        ((PASS++))
    else
        STATUS="FAIL"
        ((FAIL++))
        FAIL_LIST+=("${name}")
    fi

    SCENE_DUR=$(( $(date +%s) - SCENE_START ))
    TIME_LIST+=("$(printf "  %-20s %s  %dh%dm%ds" "${name}" "${STATUS}" $((SCENE_DUR/3600)) $(((SCENE_DUR%3600)/60)) $((SCENE_DUR%60)))")

    sleep 5
done

echo ""
echo "############################################################"
echo "#  完成: 成功 ${PASS}/${TOTAL}  失败 ${FAIL}/${TOTAL}"
[ ${FAIL} -gt 0 ] && echo "#  失败: ${FAIL_LIST[*]}"
echo "#"
for t in "${TIME_LIST[@]}"; do echo "#${t}"; done
echo "############################################################"
