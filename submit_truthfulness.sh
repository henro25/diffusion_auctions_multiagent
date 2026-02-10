#!/bin/bash
# Central submission script for truthfulness experiment.
# Submits all generation + alignment jobs across both accounts.
#
# Usage: ./submit_truthfulness.sh [gen|align|all]
#   gen   - submit generation jobs only (default)
#   align - submit alignment jobs only
#   all   - submit both

set -e

BASE=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent
WORKER="$BASE/run_truthfulness_worker.sh"
LOGS="$BASE/logs"
NUM_CHUNKS=6
MODE="${1:-gen}"

mkdir -p "$LOGS"

# Account/partition configs
KEMP_ACCOUNT="kempner_ydu_lab"
KEMP_PARTITIONS="kempner_h100"
KEMP_GRES="gpu:1"
HLAK_ACCOUNT="hlakkaraju_lab"
HLAK_PARTITIONS="seas_gpu,gpu,gpu_h200"
HLAK_GRES="gpu:nvidia_a100-sxm4-80gb:1"

submit_gen_job() {
    local job_name=$1
    local config=$2
    local prompt_idx=$3
    local chunk=$4
    local account=$5
    local partitions=$6
    local gres=$7

    sbatch \
        --job-name="$job_name" \
        --partition="$partitions" \
        --gres="$gres" \
        --mem=200gb \
        -t 0-01:00 \
        -o "$LOGS/%x_%j.out" \
        -e "$LOGS/%x_%j.err" \
        --mail-user=lilliansun@college.harvard.edu \
        --mail-type=END,FAIL \
        --account="$account" \
        "$WORKER" "$config" "$prompt_idx" "$chunk" "$NUM_CHUNKS"
}

submit_align_job() {
    local job_name=$1
    local config=$2
    local account=$3
    local partitions=$4
    local gres=$5

    sbatch \
        --job-name="$job_name" \
        --partition="$partitions" \
        --gres="$gres" \
        --mem=200gb \
        -t 0-02:00 \
        -o "$LOGS/%x_%j.out" \
        -e "$LOGS/%x_%j.err" \
        --mail-user=lilliansun@college.harvard.edu \
        --mail-type=END,FAIL \
        --account="$account" \
        "$BASE/run_truthfulness_alignment_worker.sh" "$config"
}

if [ "$MODE" = "gen" ] || [ "$MODE" = "all" ]; then
    echo "=== Submitting generation jobs (${NUM_CHUNKS} chunks per prompt) ==="

    REG_CONFIG="../config/truthfulness_generation_regular.json"
    COMP_CONFIG="../config/truthfulness_generation_competitive.json"

    # Regular prompts 0-3 on kempner
    for p in 0 1 2 3; do
        for c in $(seq 0 $((NUM_CHUNKS - 1))); do
            submit_gen_job "tr_rp${p}c${c}" "$REG_CONFIG" "$p" "$c" "$KEMP_ACCOUNT" "$KEMP_PARTITIONS" "$KEMP_GRES"
        done
    done
    echo "  Submitted regular p0-p3 (24 jobs) on $KEMP_ACCOUNT"

    # Regular prompts 4-5 on hlakkaraju
    for p in 4 5; do
        for c in $(seq 0 $((NUM_CHUNKS - 1))); do
            submit_gen_job "tr_rp${p}c${c}" "$REG_CONFIG" "$p" "$c" "$HLAK_ACCOUNT" "$HLAK_PARTITIONS" "$HLAK_GRES"
        done
    done
    echo "  Submitted regular p4-p5 (12 jobs) on $HLAK_ACCOUNT"

    # Competitive prompts 0-2 on hlakkaraju
    for p in 0 1 2; do
        for c in $(seq 0 $((NUM_CHUNKS - 1))); do
            submit_gen_job "tr_cp${p}c${c}" "$COMP_CONFIG" "$p" "$c" "$HLAK_ACCOUNT" "$HLAK_PARTITIONS" "$HLAK_GRES"
        done
    done
    echo "  Submitted competitive p0-p2 (18 jobs) on $HLAK_ACCOUNT"

    echo "  Total: 54 generation jobs submitted"
fi

if [ "$MODE" = "align" ] || [ "$MODE" = "all" ]; then
    echo "=== Submitting alignment jobs ==="

    submit_align_job "tr_align_reg" \
        "../config/truthfulness_alignment_clip_regular.json" \
        "$KEMP_ACCOUNT" "$KEMP_PARTITIONS" "$KEMP_GRES"

    submit_align_job "tr_align_comp" \
        "../config/truthfulness_alignment_clip_competitive.json" \
        "$HLAK_ACCOUNT" "$HLAK_PARTITIONS" "$HLAK_GRES"

    echo "  Submitted 2 alignment jobs"
fi

echo ""
echo "Monitor with: squeue -u lilliansun"
echo "Check logs:   ls -lt $LOGS/tr_*.out | head"
