#!/bin/bash
# Submission script for rebuttal experiments (Q1: dominant prompts, Q2: numerical baseline).
#
# Usage:
#   ./submit_rebuttal.sh pilot       # Phase 0: prompt 0 only for Q1 and Q2
#   ./submit_rebuttal.sh q1_full     # Q1 prompts 1-4 (after pilot review)
#   ./submit_rebuttal.sh q2_full     # Q2 prompts 1-4 (after pilot review)
#   ./submit_rebuttal.sh full_all    # All remaining prompts for Q1 and Q2
#   ./submit_rebuttal.sh align       # Alignment for both Q1 and Q2 (run after gen completes)
#   ./submit_rebuttal.sh align_q1    # Alignment Q1 only
#   ./submit_rebuttal.sh align_q2    # Alignment Q2 only
#
# Env vars:
#   CLUSTER=kempner|hlak     # default: kempner. Pick which cluster to submit to.
#   DUAL=1                   # if set, submit duplicate copy on the OTHER cluster (skip-existing makes it safe).

set -e

BASE=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent
WORKER_AUCTION="$BASE/run_truthfulness_worker.sh"
WORKER_NUMERICAL="$BASE/run_q2_numerical_worker.sh"
ALIGN_WORKER="$BASE/run_truthfulness_alignment_worker.sh"
LOGS="$BASE/logs"
MODE="${1:-pilot}"
CLUSTER="${CLUSTER:-kempner}"

mkdir -p "$LOGS"

# Cluster configurations
KEMP_ACCOUNT="kempner_ydu_lab"
KEMP_PARTITIONS="kempner_h100"
KEMP_GRES="gpu:1"
HLAK_ACCOUNT="hlakkaraju_lab"
HLAK_PARTITIONS="seas_gpu,gpu,gpu_h200"
HLAK_GRES="gpu:nvidia_a100-sxm4-80gb:1"

cluster_config() {
    case "$1" in
        kempner) echo "$KEMP_ACCOUNT|$KEMP_PARTITIONS|$KEMP_GRES" ;;
        hlak)    echo "$HLAK_ACCOUNT|$HLAK_PARTITIONS|$HLAK_GRES" ;;
        *) echo "ERR" ;;
    esac
}

other_cluster() {
    case "$1" in
        kempner) echo "hlak" ;;
        hlak)    echo "kempner" ;;
    esac
}

submit_q1_gen_on() {
    local cluster=$1 p=$2
    IFS='|' read -r ACC PART GRES <<< "$(cluster_config $cluster)"
    sbatch \
        --job-name="q1_p${p}_${cluster}" \
        --partition="$PART" \
        --gres="$GRES" \
        --mem=200gb \
        -t 0-00:30 \
        -o "$LOGS/q1_p${p}_${cluster}_%j.out" \
        -e "$LOGS/q1_p${p}_${cluster}_%j.err" \
        --mail-user=lilliansun@college.harvard.edu \
        --mail-type=END,FAIL \
        --account="$ACC" \
        "$WORKER_AUCTION" "config/q1_dominant_gen.json" "$p" "0" "1"
}

submit_q2_gen_on() {
    local cluster=$1 p=$2 config="${3:-config/q2_numerical_gen.json}" timelim="${4:-0-01:00}"
    IFS='|' read -r ACC PART GRES <<< "$(cluster_config $cluster)"
    sbatch \
        --job-name="q2_p${p}_${cluster}" \
        --partition="$PART" \
        --gres="$GRES" \
        --mem=200gb \
        -t "$timelim" \
        -o "$LOGS/q2_p${p}_${cluster}_%j.out" \
        -e "$LOGS/q2_p${p}_${cluster}_%j.err" \
        --mail-user=lilliansun@college.harvard.edu \
        --mail-type=END,FAIL \
        --account="$ACC" \
        "$WORKER_NUMERICAL" "$config" "$p" "0" "1"
}

submit_align_on() {
    local cluster=$1 tag=$2 config=$3 time_limit=$4
    IFS='|' read -r ACC PART GRES <<< "$(cluster_config $cluster)"
    sbatch \
        --job-name="${tag}_align_${cluster}" \
        --partition="$PART" \
        --gres="$GRES" \
        --mem=200gb \
        -t "$time_limit" \
        -o "$LOGS/${tag}_align_${cluster}_%j.out" \
        -e "$LOGS/${tag}_align_${cluster}_%j.err" \
        --mail-user=lilliansun@college.harvard.edu \
        --mail-type=END,FAIL \
        --account="$ACC" \
        "$ALIGN_WORKER" "$config"
}

submit_q1_gen() {
    submit_q1_gen_on "$CLUSTER" "$1"
    [ "${DUAL:-0}" = "1" ] && submit_q1_gen_on "$(other_cluster $CLUSTER)" "$1"
    return 0
}

submit_q2_gen() {
    submit_q2_gen_on "$CLUSTER" "$1"
    [ "${DUAL:-0}" = "1" ] && submit_q2_gen_on "$(other_cluster $CLUSTER)" "$1"
    return 0
}

submit_q2_gen_pilot_k5() {
    submit_q2_gen_on "$CLUSTER" "$1" "config/q2_numerical_pilot_k5_gen.json" "0-00:30"
    [ "${DUAL:-0}" = "1" ] && submit_q2_gen_on "$(other_cluster $CLUSTER)" "$1" "config/q2_numerical_pilot_k5_gen.json" "0-00:30"
    return 0
}

submit_align() {
    local tag=$1 config=$2 time_limit=$3
    submit_align_on "$CLUSTER" "$tag" "$config" "$time_limit"
    [ "${DUAL:-0}" = "1" ] && submit_align_on "$(other_cluster $CLUSTER)" "$tag" "$config" "$time_limit"
    return 0
}

case "$MODE" in
    pilot)
        echo "=== Phase 0: Pilot (prompt 0 for Q1 and Q2, k=5) ==="
        submit_q1_gen 0
        submit_q2_gen_pilot_k5 0
        echo "  2 generation jobs submitted (Q2 uses k=5 pilot config)"
        echo ""
        echo "After both finish, run: ./submit_rebuttal.sh align_pilot"
        ;;
    q1_full)
        echo "=== Q1 full (prompts 1-4) ==="
        for p in 1 2 3 4; do
            submit_q1_gen $p
        done
        echo "  4 Q1 generation jobs submitted"
        ;;
    q2_full)
        echo "=== Q2 full (prompts 1-4) ==="
        for p in 1 2 3 4; do
            submit_q2_gen $p
        done
        echo "  4 Q2 generation jobs submitted"
        ;;
    full_all)
        echo "=== Q1 + Q2 full (prompts 1-4 each) ==="
        for p in 1 2 3 4; do
            submit_q1_gen $p
            submit_q2_gen $p
        done
        echo "  8 generation jobs submitted"
        ;;
    align)
        echo "=== Alignment for Q1 + Q2 (full k=20 Q2) ==="
        submit_align "q1" "config/q1_dominant_alignment_clip.json" "0-00:30"
        submit_align "q2" "config/q2_numerical_alignment_clip.json" "0-01:00"
        echo "  2 alignment jobs submitted"
        ;;
    align_pilot)
        echo "=== Pilot alignment (Q1 + Q2 k=5 only) ==="
        submit_align "q1" "config/q1_dominant_alignment_clip.json" "0-00:30"
        submit_align "q2_k5" "config/q2_numerical_pilot_k5_alignment_clip.json" "0-00:30"
        echo "  2 pilot alignment jobs submitted"
        ;;
    align_q1)
        submit_align "q1" "config/q1_dominant_alignment_clip.json" "0-00:30"
        ;;
    align_q2)
        submit_align "q2" "config/q2_numerical_alignment_clip.json" "0-01:00"
        ;;
    align_q2_k5)
        submit_align "q2_k5" "config/q2_numerical_pilot_k5_alignment_clip.json" "0-00:30"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {pilot|q1_full|q2_full|full_all|align|align_pilot|align_q1|align_q2|align_q2_k5}"
        exit 1
        ;;
esac

echo ""
echo "Monitor: squeue -u lilliansun"
echo "Logs:    ls -lt $LOGS/q*.out | head"
