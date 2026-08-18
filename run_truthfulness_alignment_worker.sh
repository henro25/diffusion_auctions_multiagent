#!/bin/bash
# Worker script for truthfulness CLIP alignment jobs.
# Arguments: $1=config_path

PROJECT_DIR=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent
# Resolve config path relative to project root if not absolute
if [[ "$1" == /* ]]; then
    CONFIG="$1"
else
    CONFIG="$PROJECT_DIR/$1"
fi

export HF_HOME=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

set -a; source /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/.env; set +a

module load Mambaforge/23.11.0-fasrc01
conda activate flux

cd /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/scripts
python alignment_clip.py --config "$CONFIG"
