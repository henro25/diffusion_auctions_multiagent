#!/bin/bash
# Worker script for truthfulness generation jobs.
# Arguments: $1=config_path $2=prompt_index $3=bid_chunk $4=num_chunks

PROJECT_DIR=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent
# Resolve config path relative to project root if not absolute
if [[ "$1" == /* ]]; then
    CONFIG="$1"
else
    CONFIG="$PROJECT_DIR/$1"
fi
PROMPT_INDEX=$2
BID_CHUNK=$3
NUM_CHUNKS=$4

export HF_HOME=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

set -a; source /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/.env; set +a

module load Mambaforge/23.11.0-fasrc01
conda activate flux

cd /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/scripts
python generate_images.py --config "$CONFIG" --prompt_index "$PROMPT_INDEX" --bid_chunk "$BID_CHUNK" --num_chunks "$NUM_CHUNKS"
