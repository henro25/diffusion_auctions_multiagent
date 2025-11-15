#!/bin/bash
#SBATCH --job-name=run_2_agent_image_gen
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j_run_2_agent_image_gen.out
#SBATCH --error=logs/%j_run_2_agent_image_gen.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hhuang@college.harvard.edu

set -e
set -x

module load cuda
source ~/.bashrc
mamba activate diffusion_auctions

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/henhua/diffusion_auctions_multiagent

echo "=== Running generate_images.py script for 2 agents image generation ==="
python scripts/generate_images.py --config config/config_2_agents.json

echo "=== generate_images.py script for 2 agents image generation completed ==="