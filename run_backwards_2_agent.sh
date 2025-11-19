#!/bin/bash
#SBATCH --job-name=run_backwards_2_agent
#SBATCH --account=kempner_ydu_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/%j_run_backwards_2_agent.out
#SBATCH --error=logs/%j_run_backwards_2_agent.err
#SBATCH --mail-type=END
#SBATCH --mail-user=hhuang@college.harvard.edu

set -e
set -x

module load cuda
source ~/.bashrc
mamba activate diffusion_auctions

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/henhua/diffusion_auctions_multiagent

echo "=== Running generate_images.py script for backwards 2 agents ==="
python scripts/generate_images.py --config config/backwards_config_2_agents.json

echo "=== generate_images.py script for backwards 2 agents completed ==="