# Diffusion Auctions Multi-Agent

**Authors:** Lillian Sun, Warren Zhu, Henry Huang

## Overview

This project implements a novel auction mechanism for diffusion models where multiple agents (2-20) bid to influence image generation. Higher bids result in greater visual representation in the final generated image. The system uses iterative pairwise score composition to blend agent influences proportionally to their bids.

**Key idea:** Traditional auctions have a single winner, but generative AI enables **multi-winner auctions** where multiple bidders can influence the outcome proportionally.

**Example scenario:**
- Base prompt: "Two friends chatting over coffee at a cafe"
- Agent 1 bids 0.6 for "Cappuccino drink"
- Agent 2 bids 0.3 for "Microsoft Surface laptop"
- Agent 3 bids 0.1 for "USM Haller"

Result: The generated image shows friends at a cafe with a prominent cappuccino, a visible laptop, and subtle furniture presence.

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (8-12GB VRAM)
- 16GB+ RAM

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd diffusion_auctions_multiagent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace authentication
echo "HF_TOKEN=your_token_here" > .env
# Or: huggingface-cli login
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

### Generate Images
```bash
cd scripts
python generate_images.py --config ../config/config_3_agents.json
```

Output images are saved to the `output_dir` specified in the config file.

## How It Works

### Iterative Pairwise Score Composition

The auction mechanism works through iterative pairwise composition of noise predictions:

1. **Sort agents** by bid amount (ascending)
2. **Select top two bidders** from the remaining agents
3. **Normalize bids** relative to each other: `norm_bid = bid_high / (bid_high + bid_low)`
4. **Calculate weight**: `w_dom = clamp(2 * norm_bid - 1, 0, 1)`
5. **Compose predictions**: `combined = (1 - w_dom) * noise_shared + w_dom * noise_dominant`
6. **Create composite agent** with summed bids and concatenated prompts
7. **Repeat** until one composite agent remains

### Bidding Examples

#### 2-Agent Scenarios
| Bids | Effect |
|------|--------|
| `[0.0, 0.0]` | Base prompt only (no agent influence) |
| `[1.0, 0.0]` | Agent 1 completely dominates |
| `[0.5, 0.5]` | Equal influence from both agents |
| `[0.7, 0.3]` | Agent 1 has more influence |

#### 3-Agent Scenarios
| Bids | Effect |
|------|--------|
| `[0.0, 0.0, 0.0]` | Base prompt only |
| `[1.0, 0.0, 0.0]` | Agent 1 only |
| `[0.33, 0.33, 0.33]` | All agents equal |
| `[0.6, 0.3, 0.1]` | Clear hierarchy: A1 > A2 > A3 |

The system scales to 5, 10, or 20 agents with the same algorithm.

## Project Structure

```
diffusion_auctions_multiagent/
├── pipelines/
│   ├── flux_auction_pipeline.py      # FluxPipelineAuction (N-agent iterative)
│   └── flux_auction_pipeline_old.py  # Legacy 3-agent recursive version
├── scripts/
│   ├── generate_images.py            # Config-driven image generation
│   ├── alignment_clip.py             # CLIP alignment analysis
│   ├── alignment_pickscore.py        # PickScore alignment analysis
│   ├── calculate_alignment_2_agent.py # 2-agent alignment (with VLM support)
│   ├── calculate_alignment_3_agent.py # 3-agent alignment (with VLM support)
│   ├── quality_laion.py              # LAION aesthetic quality
│   ├── multi_gpu_config.py           # Multi-GPU utilities
│   ├── run_with_cache.sh             # Cache-optimized runner (3-agent)
│   └── run_with_cache_2_agent.sh     # Cache-optimized runner (2-agent)
├── config/                           # New experiment configs go here
├── old_configs/                      # 56 legacy configs (from henry branch)
│   ├── config_{2,3,5,10,20}_agents.json
│   ├── backwards_config_*.json
│   ├── alignment_{clip,pickscore}_config_*.json
│   ├── quality_laion_config_*.json
│   ├── vlm_config.json
│   └── README.md
├── prompts/
│   ├── agent_prompts.json            # Standard prompts (20 agents each)
│   ├── agent_prompts_competitive.json # Competitive variant
│   ├── base_prompts.json             # Base scene descriptions
│   └── base_prompts_competitive.json
├── analysis/                         # Jupyter notebooks for welfare & quality
├── alignment/                        # Alignment analysis results
├── quality/                          # Quality assessment results
├── results/                          # Aggregated charts & data
├── helpers/                          # Cache & utility scripts
├── vlm_quality_assessor.py           # VLM quality module
├── pickscore_predictor.py            # PickScore utility
├── VLM_SETUP.md                      # VLM setup guide
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Configuration System

All scripts are driven by JSON config files. Legacy configs from the henry branch are in `old_configs/`. New experiment configs go in `config/`.

### Config Schema

```json
{
  "num_agents": 3,
  "prompts_path": "prompts/agent_prompts.json",
  "output_dir": "output/images_3_agents",
  "num_samples_per_combination": 20,
  "num_prompts_to_process": null,
  "process_prompts_forward": true,
  "guidance_scale": 10.0,
  "num_inference_steps": 5,
  "bidding_combinations": [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.33, 0.33, 0.33],
    [0.6, 0.3, 0.1]
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `num_agents` | int | Number of agents in the auction |
| `prompts_path` | str | Path to prompts JSON file |
| `output_dir` | str | Output directory for results |
| `num_samples_per_combination` | int | Samples per bid combination |
| `num_prompts_to_process` | int/null | Limit prompts (null = all) |
| `process_prompts_forward` | bool | Process order (true = 0-N, false = N-0) |
| `guidance_scale` | float | Diffusion guidance scale |
| `num_inference_steps` | int | Number of denoising steps |
| `bidding_combinations` | list | List of bid vectors to test |

### Available Configs

| Config | Description |
|--------|-------------|
| `config_2_agents.json` | 2-agent scenarios (8 bid combinations) |
| `config_3_agents.json` | 3-agent scenarios (10 bid combinations) |
| `config_5_agents.json` | 5-agent scenarios |
| `config_10_agents.json` | 10-agent scenarios |
| `config_20_agents.json` | 20-agent scenarios |
| `backwards_config_*.json` | Reverse-order processing variants |

### Creating Custom Configs

1. Copy an existing config file
2. Update `num_agents` to desired count
3. Define `bidding_combinations` as list of bid vectors
4. Each bid vector should have length equal to `num_agents`
5. Always include `[0.0, ..., 0.0]` as a baseline

## Usage Guide

### Image Generation

```bash
cd scripts

# Standard generation with a config
python generate_images.py --config ../config/config_3_agents.json

# The script:
# - Loads the pipeline (FLUX.1-schnell)
# - Iterates over prompts x bid_combinations x samples
# - Saves images to output_dir/prompt_XXX/
# - Writes generation_log.json
# - Skips existing images (safe to resume)
```

### Alignment Analysis

Measures how well generated images align with each agent's prompt.

```bash
cd scripts

# CLIP-based alignment
python alignment_clip.py --config ../config/alignment_clip_config_3_agents.json

# PickScore-based alignment
python alignment_pickscore.py --config ../config/alignment_pickscore_config_3_agents.json
```

Output: Per-image JSON files with alignment scores, quality metrics, and welfare calculations.

### Quality Assessment

```bash
cd scripts

# LAION aesthetic quality (uses ViT-L-14 + aesthetic linear head)
python quality_laion.py --config ../config/quality_laion_config_3_agents.json

# VLM quality (requires Qwen2.5-VL, ~29GB VRAM) — see VLM_SETUP.md
python calculate_alignment_2_agent.py --enable_vlm
python calculate_alignment_3_agent.py --enable_vlm
```

### Analysis Notebooks

Jupyter notebooks in `analysis/` for aggregating results:

| Notebook | Purpose |
|----------|---------|
| `calculate_welfare_clip.ipynb` | CLIP-based welfare analysis across agent counts |
| `calculate_welfare_pickscore.ipynb` | PickScore-based welfare analysis |
| `analyze_quality_laion.ipynb` | LAION aesthetic quality analysis |
| `*_competitive.ipynb` | Competitive bidding scenario analysis |

Results (charts, data.csv) are saved to `results/{N}_agents/`.

### SLURM Job Submission

Example sbatch script for the cluster:

```bash
#!/bin/bash
#SBATCH --job-name=diffusion_auction
#SBATCH --partition=kempner_h100
#SBATCH --gres=gpu:2
#SBATCH --mem=200gb
#SBATCH -t 0-12:00
#SBATCH -o /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/results/%x_%j.out
#SBATCH -e /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/results/%x_%j.err
#SBATCH --mail-user=lilliansun@college.harvard.edu
#SBATCH --mail-type=ALL
#SBATCH --account=kempner_ydu_lab

export HF_HOME=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

# Load HF_TOKEN from .env (gitignored)
set -a; source /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/.env; set +a

module load Mambaforge/23.11.0-fasrc01
conda activate flux

cd /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/scripts
python generate_images.py --config ../config/config_3_agents.json
```

**Available accounts and partitions:**

| Account | Partitions |
|---------|------------|
| `kempner_ydu_lab` (primary) | `kempner_h100`, `kempner` |
| `hlakkaraju_lab` (secondary) | `seas_gpu`, `gpu`, `gpu_h200` |

### Cluster Cache Setup

For faster model downloads on cluster nodes with local SSD:

```bash
cd scripts
./run_with_cache.sh           # 3-agent with cache
./run_with_cache_2_agent.sh   # 2-agent with cache

# Manual cache management
python helpers/manage_cache.py list
python helpers/manage_cache.py usage
python helpers/manage_cache.py clean [model_name]
```

## Prompt Format

### Standard Prompts (`agent_prompts.json`)

Each prompt entry supports up to 20 agents. The script extracts the first N based on `num_agents`:

```json
[
  {
    "base_prompt": "Two friends chatting over coffee at a cafe",
    "agent1_prompt": "Cappuccino drink",
    "agent2_prompt": "Microsoft Surface laptop",
    "agent3_prompt": "USM Haller",
    "agent4_prompt": "Marriott key card",
    ...
    "agent20_prompt": "Nike running shoes"
  }
]
```

### Competitive Prompts (`agent_prompts_competitive.json`)

Same structure but with competitive/overlapping brand scenarios.

### Base Prompts (`base_prompts.json`)

Standalone scene descriptions used for baseline-only generation:

```json
["Two friends chatting over coffee at a cafe", "People enjoying a sunny day at the beach", ...]
```

## Output Format

### Generated Images

```
output/images_3_agents/
├── prompt_000/
│   ├── idx000_b1_0.00_b2_0.00_b3_0.00_s00.png   # Base only
│   ├── idx000_b1_1.00_b2_0.00_b3_0.00_s00.png   # Agent 1 dominant
│   ├── idx000_b1_0.33_b2_0.33_b3_0.33_s00.png   # Equal influence
│   └── ...
├── prompt_001/
│   └── ...
└── generation_log.json
```

Naming: `idx{prompt_idx:03d}_b1_{bid1:.2f}_b2_{bid2:.2f}..._s{sample:02d}.png`

### Generation Log (`generation_log.json`)

```json
[
  {
    "item_index": 0,
    "bids": [0.6, 0.3, 0.1],
    "sample_index": 0,
    "agent_prompts": ["Cappuccino drink", "Microsoft Surface laptop", "USM Haller"],
    "base_prompt": "Two friends chatting over coffee at a cafe",
    "image_path": "output/images_3_agents/prompt_000/idx000_b1_0.60_b2_0.30_b3_0.10_s00.png"
  }
]
```

### Alignment Results (per-image JSON)

```json
{
  "metadata": {
    "prompt_index": 0,
    "bids": [0.6, 0.3, 0.1],
    "sample_index": 0,
    "image_path": "..."
  },
  "alignment_scores": {
    "base_alignment": 0.85,
    "agent1_alignment": 0.92,
    "agent2_alignment": 0.78,
    "agent3_alignment": 0.65
  },
  "quality_assessment": {
    "clip_quality": 0.82
  },
  "welfare_metrics": {
    "weighted_alignment": 0.856,
    "total_welfare": 1.70
  }
}
```

### Result Charts (`results/`)

Aggregated visualizations per agent count:
- `bid_monotonicity_*.png` — Alignment vs bid amount
- `welfare_*_over_bids_*.png` — Welfare metrics across bid combinations
- `welfare_*_over_k_*.png` — Welfare metrics across agent counts
- `laion_quality_*.png` — Aesthetic quality analysis
- `data.csv` — Numerical results

## Evaluation Metrics

| Metric | Script | Description |
|--------|--------|-------------|
| **CLIP Alignment** | `alignment_clip.py` | Cosine similarity between image CLIP embedding and text prompt |
| **PickScore** | `alignment_pickscore.py` | Image preference prediction using PickScore_v1 model |
| **LAION Aesthetics** | `quality_laion.py` | Aesthetic quality score using ViT-L-14 + linear predictor |
| **VLM Quality** | `vlm_quality_assessor.py` | Multi-dimensional quality via Qwen2.5-VL (sharpness, coherence, color, composition, detail) |
| **Welfare** | Analysis notebooks | Weighted alignment and total welfare across agents |

## Troubleshooting

### Common Issues

- **CUDA out of memory**: Reduce `num_inference_steps` in config or use fewer GPUs
- **Path errors**: Scripts expect to be run from `scripts/` directory; configs use relative paths
- **Slow model downloads**: Use `run_with_cache.sh` scripts for cluster environments
- **Missing dependencies**: `pip install -r requirements.txt`
- **VLM loading fails**: Ensure `transformers>=4.49.0`, see `VLM_SETUP.md`

### Performance Notes

- Generation: ~5-10 seconds per image on GPU
- Memory: ~8-12GB VRAM for FLUX.1-schnell
- VLM assessment: ~2-5 seconds per image, ~29GB VRAM
- Storage: ~2-5MB per generated image

### Resuming Interrupted Runs

All generation scripts automatically skip existing images. Safe to re-run after interruption.

## References

- [1] [Auctions with LLM Summaries](https://arxiv.org/abs/2404.08126)
- [2] [Ad Auctions for LLMs via Retrieval Augmented Generation](https://arxiv.org/abs/2406.09459)
- [3] [Mechanism Design for Large Language Models](https://arxiv.org/pdf/2310.10826)
- [4] [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [5] [Classifier-Free Guidance Is a Predictor-Corrector](https://machinelearning.apple.com/research/predictor-corrector)
