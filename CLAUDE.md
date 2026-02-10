# CLAUDE.md

## Project Overview

**Diffusion Auctions Multi-Agent** — Multi-agent auction mechanism for diffusion models where N agents (2-20) bid to influence image generation. Higher bids result in greater visual representation in the final image.

**Authors:** Lillian Sun, Warren Zhu, Henry Huang
**Context:** 4th year research project on multi-winner auctions for generative AI

## Architecture & Algorithm

### Pipeline

`FluxPipelineAuction` extends `FluxPipeline` (from HuggingFace diffusers) in `pipelines/flux_auction_pipeline.py`. Model: FLUX.1-schnell (`black-forest-labs/FLUX.1-schnell`).

### Iterative Pairwise Score Composition

The core algorithm in `_apply_iterative_score_composition` (line 506):

1. Agents sorted ascending by bid (`_organize_and_sort_agents`)
2. Pop two highest bidders (random tie-breaking via `_select_two_highest_bidders`)
3. Pairwise normalize: `norm_bid_n = bid_n / (bid_n + bid_n_minus_1)`
4. Dominant weight: `w_dom = clamp(2 * norm_bid_n - 1, 0.0, 1.0)`
5. Compose noise: `combined = (1 - w_dom) * noise_shared + w_dom * noise_dom`
6. Create composite agent with summed bids and concatenated prompts, add back, re-sort
7. Repeat until one agent remains

### Prompt Encoding

- Individual agent: `"{base_prompt} with {agent_prompt}"`
- Combined pair: `"{base_prompt} with {agentA_prompt} and {agentB_prompt}"`
- All-zero bids: encodes only `base_prompt`

### Edge Cases

- All bids zero (`total_bid < 1e-9`): returns base-prompt-only noise prediction
- Both bids in pair near-zero (`bid_sum < 1e-9`): skips composition, re-adds agent
- Tied highest bids: random selection among tied agents
- Mismatched prompt/bid lengths: raises `ValueError`

### Key Methods

| Method | Purpose |
|--------|---------|
| `__call__` | Main entry point (line 88) |
| `_apply_iterative_score_composition` | Core auction algorithm (line 506) |
| `_select_two_highest_bidders` | Picks top 2 with tie-breaking (line 370) |
| `_organize_and_sort_agents` | Sorts agents ascending by bid (line 396) |
| `_denoising_loop_with_score_composition` | Runs denoising timesteps (line 430) |
| `_get_noise_prediction` | Transformer inference wrapper (line 685) |

## Project Structure

```
diffusion_auctions_multiagent/
├── pipelines/
│   ├── flux_auction_pipeline.py      # FluxPipelineAuction (current, N-agent iterative)
│   └── flux_auction_pipeline_old.py  # Legacy 3-agent recursive version
├── scripts/
│   ├── generate_images.py            # Config-driven image generation (any N agents)
│   ├── alignment_clip.py             # CLIP alignment analysis (config-driven)
│   ├── alignment_pickscore.py        # PickScore alignment analysis (config-driven)
│   ├── calculate_alignment_2_agent.py # 2-agent alignment (hardcoded paths, VLM support)
│   ├── calculate_alignment_3_agent.py # 3-agent alignment (hardcoded paths, VLM support)
│   ├── quality_laion.py              # LAION aesthetic quality (config-driven)
│   ├── multi_gpu_config.py           # MultiGPUManager class
│   ├── run_with_cache.sh             # Cache-optimized runner (3-agent)
│   └── run_with_cache_2_agent.sh     # Cache-optimized runner (2-agent)
├── config/                           # New configs go here
├── old_configs/                      # 56 legacy JSON configs (from henry branch)
│   ├── config_{2,3,5,10,20}_agents.json        # Generation configs
│   ├── backwards_config_*.json                  # Reverse-order processing
│   ├── alignment_{clip,pickscore}_config_*.json # Alignment analysis configs
│   ├── quality_laion_config_*.json              # Quality assessment configs
│   ├── vlm_config.json                          # VLM model configuration
│   └── README.md                                # Config schema docs
├── prompts/
│   ├── agent_prompts.json            # 50+ prompts, 20 agents each
│   ├── agent_prompts_competitive.json
│   ├── base_prompts.json             # 50+ base scene descriptions
│   └── base_prompts_competitive.json
├── analysis/                         # Jupyter notebooks
│   ├── calculate_welfare_clip.ipynb
│   ├── calculate_welfare_pickscore.ipynb
│   ├── analyze_quality_laion.ipynb
│   └── *_competitive.ipynb, *_old.ipynb variants
├── alignment/                        # CLIP & PickScore alignment results
├── quality/                          # LAION aesthetic quality results
├── results/                          # Aggregated charts & data.csv
│   └── {2,3,5,10,20}_agents/        # Per-agent-count visualizations
├── helpers/
│   ├── manage_cache.py               # HF cache management
│   ├── setup_cache.sh                # Cache initialization
│   ├── zip_*.sh                      # Result compression utilities
│   └── CLUSTER_SETUP.md
├── vlm_quality_assessor.py           # VLM quality assessment module (Qwen2.5-VL)
├── pickscore_predictor.py            # PickScore utility
├── VLM_SETUP.md
├── requirements.txt
├── README.md
└── CLAUDE.md
```

## Configuration System

All main scripts accept `--config path/to/config.json`. Config schema:

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
  "bidding_combinations": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.33, 0.33, 0.33]]
}
```

Legacy configs are in `old_configs/` (from henry branch). New experiment configs go in `config/`.

Config categories:
- **Generation**: `config_{N}_agents.json` — image generation for N agents
- **Backward**: `backwards_config_*.json` — process prompts in reverse order
- **Alignment**: `alignment_{clip,pickscore}_config_*.json` — CLIP/PickScore analysis
- **Quality**: `quality_laion_config_*.json` — LAION aesthetic scoring
- **VLM**: `vlm_config.json` — Vision Language Model settings

**Image storage**: New experiments should store images at `/net/holy-isilon/ifs/rc_labs/ydu_lab/lilliansun/diffusion_auction_images/` (not local `output/`).

## Key Commands

```bash
# Image generation (config-driven)
cd scripts && python generate_images.py --config ../config/config_3_agents.json

# CLIP alignment analysis
cd scripts && python alignment_clip.py --config ../config/alignment_clip_config_3_agents.json

# PickScore alignment analysis
cd scripts && python alignment_pickscore.py --config ../config/alignment_pickscore_config_3_agents.json

# LAION quality assessment
cd scripts && python quality_laion.py --config ../config/quality_laion_config_3_agents.json

# Legacy 2/3-agent alignment (hardcoded paths, optional VLM)
cd scripts && python calculate_alignment_2_agent.py [--enable_vlm] [--prompt_index N]
cd scripts && python calculate_alignment_3_agent.py [--enable_vlm]

# Cache management
python helpers/manage_cache.py list
python helpers/manage_cache.py usage
python helpers/manage_cache.py clean [model_name]
```

## SLURM / Cluster

No sbatch scripts exist in the repo yet. Cluster details for job scripts:

| Setting | Primary | Secondary |
|---------|---------|-----------|
| Account | `kempner_ydu_lab` | `hlakkaraju_lab` |
| Partitions | `kempner_h100`, `kempner` | `seas_gpu`, `gpu`, `gpu_h200` |

Typical resources: `--gres=gpu:2 --mem=200gb -t 0-12:00`
Mail notifications: `lilliansun@college.harvard.edu`
Log directory: project-level results directory

### Job Environment Setup

Every sbatch script should include this preamble before running any Python:

```bash
export HF_HOME=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"

# Load HF_TOKEN from .env (gitignored)
set -a; source /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/.env; set +a

module load Mambaforge/23.11.0-fasrc01
conda activate flux

cd /n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent/scripts
```

## Data Formats

### Prompts (`agent_prompts.json`)
```json
{
  "base_prompt": "Two friends chatting over coffee at a cafe",
  "agent1_prompt": "Cappuccino drink",
  "agent2_prompt": "Microsoft Surface laptop",
  ...
  "agent20_prompt": "Nike running shoes"
}
```
Script extracts first N agent prompts based on `num_agents` in config.

### Output Image Naming
`idx{idx:03d}_b1_{bid1:.2f}_b2_{bid2:.2f}..._s{sample:02d}.png`
Saved to `{output_dir}/prompt_{idx:03d}/`

### Alignment Output (JSON per image)
```json
{
  "metadata": {"prompt_index": 0, "bids": [0.6, 0.3, 0.1], "sample_index": 0, "image_path": "..."},
  "alignment_scores": {"base_alignment": 0.85, "agent1_alignment": 0.92, ...},
  "quality_assessment": {"clip_quality": 0.82},
  "welfare_metrics": {"weighted_alignment": 0.856, "total_welfare": 1.70}
}
```

### Generation Log (`generation_log.json`)
```json
{"item_index": 0, "bids": [0.6, 0.3, 0.1], "sample_index": 0, "agent_prompts": [...], "base_prompt": "...", "image_path": "..."}
```

## Dependencies

Core: `torch>=2.4.0`, `diffusers>=0.30.0`, `transformers>=4.49.0`, `accelerate>=0.24.0`
Evaluation: `open-clip-torch`, `qwen-vl-utils[decord]`, `flash-attn>=2.0.0`, `einops`, `timm`
Supporting: `numpy`, `tqdm`, `matplotlib`, `pillow`, `seaborn`, `sentencepiece`

Hardware: CUDA GPU required. 8-12GB VRAM for generation, ~29GB for VLM assessment.

## Git

- **Current branch**: `lillian` (copied from `henry`)
- **Main branch**: `main`
- **Remote branches**: `origin/main`, `origin/henry`, `origin/gdaras`
