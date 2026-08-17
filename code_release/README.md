# Multi-Agent Diffusion Auctions

Code for running multi-agent auctions over diffusion-model image generation.
N agents (2+) submit bids and text prompts; the mechanism composes their
prompts into a single image whose visual content reflects the bids, via
iterative pairwise score composition inside the denoising loop of
FLUX.1-schnell.

All scripts are config-driven and prompt-agnostic: you supply your own prompts
file and bid combinations, and every path is set in a JSON config or CLI flag.
Nothing is hardcoded.

## Repository layout

```
code_release/
├── pipelines/
│   └── flux_auction_pipeline.py   # FluxPipelineAuction (core mechanism)
├── scripts/
│   ├── generate_images.py         # image generation for any N agents
│   ├── alignment_clip.py          # CLIP alignment scoring
│   ├── alignment_pickscore.py     # PickScore alignment scoring
│   └── quality_laion.py           # LAION aesthetic quality scoring
├── analysis/
│   ├── analyze_welfare.py         # welfare / efficiency vs single-winner baseline
│   ├── analyze_quality.py         # aesthetic quality analysis
│   ├── analyze_myerson_regret.py  # Myerson-payment truthfulness (regret) analysis
│   └── analyze_vcg_regret.py      # Clarke/VCG-payment truthfulness (regret) analysis
├── configs/                       # example configs (edit paths & bids to taste)
├── prompts/                       # example prompt files (replace with your own)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU (8–12 GB VRAM for generation). The first run downloads
FLUX.1-schnell (`black-forest-labs/FLUX.1-schnell`) from Hugging Face.

## Prompt file format

A JSON list. Each entry has a `base_prompt` (the shared scene) and one prompt
per agent (`agent1_prompt`, `agent2_prompt`, ..., `agentN_prompt`):

```json
[
  {
    "base_prompt": "Two friends chatting over coffee at a cafe",
    "agent1_prompt": "Cappuccino drink",
    "agent2_prompt": "Laptop computer"
  }
]
```

Scripts read the first `num_agents` agent prompts from each entry, so a single
prompts file can serve experiments at multiple agent counts. Example files
(the prompt sets used in the paper) are in `prompts/`.

## Config file format

```json
{
  "num_agents": 2,
  "prompts_path": "prompts/example_prompts.json",
  "output_dir": "output/images_2_agents",
  "num_samples_per_combination": 20,
  "num_prompts_to_process": null,
  "process_prompts_forward": true,
  "guidance_scale": 10.0,
  "num_inference_steps": 5,
  "bidding_combinations": [[0.5, 0.5], [0.7, 0.3], ...]
}
```

Alignment/quality configs additionally take `"images_dir"` (where the
generated images live). Relative paths are resolved against the working
directory — run everything from the `code_release/` root, or use absolute
paths in the config.

## Workflow

### 1. Generate images

```bash
python scripts/generate_images.py --config configs/example_generation_2_agents.json
```

Images are written to
`{output_dir}/prompt_{idx:03d}/idx{idx:03d}_b1_{b1:.2f}_b2_{b2:.2f}_s{sample:02d}.png`.
Existing images are skipped, so interrupted runs can simply be restarted.

For parallel execution, shard by prompt and/or bid chunk:

```bash
python scripts/generate_images.py --config CONFIG --prompt_index 0 --bid_chunk 0 --num_chunks 4
```

### 2. Score alignment

```bash
python scripts/alignment_clip.py --config configs/example_alignment_clip_2_agents.json
# and/or
python scripts/alignment_pickscore.py --config configs/example_alignment_pickscore_2_agents.json
```

Writes one JSON per image with per-agent alignment scores and welfare metrics.

### 3. Score aesthetic quality (optional)

```bash
python scripts/quality_laion.py --config configs/example_quality_laion_2_agents.json
```

### 4. Analyze

Welfare / efficiency / bid monotonicity:

```bash
python analysis/analyze_welfare.py \
  --alignment_dir alignment/alignment_clip_2_agents \
  --config configs/example_alignment_clip_2_agents.json \
  --output_dir results/welfare_2_agents
```

Aesthetic quality:

```bash
python analysis/analyze_quality.py \
  --quality_dir quality/quality_laion_2_agents \
  --config configs/example_quality_laion_2_agents.json \
  --output_dir results/quality_2_agents
```

Truthfulness (regret) — requires the fine-grained bid sweep produced by
`configs/example_generation_truthfulness.json` (one agent's bid sweeps
[0, 1] in 0.05 steps while the opponent is fixed at 0.3 / 0.5 / 0.7, k=25
samples per combination):

```bash
python analysis/analyze_myerson_regret.py \
  --alignment_dir alignment/alignment_clip_truthfulness \
  --prompts prompts/example_prompts_truthfulness.json \
  --output_dir results/myerson_regret

python analysis/analyze_vcg_regret.py \
  --alignment_dir alignment/alignment_clip_truthfulness \
  --prompts prompts/example_prompts_truthfulness.json \
  --output_dir results/vcg_regret
```

Both regret analyses estimate the *expected* (ex-ante) allocation by splitting
the k samples per bid into `--num_batches` batches of `--batch_size`, taking
the welfare-optimal image within each batch, and averaging alignments across
batch winners (defaults: 5 batches of 5, i.e. k=25).

## Mechanism summary

At every denoising step, agents are sorted by bid ascending and combined
pairwise from the top: the two highest bidders' bids are normalized within the
pair (`norm = b_top / (b_top + b_second)`), a dominance weight
`w_dom = clamp(2*norm - 1, 0, 1)` is computed, and the pair's noise
predictions are blended as
`(1 - w_dom) * noise(joint prompt) + w_dom * noise(top prompt)`. The pair is
replaced by a composite agent (summed bids, concatenated prompts) and the
process repeats until one agent remains. See
`pipelines/flux_auction_pipeline.py` for details.

Payments can then be computed from the resulting empirical allocation curves
using either Myerson's lemma (`analyze_myerson_regret.py`) or a Clarke/VCG
externality rule (`analyze_vcg_regret.py`).
