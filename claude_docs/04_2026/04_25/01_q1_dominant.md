# 01 — Q1: Visually Dominant Prompts

## Question being addressed
Reviewer Q: "If a low-bid agent's prompt asks for something that is visually dominant, it might disproportionately decrease the alignment toward other (higher-bid) prompts."

## Hypothesis
1. Our auction still gives non-trivial CLIP alignment to the dominant low-bid agent (prompt does not vanish).
2. VCG payments scale with the negative externality the dominant prompt imposes on the other agent.

## Setup

### Prompts (`prompts/q1_dominant.json`)
| idx | base | agent1 (dominant, scene-forcing) | agent2 (neutral) |
|---|---|---|---|
| 0 | A person posing for a portrait | Canada Goose parka | Apple MacBook Air |
| 1 | A person photographed indoors | Full scuba diving gear | Hardcover novel |
| 2 | A person standing in a room | White wedding dress | DeWalt power drill |
| 3 | A person enjoying a hot drink | Snowboarding gear on a snowy mountain | Espresso machine |
| 4 | A person seated at a desk | Black wetsuit | Vintage typewriter |

Bases are deliberately neutral so the conflict is induced by the agent prompts themselves.

### Pipeline
- `FluxPipelineAuction` (existing auction pipeline) via `scripts/generate_images.py`
- Worker script: `run_truthfulness_worker.sh` (no new worker needed)
- Model: FLUX.1-schnell, guidance 10.0, 5 inference steps

### Bid grid
Linear sweep: `b1 ∈ {0.0, 0.1, 0.2, ..., 1.0}`, `b2 = 1 − b1` → 11 combos.

### Volumes
- 5 prompts × 11 combos × k=5 samples = **275 images**
- ~3 min generation per prompt + ~2 min alignment overall
- Pilot (prompt 0): 55 images, ~3 min gen + ~2 min align

### Configs
- Generation: `config/q1_dominant_gen.json`
- CLIP alignment: `config/q1_dominant_alignment_clip.json`

### Storage paths
- Raw images: `/net/holy-isilon/ifs/rc_labs/ydu_lab/lilliansun/diffusion_auction_images/q1_dominant/prompt_{NNN}/`
- Alignment JSONs: `alignment/q1_dominant_clip/prompt_{NNN}/`
- Plots/CSVs: `results/q1_dominant/`
- Notebook: `analysis/q1_dominant.ipynb`

## SBATCH

```bash
sbatch \
  --job-name="q1_p${p}" \
  --partition=kempner_h100 --gres=gpu:1 --mem=200gb \
  -t 0-00:30 \
  -o "logs/q1_p${p}_%j.out" -e "logs/q1_p${p}_%j.err" \
  --mail-user=lilliansun@college.harvard.edu --mail-type=END,FAIL \
  --account=kempner_ydu_lab \
  run_truthfulness_worker.sh config/q1_dominant_gen.json $p 0 1
```

Submission via `./submit_rebuttal.sh pilot|q1_full`.

## Analysis (notebook — `analysis/q1_dominant.ipynb`)

1. Per-prompt allocation curves: agent1 + agent2 CLIP alignment vs `b1` (with `b2 = 1 − b1`). Both agents should remain > 0 across the bid range.
2. Mean across prompts (with shaded std band).
3. VCG payment vs `b1` for each agent.
4. Externality measure: difference between dominant-pair payment and matched neutral-pair payment (using existing `alignment/alignment_clip_2_agents/` data) at the same bids.
5. Sample image grid: representative samples at `b1 ∈ {0.1, 0.5, 0.9}` for each prompt — visual evidence of the externality.

## Bugs hit / fixes

(populated as we encounter them)

