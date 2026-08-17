# Rebuttal Experiments Plan — 2026-04-25

Two experiments to address reviewer questions:
- **Q1**: Visually dominant low-bid prompts and negative externalities
- **Q2**: Numerical-prompt baseline (verbalize bids as percentages)

All generation and alignment runs as **sbatch jobs** — no interactive `python` calls.
Login shell does not have `torch`/CUDA, only the GPU partitions do.

---

## File / path conventions

| Type | Path |
|---|---|
| Raw images (Q1) | `/net/holy-isilon/ifs/rc_labs/ydu_lab/lilliansun/diffusion_auction_images/q1_dominant/` |
| Raw images (Q2) | `/net/holy-isilon/ifs/rc_labs/ydu_lab/lilliansun/diffusion_auction_images/q2_numerical_baseline/` |
| Configs | `config/q1_*.json`, `config/q2_*.json` |
| Prompts | `prompts/q1_dominant.json`, `prompts/q2_numerical.json` |
| Alignment JSONs | `alignment/q1_dominant_clip/`, `alignment/q2_numerical_clip/` |
| Notebooks | `analysis/q1_dominant.ipynb`, `analysis/q2_numerical_baseline.ipynb` |
| Plots / CSVs | `results/q1_dominant/`, `results/q2_numerical_baseline/` |
| sbatch logs | `logs/q1_*.{out,err}`, `logs/q2_*.{out,err}` |
| Submission script | `submit_rebuttal.sh` (project root) |

---

## SLURM conventions (matches existing `submit_truthfulness.sh`)

Two accounts available:

| Account | Partitions | GRES |
|---|---|---|
| `kempner_ydu_lab` | `kempner_h100` | `gpu:1` |
| `hlakkaraju_lab` | `seas_gpu,gpu,gpu_h200` | `gpu:nvidia_a100-sxm4-80gb:1` |

Standard sbatch flags (per `submit_truthfulness.sh`):
```bash
sbatch \
  --job-name=<name> \
  --partition=<partitions> \
  --gres=<gres> \
  --mem=200gb \
  -t 0-01:00 \                            # bump to 0-02:00 for alignment
  -o "$LOGS/%x_%j.out" \
  -e "$LOGS/%x_%j.err" \
  --mail-user=lilliansun@college.harvard.edu \
  --mail-type=END,FAIL \
  --account=<account> \
  "$WORKER" "$config" [worker args...]
```

Worker preamble (per `run_truthfulness_worker.sh`):
```bash
PROJECT_DIR=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/diffusion_auctions_multiagent
export HF_HOME=/n/holylabs/LABS/ydu_lab/Lab/lilliansun/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/models"
set -a; source "$PROJECT_DIR/.env"; set +a
module load Mambaforge/23.11.0-fasrc01
conda activate flux
cd "$PROJECT_DIR/scripts"
```

Resolve `--config` argument as absolute path or relative-to-project-root inside the worker (same trick as existing scripts).

---

## Q1: Visually Dominant Prompts

### Goal
Show that (a) our auction still gives non-trivial CLIP alignment to the dominant
low-bid agent, and (b) VCG payments scale with the negative externality the
dominant prompt imposes.

### Prompt set — `prompts/q1_dominant.json`

5 pairs where agent1 is environment-forcing and agent2 is neutral:

| idx | base_prompt | agent1 (dominant) | agent2 (neutral) |
|---|---|---|---|
| 0 | A person posing for a portrait | Canada Goose parka | Apple MacBook Air |
| 1 | A person photographed indoors | Full scuba diving gear | Hardcover novel |
| 2 | A person standing in a room | White wedding dress | DeWalt power drill |
| 3 | A person enjoying a hot drink | Snowboarding gear on a snowy mountain | Espresso machine |
| 4 | A person seated at a desk | Black wetsuit | Vintage typewriter |

### Generation config — `config/q1_dominant_gen.json`

```json
{
  "num_agents": 2,
  "prompts_path": "../prompts/q1_dominant.json",
  "output_dir": "/net/holy-isilon/ifs/rc_labs/ydu_lab/lilliansun/diffusion_auction_images/q1_dominant",
  "num_samples_per_combination": 5,
  "num_prompts_to_process": null,
  "process_prompts_forward": true,
  "guidance_scale": 10.0,
  "num_inference_steps": 5,
  "bidding_combinations": [
    [0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6],
    [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.0, 0.0]
  ]
}
```

### Alignment config — `config/q1_dominant_alignment_clip.json`

Same bidding combos, points at `q1_dominant` images. Output to `alignment/q1_dominant_clip/`.

### sbatch jobs

5 prompts × 1 chunk each = **5 generation jobs** + 1 alignment job.

```bash
for p in 0 1 2 3 4; do
  sbatch \
    --job-name="q1_p${p}" \
    --partition=kempner_h100 --gres=gpu:1 --mem=200gb \
    -t 0-00:30 \
    -o "logs/q1_p${p}_%j.out" -e "logs/q1_p${p}_%j.err" \
    --mail-user=lilliansun@college.harvard.edu --mail-type=END,FAIL \
    --account=kempner_ydu_lab \
    run_truthfulness_worker.sh config/q1_dominant_gen.json $p 0 1
done
```

After all 5 finish:
```bash
sbatch \
  --job-name="q1_align" \
  --partition=kempner_h100 --gres=gpu:1 --mem=200gb \
  -t 0-01:00 \
  -o "logs/q1_align_%j.out" -e "logs/q1_align_%j.err" \
  --mail-user=lilliansun@college.harvard.edu --mail-type=END,FAIL \
  --account=kempner_ydu_lab \
  run_truthfulness_alignment_worker.sh config/q1_dominant_alignment_clip.json
```

### Volumes & timing
- 5 prompts × 11 combos × 5 samples = **275 images total**
- ~12 min generation + ~5 min alignment
- **Wall clock ≈ 20–25 min** once jobs start

### Analysis (notebook `analysis/q1_dominant.ipynb`)
1. Per-prompt CLIP alignment plot: `agent1_alignment` and `agent2_alignment` vs `b1` (with `b2 = 1 - b1`). Verify both stay > 0 across the bid range.
2. VCG payment vs `b1` for each agent.
3. Compare VCG payments against neutral-vs-neutral baseline at matched bids (use existing `alignment/alignment_clip_2_agents/` data).
4. Visual grid: pick representative samples at `b1 ∈ {0.1, 0.5, 0.9}` for the externality argument.

---

## Q2: Numerical Prompt Baseline

### Goal
Show that explicitly verbalizing bid-as-percentage in the prompt does **not**
produce bid-monotone allocation, while the auction does. Compares against the
already-existing 2-agent auction alignment data — no auction regeneration needed.

### Prompt set — `prompts/q2_numerical.json`

First 5 prompts from `prompts/agent_prompts.json` + `prompts/base_prompts.json`:

| idx | base | agent1 | agent2 |
|---|---|---|---|
| 0 | Two friends chatting over coffee at a cafe | Cappuccino drink | Microsoft Surface laptop |
| 1 | People enjoying a sunny day at the beach | Spalding basketball | Target logo |
| 2 | A person relaxing on a living-room sofa | Clinique skincare | Hulu series |
| 3 | A hiker resting on a mountain trail | Keen boots | Jura espresso machine |
| 4 | A professional working at a desk in a bright office | Wine glass | Chromecast dongle |

### Prompt template (numerical baseline)

```
"{base_prompt} that is {b1*100:.0f}% aligned with {agent1_prompt}
 and {b2*100:.0f}% aligned with {agent2_prompt}"
```

Edge cases:
- `(0, 0)` → just `base_prompt`
- `b1 = 0` → drop the `b1` clause, keep `b2`
- `b2 = 0` → drop the `b2` clause, keep `b1`

### Bid combos
Match `config_2_agents.json` exactly so we can compare against existing auction alignment:
```
[(0.0, 0.0), (1.0, 0.0), (0.1, 0.9), (0.9, 0.1), (0.0, 1.0),
 (0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]
```

### New code: `scripts/generate_numerical_baseline.py`

Bypasses the auction pipeline. Uses `FluxPipeline` (not `FluxPipelineAuction`) directly.
- Same `--config` argparse interface as `generate_images.py`
- Same `--prompt_index`, `--bid_chunk`, `--num_chunks` for chunked sbatch
- Same output naming: `idx{idx:03d}_b1_{b1:.2f}_b2_{b2:.2f}_s{sample:02d}.png` for compatibility with existing alignment scripts
- Loads FLUX.1-schnell once, reuses for all combos

### Generation config — `config/q2_numerical_gen.json`

```json
{
  "num_agents": 2,
  "prompts_path": "../prompts/q2_numerical.json",
  "output_dir": "/net/holy-isilon/ifs/rc_labs/ydu_lab/lilliansun/diffusion_auction_images/q2_numerical_baseline",
  "num_samples_per_combination": 20,
  "num_prompts_to_process": null,
  "process_prompts_forward": true,
  "guidance_scale": 10.0,
  "num_inference_steps": 5,
  "bidding_combinations": [
    [0.0, 0.0], [1.0, 0.0], [0.1, 0.9], [0.9, 0.1],
    [0.0, 1.0], [0.5, 0.5], [0.7, 0.3], [0.3, 0.7]
  ]
}
```

### Worker script — `run_q2_numerical_worker.sh`

Identical to `run_truthfulness_worker.sh` but calls `generate_numerical_baseline.py` instead of `generate_images.py`.

### sbatch jobs

```bash
for p in 0 1 2 3 4; do
  sbatch \
    --job-name="q2_p${p}" \
    --partition=kempner_h100 --gres=gpu:1 --mem=200gb \
    -t 0-01:00 \
    -o "logs/q2_p${p}_%j.out" -e "logs/q2_p${p}_%j.err" \
    --mail-user=lilliansun@college.harvard.edu --mail-type=END,FAIL \
    --account=kempner_ydu_lab \
    run_q2_numerical_worker.sh config/q2_numerical_gen.json $p 0 1
done
```

Alignment (after all 5 finish):
```bash
sbatch \
  --job-name="q2_align" \
  --partition=kempner_h100 --gres=gpu:1 --mem=200gb \
  -t 0-01:00 \
  -o "logs/q2_align_%j.out" -e "logs/q2_align_%j.err" \
  --mail-user=lilliansun@college.harvard.edu --mail-type=END,FAIL \
  --account=kempner_ydu_lab \
  run_truthfulness_alignment_worker.sh config/q2_numerical_alignment_clip.json
```

### Volumes & timing
- 5 prompts × 8 combos × 20 samples = **800 images total**
- ~30 min generation + ~10 min alignment
- **Wall clock ≈ 45 min** once jobs start

### Analysis (notebook `analysis/q2_numerical_baseline.ipynb`)
1. For each prompt, plot agent alignment vs `b1` for both conditions (auction vs numerical baseline) on the same axes.
2. Welfare comparison at each bid combo.
3. Monotonicity comparison: how often does the numerical baseline produce
   non-monotone allocation curves vs the auction?

Auction comparison data is already in `alignment/alignment_clip_2_agents/prompt_{000..004}/`.

---

## Phased rollout

### Phase 0: Pilot — 1 prompt per question

Before committing to the full 5×2 = 10 prompts, run **prompt 0 only** for both Q1 and Q2 to sanity-check the pipeline, sbatch flags, output paths, and the analysis notebooks.

- **Q1 pilot**: prompt 0 only (Canada Goose / MacBook), 11 combos × 5 samples = **55 images** → ~3 min gen + ~2 min align
- **Q2 pilot**: prompt 0 only (Cappuccino / Surface laptop), 8 combos × 20 samples = **160 images** → ~7 min gen + ~3 min align
- Both run in parallel on separate sbatch jobs
- **Phase 0 wall clock ≈ 15 min** once jobs start
- Pilot analysis: lightweight notebook cells producing one allocation plot per question

User reviews Phase 0 results, confirms direction, then proceed to Phase 1.

### Phase 1: Full 5 prompts per question

Submit the remaining 4 prompts per question (using the same configs, just different `--prompt_index` args).
- Q1 remaining: 4 prompts × 55 images = 220 images → ~10 min gen + ~5 min align
- Q2 remaining: 4 prompts × 160 images = 640 images → ~25 min gen + ~10 min align
- Parallel → **~35 min wall clock**
- Then write/run the full analysis notebooks

### Combined timing

- **Phase 0 (pilot)**: ~15 min wall clock + ~30 min upfront for configs, prompts, `generate_numerical_baseline.py`
- **Phase 1 (full)**: ~35 min wall clock + ~30 min for full notebooks
- **Total: ~110 min** spread across two checkpoints

---

## Submission script: `submit_rebuttal.sh`

Mirror `submit_truthfulness.sh` structure with `pilot` and full modes:
```bash
./submit_rebuttal.sh pilot     # Phase 0: prompt 0 only for Q1 and Q2
./submit_rebuttal.sh q1        # Q1 prompts 1-4 (after pilot is reviewed)
./submit_rebuttal.sh q2        # Q2 prompts 1-4 (after pilot is reviewed)
./submit_rebuttal.sh all       # all prompts for both (skip pilot)
```

Uses `kempner_ydu_lab` for everything (small experiment, no need to spread across accounts).

---

## Order of operations

### Phase 0 (pilot)
1. Write `prompts/q1_dominant.json`, `prompts/q2_numerical.json` (full 5 entries each — pilot just runs index 0)
2. Write `config/q1_*.json`, `config/q2_*.json`
3. Write `scripts/generate_numerical_baseline.py`
4. Write `run_q2_numerical_worker.sh`
5. Write `submit_rebuttal.sh`
6. `./submit_rebuttal.sh pilot` — submits prompt 0 for Q1 and Q2 in parallel
7. Wait ~15 min for jobs to finish
8. Quick sanity-check notebook cells: 1 plot per question
9. **Checkpoint with user — confirm direction before Phase 1**

### Phase 1 (full)
10. `./submit_rebuttal.sh q1 && ./submit_rebuttal.sh q2` (or `all` if not running pilot)
11. Wait ~35 min for all jobs to finish
12. Full notebooks `analysis/q1_dominant.ipynb`, `analysis/q2_numerical_baseline.ipynb`
13. Generate plots into `results/q1_dominant/`, `results/q2_numerical_baseline/`
14. Draft rebuttal text using the figures
