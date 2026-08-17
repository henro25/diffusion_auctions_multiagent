# 03 — Pilot Results (prompt 0 of each, k=5)

**Status**: ✅ complete (2026-04-25 19:23 PDT / 16:23 PT).

## What was run
- Q1 prompt 0: Canada Goose parka / Apple MacBook Air, base = "A person posing for a portrait". 11 bid combos × k=5 samples = 55 images (job `q1_p0=8475925`, ~23 min wall clock).
- Q2 prompt 0: Cappuccino drink / Microsoft Surface laptop, base = "Two friends chatting over coffee at a cafe". 8 bid combos × k=5 samples = 40 images for the numerical baseline (job `q2_p0_k5=8476601`, ~5 min wall clock). Comparison: existing auction data at `alignment/alignment_clip_2_agents/prompt_000/` (50 samples per combo).

## Key findings

### Q1: Visually dominant prompts produce a clear phase transition

`results/q1_dominant/allocation_per_prompt.png`:
- **Canada Goose alignment** rises smoothly from 0.080 (b1=0.0) to 0.255 (b1=1.0).
- **MacBook alignment** stays nearly flat at 0.22–0.24 from b1=0.0 through b1=0.7, then **drops sharply to 0.11 at b1=0.8** and stays there.
- Both agents have **non-trivial positive alignment across the bid range** — supports hypothesis 1.
- The "phase transition" at b1≈0.7→0.8 is the visual externality kicking in: the parka takes over the scene and pushes the laptop out.

`results/q1_dominant/vcg_payments.png`:
- Agent 1's VCG payment (negative externality on Agent 2) stays near 0 for b1∈[0.0, 0.7], then **jumps to 0.021 at b1=0.8** and decreases to 0 at b1=1.0 (because at b1=1.0, b2=0 so payment is multiplied by 0).
- Agent 2's VCG payment scales with b1 throughout (Agent 2 imposes externality on Agent 1 at all non-zero b1).
- **The auction correctly captures the externality**: Agent 1 only pays when its dominance is actually visible.

### Q2: Numerical baseline produces flatter, less responsive allocation

`results/q2_numerical_baseline/comparison_per_prompt.png` (for prompt 0; prompts 1-4 numerical baseline data still being generated):
- **Auction (Agent 1, cappuccino)**: 0.148 → 0.205 (slope ≈ +0.057)
- **Numerical baseline (Agent 1, cappuccino)**: 0.157 → 0.204 (slope ≈ +0.047, but mostly flat)
- **Auction (Agent 2, laptop)**: 0.20 → 0.13 (slope ≈ -0.07, monotone decrease)
- **Numerical baseline (Agent 2, laptop)**: 0.17 → 0.12 (less clear pattern, near-flat with noise)
- The auction has **steeper, more bid-responsive** allocation curves.

## Files produced

| Type | Path |
|---|---|
| Q1 per-prompt allocation | `results/q1_dominant/allocation_per_prompt.png` |
| Q1 mean allocation | `results/q1_dominant/allocation_mean.png` |
| Q1 VCG payments | `results/q1_dominant/vcg_payments.png` |
| Q1 allocation CSV | `results/q1_dominant/allocation_data.csv` |
| Q1 VCG CSV | `results/q1_dominant/vcg_payments.csv` |
| Q2 per-prompt comparison | `results/q2_numerical_baseline/comparison_per_prompt.png` |
| Q2 welfare comparison | `results/q2_numerical_baseline/welfare_comparison.png` |
| Q2 monotonicity summary | `results/q2_numerical_baseline/monotonicity_summary.csv` |
| Q2 comparison CSV | `results/q2_numerical_baseline/comparison_data.csv` |

## Sanity-check observations
- Both notebooks executed without errors.
- All values in plots/CSVs are derived from JSON files on disk — nothing hardcoded.
- The phase-transition pattern in Q1 (parka takeover at b1≈0.8) is exactly what the reviewer was concerned about: the dominant prompt CAN suppress the other agent. Critical: the AUCTION'S VCG PAYMENT correctly responds to this by charging Agent 1 more.

## Decision: continue to Phase 1
✅ Yes. Phase 1 jobs (prompts 1-4 for both Q1 and Q2 with k=5) already submitted at 16:21 PDT. Once those finish, re-execute notebooks to get full 5-prompt analysis.
