# 04 — Full Results (all 5 prompts, k=5)

**Status**: ✅ complete (2026-04-25 16:55 PDT).

## What was run
- **Q1**: 5 prompts × 11 bid combos × 5 samples = 275 images (auction pipeline). Submitted as 5 sbatch jobs (`q1_p0=8475925`, `q1_p1=8477515`, `q1_p2=8477517`, `q1_p3=8477519`, `q1_p4=8477521`) on `kempner_h100`. Total wall clock ~25 min from first start to last finish. Real per-image rate: ~25 sec/image.
- **Q2 (numerical baseline)**: 5 prompts × 8 bid combos × 5 samples = 200 images (FluxPipeline). Submitted as 5 sbatch jobs (`q2_p0_k5=8476601`, `q2_p1_k5=8477516`, `q2_p2_k5=8477518`, `q2_p3_k5=8477520`, `q2_p4_k5=8477522`). Total wall clock ~7 min per job. Real per-image rate: ~10 sec/image.
- **Q2 (auction comparison data)**: existing `alignment/alignment_clip_2_agents/prompt_{000..004}/` (5 prompts × 8 combos × 50 samples per combo = 400 alignment files per prompt).

## Q1 results

### Allocation (mean across 5 prompts)

| `b1` | Agent 1 (dominant) | Agent 2 (neutral) |
|---|---|---|
| 0.0 | 0.078 | 0.226 |
| 0.1 | 0.073 | 0.224 |
| 0.2 | 0.110 | 0.228 |
| 0.3 | 0.192 | 0.219 |
| 0.4 | 0.211 | 0.216 |
| 0.5 | 0.220 | 0.218 |
| 0.6 | 0.229 | 0.221 |
| 0.7 | 0.213 | 0.183 |
| 0.8 | 0.230 | **0.126** ← phase transition |
| 0.9 | 0.236 | 0.110 |
| 1.0 | 0.244 | 0.105 |

**Pattern**:
- Agent 1 (dominant prompt) rises 3× from 0.078 to 0.244 across the bid range. Sharpest growth at b1=0.2→0.3 (where the dominant prompt becomes visible).
- Agent 2 (neutral prompt) stays near 0.22 from b1=0 through b1=0.6, then phase-transitions to ~0.11 at b1≥0.8. This is the "dominant takeover" the reviewer asked about.
- **Critically: both agents have positive alignment at every bid combo.** No prompt is ever fully suppressed.

### VCG payments (mean across 5 prompts)

| `b1` | Agent 1 VCG | Agent 2 VCG |
|---|---|---|
| 0.0 | 0.000 | 0.000 |
| 0.2 | 0.003 | **0.027** |
| 0.4 | 0.008 | 0.013 |
| 0.5 | 0.004 | 0.012 |
| 0.7 | 0.013 | **0.021** |
| 0.8 | **0.020** | 0.011 |
| 0.9 | 0.012 | 0.007 |

**Pattern**:
- Agent 1's VCG payment **peaks at b1=0.8 (0.020)** — exactly at the phase transition where the dominant prompt suppresses the neutral one. This is the auction correctly charging Agent 1 for the externality it imposes.
- Agent 2's VCG payment is non-trivial whenever b1 < 1 (Agent 2 is also competing for image space). Peaks at b1=0.2 and b1=0.7.
- **The auction internalizes the negative externality**: dominant low-bid prompts pay a measurable VCG premium when their visual takeover actively hurts the other agent.

### Plots
- `results/q1_dominant/allocation_per_prompt.png` — per-prompt curves (5 panels)
- `results/q1_dominant/allocation_mean.png` — mean across prompts with std bands
- `results/q1_dominant/vcg_payments.png` — per-prompt VCG payments
- `results/q1_dominant/allocation_data.csv` — raw 55-row data
- `results/q1_dominant/vcg_payments.csv` — raw VCG data

## Q2 results

### Welfare (mean across 5 prompts)

| (b1, b2) | Auction | Numerical baseline | Δ% |
|---|---|---|---|
| (0.0, 1.0) | 0.204 | 0.182 | **+12.1%** |
| (0.1, 0.9) | 0.194 | 0.176 | **+10.0%** |
| (0.3, 0.7) | 0.187 | 0.187 | +0.3% |
| (0.5, 0.5) | 0.189 | 0.184 | +2.7% |
| (0.7, 0.3) | 0.185 | 0.170 | **+8.4%** |
| (0.9, 0.1) | 0.188 | 0.166 | **+13.2%** |
| (1.0, 0.0) | 0.197 | 0.177 | **+11.1%** |

**Pattern**: Auction welfare is **higher than numerical baseline at every bid combo**, with an average improvement of ~8% (range +0.3% to +13.2%). The gap is largest at extreme bids (one agent dominant) — exactly where the verbal "X% aligned" instruction fails to actually steer the diffusion model.

### Average welfare improvement summary

**Overall mean: +8.26%** improvement of auction over baseline (averaged across 7 non-zero bid combos).

Per-combo (from `welfare_improvement_summary.csv`):

| (b1, b2) | Auction | Baseline | Improvement |
|---|---|---|---|
| (0.0, 1.0) | 0.2036 | 0.1816 | **+12.07%** |
| (0.1, 0.9) | 0.1938 | 0.1761 | **+10.05%** |
| (0.3, 0.7) | 0.1871 | 0.1865 | +0.32% |
| (0.5, 0.5) | 0.1891 | 0.1842 | +2.68% |
| (0.7, 0.3) | 0.1848 | 0.1704 | **+8.42%** |
| (0.9, 0.1) | 0.1882 | 0.1663 | **+13.16%** |
| (1.0, 0.0) | 0.1967 | 0.1771 | **+11.09%** |

### Per-agent average alignment (proves the auction is more bid-monotone)

Slope of mean alignment vs `b1` (linear regression across all 7 non-zero bid combos):

| Agent | Auction slope | Baseline slope | Auction more responsive? |
|---|---|---|---|
| Agent 1 | **+0.0862** | +0.0471 | yes — 1.83× steeper |
| Agent 2 | **−0.0515** | −0.0069 | yes — 7.5× steeper |

The numerical baseline produces a nearly-flat Agent 2 curve (slope ≈ -0.007) — the diffusion model essentially ignores the verbalized percentage. The auction produces a clear monotone decrease (slope = -0.052).

### Per-prompt comparison
At symmetric (b1=0.5, b2=0.5) for prompt 0:
- Auction: a1=0.165, a2=0.208
- Baseline: a1=0.165, a2=0.177

Both methods produce monotone allocation curves (5/5 prompts for both agents) at the coarse `mean-monotone-in-own-bid` level, but the **auction's slopes are dramatically steeper**, making each $1 of bid translate to more visual representation.

### Plots / files
- `results/q2_numerical_baseline/comparison_per_prompt.png` — auction vs baseline per prompt × per agent (5×2 grid)
- `results/q2_numerical_baseline/welfare_comparison.png` — welfare bar chart per bid combo
- `results/q2_numerical_baseline/welfare_improvement_by_combo.png` — % improvement bar chart (overall mean: +8.26%)
- `results/q2_numerical_baseline/alignment_per_agent_avg.png` — per-agent avg alignment, auction vs baseline (1×2 grid showing the steeper auction slopes)
- `results/q2_numerical_baseline/welfare_improvement_summary.csv` — per-combo improvement %
- `results/q2_numerical_baseline/comparison_data.csv` — raw aggregated data
- `results/q2_numerical_baseline/monotonicity_summary.csv` — pairwise monotonicity check

---

## Rebuttal text snippets

### Q1 (visually dominant prompts)

> The reviewer raises a valid concern that visually dominant prompts could disproportionately suppress higher-bid agents. We empirically tested this with five intentionally-chosen dominant/neutral prompt pairs (Canada Goose parka / MacBook Air, scuba gear / hardcover novel, wedding dress / power drill, snowboarding gear / espresso machine, wetsuit / vintage typewriter) under a linear bid sweep b₁ ∈ {0.0, 0.1, …, 1.0} with b₂ = 1 − b₁ and k=5 samples per combination.
>
> **Both agents always retain positive CLIP alignment**: at every bid combo the dominant prompt never receives less than 0.073 and the neutral prompt never less than 0.105 (mean across 5 prompts), so the auction does not fully suppress either agent. We do observe a sharp phase transition: the neutral agent's mean alignment stays near 0.22 for b₁ ∈ [0.0, 0.6] and drops to ≈ 0.11 once b₁ ≥ 0.8, when the dominant prompt's environmental constraint takes over the scene.
>
> **VCG payments correctly internalize this externality.** The dominant agent's mean VCG payment is near zero for b₁ ∈ [0.0, 0.6] (when its presence does not yet harm the neutral agent) and peaks at 0.020 alignment-units at b₁ = 0.8 — precisely where the visual takeover begins. Symmetrically, the neutral agent pays its largest VCG premium at small b₁, when *its* presence is what eats into the dominant prompt's allocation. See `results/q1_dominant/{allocation_per_prompt,vcg_payments,allocation_mean}.png`.

### Q2 (numerical-prompt baseline)

> We compare our auction against the explicit-percentage baseline the reviewer proposes. For each prompt and bid combo we generate k=5 images with `FluxPipeline` (no auction) using the prompt template `"{base} that is {b₁·100}% aligned with {agent1_prompt} and {b₂·100}% aligned with {agent2_prompt}"`, dropping any 0% clauses. We use the same 5 prompts and 8 bid combinations as our two-agent main experiments, and compare against the existing 50-samples-per-combo auction data.
>
> **The auction beats the numerical baseline on welfare at every bid combo, with a mean improvement of +8.26% across the 7 non-zero combos** (range +0.3% to +13.2%), see `results/q2_numerical_baseline/welfare_improvement_by_combo.png`. The gap is largest at extreme bids (e.g. (b₁=0.9, b₂=0.1) shows +13.2% welfare advantage), where the diffusion model fails to actually condition on the verbal percentage.
>
> **The auction is also dramatically more bid-responsive.** Plotting per-agent mean alignment vs `b₁` averaged across all 5 prompts (`results/q2_numerical_baseline/alignment_per_agent_avg.png`), the auction's slope for Agent 1 is +0.086 (vs +0.047 for the baseline; **1.83× steeper**) and for Agent 2 is −0.052 (vs −0.007 for the baseline; **7.5× steeper**). The numerical baseline's Agent 2 curve is essentially flat — the diffusion model largely ignores the verbalized percentage. Each dollar of bid translates to significantly more visual representation under the auction than under the verbal-percentage prompt.

---

## Source job IDs

| Job | Type | Started → finished |
|---|---|---|
| 8475925 | Q1 p0 generation | 19:00 → 19:18 EDT |
| 8475927 | Q2 p0 generation (k=20, cancelled at 35/160) | 19:00 → 19:08 EDT |
| 8476601 | Q2 p0 k=5 generation | 19:07 → 19:12 EDT |
| 8477507 | Q1 pilot alignment | 19:20 EDT (~20s wall) |
| 8477508 | Q2 k=5 pilot alignment | 19:21 EDT (~30s wall) |
| 8477515-8477522 | Q1 p1-p4 + Q2 p1-p4 generation | 19:38 → 19:51 EDT |
| 8480061 | Q2 full k=5 alignment | 19:50 → 19:53 EDT |
| 8480155 | Q1 full alignment | 19:51 → 19:53 EDT |
