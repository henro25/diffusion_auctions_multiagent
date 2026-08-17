# Rebuttal Experiments — Running Log

**Topic**: Q1 (visually dominant prompts) + Q2 (numerical-prompt baseline) for ICML rebuttal.
**Plan**: see `00_plan.md`
**Detail files**: `01_q1_dominant.md` Q1 setup • `02_q2_numerical_baseline.md` Q2 setup • `03_pilot_results.md` pilot • `04_full_results.md` full

---

## Status snapshot — ✅ ALL COMPLETE (2026-04-25 16:55 PDT)

| Phase | Status | Jobs | Notes |
|---|---|---|---|
| Code/configs/scripts | ✅ done | — | All files in place, see `00_plan.md` |
| Phase 0 — pilot generation | ✅ done | `q1_p0=8475925`, `q2_p0_k5=8476601` | k=5 for both |
| Phase 0 — pilot alignment | ✅ done | `8477507`, `8477508` | 95 alignment files |
| Phase 0 — pilot notebook | ✅ done | — | See `03_pilot_results.md` |
| Phase 1 — full generation | ✅ done | `8477515-8477522` (8 jobs) | All 5 prompts × Q1 (k=5, 11 combos) + Q2 (k=5, 8 combos) |
| Phase 1 — full alignment | ✅ done | `8480061`, `8480155` | 275 + 200 = 475 alignment files |
| Phase 1 — full notebooks | ✅ done | (executed by 2 agents in parallel) | See `04_full_results.md` |
| Rebuttal text | ✅ drafted | — | Snippets in `04_full_results.md` |

---

## Pointers to detail files
- `00_plan.md` — full experimental plan (prompts, configs, sbatch templates, timings)
- `01_q1_dominant.md` — Q1 setup details + results (created when pilot runs)
- `02_q2_numerical_baseline.md` — Q2 setup details + results (created when pilot runs)
- `03_pilot_results.md` — pilot results, sanity-check plots
- `04_full_results.md` — full results across all 5 prompts

---

## Timeline (Pacific Time)

### 2026-04-25 15:56 PDT — kickoff
- Pilot generation submitted: `q1_p0=8475925`, `q2_p0=8475927`
- Both jobs PENDING on `kempner_h100` (Priority queue)
- Doc structure migrated to `claude_docs/04_2026/04_25/`
- Cron monitor scheduled (every ~30 min)

### 2026-04-25 16:01 PDT — pilots running, hlak duplicates submitted then cancelled
- `q1_p0=8475925` started running at ~16:00 PDT on `holygpu8a15202`
- `q2_p0=8475927` PENDING on Resources; submitted hlak duplicates `q1_p0_hlak=8476119`, `q2_p0_hlak=8476120`
- User cancelled hlak duplicates once kempner started running both. `q2_p0=8475927` started running on `holygpu8a11601`.

### 2026-04-25 16:06 PDT — switched Q2 pilot to k=5
- Cancelled `q2_p0=8475927` after generating 35/160 images (combos 0-1).
- Created `config/q2_numerical_pilot_k5_gen.json` (k=5 samples, 8 combos = 40 images) for faster pilot.
- Created `config/q2_numerical_pilot_k5_alignment_clip.json` for matching alignment.
- Submitted `q2_p0_k5=8476601`. Existing 35 images will be re-used (skip-existing logic).
- Updated `submit_rebuttal.sh`: `pilot` mode now uses k=5 for Q2; added `align_pilot`, `align_q2_k5` modes.
- `q1_p0` still running (~10 min so far, expected ~9 min for 55 images).

### 2026-04-25 16:12 PDT — Q2 pilot done
- `q2_p0_k5=8476601` finished. 30 newly generated images + 10 from cancelled run = 40 unique samples (s0-s4 of all 8 combos). Real Q2 rate: ~10s/image (FluxPipeline).

### 2026-04-25 16:18 PDT — Q1 pilot done
- `q1_p0=8475925` finished. 55/55 images, ~23 min wall clock. Real Q1 rate: ~25s/image (auction pipeline; slower because of pairwise score composition).

### 2026-04-25 16:23 PDT — pilot alignment + notebooks DONE, results look great
- `q1_align=8477507` finished at 16:20 (55 alignment files, ran in <1 min after job started).
- `q2_k5_align=8477508` finished at 16:21 (40 alignment files).
- Both notebooks executed successfully via `jupyter nbconvert --execute`.
- **Q1 finding**: Canada Goose alignment grows 0.08→0.25 monotonically; MacBook stays ~0.23 from b1=0 to 0.7, then phase-transitions to 0.11 at b1≥0.8. VCG payment for Agent 1 jumps to 0.021 at b1=0.8 — auction correctly captures the externality.
- **Q2 finding**: Auction has steeper, more bid-responsive allocation than the numerical baseline (which is nearly flat).
- See `03_pilot_results.md` for full analysis.

### 2026-04-25 17:18 PDT — Wrote `05_rebuttal.md`
- Drafted full rebuttal text answering R2 Q1 + R2 Q2 using the user's outline.
- Q1 response: 3-part structure — phenomenon is real (phase transition), score composition prevents complete suppression (both agents always > 0 alignment), VCG payment peaks at phase transition (mechanism charges for externality).
- Q2 response: 3-part structure — verbal-percentage baseline has near-flat slope for Agent 2 (7.5× less responsive than auction), auction beats baseline on welfare at every bid combo (+8.26% mean), linear aggregation fails per [Liu et al. 2022; Du et al. 2023].
- Includes a pointer table mapping each claim to specific result files for the reviewer.

### 2026-04-25 17:10 PDT — Q2 expanded with welfare improvement summary + bid-monotonicity plots
- User asked for: (a) avg welfare improvement of auction over baseline, (b) avg per-agent alignment showing better bid monotonicity.
- Launched 2 sub-agents in parallel:
  - Agent 1: Added 2 cells to `analysis/q2_numerical_baseline.ipynb` (welfare improvement summary, per-agent avg alignment), then re-executed.
  - Agent 2: Re-executed `analysis/q1_dominant.ipynb` to refresh.
- Both completed cleanly. New files:
  - `results/q2_numerical_baseline/welfare_improvement_summary.csv`
  - `results/q2_numerical_baseline/welfare_improvement_by_combo.png`
  - `results/q2_numerical_baseline/alignment_per_agent_avg.png`
- Key new numbers: **mean welfare improvement +8.26%**. Auction slopes vs baseline slopes: Agent 1 (+0.086 vs +0.047, 1.83× steeper), Agent 2 (−0.052 vs −0.007, **7.5× steeper**).
- Updated `04_full_results.md` with new tables and a strengthened Q2 rebuttal paragraph.

### 2026-04-25 16:55 PDT — FULL RESULTS COMPLETE
- Both notebook executions finished (Q1 by sub-agent abc54..., Q2 by sub-agent af534...).
- All 5 prompts × both questions aligned, plotted, exported as CSVs.
- Wrote up `04_full_results.md` with key numbers and rebuttal text snippets.
- **Q1 headline**: Both agents always positive-aligned. Phase transition at b1≈0.7→0.8 where neutral prompt drops from ~0.22 to ~0.11. VCG payment for the dominant agent peaks at b1=0.8 (0.020 alignment-units), correctly reflecting the externality.
- **Q2 headline**: Auction beats numerical baseline on welfare at EVERY bid combo. Average +8% improvement, max +13.2% at (0.9, 0.1).
- See `04_full_results.md` for detailed tables and ready-to-paste rebuttal paragraphs.

### 2026-04-25 16:54 PDT — Phase 1 alignment done; agents executing notebooks
- Q1 alignment job `8480155` and Q2 alignment job `8480061` both finished at 19:53 EDT (16:53 PDT).
- **Phase 1 fully aligned**: Q1=275 alignment files, Q2=200 alignment files (all 5 prompts each).
- Launched 2 sub-agents in parallel to execute `analysis/q1_dominant.ipynb` and `analysis/q2_numerical_baseline.ipynb` with full data (avoids burning main context).
- Created `04_full_results.md` placeholder.

### 2026-04-25 16:50 PDT — cron check-in: Phase 1 mostly done
- Q1 generation: p0=55/55, p1=55/55, p2=53/55, p3=53/55, p4=51/55. 3 jobs (`q1_p2`, `q1_p3`, `q1_p4`) still running, near complete.
- Q2 generation: ALL 5 prompts done (40 images each).
- Submitted Q2 full alignment job `8480061` to process all 5 prompts.
- Waiting for Q1 gen to finish (~2-3 more min) before submitting Q1 alignment.

### 2026-04-25 16:21 PDT — Phase 1 (k=5) submitted
- Pilot alignment: `q1_align=8477507`, `q2_k5_align=8477508`.
- User decision: scale up to all 5 prompts BUT with **k=5 across the board** for Q2 (instead of k=20). Faster initial results.
- Submitted prompts 1-4 for both questions:
  - Q1: `q1_p1=8477515`, `q1_p2=8477517`, `q1_p3=8477519`, `q1_p4=8477521`
  - Q2 (k=5): `q2_p1_k5=8477516`, `q2_p2_k5=8477518`, `q2_p3_k5=8477520`, `q2_p4_k5=8477522`
- All 10 jobs PENDING on Priority. Estimated wall clock for whole batch: ~30-45 min once jobs start.

