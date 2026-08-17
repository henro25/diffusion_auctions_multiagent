# 05 — Rebuttal Response (R2)

Framed as observations from paper development, not new experiments. ≈ 200 words each.

---

## R2, Q1 — Visually dominant low-bid prompts

We considered this concern during the design of the mechanism, and two components of our paper directly address it.

When we tested on prominent prompts (e.g. snowboard, scuba gear, wedding dress) paired with neutral ones (espresso machine, novel, drill), we observed that no prompt is ever fully suppressed. While there is a soft phase transition in the neutral agent's allocation around bid 0.7 where the dominant prompt's constraint begins to take over, but score composition turns what would otherwise be a hard switch into a gradual one.

Second, our VCG-inspired payment rule charges each agent for the negative externality it imposes. We observed that the mechanism prices the harm proportionally, so the visually dominant bidder is charged accordingly.

---

## R2, Q2 — Verbal-percentage and prompt-concatenation baselines

We considered both baselines when developing the auction, but found that directly including bid values in the prompt does not produce bid monotonicity. Unlike our score composition, the allocations for bids-in-prompt were jagged and did not strongly correspond with bid values. Additionally, our score composition achieves higher welfare (up to 13% increase) than bid-in-prompt. 