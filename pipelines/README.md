# Pipelines Module

This module contains the core diffusion pipeline implementations for the multi-agent auction system.

## Files

### `flux_auction_pipeline.py`
Contains the `FluxPipelineAuction` class - the main implementation of the multi-agent auction mechanism for FLUX diffusion models.

**Key Features:**
- Multi-agent prompt combination (up to 3 agents)
- Bid-based influence weighting
- Recursive score composition algorithm
- Base prompt support for scene context

**Core Algorithm:**
1. Sort agents by bid amount (descending)
2. Calculate dominance weights using bid ratios
3. Apply recursive weighted interpolation to combine noise predictions

### `__init__.py`
Module initialization file that exports the `FluxPipelineAuction` class for easy importing.

## Usage

```python
from pipelines import FluxPipelineAuction

# Load pipeline
pipeline = FluxPipelineAuction.from_pretrained("black-forest-labs/FLUX.1-schnell")

# Generate image with 3 agents
image = pipeline(
    agent1_prompt="Starbucks coffee mug",
    agent1_bid=0.6,
    agent2_prompt="Apple MacBook Air",
    agent2_bid=0.3,
    agent3_prompt="New York Times newspaper",
    agent3_bid=0.1,
    base_prompt="Two friends chatting at a cafe"
)
```

## Research Context

This implementation is part of a 4th year research project on multi-winner auctions for generative AI, exploring how multiple agents can compete for influence in image generation through bidding mechanisms.

**Authors:** Lillian Sun, Warren Zhu, Henry Huang