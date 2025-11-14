# Pipeline Update Required

## Summary

The repository has been restructured to support a generalized multi-agent setup with configurable numbers of agents (2, 3, 5, 10, 20, or custom).

## Key Changes

### 1. New Script: `scripts/generate_images.py`

A single generalized script that replaces the agent-specific scripts:
- `scripts/generate_images_2_agent.py`
- `scripts/generate_images_3_agent.py`

### 2. Configuration-Driven

Configs are located in the `config/` directory:
- `config/config_2_agents.json`
- `config/config_3_agents.json`
- `config/config_5_agents.json`
- `config/config_10_agents.json`
- `config/config_20_agents.json`

### 3. Updated Flux Pipeline Call

The `generate_images.py` script calls the pipeline with **generalized list-based parameters** instead of individual agent parameters.

#### Current Approach (Old - Agent-Specific)

```python
images = pipeline(
    agent1_prompt=agent1_prompt,
    agent1_bid=bid1,
    agent2_prompt=agent2_prompt,
    agent2_bid=bid2,
    agent3_prompt=agent3_prompt,
    agent3_bid=bid3,
    base_prompt=base_prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
)
```

#### New Approach (Generalized)

```python
images = pipeline(
    agent_prompts=agent_prompts,      # List of prompts (length = num_agents)
    agent_bids=agent_bids,            # List of bids (length = num_agents)
    base_prompt=base_prompt,
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
)
```

## Implementation Required

Update `pipelines/flux_auction_pipeline.py` (specifically the `__call__` method) to:

1. Accept `agent_prompts` (list) and `agent_bids` (list) as inputs
2. Determine number of agents from the list lengths
3. Process bids and prompts in a generalized loop instead of hardcoded agent1/agent2/agent3

### Example Implementation Outline

```python
def __call__(
    self,
    agent_prompts: list,           # ["prompt1", "prompt2", "prompt3", ...]
    agent_bids: list,              # [0.6, 0.3, 0.1, ...]
    base_prompt: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 20,
    **kwargs
):
    num_agents = len(agent_prompts)

    # Generate noise predictions for each agent
    noise_preds = []
    for i, prompt in enumerate(agent_prompts):
        if agent_bids[i] > 0:  # Only generate if bid > 0
            noise_pred = self.generate_noise_pred(
                prompt,
                guidance_scale,
                num_inference_steps
            )
            noise_preds.append(noise_pred)

    # Generate base prompt noise prediction
    base_noise_pred = self.generate_noise_pred(
        base_prompt,
        guidance_scale,
        num_inference_steps
    )

    # Apply auction mechanism with bids and noise predictions
    # (existing score composition algorithm)
    final_image = self.apply_auction_mechanism(
        noise_predictions=noise_preds,
        bids=agent_bids,
        base_noise_pred=base_noise_pred
    )

    return final_image
```

## Directory Structure

```
diffusion_auctions_multiagent/
├── scripts/
│   ├── generate_images.py              # NEW: Generalized script
│   ├── generate_images_2_agent.py      # (Can keep for backwards compatibility)
│   ├── generate_images_3_agent.py      # (Can keep for backwards compatibility)
│   └── ...
├── config/                             # NEW: Configuration directory
│   ├── README.md                       # Configuration documentation
│   ├── config_2_agents.json
│   ├── config_3_agents.json
│   ├── config_5_agents.json
│   ├── config_10_agents.json
│   └── config_20_agents.json
├── output/                             # NEW: Output directory structure
│   ├── images_2_agents/
│   ├── images_3_agents/
│   ├── images_5_agents/
│   ├── images_10_agents/
│   └── images_20_agents/
├── prompts/
│   ├── agent_prompts.json              # Used for all configs
│   └── ...
├── pipelines/
│   ├── flux_auction_pipeline.py        # NEEDS UPDATE
│   └── ...
└── PIPELINE_UPDATE.md                  # This file
```

## Usage

### Generate images with 3 agents:

```bash
cd scripts
python generate_images.py ../config/config_3_agents.json
```

### Generate images with 10 agents:

```bash
cd scripts
python generate_images.py ../config/config_10_agents.json
```

## Notes

- All configurations use `prompts/agent_prompts.json` which has up to 20 agents
- The script automatically extracts the required agent prompts based on `num_agents`
- Bids don't need to sum to 1.0, but it's recommended for interpretability
- All combinations always include [0.0, 0.0, ...] for base-prompt-only reference
- Output directories are automatically created if they don't exist
