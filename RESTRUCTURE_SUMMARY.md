# Repository Restructuring Summary

## Overview

The repository has been restructured to support a **generalized, configuration-driven** multi-agent image generation pipeline. This replaces the hardcoded 2-agent and 3-agent scripts with a single flexible script that works for any number of agents.

## What Was Created

### 1. Generalized Script: `scripts/generate_images.py`

A single, reusable script that:
- Accepts a configuration JSON file as input
- Works with any number of agents (2, 3, 5, 10, 20, or more)
- Dynamically extracts agent prompts and bids from config
- Passes lists to the Flux pipeline instead of individual parameters

**Usage:**
```bash
cd scripts
python generate_images.py ../config/config_3_agents.json
```

### 2. Configuration Files: `config/config_N_agents.json`

Five pre-configured files for common use cases:

#### `config_2_agents.json`
- 2 agents
- 6 bidding combinations
- Includes: base-only, single dominance, equal split, asymmetric bids

#### `config_3_agents.json`
- 3 agents
- 10 bidding combinations
- Includes: base-only, single/dual dominance, equal, hierarchical bids

#### `config_5_agents.json`
- 5 agents
- 8 bidding combinations
- Reasonable distributions with decreasing influence

#### `config_10_agents.json`
- 10 agents
- 7 bidding combinations
- Scaled distributions for larger auctions

#### `config_20_agents.json`
- 20 agents
- 6 bidding combinations
- Patterns for maximum scalability

### 3. Configuration Structure

Each config file specifies:

```json
{
  "num_agents": N,
  "prompts_path": "../prompts/agent_prompts.json",
  "output_dir": "../output/images_N_agents",
  "num_samples_per_combination": 20,
  "num_prompts_to_process": null,
  "guidance_scale": 10.0,
  "num_inference_steps": 5,
  "bidding_combinations": [
    [0.0, 0.0, ..., 0.0],  # Base prompt only
    [1.0, 0.0, ..., 0.0],  # Agent 1 dominance
    ...
  ]
}
```

### 4. Documentation Files

- **`config/README.md`**: Complete configuration guide with examples
- **`PIPELINE_UPDATE.md`**: Details on required Flux pipeline changes
- **`RESTRUCTURE_SUMMARY.md`**: This file

## Key Design Decisions

### 1. Generalized Pipeline Call

**Before (Agent-Specific):**
```python
images = pipeline(
    agent1_prompt=agent1_prompt,
    agent1_bid=bid1,
    agent2_prompt=agent2_prompt,
    agent2_bid=bid2,
    agent3_prompt=agent3_prompt,
    agent3_bid=bid3,
    base_prompt=base_prompt,
    ...
)
```

**After (Generalized):**
```python
images = pipeline(
    agent_prompts=agent_prompts,  # List of N prompts
    agent_bids=agent_bids,        # List of N bids
    base_prompt=base_prompt,
    ...
)
```

### 2. Bidding Combinations Strategy

Each config includes:
- **[0, 0, ..., 0]**: Base prompt only (essential reference)
- **[1, 0, ..., 0]**: Single agent dominance
- **Equal splits**: [1/N, 1/N, ..., 1/N] for all agents
- **Hierarchical**: Decreasing influence patterns
- **Custom**: Specific patterns for research

### 3. Centralized Prompts

All configs use `prompts/agent_prompts.json` which contains up to 20 agent prompts. The script automatically extracts the needed agents.

### 4. Flexible Output Structure

Output directories are specified in configs and automatically created:
- `output/images_2_agents/`
- `output/images_3_agents/`
- `output/images_5_agents/`
- `output/images_10_agents/`
- `output/images_20_agents/`

## File Changes

### New Files
```
config/
├── config_2_agents.json
├── config_3_agents.json
├── config_5_agents.json
├── config_10_agents.json
├── config_20_agents.json
├── README.md
scripts/
├── generate_images.py

PIPELINE_UPDATE.md
RESTRUCTURE_SUMMARY.md
```

### Unchanged Files (Backwards Compatible)
- `scripts/generate_images_2_agent.py` (can be deprecated)
- `scripts/generate_images_3_agent.py` (can be deprecated)
- `prompts/agent_prompts.json` (used by all configs)

## Next Steps

### 1. Update Flux Pipeline (Required)

Modify `pipelines/flux_auction_pipeline.py` to accept list-based parameters:
- Change `__call__` signature to accept `agent_prompts` and `agent_bids` lists
- Update the auction mechanism to handle variable numbers of agents
- See `PIPELINE_UPDATE.md` for implementation details

### 2. Test the Generalized Script

```bash
# Test with 3 agents (recommended first test)
cd scripts
python generate_images.py ../config/config_3_agents.json

# Test with other agent counts
python generate_images.py ../config/config_2_agents.json
python generate_images.py ../config/config_5_agents.json
```

### 3. Customize Bidding Combinations (Optional)

Edit config files to adjust bidding combinations for your research needs. The script validates that configs have valid `num_agents`.

### 4. Clean Up Old Scripts (Optional)

Once the new script is tested and pipeline updated:
- Keep `generate_images_2_agent.py` and `generate_images_3_agent.py` for historical reference
- Or delete them and use `generate_images.py` exclusively

## Quick Reference

### To run for 3 agents:
```bash
cd scripts && python generate_images.py ../config/config_3_agents.json
```

### To run for 10 agents:
```bash
cd scripts && python generate_images.py ../config/config_10_agents.json
```

### To create a custom config for 7 agents:
1. Copy `config/config_5_agents.json`
2. Change `num_agents` to 7
3. Create 7 bid vectors in `bidding_combinations`
4. Run with new config

## Bidding Combinations Design

All configs follow this pattern:
1. **[0, 0, ..., 0]** - Base prompt reference (always included)
2. **[1, 0, ..., 0]** - First agent dominance (for comparison)
3. **[1/N, 1/N, ..., 1/N]** - Perfect equality (benchmark fairness)
4. **Hierarchical** - Realistic power distributions
5. **Custom** - Problem-specific scenarios

This ensures comprehensive testing across different auction scenarios.

## Notes

- Configurations are human-readable JSON - easy to edit
- All paths are relative to the config directory location
- The script includes progress bars and detailed logging
- Output includes `generation_log.json` with all metadata
- Images are organized by prompt index for easy analysis
- The script safely resumes from interruptions (checks existing files)

## Validation

The script validates:
- Config file exists and is valid JSON
- Prompts file exists at specified path
- All required config fields are present
- Output directory can be created
- Number of agents matches bidding combination sizes
