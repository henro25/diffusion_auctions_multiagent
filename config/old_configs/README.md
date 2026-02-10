# Configuration Files

This directory contains configuration files for the generalized `generate_images.py` script.

## Configuration Structure

Each config file is a JSON file with the following structure:

```json
{
  "num_agents": 2,
  "prompts_path": "../prompts/agent_prompts.json",
  "output_dir": "../output/images_2_agents",
  "num_samples_per_combination": 20,
  "num_prompts_to_process": null,
  "guidance_scale": 10.0,
  "num_inference_steps": 5,
  "bidding_combinations": [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 0.5]
  ]
}
```

### Fields

- **num_agents** (int): Number of agents in the auction
- **prompts_path** (str): Path to the prompts JSON file (relative to config directory)
- **output_dir** (str): Output directory for generated images (relative to config directory)
- **num_samples_per_combination** (int): Number of samples to generate for each bid combination
- **num_prompts_to_process** (int or null): Number of prompts to process (null = all)
- **guidance_scale** (float): Guidance scale for diffusion model
- **num_inference_steps** (int): Number of denoising steps
- **bidding_combinations** (list of lists): List of bid vectors to test

## Available Configurations

### config_2_agents.json
2-agent auction scenarios including:
- Base prompt only: [0.0, 0.0]
- Single agent dominance: [1.0, 0.0], [0.0, 1.0]
- Equal split: [0.5, 0.5]
- Asymmetric: [0.7, 0.3], [0.3, 0.7]

### config_3_agents.json
3-agent auction scenarios including:
- Base prompt only: [0.0, 0.0, 0.0]
- Single agent dominance: [1.0, 0.0, 0.0], etc.
- Equal split: [0.33, 0.33, 0.33]
- Two-agent scenarios: [0.5, 0.5, 0.0]
- Hierarchical bids: [0.6, 0.3, 0.1], [0.6, 0.2, 0.2]

### config_5_agents.json
5-agent auction scenarios with reasonable distributions across agents

### config_10_agents.json
10-agent auction scenarios with decreasing influence patterns

### config_20_agents.json
20-agent auction scenarios designed for larger-scale testing

## Usage

To generate images using a configuration:

```bash
cd scripts
python generate_images.py ../config/config_3_agents.json
```

## Output Structure

Generated images are saved to `output_dir` with the following structure:

```
output/images_3_agents/
├── prompt_000/
│   ├── idx000_b1_0.00_b2_0.00_b3_0.00_s00.png
│   ├── idx000_b1_0.00_b2_0.00_b3_0.00_s01.png
│   ├── idx000_b1_1.00_b2_0.00_b3_0.00_s00.png
│   └── ...
├── prompt_001/
│   └── ...
├── generation_log.json
```

The `generation_log.json` file contains metadata about all generated images including:
- Item index
- Bid vector
- Sample index
- Agent prompts
- Base prompt
- Image path

## Prompt Format

Prompts should be in `prompts/agent_prompts.json` with the following structure:

```json
[
  {
    "base_prompt": "Two friends chatting over coffee at a cafe",
    "agent1_prompt": "Cappuccino drink",
    "agent2_prompt": "Microsoft Surface laptop",
    "agent3_prompt": "USM Haller",
    "agent4_prompt": "Marriott key card",
    ...
    "agent20_prompt": "Nike running shoes"
  },
  ...
]
```

The script will automatically extract the required number of agent prompts based on `num_agents`.

## Customizing Configurations

To create a new configuration:

1. Copy an existing config file
2. Update `num_agents` to the desired number
3. Define `bidding_combinations` as a list of bid vectors
4. Ensure each bid vector sums to approximately 1.0 (recommended but not required)
5. Adjust `num_samples_per_combination` and other parameters as needed

## Notes

- All bid values should be between 0.0 and 1.0
- The pipeline will be called with `agent_prompts` (list) and `agent_bids` (list)
- Base prompt (all bids = 0.0) should always be included for reference
- Paths are relative to the config directory location
