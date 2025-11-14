# Usage Examples

## Quick Start

### Generate images with 3 agents
```bash
cd scripts
python generate_images.py ../config/config_3_agents.json
```

### Generate images with 10 agents
```bash
cd scripts
python generate_images.py ../config/config_10_agents.json
```

## Configuration Examples

### 2-Agent Setup
```bash
cd scripts
python generate_images.py ../config/config_2_agents.json
```

**Config includes:**
- Base only: [0.0, 0.0]
- Agent 1 dominant: [1.0, 0.0]
- Agent 2 dominant: [0.0, 1.0]
- Equal: [0.5, 0.5]
- Asymmetric: [0.7, 0.3], [0.3, 0.7]

**Output location:** `output/images_2_agents/`

---

### 3-Agent Setup
```bash
cd scripts
python generate_images.py ../config/config_3_agents.json
```

**Config includes:**
- Base only: [0.0, 0.0, 0.0]
- Single dominance: [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
- Equal: [0.33, 0.33, 0.33]
- Hierarchical: [0.6, 0.3, 0.1], [0.6, 0.2, 0.2]

**Output location:** `output/images_3_agents/`

---

### 5-Agent Setup
```bash
cd scripts
python generate_images.py ../config/config_5_agents.json
```

**Config includes:**
- Base only: [0.0, 0.0, 0.0, 0.0, 0.0]
- One dominant: [1.0, 0.0, 0.0, 0.0, 0.0]
- Equal split: [0.2, 0.2, 0.2, 0.2, 0.2]
- Decreasing: [0.4, 0.3, 0.2, 0.05, 0.05]

**Output location:** `output/images_5_agents/`

---

### 10-Agent Setup
```bash
cd scripts
python generate_images.py ../config/config_10_agents.json
```

**Config includes:**
- Base only: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
- One dominant: [1.0, 0.0, ...]
- Equal split: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
- Decreasing influence: [0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.01, 0.01]

**Output location:** `output/images_10_agents/`

---

### 20-Agent Setup
```bash
cd scripts
python generate_images.py ../config/config_20_agents.json
```

**Config includes:**
- Base only: [0.0, 0.0, ..., 0.0] (20 zeros)
- One dominant: [1.0, 0.0, ..., 0.0]
- Equal split: [0.05, 0.05, ..., 0.05] (20 agents)
- Decreasing: [0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05, ...]

**Output location:** `output/images_20_agents/`

---

## Custom Configurations

### Creating a 7-agent configuration

1. **Create new config file:** `config/config_7_agents.json`

```json
{
  "num_agents": 7,
  "prompts_path": "../prompts/agent_prompts.json",
  "output_dir": "../output/images_7_agents",
  "num_samples_per_combination": 20,
  "num_prompts_to_process": null,
  "guidance_scale": 10.0,
  "num_inference_steps": 5,
  "bidding_combinations": [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
    [0.4, 0.25, 0.15, 0.1, 0.05, 0.03, 0.02],
    [0.35, 0.3, 0.15, 0.1, 0.05, 0.03, 0.02],
    [0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05]
  ]
}
```

2. **Run the script:**
```bash
cd scripts
python generate_images.py ../config/config_7_agents.json
```

---

## Expected Output Structure

After running any configuration, you'll get:

```
output/images_3_agents/
├── prompt_000/
│   ├── idx000_b1_0.00_b2_0.00_b3_0.00_s00.png
│   ├── idx000_b1_0.00_b2_0.00_b3_0.00_s01.png
│   ├── idx000_b1_0.00_b2_0.00_b3_0.00_s02.png
│   ├── ...
│   ├── idx000_b1_1.00_b2_0.00_b3_0.00_s00.png
│   ├── ...
│   ├── idx000_b1_0.33_b2_0.33_b3_0.33_s00.png
│   └── ...
├── prompt_001/
│   ├── idx001_b1_0.00_b2_0.00_b3_0.00_s00.png
│   └── ...
├── prompt_002/
│   └── ...
└── generation_log.json
```

**generation_log.json contains:**
```json
[
  {
    "item_index": 0,
    "bids": [0.0, 0.0, 0.0],
    "sample_index": 0,
    "agent_prompts": [
      "Cappuccino drink",
      "Microsoft Surface laptop",
      "USM Haller"
    ],
    "base_prompt": "Two friends chatting over coffee at a cafe",
    "image_path": "path/to/image.png"
  },
  ...
]
```

---

## Script Options and Customization

### Limit to first N prompts (good for testing)

Edit config and set:
```json
{
  "num_prompts_to_process": 5
}
```

### Generate fewer samples per combination

```json
{
  "num_samples_per_combination": 5
}
```

### Adjust inference parameters

```json
{
  "guidance_scale": 12.0,
  "num_inference_steps": 10
}
```

---

## Common Workflows

### 1. Test with few prompts
```json
{
  "num_prompts_to_process": 3,
  "num_samples_per_combination": 2
}
```
Result: 3 prompts × ~10 combinations × 2 samples = ~60 images

### 2. Full research run
```json
{
  "num_prompts_to_process": null,
  "num_samples_per_combination": 20
}
```
Result: All prompts × combinations × 20 samples each (comprehensive)

### 3. Quick validation
```json
{
  "num_prompts_to_process": 1,
  "num_samples_per_combination": 1,
  "bidding_combinations": [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.33, 0.33, 0.33]
  ]
}
```
Result: 1 prompt × 3 combinations × 1 sample = 3 images (< 1 minute)

---

## Monitoring Progress

The script will display:
- Configuration summary
- Progress bar with number of prompts processed
- Each image being generated with its filename
- Final summary with total images created

Example output:
```
=== 3-Agent Diffusion Auctions Image Generation ===
Configuration file: ../config/config_3_agents.json

Configuration:
  - 3 agents
  - 50 prompts
  - 10 bidding combinations
  - 20 samples per combination
  - Total images to generate: 10000
  - Output directory: ../output/images_3_agents

Processing Prompts: 100%|████████| 50/50 [03:45<00:00, 4.50s/prompt]

Generation complete!
  - 10000 images generated successfully
  - Results saved to: ../output/images_3_agents/generation_log.json
```

---

## Resuming Interrupted Runs

If the script is interrupted, just run it again with the same config. The script automatically:
- Checks for existing images
- Skips already generated images
- Continues from where it left off

No data loss or duplication!

---

## Troubleshooting

### "Config file not found"
Make sure you're using the correct relative path:
```bash
# Correct (from scripts directory)
python generate_images.py ../config/config_3_agents.json

# Incorrect (from repo root)
python scripts/generate_images.py config/config_3_agents.json
```

### "Prompts file not found"
The path `../prompts/agent_prompts.json` is relative to the config directory. Check that it exists.

### Agent prompt index out of range
The config `num_agents` must match the number of agents in your prompts file. `agent_prompts.json` has up to 20 agents.

### Memory issues with large N
For 10+ agents with many prompts, consider:
- Reducing `num_samples_per_combination`
- Limiting `num_prompts_to_process`
- Using fewer bidding combinations
