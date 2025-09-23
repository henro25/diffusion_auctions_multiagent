# Diffusion Auctions Multi-Agent

**Authors:** Lillian Sun, Warren Zhu, Henry Huang

## 🎯 Overview

This project implements a novel auction mechanism for diffusion models where multiple agents bid to influence image generation. Higher bids result in greater representation in the final generated image.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd diffusion_auctions_multiagent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace authentication
# Option 1: Create .env file (recommended for clusters)
echo "HF_TOKEN=your_token_here" > .env

# Option 2: Interactive login
huggingface-cli login
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

### Running the 3-Agent Image Generation

#### Standard Usage
```bash
cd scripts
python generate_images_3_agent.py
```

#### Multi-GPU Usage
```bash
cd scripts
python generate_images_3_agent_multigpu.py
```

#### Cluster/High-Performance Computing
For faster model downloads on clusters with local SSD storage:
```bash
cd scripts
./run_with_cache.sh                # Runs 3-agent generation with cache
```
This automatically:
- Sets up local cache to avoid slow network downloads
- Loads HuggingFace token from `.env` file for authentication
- Validates authentication before starting generation

### Running the 2-Agent Image Generation

The 2-agent scripts automatically sweep across bid combinations where one agent gets `b` and the other gets `1-b`, with `b` taking values: 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0.

#### Standard Usage
```bash
cd scripts
python generate_images_2_agent.py
```

#### Multi-GPU Usage
```bash
cd scripts
python generate_images_2_agent_multigpu.py
```

#### Cluster/High-Performance Computing
For faster model downloads on clusters with local SSD storage:
```bash
cd scripts
./run_with_cache_2_agent.sh        # Runs 2-agent generation with cache
```
This automatically:
- Sets up local cache to avoid slow network downloads
- Loads HuggingFace token from `.env` file for authentication
- Validates authentication before starting generation

### Expected Output
- Generated images will be saved to `images/images_3_agent/`

#### 3-Agent Generation
- Generated images will be saved to `images/images_3_agent/`
- Each prompt gets its own subdirectory: `prompt_000/`, `prompt_001/`, etc.
- Images are named with bid information: `idx000_b1_1.00_b2_0.00_b3_0.00_s00.png`
- Base prompt only images: `idx000_b1_0.00_b2_0.00_b3_0.00_s00.png`
- Generation log saved as `generation_log.json`
- Automatically skips existing images to allow resuming interrupted runs

#### 2-Agent Generation
- Generated images will be saved to `images/images_2_agent/`
- Multi-GPU version saves to `images_2_agent_multigpu/`
- Images are named with bid information: `idx000_b1_0.70_b2_0.30_s00.png`
- Generation log saved as `generation_log.json`

## 📁 Project Structure
```
diffusion_auctions_multiagent/
├── pipelines/                        # Core pipeline implementations
│   ├── __init__.py                   # Module exports
│   ├── flux_auction_pipeline.py      # FluxPipelineAuction class
│   └── README.md                     # Pipeline documentation
├── scripts/                          # Main generation scripts
│   ├── generate_images_3_agent.py    # 3-agent single-GPU script
│   ├── generate_images_3_agent_multigpu.py # 3-agent multi-GPU script
│   ├── generate_images_2_agent.py    # 2-agent single-GPU script
│   ├── generate_images_2_agent_multigpu.py # 2-agent multi-GPU script
│   ├── multi_gpu_config.py           # Multi-GPU management
│   ├── run_with_cache.sh             # 3-agent cluster-optimized runner
│   └── run_with_cache_2_agent.sh     # 2-agent cluster-optimized runner
├── helpers/                          # Utility scripts and tools
│   ├── setup_cache.sh                # HuggingFace cache setup
│   ├── manage_cache.py               # Cache management utility
│   └── CLUSTER_SETUP.md              # Cluster deployment guide
├── prompts/                          # Prompt configurations
│   ├── prompts_3_agent.json          # 3-agent test scenarios
│   ├── prompts_2_agent.json          # 2-agent scenarios
│   └── base_prompts.json             # Base prompt library
├── images/                           # Generated outputs (gitignored)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── CLAUDE.md                         # Development context
```

## 🎲 How It Works

### Bidding System
Each agent places a bid (0.0-1.0) to influence the final image:
- **Agent 1:** Highest bidder gets the most influence
- **Agent 2:** Second highest bidder gets moderate influence
- **Agent 3:** Lowest bidder gets least influence (3-agent scenarios only)

### Score Composition Algorithm
The system uses a recursive score composition method:
1. Sorts agents by bid amount (highest to lowest)
2. Calculates dominance weights based on bid ratios
3. Combines noise predictions using weighted interpolation
4. Higher bids = greater weight in final image generation

### Example Bidding Scenarios

#### 3-Agent Scenarios
- `(0.0, 0.0, 0.0)`: Base prompt only (no agent influence)
- `(1.0, 0.0, 0.0)`: Agent 1 completely dominates
- `(0.33, 0.33, 0.33)`: All agents have equal influence
- `(0.6, 0.3, 0.1)`: Agent 1 > Agent 2 > Agent 3 with clear hierarchy

#### 2-Agent Scenarios (Auto-generated sweep)
- `(0.0, 1.0)`: Agent 2 completely dominates
- `(0.3, 0.7)`: Agent 2 has more influence than Agent 1
- `(0.5, 0.5)`: Both agents have equal influence
- `(0.7, 0.3)`: Agent 1 has more influence than Agent 2
- `(1.0, 0.0)`: Agent 1 completely dominates

## 💡 Project Concept

Traditional auctions have a single winner, but generative AI enables **multi-winner auctions** where multiple bidders can influence the outcome proportionally to their bids.

**Example scenario:**
- Base prompt: "Two friends chatting over coffee at a cafe"
- Agent 1 bids 0.6 for "Starbucks mug"
- Agent 2 bids 0.3 for "Apple MacBook Air"
- Agent 3 bids 0.1 for "New York Times newspaper"

Result: Generated image shows friends at cafe with prominent Starbucks mug, visible MacBook, and subtle newspaper presence.

## 🔧 Customization

### Adding New Prompts
Edit `prompts/prompts_3_agent.json`:
```json
{
  "base_prompt": "Your scene description",
  "agent1_prompt": "Brand/object for agent 1",
  "agent2_prompt": "Brand/object for agent 2",
  "agent3_prompt": "Brand/object for agent 3"
}
```

### Modifying Bid Combinations
- **3-Agent:** In `generate_images_3_agent.py`, edit the `BIDDING_COMBINATIONS_3_AGENT` list to test different bid scenarios.
- **2-Agent:** In `generate_images_2_agent.py`, modify the `SWEEP_VALUES` list to change the bid sweep range. The script automatically generates combinations where Agent 1 gets `b` and Agent 2 gets `1-b`.

## 🐛 Troubleshooting

### Common Issues
- **CUDA out of memory:** Reduce `num_inference_steps` or use smaller batch size
- **Path errors:** Ensure you're running from the `scripts/` directory
- **Missing dependencies:** Run `pip install -r requirements.txt`
- **Slow model downloads:** Use `./run_with_cache.sh` (3-agent) or `./run_with_cache_2_agent.sh` (2-agent) for cluster environments

### Performance Notes
- Generation time: ~5-10 seconds per image on GPU
- Memory usage: ~8-12GB VRAM for FLUX.1-schnell model
- Storage: ~2-5MB per generated image

### Cluster Environments
For detailed cluster setup and cache management, see `helpers/CLUSTER_SETUP.md`:
- Local SSD cache setup for faster downloads
- Cache management utilities
- Troubleshooting interrupted downloads


## 📚 References

- [1] [Auctions with LLM Summaries](https://arxiv.org/abs/2404.08126)
- [2] [Ad Auctions for LLMs via Retrieval Augmented Generation](https://arxiv.org/abs/2406.09459)
- [3] [Mechanism Design for Large Language Models](https://arxiv.org/pdf/2310.10826)
- [4] [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [5] [Classifier-Free Guidance Is a Predictor-Corrector](https://machinelearning.apple.com/research/predictor-corrector)
