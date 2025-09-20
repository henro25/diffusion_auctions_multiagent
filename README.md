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

# Log into HuggingFace
huggingface-cli login
```

### Running the 3-Agent Image Generation
```bash
# From the project root directory
cd scripts
python generate_images_3_agent.py
```

### Expected Output
- Generated images will be saved to `/datastor1/gdaras/diffusion_auctions_multiagent/images/images_3_agent/`
- Each prompt gets its own subdirectory: `prompt_000/`, `prompt_001/`, etc.
- Images are named with bid information: `idx000_b1_1.00_b2_0.00_b3_0.00.png`
- Generation log saved as `generation_log.json`

## 📁 Project Structure
```
diffusion_auctions_multiagent/
├── scripts/
│   └── generate_images_3_agent.py    # Main 3-agent generation script
├── prompts/
│   ├── prompts_3_agent.json          # 3-agent prompt configurations
│   ├── prompts_2_agent.json          # 2-agent prompt configurations
│   └── base_prompts.json             # Base prompt library
├── images/                           # Generated images output directory
└── requirements.txt                  # Python dependencies
```

## 🎲 How It Works

### Bidding System
Each agent places a bid (0.0-1.0) to influence the final image:
- **Agent 1:** Highest bidder gets the most influence
- **Agent 2:** Second highest bidder gets moderate influence
- **Agent 3:** Lowest bidder gets least influence

### Score Composition Algorithm
The system uses a recursive score composition method:
1. Sorts agents by bid amount (highest to lowest)
2. Calculates dominance weights based on bid ratios
3. Combines noise predictions using weighted interpolation
4. Higher bids = greater weight in final image generation

### Example Bidding Scenarios
- `(1.0, 0.0, 0.0)`: Agent 1 completely dominates
- `(0.33, 0.33, 0.33)`: All agents have equal influence
- `(0.6, 0.3, 0.1)`: Agent 1 > Agent 2 > Agent 3 with clear hierarchy

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
In `generate_images_3_agent.py`, edit the `bidding_combinations_3_agent` list to test different bid scenarios.

## 🐛 Troubleshooting

### Common Issues
- **CUDA out of memory:** Reduce `num_inference_steps` or use smaller batch size
- **Path errors:** Ensure you're running from the `scripts/` directory
- **Missing dependencies:** Run `pip install -r requirements.txt`

### Performance Notes
- Generation time: ~5-10 seconds per image on GPU
- Memory usage: ~8-12GB VRAM for FLUX.1-schnell model
- Storage: ~2-5MB per generated image


## 📚 References

- [1] [Auctions with LLM Summaries](https://arxiv.org/abs/2404.08126)
- [2] [Ad Auctions for LLMs via Retrieval Augmented Generation](https://arxiv.org/abs/2406.09459)
- [3] [Mechanism Design for Large Language Models](https://arxiv.org/pdf/2310.10826)
- [4] [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [5] [Classifier-Free Guidance Is a Predictor-Corrector](https://machinelearning.apple.com/research/predictor-corrector)
