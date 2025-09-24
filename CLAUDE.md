# CLAUDE.md - Development Context

## 🎯 Project Overview
**Diffusion Auctions Multi-Agent** - A research project implementing auction mechanisms for diffusion models where multiple agents bid to influence image generation. Higher bids result in greater representation in the final generated image.

**Authors:** Lillian Sun, Warren Zhu, Henry Huang
**Academic Context:** 4th Year research project on multi-winner auctions for generative AI

## 🏗️ Architecture & Core Concepts

### Auction Mechanism
- **2-3 agents** place bids (0.0-1.0) to influence final image generation
- **Score Composition Algorithm:** Recursive weighted interpolation based on bid ratios
- **Sorting:** Agents sorted by bid amount (highest to lowest)
- **Weight Calculation:** Uses dominance ratios to determine influence

### Key Algorithm (FluxPipelineAuction.__call__)
1. Sort agents by bid amount: `sb1 >= sb2 >= sb3`
2. Calculate dominance weights:
   - `P_dom_A = (sb1 + sb2) / S3` where `S3 = sb1 + sb2 + sb3`
   - `w_A = 2 * P_dom_A - 1`
   - `P_dom_B = sb1 / S2` where `S2 = sb1 + sb2`
   - `w_B = 2 * P_dom_B - 1`
3. Recursive composition:
   - `s_1_2_intermediate = (1-w_B) * noise_pred_s1_s2 + w_B * noise_pred_s1`
   - `final_noise_pred = (1-w_A) * noise_pred_s1_s2_s3 + w_A * s_1_2_intermediate`

## 📁 Project Structure
```
diffusion_auctions_multiagent/
├── pipelines/                        # Core pipeline implementations
│   ├── __init__.py                   # Module exports
│   ├── flux_auction_pipeline.py      # FluxPipelineAuction class
│   └── README.md                     # Pipeline documentation
├── scripts/                          # Main generation scripts
│   ├── generate_images_3_agent.py    # 3-agent single-GPU script (auto-cache)
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
├── requirements.txt                  # Dependencies
├── README.md                         # User documentation
└── CLAUDE.md                         # This file
```

## 🔧 Key Configuration (scripts/generate_images_3_agent.py)

### Configuration Section (Lines 13-55)
```python
# Path configurations
PROMPTS_PATH = "../prompts/prompts_3_agent.json"
OUTPUT_DIR = "images/images_3_agent"

# Sampling configuration
NUM_SAMPLES_PER_COMBINATION = 1      # Multiple sampling support
NUM_PROMPTS_TO_PROCESS = None        # Limit prompts (None = all)

# Generation parameters
GUIDANCE_SCALE = 10.0
NUM_INFERENCE_STEPS = 5
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16

# Bidding combinations (Lines 29-53)
BIDDING_COMBINATIONS_3_AGENT = [
    (0.0, 0.0, 0.0),      # Base prompt only (no agent influence)
    (1.0, 0.0, 0.0),      # Agent 1 dominant
    (0.33, 0.33, 0.33),   # All equal
    (0.4, 0.4, 0.2),      # A1 & A2 strong, A3 minor
    (0.6, 0.3, 0.1),      # Clear hierarchy
    (0.6, 0.2, 0.2),      # A1 > A2 = A3
]
```

### Key Functions
- **`FluxPipelineAuction.__call__`** (pipelines/flux_auction_pipeline.py): Core auction mechanism
- **`generate_and_save_image`** (scripts/generate_images_3_agent.py): Image generation wrapper
- **Multi-GPU support** (scripts/multi_gpu_config.py): Parallel processing across GPUs

## 🐛 Known Issues & Fixes Applied

### Fixed Issues
- **Image access bug** (Line 705): Fixed `images[0][0].save()` → `images.images[0].save()`
- **Path configuration**: Centralized hardcoded values to configuration section
- **Multiple sampling**: Added sample indexing for statistical analysis

### Potential Issues (Not Fixed - Need Domain Expertise)
- **Line 422**: `image_seq_len` calculation might be incorrect
- **Lines 560-582**: Complex bid edge case handling in score composition
- **Line 17**: Path `../prompts/` assumes script run from `scripts/` directory

## 📊 Data Formats

### Prompt Structure (prompts_3_agent.json)
```json
{
  "base_prompt": "Scene description",
  "agent1_prompt": "Brand/object for agent 1",
  "agent2_prompt": "Brand/object for agent 2",
  "agent3_prompt": "Brand/object for agent 3"
}
```

### Output Naming Convention
`idx{prompt_index:03d}_b1_{bid1:.2f}_b2_{bid2:.2f}_b3_{bid3:.2f}_s{sample_idx:02d}.png`

Examples:
- Base prompt only: `idx000_b1_0.00_b2_0.00_b3_0.00_s00.png`
- Agent bidding: `idx000_b1_0.60_b2_0.30_b3_0.10_s00.png`

### Generation Log (generation_log.json)
```json
{
  "item_index": 0,
  "bids": [0.6, 0.3, 0.1],
  "sample_index": 0,
  "agent1_prompt": "Starbucks mug",
  "agent2_prompt": "Apple MacBook Air",
  "agent3_prompt": "New York Times newspaper",
  "base_prompt": "Two friends chatting over coffee at a cafe",
  "image_path": "path/to/generated/image.png"
}
```

## 🚀 Usage Patterns

### Development Workflow

#### Single GPU (Recommended for Development)
1. **Modify configuration** at top of `generate_images_3_agent.py`
2. **Run from scripts directory**:
   - Standard: `cd scripts && python generate_images_3_agent.py`
   - Cluster-optimized: `cd scripts && ./run_with_cache.sh`
3. **Check outputs** in `images/images_3_agent/prompt_XXX/`
4. **Resume interrupted runs**: Script automatically skips existing images, allowing safe resumption

#### Multi-GPU (Production)
1. **Configure GPUs** in `generate_images_3_agent_multigpu.py`:
   ```python
   USE_MULTI_GPU = True
   GPU_INDICES = [0, 1, 2, 3]  # Specify exact GPUs
   ```
2. **Run**: `cd scripts && python generate_images_3_agent_multigpu.py`
3. **Check outputs** in `images/images_3_agent_multigpu/prompt_XXX/`

#### Cluster/HPC Environments
For faster model downloads on clusters:
```bash
# Automatic cache setup + generation (3-agent)
cd scripts && ./run_with_cache.sh

# Automatic cache setup + generation (2-agent)
cd scripts && ./run_with_cache_2_agent.sh

# Manual cache management
python helpers/manage_cache.py list
python helpers/manage_cache.py clean black-forest-labs/FLUX.1-schnell
```

### Common Modifications
```python
# Test multiple samples for statistical analysis
NUM_SAMPLES_PER_COMBINATION = 5

# Test specific bidding scenarios
BIDDING_COMBINATIONS_3_AGENT = [
    (1.0, 0.0, 0.0),     # Only agent 1
    (0.0, 1.0, 0.0),     # Only agent 2
    (0.0, 0.0, 1.0),     # Only agent 3
]

# Process only first few prompts during development
NUM_PROMPTS_TO_PROCESS = 3
```

## 🧪 Testing Scenarios

### Validation Tests
- **Base prompt only**: `(0.0, 0.0, 0.0)` should show only base prompt content with no agent influence
- **Single agent dominance**: `(1.0, 0.0, 0.0)` should show only agent 1's content
- **Equal influence**: `(0.33, 0.33, 0.33)` should blend all agents equally
- **Clear hierarchy**: `(0.6, 0.3, 0.1)` should show proportional influence

### Edge Cases (Commented Out)
```python
# Important test cases preserved in BIDDING_COMBINATIONS_3_AGENT comments:
# (0.0, 1.0, 0.0),      # Agent 2 dominant
# (0.5, 0.5, 0.0),      # Two-agent equal
# (0.7, 0.2, 0.1),      # Strong dominance with weak third
```

## 🔬 Research Context

### Academic Goals
- Design auction mechanisms for diffusion models
- Demonstrate proportional influence based on bidding
- Characterize the tilted distribution from score composition
- Experimental validation of multi-winner auctions

### Key Metrics to Evaluate
- **Visual influence correlation**: Higher bids → more visible brand presence
- **Fairness**: Equal bids → equal visual representation
- **Stability**: Consistent results across multiple samples
- **Scalability**: Performance with different agent counts

## 🛠️ Development Notes

### Dependencies
- **Core**: `torch`, `diffusers`, `transformers`
- **Model**: FLUX.1-schnell (black-forest-labs)
- **Hardware**: CUDA GPU recommended (8-12GB VRAM)

### Performance
- **Generation time**: ~5-10 seconds per image on GPU
- **Memory usage**: ~8-12GB VRAM
- **Storage**: ~2-5MB per generated image

### Common Commands
```bash
# Run generation (3-agent standard)
cd scripts && python generate_images_3_agent.py

# Run generation (2-agent standard)
cd scripts && python generate_images_2_agent.py

# Run generation (cluster-optimized)
cd scripts && ./run_with_cache.sh          # 3-agent
cd scripts && ./run_with_cache_2_agent.sh  # 2-agent

# Cache management
python helpers/manage_cache.py list
python helpers/manage_cache.py usage
python helpers/manage_cache.py clean [model_name]

# Project maintenance
git status  # Many files in images/ are gitignored
pip install -r requirements.txt
source .venv/bin/activate
```

## 🎨 Prompt Engineering

### Effective Base Prompts
- Clear scene descriptions work best
- Specific contexts: "cafe", "beach", "office"
- Natural scenarios where brands can be integrated

### Agent Prompt Strategies
- **Brand names**: "Starbucks mug", "Nike sneakers"
- **Product categories**: "Apple MacBook Air", "Coca-Cola cans"
- **Contextual objects**: "New York Times newspaper"

## 📝 Future Development Areas

### Potential Enhancements
1. **Dynamic bidding combinations**: Generate based on mathematical properties
2. **Evaluation metrics**: Automated visual analysis of brand presence
3. **Multi-resolution testing**: Different image sizes and aspect ratios
4. **Prompt complexity**: More complex multi-object scenes
5. **Alternative models**: Test with other diffusion models beyond FLUX

### Research Extensions
1. **4+ agent scenarios**: Extend beyond 3 agents
2. **Auction mechanism variants**: Different weighting strategies
3. **Temporal dynamics**: Bidding over multiple generation steps
4. **Interactive bidding**: Real-time bid adjustment during generation

## 🆕 Recent Updates & Repository Status

### Current Branch: `gdaras`
- **Main Branch:** `main` (for PRs)
- **Status:** Clean working directory with recent cache infrastructure improvements

### Recent Changes (Sept 2025)
- **Added separate cache runners**: Created `run_with_cache_2_agent.sh` for dedicated 2-agent cache support
- **Enhanced documentation**: Updated README.md and CLAUDE.md with 2-agent/3-agent separation
- **Repository improvements**: Better organization of cache utilities and multi-agent scenarios

### Current Repository State
- **Untracked files**: `scripts/run_with_cache_2_agent.sh` (newly created)
- **Modified files**: `README.md` (updated with new cache commands)
- **Recent commits**:
  - cb859ef: Adding 2 Agent Image Generation
  - 71060d6: Generated 20 Samples Each and Zipped 3 Agent Images
  - b9765a6: Fixing Boolean Tensor Bug 3

### Key Infrastructure Additions
1. **Separate 2-agent cache runner**: `scripts/run_with_cache_2_agent.sh`
2. **Enhanced documentation**: Clear separation between 2-agent and 3-agent workflows
3. **Project structure updates**: Better organization in both README.md and CLAUDE.md

---
*Last updated: September 2025 - Cache infrastructure and 2-agent separation*