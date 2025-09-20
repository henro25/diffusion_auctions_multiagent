"""
2-Agent Image Generation Script

This script generates images using the FluxPipelineAuction with 2 agents.
Sweeps across bid values: b=0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 with the other agent getting 1-b.
For multi-GPU support, use generate_images_2_agent_multigpu.py instead.

Authors: Lillian Sun, Warren Zhu, Henry Huang
Academic Context: 4th Year research project on multi-winner auctions for generative AI
"""

import os
import json
import sys
import torch
from tqdm import tqdm


# Setup HuggingFace cache before importing models
def setup_hf_cache():
    """Setup HuggingFace cache to use local SSD instead of slow NFS."""
    # Check if cache is already configured
    if "HF_HOME" in os.environ:
        print(f"Using existing HF cache: {os.environ['HF_HOME']}")
        return

    # Try to find a local SSD path
    user = os.environ.get("USER", os.environ.get("USERNAME", "user"))
    cache_paths = [
        f"/scratch/{user}/hf-cache",
        f"/tmp/{user}/hf-cache",
        f"/dev/shm/{user}/hf-cache",
        f"/var/tmp/{user}/hf-cache",
    ]

    cache_dir = None
    for path in cache_paths:
        parent = os.path.dirname(path)
        if os.access(parent, os.W_OK):
            cache_dir = path
            break

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(f"{cache_dir}/hub", exist_ok=True)
        os.makedirs(f"{cache_dir}/transformers", exist_ok=True)

        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = f"{cache_dir}/hub"
        os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"

        print(f"Setup HF cache at: {cache_dir}")
    else:
        print(
            "Warning: Could not find local SSD for cache, using default (may be slow)"
        )


# Setup cache before importing heavy libraries
setup_hf_cache()

# Import the FluxPipelineAuction class from pipelines module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipelines import FluxPipelineAuction

# ===== CONFIGURATION SECTION =====
# Modify these parameters as needed

# Path configurations
PROMPTS_PATH = "../prompts/prompts_2_agent.json"  # Path to prompts file
OUTPUT_DIR = "/datastor1/gdaras/diffusion_auctions_multiagent/images/images_2_agent"  # Output directory for generated images

# Sampling configuration
NUM_SAMPLES_PER_COMBINATION = 1  # Number of times to sample each prompt-bid combination
NUM_PROMPTS_TO_PROCESS = None  # Number of prompts to process (None = all prompts)

# Generation parameters
GUIDANCE_SCALE = 10.0  # Guidance scale for generation
NUM_INFERENCE_STEPS = 5  # Number of denoising steps
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16

# Bidding sweep values for 2 agents
# Agent 1 gets b, Agent 2 gets (1-b), Agent 3 gets 0.0
SWEEP_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Generate bidding combinations automatically
BIDDING_COMBINATIONS_2_AGENT = []
for b in SWEEP_VALUES:
    BIDDING_COMBINATIONS_2_AGENT.append((b, 1.0 - b, 0.0))

# ===== END CONFIGURATION SECTION =====


def load_pipeline():
    """Initialize the FluxPipelineAuction on available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading pipeline on device: {device}")

    pipeline = FluxPipelineAuction.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=TORCH_DTYPE,
    ).to(device)

    return pipeline


def generate_and_save_image(pipeline, data_item, index, output_dir, bids, sample_idx=0):
    """
    Generate and save a single image using the auction pipeline.

    Args:
        pipeline: FluxPipelineAuction instance
        data_item: Dictionary containing prompt data
        index: Prompt index
        output_dir: Output directory
        bids: Tuple of (bid1, bid2, bid3) - note: bid3 will always be 0.0 for 2-agent
        sample_idx: Sample index for multiple sampling

    Returns:
        str: Path to saved image, or None if generation failed
    """
    agent1_prompt = data_item.get("agent1_prompt", "")
    agent2_prompt = data_item.get("agent2_prompt", "")
    base_prompt = data_item.get("base_prompt", "")

    bid1, bid2, bid3 = bids  # bid3 should always be 0.0

    filename_base = f"idx{index:03d}_b1_{bid1:.2f}_b2_{bid2:.2f}_s{sample_idx:02d}"
    prompt_specific_output_dir = os.path.join(output_dir, f"prompt_{index:03d}")
    os.makedirs(prompt_specific_output_dir, exist_ok=True)
    output_path = os.path.join(prompt_specific_output_dir, f"{filename_base}.png")

    print(
        f"Generating item {index}, sample {sample_idx}: Bids=({bid1:.2f}, {bid2:.2f})"
    )
    print(f"  A1: {agent1_prompt[:30]}... | A2: {agent2_prompt[:30]}...")

    try:
        images = pipeline(
            agent1_prompt=agent1_prompt,
            agent1_bid=bid1,
            agent2_prompt=agent2_prompt,
            agent2_bid=bid2,
            agent3_prompt="",  # Empty for 2-agent scenario
            agent3_bid=bid3,  # Always 0.0 for 2-agent scenario
            base_prompt=base_prompt,
            guidance_scale=GUIDANCE_SCALE,
            num_inference_steps=NUM_INFERENCE_STEPS,
        )

        if images and hasattr(images, "images") and len(images.images) > 0:
            images.images[0].save(output_path)
            print(f"Saved image to {output_path}")
            return output_path
        else:
            print(f"No image generated for {output_path}")
            return None

    except Exception as e:
        print(f"Error generating image for {filename_base}: {e}")
        return None


def main():
    """Main execution function."""
    print("=== 2-Agent Diffusion Auctions Image Generation ===")
    print(f"Bid sweep values: {SWEEP_VALUES}")
    print("Generated bid combinations:")
    for i, combo in enumerate(BIDDING_COMBINATIONS_2_AGENT):
        print(f"  {i + 1}: Agent1={combo[0]:.1f}, Agent2={combo[1]:.1f}")

    # Load prompts
    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)

    # Calculate total images
    num_items_to_process = (
        NUM_PROMPTS_TO_PROCESS if NUM_PROMPTS_TO_PROCESS is not None else len(prompts)
    )
    total_images = (
        num_items_to_process
        * len(BIDDING_COMBINATIONS_2_AGENT)
        * NUM_SAMPLES_PER_COMBINATION
    )

    print("\nConfiguration:")
    print(f"  - {num_items_to_process} prompts")
    print(f"  - {len(BIDDING_COMBINATIONS_2_AGENT)} bidding combinations")
    print(f"  - {NUM_SAMPLES_PER_COMBINATION} samples per combination")
    print(f"  - Total images to generate: {total_images}")
    print(f"  - Output directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load pipeline
    pipeline = load_pipeline()

    # Generate images
    generation_results = []
    for i, item_data in enumerate(
        tqdm(prompts[:num_items_to_process], desc="Processing Prompts")
    ):
        for bids_tuple in BIDDING_COMBINATIONS_2_AGENT:
            for sample_idx in range(NUM_SAMPLES_PER_COMBINATION):
                generated_image_path = generate_and_save_image(
                    pipeline, item_data, i, OUTPUT_DIR, bids_tuple, sample_idx
                )
                if generated_image_path:
                    generation_results.append(
                        {
                            "item_index": i,
                            "bids": bids_tuple[
                                :2
                            ],  # Only include agent1 and agent2 bids
                            "sample_index": sample_idx,
                            "agent1_prompt": item_data["agent1_prompt"],
                            "agent2_prompt": item_data["agent2_prompt"],
                            "base_prompt": item_data["base_prompt"],
                            "image_path": generated_image_path,
                        }
                    )

    # Save results
    results_filename = os.path.join(OUTPUT_DIR, "generation_log.json")
    with open(results_filename, "w") as f:
        json.dump(generation_results, f, indent=2)

    print(f"\nGeneration complete!")
    print(f"  - {len(generation_results)} images generated successfully")
    print(f"  - Results saved to: {results_filename}")


if __name__ == "__main__":
    main()
