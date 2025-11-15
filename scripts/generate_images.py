"""
Generalized Multi-Agent Image Generation Script

This script generates images using the FluxPipelineAuction with a configurable
number of agents. The configuration is loaded from a JSON config file that
specifies the number of agents, bidding combinations, and generation parameters.

The script accepts agent prompts and bids as lists, making it flexible for
any number of agents (2, 3, 5, 10, 20, etc.).

Authors: Lillian Sun, Warren Zhu, Henry Huang
Academic Context: 4th Year research project on multi-winner auctions for generative AI
"""

import os
import json
import sys
import torch
import argparse
from tqdm import tqdm
from pathlib import Path


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


def load_config(config_path):
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config JSON file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_pipeline(torch_dtype):
    """Initialize the FluxPipelineAuction on available device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading pipeline on device: {device}")

    pipeline = FluxPipelineAuction.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        dtype=torch_dtype,
    ).to(device)

    return pipeline


def generate_and_save_image(
    pipeline,
    data_item,
    index,
    output_dir,
    bids,
    num_agents,
    guidance_scale,
    num_inference_steps,
    sample_idx=0,
):
    """
    Generate and save a single image using the auction pipeline.

    Args:
        pipeline: FluxPipelineAuction instance
        data_item: Dictionary containing prompt data
        index: Prompt index
        output_dir: Output directory
        bids: Tuple/list of bids for all agents
        num_agents: Number of agents
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of denoising steps
        sample_idx: Sample index for multiple sampling

    Returns:
        str: Path to saved image, or None if generation failed
    """
    base_prompt = data_item.get("base_prompt", "")

    # Extract agent prompts from data_item
    agent_prompts = []
    for i in range(1, num_agents + 1):
        agent_prompts.append(data_item.get(f"agent{i}_prompt", ""))

    # Create filename with bid info
    bid_str = "_".join([f"b{i+1}_{bid:.2f}" for i, bid in enumerate(bids)])
    filename_base = f"idx{index:03d}_{bid_str}_s{sample_idx:02d}"
    prompt_specific_output_dir = os.path.join(output_dir, f"prompt_{index:03d}")
    os.makedirs(prompt_specific_output_dir, exist_ok=True)
    output_path = os.path.join(prompt_specific_output_dir, f"{filename_base}.png")

    # Check if image already exists and skip if it does
    if os.path.exists(output_path):
        print(f"Skipping existing image: {output_path}")
        return output_path

    # Print generation info
    if all(bid == 0.0 for bid in bids):
        print(
            f"Generating item {index}, sample {sample_idx}: Base prompt only: {base_prompt[:60]}..."
        )
    else:
        bid_info = ", ".join([f"A{i+1}={bid:.2f}" for i, bid in enumerate(bids)])
        print(f"Generating item {index}, sample {sample_idx}: Bids=({bid_info})")
        for i, prompt in enumerate(agent_prompts):
            if prompt and bids[i] > 0.0:
                print(f"  A{i+1}: {prompt[:40]}...")

    try:
        # Call pipeline with lists of agent prompts and bids
        images = pipeline(
            agent_prompts=agent_prompts,
            agent_bids=list(bids),
            base_prompt=base_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
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
    parser = argparse.ArgumentParser(
        description="Generate images using FluxPipelineAuction with configurable agents"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file",
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Extract configuration values
    prompts_path = config.get("prompts_path")
    output_dir = config.get("output_dir")
    num_samples_per_combination = config.get("num_samples_per_combination", 20)
    num_prompts_to_process = config.get("num_prompts_to_process", None)
    guidance_scale = config.get("guidance_scale", 10.0)
    num_inference_steps = config.get("num_inference_steps", 5)
    bidding_combinations = config.get("bidding_combinations", [])
    num_agents = config.get("num_agents")

    # Validate configuration
    if not all(
        [
            num_agents,
            prompts_path,
            output_dir,
            bidding_combinations,
        ]
    ):
        print("Error: Config missing required fields (num_agents, prompts_path, output_dir, bidding_combinations)")
        sys.exit(1)

    # Convert bidding combinations to tuples
    bidding_combinations = [tuple(combo) for combo in bidding_combinations]

    # Determine torch dtype
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    print(f"\n=== {num_agents}-Agent Diffusion Auctions Image Generation ===")
    print(f"Configuration file: {config_path}")

    # Load prompts
    if not os.path.exists(prompts_path):
        print(f"Error: Prompts file not found at {prompts_path}")
        sys.exit(1)

    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    # Calculate total images
    num_items_to_process = (
        num_prompts_to_process if num_prompts_to_process is not None else len(prompts)
    )
    total_images = (
        num_items_to_process
        * len(bidding_combinations)
        * num_samples_per_combination
    )

    print("\nConfiguration:")
    print(f"  - {num_agents} agents")
    print(f"  - {num_items_to_process} prompts")
    print(f"  - {len(bidding_combinations)} bidding combinations")
    print(f"  - {num_samples_per_combination} samples per combination")
    print(f"  - Total images to generate: {total_images}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Guidance scale: {guidance_scale}")
    print(f"  - Inference steps: {num_inference_steps}")

    print("\nBidding combinations:")
    for i, combo in enumerate(bidding_combinations):
        combo_str = ", ".join([f"A{j+1}={bid:.2f}" for j, bid in enumerate(combo)])
        print(f"  {i + 1}: ({combo_str})")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline
    pipeline = load_pipeline(torch_dtype)

    # Generate images
    generation_results = []
    for i, item_data in enumerate(
        tqdm(prompts[:num_items_to_process], desc="Processing Prompts")
    ):
        for bids_tuple in bidding_combinations:
            for sample_idx in range(num_samples_per_combination):
                generated_image_path = generate_and_save_image(
                    pipeline,
                    item_data,
                    i,
                    output_dir,
                    bids_tuple,
                    num_agents,
                    guidance_scale,
                    num_inference_steps,
                    sample_idx,
                )
                if generated_image_path:
                    # Collect agent prompts and bids for results
                    agent_prompts = [
                        item_data.get(f"agent{j+1}_prompt", "")
                        for j in range(num_agents)
                    ]
                    generation_results.append(
                        {
                            "item_index": i,
                            "bids": list(bids_tuple),
                            "sample_index": sample_idx,
                            "agent_prompts": agent_prompts,
                            "base_prompt": item_data["base_prompt"],
                            "image_path": generated_image_path,
                        }
                    )

    # Save results
    results_filename = os.path.join(output_dir, "generation_log.json")
    with open(results_filename, "w") as f:
        json.dump(generation_results, f, indent=2)

    print(f"\nGeneration complete!")
    print(f"  - {len(generation_results)} images generated successfully")
    print(f"  - Results saved to: {results_filename}")


if __name__ == "__main__":
    main()
