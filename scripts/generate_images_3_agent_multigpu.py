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
PROMPTS_PATH = "../prompts/prompts_3_agent.json"  # Path to prompts file
OUTPUT_DIR = "/datastor1/gdaras/diffusion_auctions_multiagent/images/images_3_agent_multigpu"  # Output directory for generated images

# Multi-GPU configuration
USE_MULTI_GPU = True  # Set to False to use single GPU like original script
NUM_GPUS = None  # None = auto-detect, or specify number (e.g., 4)
GPU_INDICES = None  # None = auto-detect, or specify list of GPU indices (e.g., [0, 1, 3] or [1, 2])

# Sampling configuration
NUM_SAMPLES_PER_COMBINATION = 20  # Number of times to sample each prompt-bid combination
NUM_PROMPTS_TO_PROCESS = None  # Number of prompts to process (None = all prompts)

# Generation parameters
GUIDANCE_SCALE = 10.0  # Guidance scale for generation
NUM_INFERENCE_STEPS = 5  # Number of denoising steps
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16

# Bidding combinations for 3 agents (b1, b2, b3)
BIDDING_COMBINATIONS_3_AGENT = [
    (0.0, 0.0, 0.0),  # Base prompt only (no agent influence)
    (1.0, 0.0, 0.0),  # Agent 1 dominant
    # (0.0, 1.0, 0.0),  # Agent 2 dominant
    # (0.0, 0.0, 1.0),  # Agent 3 dominant
    # (0.5, 0.5, 0.0),  # Agent 1 & 2 equal, Agent 3 none
    (0.33, 0.33, 0.33),  # All equal
    # (0.45, 0.45, 0.1),  # A1 & A2 strong and equal, A3 minor voice
    (0.4, 0.4, 0.2),  # A1 & A2 strong and equal, A3 minor voice
    # (0.45, 0.1, 0.45),  # A1 & A3 strong and equal, A2 minor voice
    # (0.1, 0.45, 0.45),  # A2 & A3 strong and equal, A1 minor voice
    # (0.4, 0.3, 0.3),  # A1 slightly more influential than A2 & A3 (who are equal)
    # (0.3, 0.4, 0.3),  # A2 slightly more influential
    # (0.3, 0.3, 0.4),  # A3 slightly more influential
    # (0.7, 0.2, 0.1),  # Strong A1, moderate A2, weak A3
    # (0.1, 0.7, 0.2),  # Strong A2, moderate A3, weak A1
    # (0.2, 0.1, 0.7),  # Strong A3, moderate A1, weak A2
    # (0.5, 0.3, 0.2),  # A1 > A2 > A3 with smaller gaps
    # (0.2, 0.5, 0.3),  # A2 > A3 > A1 with smaller gaps
    # (0.3, 0.2, 0.5),  # A3 > A1 > A2 with smaller gaps
    (0.6, 0.3, 0.1),  # Agent 1 > Agent 2 > Agent 3
    # (0.1, 0.6, 0.3),  # Agent 2 > Agent 3 > Agent 1
    # (0.1, 0.3, 0.6),  # Agent 3 > Agent 2 > Agent 1
    (0.6, 0.2, 0.2),  # Agent 1 > Agent 2 = Agent 3
]

# ===== END CONFIGURATION SECTION =====


def run_single_gpu_generation(prompts, output_dir):
    """Original single-GPU generation approach"""
    print("Running single-GPU generation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe_auction = FluxPipelineAuction.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=TORCH_DTYPE,
    ).to(device)

    def generate_and_save_flux_image_3_agents(
        pipe_auction, data_item, index, output_dir, bids, sample_idx=0
    ):
        agent1_prompt = data_item.get("agent1_prompt", "")
        agent2_prompt = data_item.get("agent2_prompt", "")
        agent3_prompt = data_item.get("agent3_prompt", "")
        base_prompt = data_item.get("base_prompt", "")

        bid1, bid2, bid3 = bids

        filename_base = f"idx{index:03d}_b1_{bid1:.2f}_b2_{bid2:.2f}_b3_{bid3:.2f}_s{sample_idx:02d}"
        prompt_specific_output_dir = os.path.join(output_dir, f"prompt_{index:03d}")
        os.makedirs(prompt_specific_output_dir, exist_ok=True)
        output_path = os.path.join(prompt_specific_output_dir, f"{filename_base}.png")

        # Check if image already exists and skip if it does
        if os.path.exists(output_path):
            print(f"Skipping existing image: {output_path}")
            return output_path

        print(
            f"Generating item {index}, sample {sample_idx}: Bids=({bid1:.2f}, {bid2:.2f}, {bid3:.2f})"
        )

        # For base prompt only (0, 0, 0), show different info
        if bid1 == 0.0 and bid2 == 0.0 and bid3 == 0.0:
            print(f"  Base prompt only: {base_prompt[:60]}...")
        else:
            print(
                f"  A1: {agent1_prompt[:30]}... | A2: {agent2_prompt[:30]}... | A3: {agent3_prompt[:30]}..."
            )

        try:
            images = pipe_auction(
                agent1_prompt=agent1_prompt,
                agent1_bid=bid1,
                agent2_prompt=agent2_prompt,
                agent2_bid=bid2,
                agent3_prompt=agent3_prompt,
                agent3_bid=bid3,
                base_prompt=base_prompt,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
            )

            if images and hasattr(images, "images") and len(images.images) > 0:
                images.images[0].save(output_path)
                print(f"Saved image to {output_path}")
            else:
                print(f"No image generated for {output_path}")
                output_path = None

        except Exception as e:
            print(f"Error generating image for {filename_base}: {e}")
            output_path = None

        return output_path

    # Generation loop
    generation_results = []
    num_items_to_process = (
        NUM_PROMPTS_TO_PROCESS if NUM_PROMPTS_TO_PROCESS is not None else len(prompts)
    )

    for i, item_data in enumerate(
        tqdm(prompts[:num_items_to_process], desc="Processing Prompts")
    ):
        for bids_tuple in BIDDING_COMBINATIONS_3_AGENT:
            for sample_idx in range(NUM_SAMPLES_PER_COMBINATION):
                generated_image_path = generate_and_save_flux_image_3_agents(
                    pipe_auction, item_data, i, output_dir, bids_tuple, sample_idx
                )
                if generated_image_path:
                    generation_results.append(
                        {
                            "item_index": i,
                            "bids": bids_tuple,
                            "sample_index": sample_idx,
                            "agent1_prompt": item_data["agent1_prompt"],
                            "agent2_prompt": item_data["agent2_prompt"],
                            "agent3_prompt": item_data["agent3_prompt"],
                            "base_prompt": item_data["base_prompt"],
                            "image_path": generated_image_path,
                        }
                    )

    return generation_results


def run_multi_gpu_generation(prompts, output_dir):
    """Multi-GPU generation approach"""
    print("Running multi-GPU generation...")

    try:
        from multi_gpu_config import (
            MultiGPUManager,
            setup_multi_gpu_environment,
            get_gpu_memory_info,
        )
    except ImportError:
        print("Error: multi_gpu_config.py not found. Falling back to single GPU.")
        return run_single_gpu_generation(prompts, output_dir)

    # Setup multi-GPU environment
    setup_multi_gpu_environment()

    # Display GPU info
    gpu_info = get_gpu_memory_info()
    print("Available GPUs:")
    for info in gpu_info:
        print(
            f"  GPU {info['gpu_id']}: {info['name']} ({info['free_memory_gb']:.1f}GB free)"
        )

    # Initialize multi-GPU manager
    manager = MultiGPUManager(
        num_gpus=NUM_GPUS, gpu_indices=GPU_INDICES, torch_dtype=TORCH_DTYPE
    )
    manager.initialize_pipelines()

    # Run parallel generation
    generation_results = manager.generate_images_parallel(
        prompts=prompts,
        bidding_combinations=BIDDING_COMBINATIONS_3_AGENT,
        num_samples_per_combination=NUM_SAMPLES_PER_COMBINATION,
        output_dir=output_dir,
        num_prompts_to_process=NUM_PROMPTS_TO_PROCESS,
    )

    return generation_results


def main():
    """Main execution function"""
    print("=== Multi-GPU Diffusion Auctions Image Generation ===")

    # Load prompts
    with open(PROMPTS_PATH, "r") as f:
        prompts = json.load(f)

    # Calculate total images
    num_items_to_process = (
        NUM_PROMPTS_TO_PROCESS if NUM_PROMPTS_TO_PROCESS is not None else len(prompts)
    )
    total_images = (
        num_items_to_process
        * len(BIDDING_COMBINATIONS_3_AGENT)
        * NUM_SAMPLES_PER_COMBINATION
    )

    print(f"Configuration:")
    print(f"  - {num_items_to_process} prompts")
    print(f"  - {len(BIDDING_COMBINATIONS_3_AGENT)} bidding combinations")
    print(f"  - {NUM_SAMPLES_PER_COMBINATION} samples per combination")
    print(f"  - Total images to generate: {total_images}")
    print(f"  - Multi-GPU enabled: {USE_MULTI_GPU}")
    print(f"  - Output directory: {OUTPUT_DIR}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run generation
    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        generation_results = run_multi_gpu_generation(prompts, OUTPUT_DIR)
    else:
        if USE_MULTI_GPU:
            print(
                "Warning: Multi-GPU requested but only 1 GPU available. Using single GPU."
            )
        generation_results = run_single_gpu_generation(prompts, OUTPUT_DIR)

    # Save results
    results_filename = os.path.join(OUTPUT_DIR, "generation_log.json")
    with open(results_filename, "w") as f:
        json.dump(generation_results, f, indent=2)

    print(f"\nGeneration complete!")
    print(f"  - {len(generation_results)} images generated successfully")
    print(f"  - Results saved to: {results_filename}")


if __name__ == "__main__":
    main()
