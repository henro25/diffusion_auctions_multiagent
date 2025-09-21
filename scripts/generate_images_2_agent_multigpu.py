"""
2-Agent Multi-GPU Image Generation Script

This script generates images using the FluxPipelineAuction with 2 agents across multiple GPUs.
Sweeps across bid values: b=0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 with the other agent getting 1-b.
For single-GPU generation, use generate_images_2_agent.py instead.

Authors: Lillian Sun, Warren Zhu, Henry Huang
Academic Context: 4th Year research project on multi-winner auctions for generative AI
"""

import os
import json
import torch
from tqdm import tqdm

# Import the FluxPipelineAuction class from pipelines module
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipelines import FluxPipelineAuction

# ===== CONFIGURATION SECTION =====
# Modify these parameters as needed

# Path configurations
PROMPTS_PATH = "../prompts/prompts_2_agent.json"  # Path to prompts file
OUTPUT_DIR = "/datastor1/gdaras/diffusion_auctions_multiagent/images/images_2_agent_multigpu"  # Output directory for generated images

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

# Bidding sweep values for 2 agents
# Agent 1 gets b, Agent 2 gets (1-b), Agent 3 gets 0.0
SWEEP_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Generate bidding combinations automatically
BIDDING_COMBINATIONS_2_AGENT = []
for b in SWEEP_VALUES:
    BIDDING_COMBINATIONS_2_AGENT.append((b, 1.0 - b, 0.0))

# ===== END CONFIGURATION SECTION =====


def run_single_gpu_generation(prompts, output_dir):
    """Original single-GPU generation approach"""
    print("Running single-GPU generation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe_auction = FluxPipelineAuction.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=TORCH_DTYPE,
    ).to(device)

    def generate_and_save_flux_image_2_agents(
        pipe_auction, data_item, index, output_dir, bids, sample_idx=0
    ):
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

        try:
            images = pipe_auction(
                agent1_prompt=agent1_prompt,
                agent1_bid=bid1,
                agent2_prompt=agent2_prompt,
                agent2_bid=bid2,
                agent3_prompt="",  # Empty for 2-agent scenario
                agent3_bid=bid3,   # Always 0.0 for 2-agent scenario
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
        for bids_tuple in BIDDING_COMBINATIONS_2_AGENT:
            for sample_idx in range(NUM_SAMPLES_PER_COMBINATION):
                generated_image_path = generate_and_save_flux_image_2_agents(
                    pipe_auction, item_data, i, output_dir, bids_tuple, sample_idx
                )
                if generated_image_path:
                    generation_results.append(
                        {
                            "item_index": i,
                            "bids": bids_tuple[:2],  # Only include agent1 and agent2 bids
                            "sample_index": sample_idx,
                            "agent1_prompt": item_data["agent1_prompt"],
                            "agent2_prompt": item_data["agent2_prompt"],
                            "base_prompt": item_data["base_prompt"],
                            "image_path": generated_image_path,
                        }
                    )

    return generation_results


def run_multi_gpu_generation(prompts, output_dir):
    """Multi-GPU generation approach"""
    print("Running multi-GPU generation...")

    from multi_gpu_config import MultiGPUConfig

    # Initialize multi-GPU configuration
    gpu_config = MultiGPUConfig(num_gpus=NUM_GPUS, gpu_indices=GPU_INDICES)

    def generate_and_save_flux_image_2_agents_multigpu(
        gpu_id, pipe_auction, data_item, index, output_dir, bids, sample_idx=0
    ):
        agent1_prompt = data_item.get("agent1_prompt", "")
        agent2_prompt = data_item.get("agent2_prompt", "")
        base_prompt = data_item.get("base_prompt", "")

        bid1, bid2, bid3 = bids  # bid3 should always be 0.0

        filename_base = f"idx{index:03d}_b1_{bid1:.2f}_b2_{bid2:.2f}_s{sample_idx:02d}"
        prompt_specific_output_dir = os.path.join(output_dir, f"prompt_{index:03d}")
        os.makedirs(prompt_specific_output_dir, exist_ok=True)
        output_path = os.path.join(prompt_specific_output_dir, f"{filename_base}.png")

        print(
            f"[GPU {gpu_id}] Generating item {index}, sample {sample_idx}: Bids=({bid1:.2f}, {bid2:.2f})"
        )

        try:
            images = pipe_auction(
                agent1_prompt=agent1_prompt,
                agent1_bid=bid1,
                agent2_prompt=agent2_prompt,
                agent2_bid=bid2,
                agent3_prompt="",  # Empty for 2-agent scenario
                agent3_bid=bid3,   # Always 0.0 for 2-agent scenario
                base_prompt=base_prompt,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
            )

            if images and hasattr(images, "images") and len(images.images) > 0:
                images.images[0].save(output_path)
                print(f"[GPU {gpu_id}] Saved image to {output_path}")
            else:
                print(f"[GPU {gpu_id}] No image generated for {output_path}")
                output_path = None

        except Exception as e:
            print(f"[GPU {gpu_id}] Error generating image for {filename_base}: {e}")
            output_path = None

        return output_path

    # Create task list
    tasks = []
    num_items_to_process = (
        NUM_PROMPTS_TO_PROCESS if NUM_PROMPTS_TO_PROCESS is not None else len(prompts)
    )

    for i, item_data in enumerate(prompts[:num_items_to_process]):
        for bids_tuple in BIDDING_COMBINATIONS_2_AGENT:
            for sample_idx in range(NUM_SAMPLES_PER_COMBINATION):
                tasks.append((item_data, i, bids_tuple, sample_idx))

    print(f"Total tasks: {len(tasks)}")
    print(f"Available GPUs: {gpu_config.available_gpus}")

    # Load pipelines on each GPU
    pipelines = {}
    for gpu_id in gpu_config.available_gpus:
        print(f"Loading pipeline on GPU {gpu_id}...")
        device = torch.device(f"cuda:{gpu_id}")
        pipeline = FluxPipelineAuction.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=TORCH_DTYPE,
        ).to(device)
        pipelines[gpu_id] = pipeline

    # Distribute tasks across GPUs
    generation_results = []
    for task_idx, (item_data, i, bids_tuple, sample_idx) in enumerate(tqdm(tasks, desc="Processing Tasks")):
        # Select GPU for this task
        gpu_id = gpu_config.available_gpus[task_idx % len(gpu_config.available_gpus)]

        # Generate image
        generated_image_path = generate_and_save_flux_image_2_agents_multigpu(
            gpu_id, pipelines[gpu_id], item_data, i, output_dir, bids_tuple, sample_idx
        )

        if generated_image_path:
            generation_results.append(
                {
                    "item_index": i,
                    "bids": bids_tuple[:2],  # Only include agent1 and agent2 bids
                    "sample_index": sample_idx,
                    "agent1_prompt": item_data["agent1_prompt"],
                    "agent2_prompt": item_data["agent2_prompt"],
                    "base_prompt": item_data["base_prompt"],
                    "image_path": generated_image_path,
                    "gpu_id": gpu_id,
                }
            )

    return generation_results


def main():
    """Main execution function."""
    print("=== 2-Agent Multi-GPU Diffusion Auctions Image Generation ===")
    print(f"Bid sweep values: {SWEEP_VALUES}")
    print("Generated bid combinations:")
    for i, combo in enumerate(BIDDING_COMBINATIONS_2_AGENT):
        print(f"  {i+1}: Agent1={combo[0]:.1f}, Agent2={combo[1]:.1f}")

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
    print(f"  - Multi-GPU enabled: {USE_MULTI_GPU}")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run generation
    if USE_MULTI_GPU and torch.cuda.device_count() > 1:
        generation_results = run_multi_gpu_generation(prompts, OUTPUT_DIR)
    else:
        print("Falling back to single-GPU generation...")
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