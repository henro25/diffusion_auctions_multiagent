"""
Numerical Prompt Baseline Generation

Generates images using the standard FluxPipeline (not the auction pipeline),
with bids encoded as percentages directly in the prompt. Used as a baseline
to compare against our diffusion auction mechanism.

Prompt template: "{base_prompt} that is {b1*100}% aligned with {agent1_prompt}
and {b2*100}% aligned with {agent2_prompt}"

Edge cases:
- (0, 0) -> just base_prompt
- bid_i = 0 -> drop that clause
- (1.0, 0.0) -> "{base_prompt} that is 100% aligned with {agent1_prompt}"

Output naming matches generate_images.py for compatibility with existing alignment scripts.
"""

import os
import json
import sys
import torch
import argparse
from tqdm import tqdm


def setup_hf_cache():
    if "HF_HOME" in os.environ:
        print(f"Using existing HF cache: {os.environ['HF_HOME']}")
        return
    user = os.environ.get("USER", "user")
    for path in [f"/scratch/{user}/hf-cache", f"/tmp/{user}/hf-cache",
                 f"/dev/shm/{user}/hf-cache", f"/var/tmp/{user}/hf-cache"]:
        parent = os.path.dirname(path)
        if os.access(parent, os.W_OK):
            os.makedirs(path, exist_ok=True)
            os.environ["HF_HOME"] = path
            os.environ["HF_HUB_CACHE"] = f"{path}/hub"
            os.environ["TRANSFORMERS_CACHE"] = f"{path}/transformers"
            print(f"Setup HF cache at: {path}")
            return
    print("Warning: no local SSD cache available")


setup_hf_cache()

from diffusers import FluxPipeline


def build_numerical_prompt(base_prompt, agent_prompts, bids):
    """Build a single numerical-baseline prompt from the bid percentages.

    Examples:
      bids=(0, 0)         -> "{base}"
      bids=(1.0, 0.0)     -> "{base} that is 100% aligned with {a1}"
      bids=(0.7, 0.3)     -> "{base} that is 70% aligned with {a1} and 30% aligned with {a2}"
      bids=(0.0, 1.0)     -> "{base} that is 100% aligned with {a2}"
    """
    clauses = []
    for prompt, bid in zip(agent_prompts, bids):
        if bid > 0.0 and prompt:
            clauses.append(f"{int(round(bid * 100))}% aligned with {prompt}")

    if not clauses:
        return base_prompt
    return f"{base_prompt} that is " + " and ".join(clauses)


def load_pipeline(torch_dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading FluxPipeline on device: {device}")
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        dtype=torch_dtype,
    ).to(device)
    return pipeline


def generate_and_save(pipeline, item, prompt_idx, output_dir, bids,
                      guidance_scale, num_inference_steps, sample_idx):
    base_prompt = item.get("base_prompt", "")
    agent_prompts = [item.get("agent1_prompt", ""), item.get("agent2_prompt", "")]

    bid_str = "_".join([f"b{i+1}_{bid:.2f}" for i, bid in enumerate(bids)])
    filename = f"idx{prompt_idx:03d}_{bid_str}_s{sample_idx:02d}.png"
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx:03d}")
    os.makedirs(prompt_dir, exist_ok=True)
    out_path = os.path.join(prompt_dir, filename)

    if os.path.exists(out_path):
        print(f"Skipping existing: {out_path}")
        return out_path

    full_prompt = build_numerical_prompt(base_prompt, agent_prompts, bids)
    print(f"  prompt={full_prompt!r}")

    try:
        result = pipeline(
            prompt=full_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        if result and hasattr(result, "images") and len(result.images) > 0:
            result.images[0].save(out_path)
            print(f"Saved: {out_path}")
            return out_path
        print(f"No image generated for {filename}")
        return None
    except Exception as e:
        print(f"Error generating {filename}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Numerical-prompt baseline generation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--prompt_index", type=int, default=None)
    parser.add_argument("--bid_chunk", type=int, default=None)
    parser.add_argument("--num_chunks", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: config not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = json.load(f)

    prompts_path = config["prompts_path"]
    output_dir = config["output_dir"]
    num_samples = config.get("num_samples_per_combination", 20)
    guidance_scale = config.get("guidance_scale", 10.0)
    num_inference_steps = config.get("num_inference_steps", 5)
    bidding_combinations = [tuple(c) for c in config["bidding_combinations"]]

    # Optional bid chunking
    if args.bid_chunk is not None and args.num_chunks is not None:
        total = len(bidding_combinations)
        chunk_size = (total + args.num_chunks - 1) // args.num_chunks
        start = args.bid_chunk * chunk_size
        end = min(start + chunk_size, total)
        bidding_combinations = bidding_combinations[start:end]
        print(f"Bid chunk {args.bid_chunk}/{args.num_chunks}: combos {start}-{end-1}")

    if not os.path.exists(prompts_path):
        print(f"Error: prompts not found: {prompts_path}")
        sys.exit(1)
    with open(prompts_path) as f:
        prompts = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Numerical Baseline Generation ===")
    print(f"Config: {args.config}")
    print(f"Prompts: {len(prompts)}, Combos: {len(bidding_combinations)}, Samples: {num_samples}")
    print(f"Output: {output_dir}")

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    pipeline = load_pipeline(torch_dtype)

    # Filter to single prompt if requested
    prompt_items = list(enumerate(prompts))
    if args.prompt_index is not None:
        prompt_items = [(i, p) for i, p in prompt_items if i == args.prompt_index]
        if not prompt_items:
            print(f"Error: prompt_index {args.prompt_index} not found")
            sys.exit(1)

    log = []
    for i, item in tqdm(prompt_items, desc="Prompts"):
        for bids in bidding_combinations:
            for s in range(num_samples):
                path = generate_and_save(
                    pipeline, item, i, output_dir, bids,
                    guidance_scale, num_inference_steps, s,
                )
                if path:
                    log.append({
                        "item_index": i,
                        "bids": list(bids),
                        "sample_index": s,
                        "agent_prompts": [item.get("agent1_prompt", ""), item.get("agent2_prompt", "")],
                        "base_prompt": item.get("base_prompt", ""),
                        "numerical_prompt": build_numerical_prompt(
                            item.get("base_prompt", ""),
                            [item.get("agent1_prompt", ""), item.get("agent2_prompt", "")],
                            bids,
                        ),
                        "image_path": path,
                    })

    log_path = os.path.join(output_dir, "generation_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. {len(log)} images logged at {log_path}")


if __name__ == "__main__":
    main()
