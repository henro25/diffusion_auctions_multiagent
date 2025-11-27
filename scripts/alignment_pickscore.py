"""
PickScore Alignment Analysis for Multi-Agent Diffusion Auctions
"""

import os
import sys
import json
from typing import Dict, List, Optional
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_hf_cache():
    if "HF_HOME" in os.environ:
        print(f"Using existing HF cache: {os.environ['HF_HOME']}")
        return
    user = os.environ.get("USER", os.environ.get("USERNAME", "user"))
    cache_paths = [f"/scratch/{user}/hf-cache", f"/tmp/{user}/hf-cache", f"/dev/shm/{user}/hf-cache", f"/var/tmp/{user}/hf-cache"]
    cache_dir = None
    for path in cache_paths:
        parent = os.path.dirname(path)
        if os.access(parent, os.W_OK):
            cache_dir = path
            break
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        print(f"Setup HF cache at: {cache_dir}")


setup_hf_cache()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_NAME = "yuvalkirstain/PickScore_v1"


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_prompts(prompts_path: str) -> List[Dict]:
    with open(prompts_path, "r") as f:
        return json.load(f)


def get_pickscore_alignment(image_path: str, text: str, processor, model) -> Optional[float]:
    try:
        image = Image.open(image_path).convert("RGB")
        image_inputs = processor(images=[image], padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
        text_inputs = processor(text=[text], padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            score = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            return float(score.item())
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def construct_image_filename(prompt_idx: int, bids: List[float], sample_idx: int, num_agents: int) -> str:
    bid_str = "_".join([f"b{i+1}_{bid:.2f}" for i, bid in enumerate(bids)])
    return f"idx{prompt_idx:03d}_{bid_str}_s{sample_idx:02d}.png"


def construct_alignment_filename(prompt_idx: int, bids: List[float], sample_idx: int) -> str:
    bid_str = "_".join([f"{bid:.2f}" for bid in bids])
    return f"alignment_p{prompt_idx:03d}_b{bid_str}_s{sample_idx:02d}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    num_agents = config.get("num_agents")
    prompts_path = config.get("prompts_path")
    images_dir = config.get("images_dir")
    output_dir = config.get("output_dir")
    num_samples_per_combination = config.get("num_samples_per_combination", 20)
    num_prompts_to_process = config.get("num_prompts_to_process", None)
    process_prompts_forward = config.get("process_prompts_forward", True)
    bidding_combinations = [list(combo) for combo in config.get("bidding_combinations", [])]

    print(f"\n=== {num_agents}-Agent PickScore Alignment Analysis ===")

    prompts = load_prompts(prompts_path)
    num_items_to_process = num_prompts_to_process if num_prompts_to_process is not None else len(prompts)

    os.makedirs(output_dir, exist_ok=True)

    print("Loading PickScore model...")
    processor = AutoProcessor.from_pretrained(PROCESSOR_NAME, cache_dir=CACHE_DIR)
    model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).eval().to(DEVICE)
    print("PickScore model loaded")

    processed_count = 0
    skipped_missing = 0
    skipped_existing = 0
    total_expected = num_items_to_process * len(bidding_combinations) * num_samples_per_combination

    prompt_indices = range(num_items_to_process) if process_prompts_forward else range(num_items_to_process - 1, -1, -1)

    with tqdm(total=total_expected, desc="Processing images") as pbar:
        for prompt_idx in prompt_indices:
            if prompt_idx >= len(prompts):
                pbar.update(len(bidding_combinations) * num_samples_per_combination)
                continue

            prompt_data = prompts[prompt_idx]
            prompt_output_dir = os.path.join(output_dir, f"prompt_{prompt_idx:03d}")
            os.makedirs(prompt_output_dir, exist_ok=True)

            for bids in bidding_combinations:
                for sample_idx in range(num_samples_per_combination):
                    image_filename = construct_image_filename(prompt_idx, bids, sample_idx, num_agents)
                    image_path = os.path.join(images_dir, f"prompt_{prompt_idx:03d}", image_filename)
                    alignment_filename = construct_alignment_filename(prompt_idx, bids, sample_idx)
                    output_path = os.path.join(prompt_output_dir, alignment_filename)

                    if os.path.exists(output_path):
                        skipped_existing += 1
                        pbar.update(1)
                        continue

                    if not os.path.exists(image_path):
                        skipped_missing += 1
                        pbar.update(1)
                        continue

                    base_score = get_pickscore_alignment(image_path, prompt_data["base_prompt"], processor, model)
                    if base_score is None:
                        pbar.update(1)
                        continue

                    agent_scores = {}
                    agent_prompts_dict = {}
                    for i in range(1, num_agents + 1):
                        agent_key = f"agent{i}_prompt"
                        agent_prompts_dict[agent_key] = prompt_data.get(agent_key, "")
                        score = get_pickscore_alignment(image_path, prompt_data.get(agent_key, ""), processor, model)
                        agent_scores[f"agent{i}_alignment"] = score if score is not None else 0.0

                    total_welfare = sum(agent_scores[f"agent{i+1}_alignment"] * bids[i] for i in range(num_agents))
                    weighted_alignment = total_welfare / sum(bids) if sum(bids) > 0 else 0.0

                    result = {
                        "metadata": {"prompt_index": prompt_idx, "bids": bids, "sample_index": sample_idx, "image_path": image_path, "filename": os.path.basename(image_path)},
                        "prompts": {"base_prompt": prompt_data["base_prompt"], **agent_prompts_dict},
                        "alignment_scores": {"base_alignment": base_score, **agent_scores},
                        "welfare_metrics": {"weighted_alignment": weighted_alignment, "total_welfare": total_welfare},
                    }
                    with open(output_path, "w") as f:
                        json.dump(result, f, indent=2)
                    processed_count += 1
                    pbar.update(1)

    print(f"\n=== Complete === Processed: {processed_count}, Skipped existing: {skipped_existing}, Skipped missing: {skipped_missing}")
    del model
    del processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
