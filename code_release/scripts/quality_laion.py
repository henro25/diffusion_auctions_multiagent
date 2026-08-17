"""
LAION Aesthetic Quality Assessment for Multi-Agent Diffusion Auctions
"""

import os
import sys
import json
from typing import Dict, List, Optional
import argparse
from os.path import expanduser

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import open_clip
import requests

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


def get_aesthetic_model(clip_model="vit_l_14"):
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_" + clip_model + "_linear.pth"
    
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_" + clip_model + "_linear.pth?raw=true"
        print(f"Downloading aesthetic model to {path_to_model}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url_model, headers=headers, timeout=60)
        response.raise_for_status()
        with open(path_to_model, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    
    m = nn.Linear(768, 1) if clip_model == "vit_l_14" else nn.Linear(512, 1)
    s = torch.load(path_to_model, map_location=DEVICE)
    m.load_state_dict(s)
    m.eval()
    return m


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_prompts(prompts_path: str) -> List[Dict]:
    with open(prompts_path, "r") as f:
        return json.load(f)


def calculate_aesthetic_score(image_path: str, clip_model, preprocess, aesthetic_model) -> Optional[float]:
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            if DEVICE.type == "cuda":
                image_features = image_features.float()
            prediction = aesthetic_model(image_features)
            return float(prediction.item())
    except Exception as e:
        print(f"Error calculating aesthetic score for {image_path}: {e}")
        return None


def construct_image_filename(prompt_idx: int, bids: List[float], sample_idx: int, num_agents: int) -> str:
    bid_str = "_".join([f"b{i+1}_{bid:.2f}" for i, bid in enumerate(bids)])
    return f"idx{prompt_idx:03d}_{bid_str}_s{sample_idx:02d}.png"


def construct_quality_filename(prompt_idx: int, bids: List[float], sample_idx: int) -> str:
    bid_str = "_".join([f"{bid:.2f}" for bid in bids])
    return f"quality_p{prompt_idx:03d}_b{bid_str}_s{sample_idx:02d}.json"


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

    print(f"\n=== {num_agents}-Agent LAION Aesthetic Quality Assessment ===")

    prompts = load_prompts(prompts_path)
    num_items_to_process = num_prompts_to_process if num_prompts_to_process is not None else len(prompts)

    os.makedirs(output_dir, exist_ok=True)

    print("Loading CLIP model and aesthetic predictor...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()
    aesthetic_model = get_aesthetic_model(clip_model="vit_l_14").to(DEVICE)
    aesthetic_model.eval()
    print("Models loaded")

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
                    quality_filename = construct_quality_filename(prompt_idx, bids, sample_idx)
                    output_path = os.path.join(prompt_output_dir, quality_filename)

                    if os.path.exists(output_path):
                        skipped_existing += 1
                        pbar.update(1)
                        continue

                    if not os.path.exists(image_path):
                        skipped_missing += 1
                        pbar.update(1)
                        continue

                    aesthetic_score = calculate_aesthetic_score(image_path, clip_model, preprocess, aesthetic_model)
                    if aesthetic_score is not None:
                        agent_prompts_dict = {f"agent{i}_prompt": prompt_data.get(f"agent{i}_prompt", "") for i in range(1, num_agents + 1)}
                        result = {
                            "metadata": {"prompt_index": prompt_idx, "bids": bids, "sample_index": sample_idx, "image_path": image_path, "filename": os.path.basename(image_path)},
                            "prompts": {"base_prompt": prompt_data["base_prompt"], **agent_prompts_dict},
                            "quality_assessment": {"aesthetic_score": aesthetic_score},
                        }
                        with open(output_path, "w") as f:
                            json.dump(result, f, indent=2)
                        processed_count += 1
                    pbar.update(1)

    print(f"\n=== Complete === Processed: {processed_count}, Skipped existing: {skipped_existing}, Skipped missing: {skipped_missing}")
    del clip_model
    del aesthetic_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
