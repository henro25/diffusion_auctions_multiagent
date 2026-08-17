#!/usr/bin/env python3
"""Aesthetic quality analysis for multi-agent diffusion auction results.

Processes ONE experiment (one quality directory + one config) per invocation
and produces:

  1. quality_over_bids.png
       Bar plot of the mean LAION aesthetic score per bid combination
       (with standard-error bars).
  2. quality_over_bids.csv / data.csv
       Aggregated per-bid-combination quality statistics.

Inputs
------
--quality_dir : directory containing prompt_NNN/ subdirectories of per-image
    LAION aesthetic score JSONs named quality_pNNN_b{bid1}_{bid2}..._sSS.json
    with schema:
      {"metadata": {"prompt_index", "bids", "sample_index"},
       "quality_assessment": {"aesthetic_score": <float>}}
--config : the quality config JSON used to produce the data. Must contain
    "num_agents" and "bidding_combinations"; "num_samples_per_combination"
    is optional (default 20).
--output_dir : directory to write the plot and CSVs (created if missing).

Example
-------
    python analyze_quality.py \
        --quality_dir path/to/quality_laion_3_agents \
        --config path/to/quality_laion_config_3_agents.json \
        --output_dir path/to/results/3_agents

Only prompts with complete data (every bid combination x every sample present)
are included, so that all statistics are computed over the same prompt set.
"""

import argparse
import json
import os
import re
import sys
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def construct_quality_filename(prompt_idx: int, bids: List[float],
                               sample_idx: int) -> str:
    """Construct the quality JSON filename for one image."""
    bid_str = "_".join(f"{bid:.2f}" for bid in bids)
    return f"quality_p{prompt_idx:03d}_b{bid_str}_s{sample_idx:02d}.json"


def find_prompt_indices(quality_dir: str) -> List[int]:
    """Find prompt indices from prompt_NNN subdirectories."""
    indices = []
    for name in sorted(os.listdir(quality_dir)):
        match = re.fullmatch(r"prompt_(\d+)", name)
        if match and os.path.isdir(os.path.join(quality_dir, name)):
            indices.append(int(match.group(1)))
    return indices


def load_quality_data(quality_dir: str, num_agents: int,
                      bidding_combinations: List[List[float]],
                      num_samples: int) -> pd.DataFrame:
    """Load quality JSONs into a DataFrame.

    Only prompts with complete data (all bid combinations x all samples) are
    kept so every aggregate is computed over the same prompt set.
    """
    all_rows = []
    complete_prompts = 0
    prompt_indices = find_prompt_indices(quality_dir)

    if not prompt_indices:
        raise FileNotFoundError(
            f"No prompt_NNN subdirectories found in {quality_dir}")

    for prompt_idx in prompt_indices:
        prompt_dir = os.path.join(quality_dir, f"prompt_{prompt_idx:03d}")
        prompt_rows = []
        is_complete = True

        for bids in bidding_combinations:
            for sample_idx in range(num_samples):
                filepath = os.path.join(
                    prompt_dir,
                    construct_quality_filename(prompt_idx, bids, sample_idx))
                if not os.path.exists(filepath):
                    is_complete = False
                    break
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    row = {
                        "prompt": data["metadata"]["prompt_index"],
                        "sample": data["metadata"]["sample_index"],
                        "aesthetic_score":
                            data["quality_assessment"]["aesthetic_score"],
                        "is_base_only": all(b == 0 for b in bids),
                    }
                    for i, bid in enumerate(bids):
                        row[f"bid_{i + 1}"] = bid
                    prompt_rows.append(row)
                except (json.JSONDecodeError, KeyError) as exc:
                    print(f"Warning: could not parse {filepath}: {exc}",
                          file=sys.stderr)
                    is_complete = False
                    break
            if not is_complete:
                break

        expected = len(bidding_combinations) * num_samples
        if is_complete and len(prompt_rows) == expected:
            all_rows.extend(prompt_rows)
            complete_prompts += 1
        else:
            print(f"Skipping prompt {prompt_idx} (incomplete data)",
                  file=sys.stderr)

    print(f"Loaded {complete_prompts} complete prompts "
          f"({len(all_rows)} rows) from {quality_dir}")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------

def plot_quality_over_bids(df: pd.DataFrame, num_agents: int,
                           output_dir: str) -> pd.DataFrame:
    """Bar plot: mean aesthetic score per bid combination."""
    df = df.copy()
    df["bid_combo"] = df.apply(
        lambda r: "/".join(f"{r[f'bid_{i + 1}']:.2f}"
                           for i in range(num_agents)), axis=1)

    stats = df.groupby("bid_combo")["aesthetic_score"].agg(
        ["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.set_facecolor("white")
    ax.bar(range(len(stats)), stats["mean"], yerr=stats["se"], capsize=4,
           alpha=0.8, color="steelblue", edgecolor="navy", linewidth=1.2)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats["bid_combo"], rotation=45, ha="right",
                       fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel(f"Bid Combination ({num_agents} agents)", fontsize=14)
    ax.set_ylabel("Mean LAION Aesthetic Score ± SE", fontsize=14)
    ax.set_title(f"{num_agents}-Agent: LAION Aesthetic Quality by Bid "
                 f"Combination", fontsize=15, pad=20)
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "quality_over_bids.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    csv_path = os.path.join(output_dir, "quality_over_bids.csv")
    stats[["bid_combo", "mean", "std", "se", "count"]].to_csv(
        csv_path, index=False)
    print(f"Saved: {csv_path}")
    return stats


def print_summary(df: pd.DataFrame) -> None:
    """Print overall and base-only vs with-agents quality statistics."""
    print("\nSummary statistics:")
    print(f"  Overall mean aesthetic score: "
          f"{df['aesthetic_score'].mean():.3f}")
    print(f"  Overall std aesthetic score:  "
          f"{df['aesthetic_score'].std():.3f}")

    base_only = df[df["is_base_only"]]["aesthetic_score"]
    with_agents = df[~df["is_base_only"]]["aesthetic_score"]
    if len(base_only) > 0:
        print(f"  Base prompt only mean:        {base_only.mean():.3f}")
    if len(with_agents) > 0:
        print(f"  With agent prompts mean:      {with_agents.mean():.3f}")
        if len(base_only) > 0:
            print(f"  Difference (agents - base):   "
                  f"{with_agents.mean() - base_only.mean():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LAION aesthetic quality scores from multi-agent "
                    "diffusion auction results. Produces a quality-vs-bid-"
                    "combination bar plot and CSV export for a single "
                    "experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--quality_dir", required=True,
        help="Directory containing prompt_NNN/ subdirectories of LAION "
             "aesthetic score JSON files.")
    parser.add_argument(
        "--config", required=True,
        help="Quality config JSON used to produce the data (must contain "
             "num_agents and bidding_combinations).")
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write the plot and CSV files (created if "
             "missing).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    num_agents = config["num_agents"]
    bidding_combinations = config["bidding_combinations"]
    num_samples = config.get("num_samples_per_combination", 20)

    for combo in bidding_combinations:
        if len(combo) != num_agents:
            raise ValueError(
                f"Bid combination {combo} does not match num_agents="
                f"{num_agents}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_quality_data(args.quality_dir, num_agents,
                           bidding_combinations, num_samples)
    if df.empty:
        print("No complete data found; nothing to analyze.", file=sys.stderr)
        sys.exit(1)

    stats = plot_quality_over_bids(df, num_agents, args.output_dir)

    data_csv = os.path.join(args.output_dir, "data.csv")
    stats[["bid_combo", "mean", "std", "se", "count"]].to_csv(
        data_csv, index=False)
    print(f"Saved: {data_csv}")

    print_summary(df)


if __name__ == "__main__":
    main()
