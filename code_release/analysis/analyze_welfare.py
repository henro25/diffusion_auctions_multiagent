#!/usr/bin/env python3
"""Welfare analysis for multi-agent diffusion auction alignment results.

Processes ONE experiment (one alignment directory + one config) per invocation
and produces:

  1. welfare_over_bids.png / .csv
       Mean welfare improvement of the best-of-k image per bid combination,
       relative to the single-winner baseline (the best welfare achievable by
       any single-agent-monopoly image, evaluated under the same bids).
  2. welfare_over_k.png / .csv
       Average welfare improvement as a function of the number of Monte Carlo
       samples k used per bid combination.
  3. bid_monotonicity.png / .csv
       Each agent's mean alignment score as a function of its own bid.
  4. data.csv
       Aggregated per-bid-combination welfare statistics.

Inputs
------
--alignment_dir : directory containing prompt_NNN/ subdirectories of per-image
    alignment JSONs named alignment_pNNN_b{bid1}_{bid2}..._sSS.json with schema:
      {"metadata": {"prompt_index", "bids", "sample_index"},
       "alignment_scores": {"base_alignment", "agent1_alignment", ...},
       "welfare_metrics": {...}}
--config : the alignment config JSON used to produce the data. Must contain
    "num_agents" and "bidding_combinations"; "num_samples_per_combination"
    is optional (default 20).
--output_dir : directory to write plots and CSVs (created if missing).
--metric : name of the alignment metric (e.g. "clip"), used only for labeling.

Example
-------
    python analyze_welfare.py \
        --alignment_dir path/to/alignment_clip_3_agents \
        --config path/to/alignment_clip_config_3_agents.json \
        --output_dir path/to/results/3_agents \
        --metric clip

Only prompts with complete data (every bid combination x every sample present)
are included, so that all statistics are computed over the same prompt set.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_K_VALUES = [1, 2, 3, 5, 10, 15, 20]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def construct_alignment_filename(prompt_idx: int, bids: List[float],
                                 sample_idx: int) -> str:
    """Construct the alignment JSON filename for one image."""
    bid_str = "_".join(f"{bid:.2f}" for bid in bids)
    return f"alignment_p{prompt_idx:03d}_b{bid_str}_s{sample_idx:02d}.json"


def find_prompt_indices(alignment_dir: str) -> List[int]:
    """Find prompt indices from prompt_NNN subdirectories."""
    indices = []
    for name in sorted(os.listdir(alignment_dir)):
        match = re.fullmatch(r"prompt_(\d+)", name)
        if match and os.path.isdir(os.path.join(alignment_dir, name)):
            indices.append(int(match.group(1)))
    return indices


def load_alignment_data(alignment_dir: str, num_agents: int,
                        bidding_combinations: List[List[float]],
                        num_samples: int) -> pd.DataFrame:
    """Load alignment JSONs into a DataFrame.

    Only prompts with complete data (all bid combinations x all samples) are
    kept so every aggregate is computed over the same prompt set.
    """
    all_rows = []
    complete_prompts = 0
    prompt_indices = find_prompt_indices(alignment_dir)

    if not prompt_indices:
        raise FileNotFoundError(
            f"No prompt_NNN subdirectories found in {alignment_dir}")

    for prompt_idx in prompt_indices:
        prompt_dir = os.path.join(alignment_dir, f"prompt_{prompt_idx:03d}")
        prompt_rows = []
        is_complete = True

        for bids in bidding_combinations:
            for sample_idx in range(num_samples):
                filepath = os.path.join(
                    prompt_dir,
                    construct_alignment_filename(prompt_idx, bids, sample_idx))
                if not os.path.exists(filepath):
                    is_complete = False
                    break
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    row = {
                        "prompt": data["metadata"]["prompt_index"],
                        "sample": data["metadata"]["sample_index"],
                        "base_alignment":
                            data["alignment_scores"].get("base_alignment", 0.0),
                    }
                    for i, bid in enumerate(bids):
                        row[f"bid_{i + 1}"] = bid
                    for i in range(num_agents):
                        row[f"agent{i + 1}_alignment"] = (
                            data["alignment_scores"].get(
                                f"agent{i + 1}_alignment", 0.0))
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
          f"({len(all_rows)} rows) from {alignment_dir}")
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Welfare computation
# ---------------------------------------------------------------------------

def get_bid_columns(num_agents: int) -> List[str]:
    return [f"bid_{i + 1}" for i in range(num_agents)]


def calculate_welfare(df: pd.DataFrame, num_agents: int) -> pd.DataFrame:
    """Welfare of an image = sum_i bid_i * agent_i_alignment."""
    df = df.copy()
    df["welfare"] = sum(df[f"bid_{i + 1}"] * df[f"agent{i + 1}_alignment"]
                        for i in range(num_agents))
    return df


def is_single_agent_monopoly(row: pd.Series, num_agents: int) -> bool:
    """True if exactly one agent bids 1.0 and all others bid 0.0."""
    bids = [row[f"bid_{i + 1}"] for i in range(num_agents)]
    return (sum(1 for b in bids if b == 1.0) == 1
            and sum(1 for b in bids if b == 0.0) == num_agents - 1)


def calculate_single_winner_baseline(prompt_df: pd.DataFrame,
                                     num_agents: int) -> pd.Series:
    """Single-winner (VCG-style) baseline for every row of one prompt.

    For each row's bid vector, the baseline is the maximum welfare achievable
    by any single-agent-monopoly image (one agent bids 1.0, all others 0.0),
    evaluated under that row's bids.
    """
    baseline_mask = prompt_df.apply(
        lambda r: is_single_agent_monopoly(r, num_agents), axis=1)
    baseline_rows = prompt_df[baseline_mask]

    if len(baseline_rows) == 0:
        return pd.Series([0.0] * len(prompt_df), index=prompt_df.index)

    baselines = []
    for _, row in prompt_df.iterrows():
        max_welfare = 0.0
        for _, baseline_row in baseline_rows.iterrows():
            welfare = sum(
                baseline_row[f"agent{i + 1}_alignment"] * row[f"bid_{i + 1}"]
                for i in range(num_agents))
            max_welfare = max(max_welfare, welfare)
        baselines.append(max_welfare)
    return pd.Series(baselines, index=prompt_df.index)


def compute_welfare_summary(df: pd.DataFrame, num_agents: int,
                            k: int) -> pd.DataFrame:
    """Per prompt x bid combination: best-of-k welfare vs baseline.

    Returns one row per (prompt, bid combination) with columns
    best_welfare, baseline, and diff = relative improvement over baseline.
    """
    bid_cols = get_bid_columns(num_agents)
    df_k = df[df["sample"] < k].copy()
    df_k = calculate_welfare(df_k, num_agents)
    df_k["best_welfare"] = df_k.groupby(["prompt"] + bid_cols)[
        "welfare"].transform("max")
    df_k["baseline"] = df_k.groupby("prompt", group_keys=False).apply(
        lambda x: calculate_single_winner_baseline(x, num_agents))

    welfare_df = df_k.groupby(["prompt"] + bid_cols).first().reset_index()
    welfare_df = welfare_df[["prompt"] + bid_cols
                            + ["best_welfare", "baseline"]]
    welfare_df["diff"] = np.where(
        welfare_df["baseline"] > 0,
        (welfare_df["best_welfare"] - welfare_df["baseline"])
        / welfare_df["baseline"],
        0)
    welfare_df["bid_combo"] = welfare_df.apply(
        lambda r: "/".join(f"{r[f'bid_{i + 1}']:.2f}"
                           for i in range(num_agents)), axis=1)
    return welfare_df


def non_degenerate_combos(welfare_df: pd.DataFrame,
                          num_agents: int) -> pd.DataFrame:
    """Drop all-zero and single-agent-monopoly bid combinations.

    Monopoly combinations define the baseline itself, so their improvement is
    not meaningful; all-zero combinations have zero welfare by definition.
    """
    bid_cols = get_bid_columns(num_agents)
    total = welfare_df[bid_cols].sum(axis=1)
    monopoly = welfare_df.apply(
        lambda r: is_single_agent_monopoly(r, num_agents), axis=1)
    return welfare_df[(total > 0) & (~monopoly)]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_welfare_over_bids(df: pd.DataFrame, num_agents: int, k: int,
                           metric: str, output_dir: str) -> pd.DataFrame:
    """Bar plot: mean welfare improvement per bid combination."""
    welfare_df = non_degenerate_combos(
        compute_welfare_summary(df, num_agents, k), num_agents)

    stats = welfare_df.groupby("bid_combo")["diff"].agg(
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
    ax.set_ylabel("Mean Welfare Improvement ± SE", fontsize=14)
    ax.set_title(f"{num_agents}-Agent Welfare Improvement Over the "
                 f"Single-Winner Baseline (k={k}, {metric.upper()})",
                 fontsize=15, pad=20)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, linewidth=2,
               label="No Improvement")
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="best", fontsize=12)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "welfare_over_bids.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    csv_path = os.path.join(output_dir, "welfare_over_bids.csv")
    stats[["bid_combo", "mean", "std", "se", "count"]].to_csv(
        csv_path, index=False)
    print(f"Saved: {csv_path}")
    return stats


def plot_welfare_over_k(df: pd.DataFrame, num_agents: int,
                        k_list: List[int], metric: str,
                        output_dir: str) -> pd.DataFrame:
    """Bar plot: average welfare improvement vs number of samples k."""
    rows = []
    for k in k_list:
        welfare_df = non_degenerate_combos(
            compute_welfare_summary(df, num_agents, k), num_agents)
        rows.append({
            "k": k,
            "mean": welfare_df["diff"].mean(),
            "std": welfare_df["diff"].std(),
            "se": welfare_df["diff"].std() / np.sqrt(len(welfare_df)),
            "count": len(welfare_df),
        })
    k_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.set_facecolor("white")
    bars = ax.bar([str(k) for k in k_list], k_df["mean"], yerr=k_df["se"],
                  capsize=5, alpha=0.8, color="darkgreen",
                  edgecolor="darkblue")
    for bar, row in zip(bars, k_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + row.se + 0.0003,
                f"{row.mean:.4f}", ha="center", va="bottom", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Number of Samples (k)", fontsize=14)
    ax.set_ylabel("Average Welfare Improvement ± SE", fontsize=14)
    ax.set_title(f"{num_agents}-Agent: Welfare Improvement vs Sample Size "
                 f"({metric.upper()})", fontsize=15, pad=20)
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "welfare_over_k.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    csv_path = os.path.join(output_dir, "welfare_over_k.csv")
    k_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return k_df


def plot_bid_monotonicity(df: pd.DataFrame, num_agents: int, metric: str,
                          output_dir: str) -> pd.DataFrame:
    """Line plot: each agent's mean alignment score vs its own bid."""
    cmap = plt.get_cmap("tab10" if num_agents <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.set_facecolor("white")

    all_stats = []
    for i in range(num_agents):
        bid_col = f"bid_{i + 1}"
        align_col = f"agent{i + 1}_alignment"
        stats = df.groupby(bid_col)[align_col].agg(
            ["mean", "std", "count"]).reset_index()
        stats["se"] = stats["std"] / np.sqrt(stats["count"])
        stats = stats.rename(columns={bid_col: "bid"})
        stats.insert(0, "agent", i + 1)
        all_stats.append(stats)

        color = cmap(i % cmap.N)
        ax.plot(stats["bid"], stats["mean"], "o-", linewidth=2, markersize=6,
                color=color, label=f"Agent {i + 1}")
        ax.fill_between(stats["bid"], stats["mean"] - stats["se"],
                        stats["mean"] + stats["se"], alpha=0.2, color=color)

    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Agent's Own Bid", fontsize=14)
    ax.set_ylabel("Average Alignment Score ± SE", fontsize=14)
    ax.set_title(f"{num_agents}-Agent Bid Monotonicity: Alignment vs Own Bid "
                 f"({metric.upper()})", fontsize=15, pad=20)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(fontsize=11, loc="best",
              ncol=1 if num_agents <= 5 else 2)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "bid_monotonicity.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    stats_df = pd.concat(all_stats, ignore_index=True)
    csv_path = os.path.join(output_dir, "bid_monotonicity.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return stats_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze welfare from multi-agent diffusion auction "
                    "alignment results. Produces welfare-over-bids, "
                    "welfare-over-k, and bid-monotonicity plots plus CSV "
                    "exports for a single experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--alignment_dir", required=True,
        help="Directory containing prompt_NNN/ subdirectories of alignment "
             "JSON files.")
    parser.add_argument(
        "--config", required=True,
        help="Alignment config JSON used to produce the data (must contain "
             "num_agents and bidding_combinations).")
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write plots and CSV files (created if missing).")
    parser.add_argument(
        "--metric", default="clip",
        help="Name of the alignment metric, used for plot labels.")
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

    df = load_alignment_data(args.alignment_dir, num_agents,
                             bidding_combinations, num_samples)
    if df.empty:
        print("No complete data found; nothing to analyze.", file=sys.stderr)
        sys.exit(1)

    # (a) Welfare over bid combinations (best-of-all-samples)
    combo_stats = plot_welfare_over_bids(df, num_agents, num_samples,
                                         args.metric, args.output_dir)

    # (b) Welfare over number of samples k
    k_values = sorted({k for k in DEFAULT_K_VALUES if k <= num_samples}
                      | {num_samples})
    plot_welfare_over_k(df, num_agents, k_values, args.metric,
                        args.output_dir)

    # (c) Bid monotonicity
    plot_bid_monotonicity(df, num_agents, args.metric, args.output_dir)

    # (d) Aggregated data export
    data_csv = os.path.join(args.output_dir, "data.csv")
    combo_stats[["bid_combo", "mean", "std", "se", "count"]].to_csv(
        data_csv, index=False)
    print(f"Saved: {data_csv}")

    print("\nSummary (welfare improvement over single-winner baseline):")
    print(f"  mean = {combo_stats['mean'].mean():.4f}")
    print(f"  min  = {combo_stats['mean'].min():.4f}")
    print(f"  max  = {combo_stats['mean'].max():.4f}")


if __name__ == "__main__":
    main()
