"""Clarke-VCG truthfulness / regret analysis for 2-agent diffusion auctions.

Given CLIP alignment scores for images generated over a grid of 2-agent bid
combinations, this script:

1. Loads per-image alignment JSONs (one directory of ``prompt_NNN/`` subdirs).
2. Computes ex-ante allocation curves via batch averaging: for each bid combo,
   samples are split into ``num_batches`` batches of ``batch_size``; the
   welfare-maximizing image is selected within each batch, and alignment
   scores are averaged across batch winners (default 5x5 = k=25 samples).
3. Builds interpolated allocation curves for BOTH the deviating agent's own
   alignment x_self(b) and the opponent's alignment x_opp(b) as functions of
   the deviator's bid (opponent bid held fixed).
4. Computes the Clarke-VCG payment: each agent pays the externality it
   imposes on the other agent,
       p(b_self) = b_other * max(x_opp(0) - x_opp(b_self), 0),
   i.e. the opponent's (bid-weighted) alignment loss caused by the deviator's
   participation.
5. Computes relative regret of truthful bidding vs. the best deviation over a
   bid grid, and saves plots plus a data.csv of regret values.

Usage:
    python analyze_vcg_regret.py \\
        --alignment_dir path/to/alignment_results \\
        --prompts path/to/prompts.json \\
        --output_dir path/to/output \\
        [--num_batches 5] [--batch_size 5] \\
        [--fixed_values 0.3 0.5 0.7] [--prompt_indices 0 1 2]

Expected alignment JSON schema (one file per image, ``alignment_*.json``):
    {
      "metadata": {"prompt_index": int, "bids": [b1, b2], "sample_index": int},
      "alignment_scores": {"base_alignment": float,
                            "agent1_alignment": float,
                            "agent2_alignment": float},
      "welfare_metrics": {"total_welfare": float}
    }

The prompts file is a JSON list of objects with keys
``base_prompt``, ``agent1_prompt``, ``agent2_prompt``.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Bid grid over which deviations are searched, and true values for regret.
ALL_BIDS = np.round(np.arange(0.0, 1.05, 0.05), 2)
REGRET_TRUE_VALUES = np.round(np.arange(0.1, 1.0, 0.1), 2)
FINE_TRUE_VALUES = np.round(np.arange(0.05, 1.0, 0.05), 2)

_COLOR_CYCLE = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple',
                'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


def _color_map(fixed_values):
    """Assign a stable color to each fixed opponent value."""
    return {fv: _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
            for i, fv in enumerate(fixed_values)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_alignment_data(alignment_dir, prompts_path, prompt_indices=None):
    """Load all alignment JSONs from a directory into a DataFrame.

    Returns a DataFrame with columns:
        prompt_idx, b1, b2, sample_idx, base_alignment,
        agent1_alignment, agent2_alignment, total_welfare
    """
    if not os.path.exists(alignment_dir):
        print(f'Alignment dir not found: {alignment_dir}')
        return pd.DataFrame()

    with open(prompts_path) as f:
        prompts = json.load(f)

    if prompt_indices is None:
        prompt_indices = list(range(len(prompts)))

    records = []
    for prompt_idx in prompt_indices:
        prompt_dir = os.path.join(alignment_dir, f'prompt_{prompt_idx:03d}')
        if not os.path.isdir(prompt_dir):
            print(f'  Skipping prompt {prompt_idx} (not found)')
            continue

        json_files = glob.glob(os.path.join(prompt_dir, 'alignment_*.json'))
        if not json_files:
            print(f'  Skipping prompt {prompt_idx} (no alignment files)')
            continue

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            meta = data['metadata']
            bids = meta['bids']
            scores = data['alignment_scores']
            welfare = data['welfare_metrics']

            records.append({
                'prompt_idx': meta['prompt_index'],
                'b1': bids[0],
                'b2': bids[1],
                'sample_idx': meta['sample_index'],
                'base_alignment': scores['base_alignment'],
                'agent1_alignment': scores['agent1_alignment'],
                'agent2_alignment': scores['agent2_alignment'],
                'total_welfare': welfare['total_welfare'],
            })

    df = pd.DataFrame(records)
    if len(df) > 0:
        print(f'Loaded {len(df)} alignment records from {alignment_dir}')
        sample_counts = df.groupby(['prompt_idx', 'b1', 'b2']).size()
        print(f'  Prompts: {sorted(df.prompt_idx.unique())}')
        print(f'  Bid combos: {len(sample_counts)}')
        print(f'  Samples per combo: min={sample_counts.min()}, '
              f'max={sample_counts.max()}, median={sample_counts.median()}')
    else:
        print(f'No alignment data found in {alignment_dir}')
    return df


# ---------------------------------------------------------------------------
# Batch-averaged (ex-ante) allocations
# ---------------------------------------------------------------------------

def batch_welfare_average(df, num_batches=5, batch_size=5):
    """For each (prompt_idx, b1, b2), split samples into batches, pick the
    welfare-optimal image from each batch, and average alignments across
    batch winners.

    Only bid combos with the full ``num_batches * batch_size`` samples are
    included; incomplete combos are skipped.

    Returns a DataFrame with one row per (prompt_idx, b1, b2) containing the
    averaged alignment scores plus per-batch values for error bars.
    """
    if len(df) == 0:
        return df

    total_needed = num_batches * batch_size
    records = []
    skipped = 0

    for (pidx, b1, b2), group in df.groupby(['prompt_idx', 'b1', 'b2']):
        group_sorted = group.sort_values('sample_idx').reset_index(drop=True)

        if len(group_sorted) < total_needed:
            skipped += 1
            continue

        batch_winners_a1 = []
        batch_winners_a2 = []
        batch_winners_base = []

        for batch_i in range(num_batches):
            start = batch_i * batch_size
            batch = group_sorted.iloc[start:start + batch_size]

            # Pick welfare-optimal image within this batch
            winner = batch.loc[batch['total_welfare'].idxmax()]

            batch_winners_a1.append(winner['agent1_alignment'])
            batch_winners_a2.append(winner['agent2_alignment'])
            batch_winners_base.append(winner['base_alignment'])

        records.append({
            'prompt_idx': pidx,
            'b1': b1,
            'b2': b2,
            'agent1_alignment': np.mean(batch_winners_a1),
            'agent2_alignment': np.mean(batch_winners_a2),
            'base_alignment': np.mean(batch_winners_base),
            'agent1_alignment_std': np.std(batch_winners_a1),
            'agent2_alignment_std': np.std(batch_winners_a2),
            'agent1_batch_vals': list(batch_winners_a1),
            'agent2_batch_vals': list(batch_winners_a2),
            'n_batches': num_batches,
        })

    result = pd.DataFrame(records)
    print(f'Batch-averaged {len(result)} bid combos '
          f'({skipped} skipped for < {total_needed} samples)')
    return result


# ---------------------------------------------------------------------------
# Allocation curves (self and opponent)
# ---------------------------------------------------------------------------

def build_allocation_curves(df_avg, agent_col, vary_col, fixed_col,
                            fixed_values, min_points=5):
    """Build interpolated allocation curves from batch-averaged data.

    Args:
        df_avg: batch-averaged DataFrame.
        agent_col: 'agent1_alignment' or 'agent2_alignment'.
        vary_col: 'b1' or 'b2' (the bid that varies).
        fixed_col: 'b2' or 'b1' (the fixed opponent bid).
        fixed_values: opponent bid values at which curves are anchored.
        min_points: minimum number of bid points needed to form a curve.

    Returns:
        dict: {(prompt_idx, fixed_val): {'bids', 'allocations', 'stds',
               'interp_fn', 'batch_interp_fns'}}
    """
    curves = {}
    if len(df_avg) == 0:
        return curves

    std_col = agent_col + '_std'
    batch_col = agent_col.replace('_alignment', '_batch_vals')

    for prompt_idx in df_avg['prompt_idx'].unique():
        for fixed_val in fixed_values:
            mask = ((df_avg['prompt_idx'] == prompt_idx)
                    & (np.isclose(df_avg[fixed_col], fixed_val)))
            subset = df_avg[mask].sort_values(vary_col)

            if len(subset) < min_points:
                continue

            bids = subset[vary_col].values
            allocations = subset[agent_col].values
            stds = (subset[std_col].values if std_col in subset.columns
                    else np.zeros_like(allocations))

            interp_fn = interpolate.interp1d(
                bids, allocations, kind='linear',
                fill_value='extrapolate', bounds_error=False)

            # Per-batch interpolation functions (for per-batch regret plots)
            batch_interp_fns = []
            if batch_col in subset.columns:
                batch_vals_list = subset[batch_col].tolist()
                n_batches = len(batch_vals_list[0]) if batch_vals_list else 0
                for bi in range(n_batches):
                    batch_allocs = np.array([bv[bi] for bv in batch_vals_list])
                    batch_interp_fns.append(interpolate.interp1d(
                        bids, batch_allocs, kind='linear',
                        fill_value='extrapolate', bounds_error=False))

            curves[(prompt_idx, fixed_val)] = {
                'bids': bids,
                'allocations': allocations,
                'stds': stds,
                'interp_fn': interp_fn,
                'batch_interp_fns': batch_interp_fns,
            }

    return curves


def build_opponent_curves(df_avg, deviator_col, opp_alignment_col, fixed_col,
                          fixed_values, min_points=5):
    """For each (prompt_idx, fixed_val), build an interpolator for the
    OPPONENT's alignment as a function of the deviating agent's bid. Used
    inside the VCG payment computation.

    Args:
        df_avg: batch-averaged DataFrame.
        deviator_col: 'b1' if agent 1 is deviating, 'b2' if agent 2 is.
        opp_alignment_col: 'agent2_alignment' if agent 1 deviating, else
            'agent1_alignment'.
        fixed_col: 'b2' if agent 1 deviating, 'b1' if agent 2 deviating.
        fixed_values: opponent bid values at which curves are anchored.
        min_points: minimum number of bid points needed to form a curve.

    Returns:
        dict: {(prompt_idx, fixed_val): {'bids', 'allocations', 'interp_fn',
               'batch_interp_fns'}}
    """
    curves = {}
    if len(df_avg) == 0:
        return curves

    batch_col = opp_alignment_col.replace('_alignment', '_batch_vals')
    for prompt_idx in df_avg['prompt_idx'].unique():
        for fv in fixed_values:
            mask = ((df_avg['prompt_idx'] == prompt_idx)
                    & (np.isclose(df_avg[fixed_col], fv)))
            sub = df_avg[mask].sort_values(deviator_col)
            if len(sub) < min_points:
                continue

            bids = sub[deviator_col].values
            opp_allocs = sub[opp_alignment_col].values
            interp_fn = interpolate.interp1d(
                bids, opp_allocs, kind='linear',
                fill_value='extrapolate', bounds_error=False)

            batch_fns = []
            if batch_col in sub.columns:
                bv_list = sub[batch_col].tolist()
                n_b = len(bv_list[0]) if bv_list else 0
                for bi in range(n_b):
                    batch_allocs = np.array([bv[bi] for bv in bv_list])
                    batch_fns.append(interpolate.interp1d(
                        bids, batch_allocs, kind='linear',
                        fill_value='extrapolate', bounds_error=False))

            curves[(prompt_idx, fv)] = {
                'bids': bids,
                'allocations': opp_allocs,
                'interp_fn': interp_fn,
                'batch_interp_fns': batch_fns,
            }
    return curves


# ---------------------------------------------------------------------------
# VCG payments, utilities, regret
# ---------------------------------------------------------------------------

def compute_vcg_payment_2agent(opp_interp_fn, b_self, b_other_fixed):
    """Clarke-VCG payment for the deviating agent (2-agent case).

    p_vcg(b_self) = b_other * max(x_opp(0) - x_opp(b_self), 0)

    The deviator pays the externality it imposes on the opponent: the
    opponent's bid-weighted alignment loss relative to the deviator bidding
    zero. Both terms are read from ``opp_interp_fn``; clamped at zero.
    """
    opp_at_zero = float(opp_interp_fn(0.0))
    opp_at_self = float(opp_interp_fn(b_self))
    return max(b_other_fixed * (opp_at_zero - opp_at_self), 0.0)


def compute_vcg_utility(true_value, b_self, b_other_fixed, alloc_fn,
                        opp_interp_fn):
    """Quasilinear utility u(v, b) = v * x_self(b) - p_vcg(b)."""
    x_self = float(alloc_fn(b_self))
    p = compute_vcg_payment_2agent(opp_interp_fn, b_self, b_other_fixed)
    return true_value * x_self - p


def compute_vcg_regret_curve(alloc_fn, opp_interp_fn, b_other_fixed,
                             true_values=None, bid_grid=None):
    """Relative regret of truthful bidding under VCG payments.

    regret(v) = (max_b u(v, b) - u(v, v)) / |u(v, v)|
    """
    if true_values is None:
        true_values = REGRET_TRUE_VALUES
    if bid_grid is None:
        bid_grid = ALL_BIDS

    rows = []
    for v in true_values:
        u_truthful = compute_vcg_utility(v, v, b_other_fixed, alloc_fn,
                                         opp_interp_fn)
        utilities = [compute_vcg_utility(v, b, b_other_fixed, alloc_fn,
                                         opp_interp_fn) for b in bid_grid]
        best_idx = int(np.argmax(utilities))
        u_optimal = utilities[best_idx]
        b_optimal = float(bid_grid[best_idx])

        if abs(u_truthful) > 1e-10:
            rel = (u_optimal - u_truthful) / abs(u_truthful)
        else:
            rel = 0.0 if abs(u_optimal - u_truthful) < 1e-10 else np.inf

        rows.append({'true_value': v, 'truthful_utility': u_truthful,
                     'optimal_utility': u_optimal, 'optimal_bid': b_optimal,
                     'relative_regret': rel})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_vcg_regret_per_prompt(self_curves, opp_curves, agent_name, prompts,
                               fixed_values):
    """Per-prompt panels. Top row = allocation (self solid + opponent
    dashed), bottom row = VCG regret."""
    if not self_curves or not opp_curves:
        print(f'No curves for per-prompt regret plot ({agent_name})')
        return None

    prompt_indices = sorted(set(k[0] for k in self_curves.keys()))
    colors = _color_map(fixed_values)
    fine_bids = np.linspace(0, 1, 200)
    n_cols = len(prompt_indices)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10),
                             squeeze=False)

    for col, pidx in enumerate(prompt_indices):
        ax_alloc = axes[0, col]
        ax_reg = axes[1, col]
        for fv in fixed_values:
            key = (pidx, fv)
            if key not in self_curves or key not in opp_curves:
                continue
            color = colors.get(fv, 'tab:gray')
            self_data = self_curves[key]
            opp_data = opp_curves[key]

            # Allocation: self (solid) and opponent (dashed)
            ax_alloc.plot(fine_bids, self_data['interp_fn'](fine_bids),
                          color=color, linewidth=2, label=f'self (opp={fv})')
            ax_alloc.errorbar(self_data['bids'], self_data['allocations'],
                              yerr=self_data['stds'], fmt='o', color=color,
                              markersize=4, alpha=0.6)
            ax_alloc.plot(fine_bids, opp_data['interp_fn'](fine_bids),
                          color=color, linewidth=2, linestyle='--',
                          label=f'opponent (opp={fv})')

            # VCG regret curve
            rdf = compute_vcg_regret_curve(
                self_data['interp_fn'], opp_data['interp_fn'],
                b_other_fixed=fv, true_values=FINE_TRUE_VALUES)
            ax_reg.plot(rdf['true_value'], rdf['relative_regret'],
                        color=color, linewidth=2, marker='o', markersize=4,
                        label=f'Opponent={fv}')

        ax_alloc.set_title(
            f'P{pidx}: {prompts[pidx]["base_prompt"][:35]}...', fontsize=10)
        ax_alloc.set_xlabel(f'{agent_name} Bid')
        ax_alloc.set_ylabel('Alignment (batch-averaged)')
        ax_alloc.set_xlim(0, 1)
        ax_alloc.legend(fontsize=8)

        ax_reg.set_xlabel('Agent True Value')
        ax_reg.set_ylabel('Relative Regret')
        ax_reg.set_xlim(0, 1)
        ax_reg.axhline(0, color='gray', linewidth=0.5, linestyle='-')
        ax_reg.legend(fontsize=9)

    fig.suptitle(f'VCG — {agent_name} Allocation & Regret (batch-averaged)',
                 fontsize=14)
    plt.tight_layout()
    return fig


def plot_vcg_regret_mean_across_prompts(self_curves, opp_curves, agent_name,
                                        fixed_values):
    """1x2 figure. Left = mean allocation (self solid + opponent dashed),
    right = mean VCG regret across prompts (with std bands)."""
    if not self_curves or not opp_curves:
        print(f'No curves for mean regret plot ({agent_name})')
        return None

    fig, (ax_alloc, ax_reg) = plt.subplots(1, 2, figsize=(14, 6))
    colors = _color_map(fixed_values)
    fine_bids = np.linspace(0, 1, 200)

    for fv in fixed_values:
        color = colors.get(fv, 'tab:gray')
        keys = [k for k in self_curves if k[1] == fv and k in opp_curves]
        if not keys:
            continue

        # Mean allocation
        self_allocs = np.array([self_curves[k]['interp_fn'](fine_bids)
                                for k in keys])
        opp_allocs = np.array([opp_curves[k]['interp_fn'](fine_bids)
                               for k in keys])
        ax_alloc.plot(fine_bids, self_allocs.mean(axis=0), color=color,
                      linewidth=2, label=f'self (opp={fv})')
        ax_alloc.fill_between(fine_bids,
                              self_allocs.mean(axis=0) - self_allocs.std(axis=0),
                              self_allocs.mean(axis=0) + self_allocs.std(axis=0),
                              color=color, alpha=0.15)
        ax_alloc.plot(fine_bids, opp_allocs.mean(axis=0), color=color,
                      linewidth=2, linestyle='--', label=f'opponent (opp={fv})')

        # Mean regret across prompts
        regret_per_prompt = np.array([
            compute_vcg_regret_curve(
                self_curves[k]['interp_fn'], opp_curves[k]['interp_fn'],
                b_other_fixed=fv, true_values=FINE_TRUE_VALUES)
            ['relative_regret'].values
            for k in keys])
        mean_reg = regret_per_prompt.mean(axis=0)
        std_reg = regret_per_prompt.std(axis=0)
        ax_reg.plot(FINE_TRUE_VALUES, mean_reg, color=color, linewidth=2,
                    marker='o', markersize=4, label=f'Opponent={fv}')
        ax_reg.fill_between(FINE_TRUE_VALUES, mean_reg - std_reg,
                            mean_reg + std_reg, color=color, alpha=0.15)

    ax_alloc.set_title(f'{agent_name}: mean allocation across prompts')
    ax_alloc.set_xlabel(f'{agent_name} Bid')
    ax_alloc.set_ylabel('Alignment')
    ax_alloc.set_xlim(0, 1)
    ax_alloc.legend(fontsize=9)

    ax_reg.set_title(f'{agent_name}: mean VCG regret across prompts')
    ax_reg.set_xlabel('Agent True Value')
    ax_reg.set_ylabel('Relative Regret')
    ax_reg.set_xlim(0, 1)
    ax_reg.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    ax_reg.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_vcg_regret_per_batch(self_curves, opp_curves, agent_name,
                              fixed_values):
    """For each (fixed value, prompt): the per-batch VCG regret curves plus
    their mean (bold black)."""
    if not self_curves or not opp_curves:
        print(f'No curves for per-batch regret plot ({agent_name})')
        return None

    prompt_indices = sorted(set(k[0] for k in self_curves.keys()))
    n_rows = max(len(fixed_values), 1)
    n_cols = max(len(prompt_indices), 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for row, fv in enumerate(fixed_values):
        for col, pidx in enumerate(prompt_indices):
            ax = axes[row, col]
            key = (pidx, fv)
            if key not in self_curves or key not in opp_curves:
                ax.set_visible(False)
                continue

            self_batch_fns = self_curves[key]['batch_interp_fns']
            opp_batch_fns = opp_curves[key]['batch_interp_fns']
            n_b = min(len(self_batch_fns), len(opp_batch_fns))
            if n_b == 0:
                ax.set_visible(False)
                continue

            batch_regrets = []
            for bi in range(n_b):
                rdf = compute_vcg_regret_curve(
                    self_batch_fns[bi], opp_batch_fns[bi],
                    b_other_fixed=fv, true_values=FINE_TRUE_VALUES)
                ax.plot(rdf['true_value'], rdf['relative_regret'], alpha=0.4,
                        linewidth=1,
                        label=f'batch {bi + 1}' if (row == 0 and col == 0)
                        else None)
                batch_regrets.append(rdf['relative_regret'].values)

            batch_regrets = np.array(batch_regrets)
            ax.plot(FINE_TRUE_VALUES, batch_regrets.mean(axis=0),
                    color='black', linewidth=2.5,
                    label='mean' if (row == 0 and col == 0) else None)
            ax.set_title(f'Prompt {pidx}, opponent bid={fv}')
            ax.set_xlabel('Agent True Value')
            ax.set_ylabel('Relative Regret')
            ax.set_xlim(0, 1)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(f'VCG per-batch regret — {agent_name}', fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_regret_rows(self_curves, opp_curves, agent_label):
    """Regret rows over the coarse true-value grid for the data.csv export."""
    rows = []
    for (pidx, fv), self_data in sorted(self_curves.items()):
        if (pidx, fv) not in opp_curves:
            continue
        opp_data = opp_curves[(pidx, fv)]
        rdf = compute_vcg_regret_curve(
            self_data['interp_fn'], opp_data['interp_fn'],
            b_other_fixed=fv, true_values=REGRET_TRUE_VALUES)
        rdf['agent'] = agent_label
        rdf['prompt_idx'] = pidx
        rdf['fixed_val'] = fv
        rows.append(rdf)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Clarke-VCG truthfulness / regret analysis for 2-agent '
                    'diffusion auction experiments.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--alignment_dir', required=True,
                        help='Directory containing prompt_NNN/ subdirectories '
                             'of alignment_*.json files.')
    parser.add_argument('--prompts', required=True,
                        help='JSON list of {base_prompt, agent1_prompt, '
                             'agent2_prompt} objects.')
    parser.add_argument('--output_dir', required=True,
                        help='Directory for output plots and data.csv.')
    parser.add_argument('--num_batches', type=int, default=5,
                        help='Number of batches for ex-ante averaging.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Samples per batch (welfare-max within a batch).')
    parser.add_argument('--fixed_values', type=float, nargs='+',
                        default=[0.3, 0.5, 0.7],
                        help='Opponent bid values at which allocation curves '
                             'are anchored.')
    parser.add_argument('--prompt_indices', type=int, nargs='+', default=None,
                        help='Subset of prompt indices to analyze '
                             '(default: all prompts).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    total_samples = args.num_batches * args.batch_size
    print(f'Batch config: {args.num_batches} batches x {args.batch_size} '
          f'samples = {total_samples} total (k={total_samples})')

    with open(args.prompts) as f:
        prompts = json.load(f)

    df = load_alignment_data(args.alignment_dir, args.prompts,
                             prompt_indices=args.prompt_indices)
    if len(df) == 0:
        print('No alignment data found; nothing to do.')
        return

    df_avg = batch_welfare_average(df, args.num_batches, args.batch_size)
    if len(df_avg) == 0:
        print(f'No bid combos with the full {total_samples} samples; '
              'nothing to analyze.')
        return

    # Self allocation curves (deviating agent's own alignment)
    curves_a1 = build_allocation_curves(df_avg, 'agent1_alignment', 'b1', 'b2',
                                        args.fixed_values)
    curves_a2 = build_allocation_curves(df_avg, 'agent2_alignment', 'b2', 'b1',
                                        args.fixed_values)
    # Opponent allocation curves (other agent's alignment vs. self bid)
    opp_curves_a1 = build_opponent_curves(
        df_avg, deviator_col='b1', opp_alignment_col='agent2_alignment',
        fixed_col='b2', fixed_values=args.fixed_values)
    opp_curves_a2 = build_opponent_curves(
        df_avg, deviator_col='b2', opp_alignment_col='agent1_alignment',
        fixed_col='b1', fixed_values=args.fixed_values)
    print(f'Self curves: agent 1 = {len(curves_a1)}, '
          f'agent 2 = {len(curves_a2)}')
    print(f'Opponent curves: agent 1 = {len(opp_curves_a1)}, '
          f'agent 2 = {len(opp_curves_a2)}')

    for agent_name, self_c, opp_c, suffix in [
            ('Agent 1', curves_a1, opp_curves_a1, 'a1'),
            ('Agent 2', curves_a2, opp_curves_a2, 'a2')]:
        if not self_c or not opp_c:
            print(f'No curves for {agent_name}; skipping plots.')
            continue

        fig = plot_vcg_regret_per_prompt(self_c, opp_c, agent_name, prompts,
                                         args.fixed_values)
        if fig is not None:
            out = os.path.join(args.output_dir,
                               f'vcg_regret_mean_{suffix}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')

        fig = plot_vcg_regret_mean_across_prompts(self_c, opp_c, agent_name,
                                                  args.fixed_values)
        if fig is not None:
            out = os.path.join(args.output_dir,
                               f'vcg_regret_mean_across_{suffix}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')

        fig = plot_vcg_regret_per_batch(self_c, opp_c, agent_name,
                                        args.fixed_values)
        if fig is not None:
            out = os.path.join(args.output_dir,
                               f'vcg_regret_batch_{suffix}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')

    # Export regret table
    all_rows = (collect_regret_rows(curves_a1, opp_curves_a1, 'Agent 1')
                + collect_regret_rows(curves_a2, opp_curves_a2, 'Agent 2'))
    if all_rows:
        data_csv = pd.concat(all_rows, ignore_index=True)
        cols = ['agent', 'prompt_idx', 'fixed_val', 'true_value',
                'truthful_utility', 'optimal_utility', 'optimal_bid',
                'relative_regret']
        data_csv = data_csv[cols]
        out = os.path.join(args.output_dir, 'data.csv')
        data_csv.to_csv(out, index=False)
        print(f'Saved {len(data_csv)} rows to {out}')

        summary = data_csv.groupby(['agent', 'fixed_val'])[
            'relative_regret'].agg(['mean', 'max']).reset_index()
        print('\nMean / max relative regret per (agent, fixed value):')
        print(summary.to_string(index=False))
    else:
        print('No regret data to export.')


if __name__ == '__main__':
    main()
