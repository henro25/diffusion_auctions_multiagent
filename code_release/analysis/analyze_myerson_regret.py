"""Myerson-payment truthfulness / regret analysis for 2-agent diffusion auctions.

Given CLIP alignment scores for images generated over a grid of 2-agent bid
combinations, this script:

1. Loads per-image alignment JSONs (one directory of ``prompt_NNN/`` subdirs).
2. Computes ex-ante allocation curves via batch averaging: for each bid combo,
   samples are split into ``num_batches`` batches of ``batch_size``; the
   welfare-maximizing image is selected within each batch, and alignment
   scores are averaged across batch winners (default 5x5 = k=25 samples).
3. Builds interpolated allocation curves x(b) per (prompt, fixed opponent bid).
4. Computes the Myerson payment implied by each allocation curve
   (Myerson's lemma): p(b) = b * x(b) - integral_0^b x(z) dz, evaluated with
   trapezoid quadrature. Under this payment rule, truthful bidding is optimal
   whenever x is monotone; deviations from monotonicity in the empirical
   curves show up as positive regret.
5. Computes relative regret of truthful bidding vs. the best deviation over a
   bid grid, and saves plots plus a data.csv of regret values.

Usage:
    python analyze_myerson_regret.py \\
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
_MARKER_CYCLE = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def _style_maps(fixed_values):
    """Assign a stable color / marker to each fixed opponent value."""
    colors = {fv: _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
              for i, fv in enumerate(fixed_values)}
    markers = {fv: _MARKER_CYCLE[i % len(_MARKER_CYCLE)]
               for i, fv in enumerate(fixed_values)}
    return colors, markers


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
# Allocation curves
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


# ---------------------------------------------------------------------------
# Myerson payments, utilities, regret
# ---------------------------------------------------------------------------

def compute_myerson_payment(interp_fn, bid, n_quadrature=1000):
    """Myerson's lemma payment: p(b) = b * x(b) - integral_0^b x(z) dz.

    The integral is evaluated with trapezoid quadrature over ``n_quadrature``
    points; the payment is clamped at zero.
    """
    x_at_bid = float(interp_fn(bid))
    z_vals = np.linspace(0, bid, n_quadrature)
    x_vals = interp_fn(z_vals)
    integral = np.trapz(x_vals, z_vals)
    return max(bid * x_at_bid - integral, 0.0)


def compute_utility(true_value, bid, interp_fn):
    """Quasilinear utility u(v, b) = v * x(b) - p(b)."""
    x_at_bid = float(interp_fn(bid))
    payment = compute_myerson_payment(interp_fn, bid)
    return true_value * x_at_bid - payment


def compute_regret_curve(interp_fn, true_values=None, bid_grid=None):
    """Relative regret of truthful bidding for each true value.

    regret(v) = (max_b u(v, b) - u(v, v)) / |u(v, v)|
    """
    if true_values is None:
        true_values = REGRET_TRUE_VALUES
    if bid_grid is None:
        bid_grid = ALL_BIDS

    results = []
    for v in true_values:
        u_truthful = compute_utility(v, v, interp_fn)
        utilities = [compute_utility(v, b, interp_fn) for b in bid_grid]
        best_idx = int(np.argmax(utilities))
        u_optimal = utilities[best_idx]
        b_optimal = float(bid_grid[best_idx])

        if abs(u_truthful) > 1e-10:
            rel_regret = (u_optimal - u_truthful) / abs(u_truthful)
        else:
            rel_regret = 0.0 if abs(u_optimal - u_truthful) < 1e-10 else np.inf

        results.append({
            'true_value': v,
            'truthful_utility': u_truthful,
            'optimal_utility': u_optimal,
            'optimal_bid': b_optimal,
            'relative_regret': rel_regret,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_regret_per_prompt(curves, agent_name, prompts, fixed_values):
    """Per-prompt panels: top row = allocation curve, bottom row = regret."""
    if not curves:
        print(f'No curves for per-prompt regret plot ({agent_name})')
        return None

    prompt_indices = sorted(set(k[0] for k in curves.keys()))
    available_fvs = sorted(set(k[1] for k in curves.keys()))
    colors, markers = _style_maps(fixed_values)
    fine_bids = np.linspace(0, 1, 200)

    fig, axes = plt.subplots(2, len(prompt_indices),
                             figsize=(7 * len(prompt_indices), 9),
                             squeeze=False)

    for col, pidx in enumerate(prompt_indices):
        ax_alloc = axes[0, col]
        ax_regret = axes[1, col]

        for fv in available_fvs:
            key = (pidx, fv)
            if key not in curves:
                continue
            c = curves[key]
            color = colors.get(fv, 'tab:gray')

            # Allocation curve with per-batch std error bars
            ax_alloc.plot(fine_bids, c['interp_fn'](fine_bids), color=color,
                          linewidth=2, label=f'Opponent={fv}')
            ax_alloc.errorbar(c['bids'], c['allocations'], yerr=c['stds'],
                              fmt='o', color=color, markersize=3, capsize=2,
                              alpha=0.5, zorder=5)

            # Regret curve
            rdf = compute_regret_curve(c['interp_fn'],
                                       true_values=FINE_TRUE_VALUES)
            ax_regret.plot(rdf['true_value'], rdf['relative_regret'],
                           marker=markers.get(fv, 'o'), color=color,
                           label=f'Opponent={fv}', linewidth=2, markersize=4)

        ax_alloc.set_xlabel(f'{agent_name} Bid')
        ax_alloc.set_ylabel('Alignment (batch-averaged)')
        ax_alloc.set_title(
            f'P{pidx}: {prompts[pidx]["base_prompt"][:35]}...', fontsize=10)
        ax_alloc.legend(fontsize=8)
        ax_alloc.set_xlim(-0.02, 1.02)

        ax_regret.set_xlabel('Agent True Value')
        ax_regret.set_ylabel('Relative Regret')
        ax_regret.legend(fontsize=9)
        ax_regret.set_xlim(0.0, 1.0)
        ax_regret.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    fig.suptitle(f'{agent_name} — Allocation & Myerson Regret '
                 f'(batch-averaged)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_regret_mean_across_prompts(curves, agent_name, fixed_values):
    """Left panel = mean allocation across prompts, right panel = mean regret."""
    if not curves:
        print(f'No curves for mean regret plot ({agent_name})')
        return None

    available_fvs = sorted(set(k[1] for k in curves.keys()))
    colors, markers = _style_maps(fixed_values)
    fine_bids = np.linspace(0, 1, 200)

    fig, (ax_alloc, ax_regret) = plt.subplots(1, 2, figsize=(14, 5))

    for fv in available_fvs:
        prompt_keys = [k for k in curves if k[1] == fv]
        if not prompt_keys:
            continue
        color = colors.get(fv, 'tab:gray')

        # Mean allocation
        all_allocs = np.array([curves[k]['interp_fn'](fine_bids)
                               for k in prompt_keys])
        ax_alloc.plot(fine_bids, all_allocs.mean(axis=0), color=color,
                      linewidth=2, label=f'Opponent={fv}')
        ax_alloc.fill_between(fine_bids,
                              all_allocs.mean(axis=0) - all_allocs.std(axis=0),
                              all_allocs.mean(axis=0) + all_allocs.std(axis=0),
                              color=color, alpha=0.15)

        # Mean regret
        all_regrets = [
            compute_regret_curve(curves[k]['interp_fn'],
                                 true_values=FINE_TRUE_VALUES)
            ['relative_regret'].values
            for k in prompt_keys]
        mean_regret = np.mean(all_regrets, axis=0)
        ax_regret.plot(FINE_TRUE_VALUES, mean_regret,
                       marker=markers.get(fv, 'o'), color=color,
                       label=f'Opponent={fv}', linewidth=2, markersize=4)

    ax_alloc.set_xlabel(f'{agent_name} Bid')
    ax_alloc.set_ylabel('Alignment (mean across prompts)')
    ax_alloc.set_title('Allocation Curves')
    ax_alloc.legend(fontsize=9)
    ax_alloc.set_xlim(-0.02, 1.02)

    ax_regret.set_xlabel('Agent True Value')
    ax_regret.set_ylabel('Relative Regret')
    ax_regret.set_title('Relative Regret')
    ax_regret.legend(fontsize=10)
    ax_regret.set_xlim(0.0, 1.0)
    ax_regret.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    fig.suptitle(f'{agent_name} — Mean Allocation & Myerson Regret '
                 f'(batch-averaged)', fontsize=14)
    plt.tight_layout()
    return fig


def plot_regret_per_batch(curves, agent_name, prompts):
    """Per-batch allocation + regret. For each (fixed value, prompt):
    top = allocation (per-batch lines + mean), bottom = regret."""
    if not curves:
        print(f'No curves for per-batch regret plot ({agent_name})')
        return None

    prompt_indices = sorted(set(k[0] for k in curves.keys()))
    available_fvs = sorted(set(k[1] for k in curves.keys()))
    fine_bids = np.linspace(0, 1, 200)
    batch_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                    'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

    n_cols = max(len(prompt_indices), 1)
    n_rows = 2 * max(len(available_fvs), 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 4 * n_rows), squeeze=False)

    for fv_idx, fv in enumerate(available_fvs):
        row_alloc = 2 * fv_idx
        row_regret = 2 * fv_idx + 1

        for col, pidx in enumerate(prompt_indices):
            ax_a = axes[row_alloc, col]
            ax_r = axes[row_regret, col]
            key = (pidx, fv)
            if key not in curves:
                ax_a.set_visible(False)
                ax_r.set_visible(False)
                continue

            curve_data = curves[key]
            for bi, bfn in enumerate(curve_data.get('batch_interp_fns', [])):
                bc = batch_colors[bi % len(batch_colors)]
                ax_a.plot(fine_bids, bfn(fine_bids), color=bc, linewidth=1,
                          alpha=0.5, label=f'Batch {bi + 1}')
                rdf = compute_regret_curve(bfn, true_values=FINE_TRUE_VALUES)
                ax_r.plot(rdf['true_value'], rdf['relative_regret'], color=bc,
                          linewidth=1, alpha=0.6, label=f'Batch {bi + 1}')

            # Mean allocation and regret (bold)
            ax_a.plot(fine_bids, curve_data['interp_fn'](fine_bids),
                      color='black', linewidth=2.5, label='Mean', zorder=10)
            rdf_mean = compute_regret_curve(curve_data['interp_fn'],
                                            true_values=FINE_TRUE_VALUES)
            ax_r.plot(rdf_mean['true_value'], rdf_mean['relative_regret'],
                      color='black', linewidth=2.5, label='Mean', zorder=10)

            ax_a.set_xlim(-0.02, 1.02)
            ax_r.set_xlim(0.0, 1.0)
            ax_r.axhline(0, color='gray', linewidth=0.5, linestyle='-')

            if col == 0:
                ax_a.set_ylabel(f'Opp={fv}\nAlignment')
                ax_r.set_ylabel(f'Opp={fv}\nRelative Regret')
            if fv_idx == 0:
                ax_a.set_title(
                    f'P{pidx}: {prompts[pidx]["base_prompt"][:35]}...',
                    fontsize=10)
            if fv_idx == len(available_fvs) - 1:
                ax_r.set_xlabel('Agent True Value')
            ax_a.legend(fontsize=6, ncol=4)
            ax_r.legend(fontsize=6, ncol=4)

    fig.suptitle(f'Per-Batch Allocation & Myerson Regret — {agent_name}',
                 fontsize=14)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_regret_rows(curves, agent_label):
    """Regret rows over the coarse true-value grid for the data.csv export."""
    rows = []
    for (pidx, fv), curve_data in sorted(curves.items()):
        rdf = compute_regret_curve(curve_data['interp_fn'])
        rdf['agent'] = agent_label
        rdf['prompt_idx'] = pidx
        rdf['fixed_val'] = fv
        rows.append(rdf)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Myerson-payment truthfulness / regret analysis for '
                    '2-agent diffusion auction experiments.',
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

    curves_a1 = build_allocation_curves(df_avg, 'agent1_alignment', 'b1', 'b2',
                                        args.fixed_values)
    curves_a2 = build_allocation_curves(df_avg, 'agent2_alignment', 'b2', 'b1',
                                        args.fixed_values)
    print(f'Allocation curves: agent 1 = {len(curves_a1)}, '
          f'agent 2 = {len(curves_a2)}')

    for agent_name, curves, suffix in [('Agent 1', curves_a1, 'a1'),
                                       ('Agent 2', curves_a2, 'a2')]:
        if not curves:
            print(f'No curves for {agent_name}; skipping plots.')
            continue

        fig = plot_regret_per_prompt(curves, agent_name, prompts,
                                     args.fixed_values)
        if fig is not None:
            out = os.path.join(args.output_dir, f'regret_mean_{suffix}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')

        fig = plot_regret_mean_across_prompts(curves, agent_name,
                                              args.fixed_values)
        if fig is not None:
            out = os.path.join(args.output_dir,
                               f'regret_mean_across_{suffix}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')

        fig = plot_regret_per_batch(curves, agent_name, prompts)
        if fig is not None:
            out = os.path.join(args.output_dir, f'regret_batch_{suffix}.png')
            fig.savefig(out, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved {out}')

    # Export regret table
    all_rows = (collect_regret_rows(curves_a1, 'Agent 1')
                + collect_regret_rows(curves_a2, 'Agent 2'))
    if all_rows:
        all_regret = pd.concat(all_rows, ignore_index=True)
        cols = ['agent', 'prompt_idx', 'fixed_val', 'true_value',
                'truthful_utility', 'optimal_utility', 'optimal_bid',
                'relative_regret']
        all_regret = all_regret[cols]
        out = os.path.join(args.output_dir, 'data.csv')
        all_regret.to_csv(out, index=False)
        print(f'Saved {len(all_regret)} rows to {out}')

        summary = all_regret.groupby('agent').agg(
            mean_regret=('relative_regret', 'mean'),
            max_regret=('relative_regret', 'max'),
            median_regret=('relative_regret', 'median'),
            std_regret=('relative_regret', 'std'),
        ).reset_index()
        print('\nSummary:')
        print(summary.to_string(index=False))
    else:
        print('No regret data to export.')


if __name__ == '__main__':
    main()
