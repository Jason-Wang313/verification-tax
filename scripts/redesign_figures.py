#!/usr/bin/env python3
"""
Regenerate the 4 main-text experiment figures with NeurIPS publication-quality styling.

Figures:
  1. fig_self_eval_zero.pdf   — validates TB (self-verification impossibility)
  2. fig_active_real.pdf      — validates TA (active verification rate)
  3. fig_compositional.pdf    — validates TC (compositional verification tax)
  4. fig_leaderboard_noise.pdf — validates demolition (leaderboard noise)
"""

import os
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# ──────────────────────────────────────────────
# Global NeurIPS-quality settings
# ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-safe palette (IBM Design)
COLORS = {
    'blue':    '#648FFF',
    'purple':  '#785EF0',
    'magenta': '#DC267F',
    'orange':  '#FE6100',
    'yellow':  '#FFB000',
    'gray':    '#666666',
}

# Model color assignments (consistent across all figures)
MODEL_COLORS = {
    'Llama-3.1-405B':    COLORS['blue'],
    'LLaMA-3.1-405B':    COLORS['blue'],
    'Llama-4-Maverick':  COLORS['magenta'],
    'LLaMA-4-Maverick':  COLORS['magenta'],
    'Qwen3-Next-80B':    COLORS['orange'],
}

MODEL_DISPLAY = {
    'Llama-3.1-405B':    'Llama-3.1-405B',
    'LLaMA-3.1-405B':    'Llama-3.1-405B',
    'Llama-4-Maverick':  'Llama-4-Maverick',
    'LLaMA-4-Maverick':  'Llama-4-Maverick',
    'Qwen3-Next-80B':    'Qwen3-Next-80B',
}

# Paths
BASE = r'C:\Users\wangz\verification tax'
RESULTS = os.path.join(BASE, 'results', 'analysis')
FIGURES = os.path.join(BASE, 'figures')
os.makedirs(FIGURES, exist_ok=True)


def _get_color(name):
    return MODEL_COLORS.get(name, COLORS['gray'])


def _get_display(name):
    return MODEL_DISPLAY.get(name, name)


# ──────────────────────────────────────────────
# Figure 1: fig_self_eval_zero  (validates TB)
# ──────────────────────────────────────────────
def figure1():
    print("Generating Figure 1: fig_self_eval_zero ...")
    with open(os.path.join(RESULTS, 'self_eval_correlations.json')) as f:
        data = json.load(f)

    within = data['within_model']
    model_names = list(within.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    markers = ['o', 's', 'D']

    for idx, mname in enumerate(model_names):
        mdata = within[mname]
        bins = mdata['bins']

        confs = np.array([b['mean_conf'] for b in bins])
        accs = np.array([b['accuracy'] for b in bins])
        cal_gaps = np.array([abs(b['calibration_gap']) for b in bins])
        n_items = np.array([b['n_items'] for b in bins])

        # Marker size proportional to sqrt(n_items), scaled for visibility, capped
        sizes = np.clip(np.sqrt(n_items) * 2.5, 15, 120)

        color = _get_color(mname)
        display = _get_display(mname)
        marker = markers[idx % len(markers)]

        # Left panel: confidence vs accuracy
        ax1.scatter(confs, accs, s=sizes, c=color, marker=marker,
                    alpha=0.75, edgecolors='white', linewidths=0.3,
                    label=display, zorder=3)

        # Right panel: confidence vs |calibration gap|
        ax2.scatter(confs, cal_gaps, s=sizes, c=color, marker=marker,
                    alpha=0.75, edgecolors='white', linewidths=0.3,
                    label=display, zorder=3)

    # Spearman annotations for left panel (conf vs accuracy)
    y_offset = 0.95
    for idx, mname in enumerate(model_names):
        mdata = within[mname]
        r_val = mdata['spearman_conf_vs_accuracy']['r']
        p_val = mdata['spearman_conf_vs_accuracy']['p']
        color = _get_color(mname)
        display = _get_display(mname)
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
        ax1.text(0.03, y_offset - idx * 0.08,
                 f'{display}: $r_s$ = {r_val:.2f} {sig}',
                 transform=ax1.transAxes, fontsize=7.5, color=color,
                 verticalalignment='top')

    # Spearman annotations for right panel (conf vs |cal gap|)
    y_offset = 0.95
    for idx, mname in enumerate(model_names):
        mdata = within[mname]
        r_val = mdata['spearman_conf_vs_calibration_gap']['r']
        p_val = mdata['spearman_conf_vs_calibration_gap']['p']
        color = _get_color(mname)
        display = _get_display(mname)
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
        ax2.text(0.03, y_offset - idx * 0.08,
                 f'{display}: $r_s$ = {r_val:.2f} {sig}',
                 transform=ax2.transAxes, fontsize=7.5, color=color,
                 verticalalignment='top')

    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('(a) Confidence vs. Accuracy')
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)

    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('|Calibration Gap|')
    ax2.set_title('(b) Confidence vs. |Calibration Gap|')
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig.tight_layout(w_pad=3.0)

    out_pdf = os.path.join(FIGURES, 'fig_self_eval_zero.pdf')
    out_png = os.path.join(FIGURES, 'fig_self_eval_zero.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")


# ──────────────────────────────────────────────
# Figure 2: fig_active_real  (validates TA)
# ──────────────────────────────────────────────
def figure2():
    print("Generating Figure 2: fig_active_real ...")
    with open(os.path.join(RESULTS, 'active_real_results.json')) as f:
        data = json.load(f)

    m_values = data['m_values']
    models = data['models']
    model_names = list(models.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # ---- Left panel: log-log estimation error vs m ----
    for idx, mname in enumerate(model_names):
        mdata = models[mname]
        color = _get_color(mname)
        display = _get_display(mname)

        # Passive
        passive_means = [mdata['passive'][str(m)]['mean'] for m in m_values]
        ax1.plot(m_values, passive_means, '-o', color=color, markersize=4,
                 label=f'{display} (passive)', zorder=3)

        # Active
        active_means = [mdata['active'][str(m)]['mean'] for m in m_values]
        ax1.plot(m_values, active_means, '--^', color=color, markersize=4,
                 alpha=0.8, label=f'{display} (active)', zorder=3)

    # Theory reference lines
    m_arr = np.array(m_values, dtype=float)
    # m^{-1/3} reference (passive theory)
    ref_passive = 0.5 * (m_arr / m_arr[0]) ** (-1/3)
    ax1.plot(m_arr, ref_passive, ':', color=COLORS['gray'], linewidth=1.2,
             label=r'$m^{-1/3}$ (passive theory)', zorder=1)

    # m^{-1/2} reference (active theory)
    ref_active = 0.35 * (m_arr / m_arr[0]) ** (-1/2)
    ax1.plot(m_arr, ref_active, '-.', color=COLORS['gray'], linewidth=1.2,
             label=r'$m^{-1/2}$ (active theory)', zorder=1)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of samples $m$')
    ax1.set_ylabel('Mean ECE estimation error')
    ax1.set_title('(a) Estimation Error vs. Sample Size')
    ax1.legend(fontsize=6.5, ncol=2, loc='lower left', framealpha=0.9)

    # ---- Right panel: bar chart at m=2000 ----
    m_target = '2000'
    bar_width = 0.25
    x_pos = np.arange(len(model_names))

    passive_errs = []
    active_errs = []
    for mname in model_names:
        mdata = models[mname]
        passive_errs.append(mdata['passive'][m_target]['mean'])
        active_errs.append(mdata['active'][m_target]['mean'])

    bars_passive = ax2.bar(x_pos - bar_width/2, passive_errs, bar_width,
                           color=[_get_color(m) for m in model_names],
                           alpha=0.85, label='Passive', edgecolor='white',
                           linewidth=0.5)
    bars_active = ax2.bar(x_pos + bar_width/2, active_errs, bar_width,
                          color=[_get_color(m) for m in model_names],
                          alpha=0.55, label='Active', edgecolor='black',
                          linewidth=0.5, hatch='///')

    # Add L_hat annotations above active bars
    for i, mname in enumerate(model_names):
        mdata = models[mname]
        L_hat = mdata['L_hat']
        ax2.text(x_pos[i] + bar_width/2, active_errs[i] + 0.0003,
                 f'$\\hat{{L}}$={L_hat:.1f}', ha='center', va='bottom',
                 fontsize=7, color=COLORS['gray'])

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([_get_display(m) for m in model_names],
                         fontsize=8, rotation=15, ha='right')
    ax2.set_ylabel('Mean ECE estimation error')
    ax2.set_title(f'(b) Error at $m = {m_target}$')
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout(w_pad=3.0)

    out_pdf = os.path.join(FIGURES, 'fig_active_real.pdf')
    out_png = os.path.join(FIGURES, 'fig_active_real.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")


# ──────────────────────────────────────────────
# Figure 3: fig_compositional  (validates TC)
# ──────────────────────────────────────────────
def figure3():
    print("Generating Figure 3: fig_compositional ...")
    with open(os.path.join(RESULTS, 'compositional_results.json')) as f:
        data = json.load(f)

    params = data['parameters']
    ms = params['ms']
    Ks = params['Ks']
    per_K = data['per_K_results']
    cost_vs_K = data['cost_vs_K']
    exp_fit = data['exponential_fit']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Sequential colormap: lighter for fewer stages
    cmap = matplotlib.colormaps['YlOrRd']
    K_colors = {K: cmap(0.25 + 0.6 * (K - 1) / (max(Ks) - 1)) for K in Ks}

    # ---- Left panel: log-log estimation error vs m for each K ----
    for K in Ks:
        kdata = per_K[str(K)]
        errors = kdata['errors']
        color = K_colors[K]
        ax1.plot(ms, errors, '-o', color=color, markersize=4,
                 label=f'$K={K}$', zorder=3)

        # Theory reference line: (L_sys * eps / m)^{1/3}
        L_sys = kdata['L_sys_empirical']
        eps = params['eps']
        m_arr = np.array(ms, dtype=float)
        theory = (L_sys * eps / m_arr) ** (1/3)
        ax1.plot(ms, theory, '--', color=color, linewidth=1.0, alpha=0.5,
                 zorder=1)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of samples $m$')
    ax1.set_ylabel('Mean ECE estimation error')
    ax1.set_title('(a) Error vs. $m$ by Pipeline Depth')
    ax1.legend(fontsize=8, loc='lower left', framealpha=0.9)

    # ---- Right panel: cost-to-reach-delta vs K ----
    Ks_arr = np.array(Ks, dtype=float)
    m_required = np.array([cost_vs_K[str(K)]['m_required'] for K in Ks])

    ax2.scatter(Ks_arr, m_required, s=60, color=COLORS['magenta'],
                edgecolors='white', linewidths=0.5, zorder=3)

    # Exponential fit line
    slope_emp = exp_fit['empirical']['slope']
    base_emp = exp_fit['empirical']['base']
    K_fine = np.linspace(0.8, 5.2, 100)
    fit_line = np.exp(np.log(m_required[0]) + slope_emp * (K_fine - 1))
    ax2.plot(K_fine, fit_line, '-', color=COLORS['magenta'], linewidth=1.5,
             label=f'Empirical fit (base={base_emp:.2f})', zorder=2)

    # L^K reference slope
    L_stage = params['L_stage']
    log_L = exp_fit['reference']['slope_log_L']
    ref_line = m_required[0] * np.exp(log_L * (K_fine - 1))
    ax2.plot(K_fine, ref_line, '--', color=COLORS['gray'], linewidth=1.2,
             label=f'$L^K$ reference ($L$={L_stage})', zorder=1)

    ax2.set_yscale('log')
    ax2.set_xlabel('Pipeline depth $K$')
    ax2.set_ylabel('Samples to reach $\\delta$')
    ax2.set_title('(b) Cost vs. Pipeline Depth')
    ax2.set_xticks(Ks)
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout(w_pad=3.0)

    out_pdf = os.path.join(FIGURES, 'fig_compositional.pdf')
    out_png = os.path.join(FIGURES, 'fig_compositional.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")


# ──────────────────────────────────────────────
# Figure 4: fig_leaderboard_noise  (validates demolition)
# ──────────────────────────────────────────────
def figure4():
    print("Generating Figure 4: fig_leaderboard_noise ...")
    with open(os.path.join(RESULTS, 'leaderboard_noise.json')) as f:
        data = json.load(f)

    summary = data['summary']
    subjects = data['subjects']

    # Collect pairwise gaps and verification status
    all_gaps = []
    all_verifiable = []
    all_delta_floors = []
    for sname, sdata in subjects.items():
        for pair, pdata in sdata['pairwise'].items():
            all_gaps.append(pdata['gap'])
            all_verifiable.append(pdata['verifiable'])
            all_delta_floors.append(pdata['delta_floor'])

    all_gaps = np.array(all_gaps)
    all_verifiable = np.array(all_verifiable)
    all_delta_floors = np.array(all_delta_floors)

    median_floor = summary['median_delta_floor']

    # Collect subject-level data for scatter
    subj_sizes = []
    subj_instabilities = []
    subj_fully_rankable = []
    for sname, sdata in subjects.items():
        subj_sizes.append(sdata['n_items_avg'])
        subj_instabilities.append(sdata['instability'])
        all_v = all(pdata['verifiable'] for pdata in sdata['pairwise'].values())
        subj_fully_rankable.append(all_v)

    subj_sizes = np.array(subj_sizes)
    subj_instabilities = np.array(subj_instabilities)
    subj_fully_rankable = np.array(subj_fully_rankable)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # ---- Left panel: histogram of pairwise accuracy gaps ----
    bins_edges = np.linspace(0, max(all_gaps) * 1.05, 30)

    # Separate into above and below floor
    above = all_gaps[all_verifiable]
    below = all_gaps[~all_verifiable]

    ax1.hist([below, above], bins=bins_edges,
             color=[COLORS['magenta'], COLORS['blue']], alpha=0.8,
             label=[f'Below floor ({len(below)})', f'Above floor ({len(above)})'],
             edgecolor='white', linewidth=0.4, stacked=True, zorder=2)

    # Vertical line at median verification floor
    ax1.axvline(median_floor, color=COLORS['orange'], linewidth=2.0,
                linestyle='--', label=f'Median $\\delta_{{\\mathrm{{floor}}}}$ = {median_floor:.3f}',
                zorder=4)

    ax1.set_xlabel('Pairwise Accuracy Gap')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) Distribution of Pairwise Gaps')
    ax1.legend(fontsize=7.5, framealpha=0.9)

    # ---- Right panel: subject size vs ranking instability ----
    # Color by fully rankable
    colors_scatter = np.where(subj_fully_rankable, COLORS['blue'], COLORS['magenta'])

    ax2.scatter(subj_sizes, subj_instabilities, s=35, c=colors_scatter,
                alpha=0.7, edgecolors='white', linewidths=0.3, zorder=3)

    # Reference curve 1/sqrt(n)
    n_range = np.linspace(80, max(subj_sizes) * 1.05, 200)
    ref_curve = 1.0 / np.sqrt(n_range)
    # Scale reference to data range
    scale_factor = np.median(subj_instabilities) / np.median(1.0 / np.sqrt(subj_sizes))
    ax2.plot(n_range, scale_factor * ref_curve, '--', color=COLORS['gray'],
             linewidth=1.5, label=r'$\propto 1/\sqrt{n}$ reference', zorder=1)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['blue'],
               markersize=7, label='Fully rankable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['magenta'],
               markersize=7, label='Not fully rankable'),
        Line2D([0], [0], linestyle='--', color=COLORS['gray'],
               linewidth=1.5, label=r'$\propto 1/\sqrt{n}$ reference'),
    ]
    ax2.legend(handles=legend_elements, fontsize=7.5, framealpha=0.9,
               loc='upper right')

    ax2.set_xlabel('Subject Size (number of items)')
    ax2.set_ylabel('Ranking Instability Fraction')
    ax2.set_title('(b) Size vs. Ranking Instability')

    fig.tight_layout(w_pad=3.0)

    out_pdf = os.path.join(FIGURES, 'fig_leaderboard_noise.pdf')
    out_png = os.path.join(FIGURES, 'fig_leaderboard_noise.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  Saved: {out_pdf}")
    print(f"  Saved: {out_png}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Regenerating publication-quality figures")
    print("=" * 60)
    figure1()
    figure2()
    figure3()
    figure4()
    print("=" * 60)
    print("All figures generated successfully.")
    print("=" * 60)
