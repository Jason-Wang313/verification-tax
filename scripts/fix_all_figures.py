"""Fix all 5 figure issues reported by user."""
import json, numpy as np, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "C:/Users/wangz/verification tax/figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3
})
C = ['#648FFF', '#DC267F', '#FE6100']

# ====== FIG 2: Leaderboard noise — BIGGER right panel ======
data = json.load(open('C:/Users/wangz/verification tax/results/analysis/leaderboard_noise.json'))
gaps_below = [g['gap'] for g in data['all_pairwise_gaps'] if not g['verifiable']]
gaps_above = [g['gap'] for g in data['all_pairwise_gaps'] if g['verifiable']]
median_floor = data['summary']['median_delta_floor']
sizes = [s['n_items_avg'] for s in data['subjects'].values()]
inst = [s['instability'] for s in data['subjects'].values()]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
bins = np.linspace(0, 0.8, 35)
ax1.hist(gaps_below, bins=bins, color='#DC267F', alpha=0.85, label=f'Below floor ({len(gaps_below)})')
ax1.hist(gaps_above, bins=bins, color='#648FFF', alpha=0.85, label=f'Above floor ({len(gaps_above)})')
ax1.axvline(median_floor, color='#FE6100', lw=2.5, ls='--', label=f'Median floor = {median_floor:.3f}')
ax1.set_xlabel('Pairwise Accuracy Gap'); ax1.set_ylabel('Count')
ax1.set_title('(a) Distribution of Pairwise Gaps'); ax1.legend(fontsize=11)

ax2.scatter(sizes, inst, c='#DC267F', s=90, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)
x_ref = np.linspace(80, 1700, 200)
ax2.plot(x_ref, 4.5/np.sqrt(x_ref), '--', color='gray', lw=1.5, alpha=0.6)
ax2.set_xlabel('Subject Size (number of items)'); ax2.set_ylabel('Ranking Instability Fraction')
ax2.set_title('(b) Size vs. Ranking Instability')
plt.tight_layout(w_pad=3)
plt.savefig(f'{OUT}/fig_leaderboard_noise.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig_leaderboard_noise.png', dpi=300, bbox_inches='tight')
plt.close(); print('Fig 2 done')

# ====== FIG 3: Active real — L-hat in x-labels, not floating ======
data = json.load(open('C:/Users/wangz/verification tax/results/analysis/active_real_results.json'))
ms_list = data['m_values']; md = data['models']; mnames = list(md.keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
for i, model in enumerate(mnames):
    p = [md[model]['passive'][str(m)]['mean'] for m in ms_list]
    a = [md[model]['active'][str(m)]['mean'] for m in ms_list]
    ax1.loglog(ms_list, p, '-o', color=C[i], ms=5, lw=1.8, alpha=0.85)
    ax1.loglog(ms_list, a, '--^', color=C[i], ms=5, lw=1.8, alpha=0.85)
ms_r = np.array([80, 12000])
ax1.loglog(ms_r, 0.35*ms_r**(-1/3), ':', color='gray', lw=1.2)
ax1.loglog(ms_r, 0.18*ms_r**(-1/2), ':', color='gray', lw=1.2)
ax1.text(9000, 0.35*9000**(-1/3)*1.4, '$m^{-1/3}$', fontsize=9, color='gray')
ax1.text(9000, 0.18*9000**(-1/2)*0.65, '$m^{-1/2}$', fontsize=9, color='gray')
leg = [Line2D([0],[0], color='dimgray', ls='-', marker='o', ms=5, lw=1.8, label='Passive (3 models)'),
       Line2D([0],[0], color='dimgray', ls='--', marker='^', ms=5, lw=1.8, label='Active (3 models)'),
       Line2D([0],[0], color='gray', ls=':', lw=1.2, label='Theory references')]
ax1.legend(handles=leg, loc='lower left', fontsize=10); ax1.set_xlabel('Number of samples $m$')
ax1.set_ylabel('Mean ECE estimation error'); ax1.set_title('(a) Active vs. Passive on Real MMLU')

m_key = '2000'; x = np.arange(len(mnames)); w = 0.32
pv = [md[m]['passive'][m_key]['mean'] for m in mnames]
av = [md[m]['active'][m_key]['mean'] for m in mnames]
ax2.bar(x-w/2, pv, w, color=C, alpha=0.85, label='Passive')
ax2.bar(x+w/2, av, w, color=C, alpha=0.5, hatch='///', edgecolor='gray', label='Active')
lhats = [md[m]['L_hat'] for m in mnames]
xlabels = [f'405B\n($\\hat{{L}}$={lhats[0]:.1f})', f'Maverick\n($\\hat{{L}}$={lhats[1]:.1f})', f'Qwen3-80B\n($\\hat{{L}}$={lhats[2]:.1f})']
ax2.set_xticks(x); ax2.set_xticklabels(xlabels, fontsize=9)
ax2.set_ylabel('Mean ECE estimation error'); ax2.set_title('(b) Error at $m$=2,000')
ax2.legend(fontsize=10)
plt.tight_layout(w_pad=3)
plt.savefig(f'{OUT}/fig_active_real.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig_active_real.png', dpi=300, bbox_inches='tight')
plt.close(); print('Fig 3 done')

# ====== FIG 5: Sun comparison — move grey text to bottom ======
eps = np.logspace(np.log10(0.01), np.log10(0.50), 500); m=10000; L=1
sun = (L/m)**(1/3)*np.ones_like(eps); pas = (L*eps/m)**(1/3); act = np.sqrt(eps/m)
m_pas = 8*L/eps**2; m_act = 4/eps

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
ax1.loglog(eps, sun, color='#1B4F72', lw=2, label='Sun et al.  $(L/m)^{1/3}$')
ax1.loglog(eps, pas, color='#C0392B', lw=2, label='Our passive  $(L\\varepsilon/m)^{1/3}$')
ax1.loglog(eps, act, color='#1E8449', lw=2, label='Our active  $\\sqrt{\\varepsilon/m}$')
ax1.set_xlim(0.01, 0.50); ax1.set_ylim(5e-4, 0.12)
ax1.axvline(0.30, color='gray', ls='--', lw=0.8, alpha=0.4)
ax1.axvline(0.05, color='gray', ls='--', lw=0.8, alpha=0.4)
# Labels at BOTTOM to avoid overlap with top
ax1.text(0.30, 6e-4, 'General', fontsize=7, ha='center', color='gray', style='italic')
ax1.text(0.05, 6e-4, 'Frontier', fontsize=7, ha='center', color='gray', style='italic')
sun_at = (L/m)**(1/3); pas_at = (L*0.05/m)**(1/3)
ax1.annotate('', xy=(0.065, sun_at), xytext=(0.065, pas_at),
    arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=1.2, shrinkA=2, shrinkB=2))
ax1.text(0.08, np.sqrt(sun_at*pas_at), f'{sun_at/pas_at:.1f}$\\times$ gap', fontsize=8, color='#E74C3C', fontweight='bold')
ax1.set_xlabel('Error rate $\\varepsilon$'); ax1.set_ylabel('Minimax estimation error')
ax1.set_title('(a) Estimation error vs. error rate', fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)

ax2.loglog(eps, m_pas, color='#C0392B', lw=2, label='Passive: $m=8L/\\varepsilon^2$')
ax2.loglog(eps, m_act, color='#1E8449', lw=2, label='Active: $m=4/\\varepsilon$')
ax2.set_xlim(0.01, 0.50); ax2.set_ylim(5, 2e7)
ax2.axhline(1e4, color='gray', ls=':', lw=0.8, alpha=0.5)
ax2.axhline(1e5, color='gray', ls=':', lw=0.8, alpha=0.5)
ax2.text(0.45, 1.3e4, '$m$=10k', fontsize=8, ha='right', color='gray')
ax2.text(0.45, 1.3e5, '$m$=100k', fontsize=8, ha='right', color='gray')
ax2.fill_between(eps, 1e5, 2e7, color='#F2F3F4', alpha=0.3, zorder=0)
for ev in [0.05, 0.01]:
    ax2.plot(ev, 8*L/ev**2, 'o', color='#C0392B', ms=6, zorder=5, markeredgecolor='white', markeredgewidth=0.8)
ax2.set_xlabel('Error rate $\\varepsilon$'); ax2.set_ylabel('Required samples $m$')
ax2.set_title('(b) Sample cost of meaningful verification', fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout(w_pad=3)
plt.savefig(f'{OUT}/fig_sun_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig_sun_comparison.png', dpi=300, bbox_inches='tight')
plt.close(); print('Fig 5 done')

# ====== FIG 6: All benchmarks — NO value labels, NO "floor>0.05" text ======
summary = json.load(open('C:/Users/wangz/verification tax/results/analysis/all_benchmarks_summary.json'))
bf = summary['mean_verification_floor_per_benchmark']
names = list(bf.keys()); floors = [bf[n] for n in names]
colors_bar = ['#648FFF', '#FE6100', '#1E8449', '#785EF0', '#DC267F']

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(names, floors, color=colors_bar[:len(names)], alpha=0.85, edgecolor='white', linewidth=0.5)
ax.axhline(0.05, color='red', ls='--', lw=2, alpha=0.7, label='$\\delta$=0.05 (typical improvement)')
ax.set_ylabel('Verification floor $\\delta_{\\mathrm{floor}}$', fontsize=12)
ax.set_title('Verification Floor by Benchmark', fontsize=13)
ax.set_ylim(0, max(floors)*1.15)
ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_all_benchmarks.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig_all_benchmarks.png', dpi=300, bbox_inches='tight')
plt.close(); print('Fig 6 done')

# ====== FIG 8: Pipeline — CLEAN, no value annotations ======
pd = json.load(open('C:/Users/wangz/verification tax/results/analysis/pipeline_real_results.json'))
sub = pd['subsampling']
ms = [m for m in sub.get('m_values', [100,200,500,1000,2000,5000,10000]) if str(m) in sub['pipeline']]
pe = [sub['pipeline'][str(m)]['mean_abs_error'] for m in ms]
se = [sub['single_model'][str(m)]['mean_abs_error'] for m in ms]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.loglog(ms, pe, '-s', color='#FE6100', lw=2, ms=6, label='2-stage pipeline', markeredgecolor='white')
ax1.loglog(ms, se, '-o', color='#648FFF', lw=2, ms=6, label='Single model (405B)', markeredgecolor='white')
ms_a = np.array(ms)
pL = pd['pipeline']['L_hat']; pE = pd['pipeline']['eps']
sL = pd['single_model']['L_hat']; sE = pd['single_model']['eps']
ax1.loglog(ms_a, (pL*pE/ms_a)**(1/3), '--', color='#FE6100', lw=1, alpha=0.4)
ax1.loglog(ms_a, (sL*sE/ms_a)**(1/3), '--', color='#648FFF', lw=1, alpha=0.4)
ax1.set_xlabel('Sample size $m$'); ax1.set_ylabel('Mean $|\\hat{ECE} - ECE_{true}|$')
ax1.set_title('(a) Pipeline vs. single-model error'); ax1.legend(fontsize=10)

# Right panel: simple comparison, NO messy annotations
labels = ['Single model\n(Llama-405B)', '2-stage\npipeline']
l_vals = [sL, pL]
x = np.arange(2)
ax2.bar(x, l_vals, 0.5, color=['#648FFF', '#FE6100'], alpha=0.85, edgecolor='white')
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel('$\\hat{L}$ (Lipschitz constant)', fontsize=12)
ax2.set_title('(b) System Lipschitz: single vs. pipeline')
# Annotate ratio
ax2.text(0.5, max(l_vals)*0.5, f'{l_vals[1]/l_vals[0]:.1f}$\\times$', fontsize=14, ha='center',
         fontweight='bold', color='#333333')
plt.tight_layout(w_pad=3)
plt.savefig(f'{OUT}/fig_pipeline_real.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig_pipeline_real.png', dpi=300, bbox_inches='tight')
plt.close(); print('Fig 8 done')

# Standalone appendix panels for the real pipeline figure
fig_err, ax_err = plt.subplots(figsize=(8.5, 4.8))
ax_err.loglog(ms, pe, '-s', color='#FE6100', lw=2, ms=7, label='2-stage pipeline', markeredgecolor='white')
ax_err.loglog(ms, se, '-o', color='#648FFF', lw=2, ms=7, label='Single model (405B)', markeredgecolor='white')
ax_err.loglog(ms_a, (pL*pE/ms_a)**(1/3), '--', color='#FE6100', lw=1.2, alpha=0.4)
ax_err.loglog(ms_a, (sL*sE/ms_a)**(1/3), '--', color='#648FFF', lw=1.2, alpha=0.4)
ax_err.set_xlabel('Sample size $m$')
ax_err.set_ylabel('Mean $|\\hat{ECE} - ECE_{true}|$')
ax_err.set_title('Pipeline vs. Single-Model Error', fontweight='bold')
ax_err.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_pipeline_real_error.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig_pipeline_real_error.png', dpi=300, bbox_inches='tight')
plt.close(fig_err)

fig_lip, ax_lip = plt.subplots(figsize=(7.5, 4.8))
ax_lip.bar(x, l_vals, 0.5, color=['#648FFF', '#FE6100'], alpha=0.85, edgecolor='white')
ax_lip.set_xticks(x)
ax_lip.set_xticklabels(labels, fontsize=10)
ax_lip.set_ylabel('$\\hat{L}$ (Lipschitz constant)', fontsize=12)
ax_lip.set_title('System Lipschitz: Single vs. Pipeline', fontweight='bold')
ax_lip.text(0.5, max(l_vals)*0.5, f'{l_vals[1]/l_vals[0]:.1f}$\\times$', fontsize=14, ha='center',
            fontweight='bold', color='#333333')
plt.tight_layout()
plt.savefig(f'{OUT}/fig_pipeline_real_lipschitz.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT}/fig_pipeline_real_lipschitz.png', dpi=300, bbox_inches='tight')
plt.close(fig_lip)

print('\nALL 5 FIGURES FIXED')
