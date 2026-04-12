"""
fig_sun_comparison.py
Produces the key figure comparing the verification tax's epsilon-dependent
minimax rates against the epsilon-independent rate of Sun et al.

Left panel:  Minimax estimation error vs epsilon (fixed m = 10,000)
Right panel: Required sample size vs epsilon for meaningful verification (delta = epsilon/2)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------- paths ----------
OUT_DIR = "C:/Users/wangz/verification tax/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- style ----------
sns.set_context("paper")
sns.set_palette("colorblind")

# Professional colors
DEEP_BLUE    = "#1B4F72"
VERMILLION   = "#C0392B"
FOREST_GREEN = "#1E8449"
GRAY_REF     = "#7F8C8D"
LIGHT_GRAY   = "#BDC3C7"

plt.rcParams.update({
    "font.size":          11,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.grid":          True,
    "grid.color":         "#CCCCCC",
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "figure.dpi":         300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "text.usetex":        False,
    "mathtext.fontset":   "cm",
})

LW_MAIN = 2.0
LW_REF  = 1.0

# ---------- data ----------
eps = np.logspace(np.log10(0.01), np.log10(0.50), 500)
m = 10_000
L = 1
C_sun = 1.0  # proportionality constant (same for all for fair comparison)

# Left panel curves
sun_rate     = C_sun * (L / m) ** (1 / 3) * np.ones_like(eps)   # flat
passive_rate = (L * eps / m) ** (1 / 3)
active_rate  = np.sqrt(eps / m)

# Right panel curves: required m for delta = eps/2
m_passive = 8 * L / eps**2
m_active  = 4 / eps

# ---------- figure ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ===================================================================
#  LEFT PANEL — Minimax estimation error vs epsilon
# ===================================================================
ax1.loglog(eps, sun_rate,     color=DEEP_BLUE,   lw=LW_MAIN,
           label=r"Sun et al.  $(L/m)^{1/3}$", zorder=3)
ax1.loglog(eps, passive_rate, color=VERMILLION,   lw=LW_MAIN,
           label=r"Our passive  $(L\varepsilon/m)^{1/3}$", zorder=3)
ax1.loglog(eps, active_rate,  color=FOREST_GREEN, lw=LW_MAIN,
           label=r"Our active  $\sqrt{\varepsilon/m}$", zorder=3)

ax1.set_xlim(0.01, 0.50)
ax1.set_ylim(5e-4, 0.12)

# Reference vertical lines — label at top of panel using transform
ax1.axvline(0.30, color=GRAY_REF, ls="--", lw=LW_REF, alpha=0.6, zorder=1)
ax1.axvline(0.05, color=GRAY_REF, ls="--", lw=LW_REF, alpha=0.6, zorder=1)

# Labels near top of each vline (in data coords)
ax1.text(0.30, 0.095, "General regime\n(Sun et al.)", fontsize=7,
         ha="center", va="bottom", color=GRAY_REF, style="italic")
ax1.text(0.05, 0.095, "Frontier\nmodels", fontsize=7,
         ha="center", va="bottom", color=GRAY_REF, style="italic")

# Highlight the gap at epsilon = 0.05 with a double-headed arrow
eps_gap = 0.05
sun_at_gap     = C_sun * (L / m) ** (1 / 3)
passive_at_gap = (L * eps_gap / m) ** (1 / 3)
active_at_gap  = np.sqrt(eps_gap / m)

# Arrow from passive rate up to Sun rate at eps = 0.05
ax1.annotate("",
             xy=(eps_gap * 1.15, sun_at_gap), xytext=(eps_gap * 1.15, passive_at_gap),
             arrowprops=dict(arrowstyle="<->", color="#E74C3C", lw=1.2,
                             shrinkA=2, shrinkB=2))
ratio_val = sun_at_gap / passive_at_gap
ax1.text(eps_gap * 1.35, np.sqrt(sun_at_gap * passive_at_gap),
         f"{ratio_val:.1f}$\\times$\ngap",
         fontsize=7.5, color="#E74C3C", ha="left", va="center", fontweight="bold")

ax1.set_xlabel(r"Error rate $\varepsilon$")
ax1.set_ylabel("Minimax estimation error")
ax1.set_title("(a)  Estimation error vs. error rate",
              fontsize=11, fontweight="bold", pad=10)
ax1.legend(loc="lower right", framealpha=0.92, edgecolor="none",
           borderpad=0.6, handlelength=1.8)

# ===================================================================
#  RIGHT PANEL — Required sample size vs epsilon
# ===================================================================
ax2.loglog(eps, m_passive, color=VERMILLION,   lw=LW_MAIN,
           label=r"Passive: $m = 8L/\varepsilon^{2}$", zorder=3)
ax2.loglog(eps, m_active,  color=FOREST_GREEN, lw=LW_MAIN,
           label=r"Active: $m = 4/\varepsilon$", zorder=3)

ax2.set_xlim(0.01, 0.50)
ax2.set_ylim(5, 2e7)

# Horizontal benchmark lines
ax2.axhline(1e4, color=GRAY_REF, ls=":", lw=LW_REF, alpha=0.65, zorder=1)
ax2.axhline(1e5, color=GRAY_REF, ls=":", lw=LW_REF, alpha=0.65, zorder=1)
ax2.text(0.42, 1.20e4, "$m = 10\\!,\\!000$\n(typical benchmark)", fontsize=7,
         ha="right", va="bottom", color=GRAY_REF, linespacing=1.2)
ax2.text(0.42, 1.20e5, "$m = 100\\!,\\!000$\n(large benchmark)", fontsize=7,
         ha="right", va="bottom", color=GRAY_REF, linespacing=1.2)

# Shade the infeasible region (above 100k)
ax2.fill_between(eps, 1e5, 2e7, color="#F2F3F4", alpha=0.6, zorder=0)
ax2.text(0.10, 4e6, "Infeasible region", fontsize=8, color=GRAY_REF,
         ha="center", va="center", style="italic")

# Annotate specific epsilon values on the passive curve
annotations = [
    (0.05, r"$\varepsilon\!=\!0.05$" + "\n" + r"$m_{\mathrm{pass}}\!=\!3{,}200$",
     2.2, 0.18),   # offset_x factor, offset_y factor
    (0.01, r"$\varepsilon\!=\!0.01$" + "\n" + r"$m_{\mathrm{pass}}\!=\!80{,}000$",
     2.5, 3.5),
]

for eps_val, label, ox, oy in annotations:
    m_val = 8 * L / eps_val**2
    ax2.plot(eps_val, m_val, "o", color=VERMILLION, ms=5, zorder=5,
             markeredgecolor="white", markeredgewidth=0.5)
    ax2.annotate(label,
                 xy=(eps_val, m_val),
                 xytext=(eps_val * ox, m_val * oy),
                 fontsize=7.5, color=VERMILLION,
                 arrowprops=dict(arrowstyle="->", color=VERMILLION,
                                 lw=0.8, connectionstyle="arc3,rad=-0.15"),
                 ha="left", va="center", linespacing=1.15)

# Also mark the "verification horizons" — where curves cross the benchmark lines
# Passive crosses 10k at eps = sqrt(8L/1e4) = sqrt(8e-4) ~ 0.0283
# Passive crosses 100k at eps = sqrt(8L/1e5) ~ 0.00894
# Active crosses 10k at eps = 4/1e4 = 4e-4 (off chart)
eps_cross_10k  = np.sqrt(8 * L / 1e4)
eps_cross_100k = np.sqrt(8 * L / 1e5)

ax2.plot(eps_cross_10k, 1e4, "s", color=VERMILLION, ms=6, zorder=5,
         markeredgecolor="white", markeredgewidth=0.6)
ax2.annotate(r"$\varepsilon^{*}\!\approx\!0.028$" + "\nverification\nhorizon",
             xy=(eps_cross_10k, 1e4),
             xytext=(eps_cross_10k * 2.5, 1e4 * 0.06),
             fontsize=7, color=VERMILLION, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=VERMILLION, lw=0.7,
                             connectionstyle="arc3,rad=0.25"),
             linespacing=1.1)

ax2.set_xlabel(r"Error rate $\varepsilon$")
ax2.set_ylabel(r"Required samples $m$ for $\delta = \varepsilon/2$")
ax2.set_title("(b)  Sample cost of meaningful verification",
              fontsize=11, fontweight="bold", pad=10)
ax2.legend(loc="upper right", framealpha=0.92, edgecolor="none",
           borderpad=0.6, handlelength=1.8)

# ---------- layout + save ----------
plt.tight_layout(w_pad=3.0)

fig.savefig(os.path.join(OUT_DIR, "fig_sun_comparison.pdf"), format="pdf")
fig.savefig(os.path.join(OUT_DIR, "fig_sun_comparison.png"), format="png", dpi=300)
plt.close(fig)

print("Saved:")
print(f"  {os.path.join(OUT_DIR, 'fig_sun_comparison.pdf')}")
print(f"  {os.path.join(OUT_DIR, 'fig_sun_comparison.png')}")
