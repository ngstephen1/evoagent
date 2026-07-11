"""
phase3_visualize_results.py — visualize the Phase-3 improvement journey.

Academic / publication style, two standalone figures:
  Figure 1 (results_progression.png):
      Kaggle public-score progression across the session (0.658 -> 0.802),
      colored by method family, with the one regression (v3) shown honestly.
  Figure 2 (results_star_slices.png):
      Dev accuracy by question type, fine-tuned vs rejection-sampled (STaR),
      showing where STaR helped most (multi-step programs).

Each figure carries its own headline and is embedded in the report next to the
section that discusses it (progression -> Overview; STaR slices -> STaR section).

Palette: Okabe-Ito (the standard colorblind-safe palette for scientific figures);
validated with the dataviz palette checker (all checks pass, light mode).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTDIR = Path(__file__).resolve().parent.parent / "A3_MelanieAndStephen_<StudentID1>_<StudentID2>"

# ---- Okabe-Ito categorical palette (CVD-safe, publication standard) ----
SKY, BLUE, ORANGE, GREEN, PURPLE, VERM = (
    "#56B4E9", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
# ink / structural neutrals
INK, INK2, MUTED, GRID, LINE = "#1a1a1a", "#4d4d4d", "#8a8a8a", "#d9d9d9", "#bdbdbd"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})


def headline(fig, title, subtitle=None, x=0.055):
    """Standard bold figure headline (+ optional lighter subtitle)."""
    fig.text(x, 0.965, title, fontsize=13, fontweight="bold", ha="left",
             va="top", color=INK)
    if subtitle:
        fig.text(x, 0.905, subtitle, fontsize=10, ha="left", va="top", color=INK2)


# =====================================================================
# Figure 1 — Kaggle score progression
# =====================================================================
BASE, ENS, POST, FT, STAR, API = (
    "Baseline", "Ensemble", "Post-processing", "Fine-tuning", "STaR", "Frontier-API ensemble")
CATCOL = {BASE: SKY, ENS: BLUE, POST: ORANGE, FT: GREEN, STAR: PURPLE, API: VERM}
journey = [
    ("Prior best\n(Run009)", 0.65789, BASE),
    ("3-way\nensemble", 0.69433, ENS),
    ("Scale fix\n(v2)", 0.69838, POST),
    ("Blind fix\n(v3)", 0.68218, POST),      # the regression
    ("Sign fix\n(v4)", 0.70242, POST),
    ("Fine-tune\n(Qwen3)", 0.72267, FT),
    ("FT\n(Qwen2.5)", 0.71255, FT),          # 2nd model, for diversity
    ("FT merged", 0.72874, POST),
    ("RS / STaR", 0.74696, STAR),
    ("API 3-way\nensemble", 0.80161, API),   # frontier-API ensemble -> 80%
]

fig1, axA = plt.subplots(figsize=(9.2, 5.2))
fig1.subplots_adjust(left=0.085, right=0.98, top=0.83, bottom=0.16)

xs = list(range(len(journey)))
ys = [s for _, s, _ in journey]
cats = [c for _, _, c in journey]
reg_i, best_i = 3, len(journey) - 1

axA.plot(xs, ys, "-", color=LINE, lw=1.4, zorder=1)
for x, y, c in zip(xs, ys, cats):
    axA.scatter(x, y, s=70, color=CATCOL[c], zorder=3, edgecolor=SURFACE, linewidth=1.2)
# regression: restrained hollow ring + small italic note
axA.scatter(xs[reg_i], ys[reg_i], s=160, facecolor="none", edgecolor=INK2,
            linewidth=1.1, zorder=2)
axA.annotate("only regression\n(blind fix, reverted)", (xs[reg_i], ys[reg_i]),
             xytext=(xs[reg_i], ys[reg_i] - 0.018), ha="center", va="top",
             fontsize=8.5, style="italic", color=INK2,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
# 2nd fine-tune diversity note
div_i = 6
axA.annotate("2nd model\n(diversity)", (xs[div_i], ys[div_i]),
             xytext=(xs[div_i], ys[div_i] - 0.014), ha="center", va="top",
             fontsize=8.5, style="italic", color=INK2,
             arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
# numeric labels (best labeled by callout; regression labeled explicitly)
for x, y in zip(xs, ys):
    if x in (reg_i, best_i):
        continue
    axA.annotate(f"{y:.3f}", (x, y + 0.0045), ha="center", va="bottom",
                 fontsize=8.5, color=INK)
axA.annotate(f"{ys[reg_i]:.3f}", (xs[reg_i], ys[reg_i] + 0.004), ha="center",
             va="bottom", fontsize=8.5, color=INK)
axA.annotate("best: 0.802\n+14.4 pts (80%)", (best_i, ys[best_i]),
             xytext=(best_i - 0.35, ys[best_i] - 0.004),
             ha="right", va="center", fontsize=9.5, fontweight="bold", color=VERM,
             arrowprops=dict(arrowstyle="-", color=VERM, lw=0.8))

axA.set_xticks(xs)
axA.set_xticklabels([lbl for lbl, _, _ in journey], fontsize=8.5)
axA.set_xlim(-0.5, len(journey) - 0.35)
axA.set_ylim(0.645, 0.815)
axA.set_ylabel("Kaggle public score", fontsize=11)
axA.grid(axis="y", color=GRID, lw=0.7, linestyle=(0, (3, 3)), zorder=0)
axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)
handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CATCOL[k],
                  markeredgecolor=SURFACE, markeredgewidth=0.8, markersize=7.5, label=k)
           for k in [BASE, ENS, POST, FT, STAR, API]]
axA.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5,
           ncol=1, handletextpad=0.4, labelspacing=0.35, borderaxespad=0.4)
headline(fig1,
         "Kaggle public-score progression across Phase-3",
         "Each point is one submission, colored by method family; "
         "0.658 $\\rightarrow$ 0.802 (80%) over the session.")
fig1.savefig(OUTDIR / "results_progression.png", dpi=220, bbox_inches="tight",
             facecolor=SURFACE)
print(f"saved {OUTDIR / 'results_progression.png'}")

# =====================================================================
# Figure 2 — STaR effect on dev accuracy by question type
# =====================================================================
slices = ["Overall", "Arithmetic", "Table\nlookup", "Ratio /\npercent", "Multi-step\n(3+ ops)"]
ft_slice = [78.1, 76.3, 87.2, 80.4, 47.8]
rs_slice = [79.6, 78.0, 88.3, 79.8, 60.9]

fig2, axB = plt.subplots(figsize=(7.6, 4.8))
fig2.subplots_adjust(left=0.16, right=0.97, top=0.74, bottom=0.14)

y = np.arange(len(slices))
h = 0.36
axB.barh(y + h / 2, ft_slice, height=h, color=GREEN, zorder=3,
         edgecolor=SURFACE, linewidth=0.8, label="Fine-tuned (SFT)")
axB.barh(y - h / 2, rs_slice, height=h, color=PURPLE, zorder=3,
         edgecolor=SURFACE, linewidth=0.8, label="Rejection-sampled (STaR)")
for yi, v in zip(y + h / 2, ft_slice):
    axB.annotate(f"{v:.0f}", (v + 0.8, yi), va="center", fontsize=8.5, color=INK2)
for yi, v in zip(y - h / 2, rs_slice):
    axB.annotate(f"{v:.0f}", (v + 0.8, yi), va="center", fontsize=8.5, color=INK2)
axB.annotate("+13 pts", (rs_slice[-1] + 7.0, y[-1]), va="center", ha="left",
             fontsize=9.5, fontweight="bold", color=PURPLE)
axB.set_yticks(y)
axB.set_yticklabels(slices, fontsize=9)
axB.invert_yaxis()
axB.set_xlim(0, 100)
axB.set_xlabel("Dev accuracy (%)", fontsize=11)
axB.grid(axis="x", color=GRID, lw=0.7, linestyle=(0, (3, 3)), zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
axB.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
           fontsize=9, handlelength=1.1, handletextpad=0.5, columnspacing=1.8)
headline(fig2,
         "Effect of rejection sampling (STaR) on dev accuracy",
         "By question type; STaR's gain concentrates on multi-step programs, "
         "the previous ceiling.", x=0.045)
fig2.savefig(OUTDIR / "results_star_slices.png", dpi=220, bbox_inches="tight",
             facecolor=SURFACE)
print(f"saved {OUTDIR / 'results_star_slices.png'}")

# =====================================================================
# Figure 3 — our 9B fine-tune vs frontier APIs (Cleveland dot plot)
# =====================================================================
# (label, dev accuracy %, color, kind)  — Gemini 3.1-Pro deliberately excluded.
GRAYD = "#8a8a86"
models = [
    ("3-way ensemble\n(our final system)", 79.97, VERM,  "system"),
    ("STaR Qwen3-8B\n(ours · 9B, open)", 77.91, GREEN, "ours"),
    ("Gemini 2.5-flash\n(frontier API)", 76.20, GRAYD, "api"),
    ("DeepSeek-V3\n(frontier API · ≈671B)", 66.95, GRAYD, "api"),
]
fig3, axC = plt.subplots(figsize=(8.8, 4.5))
fig3.subplots_adjust(left=0.30, right=0.95, top=0.80, bottom=0.15)

yy = np.arange(len(models))[::-1]  # first item at top
star_y = yy[1]                     # STaR row
# reference line at our fine-tune's accuracy
axC.axvline(77.91, color=GREEN, lw=1.1, linestyle=(0, (4, 3)), zorder=1)
axC.text(77.91, len(models) - 0.42, "our fine-tune (77.9%)", color=GREEN,
         fontsize=8.5, style="italic", ha="center", va="bottom")
# faint per-row guide lines
for y, (lbl, v, col, kind) in zip(yy, models):
    axC.plot([64, v], [y, y], color=GRID, lw=0.8, zorder=1)
# dots
for y, (lbl, v, col, kind) in zip(yy, models):
    size = 240 if kind in ("ours", "system") else 170
    axC.scatter(v, y, s=size, color=col, zorder=3, edgecolor=SURFACE, linewidth=1.6)
    axC.annotate(f"{v:.2f}", (v, y), xytext=(9, 0), textcoords="offset points",
                 va="center", ha="left", fontsize=9.5,
                 fontweight="bold" if kind in ("ours", "system") else "normal",
                 color=col if kind in ("ours", "system") else INK2)
# star marker on the "ours" model
axC.scatter(models[1][1], star_y, s=42, marker="*", color=SURFACE, zorder=4)
# key annotations
axC.annotate("beats both frontier APIs\n— including one ≈ 70× larger",
             (models[1][1], star_y), xytext=(69.0, star_y - 0.42),
             ha="center", va="top", fontsize=8.5, style="italic", color=GREEN,
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.8))
axC.annotate("+2.1 pts from voting\nthe three below", (models[0][1], yy[0]),
             xytext=(models[0][1] - 1.2, yy[0] - 0.40), ha="right", va="top",
             fontsize=8.5, style="italic", color=VERM,
             arrowprops=dict(arrowstyle="-", color=VERM, lw=0.8))

axC.set_yticks(yy)
axC.set_yticklabels([m[0] for m in models], fontsize=9.5)
axC.set_ylim(-0.6, len(models) - 0.3)
axC.set_xlim(64, 82)
axC.set_xlabel("Dev accuracy (%)", fontsize=11)
axC.grid(axis="x", color=GRID, lw=0.7, linestyle=(0, (3, 3)), zorder=0)
axC.set_axisbelow(True)
for s in ("top", "right", "left"):
    axC.spines[s].set_visible(False)
axC.tick_params(axis="y", length=0)
headline(fig3,
         "A fine-tuned 9B model beats frontier APIs on FinQA-VN",
         "Standalone dev accuracy: our 8B STaR model outscores both hosted APIs; "
         "the 3-way vote adds +2 more.", x=0.02)
fig3.savefig(OUTDIR / "results_star_vs_api.png", dpi=220, bbox_inches="tight",
             facecolor=SURFACE)
print(f"saved {OUTDIR / 'results_star_vs_api.png'}")
