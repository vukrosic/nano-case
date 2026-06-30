"""Generate the report figures from the verified benchmark numbers.

Numbers are the locked 3-seed results (seeds 0/1/2, held-out seed 987654321,
N=4000): overall 99.8%/61.8%, smushed slice (N=1410) 99.7%/8.2%; OOD probe
in-vocab 100%, OOV 2%, 5-6 word chains 33%.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1a1a1a"
MODEL = "#2563eb"   # blue
REGEX = "#9ca3af"   # gray
plt.rcParams.update({"font.family": "serif", "font.size": 11,
                     "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK})

# ---- Fig 1: model vs regex, overall + the regex-killer slice ----
fig, ax = plt.subplots(figsize=(6.4, 3.2))
groups = ["Overall\n(N=4000)", "Smushed slice\n(N=1410)"]
model = [99.8, 99.7]
regex = [61.8, 8.2]
x = range(len(groups))
w = 0.36
b1 = ax.bar([i - w / 2 for i in x], model, w, label="nano-case (model)", color=MODEL)
b2 = ax.bar([i + w / 2 for i in x], regex, w, label="standard regex converter", color=REGEX)
ax.set_ylabel("exact-match accuracy (%)")
ax.set_ylim(0, 115)
ax.set_xticks(list(x)); ax.set_xticklabels(groups)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          ncol=2, fontsize=9.5)
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig1_model_vs_regex.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Fig 2: OOD breaking-point probe (seed-0 model) ----
fig, ax = plt.subplots(figsize=(6.4, 2.9))
labels = ["in-vocab smushed\n(trained slice)", "out-of-vocab\nwords smushed", "5-6 word chains\n(longer than trained)"]
vals = [100, 2, 33]
colors = [MODEL, "#dc2626", "#f59e0b"]
bars = ax.bar(labels, vals, color=colors, width=0.6)
ax.set_ylabel("accuracy (%)")
ax.set_ylim(0, 109)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%",
            ha="center", va="bottom", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig("fig2_ood.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig1_model_vs_regex.png, fig2_ood.png")
