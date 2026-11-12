"""Recreate figures/wrn_kd_curves.pdf (report figure): held-out accuracy of the WRN-28 student
with and without distillation, and of its WRN-28-10 teacher, from the logged metrics.

    python plot_kd_curves.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [("s45", "WRN-28 student alone", "#7f7f7f", -7),
        ("kd45", "WRN-28 + KD", "#3a75c4", 1),
        ("t45", "WRN-28-10 teacher (36.5M)", "#d1495b", -2)]

plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.3, 2.1))
for name, label, color, dy in RUNS:
    hist = json.load(open(os.path.join(HERE, "results", f"{name}_metrics.json")))["history"]
    pts = [r for r in hist if "val_acc_ema" in r]
    x, y = [r["epoch"] for r in pts], [100 * r["val_acc_ema"] for r in pts]
    ax.plot(x, y, color=color, lw=1.4, label=label)
    ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(3, dy), textcoords="offset points", fontsize=7, color=color)
ax.set_xlim(0, 275)
ax.set_ylim(88, 98.6)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy (%)")
ax.set_title("EMA validation accuracy, single view", fontsize=8.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=7)
fig.tight_layout(pad=0.3)
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "wrn_kd_curves.pdf"))
print("saved figures/wrn_kd_curves.pdf")
