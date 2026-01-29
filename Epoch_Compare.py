import matplotlib.pyplot as plt
import numpy as np
import os

SAVE_DIR = r"~\output_dir"
metrics = ["SIL", "ARI", "NMI", "ACC"]

#The tables can be modified by adjusting the number of epochs in the experimental setup.

data_1epoch = {
    "BBC News":   [0.72, 0.89, 0.87, 0.96],
    "AG News": [0.64, 0.61, 0.59, 0.82],
    "Huffpost News":   [0.48, 0.17, 0.34, 0.32],
    "Yahoo":      [0.61, 0.39, 0.46, 0.63],
}

data_5epoch = {
    "BBC News":   [0.89, 0.87, 0.85, 0.95],
    "AG News": [0.82, 0.56, 0.55, 0.80],
    "Huffpost News":   [0.62, 0.17, 0.35, 0.32],
    "Yahoo":      [0.82, 0.39, 0.46, 0.62],
}

data_30epoch = {
    "BBC News":   [0.93, 0.87, 0.84, 0.94],
    "AG News": [0.92, 0.54, 0.53, 0.79],
    "Huffpost News":   [0.92, 0.17, 0.34, 0.31],
    "Yahoo":      [0.94, 0.37, 0.45, 0.61],
}

os.makedirs(SAVE_DIR, exist_ok=True)

# datasets = sorted(set(data_1epoch) & set(data_5epoch) & set(data_30epoch))

order = ["BBC News", "AG News", "Huffpost News", "Yahoo"]
datasets = [ds for ds in order if ds in data_1epoch and ds in data_5epoch and ds in data_30epoch]


x = np.arange(len(metrics))
width = 0.22

fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), dpi=200)
axes = axes.flatten()

for i, ds in enumerate(datasets):
    ax = axes[i]

    v1 = data_1epoch[ds]
    v5 = data_5epoch[ds]
    v30 = data_30epoch[ds]

    ax.bar(x - width, v1, width, label="1 epoch", color="#4C72B0",
           edgecolor="black", linewidth=0.6)
    ax.bar(x,         v5, width, label="5 epoch", color="#55A868",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width, v30, width, label="30 epoch", color="#DD8452",
           edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Skor")
    ax.set_title(ds)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for j in range(len(datasets), len(axes)):
    axes[j].axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=3,
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

out_file = os.path.join(SAVE_DIR, "epoch_comparison.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] saved: {out_file}")
