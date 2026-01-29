import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import string

DATASET_N_CLASSES = {
    "bbc-text": 5,
    "huffpost-news": 42,
    "ag-news": 4,
    "yahoo": 10,
}


def make_muted_discrete_cmap(n_classes: int, sat: float = 0.45, val: float = 0.90):
    colors = []
    for i in range(n_classes):
        h = i / n_classes
        r, g, b = colorsys.hsv_to_rgb(h, sat, val)
        colors.append((r, g, b))
    return mpl.colors.ListedColormap(colors, name="muted_hsv")


def visualize_many_tsne(
        paths,
        out_path,
        dataset_names,
        titles=None,
        y_true_dict=None,
        random_state=0,
        perplexity=30.0,
        point_size=6.0,
        alpha=0.75,
        ncols=3,
        caption_y=-0.14,
        caption_fontsize=10,
):
    assert len(paths) == len(dataset_names), "paths and dataset_names must be same length."
    M = len(paths)
    if titles is None:
        titles = [f"Panel {i + 1}" for i in range(M)]

    panel_labels = [f"({c})" for c in string.ascii_lowercase[:M]]
    X_list = [np.load(p) for p in paths]

    nrows = int(np.ceil(M / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.8 * nrows), dpi=200)
    axes = np.array(axes).reshape(-1)

    for i, (ax, X, dname, title, lbl) in enumerate(zip(axes, X_list, dataset_names, titles, panel_labels)):
        n_classes = int(DATASET_N_CLASSES[dname])

        cmap = make_muted_discrete_cmap(n_classes, sat=0.45, val=0.90)
        boundaries = np.arange(-0.5, n_classes + 0.5, 1.0)
        norm = mpl.colors.BoundaryNorm(boundaries, n_classes)

        if y_true_dict is not None and dname in y_true_dict and y_true_dict[dname] is not None:
            y = y_true_dict[dname]
        else:
            Xs_km = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=n_classes, random_state=random_state, n_init="auto")
            y = km.fit_predict(Xs_km)

        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, (X.shape[0] - 1) / 3.0),
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        Z = tsne.fit_transform(StandardScaler().fit_transform(X))

        ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, norm=norm, s=point_size, alpha=alpha, linewidths=0)

        if " - " in title:
            method_name = title.split(" - ", 1)[1]
        else:
            method_name = title

        caption = r"$\mathbf{" + lbl + "}$ " + method_name
        ax.text(0.5, caption_y, caption, transform=ax.transAxes, ha="center", va="top", fontsize=caption_fontsize)

        ax.set_xticks([]);
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # --- HEADLINES (Dataset names) ---
    row_titles = [
        "BBC News",
        "AG News",
        "HuffPost News",
        "Yahoo! Answers"
    ]

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.15, top=0.92, bottom=0.08)

    for r in range(nrows):
        first_ax_in_row = axes[r * ncols]
        pos = first_ax_in_row.get_position()


        fig.text(
            0.5,
            pos.y1 + 0.02,
            row_titles[r],
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold"
        )

    for j in range(M, len(axes)):
        axes[j].axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    # Path list
    paths = [
        r"~\embedding_paths",
    ]

    # path order for image
    order = ['''3, 4, 5, 6, 7, 8, 0, 1, 2, 9, 10, 11''']
    paths_12 = [paths[i] for i in order]

    dataset_names_12 = (["bbc-text"] * 3 + ["ag-news"] * 3 + ["huffpost-news"] * 3 + ["yahoo"] * 3)

    titles_12 = [
        "BBC News - Base-SBERT-Clustering", "BBC News - UMAP-Clustering", "BBC News - Single-E-Clust",
        "AG News - Base-SBERT-Clustering", "AG News - UMAP-Clustering", "AG News - Single-E-Clust",
        "HuffPost News - Base-SBERT-Clustering", "HuffPost News - UMAP-Clustering", "HuffPost News - Single-E-Clust",
        "Yahoo! Answers - Base-SBERT-Clustering", "Yahoo! Answers - UMAP-Clustering", "Yahoo! Answers - Single-E-Clust",
    ]

    visualize_many_tsne(
        paths=paths_12,
        out_path=r"~\output_path",
        dataset_names=dataset_names_12,
        titles=titles_12,
        point_size=5.0,
        alpha=0.70,
        ncols=3
    )