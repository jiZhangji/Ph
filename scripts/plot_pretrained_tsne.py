#!/usr/bin/env python3
"""Plot frozen pretrained SAR representations as combined and individual t-SNE figures."""

from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize


METHODS = (
    ("mae", "MAE"),
    ("lomar", "LoMaR"),
    ("fg_mae", "FG-MAE"),
    ("i_jepa", "I-JEPA"),
    ("sar_jepa", "SAR-JEPA"),
    ("phyd_mae", "PhyD-MAE"),
)
DATASETS = (
    ("MSTAR_SOC", "MSTAR"),
    ("New_FUSAR", "FUSAR-Ship"),
    ("SAR_ACD", "SAR-ACD"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("few_shot_classification/finetune/pretrained_tsne_features"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_visualizations/pretrained_tsne")
    )
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_group(directory):
    group = {}
    for method, display_name in METHODS:
        path = directory / f"{method}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing feature file: {path}")
        with np.load(path, allow_pickle=False) as data:
            group[method] = {
                "display_name": display_name,
                "features": data["features"].astype(np.float32, copy=False),
                "labels": data["labels"].astype(np.int64, copy=False),
                "paths": data["paths"].astype(str),
                "classnames": data["classnames"].astype(str),
                "source": path,
            }
    return group


def align_group(group):
    entries = list(group.values())
    common = set(entries[0]["paths"])
    for entry in entries[1:]:
        common.intersection_update(entry["paths"])
    ordered_paths = sorted(common)
    if not ordered_paths:
        raise RuntimeError("Feature files have no common test image paths")

    reference_labels = None
    for entry in entries:
        path_to_index = {path: index for index, path in enumerate(entry["paths"])}
        indices = np.asarray([path_to_index[path] for path in ordered_paths])
        entry["features"] = entry["features"][indices]
        entry["labels"] = entry["labels"][indices]
        if reference_labels is None:
            reference_labels = entry["labels"]
        elif not np.array_equal(reference_labels, entry["labels"]):
            raise RuntimeError("Class labels disagree after image-path alignment")
    return reference_labels


def balanced_indices(labels, max_per_class, random_state):
    rng = np.random.default_rng(random_state)
    selected = []
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) > max_per_class:
            indices = np.sort(rng.choice(indices, max_per_class, replace=False))
        selected.extend(indices.tolist())
    return np.asarray(sorted(selected))


def embed(features, perplexity, random_state):
    analysis_features = normalize(features, norm="l2")
    pca_dim = min(50, analysis_features.shape[1], analysis_features.shape[0] - 1)
    if 2 <= pca_dim < analysis_features.shape[1]:
        analysis_features = PCA(
            n_components=pca_dim, random_state=random_state
        ).fit_transform(analysis_features)
    effective_perplexity = min(perplexity, max(5.0, (len(features) - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": random_state,
    }
    iteration_name = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    kwargs[iteration_name] = 1500
    try:
        embedding = TSNE(**kwargs).fit_transform(analysis_features)
    except AttributeError as error:
        message = str(error)
        if "get_config" not in message and "NoneType" not in message:
            raise
        print("Barnes-Hut unavailable in this environment; using exact t-SNE")
        kwargs["method"] = "exact"
        embedding = TSNE(**kwargs).fit_transform(analysis_features)
    return embedding, analysis_features


def class_legend(class_ids, classnames, colors):
    handles = []
    for color, class_id in zip(colors, class_ids):
        label = classnames[class_id] if class_id < len(classnames) else str(class_id)
        handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="", markersize=5,
                markerfacecolor=color, markeredgewidth=0, label=label,
            )
        )
    return handles


def draw_embedding(axis, embedding, labels, class_ids, colors, title):
    for color, class_id in zip(colors, class_ids):
        mask = labels == class_id
        axis.scatter(
            embedding[mask, 0], embedding[mask, 1], s=8, alpha=0.72,
            color=color, linewidths=0,
        )
    axis.set_title(title, fontsize=11)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#777777")


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset(group, dataset_key, dataset_name, output_dir, args, metrics):
    labels = align_group(group)
    indices = balanced_indices(labels, args.max_per_class, args.random_state)
    labels = labels[indices]
    classnames = next(iter(group.values()))["classnames"]
    class_ids = np.unique(labels)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(class_ids)))
    handles = class_legend(class_ids, classnames, colors)
    embeddings = {}

    for method, display_name in METHODS:
        entry = group[method]
        embedding, analysis_features = embed(
            entry["features"][indices], args.perplexity, args.random_state
        )
        embeddings[method] = embedding
        metrics.append(
            {
                "dataset": dataset_name,
                "method": display_name,
                "samples": len(labels),
                "silhouette": silhouette_score(analysis_features, labels),
                "davies_bouldin": davies_bouldin_score(analysis_features, labels),
                "calinski_harabasz": calinski_harabasz_score(analysis_features, labels),
                "random_state": args.random_state,
                "source": entry["source"].as_posix(),
            }
        )

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2))
    for axis, (method, display_name) in zip(axes.flat, METHODS):
        draw_embedding(axis, embeddings[method], labels, class_ids, colors, display_name)
    fig.suptitle(f"Frozen pre-trained representations on {dataset_name}", fontsize=13)
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, len(handles)), frameon=False, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    save_figure(fig, output_dir / "combined", f"tsne_{dataset_key}_all_methods")

    for method, display_name in METHODS:
        fig, axis = plt.subplots(figsize=(6.4, 5.8))
        draw_embedding(axis, embeddings[method], labels, class_ids, colors, display_name)
        fig.suptitle(f"{display_name} on {dataset_name}", fontsize=13)
        fig.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
            ncol=min(5, len(handles)), frameon=False, fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.11, 1, 0.95))
        save_figure(fig, output_dir / "individual", f"tsne_{dataset_key}_{method}")


def main():
    args = parse_args()
    feature_root = args.feature_root.resolve()
    output_dir = args.output_dir.resolve()
    metrics = []
    for dataset_key, dataset_name in DATASETS:
        group = load_group(feature_root / dataset_key)
        plot_dataset(group, dataset_key, dataset_name, output_dir, args, metrics)

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "feature_cluster_metrics.csv"
    fields = (
        "dataset", "method", "samples", "silhouette", "davies_bouldin",
        "calinski_harabasz", "random_state", "source",
    )
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    print("Created 3 combined and 18 individual t-SNE figures (PDF + PNG)")
    print(f"Metrics: {metric_path}")


if __name__ == "__main__":
    main()
