#!/usr/bin/env python3
"""Create six-method t-SNE figures from exported downstream features."""

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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


METHODS = (
    ("mae", "MAE"),
    ("lomar", "LoMaR"),
    ("fg_mae", "FG-MAE"),
    ("i_jepa", "I-JEPA"),
    ("sar_jepa", "SAR-JEPA"),
    ("phyd_mae", "PhyD-MAE"),
)
DATASETS = ("MSTAR_SOC", "New_FUSAR", "SAR_ACD")
SHOTS = (10, 20, 40)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("few_shot_classification/finetune/tsne_features"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_visualizations/tsne")
    )
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_group(directory):
    group = {}
    for method, label in METHODS:
        path = directory / f"{method}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            group[method] = {
                "label": label,
                "features": data["features"].astype(np.float32, copy=False),
                "labels": data["labels"].astype(np.int64, copy=False),
                "paths": data["paths"].astype(str),
                "classnames": data["classnames"].astype(str),
                "source": path,
            }
    return group


def align_group(group):
    entries = list(group.values())
    use_paths = all(len(entry["paths"]) == len(entry["labels"]) for entry in entries)
    if use_paths:
        common = set(entries[0]["paths"])
        for entry in entries[1:]:
            common.intersection_update(entry["paths"])
        ordered_paths = sorted(common)
        if not ordered_paths:
            raise RuntimeError("Feature files have no common test image paths")
        reference_labels = None
        for entry in entries:
            index = {path: i for i, path in enumerate(entry["paths"])}
            indices = np.asarray([index[path] for path in ordered_paths])
            entry["features"] = entry["features"][indices]
            entry["labels"] = entry["labels"][indices]
            if reference_labels is None:
                reference_labels = entry["labels"]
            elif not np.array_equal(reference_labels, entry["labels"]):
                raise RuntimeError("Class labels disagree after image-path alignment")
        return reference_labels

    reference_labels = entries[0]["labels"]
    for entry in entries[1:]:
        if not np.array_equal(reference_labels, entry["labels"]):
            raise RuntimeError("Feature files have different test-sample ordering")
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
    features = normalize(features, norm="l2")
    pca_dim = min(50, features.shape[1], features.shape[0] - 1)
    if pca_dim >= 2 and pca_dim < features.shape[1]:
        features = PCA(n_components=pca_dim, random_state=random_state).fit_transform(
            features
        )
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
        embedding = TSNE(**kwargs).fit_transform(features)
    except AttributeError as error:
        # Some Windows OpenBLAS builds expose no version string to
        # threadpoolctl, which breaks sklearn's Barnes-Hut neighbor search.
        message = str(error)
        if "get_config" not in message and "NoneType" not in message:
            raise
        print("Barnes-Hut unavailable in this environment; using exact t-SNE")
        kwargs["method"] = "exact"
        embedding = TSNE(**kwargs).fit_transform(features)
    return embedding, features


def plot_group(group, title, stem, output_dir, args, metrics):
    labels = align_group(group)
    indices = balanced_indices(labels, args.max_per_class, args.random_state)
    labels = labels[indices]
    classnames = next(iter(group.values()))["classnames"]
    class_ids = np.unique(labels)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(class_ids)))

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2))
    for axis, (method, method_label) in zip(axes.flat, METHODS):
        entry = group[method]
        embedding, analysis_features = embed(
            entry["features"][indices], args.perplexity, args.random_state
        )
        score = silhouette_score(analysis_features, labels, metric="euclidean")
        metrics.append(
            {
                "configuration": stem,
                "method": method,
                "samples": len(labels),
                "silhouette": score,
                "source": entry["source"].as_posix(),
            }
        )
        for color, class_id in zip(colors, class_ids):
            mask = labels == class_id
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=7,
                alpha=0.72,
                color=color,
                linewidths=0,
            )
        axis.set_title(method_label, fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#777777")

    handles = []
    for color, class_id in zip(colors, class_ids):
        name = classnames[class_id] if class_id < len(classnames) else str(class_id)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5,
                markerfacecolor=color,
                markeredgewidth=0,
                label=name,
            )
        )
    fig.suptitle(title, fontsize=13)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, len(handles)),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Created {output_dir / f'{stem}.pdf'}")


def main():
    args = parse_args()
    feature_root = args.feature_root.resolve()
    output_dir = args.output_dir.resolve()
    metrics = []
    created = 0

    for dataset in DATASETS:
        for shots in SHOTS:
            directory = feature_root / dataset / "MIM_finetune" / f"{shots}shot"
            group = load_group(directory)
            if group is None:
                print(f"Incomplete, skip: {directory}")
                continue
            stem = f"tsne_{dataset}_finetune_{shots}shot"
            plot_group(
                group,
                f"{dataset} fine-tuning, {shots}-shot",
                stem,
                output_dir,
                args,
                metrics,
            )
            created += 1

        directory = feature_root / dataset / "MIM_linear" / "encoder"
        group = load_group(directory)
        if group is None:
            print(f"Incomplete, skip: {directory}")
            continue
        stem = f"tsne_{dataset}_linear_encoder"
        plot_group(
            group,
            f"{dataset} linear probing (frozen encoder)",
            stem,
            output_dir,
            args,
            metrics,
        )
        created += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "feature_cluster_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("configuration", "method", "samples", "silhouette", "source")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Figures created: {created}/12")
    print(f"Metrics: {metric_path}")


if __name__ == "__main__":
    main()
