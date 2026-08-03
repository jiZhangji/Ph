#!/usr/bin/env python3
"""Compose compact cross-method Grad-CAM comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METHODS = ("mae", "lomar", "fg_mae", "i_jepa", "sar_jepa", "phyd_mae")
METHOD_LABELS = {
    "mae": "MAE",
    "lomar": "LoMaR",
    "fg_mae": "FG-MAE",
    "i_jepa": "I-JEPA",
    "sar_jepa": "SAR-JEPA",
    "phyd_mae": "PhyD-MAE",
}
DATASETS = ("MSTAR_SOC", "New_FUSAR", "SAR_ACD")


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "paper_visualizations" / "attention_heatmaps_40shot",
    )
    parser.add_argument("--panel-size", type=int, default=180)
    parser.add_argument("--header-height", type=int, default=25)
    parser.add_argument("--expected-per-dataset", type=int, default=50)
    return parser.parse_args()


def load_index(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["source_path"]: row for row in rows}


def load_font(size):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def labeled_panel(image, label, panel_size, header_height, font):
    resampling = getattr(Image, "Resampling", Image)
    panel = Image.new("RGB", (panel_size, panel_size + header_height), "white")
    image = image.convert("RGB").resize(
        (panel_size, panel_size), resample=resampling.LANCZOS
    )
    panel.paste(image, (0, header_height))
    draw = ImageDraw.Draw(panel)
    box = draw.textbbox((0, 0), label, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    draw.text(
        ((panel_size - text_width) / 2, (header_height - text_height) / 2 - 1),
        label,
        fill="black",
        font=font,
    )
    return panel


def main():
    args = parse_args()
    root = args.input_root.resolve()
    merged_root = root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    font = load_font(max(11, args.header_height - 10))
    output_rows = []
    dataset_counts = {}

    for dataset in DATASETS:
        method_rows = {
            method: load_index(root / "methods" / dataset / method / "index.csv")
            for method in METHODS
        }
        reference_paths = set(method_rows[METHODS[0]])
        for method in METHODS[1:]:
            paths = set(method_rows[method])
            if paths != reference_paths:
                raise RuntimeError(
                    f"Sample mismatch for {dataset}/{method}: "
                    f"missing={len(reference_paths - paths)} "
                    f"extra={len(paths - reference_paths)}"
                )
        if len(reference_paths) != args.expected_per_dataset:
            raise RuntimeError(
                f"Expected {args.expected_per_dataset} samples for {dataset}, "
                f"found {len(reference_paths)}"
            )

        ordered_paths = sorted(
            reference_paths,
            key=lambda path: int(method_rows[METHODS[0]][path]["sample_index"]),
        )
        dataset_counts[dataset] = len(ordered_paths)
        for source_path in ordered_paths:
            reference = method_rows[METHODS[0]][source_path]
            original_path = (
                root / "originals" / dataset / reference["original_path"]
            )
            panels = [
                labeled_panel(
                    Image.open(original_path),
                    "Input",
                    args.panel_size,
                    args.header_height,
                    font,
                )
            ]
            for method in METHODS:
                row = method_rows[method][source_path]
                overlay_path = (
                    root / "methods" / dataset / method / row["overlay_path"]
                )
                panels.append(
                    labeled_panel(
                        Image.open(overlay_path),
                        METHOD_LABELS[method],
                        args.panel_size,
                        args.header_height,
                        font,
                    )
                )

            width = sum(panel.width for panel in panels)
            height = max(panel.height for panel in panels)
            merged = Image.new("RGB", (width, height), "white")
            left = 0
            for panel in panels:
                merged.paste(panel, (left, 0))
                left += panel.width

            relative_output = (
                Path(dataset)
                / reference["classname"]
                / Path(reference["original_path"]).name
            )
            output_path = merged_root / relative_output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.save(
                output_path,
                format="JPEG",
                quality=92,
                subsampling=0,
                optimize=True,
            )

            output_row = {
                "dataset": dataset,
                "sample_id": reference["sample_id"],
                "source_path": source_path,
                "label": reference["label"],
                "classname": reference["classname"],
                "original_path": (
                    Path("originals") / dataset / reference["original_path"]
                ).as_posix(),
                "merged_path": (
                    Path("merged") / relative_output
                ).as_posix(),
            }
            for method in METHODS:
                row = method_rows[method][source_path]
                output_row[f"{method}_prediction"] = row["predicted_class"]
                output_row[f"{method}_confidence"] = row["confidence"]
                output_row[f"{method}_correct"] = row["correct"]
            output_rows.append(output_row)

    fields = list(output_rows[0])
    with (root / "merged_index.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    marker = {
        "datasets": dataset_counts,
        "methods": list(METHODS),
        "total_merged": len(output_rows),
        "panel_size": args.panel_size,
        "header_height": args.header_height,
    }
    (root / "MERGED_EXPORT_COMPLETE.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"MERGED_ATTENTION_EXPORT root={root} "
        f"samples={len(output_rows)} methods={len(METHODS)}"
    )


if __name__ == "__main__":
    main()
