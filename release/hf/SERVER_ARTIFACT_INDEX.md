# Server Artifact Index

All paths below are relative to the Ph repository root unless an absolute root is shown.

Server repository root:

```text
/inspire/hdd/global_user/liuxiaotong-253108540242/yanggang/lihao/lh/or/SAR-Generation/Ph
```

## Checkpoints Worth Preserving

| Priority | ID | Server checkpoint | Purpose |
|---|---|---|---|
| Main | `phyd-best-ckpt300` | `runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth` | Best confirmed strict-official downstream model. |
| Reproduction | `phyd-stage1-ckpt250` | `runs/sarjepa_official_phyd_2xh100/checkpoint-250.pth` | Stage-I state needed to reproduce the best two-stage recipe. |
| Baseline | `sarjepa-official-reproduction-ckpt200` | `runs/sarjepa_pretrain_2xh100/checkpoint-200.pth` | Local official SAR-JEPA reproduction. |
| Analysis | `phyd-warmstart-drift-ckpt299` | `runs/sarjepa_official_phyd_warmstart_bestckpt300_bs1088_lfst0p1_image_20260709_174032/checkpoint-299.pth` | Complete strict-official result matrix and representation-drift analysis. |
| Experimental | `phyd-sfafm7-ckpt20` | `runs/phyd_sfafm7_every2end_from_best300_g1_lfst0p1_image_bs768_300e_2xh200/checkpoint-20.pth` | Best early seven-SFAFM checkpoint; not paper-ready. |

Optional historical checkpoints:

```text
runs/overnight_ablation_8runs_sasgt_only_image/checkpoint-24.pth
runs/pretrain_2xh100_rerun_bs256_lr1e-4/checkpoint-99.pth
runs/sarjepa_official_phyd_ft299_bs1024_lfst0p2_2xh200/checkpoint-321.pth
```

The first two used legacy downstream pipelines. The LFST=0.2 checkpoint was weaker than the main checkpoint and is not included by default.

## Confirmed Downstream Result Directories

Main checkpoint-300, strict official-style downstream, 3 datasets x 2 protocols x 3 shots x up to 20 seeds:

```text
few_shot_classification/finetune/output_phyd_ft250_ckpt300_official_all10_mstar
few_shot_classification/finetune/output_phyd_ft250_ckpt300_official_all10_fusar
few_shot_classification/finetune/output_phyd_ft250_ckpt300_official_all10_saracd
```

Warm-start checkpoint-299, strict official-style downstream, complete 360/360:

```text
few_shot_classification/finetune/output_warmstart_bestckpt300_ckpt299_official_all_fast3
```

Seven-SFAFM checkpoint sweep and checkpoint-20 follow-up:

```text
few_shot_classification/finetune/output_phyd_sfafm7_every2_sweep
few_shot_classification/finetune/output_phyd_sfafm7_ckpt20_fusar_saracd_finetune_5seeds
```

Historical checkpoint-250 downstream output:

```text
few_shot_classification/finetune/output_sarjepa_official_phyd_ckpt250_all_20seeds
```

The original official SAR-JEPA checkpoint-200 output directory was not uniquely preserved in the durable ledger because early runs reused shared output paths. Locate its matching logs by checkpoint content instead of guessing a directory:

```bash
grep -RFl "sarjepa_pretrain_2xh100/checkpoint-200.pth" \
  few_shot_classification/finetune runs --include=log.txt
```

## Durable Seed-Level Records

The release package includes these local records under `results/confirmed_multiseed/`:

```text
best_ckpt300_seed_results.csv
warmstart_ckpt299_seed_results.csv
ckpt250_historical_seed_accuracy.csv
raw_best_ckpt300_official_stats.txt
raw_warmstart_ckpt299_official_stats.txt
raw_ckpt250_historical_stats.txt
```

The main and warm-start CSV files contain per-seed accuracy and macro-F1. Historical checkpoint-250 per-seed macro-F1 was not present in the original output and must not be reconstructed from aggregate values.

## Full Run Packaging

To preserve complete run directories rather than model-only checkpoints, use:

```bash
PACKAGE_DIR="$PWD/hf_release/phyd-sar-full-runs" \
INCLUDE_FULL_RUNS=1 \
INCLUDE_MODEL_ONLY=0 \
INCLUDE_RAW_LOGS=0 \
INCLUDE_HISTORICAL=0 \
bash scripts/package_hf_release.sh
```

The package keeps each original `runs/<run-name>/` relative path. Files are hard-linked when the package and source are on the same filesystem, falling back to copies when hard links are unavailable. Matching files from `logs/<run-name>*` are placed under `external_logs/<run-name>/`.
