# Paper baseline reproduction and controlled speckle robustness

This suite evaluates the paper baselines with a shared downstream protocol and
adds deterministic, test-only multiplicative speckle corruption.

## Supported checkpoints

| Method | Encoder | Checkpoint state | Pooling |
|---|---|---|---|
| MAE | plain single-channel ViT-B/16 | `model` | class token |
| LoMaR | plain downstream ViT-B/16 | `model` | class token |
| FG-MAE | plain single-channel ViT-B/16 | `model` | class token |
| I-JEPA | no-class-token ViT-B/16 | `encoder` | mean patch token |
| SAR-JEPA | plain downstream ViT-B/16 | `model` | class token |
| PhyD-MAE | project ViT-B/16 | `model` | class token |

The loader fails if the patch projection, positional embedding, any Transformer
block, or any other required encoder tensor is absent or shape-mismatched.
Pre-training-only decoders and method-specific relative-position tensors are
reported but are not treated as downstream backbone parameters, following the
original SAR-JEPA downstream setup.

Audit downloaded baseline weights before launching a long experiment:

```bash
python scripts/audit_paper_baseline_checkpoints.py \
  --weights-root /mnt/data5/zhangji/SAR-JEPA/weights \
  --forward
```

Run the complete clean downstream matrix:

```bash
CUDA_VISIBLE_DEVICES=0 \
WEIGHTS_ROOT=/mnt/data5/zhangji/SAR-JEPA/weights \
METHODS="mae lomar fg_mae i_jepa sar_jepa" \
bash scripts/run_paper_baseline_fewshot.sh
```

The runner uses three datasets, fine-tuning and linear probing, 10/20/40 shots,
and downstream seeds 0--4 by default. Completed logs are skipped.

## Controlled additional speckle

The robustness experiment does not claim to recover or replace the unknown
number of looks of the processed benchmark images. It applies additional
intensity speckle at test time:

```text
N ~ Gamma(L_add, 1 / L_add)
A_noisy = clamp(A * sqrt(N), 0, 1)
```

`L_add=clean` leaves the image unchanged. Smaller positive values produce
stronger additional corruption. Noise is derived from the image path and noise
seed, so every method receives the same realization independent of data-loader
order. Training images remain clean.

The default robustness experiment trains each downstream model once on clean
MSTAR 20-shot data with linear probing, then evaluates it at
`clean/8/4/2/1`, using five downstream seeds and three noise seeds:

```bash
CUDA_VISIBLE_DEVICES=0 \
WEIGHTS_ROOT=/mnt/data5/zhangji/SAR-JEPA/weights \
PHYD_CHECKPOINT="$PWD/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth" \
bash scripts/run_speckle_robustness.sh
```

The final CSV contains accuracy, absolute drop from clean accuracy, and
retention percentage. Incomplete seed groups cause the summarizer to fail
instead of silently producing a partial table.
