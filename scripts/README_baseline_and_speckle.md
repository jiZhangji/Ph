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

The paper robustness experiment trains the linear probes on clean data from
MSTAR, FUSAR-Ship, and SAR-ACD at 10/20/40 shots, then evaluates each probe at
`clean/8/4/2/1`, using ten downstream seeds and three noise seeds. Baselines
use the shared linear-probe learning rate of `1e-4`; PhyD-MAE uses the `1e-3`
linear-probe setting associated with its main-table checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
WEIGHTS_ROOT=/mnt/data5/zhangji/SAR-JEPA/weights \
PHYD_CHECKPOINT="$PWD/runs/sarjepa_official_phyd_ft250_bs1024_lfst0p1_image_2xh200/checkpoint-300.pth" \
bash scripts/launch_paper_speckle_4090.sh
```

The final CSV contains accuracy, absolute drop from clean accuracy, and
retention percentage. Incomplete seed groups cause the summarizer to fail
instead of silently producing a partial table.

For the full paper matrix covering both fine-tuning and linear probing at
10/20/40 shots, prepare the output once and then launch the matching profile
on each GPU instance:

```bash
bash scripts/prepare_paper_speckle_full_output.sh
bash scripts/launch_paper_speckle_full.sh 4090
bash scripts/launch_paper_speckle_full.sh 2h100
bash scripts/launch_paper_speckle_full.sh 1h100
bash scripts/launch_paper_speckle_full.sh 2h200
```

The preparation step archives the earlier PhyD-MAE `LR=1e-4` linear pilot.
The full run uses `LR=1e-4` for fine-tuning and baseline linear probes, and
`LR=1e-3` for the PhyD-MAE linear probe to match its main-table evaluation.
