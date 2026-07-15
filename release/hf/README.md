---
license: cc-by-nc-4.0
library_name: pytorch
tags:
- synthetic-aperture-radar
- self-supervised-learning
- vision-transformer
- masked-image-modeling
- few-shot-classification
---

# PhyD SAR Pretraining Checkpoints

This release collects the best confirmed PhyD checkpoint, the checkpoints needed to reproduce its two-stage training, a local official SAR-JEPA reproduction, and selected analysis checkpoints. The implementation is based on the official SAR-JEPA training framework with PhyD spatial-gradient (SASGT) and frequency-domain (LFST) targets.

Source code: <https://github.com/jiZhangji/Ph>

## Recommended Checkpoint

Use `models/main/phyd-best-ckpt300/model.pth` for downstream experiments.

Training configuration:

- Backbone: ViT-Base/16 in the official SAR-JEPA framework.
- Stage I: `phyd-stage1-ckpt250`, gradient/LFST weights `1.0/1.0`, patch target normalization.
- Stage II: continued to checkpoint 300 with gradient/LFST weights `1.0/0.1` and image target normalization.
- SFAFM: disabled.
- Downstream evaluation: strict official-style finetuning and linear probing.

## Strict Official-Style Downstream Results

Values are mean `+-` population standard deviation. Full per-seed values are in `results/confirmed_multiseed/`.

| Dataset | Protocol | Shots | Accuracy | Macro-F1 | Seeds |
|---|---|---:|---:|---:|---:|
| MSTAR_SOC | finetune | 10 | 70.22 +- 3.00 | 67.78 +- 3.00 | 20 |
| MSTAR_SOC | finetune | 20 | 83.34 +- 2.33 | 81.91 +- 2.53 | 18 |
| MSTAR_SOC | finetune | 40 | 89.89 +- 2.16 | 89.24 +- 2.28 | 20 |
| MSTAR_SOC | linear | 10 | 63.47 +- 0.83 | 61.06 +- 1.02 | 20 |
| MSTAR_SOC | linear | 20 | 72.25 +- 1.06 | 70.35 +- 1.07 | 20 |
| MSTAR_SOC | linear | 40 | 77.31 +- 1.16 | 75.81 +- 1.18 | 20 |
| New_FUSAR | finetune | 10 | 80.36 +- 1.95 | 67.83 +- 2.41 | 19 |
| New_FUSAR | finetune | 20 | 82.94 +- 1.19 | 71.22 +- 1.28 | 20 |
| New_FUSAR | finetune | 40 | 85.88 +- 0.86 | 75.34 +- 0.92 | 20 |
| New_FUSAR | linear | 10 | 80.44 +- 1.88 | 68.06 +- 2.37 | 20 |
| New_FUSAR | linear | 20 | 83.26 +- 1.02 | 71.68 +- 1.63 | 20 |
| New_FUSAR | linear | 40 | 85.98 +- 0.68 | 75.59 +- 0.85 | 20 |
| SAR_ACD | finetune | 10 | 53.82 +- 1.81 | 53.44 +- 1.90 | 20 |
| SAR_ACD | finetune | 20 | 61.75 +- 2.08 | 61.69 +- 2.05 | 20 |
| SAR_ACD | finetune | 40 | 71.88 +- 1.98 | 71.62 +- 2.02 | 20 |
| SAR_ACD | linear | 10 | 53.51 +- 1.54 | 53.20 +- 1.68 | 20 |
| SAR_ACD | linear | 20 | 59.89 +- 1.27 | 59.84 +- 1.28 | 20 |
| SAR_ACD | linear | 40 | 65.52 +- 1.67 | 65.54 +- 1.63 | 20 |

Three runs were excluded because their logs did not match the expected checkpoint: MSTAR finetune 20-shot seeds 9 and 11, and New_FUSAR finetune 10-shot seed 4.

## Checkpoint Index

| ID | Role | Notes |
|---|---|---|
| `phyd-best-ckpt300` | Main | Best confirmed paper-facing checkpoint. |
| `phyd-stage1-ckpt250` | Reproduction | Stage-I initialization used by the main checkpoint. |
| `sarjepa-official-reproduction-ckpt200` | Baseline | Local official SAR-JEPA reproduction. |
| `phyd-warmstart-drift-ckpt299` | Analysis | Complete 360-run downstream ledger; prolonged warm start reduced transfer on most settings. |
| `phyd-sfafm7-ckpt20` | Experimental | Seven-SFAFM candidate; only early single-seed evidence is available. |

The exact included files and source paths are recorded in `manifest.json`. By default, files under `models/` are model-only checkpoints and omit optimizer/scaler state.

## Loading

The model-only files retain the checkpoint structure expected by this repository:

```python
import torch

checkpoint = torch.load("models/main/phyd-best-ckpt300/model.pth", map_location="cpu")
state_dict = checkpoint["model"]
```

For the repository's strict official-style downstream evaluation:

```bash
CHECKPOINT=/path/to/models/main/phyd-best-ckpt300/model.pth \
DATASETS=MSTAR_SOC \
PROTOCOLS="finetune linear" \
SHOTS="10 20 40" \
SEEDS="0 1 2 3 4" \
USE_SFAFM=0 \
FORCE=0 \
bash scripts/run_sarjepa_fewshot_all.sh
```

For `phyd-sfafm7-ckpt20`, use the matching seven-stage SFAFM architecture and set `USE_SFAFM=1`.

## Comparability and Limitations

- Paper-facing comparisons should use only the strict official-style downstream results.
- Historical pipeline results in the ledger are diagnostic and are not controlled epoch-only comparisons.
- Lower self-supervised training loss did not consistently imply better transfer performance.
- The SFAFM checkpoint is experimental and should not replace the main model without multi-seed validation.
- No training or downstream dataset is included in this model repository.
- Users are responsible for obtaining datasets under their respective licenses and for checking whether their intended use complies with all upstream code, model, and dataset licenses.

## License

The repository code is distributed under CC BY-NC 4.0. Check upstream SAR-JEPA components and each dataset's terms before redistribution or commercial use.
