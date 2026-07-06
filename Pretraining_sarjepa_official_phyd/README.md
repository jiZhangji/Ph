# SAR-JEPA Official Framework + PhyD Targets

This directory keeps the SAR-JEPA/LoMaR pretraining framework, optimizer flow,
window masking, scheduler, and checkpoint cadence aligned with the official
SAR-JEPA implementation. The only intended algorithmic change is replacing the
official reconstruction target with the PhyD dual physical targets:

- `SASGTTarget`: speckle-adaptive spatial/gradient target.
- `LFSTTarget`: low-frequency structural target.

Use `scripts/run_sarjepa_official_phyd_pretrain_2xh100.sh` to run this branch.
Keep experiment outputs separate from older Ph/SAR-JEPA runs.
