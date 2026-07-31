# Paper ablation completion

`run_paper_ablation_completion.sh` runs the complete paper ablation matrix with
the same downstream settings used by the existing PhyD evaluations.

The default matrix is:

- Core target ablation: LFST-only and SASGT-only, three datasets, two protocols,
  10/20/40 shots, seeds 0--4.
- SASGT internal ablation: uniform multi-scale gradient, adaptive gradient, and
  complete SASGT on New_FUSAR, two protocols, 10/20/40 shots, seeds 0--4.
- LFST sensitivity: raw/log input with cutoff 20/30/40 on New_FUSAR, two
  protocols, 10/20/40 shots, seeds 0--4.
- Filtering comparison: raw FFT-LFST versus spatial LPF at cutoff 40 on
  New_FUSAR, two protocols, 10/20/40 shots, seeds 0--4.

The script initializes four missing checkpoints from the final PhyD
checkpoint-300 encoder: uniform SASGT, adaptive SASGT, raw LFST cutoff 20, and
raw LFST cutoff 40. Each is trained for 30 epochs. Downstream evaluation then
runs serially on GPU 0. Existing completed logs are reused when their checkpoint
and downstream configuration match. Incomplete result directories are safe to
rerun.

On the server:

```bash
cd /path/to/Ph
conda activate sar-pretrain
nohup env \
  EXPERIMENT_GROUPS=all \
  bash scripts/run_paper_ablation_completion.sh \
  > logs/paper_ablation_completion_5seeds.nohup.log 2>&1 &
echo "PID=$!"
tail -f logs/paper_ablation_completion_5seeds.nohup.log
```

The command is resumable. Rerun the same command after an interruption. To
only inspect or summarize the current matrix:

```bash
ACTION=summary bash scripts/run_paper_ablation_completion.sh
```

Results are written to:

```text
few_shot_classification/finetune/output_paper_ablation_completion_5seeds/
```

The final files are `results_per_seed.csv` and `results_mean_std.csv`.

To run only selected groups, provide any subset of `core`, `sasgt_internal`,
`lfst_sensitivity`, and `filter_comparison`, for example:

```bash
EXPERIMENT_GROUPS="core sasgt_internal" \
  bash scripts/run_paper_ablation_completion.sh
```
