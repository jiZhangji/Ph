# Paper ablation completion

`run_paper_ablation_completion.sh` runs the reduced paper ablation matrix with
the same downstream settings used by the existing PhyD evaluations.

The default matrix is:

- Core target ablation: LFST-only and SASGT-only, three datasets, two protocols,
  10/20/40 shots, seeds 0--4.
- SASGT internal control: uniform multi-scale gradient versus complete SASGT on
  New_FUSAR, two protocols, 10/20/40 shots, seeds 0--4.

The script initializes the missing uniform-gradient checkpoint from the final
PhyD checkpoint-300 encoder, trains it for 30 epochs, then runs downstream
evaluation serially on GPU 0. Existing completed logs are reused when their
checkpoint and downstream configuration match. Incomplete result directories
are safe to rerun.

On the server:

```bash
cd /path/to/Ph
conda activate sar-pretrain
nohup env \
  EXPERIMENT_GROUPS="core sasgt_internal" \
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

The larger LFST sensitivity and FFT-vs-spatial-LPF suites are intentionally
opt-in. They can be enabled later with:

```bash
EXPERIMENT_GROUPS="core sasgt_internal lfst_sensitivity filter_comparison" \
  bash scripts/run_paper_ablation_completion.sh
```
