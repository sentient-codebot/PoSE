# Slurm templates

`run_cells.sbatch` runs one task per line of a config list. The placeholders
`ACCOUNT`, `PARTITION` and the time limit must be set to the values of your
site. The job needs one GPU per task, as the paper runs did (one H100 per
array task).

```bash
export POSE_CONFIG=configs/cells_ausgrid_fluvius.txt
export POSE_PYTHON=.venv/bin/python
sbatch --array=0-$(( $(wc -l < "$POSE_CONFIG") - 1 )) slurm/run_cells.sbatch
```

The same template runs prior training and tuning if the list holds the
arguments of those scripts, but the program name is fixed to `run_cell.py`.
For a mixed chain, submit separate lists with a small wrapper or use the
commands below directly.

## Example chain

```bash
# 1. Priors
export POSE_CONFIG=configs/priors_ausgrid_fluvius.txt
sbatch --array=0-$(( $(wc -l < "$POSE_CONFIG") - 1 )) slurm/run_cells.sbatch

# 2. Baseline tuning (one command per line, run directly or with a wrapper)
while read -r task; do $POSE_PYTHON scripts/tune_baselines.py $task; done \
  < configs/tune_ausgrid_fluvius_stage1.txt

# 3. Baseline cells
export POSE_CONFIG=configs/cells_ausgrid_fluvius.txt
sbatch --array=0-$(( $(wc -l < "$POSE_CONFIG") - 1 )) slurm/run_cells.sbatch

# 4. PoSE cells in shards, then merge
export POSE_CONFIG=configs/cells_ausgrid_fluvius_pose_shards.txt
sbatch --array=0-$(( $(wc -l < "$POSE_CONFIG") - 1 )) slurm/run_cells.sbatch
$POSE_PYTHON scripts/merge_shards.py --shards results/ausgrid/shards/.../shard0 ... \
  --output_dir results/ausgrid/cells/...

# 5. Report and figures
$POSE_PYTHON scripts/report/extended_report.py --root results/ausgrid \
  --scenarios ausgrid ausgrid_hpv --output_dir results/ausgrid/report
$POSE_PYTHON scripts/figures/plot_paper.py \
  --extended results/ausgrid/report/extended_summary.csv \
  --world results/world/report/learning_curve.csv \
  --posterior_stem results/ausgrid_p/cells/ausgrid/real/ausgrid_S2_nominal_ \
  --output_dir results/figures
```

The full paper runs used 1,000 test snapshots, 100 samples and 200 steps per
cell. A quick CPU check uses `--max_snapshots 2 --num_samples 4 --num_steps 4`
and `--device cpu`.
