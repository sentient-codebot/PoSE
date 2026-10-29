#!/usr/bin/env bash
#SBATCH --job-name=pose
#SBATCH --account=ACCOUNT
#SBATCH --partition=PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/pose_%A_%a.out
#SBATCH --error=logs/pose_%A_%a.err

# One array task per non-empty line of a config list. Each line holds the
# arguments of scripts/run_cell.py, without the program name.
#
#   export POSE_CONFIG=configs/cells_ausgrid_fluvius.txt
#   export POSE_PYTHON=.venv/bin/python
#   sbatch --array=0-$(( $(wc -l < "$POSE_CONFIG") - 1 )) slurm/run_cells.sbatch
#
# Set ACCOUNT, PARTITION and the time limit above to the values of your site.

set -euo pipefail

: "${POSE_CONFIG:?set POSE_CONFIG to a config list}"
: "${POSE_PYTHON:=python}"

mkdir -p logs
mapfile -t TASKS < <(grep -v '^[[:space:]]*$' "$POSE_CONFIG")
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "task ${SLURM_ARRAY_TASK_ID}/${#TASKS[@]}: ${TASK}"
# shellcheck disable=SC2086
$POSE_PYTHON scripts/run_cell.py $TASK
