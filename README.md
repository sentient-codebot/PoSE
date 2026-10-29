# PoSE: posterior sampling for distribution system state estimation

PoSE formulates distribution system state estimation as posterior sampling. A
flow matching prior learns the joint distribution of secondary-substation
injections, and a noise-aware Kalman-type endpoint correction brings in
measurements through the AC power flow during sampling. The network never
approximates the power flow, and one trained prior serves any measurement set,
noise level and topology.

On the IEEE 33-bus feeder, PoSE lowers the probabilistic injection error
(CRPS) by 6.7--7.5% against a full-covariance Gaussian prior in all 12 tested
settings. Its 90% intervals contain the true injection almost exactly as often
as exact posterior samples would. On the 240-bus unbalanced three-phase Iowa
feeder, it gives the lowest injection, voltage and branch-current errors of all
estimators. The fast variant, PoSE-fast, matches this accuracy at 0.2--0.6 s
per snapshot and, without retraining, beats Gaussian priors and retrained
supervised estimators on seven reconfigured topologies.

This repository contains the code, configuration lists and project page. The
paper is a preprint; the arXiv link and BibTeX will be updated when the URL
exists. Trained priors are released as assets of
[release v1.0.0](https://github.com/sentient-codebot/PoSE/releases/tag/v1.0.0).

- Project page: <https://sentient-codebot.github.io/PoSE/>
- Source repository: <https://github.com/sentient-codebot/PoSE>
- Mirror: <https://github.com/distributionnetworksTUDelft/PoSE>

## Install

Python 3.12 or newer is required. The paper runs used PyTorch 2.8 with CUDA 12.8
on one NVIDIA H100 per array task. The CPU tests in this release ran with
Python 3.12.13 and PyTorch 2.15.0.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[plots]"
```

Add the validation extra for the pandapower and OpenDSS checks:

```bash
pip install -e ".[validation,plots]"
```

## Data

No measurement data are included. The three data sets are public; download
them from their owners, check the SHA-256 values, and assemble the snapshot
files with the scripts below. The assembled files are written under `data/`,
which is ignored by Git.

| Data set | Used for | Output |
|---|---|---|
| Ausgrid Solar Home Electricity Data | IEEE 33-bus, base and high PV, joint P and Q | `data/ausgrid/snapshots.npz`, `data/ausgrid_high_pv/snapshots.npz` |
| Ausgrid, active power only | Posterior-shape figure | `data/ausgrid_p/snapshots.npz` |
| Fluvius 2024 profiles | IEEE 33-bus data-dependence study, joint P and Q | `data/fluvius/snapshots.npz` |
| Fluvius 2024 profiles, active power only | Data-scaling study teachers | `data/fluvius_base/snapshots.npz`, `data/fluvius_high_pv/snapshots.npz` |
| Iowa 240-bus test system | Balanced and unbalanced three-phase feeder | `data/iowa/snapshots.npz`, `data/iowa_3ph/feeder3ph.json` |

### Ausgrid solar home data

Owner: Ausgrid. Licence: CC BY 3.0 AU. The archive mirror used for the paper is
<https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip>
(SHA-256 `5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82`).
Extract the three CSV files into `data/ausgrid/raw/`. The assembly script
checks their SHA-256 against the values in `EXPECTED_SHA256` and prints them.

```bash
python scripts/data/assemble_ausgrid.py --raw_dir data/ausgrid/raw \
  --output data/ausgrid/snapshots.npz --state pq
python scripts/data/assemble_ausgrid.py --raw_dir data/ausgrid/raw \
  --output data/ausgrid_high_pv/snapshots.npz --state pq \
  --pv_scale 3 --scenario ausgrid_hpv
python scripts/data/assemble_ausgrid.py --raw_dir data/ausgrid/raw \
  --output data/ausgrid_p/snapshots.npz --state p
```

### Fluvius 2024 profiles

Owner: Fluvius. Licence: Fluvius open data licence, with attribution
("Fluvius, 2024 15-minute digital meter profiles"). The download script fetches
the two 2024 sample archives and the legend, and writes `SHA256SUMS.txt`.

```bash
bash scripts/data/download_fluvius.sh data/fluvius/raw
python scripts/data/assemble_fluvius.py --raw_dir data/fluvius/raw \
  --scenarios base --state pq --output data/fluvius/snapshots.npz
python scripts/data/assemble_fluvius.py --raw_dir data/fluvius/raw \
  --scenarios base --state p --output data/fluvius_base/snapshots.npz
python scripts/data/assemble_fluvius.py --raw_dir data/fluvius/raw \
  --scenarios high_pv --state p --output data/fluvius_high_pv/snapshots.npz
```

Expected archive SHA-256:

```text
0f8addd57bd9f7bc0fd0c28b019a4b6c8753a19e3fd301de758f5a5858c9d86c  P6269_1_50_DMK_Sample_Elek_2024_Deel_1.zip
cfef37973c40e503fe53afa4966bbdcd7a12525fb8b20881f365d0ee8616a3d6  P6269_1_50_DMK_Sample_Elek_2024_Deel_2.zip
344ac3a69adab39bb0e6cdf490b63380ea5ebb0c83cc26ab964c73430c1b0acf  1.50 Aanleveirng DMK Elek Fluvius Legende 25200929.xlsx
```

### Iowa 240-bus test system

Owner: Iowa State University. Cite Bu, Yuan, Wang and Meagher, "A time-series
distribution test system based on real utility data," NAPS 2019. Download from
<https://wzy.ece.iastate.edu/Testsystem.html>:

```bash
mkdir -p data/iowa/raw/extracted
curl -fL -o "data/iowa/raw/OpenDSS Model.zip" \
  "https://wzy.ece.iastate.edu/publication/OpenDSS%20Model.zip"
curl -fL -o "data/iowa/raw/Smart Meter Data.zip" \
  "https://wzy.ece.iastate.edu/publication/Smart%20Meter%20Data.zip"
sha256sum "data/iowa/raw/OpenDSS Model.zip" "data/iowa/raw/Smart Meter Data.zip"
unzip -q "data/iowa/raw/OpenDSS Model.zip" -d data/iowa/raw/extracted
unzip -q "data/iowa/raw/Smart Meter Data.zip" -d data/iowa/raw/extracted
```

Expected archive SHA-256:

```text
146c9a5278439422e3ee7bcea51e86b3a78de97484e6ff2d9c353bd212d44da5  OpenDSS Model.zip
11d6b19fc049c113dcf7c21324504848e99e8c871622e311a86cb2b3a59ad9a0  Smart Meter Data.zip
```

Assemble the balanced feeder and the three-phase feeder file:

```bash
python scripts/data/assemble_iowa.py \
  --raw_dir data/iowa/raw/extracted --output data/iowa/snapshots.npz
python scripts/data/assemble_iowa_3ph.py \
  --raw_dir "data/iowa/raw/extracted/OpenDSS Model (08.13.2020)" \
  --meta data/iowa/snapshots_meta.json --output data/iowa_3ph/feeder3ph.json
```

The OpenDSS reference used by `validate_iowa.py` is produced with

```bash
python scripts/data/iowa_opendss_export.py --data data/iowa/snapshots.npz \
  --master "data/iowa/raw/extracted/OpenDSS Model (08.13.2020)/Master.dss" \
  --output results/iowa/opendss_reference.npz
python scripts/validate/validate_iowa.py --data data/iowa/snapshots.npz \
  --opendss results/iowa/opendss_reference.npz --output results/iowa/opendss_validation.json
```

## Priors

The released priors are assets of `v1.0.0`. Download and verify them with

```bash
bash scripts/download_priors.sh
```

| File | Used for |
|---|---|
| `prior_ausgrid_base.pt` | IEEE 33-bus, joint P and Q, base PV |
| `prior_ausgrid_high_pv.pt` | IEEE 33-bus, joint P and Q, 3x PV |
| `prior_fluvius.pt` | Fluvius data-dependence study |
| `prior_iowa.pt` | Iowa balanced and unbalanced three-phase |
| `prior_model_world_base.pt` | Data-scaling teacher, base PV |
| `prior_model_world_high_pv.pt` | Data-scaling teacher, 3x PV |
| `prior_ausgrid_active_power.pt` | Posterior-shape figure, active power only |

`checkpoints/SHA256SUMS` lists the hashes. The download script rejects a file
whose hash differs.

## Quickstart

After installing and assembling the Ausgrid data:

```bash
bash scripts/download_priors.sh
python scripts/run_cell.py --data data/ausgrid/snapshots.npz \
  --prior checkpoints/prior_ausgrid_base.pt --mset S2 --noise nominal \
  --method M2 --design extended --max_snapshots 2 --num_samples 4 \
  --num_steps 4 --chunk 2 --device cpu --output_dir results/quickstart
```

`results/quickstart/metrics.json` holds the aggregates and
`per_snapshot.csv` the per-snapshot metrics. Drop `--max_snapshots`,
`--num_samples` and `--num_steps` to run the paper setting: 1,000 snapshots,
100 samples, 200 steps. A GPU is strongly recommended for the full matrix.

## Reproducing the paper

The lists under `configs/` are the inputs of `scripts/run_cell.py`,
`scripts/train_prior.py`, `scripts/select_prior.py`,
`scripts/tune_baselines.py` and the report scripts. Paths inside the lists are
relative to the repository root. The paper runs used one H100 per array task;
`slurm/run_cells.sbatch` runs one line per task and `slurm/README.md` shows an
example chain.

| Paper part | Config lists and commands |
|---|---|
| Main table, CRPS and coverage figure, ablation, cost, DSO metrics | `configs/priors_ausgrid_fluvius.txt`, `configs/tune_ausgrid_fluvius_stage1.txt`, `configs/tune_ausgrid_fluvius_stage2.txt`, `configs/cells_ausgrid_fluvius.txt`, `configs/cells_ausgrid_fluvius_pose_shards.txt`; then `scripts/report/extended_report.py`, `scripts/report/reference_table.py`, `scripts/figures/plot_paper.py` |
| Fluvius data dependence | The last three lines of `configs/cells_ausgrid_fluvius.txt` and the Fluvius entries of the prior and tuning lists |
| Exactness at feeder scale | `python scripts/exactness_test.py --data data/fluvius/snapshots.npz --output results/exactness.json` |
| Data-scaling figure | `configs/priors_data_scaling_teacher.txt`, `scripts/world_data.py`, `configs/priors_data_scaling_students.txt`, `configs/cells_data_scaling.txt`, `scripts/report/world_report.py` |
| Posterior-shape figure | `configs/priors_ausgrid_active_power.txt`, `scripts/select_prior.py`, `configs/cells_posterior_shapes.txt`, `scripts/figures/plot_paper.py` |
| Iowa balanced, pooled Gaussian | `configs/priors_iowa.txt`, `configs/tune_iowa.txt`, `configs/cells_iowa.txt`, `configs/cells_iowa_pose_shards.txt`, `scripts/data/assemble_iowa.py`, `scripts/data/iowa_opendss_export.py`, `scripts/validate/validate_iowa.py` |
| Iowa unbalanced three-phase | `scripts/data/assemble_iowa_3ph.py`, `scripts/validate/iowa_3ph_probe.py`, `scripts/validate/validate_iowa_3ph.py`, `configs/cells_iowa_3ph.txt`, `scripts/report/iowa_3ph_report.py` |
| PoSE-fast | Any M2 line with `--sigma_mode meas`; `scripts/report/timing.py` for the runtime |
| Reconfigured feeders | `configs/cells_reconfigured.txt`, `scripts/reconf_pairing.py`, `scripts/report/reconf_report.py` |
| Example week figure | `configs/week_ids.json`, `scripts/figures/select_week.py`, `scripts/figures/prior_samples.py`, `scripts/figures/pack_week.py`, `scripts/figures/fig_week.py` |

The week figure uses the whole test split once to choose the week on the truth
only, then six cells on the 336 snapshots of that week:

```bash
python scripts/run_cell.py --data data/ausgrid_high_pv/snapshots.npz \
  --mset S2 --noise nominal --design test --method B1 --chunk 25 \
  --output_dir results/week/stage1
python scripts/figures/select_week.py configs/week_ids.json \
  results/week/stage1/per_snapshot.csv results/week/selected_ids.json
# run configs/cells_week.txt, then the midday prior samples
python scripts/figures/prior_samples.py --data data/ausgrid_high_pv/snapshots.npz \
  --prior checkpoints/prior_ausgrid_high_pv.pt \
  --snapshot_ids results/week/selected_ids.json --ids_key midday \
  --output results/week/prior/prior.npz
python scripts/figures/pack_week.py results/week
python scripts/figures/fig_week.py results/week/fig_week_data.npz \
  results/figures/fig_week.pdf
```

Typical report commands:

```bash
python scripts/report/extended_report.py --root results/ausgrid \
  --scenarios ausgrid ausgrid_hpv --msets S1 S2 S3 --noises nominal high \
  --output_dir results/ausgrid/report
python scripts/report/world_report.py --cells_root results/world/cells \
  --output_dir results/world/report
python scripts/report/reconf_report.py --root results/reconf \
  --output_dir results/reconf/report
python scripts/figures/plot_paper.py \
  --extended results/ausgrid/report/extended_summary.csv \
  --world results/world/report/learning_curve.csv \
  --posterior_stem results/ausgrid_p/cells/ausgrid/real/ausgrid_S2_nominal_ \
  --output_dir results/figures
```

The Iowa, three-phase and timing reports take their input directories on the
command line; run them with `--help` for the exact paths.

The main matrix has 12 settings (three measurement sets, two noise levels, two
PV scenarios) and 1,000 test snapshots per setting. PoSE uses 200 Euler steps
and 100 samples; PoSE-fast uses the measurement-space covariance with the same
sampler. On the IEEE 33-bus feeder, PoSE takes about 15 s per snapshot on one
H100; PoSE-fast takes 0.23 s. On the balanced Iowa feeder PoSE-fast takes 0.53 s, and the three-phase Iowa
feeder 0.58 s.

## Method codes

The code and CLI keep the method codes. The paper names are:

| Code | Paper name |
|---|---|
| `M2` | PoSE |
| `M2` with `--sigma_mode meas` | PoSE-fast |
| `M0` | projection variant (ablation) |
| `M1` | isotropic variant (ablation) |
| `B1` | WLS |
| `B2` | correlated Gaussian |
| `B3` | Gaussian mixture |
| `B4` | DPS (ablation) |
| `B5` | supervised MLP |
| `B6` | supervised flow |
| `B7` | pooled Gaussian |

## Paired randomness

Every method sees the same truths, observations and initial noise. One random
generator per role and snapshot is seeded from strings in
`src/pose/experiment.py`:

- `obs`: scenario, measurement set, noise level and snapshot;
- `init`: scenario and snapshot, the initial noise of every sampler;
- `ancestral`: scenario and snapshot, the noise of the ancestral steps;
- `draw`: scenario and snapshot, Gaussian and mixture posterior draws.

Batching and sharding therefore do not change a cell's samples.

## Tests

```bash
pytest -q
```

The CPU tests cover the power flow against finite-difference Jacobians, the
Tweedie covariance, M0 as a projection, M2 on a linear Gaussian, the metrics
and split, data assembly, reconfigurations and the three-phase model. The
end-to-end smoke test trains a tiny prior and runs one PoSE cell.

## Citation

```bibtex
@unpublished{lin_pose,
  author = {Lin, Nan and Palensky, Peter and Vergara, Pedro P.},
  title  = {Bayesian Distribution System State
            Estimation with Flow Matching},
  note   = {Manuscript submitted for publication},
  year   = {2026}
}
```

## Licence and contact

MIT licence; see `LICENSE`. The fonts on the project page keep their OFL
licences under `page/assets/fonts/LICENSES/`.

Contact: Nan Lin, Delft University of Technology, <N.Lin@tudelft.nl>.

## Acknowledgement

This work used the Dutch national e-infrastructure with the support of the SURF
Cooperative using grant no. EINF-17304. Nan Lin is funded by NWO Align4Energy
Project NWA.1389.20.251.
