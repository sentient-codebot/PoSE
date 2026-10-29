# Prior checkpoints

The seven checkpoints are assets of release `v1.0.0`. Run
`bash scripts/download_priors.sh` from the repository root to fetch and verify
them; `SHA256SUMS` lists the hashes.

| File | State dimension | Used for |
|---|---|---|
| `prior_ausgrid_base.pt` | 64 | IEEE 33-bus, joint P and Q, base PV |
| `prior_ausgrid_high_pv.pt` | 64 | IEEE 33-bus, joint P and Q, 3x PV |
| `prior_fluvius.pt` | 64 | Fluvius data-dependence study |
| `prior_iowa.pt` | 386 | Iowa balanced and unbalanced three-phase |
| `prior_model_world_base.pt` | 32 | Data-scaling teacher, base PV |
| `prior_model_world_high_pv.pt` | 32 | Data-scaling teacher, 3x PV |
| `prior_ausgrid_active_power.pt` | 32 | Posterior-shape figure, active power only |

Each file is a plain dictionary with `state_dict`, `config` and
`standardization` (`mean_kw`, `std_kw`). `pose.prior.load_prior` reads this
format and the older format with `mean_kw` and `std_kw` at the top level.
