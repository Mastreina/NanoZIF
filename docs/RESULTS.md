# Results Layout

Simulation artifacts were moved under `results/` to keep the project root clean and to make it obvious which datasets are authoritative.

## Current directories

| Path | Contents | Notes |
| --- | --- | --- |
| `results/mof52/10k` | Dense MOF 5:2 sweep with 10k steps. | Supersedes the old `outputs_dense_mof52_10k`. |
| `results/mof52/strict` | Strict-constraint variant. | Former `outputs_dense_mof52_strict`. |
| `results/mof52/strict2` | Alternate strict tuning (seed/step variant). | Former `outputs_dense_mof52_strict2`. |

The intermediate 5k-step run was dropped to avoid duplicating the 10k dataset.

## R3 family

| Path | Contents | Notes |
| --- | --- | --- |
| `results/r3/dense` | Baseline dense packing run. | Former `outputs_dense_r3`. |
| `results/r3/strict` | Tight constraint setup. | Former `outputs_dense_r3_strict`. |
| `results/r3/energy` | Energy sampling / diagnostics. | Former `outputs_dense_r3_energy`. |
| `results/r3/force` | Force analysis sweep. | Former `outputs_dense_r3_force`. |
| `results/r3/overlap` | Overlap tracking output. | Former `outputs_dense_r3_overlap`. |

## Cube and truncated-cube studies

| Path | Contents | Notes |
| --- | --- | --- |
| `results/cube52/5k` | 5k-step dense cube (tiling-focused). | Former `outputs_dense_cube52_5k`. |
| `results/cube52/5k_html` | HTML/plot exports paired with the run above. | Former `outputs_dense_cube52_5k_html`. |
| `results/truncated_cube/210` | Truncated cube sweep (#210). | Former `outputs_truncated_cube_210`. |

## Other packed shapes

| Path | Contents | Notes |
| --- | --- | --- |
| `results/cube/5p1um` | Cube packing at 5.1 µm. | Former `outputs_cube_5p1um`. |
| `results/5p-series/5p2um` | Companion run at 5.2 µm. | Former `outputs_5p2um`. |
| `results/dense_small/run1` | First dense-small trial. | Former `outputs_dense_small`. |
| `results/dense_small/run2` | Second dense-small trial. | Former `outputs_dense_small2`. |
| `results/square/dense` | Dense square tilings. | Former `outputs_dense_square`. |
| `results/hexprism/5p8um` | Hexagonal prism run at 5.8 µm. | Former `outputs_hexprism_5p8um`. |

## Testbeds

| Path | Contents | Notes |
| --- | --- | --- |
| `results/tests/seed` | Seed sensitivity check. | Former `outputs_test_seed`. |
| `results/tests/trunc` | Truncation test run. | Former `outputs_test_trunc`. |

## Working with new outputs

1. Write new runs to `results/<project>/<label>` so that related data stays grouped.
2. Keep raw GSD/CSV/NPZ payloads inside a `raw/` subfolder (already ignored via `.gitignore`) and surface only lightweight summaries (plots, JSON metadata) at the directory root.
3. When a newer run supersedes an older one, move the obsolete folder into `results/archive/<YYYYMM>/` or delete it after confirming nothing depends on it.
