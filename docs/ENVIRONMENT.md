# Environment Summary

## Unified workspace environment (`.venv`)

All Monte Carlo, visualization, and HOOMD workflows now live inside the local Conda prefix `.venv`:

| Package (conda-forge) | Version | Notes |
| --- | --- | --- |
| `python` | 3.11.14 | Matches the new virtual environment interpreter. |
| `hoomd` | 5.4.0 (cpu) | Replaces the former `hoomd`, `hoomd53`, and `mc-hoomd` mamba environments. |
| `fresnel` | 0.13.8 | GPU/CPU rendering of HOOMD snapshots (`render_fresnel.py`). |
| `gsd` | 4.2.0 | Reads/writes simulation trajectories (`interactive_gsd_plot.py`, exporters). |
| `numpy` / `scipy` | 2.3.4 / 1.16.3 | Core numerics for Monte Carlo + post-processing. |
| `matplotlib` / `plotly` | 3.10.7 / 6.4.0 | Static + interactive visualization stacks. |
| `pillow` / `tqdm` / `narwhals` | 12.0.0 / 4.67.1 / 2.11.0 | Image IO, progress bars, and DataFrame helpers used throughout scripts. |

Recreate this environment with:

```bash
mamba create -p .venv -c conda-forge \
  python=3.11 hoomd=5.4.0 fresnel gsd \
  numpy scipy matplotlib plotly pillow tqdm
```

## Lock file

`uv.lock` still captures the subset of dependencies that are available from PyPI (pure Python + visualization tooling). It can be reused for lightweight, HOOMD-free setups, but the full `.venv` described above is the authoritative runtime for this repository.
