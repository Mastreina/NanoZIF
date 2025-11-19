#!/usr/bin/env python3
import pathlib
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = pathlib.Path(__file__).with_name("log.csv")
if not LOG_PATH.exists():
    raise SystemExit(f"找不到 {LOG_PATH}，请先运行模拟生成 log.csv。")

data = np.genfromtxt(LOG_PATH, delimiter=",", names=True, autostrip=True)
field_t, field_T, field_E = data.dtype.names
timestep = data[field_t]
temperature = data[field_T]
energy = data[field_E]

fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax1.set_xlabel("Simulation timestep")
ax1.set_ylabel("Kinetic temperature (kT)", color="tab:blue")
ax1.plot(timestep, temperature, color="tab:blue", label="kinetic T")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("WCA energy", color="tab:red")
ax2.plot(timestep, energy, color="tab:red", label="WCA energy")
ax2.tick_params(axis="y", labelcolor="tab:red")

fig.suptitle("MD RATTLE stage observables")
fig.tight_layout()
output = LOG_PATH.with_name("log_plot.png")
fig.savefig(output, dpi=160)
print(f"完成：{output}")
