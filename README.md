# 项目说明与操作指南

## 运行环境（uv 全流程）
仓库默认在根目录 `.venv` 中运行所有蒙特卡洛、可视化与 HOOMD 相关脚本。核心版本（均基于 Python 3.11，本地编译 HOOMD/Fresnel）：

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| python | 3.11.x | 请使用 uv 提供的同架构解释器。 |
| hoomd | 5.4.0 (CPU 构建) | 从源码编译安装。 |
| fresnel | 0.13.8 | CPU 渲染，依赖 Embree/TBB。 |
| gsd | 4.2.0 | 轨迹读写。 |
| numpy / scipy | 2.3.4 / 1.16.3 | 数值运算。 |
| matplotlib / plotly | 3.10.7 / 6.4.0 | 可视化。 |
| pillow / tqdm / narwhals | 12.0.0 / 4.67.1 / 2.11.0 | IO、进度条、数据帧工具。 |

`uv.lock` 仅锁定纯 PyPI 依赖（不含 HOOMD/Fresnel）；权威环境以 `.venv` 为准。

## 使用 uv 重建完整环境（macOS，ARM64 与 x86_64 通用）
> 在 Apple Silicon 上请确认所需架构：`uname -m` 显示 `arm64`；如需 x86_64 兼容，可在 Rosetta Shell 里执行同样命令。所有步骤均无需 mamba/conda/brew。

1) **准备 uv 与 Python**
   - 确保安装了 Xcode Command Line Tools（提供 clang/ld）。
   - 让 uv 下载匹配架构的 Python：`uv python install 3.11.11`（或其他 3.11.x），然后 `uv venv --python 3.11.11 .venv`。
   - 激活：`source .venv/bin/activate`。

2) **安装 PyPI 依赖（锁定版本）**
   - 快速同步：`uv pip sync --python .venv/bin/python uv.lock`
   - 如需显式：`uv pip install --python .venv/bin/python numpy==2.3.4 scipy==1.16.3 matplotlib==3.10.7 plotly==6.4.0 pillow==12.0.0 tqdm==4.67.1 narwhals==2.11.0 gsd==4.2.0`

3) **编译工具链**
   - `uv pip install --python .venv/bin/python "cmake>=3.28" ninja`

4) **本地 C/C++ 依赖（统一安装到 `.venv/opt`）**
   - Eigen 3.4.0  
     ```
     curl -L -o /tmp/eigen-3.4.0.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
     tar -xzf /tmp/eigen-3.4.0.tar.gz -C /tmp
     cmake -S /tmp/eigen-3.4.0 -B /tmp/eigen-build -GNinja \
       -DCMAKE_MAKE_PROGRAM=.venv/bin/ninja -DCMAKE_INSTALL_PREFIX=.venv/opt/eigen -DCMAKE_BUILD_TYPE=Release
     cmake --build /tmp/eigen-build --target install
     ```
   - cereal 1.3.2  
     ```
     curl -L -o /tmp/cereal-1.3.2.tar.gz https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz
     tar -xzf /tmp/cereal-1.3.2.tar.gz -C /tmp
     cmake -S /tmp/cereal-1.3.2 -B /tmp/cereal-build -GNinja \
       -DCMAKE_MAKE_PROGRAM=.venv/bin/ninja -DCMAKE_INSTALL_PREFIX=.venv/opt/cereal -DCMAKE_BUILD_TYPE=Release -DJUST_INSTALL_CEREAL=ON
     cmake --build /tmp/cereal-build --target install
     ```
   - oneTBB 2021.13.0  
     ```
     curl -L -o /tmp/onetbb-2021.13.0.tar.gz https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.13.0.tar.gz
     tar -xzf /tmp/onetbb-2021.13.0.tar.gz -C /tmp
     cmake -S /tmp/oneTBB-2021.13.0 -B /tmp/oneTBB-build -GNinja \
       -DCMAKE_MAKE_PROGRAM=.venv/bin/ninja -DCMAKE_INSTALL_PREFIX=.venv/opt/tbb -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF
     cmake --build /tmp/oneTBB-build --target install
     ```
   - Embree 4.3.3（CPU 渲染后台）  
     ```
     curl -L -o /tmp/embree-4.3.3.tar.gz https://github.com/embree/embree/archive/refs/tags/v4.3.3.tar.gz
     tar -xzf /tmp/embree-4.3.3.tar.gz -C /tmp
     cmake -S /tmp/embree-4.3.3 -B /tmp/embree-build -GNinja \
       -DCMAKE_MAKE_PROGRAM=.venv/bin/ninja -DCMAKE_INSTALL_PREFIX=.venv/opt/embree -DCMAKE_BUILD_TYPE=Release \
       -DTBB_ROOT=.venv/opt/tbb -DEMBREE_TUTORIALS=OFF -DEMBREE_ISPC_SUPPORT=OFF -DEMBREE_TESTING=OFF
     cmake --build /tmp/embree-build --target install
     ```
   > 如需跨架构编译，可在每个 CMake 命令添加 `-DCMAKE_OSX_ARCHITECTURES=$(uname -m)`；Apple Silicon 在 Rosetta shell 下执行可构建 x86_64 版本。

5) **编译安装 HOOMD 5.4.0 (CPU)**
   ```
   git clone --branch v5.4.0 --recursive https://github.com/glotzerlab/hoomd-blue.git /tmp/hoomd-blue-5.4.0-src
   cmake -S /tmp/hoomd-blue-5.4.0-src -B /tmp/hoomd-build -GNinja \
     -DCMAKE_MAKE_PROGRAM=.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/.venv \
     -DPython_EXECUTABLE=$PWD/.venv/bin/python \
     -DCMAKE_PREFIX_PATH="$PWD/.venv/opt/eigen;$PWD/.venv/opt/cereal;$PWD/.venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11;$PWD/.venv"
   cmake --build /tmp/hoomd-build --target install
   ```
   若只需纯 PyPI 功能可跳过 HOOMD 构建；`dense-pack` 命令将不能运行。

6) **编译安装 Fresnel 0.13.8 (CPU 渲染)**
   ```
   curl -L -o /tmp/fresnel-0.13.8.tar.gz https://github.com/glotzerlab/fresnel/releases/download/v0.13.8/fresnel-0.13.8.tar.gz
   tar -xzf /tmp/fresnel-0.13.8.tar.gz -C /tmp
   cmake -S /tmp/fresnel-0.13.8 -B /tmp/fresnel-build -GNinja \
     -DCMAKE_MAKE_PROGRAM=.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$PWD/.venv \
     -DPYTHON_EXECUTABLE=$PWD/.venv/bin/python \
     -DCMAKE_PREFIX_PATH="$PWD/.venv/opt/embree;$PWD/.venv/opt/tbb;$PWD/.venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11;$PWD/.venv" \
     -DENABLE_OPTIX=OFF -DENABLE_EMBREE=ON -DTBB_ROOT=$PWD/.venv/opt/tbb
   cmake --build /tmp/fresnel-build --target install
   # 解决 macOS 运行时搜索路径
   install_name_tool -add_rpath "$PWD/.venv/opt/embree/lib" "$PWD/.venv/lib/python3.11/site-packages/fresnel/_cpu.cpython-311-darwin.so"
   install_name_tool -add_rpath "$PWD/.venv/opt/tbb/lib" "$PWD/.venv/lib/python3.11/site-packages/fresnel/_cpu.cpython-311-darwin.so"
   install_name_tool -add_rpath "$PWD/.venv/opt/tbb/lib" "$PWD/.venv/opt/embree/lib/libembree4.4.dylib"
   ```

7) **校验**
   ```
   python - <<'PY'
   import hoomd, fresnel, gsd, numpy, scipy, matplotlib, plotly
   from PIL import Image
   print("hoomd", hoomd.version.version)
   print("fresnel", fresnel.version.version)
   print("gsd", getattr(getattr(gsd, "version", None), "version", None))
   print("numpy", numpy.__version__)
   print("scipy", scipy.__version__)
   print("matplotlib", matplotlib.__version__)
   print("plotly", plotly.__version__)
   print("pillow", Image.__version__)
   PY
   ```

## 统一 CLI 入口
所有功能通过 `python -m src.cli.main <子命令> [参数]` 调用，推荐先 `source .venv/bin/activate`。子命令与用途：

| 子命令 | 说明 | 常见参数 |
| --- | --- | --- |
| `reversible` | 球面 RTT 凸体的可逆压缩 MC | `--mode run/insert --n --R --steps` |
| `hard-poly` | 球内硬多面体压缩 | `--shape --N --R --max-sweeps` |
| `tiling` | 正多边形球面铺展 | `--shape --radius --tile-side --output-dir` |
| `unidirectional` | 多边形单向吸附（含 `equirect`/`interactive`/`coverage`） | `--shape --R --edge/--cube-edge/...` |
| `viz-npz` | NPZ 渲染工具（硬多面体/可逆压缩） | `--state --png --html` |
| `dense-pack` | HOOMD 两阶段致密堆积（需已编译 HOOMD） | `--R-sphere --cube-edge --coverage --md-relax` |

### 常见命令示例
```
# 可逆压缩
python -m src.cli.main reversible --mode run --n 50 --R 8 --steps 1000 --out outputs/my_reversible.npz
# 可视化 NPZ
python -m src.cli.main viz-npz --state outputs/my_reversible.npz --png outputs/my_reversible.png

# 硬多面体
python -m src.cli.main hard-poly --shape cube --N 20 --R 6 --max-sweeps 50 --out outputs/my_hard_poly.npz

# 球面铺展
python -m src.cli.main tiling --shape square --radius 4 --tile-side 1.0 --output-dir outputs/my_tiling

# 单向吸附 equirect 投影
python -m src.cli.main unidirectional equirect --shape square --R 8 --edge 1.5 --outdir outputs/my_adsorption
```

### 球面铺展结果复现
- **断点重渲染**：目录含 `checkpoint*.npz` 时，`python -m src.cli.main tiling --resume-from <checkpoint> --output-dir <目录>` 可快速生成投影/3D/HTML。
- **完整重跑**：用 `result_summary.json` 字段映射参数：`radius_nm -> --radius`、`tile_side_nm -> --tile-side`、`shape -> --shape`、`max_steps -> --max-steps`、`stall_steps -> --stall-steps`、`insertion_probability -> --insert-prob`、`translation_step_deg -> --translation-step`、`rotation_step_deg -> --rotation-step`，其余 gap/targeted/global/energy/force 字段直接同名传入。长任务可加 `--progress`。

### 已复现的数据
`outputs/reproduction/` 提供代表性示例：
- 可逆压缩：`reversible.npz` → `viz-npz` 生成 `reversible.png`。
- 硬多面体：`hard_poly.npz/json` → `viz-npz` 生成 `hard_poly.png`。
- 球面镶嵌：`tiling` 自动生成 `projection.png`、`view3d.png`（如传 HTML 也会输出）。
- 单向吸附：`unidirectional equirect` 得到 `adsorption_square.png`。
- 致密堆积：依赖 HOOMD；若未安装 HOOMD，CLI 会提示缺依赖。

### 结果目录组织（摘要）
为保持根目录整洁，权威数据放在 `outputs/` 下：
- MOF 5:2 系列：`results_mof52_10k`、`results_mof52_strict`、`results_mof52_strict2`
- R3 系列：`results_r3_dense`、`results_r3_strict`、`results_r3_energy`、`results_r3_force`、`results_r3_overlap`
- 立方体及截断立方体：`results_cube52_5k`、`results_cube52_5k_html`、`results_truncated_cube_210`
- 其他形状：`results_cube_5p1um`、`results_5p2um`、`results_dense_small_run1/2`、`results_square_dense`、`results_hexprism_5p8um`
- 测试集：`results_tests_seed`、`results_tests_trunc`

新增结果建议放在 `outputs/<项目>/<标签>`，大体积原始数据置于子目录（如 `raw/`）；若替代旧版本，可移入 `outputs/archive/<年月>/` 或删除。
