# AGENTS.md: Token-Minimal Local-Compute Protocol for AI Agents

When helping users design, sweep, fit, or lay out Photonic IC microring resonators, chip I/O couplers (Butt/Edge & Grating Couplers), or High-Q cavities in this repository, **strictly minimize LLM token usage** and **offload 100% of numerical work to local CPU / Apple Silicon AMX / NVIDIA CUDA compute**.

## Golden Token-Saving Rules
1. **Do NOT read `src/microring/*.py` source files** unless the user explicitly asks to modify solver internals.
2. **Do NOT print or inspect raw spectrum/polygon arrays** in stdout. All `microring agent` commands automatically save full `.csv` / `.gds` / `.lyp` artifacts under `outputs/agent_runs/` and return **<150 tokens of single-line scalar JSON** (plus persistent local SSD eigenmode caching in `outputs/.mode_cache/`).
3. Prefer MCP tools (`microring_*` if configured via `.mcp.json`) or the single-command `microring agent` CLI below (`uv run microring agent ...` or `PYTHONPATH=src python -m microring.cli agent ...`).

## One-Command Local-Compute CLI (`<150 tokens` stdout)

| Task | Command |
| :--- | :--- |
| **Probe GPU/AMX/CPU & valid keys** | `uv run microring agent status` |
| **2D FEM single-mode width cutoff** | `uv run microring agent width-cutoff --mat Si3N4 --h 0.40 --wl 1550 --pol TE0 --wmin 0.6 --wmax 2.0` |
| **Synthesize MRR + Critical Couple + GDS** | `uv run microring agent design --top pulley_ap --mat 4H-SiC --h 0.60 --w-ring 1.10 --w-bus 0.82 --fsr 150 --gap 0.24 --cc --gds` |
| **Multi-point parameter sweep (+ MPW GDS)** | `uv run microring agent sweep --top pulley_ap --mat 4H-SiC --param pulley_angle_thru_deg --min 6 --max 24 --n 7 --array-gds` |
| **Extract geometry from `.gds` / `.oas`** | `uv run microring agent inspect-gds path/to/layout.gds` |
| **Fit lab spectrum CSV (`Q_int`, doublet)** | `uv run microring agent fit path/to/trace.csv --model auto --regime under` |
| **Positive/Negative PR dose & CD error** | `uv run microring agent fab-tolerance --mat Si3N4 --top pulley_ap --resist positive_pr --dose-pct 15 --bias-gds` |
| **2D-FEM TO/EO/Si-PN tuning & INTERCONNECT** | `uv run microring agent active-tuning --mat Si --h 0.22 --slab 0.09 --w-ring 0.48 --w-bus 0.44 --r 20 --mech si_pn_depletion --biases "0,-2,-5" --interconnect` |
| **Butt/Edge or 2D-FDFD Grating Coupler** | `uv run microring agent io-coupler --type grating_coupler --mat Si --h 0.22 --etch 0.07 --angle 10 --loopback --interconnect` |
| **High-Q convergence & 0.8 ps FDTD cutout** | `uv run microring agent high-q --mat Si3N4 --top pulley_ap --r 100 --loss 0.08 --gap 0.42 --cutout` |
| **2D-FEM/EME MZI (`DC`/`MMI`/`Y`) & Modulator** | `uv run microring agent mzi --mat Si --h 0.22 --slab 0.09 --splitter mmi_2x2 --l1 1500 --dl 60 --mech si_pn_depletion --gds --interconnect` |

## Valid Keys Quick Reference
- **7 Topologies (`--top`)**: `point_circular_ap`, `point_circular_ad`, `racetrack_ap`, `racetrack_ad`, `pulley_ap`, `pulley_ad`, `vernier_crow_ad`
- **24 Multi-Waveband Materials (`--mat`, `--clad`, `--top-clad` from `400 nm` Visible to `14 um` Mid-IR)**:
  - *Visible / VNIR (`400–950 nm`)*: `TiO2`, `Ta2O5`, `GaN`, `Diamond`, `Al2O3`, `AlN`, `Si3N4`, `4H-SiC`, `LN_Zcut`, `LN_Xcut`
  - *O-Band (`1310 nm`), C-Band (`1550 nm`) & SWIR (`2000 nm`)*: `Si`, `a-Si`, `Si3N4`, `SiRich_SiNx`, `4H-SiC`, `3C-SiC`, `AlN`, `ScAlN_10`, `ScAlN_20`, `ScAlN_30`, `LN_Zcut`, `LN_Xcut`, `GSST_amorphous`, `GSST_crystalline`
  - *Mid-IR (`3.0–14.0 um`)*: `Ge`, `Diamond`, `Si`, `CaF2`, `MgF2`, `SiO2`, `Air`
