# Step-by-Step Guide: Conversational Copilot & Bidirectional GDSII Workflows

This tutorial walks through the core interactive workflows in **MicroRing Studio v3.6.0**.

---

## 1. Conversational Design Prompts Cheat-Sheet

The **Photonics Design Copilot** (`src/microring/copilot.py`) parses physical specifications in English and Simplified Chinese at any point in the conversation:

| Goal | Example Natural-Language Prompts | What MicroRing Executes |
| :--- | :--- | :--- |
| **Set Platform & Stack** | `"220 nm SOI strip, SiO2 clad, TE0 at 1550 nm"`<br/>`"600 nm X-cut TFLN with 300 nm slab, 70 deg sidewall"`<br/>`"800 nm ScAlN 20% with 200 nm slab"` | Updates `MRRDesignConfig`, runs a 7-point 2D FEM width sweep, finds `TE1`/`TM1` single-mode cutoff, and sets `W_bus` & `W_ring`. |
| **Hit a Target FSR** | `"FSR 200 GHz, point circular all-pass"`<br/>`"FSR 100 GHz, 4-port pulley add-drop"` | Uses FEM group index $n_g$ to solve exact radius $R$, runs bent FEM + spatial CMT, and plots Through/Drop spectra. |
| **Critical Coupling** | `"Apply critical coupling"` / `"临界耦合"` | Sets `gap_thru_um`, `pulley_angle_thru_deg`, or `straight_length_um` to exact critical coupling ($Q_{\mathrm{ext}} = Q_{\mathrm{int}}$). |
| **Chip I/O Couplers (Butt & GC)** | `"Design apodized focusing grating coupler on 220 nm Si"`<br/>`"Design butt coupler to 2.5 um lensed fiber"`<br/>`"Export focusing GC 127um fiber array loopback GDS"` | Runs native 2D Full-Wave `FDFD` (Grating Coupler) or 2D Vector FEM + Poynting overlap (Butt/Edge Coupler), plots `IL(λ)` / 2D radiated $|E(z,y)|$ / alignment tolerances, and exports `.GDS` or `INTERCONNECT` `SPAR`. |
| **High-Q Convergence vs FDTD** | `"Why is high-Q Si3N4 ring slow in Lumerical FDTD?"`<br/>`"Export high-Q coupler cutout GDS and lsf"` | Computes exact $M \to \infty$ steady-state resonance vs truncated-FDTD sinc-ringing, required `1e-5` simulation time, and exports `< 1 ps` Coupler-Only Cutout `.GDS` + `.lsf`. |
| **MZI (`Y-Branch` / `DC` / `MMI`) & Modulators** | `"Design a 220 nm SOI 2x2 MMI push-pull PN MZI"`<br/>`"600 nm X-cut TFLN Y-branch push-pull EO MZI arm length 5 mm"`<br/>`"Export mmi_2x2 MZI GDS and INTERCONNECT"` | Solves `DC` supermodes, `MMI` multi-mode 2D-FEM `EME`, or `Y-Branch` supermode overlap without 3D FDTD, computes $V_\pi L$, $P_\pi$, non-cavity `3-dB` BW, 2D MMI $|E(x,z)|$ self-imaging, `.GDS`, and INTERCONNECT `SPAR + OMM`. |
| **Run Tolerance Sweeps** | `"Sweep width 0.5 to 1.8 um"`<br/>`"Sweep gap 150 to 320 nm"`<br/>`"Sweep pulley angle 8 to 35 deg"` | Runs multi-point 2D FEM / CMT sweeps and plots `ER [dB]` or `n_eff(W)` on the right workbench. |
| **Active Tuning & INTERCONNECT** | `"Si PN junction tuning spectra at 0, -2, -5 V"`<br/>`"Temperature tuning spectra at 0, 20, 50 K"`<br/>`"E-field tuning spectra at -10, 0, 10 V/um"`<br/>`"Export Lumerical INTERCONNECT compact model"` | Solves 2D-FEM cross-section multiphysics + perturbed 2D vector eigenmodes (zero COMSOL), plots multi-state spectra, and exports a turnkey `.zip` bundle (`SPAR .dat`, `OMM .txt`, `.lsf`, `lumapi`). |
| **Create Foundry GDS** | `"Export single GDS layout"`<br/>`"Export 5-ring parameter sweep array GDS"` | Writes `.gds` + KLayout `.lyp` to `outputs/gds_exports/` and opens interactive polygon preview + download buttons. |

---

## 2. Inspecting & Simulating External `.GDS` / `.OAS` Files

`microring.gds_bridge.analyze_uploaded_gds` operates directly on binary polygons (`gdstk` + `shapely.ops.unary_union`), so it works on layouts drawn in **KLayout**, **`gdsfactory`**, **Luceda IPKISS**, or manual scripts:

1. **Core Layer Selection**: Automatically selects `(1, 0)` if present, or scans all `(layer, datatype)` pairs for closed annular cavities (`len(polygon.interiors) >= 1`). You can also pass `core_layer=(layer, datatype)` explicitly in Python or CLI.
2. **Topological Classification**:
   - **2+ Annular Cavities** $\rightarrow$ `vernier_crow_ad` (extracts $R_1$, $R_2$, and inter-ring gap).
   - **Elongated Bounding Box ($|D_x - D_y| \ge 0.40\ \mu\mathrm{m}$)** $\rightarrow$ `racetrack_ap` or `racetrack_ad` (extracts $R$ and straight coupler length $L_s$).
   - **Concentric Curved Bus Wrapping Around Ring** $\rightarrow$ `pulley_ap` or `pulley_ad` (compares angular proximity arc against a tangent straight-bus chord to measure `pulley_angle_thru_deg`).
   - **Tangent Straight Bus(es)** $\rightarrow$ `point_circular_ap` (1 bus) or `point_circular_ad` (2 buses).

### CLI Inspection Example

```bash
uv run microring analyze-gds outputs/cookbook_gds/sin_pulley_add_drop_single.gds
```

---

## 3. Viewing Exported Layouts in KLayout

Every GDS export automatically writes a matching `.lyp` layer-properties file:

| Role | GDS Layer/Datatype | Default Hex Color | Purpose |
| :--- | :--- | :--- | :--- |
| `core` | `1/0` | `#38bdf8` | Full-etch or rib waveguide core |
| `slab` | `2/0` | `#0284c7` | Partial-etch slab region (`H_slab > 0`) |
| `heater` | `10/0` | `#f59e0b` | Thermo-optic TiN micro-heater arc |
| `electrode` | `11/0` | `#eab308` | Electro-optic GSG RF/DC pads |
| `pin` | `65/0` | `#34d399` | Optical port markers (`In`, `Through`, `Add`, `Drop`) |
| `label` | `66/0` | `#e2e8f0` | Human-readable parameter annotations |

Open directly with colors applied:

```bash
klayout outputs/cookbook_gds/sin_pulley_5x_angle_sweep.gds -l outputs/cookbook_gds/sin_pulley_5x_angle_sweep.lyp
```
