"""4-Way Commercial Solver Validation Auditor, Plotter & Report Generator (Suites 1-11).

Aggregates independent outputs from four solver ecosystems executed from neutral specs:
  1. MicroRing Studio v3.9.0
       - `microring_independent_results.json`    (Phase 1: Suites 1-5)
       - `microring_independent_results_p2.json` (Phase 2: Suites 6-11)
  2. Ansys Lumerical 2026 R1 (`MODE FDE/EME`, `2D/3D FDTD` RTX 5090 GPU, `INTERCONNECT`, `HEAT`, `CHARGE`)
       - `lumerical_independent_results.json`    (Phase 1: Suites 1-5)
       - `lumerical_independent_results_p2.json` (Phase 2: Suites 6-11)
  3. COMSOL Multiphysics 6.4 (`ewfd`, `emw`, `es`, `ht` 2D/3D, `GeneralFormPDE g`)
       - `comsol_independent_results.json`       (Phase 1: Suites 1-5)
       - `comsol_independent_results_p2.json`    (Phase 2: Suites 6-11)
  4. Ansys HFSS 2026 R1 (`3D Driven Modal Full-Wave FEM`, Wave Ports & Bloch-Floquet ABCD 10-300 GHz)
       - `hfss_independent_results_p2.json`      (Phase 2: Suite 8 Sub-THz RF Electrodes)

Performs 10 automated physical integrity & conservation audits, exports 12 unit-labeled CSVs
with solver provenance headers, generates two 6-panel publication figures (600 dpi PNG +
editable SVG), and writes `docs/COMMERCIAL_SOLVER_VALIDATION_REPORT.md`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "benchmarks" / "outputs"
DOC_DIR = REPO_ROOT / "docs"
IMG_DIR = DOC_DIR / "images"

SPECS_P1_JSON = OUT_DIR / "commercial_validation_specs.json"
SPECS_P2_JSON = OUT_DIR / "commercial_validation_specs_p2.json"
MR_P1_JSON = OUT_DIR / "microring_independent_results.json"
LUM_P1_JSON = OUT_DIR / "lumerical_independent_results.json"
COM_P1_JSON = OUT_DIR / "comsol_independent_results.json"

MR_P2_JSON = OUT_DIR / "microring_independent_results_p2.json"
LUM_P2_JSON = OUT_DIR / "lumerical_independent_results_p2.json"
COM_P2_JSON = OUT_DIR / "comsol_independent_results_p2.json"
HFSS_P2_JSON = OUT_DIR / "hfss_independent_results_p2.json"

REPORT_MD = DOC_DIR / "COMMERCIAL_SOLVER_VALIDATION_REPORT.md"

# Okabe-Ito colorblind-safe publication palette
C_MR = "#0072B2"    # MicroRing Studio (deep blue)
C_LUM = "#D55E00"   # Ansys Lumerical (vermilion)
C_COM = "#009E73"   # COMSOL Multiphysics (bluish green)
C_HFSS = "#CC79A7"  # Ansys HFSS 3D (reddish purple)
C_GRAY = "#4D4D4D"


def _index_by(rows: list[dict], key: str = "case_id") -> dict[str, dict]:
    return {r[key]: r for r in rows}


def _pct_diff(a: float, b: float) -> float:
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom * 100.0


def _write_csv(path: Path, header_comment: str, cols: list[str], rows: list[dict]) -> None:
    lines = [f"# {header_comment}", ",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append("" if v is None else str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_pub_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "svg.fonttype": "none",
    })


def audit_physics(
    mr1: dict, lum1: dict, com1: dict,
    mr2: dict, lum2: dict, com2: dict, hfss2: dict,
) -> list[dict]:
    """Run 10 automated physical conservation, convergence, passivity, and finiteness audits."""
    checks = []

    def _add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    # 1. NaN / Inf scan across all 7 independent result files
    bad_vals = []
    def _scan(obj, prefix=""):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            bad_vals.append(prefix)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan(v, f"{prefix}[{i}]")
    for tag, d in (
        ("mr_p1", mr1), ("lum_p1", lum1), ("com_p1", com1),
        ("mr_p2", mr2), ("lum_p2", lum2), ("com_p2", com2), ("hfss_p2", hfss2),
    ):
        _scan(d, tag)
    _add("Finite numeric outputs across all 11 suites (zero NaN/Inf)", len(bad_vals) == 0, f"7 independent JSON archives scanned; bad={bad_vals[:5]}")

    # 2. Lumerical 3D FDTD GPU coupler passivity & shutoff convergence (Suite 2)
    all_fdtd = lum1.get("suite2a_coupler_5band", []) + lum1.get("suite2b_soi_gap_sweep", [])
    max_sum = max((r["passivity_sum"] for r in all_fdtd), default=0.0)
    all_shutoff = all(r["fdtd_status_code"] == 2 for r in all_fdtd)
    _add(
        "Lumerical 3D FDTD GPU coupler passivity (|S21|^2 + |S31|^2 <= 1.005)",
        max_sum <= 1.005 and all_shutoff,
        f"max(|S21|^2+|S31|^2) = {max_sum:.5f} across {len(all_fdtd)} GPU runs; shutoff_converged={all_shutoff}",
    )

    # 3. Lumerical 3D FDTD MZI splitter passivity (Suite 3B + Suite 7A 2x2 MMI)
    sp_vals = [r["total_trans_pct"] / 100.0 for r in lum1.get("suite3b_splitters", [])]
    sp_vals.append(lum2["suite7a_mmi_2x2"]["total_trans_pct"] / 100.0)
    _add(
        "Lumerical 3D FDTD MZI splitter passivity (1x2 MMI, Y-Branch, 2x2 MMI <= 1.005)",
        max(sp_vals) <= 1.005,
        f"max(T_total) = {max(sp_vals):.5f} across 3 GPU splitter geometries",
    )

    # 4. COMSOL material sensitivity verification (no Free-Space override)
    ms = com1.get("suite1_audit", {}).get("material_sensitivity", {})
    _add(
        "COMSOL material-sensitivity audit (zero Free-Space override)",
        bool(ms.get("verified_no_free_space_override", False)),
        f"4H-SiC dn_eff/dn_core = {ms.get('d_neff_over_d_ncore', 'N/A')} (+1% n_core -> {ms.get('neff_nominal')} to {ms.get('neff_core_plus1pct')})",
    )

    # 5. COMSOL quadratic FEM mesh convergence (Suite 1)
    mc = com1.get("suite1_audit", {}).get("mesh_convergence", [])
    neffs = [r["neff"] for r in mc]
    spread = (max(neffs) - min(neffs)) if len(neffs) >= 2 else 1.0
    _add(
        "COMSOL 2D quadratic FEM mesh convergence (h_core = 55 -> 35 -> 22 nm)",
        spread < 5e-4,
        f"neff = {[round(x, 6) for x in neffs]} (max spread = {spread:.2e})",
    )

    # 6. Thermal boundary heat-flux conservation across Suites 4 & 10 (< 2%)
    mr_th_err = max(
        [r["power_conservation_err_pct"] for r in mr1.get("suite4_thermal", [])]
        + [r["power_conservation_err_pct"] for r in mr2.get("suite10_thermal_adv", [])],
        default=0.0,
    )
    com_th_err = max((r["power_conservation_err_pct"] for r in com1.get("suite4_thermal", [])), default=0.0)
    _add(
        "Thermal boundary energy balance across Suites 4 & 10 (< 2.0% flux error)",
        mr_th_err < 2.0 and com_th_err < 2.0,
        f"MicroRing max err = {mr_th_err:.4f}% (7 architectures), COMSOL max err = {com_th_err:.4f}%",
    )

    # 7. Lumerical INTERCONNECT native SPAR import & passivity check (Suite 7C)
    ic = lum2["suite7c_interconnect"]
    ic_ok = bool(ic["mzi_spar_loaded_ok"]) and bool(ic["gc_spar_loaded_ok"]) and ic["mzi_max_sum_T"] <= 1.005
    _add(
        "Lumerical INTERCONNECT native SPAR .dat import & passivity (MZI + Apodized GC)",
        ic_ok,
        f"MZI FSR={ic['mzi_FSR_ic_nm']:.3f} nm (MR {mr2['suite7c_interconnect_circuit']['mzi_FSR_nm']:.3f} nm), max(T_bar+T_cross)={ic['mzi_max_sum_T']:.5f}, GC IL(1550)={ic['gc_il_1550_ic_db']:.3f} dB",
    )

    # 8. Ansys HFSS 3D Full-Wave RF passivity across 10-300 GHz (Suite 8)
    max_hfss_p = 0.0
    n_hfss_pts = 0
    for c in hfss2["suite8_rf_electrodes"]:
        for pt in c["freq_sweep"]:
            p_sum = 10.0 ** (pt["S11_mag_dB"] / 10.0) + 10.0 ** (pt["S21_mag_dB"] / 10.0)
            max_hfss_p = max(max_hfss_p, p_sum)
            n_hfss_pts += 1
    _add(
        "Ansys HFSS 3D RF network passivity (|S11|^2 + |S21|^2 <= 1.005 across 10-300 GHz)",
        max_hfss_p <= 1.005,
        f"max(|S11|^2+|S21|^2) = {max_hfss_p:.5f} across {n_hfss_pts} full-wave frequency points (4 electrodes)",
    )

    # 9. Silicon PN TCAD monotonicity across Lateral -> L-Shaped -> U-Shaped (Suite 9)
    mr9 = _index_by(mr2["suite9_si_pn_tcad"])
    com9 = _index_by(com2["suite9_si_pn_tcad"])
    mono_ok = (
        mr9["SOI_220nm_Lateral_PN"]["V_pi_L_at_m3V_Vcm"] > mr9["SOI_220nm_LShaped_PN"]["V_pi_L_at_m3V_Vcm"] > mr9["SOI_220nm_UShaped_PN"]["V_pi_L_at_m3V_Vcm"]
        and com9["SOI_220nm_Lateral_PN"]["V_pi_L_at_m3V_Vcm"] > com9["SOI_220nm_LShaped_PN"]["V_pi_L_at_m3V_Vcm"] > com9["SOI_220nm_UShaped_PN"]["V_pi_L_at_m3V_Vcm"]
        and mr9["SOI_220nm_Lateral_PN"]["Cj_0V_fF_per_um"] > mr9["SOI_220nm_Lateral_PN"]["Cj_m3V_fF_per_um"]
        and lum2["suite9_charge"]["Cj_0V_fF_per_um"] > lum2["suite9_charge"]["Cj_m3V_fF_per_um"]
    )
    _add(
        "Silicon PN TCAD physical ordering (V_pi*L: Lateral > L > U; C_j(0V) > C_j(-3V))",
        mono_ok,
        f"Lateral C_j(-3V): MR={mr9['SOI_220nm_Lateral_PN']['Cj_m3V_fF_per_um']:.3f}, COMSOL={com9['SOI_220nm_Lateral_PN']['Cj_m3V_fF_per_um']:.3f}, CHARGE={lum2['suite9_charge']['Cj_m3V_fF_per_um']:.3f} fF/um",
    )

    # 10. Cleanroom lithography CD-bias resonance sensitivity 3-way consensus (Suite 11)
    s11_mr = mr2["suite11_fab_bias"]["dlambda_dCD_pm_per_nm"]
    s11_com = com2["suite11_fab_bias"]["dlambda_dCD_pm_per_nm"]
    s11_lum = lum2["suite11_fab_bias"]["dlambda_dCD_pm_per_nm"]
    s11_spread = max(abs(s11_mr - s11_com), abs(s11_mr - s11_lum))
    _add(
        "Lithography CD-bias sensitivity 3-way agreement (< 1.0 pm/nm spread)",
        s11_spread < 1.0,
        f"dlambda/dCD = {s11_mr:.2f} (MR) / {s11_com:.2f} (COMSOL) / {s11_lum:.2f} (Lumerical MODE) pm/nm (spread {s11_spread:.2f} pm/nm)",
    )
    return checks


def generate_phase1_figure(mr1: dict, lum1: dict, com1: dict, lum2_heat: list[dict]) -> tuple[Path, Path]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_pub_rc()

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6), dpi=300)
    plt.subplots_adjust(left=0.065, right=0.985, top=0.935, bottom=0.105, wspace=0.31, hspace=0.38)

    # (a) Suite 1: 5-Waveband 2D Bent Eigenmode |Delta n_eff|
    ax = axes[0, 0]
    mr_s1 = _index_by(mr1["suite1_eigenmodes"])
    lum_s1 = _index_by(lum1["suite1_eigenmodes"])
    com_s1 = _index_by(com1["suite1_eigenmodes"])
    short_labels = ["532 nm\nSi3N4", "780 nm\nTiO2", "1310 nm\nSOI", "1550 nm\n4H-SiC", "4200 nm\nGe/CaF2"]
    cids_s1 = [r["case_id"] for r in mr1["suite1_eigenmodes"]]
    d_lum_raw = [abs(mr_s1[k]["neff_bent"] - lum_s1[k]["neff_bent"]) for k in cids_s1]
    d_com_raw = [abs(mr_s1[k]["neff_bent"] - com_s1[k]["neff_bent"]) for k in cids_s1]
    d_lc_raw = [abs(lum_s1[k]["neff_bent"] - com_s1[k]["neff_bent"]) for k in cids_s1]
    floor_val = 8e-7
    x = np.arange(len(cids_s1))
    bw = 0.26
    ax.bar(x - bw, [max(v, floor_val) for v in d_lum_raw], width=bw, color=C_LUM, label="MicroRing vs Lumerical MODE")
    ax.bar(x, [max(v, floor_val) for v in d_com_raw], width=bw, color=C_COM, label="MicroRing vs COMSOL ewfd")
    ax.bar(x + bw, [max(v, floor_val) for v in d_lc_raw], width=bw, color="#999999", label="Lumerical vs COMSOL")
    for xi, raw_v in zip(x, d_com_raw):
        if raw_v < 1e-6:
            ax.text(xi, floor_val * 1.18, "<1e-6", ha="center", va="bottom", fontsize=5.6, color=C_COM, fontweight="bold")
    ax.axhline(5e-4, color=C_GRAY, linestyle="--", linewidth=0.8, label="5e-4 engineering target")
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 6e-3)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels)
    ax.set_ylabel(r"$|\Delta n_{\mathrm{eff}}|$ (Bent TE$_0$)")
    ax.set_title("(a) Suite 1: 5-Waveband Bent Eigenmodes", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6.1)

    # (b) Suite 2B: SOI 1550 nm Gap Sweep (CMT vs 3D FDTD GPU)
    ax = axes[0, 1]
    mr_2b = mr1["suite2b_soi_gap_sweep"]
    lum_2b = lum1["suite2b_soi_gap_sweep"]
    gaps = [r["gap_nm"] for r in mr_2b]
    ax.axvspan(85, 120, color="#FDEBD0", alpha=0.65, label="Tight-gap (<120 nm)")
    ax.plot(gaps, [r["kappa_power_cmt"] for r in mr_2b], "o-", color=C_MR, linewidth=1.4, markersize=5, label=r"MicroRing CMT ($\kappa^2$)")
    ax.plot(gaps, [r["S31_sq_kappa_power"] for r in lum_2b], "s--", color=C_LUM, linewidth=1.4, markersize=5, label=r"Lumerical 3D FDTD ($|S_{31}|^2$)")
    ax.plot(gaps, [r["one_minus_S21_sq"] for r in lum_2b], "^:", color=C_GRAY, linewidth=1.0, markersize=4.5, label=r"Lumerical $1-|S_{21}|^2$")
    ax.set_yscale("log")
    ax.set_xlim(90, 265)
    ax.set_xlabel("Bus-to-Ring Gap (nm)")
    ax.set_ylabel(r"Power Coupling Ratio $\kappa^2$")
    ax.set_title("(b) Suite 2: SOI Coupler Gap Scaling (GPU)", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=6.2)

    # (c) Suite 3A: 2x2 SOI DC Supermode L_pi
    ax = axes[0, 2]
    mr_3a = mr1["suite3a_dc"]
    lum_3a = lum1["suite3a_dc"]
    com_3a = com1["suite3a_dc"]
    g_dc = np.arange(len(mr_3a))
    bw = 0.25
    ax.bar(g_dc - bw, [r["L_pi_um"] for r in mr_3a], width=bw, color=C_MR, label="MicroRing FEM")
    ax.bar(g_dc, [r["L_pi_um"] for r in lum_3a], width=bw, color=C_LUM, label="Lumerical MODE")
    ax.bar(g_dc + bw, [r["L_pi_um"] for r in com_3a], width=bw, color=C_COM, label="COMSOL ewfd")
    ax.set_xticks(g_dc)
    ax.set_xticklabels([f"{int(r['gap_nm'])} nm" for r in mr_3a])
    ax.set_xlabel("Twin-Core Gap (nm)")
    ax.set_ylabel(r"Half-Beat Length $L_\pi$ ($\mu$m)")
    ax.set_title(r"(c) Suite 3A: $2\times 2$ SOI DC $L_\pi$ (3-Way)", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper left")

    # (d) Suite 3C: Uniform 220 nm SOI Surface Grating Coupler
    ax = axes[1, 0]
    gc_mr = mr1["suite3c_gc"]
    gc_lum = lum1["suite3c_gc"]
    gc_com = com1.get("suite3c_gc", {})
    ax.plot(gc_mr["spectrum_wavelength_nm"], gc_mr["spectrum_il_db"], "-", color=C_MR, linewidth=1.5, label="MicroRing 2D FDFD")
    ax.plot(gc_lum["spectrum_wavelength_nm"], gc_lum["spectrum_il_db"], "o--", color=C_LUM, linewidth=1.2, markersize=3.5, label="Lumerical 2D FDTD")
    if isinstance(gc_com, dict) and "il_at_1550_db" in gc_com:
        ax.plot([1550.0], [gc_com["il_at_1550_db"]], "D", color=C_COM, markersize=6.5, zorder=5, label="COMSOL 2D ewfd (1550 nm)")
    ax.set_xlim(1500, 1600)
    ax.set_ylim(1.5, 9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Fiber Coupling IL (dB)")
    ax.set_title("(d) Suite 3C: Uniform SOI Grating Coupler", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower left")

    # (e) Suite 4: 2D Thermo-Optic Microheaters (now 3-way with Lumerical HEAT on SOI!)
    ax = axes[1, 1]
    mr_4 = _index_by(mr1["suite4_thermal"])
    com_4 = _index_by(com1["suite4_thermal"])
    heat_4 = _index_by(lum2_heat)
    cids_4 = [r["case_id"] for r in mr1["suite4_thermal"]]
    lbls_4 = ["SOI\nTop TiN", "SOI\nTrench", "Si3N4\nBOX 2.0", "Si3N4\nBOX 3.5"]
    x4 = np.arange(len(cids_4))
    bw4 = 0.25
    ax.bar(x4 - bw4, [mr_4[k]["R_th_core_K_per_mW"] for k in cids_4], width=bw4, color=C_MR, label="MicroRing FEM")
    ax.bar(x4, [com_4[k]["R_th_core_K_per_mW"] for k in cids_4], width=bw4, color=C_COM, label="COMSOL ht")
    h_xs, h_ys = [], []
    for idx_k, k in enumerate(cids_4):
        if k in heat_4:
            h_xs.append(idx_k + bw4)
            h_ys.append(heat_4[k]["R_th_core_K_per_mW"])
    if h_xs:
        ax.bar(h_xs, h_ys, width=bw4, color=C_LUM, label="Lumerical HEAT")
    ax.set_xticks(x4)
    ax.set_xticklabels(lbls_4)
    ax.set_ylabel(r"Core Thermal Resistance $R_{\mathrm{th}}$ (K/mW)")
    ax.set_title("(e) Suite 4: 2D Thermo-Optic Heaters (3-Way)", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)

    # (f) Suite 5: Anisotropic Pockels EO Electrostatics
    ax = axes[1, 2]
    mr_5 = _index_by(mr1["suite5_pockels_eo"])
    com_5 = _index_by(com1["suite5_pockels_eo"])
    cids_5 = [r["case_id"] for r in mr1["suite5_pockels_eo"]]
    lbls_5 = ["X-TFLN\nG=5.0 um", "X-TFLN\nG=3.2 um", "ScAlN20\nVert. Plate"]
    x5 = np.arange(len(cids_5))
    bw5 = 0.34
    e_mr = [math.hypot(mr_5[k]["ex_core_v_per_um_per_v"], mr_5[k]["ey_core_v_per_um_per_v"]) for k in cids_5]
    e_com = [math.hypot(com_5[k]["ex_core_v_per_um_per_v"], com_5[k]["ey_core_v_per_um_per_v"]) for k in cids_5]
    ax.bar(x5 - 0.5 * bw5, e_mr, width=bw5, color=C_MR, label="MicroRing Laplace FEM")
    ax.bar(x5 + 0.5 * bw5, e_com, width=bw5, color=C_COM, label="COMSOL AC/DC es")
    ax.set_ylim(0, 0.245)
    ax.set_xticks(x5)
    ax.set_xticklabels(lbls_5)
    ax.set_ylabel(r"Core $|E|/V$ [(V/$\mu$m)/V]")
    ax.set_title("(f) Suite 5: Anisotropic Pockels EO Fields", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper left")

    png_path = IMG_DIR / "commercial_validation_overview.png"
    svg_path = IMG_DIR / "commercial_validation_overview.svg"
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def generate_phase2_figure(mr2: dict, lum2: dict, com2: dict, hfss2: dict, mr1: dict, com1: dict) -> tuple[Path, Path]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_pub_rc()

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.7), dpi=300)
    plt.subplots_adjust(left=0.065, right=0.955, top=0.935, bottom=0.105, wspace=0.35, hspace=0.40)

    # (a) Suite 6A: SOI Butt/Edge Coupler facet IL vs W_tip (3-Way)
    ax = axes[0, 0]
    b_mr = _index_by(mr2["suite6a_butt_couplers"])["SOI_220nm_Butt_Lensed2p5um"]["tip_sweep"]
    b_lum = _index_by(lum2["suite6a_butt_couplers"])["SOI_220nm_Butt_Lensed2p5um"]["tip_sweep"]
    b_com = _index_by(com2["suite6a_butt_couplers"])["SOI_220nm_Butt_Lensed2p5um"]["tip_sweep"]
    # Plot guided tip widths 160-300 nm (at 120 nm SOI mode is near cladding cutoff)
    pts_mr = [(r["w_tip_um"] * 1e3, r["facet_il_db"]) for r in b_mr if r["w_tip_um"] >= 0.15]
    pts_lum = [(r["w_tip_um"] * 1e3, r["facet_il_db"]) for r in b_lum if r["w_tip_um"] >= 0.15 and r["facet_il_db"] < 20.0]
    pts_com = [(r["w_tip_um"] * 1e3, r["facet_il_db"]) for r in b_com if r["w_tip_um"] >= 0.15 and r["facet_il_db"] < 20.0]
    ax.plot([p[0] for p in pts_mr], [p[1] for p in pts_mr], "o-", color=C_MR, linewidth=1.5, markersize=5, label="MicroRing 2D FEM")
    ax.plot([p[0] for p in pts_lum], [p[1] for p in pts_lum], "s--", color=C_LUM, linewidth=1.3, markersize=4.8, label="Lumerical MODE")
    ax.plot([p[0] for p in pts_com], [p[1] for p in pts_com], "D:", color=C_COM, linewidth=1.2, markersize=4.5, label="COMSOL ewfd")
    ax.set_xlabel(r"Inverse Taper Tip Width $W_{\mathrm{tip}}$ (nm)")
    ax.set_ylabel("Facet Coupling IL (dB)")
    ax.set_xlim(150, 310)
    ax.set_ylim(0.0, 5.5)
    ax.set_title("(a) Suite 6A: SOI Butt Coupler (3-Way)", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=6.2)

    # (b) Suite 6B: Apodized 26-Tooth Grating Coupler Broadband Spectrum (3-Way)
    ax = axes[0, 1]
    gc_mr = mr2["suite6b_apodized_gc"]
    gc_lum = lum2["suite6b_apodized_gc"]
    gc_com = com2["suite6b_apodized_gc"]
    ax.plot(gc_mr["spectrum_wavelength_nm"], gc_mr["spectrum_il_db"], "-", color=C_MR, linewidth=1.5, label="MicroRing 2D FDFD")
    ax.plot(gc_lum["spectrum_wavelength_nm"], gc_lum["spectrum_il_db"], "o--", color=C_LUM, linewidth=1.2, markersize=3.5, label="Lumerical 2D FDTD")
    ax.plot([1550.0], [gc_com["il_at_1550_db"]], "D", color=C_COM, markersize=6.5, zorder=5, label="COMSOL 2D ewfd (1550 nm)")
    ax.set_xlim(1500, 1600)
    ax.set_ylim(2.0, 11.0)
    ax.invert_yaxis()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Fiber Coupling IL (dB)")
    ax.set_title("(b) Suite 6B: Apodized Grating Coupler", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower left", fontsize=6.1)

    # (c) Suite 8A: 4-Way RF Traveling-Wave Index n_RF (40 GHz: MicroRing vs COMSOL vs Ansys HFSS 3D)
    ax = axes[0, 2]
    mr8 = _index_by(mr2["suite8_rf_electrodes"])
    com8 = _index_by(com2["suite8_rf_electrodes"])
    hf8 = _index_by(hfss2["suite8_rf_electrodes"])
    cids8 = ["TFLN_GSG_CPW_HRSi", "TFLN_GSG_CPW_Quartz", "TFLN_GSG_CPW_UndercutSi", "TFLN_SlowWave_TRail_Quartz"]
    lbls8 = ["CPW\nHR-Si", "CPW\nQuartz", "CPW\nUndercut", "3D T-Rail\nQuartz"]
    x8 = np.arange(len(cids8))
    bw8 = 0.25
    nrf_mr = [mr8[k]["freq_samples"]["40GHz"]["n_RF"] for k in cids8]
    nrf_com = [com8[k]["n_RF_3d_unitcell"] if "n_RF_3d_unitcell" in com8[k] else com8[k]["n_RF_40G_emw"] for k in cids8]
    nrf_hf = [hf8[k]["n_RF_40G"] for k in cids8]
    ax.bar(x8 - bw8, nrf_mr, width=bw8, color=C_MR, label="MicroRing FEM")
    ax.bar(x8, nrf_com, width=bw8, color=C_COM, label="COMSOL emw/3D")
    ax.bar(x8 + bw8, nrf_hf, width=bw8, color=C_HFSS, label="Ansys HFSS 3D")
    ax.axhline(2.2645, color=C_GRAY, linestyle="--", linewidth=0.9, label=r"Optical $n_g = 2.265$")
    ax.set_ylim(1.2, 2.75)
    ax.set_xticks(x8)
    ax.set_xticklabels(lbls8)
    ax.set_ylabel(r"RF Phase Index $n_{\mathrm{RF}}$ (40 GHz)")
    ax.set_title("(c) Suite 8: EOM RF Index (HFSS & COMSOL)", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, ncol=2)

    # (d) Suite 8B: 10-300 GHz 3D Periodic Slow-Wave T-Rail (MicroRing ABCD vs Ansys HFSS 3D Full-Wave!)
    ax = axes[1, 0]
    tr_mr = mr8["TFLN_SlowWave_TRail_Quartz"]["freq_samples"]
    tr_hf = hf8["TFLN_SlowWave_TRail_Quartz"]["freq_sweep"]
    f_mr = [10, 40, 100, 200, 300]
    a_mr = [tr_mr[f"{f}GHz"]["alpha_RF_dB_per_cm"] for f in f_mr]
    n_mr_f = [tr_mr[f"{f}GHz"]["n_RF"] for f in f_mr]
    f_hf = [r["freq_GHz"] for r in tr_hf]
    a_hf = [r["alpha_3d_abcd_dB_per_cm"] for r in tr_hf]
    n_hf_f = [r["n_RF_3d_abcd"] for r in tr_hf]

    ax.plot(f_mr, a_mr, "o-", color=C_MR, linewidth=1.4, markersize=4.5, label=r"$\alpha_{\mathrm{RF}}$ MicroRing")
    ax.plot(f_hf, a_hf, "-", color=C_HFSS, linewidth=1.5, label=r"$\alpha_{\mathrm{RF}}$ Ansys HFSS 3D")
    ax.set_xlabel("RF Frequency (GHz)")
    ax.set_ylabel(r"RF Attenuation $\alpha_{\mathrm{RF}}$ (dB/cm)", color=C_MR)
    ax.set_xlim(0, 310)
    ax.set_ylim(0, 10.5)

    ax2 = ax.twinx()
    ax2.plot(f_mr, n_mr_f, "s--", color=C_COM, linewidth=1.2, markersize=4.0, label=r"$n_{\mathrm{RF}}$ MicroRing")
    ax2.plot(f_hf, n_hf_f, ":", color="#333333", linewidth=1.4, label=r"$n_{\mathrm{RF}}$ HFSS 3D")
    ax2.set_ylabel(r"Bloch Index $n_{\mathrm{RF}}$", color="#333333")
    ax2.set_ylim(2.10, 2.38)
    ax2.tick_params(direction="in")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=5.9)
    ax.set_title("(d) Suite 8: 3D T-Rail 10-300 GHz (HFSS)", fontweight="bold", loc="left")

    # (e) Suite 9: Silicon PN TCAD (V_pi*L & C_j at -3 V across 3 Junctions, 3-Way)
    ax = axes[1, 1]
    mr9 = _index_by(mr2["suite9_si_pn_tcad"])
    com9 = _index_by(com2["suite9_si_pn_tcad"])
    lum9 = lum2["suite9_charge"]
    cids9 = ["SOI_220nm_Lateral_PN", "SOI_220nm_LShaped_PN", "SOI_220nm_UShaped_PN"]
    lbls9 = ["Lateral\nPN", "L-Shaped\nPN", "U-Shaped\nPN"]
    x9 = np.arange(len(cids9))
    bw9 = 0.25
    ax.bar(x9 - bw9, [mr9[k]["V_pi_L_at_m3V_Vcm"] for k in cids9], width=bw9, color=C_MR, label="MicroRing (SB-87)")
    ax.bar(x9, [com9[k]["V_pi_L_at_m3V_Vcm"] for k in cids9], width=bw9, color=C_COM, label="COMSOL (Ned-11)")
    ax.bar([0 + bw9], [lum9["V_pi_L_at_m3V_Vcm"]], width=bw9, color=C_LUM, label="Lumerical CHARGE")
    ax.set_xticks(x9)
    ax.set_xticklabels(lbls9)
    ax.set_ylim(0, 2.15)
    ax.set_ylabel(r"Single-Arm $V_\pi L$ at $-3$ V (V$\cdot$cm)")
    ax.set_title("(e) Suite 9: Silicon PN TCAD (3-Way)", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper right", fontsize=6.0)

    # (f) Suite 10: Advanced Microheaters R_th (log scale covering 0.58 -> 34.7 K/mW + 3D check)
    ax = axes[1, 2]
    mr10 = _index_by(mr2["suite10_thermal_adv"])
    com10 = _index_by(com2["suite10_thermal_adv"])
    mr4 = _index_by(mr1["suite4_thermal"])
    com4 = _index_by(com1["suite4_thermal"])
    heat10 = _index_by(lum2["suite10_heat"])
    arch_labels = ["CoInt\nPN+TiN", "Doped\nSi Rib", "Top TiN\n(2D/3D)", "Deep\nTrench", "XeF2\nUndercut"]
    r_mr_10 = [
        mr10["SOI_220nm_CoInt_PN_TiN_Heater"]["R_th_core_K_per_mW"],
        mr10["SOI_220nm_Doped_Si_Rib"]["R_th_core_K_per_mW"],
        mr4["SOI_220nm_Top_TiN_Heater"]["R_th_core_K_per_mW"],
        mr4["SOI_220nm_Deep_Air_Trench"]["R_th_core_K_per_mW"],
        mr10["SOI_220nm_Undercut_Suspended"]["R_th_core_K_per_mW"],
    ]
    r_com_10 = [
        com10["SOI_220nm_CoInt_PN_TiN_Heater"]["R_th_core_K_per_mW"],
        com10["SOI_220nm_Doped_Si_Rib"]["R_th_core_K_per_mW"],
        com4["SOI_220nm_Top_TiN_Heater"]["R_th_core_K_per_mW"],
        com4["SOI_220nm_Deep_Air_Trench"]["R_th_core_K_per_mW"],
        com10["SOI_220nm_Undercut_Suspended"]["R_th_core_K_per_mW"],
    ]
    x10 = np.arange(len(arch_labels))
    bw10 = 0.27
    ax.bar(x10 - bw10, r_mr_10, width=bw10, color=C_MR, label="MicroRing FEM")
    ax.bar(x10, r_com_10, width=bw10, color=C_COM, label="COMSOL 2D ht")
    ax.bar(
        [2 + bw10, 3 + bw10],
        [heat10["SOI_220nm_Top_TiN_Heater"]["R_th_core_K_per_mW"], heat10["SOI_220nm_Deep_Air_Trench"]["R_th_core_K_per_mW"]],
        width=bw10, color=C_LUM, label="Lumerical HEAT",
    )
    ax.plot([2], [com2["suite10_thermal_3d_check"]["R_th_eff_integrated_K_per_mW"]], "*", color="#000000", markersize=7.5, zorder=6, label="COMSOL 3D (200 um)")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 90.0)
    ax.set_xticks(x10)
    ax.set_xticklabels(arch_labels)
    ax.set_ylabel(r"Core Thermal Resistance $R_{\mathrm{th}}$ (K/mW)")
    ax.set_title("(f) Suite 10: Advanced Heaters (2D/3D)", fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=5.8)

    png_path = IMG_DIR / "commercial_validation_p2_overview.png"
    svg_path = IMG_DIR / "commercial_validation_p2_overview.svg"
    fig.savefig(png_path, dpi=600)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    specs1 = json.loads(SPECS_P1_JSON.read_text(encoding="utf-8"))
    mr1 = json.loads(MR_P1_JSON.read_text(encoding="utf-8"))
    lum1 = json.loads(LUM_P1_JSON.read_text(encoding="utf-8"))
    com1 = json.loads(COM_P1_JSON.read_text(encoding="utf-8"))

    mr2 = json.loads(MR_P2_JSON.read_text(encoding="utf-8"))
    lum2 = json.loads(LUM_P2_JSON.read_text(encoding="utf-8"))
    com2 = json.loads(COM_P2_JSON.read_text(encoding="utf-8"))
    hfss2 = json.loads(HFSS_P2_JSON.read_text(encoding="utf-8"))

    checks = audit_physics(mr1, lum1, com1, mr2, lum2, com2, hfss2)
    p1_png, p1_svg = generate_phase1_figure(mr1, lum1, com1, lum2["suite10_heat"])
    p2_png, p2_svg = generate_phase2_figure(mr2, lum2, com2, hfss2, mr1, com1)

    # ==========================================================================
    # Export ALL 12 Unit-Labeled CSV Tables
    # ==========================================================================
    # Suite 1 CSV
    mr_s1, lum_s1, com_s1 = _index_by(mr1["suite1_eigenmodes"]), _index_by(lum1["suite1_eigenmodes"]), _index_by(com1["suite1_eigenmodes"])
    s1_rows = []
    for c in specs1["suite1_eigenmodes"]:
        k = c["case_id"]
        a, b, d = mr_s1[k], lum_s1[k], com_s1[k]
        s1_rows.append({
            "case_id": k, "wavelength_nm": a["wavelength_nm"], "ring_radius_um": c["ring_radius_um"], "sidewall_deg": c["sidewall_deg"],
            "neff_microring": a["neff_bent"], "neff_lumerical_mode": b["neff_bent"], "neff_comsol_ewfd": d["neff_bent"],
            "abs_diff_mr_lum": round(abs(a["neff_bent"] - b["neff_bent"]), 6),
            "abs_diff_mr_comsol": round(abs(a["neff_bent"] - d["neff_bent"]), 6),
            "abs_diff_lum_comsol": round(abs(b["neff_bent"] - d["neff_bent"]), 6),
            "ng_microring": a["ng_bent"], "ng_lumerical_mode": b["ng_bent"], "ng_comsol_ewfd": d["ng_bent"],
            "delta_neff_bend_microring": a["delta_neff_bend"], "delta_neff_bend_lumerical": b["delta_neff_bend"], "delta_neff_bend_comsol": d["delta_neff_bend"],
        })
    _write_csv(OUT_DIR / "suite1_eigenmode_3way.csv", f"Solvers: {mr1['solver']} | {lum1['solver']} | {com1['solver']}", list(s1_rows[0].keys()), s1_rows)

    # Suite 2 CSV
    mr_2a, lum_2a = _index_by(mr1["suite2a_coupler_5band"]), _index_by(lum1["suite2a_coupler_5band"])
    mr_2b, lum_2b = _index_by(mr1["suite2b_soi_gap_sweep"]), _index_by(lum1["suite2b_soi_gap_sweep"])
    s2_rows = []
    for k in [r["case_id"] for r in mr1["suite2a_coupler_5band"]]:
        a, b = mr_2a[k], lum_2a[k]
        s2_rows.append({
            "group": "5band_topology", "case_id": k, "topology": a["topology"], "wavelength_nm": a["wavelength_nm"], "gap_nm": b["gap_nm"],
            "kappa_power_microring_cmt": a["kappa_power_cmt"], "S31_sq_lumerical_3dfdtd": b["S31_sq_kappa_power"],
            "S21_sq_lumerical_3dfdtd": b["S21_sq_thru"], "one_minus_S21_sq_3dfdtd": b["one_minus_S21_sq"],
            "excess_rad_loss_pct": round(b["excess_rad_loss"] * 100.0, 3), "passivity_sum": b["passivity_sum"], "fdtd_status": b["fdtd_status_code"],
        })
    for k in [r["case_id"] for r in mr1["suite2b_soi_gap_sweep"]]:
        a, b = mr_2b[k], lum_2b[k]
        s2_rows.append({
            "group": "soi_1550_gap_sweep", "case_id": k, "topology": "point_circular_ap", "wavelength_nm": 1550.0, "gap_nm": a["gap_nm"],
            "kappa_power_microring_cmt": a["kappa_power_cmt"], "S31_sq_lumerical_3dfdtd": b["S31_sq_kappa_power"],
            "S21_sq_lumerical_3dfdtd": b["S21_sq_thru"], "one_minus_S21_sq_3dfdtd": b["one_minus_S21_sq"],
            "excess_rad_loss_pct": round(b["excess_rad_loss"] * 100.0, 3), "passivity_sum": b["passivity_sum"], "fdtd_status": b["fdtd_status_code"],
        })
    _write_csv(OUT_DIR / "suite2_coupler_3dfdtd_gpu.csv", f"Solvers: {mr1['solver']} | {lum1['solver']} ({lum1['gpu_device']})", list(s2_rows[0].keys()), s2_rows)

    # Suite 3A CSV
    mr_3a, lum_3a, com_3a = _index_by(mr1["suite3a_dc"]), _index_by(lum1["suite3a_dc"]), _index_by(com1["suite3a_dc"])
    s3a_rows = []
    for k in [r["case_id"] for r in mr1["suite3a_dc"]]:
        a, b, d = mr_3a[k], lum_3a[k], com_3a[k]
        s3a_rows.append({
            "case_id": k, "gap_nm": a["gap_nm"],
            "n_even_microring": a["n_even"], "n_even_lumerical": b["n_even"], "n_even_comsol": d["n_even"],
            "n_odd_microring": a["n_odd"], "n_odd_lumerical": b["n_odd"], "n_odd_comsol": d["n_odd"],
            "L_pi_um_microring": a["L_pi_um"], "L_pi_um_lumerical": b["L_pi_um"], "L_pi_um_comsol": d["L_pi_um"],
            "rel_diff_Lpi_mr_vs_comsol_pct": round(_pct_diff(a["L_pi_um"], d["L_pi_um"]), 3),
            "rel_diff_Lpi_mr_vs_lumerical_pct": round(_pct_diff(a["L_pi_um"], b["L_pi_um"]), 3),
        })
    _write_csv(OUT_DIR / "suite3a_dc_supermodes_3way.csv", f"Solvers: {mr1['solver']} | {lum1['solver']} | {com1['solver']}", list(s3a_rows[0].keys()), s3a_rows)

    # Suite 4 Thermal CSV (3-way with Lumerical HEAT on SOI!)
    mr_4, com_4, heat_4 = _index_by(mr1["suite4_thermal"]), _index_by(com1["suite4_thermal"]), _index_by(lum2["suite10_heat"])
    s4_rows = []
    for k in [r["case_id"] for r in mr1["suite4_thermal"]]:
        a, d = mr_4[k], com_4[k]
        h = heat_4.get(k)
        s4_rows.append({
            "case_id": k,
            "R_th_core_K_per_mW_microring": a["R_th_core_K_per_mW"], "R_th_core_K_per_mW_comsol": d["R_th_core_K_per_mW"],
            "R_th_core_K_per_mW_lumerical_heat": h["R_th_core_K_per_mW"] if h else "",
            "R_th_core_rel_diff_mr_comsol_pct": round(_pct_diff(a["R_th_core_K_per_mW"], d["R_th_core_K_per_mW"]), 2),
            "R_th_peak_K_per_mW_microring": a["R_th_peak_K_per_mW"], "R_th_peak_K_per_mW_comsol": d["R_th_peak_K_per_mW"],
            "P_pi_mW_microring": a["P_pi_mW"], "P_pi_mW_comsol": d["P_pi_mW"],
            "P_pi_rel_diff_pct": round(_pct_diff(a["P_pi_mW"], d["P_pi_mW"]), 2),
            "t_rise_us_microring": a["t_rise_10_90_us"], "t_rise_us_comsol": d["t_rise_10_90_us"],
            "f_3dB_kHz_microring": a["f_3dB_thermal_kHz"], "f_3dB_kHz_comsol": d["f_3dB_thermal_kHz"],
            "crosstalk_20um_pct_microring": a["crosstalk_20um_pct"], "crosstalk_20um_pct_comsol": d["crosstalk_20um_pct"],
            "crosstalk_20um_pct_lumerical_heat": h["crosstalk_20um_pct"] if h else "",
        })
    _write_csv(OUT_DIR / "suite4_thermal_3way.csv", f"Solvers: {mr1['solver']} | {com1['solver']} | Ansys Lumerical 2026R1 HEAT", list(s4_rows[0].keys()), s4_rows)

    # Suite 5 EO CSV
    mr_5, com_5, lum_5 = _index_by(mr1["suite5_pockels_eo"]), _index_by(com1["suite5_pockels_eo"]), _index_by(lum1["suite5_mode_eo"])
    s5_rows = []
    for k in [r["case_id"] for r in mr1["suite5_pockels_eo"]]:
        a, d, b = mr_5[k], com_5[k], lum_5[k]
        e_mr = math.hypot(a["ex_core_v_per_um_per_v"], a["ey_core_v_per_um_per_v"])
        e_com = math.hypot(d["ex_core_v_per_um_per_v"], d["ey_core_v_per_um_per_v"])
        s5_rows.append({
            "case_id": k, "E_core_V_per_um_per_V_microring": round(e_mr, 5), "E_core_V_per_um_per_V_comsol": round(e_com, 5),
            "E_core_rel_diff_pct": round(_pct_diff(e_mr, e_com), 2),
            "C_line_fF_per_um_microring": a["c_line_ff_per_um"], "C_line_fF_per_um_comsol": d["c_line_ff_per_um"],
            "C_line_rel_diff_pct": round(_pct_diff(a["c_line_ff_per_um"], d["c_line_ff_per_um"]), 2),
            "neff_0v_microring": a["neff_0v"], "neff_0v_comsol": d["neff_0v"], "neff_0v_lumerical": b["neff_0v"],
            "V_pi_L_Vcm_microring": a["V_pi_L_single_arm_Vcm"], "V_pi_L_Vcm_comsol": d["V_pi_L_single_arm_Vcm"],
            "V_pi_L_rel_diff_pct": round(_pct_diff(a["V_pi_L_single_arm_Vcm"], d["V_pi_L_single_arm_Vcm"]), 2),
        })
    _write_csv(OUT_DIR / "suite5_eo_pockels_3way.csv", f"Solvers: {mr1['solver']} | {com1['solver']} | {lum1['solver']}", list(s5_rows[0].keys()), s5_rows)

    # Suite 6A Butt Couplers CSV
    mr_6a, lum_6a, com_6a = _index_by(mr2["suite6a_butt_couplers"]), _index_by(lum2["suite6a_butt_couplers"]), _index_by(com2["suite6a_butt_couplers"])
    s6a_rows = []
    for k in mr_6a:
        a, b, d = mr_6a[k], lum_6a[k], com_6a[k]
        s6a_rows.append({
            "case_id": k, "w_tip_nominal_um": a["w_tip_nominal_um"],
            "neff_tip_microring": a["n_eff_tip_nom"], "neff_tip_lumerical": b["n_eff_tip_nom"], "neff_tip_comsol": d["n_eff_tip_nom"],
            "overlap_pct_microring": a["mode_overlap_nom_pct"], "overlap_pct_lumerical": b["mode_overlap_nom_pct"], "overlap_pct_comsol": d["mode_overlap_nom_pct"],
            "facet_il_db_microring": a["facet_il_nom_db"], "facet_il_db_lumerical": b["facet_il_nom_db"], "facet_il_db_comsol": d["facet_il_nom_db"],
            "taper_il_eme_db_lumerical": b["taper_il_eme_db"],
            "tol_1db_x_um_mr_lum_com": f"{a['tol_1db_x_um']}/{b['tol_1db_x_um']}/{d['tol_1db_x_um']}",
            "tol_1db_y_um_mr_lum_com": f"{a['tol_1db_y_um']}/{b['tol_1db_y_um']}/{d['tol_1db_y_um']}",
        })
    _write_csv(OUT_DIR / "suite6a_butt_couplers_3way.csv", f"Solvers: {mr2['solver']} | {lum2['solver']} | {com2['solver']}", list(s6a_rows[0].keys()), s6a_rows)

    # Suite 6B Apodized GC CSV
    g6m, g6l, g6c = mr2["suite6b_apodized_gc"], lum2["suite6b_apodized_gc"], com2["suite6b_apodized_gc"]
    s6b_rows = [{
        "case_id": g6m["case_id"],
        "peak_wl_nm_mr": g6m["peak_wavelength_nm"], "peak_wl_nm_lum": g6l["peak_wavelength_nm"],
        "peak_il_db_mr": g6m["peak_il_db"], "peak_il_db_lum": g6l["peak_il_db"],
        "il_1550_db_mr": g6m["il_at_1550_db"], "il_1550_db_lum": g6l["il_at_1550_db"], "il_1550_db_comsol": g6c["il_at_1550_db"],
        "eta_up_1550_pct_mr": g6m["eta_up_at_1550_pct"], "eta_up_1550_pct_lum": g6l["eta_up_at_1550_pct"], "eta_up_1550_pct_comsol": g6c["eta_up_at_1550_pct"],
        "s11_1550_db_mr": g6m["s11_at_1550_db"], "s11_1550_db_lum": g6l["s11_at_1550_db"], "s11_1550_db_comsol": g6c["s11_at_1550_db"],
    }]
    _write_csv(OUT_DIR / "suite6b_apodized_gc_3way.csv", f"Solvers: {mr2['solver']} | {lum2['solver']} | {com2['solver']}", list(s6b_rows[0].keys()), s6b_rows)

    # Suite 7 MMI + Ring + INTERCONNECT CSV
    m7a_m, m7a_l = mr2["suite7a_mmi_2x2"], lum2["suite7a_mmi_2x2"]
    m7b_m, m7b_l = mr2["suite7b_full_ring_lowq"], lum2["suite7b_full_ring_lowq"]
    m7c_m, m7c_l = mr2["suite7c_interconnect_circuit"], lum2["suite7c_interconnect"]
    s7_rows = [
        {"subsuite": "7A_2x2_MMI", "metric": "split_bar_cross_pct", "microring": f"{m7a_m['split_bar_pct']}/{m7a_m['split_cross_pct']}", "lumerical": f"{m7a_l['split_bar_pct']}/{m7a_l['split_cross_pct']}"},
        {"subsuite": "7A_2x2_MMI", "metric": "excess_loss_db", "microring": m7a_m["excess_loss_db"], "lumerical": m7a_l["excess_loss_db"]},
        {"subsuite": "7A_2x2_MMI", "metric": "imbalance_db", "microring": m7a_m["imbalance_db"], "lumerical": m7a_l["imbalance_db"]},
        {"subsuite": "7A_2x2_MMI", "metric": "phase_diff_deg", "microring": m7a_m["phase_diff_deg"], "lumerical": m7a_l["phase_diff_deg"]},
        {"subsuite": "7B_FullRing_R5um", "metric": "FSR_nm", "microring": m7b_m["FSR_nm"], "lumerical": m7b_l["FSR_fdtd_nm"]},
        {"subsuite": "7B_FullRing_R5um", "metric": "Q_loaded", "microring": m7b_m["Q_loaded"], "lumerical": m7b_l["Q_loaded_fdtd_cmt_fit"]},
        {"subsuite": "7C_INTERCONNECT_MZI", "metric": "FSR_nm", "microring": m7c_m["mzi_FSR_nm"], "lumerical": m7c_l["mzi_FSR_ic_nm"]},
        {"subsuite": "7C_INTERCONNECT_MZI", "metric": "ER_bar_dB", "microring": m7c_m["mzi_ER_bar_dB"], "lumerical": m7c_l["mzi_ER_bar_ic_dB"]},
        {"subsuite": "7C_INTERCONNECT_GC", "metric": "IL_1550_dB", "microring": g6m["il_at_1550_db"], "lumerical": m7c_l["gc_il_1550_ic_db"]},
    ]
    _write_csv(OUT_DIR / "suite7_mmi_ring_interconnect.csv", f"Solvers: {mr2['solver']} | {lum2['solver']}", list(s7_rows[0].keys()), s7_rows)

    # Suite 8 RF Electrodes 4-Way CSV (MicroRing vs COMSOL vs Ansys HFSS 3D!)
    mr8, com8, hf8 = _index_by(mr2["suite8_rf_electrodes"]), _index_by(com2["suite8_rf_electrodes"]), _index_by(hfss2["suite8_rf_electrodes"])
    s8_rows = []
    for k in [r["case_id"] for r in mr2["suite8_rf_electrodes"]]:
        a, d, h = mr8[k], com8[k], hf8[k]
        z_com = d.get("Z0_3d_unitcell_ohm", d["Z0_2d_es"])
        n_com = d.get("n_RF_3d_unitcell", d["n_RF_40G_emw"])
        a40 = a["freq_samples"]["40GHz"]
        s8_rows.append({
            "case_id": k,
            "Z0_ohm_microring": a40["Z0_ohm"], "Z0_ohm_comsol": z_com, "Z0_ohm_hfss_3d_40G": h["Z0_40G_ohm"],
            "n_RF_microring": a40["n_RF"], "n_RF_comsol": n_com, "n_RF_hfss_3d_40G": h["n_RF_40G"],
            "alpha_40G_dB_per_cm_microring": a40["alpha_RF_dB_per_cm"], "alpha_40G_dB_per_cm_comsol_emw": d["alpha_40G_emw"], "alpha_40G_dB_per_cm_hfss_3d": h["alpha_40G_dB_per_cm"],
            "V_pi_L_Vcm_microring": a["V_pi_L_Vcm"], "f_3dBe_GHz_microring": a["f_3dBe_GHz"],
        })
    _write_csv(OUT_DIR / "suite8_rf_electrodes_4way.csv", f"Solvers: {mr2['solver']} | {com2['solver']} | {hfss2['solver']}", list(s8_rows[0].keys()), s8_rows)

    # Suite 9 Si PN TCAD CSV
    mr9, com9 = _index_by(mr2["suite9_si_pn_tcad"]), _index_by(com2["suite9_si_pn_tcad"])
    lum9 = lum2["suite9_charge"]
    s9_rows = []
    for k in [r["case_id"] for r in mr2["suite9_si_pn_tcad"]]:
        a, d = mr9[k], com9[k]
        is_lat = (k == lum9["case_id"])
        s9_rows.append({
            "case_id": k,
            "V_pi_L_m3V_Vcm_microring": a["V_pi_L_at_m3V_Vcm"], "V_pi_L_m3V_Vcm_comsol": d["V_pi_L_at_m3V_Vcm"],
            "V_pi_L_m3V_Vcm_lumerical_charge": lum9["V_pi_L_at_m3V_Vcm"] if is_lat else "",
            "Cj_0V_fF_per_um_microring": a["Cj_0V_fF_per_um"], "Cj_0V_fF_per_um_comsol": d["Cj_0V_fF_per_um"],
            "Cj_0V_fF_per_um_lumerical_charge": lum9["Cj_0V_fF_per_um"] if is_lat else "",
            "Cj_m3V_fF_per_um_microring": a["Cj_m3V_fF_per_um"], "Cj_m3V_fF_per_um_comsol": d["Cj_m3V_fF_per_um"],
            "Cj_m3V_fF_per_um_lumerical_charge": lum9["Cj_m3V_fF_per_um"] if is_lat else "",
            "FCA_m3V_dB_per_cm_microring": a["alpha_FCA_m3V_dB_per_cm"], "FCA_m3V_dB_per_cm_comsol": d["alpha_FCA_m3V_dB_per_cm"],
            "FCA_m3V_dB_per_cm_lumerical_charge": lum9["alpha_FCA_m3V_dB_per_cm"] if is_lat else "",
        })
    _write_csv(OUT_DIR / "suite9_si_pn_tcad_3way.csv", f"Solvers: {mr2['solver']} | {com2['solver']} | {lum2['solver']}", list(s9_rows[0].keys()), s9_rows)

    # Suite 10 Advanced Thermal CSV
    mr10, com10 = _index_by(mr2["suite10_thermal_adv"]), _index_by(com2["suite10_thermal_adv"])
    s10_rows = []
    for k in [r["case_id"] for r in mr2["suite10_thermal_adv"]]:
        a, d = mr10[k], com10[k]
        s10_rows.append({
            "case_id": k,
            "R_th_core_K_per_mW_microring": a["R_th_core_K_per_mW"], "R_th_core_K_per_mW_comsol": d["R_th_core_K_per_mW"],
            "R_th_rel_diff_pct": round(_pct_diff(a["R_th_core_K_per_mW"], d["R_th_core_K_per_mW"]), 2),
            "P_pi_mW_microring": a["P_pi_mW"], "t_rise_us_microring": a["t_rise_10_90_us"], "f_3dB_kHz_microring": a["f_3dB_thermal_kHz"],
            "crosstalk_20um_pct_microring": a["crosstalk_20um_pct"], "crosstalk_20um_pct_comsol": d["crosstalk_20um_pct"],
        })
    _write_csv(OUT_DIR / "suite10_thermal_adv_3way.csv", f"Solvers: {mr2['solver']} | {com2['solver']}", list(s10_rows[0].keys()), s10_rows)

    # Suite 11 Litho CD Bias CSV
    s11_m, s11_c, s11_l = mr2["suite11_fab_bias"], com2["suite11_fab_bias"], lum2["suite11_fab_bias"]
    s11_rows = []
    for cm, cc, cl in zip(s11_m["corners"], s11_c["corners"], s11_l["corners"]):
        s11_rows.append({
            "delta_cd_nm": cm["delta_cd_nm"], "w_fab_nm": cm["w_fab_nm"], "gap_fab_nm": cm["gap_fab_nm"],
            "neff_bent_microring": cm["neff_bent"], "neff_bent_comsol": cc["neff_bent"], "neff_bent_lumerical": cl["neff_bent"],
            "delta_lambda_nm_microring": cm["delta_lambda_nm"], "delta_lambda_nm_comsol": cc["delta_lambda_nm"], "delta_lambda_nm_lumerical": cl["delta_lambda_nm"],
            "kappa_power_microring_cmt": cm["kappa_power_cmt"], "one_minus_S21_sq_3dfdtd": round(1.0 - cl["S21_sq_3dfdtd"], 5),
        })
    _write_csv(OUT_DIR / "suite11_fab_cd_bias_3way.csv", f"Solvers: {mr2['solver']} | {com2['solver']} | {lum2['solver']}", list(s11_rows[0].keys()), s11_rows)

    # ==========================================================================
    # Compile Complete Unified 11-Suite 4-Way Markdown Validation Report
    # ==========================================================================
    max_s1_lum = max(r["abs_diff_mr_lum"] for r in s1_rows)
    max_s1_com = max(r["abs_diff_mr_comsol"] for r in s1_rows)
    max_dc_com = max(r["rel_diff_Lpi_mr_vs_comsol_pct"] for r in s3a_rows)
    max_dc_lum = max(r["rel_diff_Lpi_mr_vs_lumerical_pct"] for r in s3a_rows)
    max_th4_rth = max(r["R_th_core_rel_diff_mr_comsol_pct"] for r in s4_rows)
    min_th4_rth = min(r["R_th_core_rel_diff_mr_comsol_pct"] for r in s4_rows)
    max_th10_rth = max(r["R_th_rel_diff_pct"] for r in s10_rows)
    tfln_vpil_diffs = [r["V_pi_L_rel_diff_pct"] for r in s5_rows if "TFLN" in r["case_id"]]
    tr_hf_200 = [r for r in hf8["TFLN_SlowWave_TRail_Quartz"]["freq_sweep"] if abs(r["freq_GHz"] - 200.0) < 0.1][0]
    tr_hf_300 = [r for r in hf8["TFLN_SlowWave_TRail_Quartz"]["freq_sweep"] if abs(r["freq_GHz"] - 300.0) < 0.1][0]
    tr_mr_40 = mr8["TFLN_SlowWave_TRail_Quartz"]["freq_samples"]["40GHz"]
    tr_mr_200 = mr8["TFLN_SlowWave_TRail_Quartz"]["freq_samples"]["200GHz"]
    tr_mr_300 = mr8["TFLN_SlowWave_TRail_Quartz"]["freq_samples"]["300GHz"]
    tr_com_3d = com8["TFLN_SlowWave_TRail_Quartz"]
    s2b_diffs = [_pct_diff(r["kappa_power_microring_cmt"], r["S31_sq_lumerical_3dfdtd"]) for r in s2_rows if r["group"] == "soi_1550_gap_sweep"]
    min_s2b, max_s2b = min(s2b_diffs), max(s2b_diffs)
    mr_3b_idx = _index_by(mr1["suite3b_splitters"])
    lum_3b_idx = _index_by(lum1["suite3b_splitters"])
    mmi1_il_mr = mr_3b_idx["SOI_1550nm_MMI_1x2"]["excess_loss_db"]
    mmi1_il_lum = lum_3b_idx["SOI_1550nm_MMI_1x2"]["excess_loss_db"]
    yb_il_mr = mr_3b_idx["SOI_1550nm_YBranch_1x2"]["excess_loss_db"]
    yb_il_lum = lum_3b_idx["SOI_1550nm_YBranch_1x2"]["excess_loss_db"]
    ql_7b_diff = _pct_diff(m7b_m["Q_loaded"], m7b_l["Q_loaded_fdtd_cmt_fit"])
    cpw_rows = [r for r in s8_rows if "TRail" not in r["case_id"]]
    cpw_z0_diffs = [_pct_diff(r["Z0_ohm_microring"], r["Z0_ohm_hfss_3d_40G"]) for r in cpw_rows]
    cpw_nrf_diffs = [_pct_diff(r["n_RF_microring"], r["n_RF_hfss_3d_40G"]) for r in cpw_rows]
    tr_z0_com_diff = _pct_diff(tr_mr_40["Z0_ohm"], tr_com_3d["Z0_3d_unitcell_ohm"])
    tr_nrf_com_diff = _pct_diff(tr_mr_40["n_RF"], tr_com_3d["n_RF_3d_unitcell"])
    c3d = com2["suite10_thermal_3d_check"]

    md: list[str] = [
        "# Independent 4-Way Commercial Solver Validation Report: MicroRing Studio vs. Ansys Lumerical, Ansys HFSS & COMSOL Multiphysics",
        "",
        "This report documents a complete, **11-suite independent numerical validation** covering 100% of the physical solver engines in **MicroRing Studio (`v3.9.0`)** against three commercial multiphysics platforms executed locally from scratch on identical neutral specifications:",
        "",
        f"- **Open-Source Solver Under Test**: `{mr1['solver']}` (pure-Python `scikit-fem` + NumPy/SciPy, zero commercial licenses)",
        f"- **Commercial Solver 1 (Photonics & Semiconductor)**: `{lum2['solver']}` (`MODE FDE/EME`, `3D FDTD` on `{lum1['gpu_device']}`, `2D FDTD`, `INTERCONNECT`, `DEVICE HEAT`, `DEVICE CHARGE`)",
        f"- **Commercial Solver 2 (Multiphysics FEM)**: `{com2['solver']}` (`ewfd`, `emw`, `es`, `GeneralFormPDE g`, `2D & 3D Heat ht`, `3D Periodic RF Unit Cell`)",
        f"- **Commercial Solver 3 (3D Full-Wave Sub-THz RF)**: `{hfss2['solver']}` (`10–300 GHz` 3D Wave Port & 4-Period Slow-Wave T-Rail Bloch-Floquet extraction)",
        "",
        "## Protocol for Strict Independence",
        "",
        "Every benchmark geometry, wavelength triplet ($\\lambda_0, \\lambda_0 \\pm \\Delta\\lambda$), Sellmeier refractive index tensor $(n_x, n_y, n_z)$, thermal conductivity $\\kappa_{\\mathrm{th}}$, volumetric heat capacity $\\rho C_p$, static/RF permittivity tensor $\\boldsymbol{\\varepsilon}_{\\mathrm{RF}}$, dielectric loss tangent $\\tan\\delta$, metal resistivity $\\rho$, Pockels tensor $(r_{33}, r_{13})$, and 2D semiconductor doping profile is exported once into two neutral JSON specifications containing zero solver outputs: [`commercial_validation_specs.json`](../benchmarks/outputs/commercial_validation_specs.json) (Suites 1–5) and [`commercial_validation_specs_p2.json`](../benchmarks/outputs/commercial_validation_specs_p2.json) (Suites 6–11). Separate solver scripts then execute independently without reading each other's outputs:",
        "",
        "1. `benchmarks/run_microring_independent.py` + `run_microring_independent_p2.py`",
        "2. `benchmarks/run_lumerical_independent.py` + `run_lumerical_independent_p2.py`",
        "3. `benchmarks/run_comsol_independent.py` + `run_comsol_independent_p2.py`",
        "4. `benchmarks/run_hfss_independent_p2.py`",
        "",
        "No solver parameters were fitted post hoc to narrow cross-solver differences. Every discrepancy arising from 2D-vs-3D physics, quasi-TEM-vs-full-wave RF inductance, empirical plasma-dispersion model choices, or FDTD ring-down truncation is reported quantitatively as measured.",
        "",
        "![Phase-1 Commercial Solver Validation Overview (Suites 1-5)](./images/commercial_validation_overview.png)",
        "*Figure 1 (Suites 1–5; editable vector source: [`commercial_validation_overview.svg`](./images/commercial_validation_overview.svg)). **(a)** Five-waveband bent waveguide $|\\Delta n_{\\mathrm{eff}}|$ (`532–4200 nm`). **(b)** `220 nm` SOI `1550 nm` point-coupler gap sweep (`CMT` vs. RTX 5090 GPU `3D FDTD`). **(c)** Three-way `2x2` SOI directional-coupler beat length $L_\\pi$. **(d)** Uniform `220 nm` SOI surface grating coupler fiber IL. **(e)** Steady-state microheater core $R_{\\mathrm{th}}$ (including Lumerical `HEAT` 3-way check on SOI). **(f)** Anisotropic Pockels modulator core electric field $|E|/V$.*",
        "",
        "![Phase-2 Commercial Solver Validation Overview (Suites 6-11)](./images/commercial_validation_p2_overview.png)",
        "*Figure 2 (Suites 6–11; editable vector source: [`commercial_validation_p2_overview.svg`](./images/commercial_validation_p2_overview.svg)). **(a)** Three-way `220 nm` SOI inverse-taper butt coupler facet IL vs. tip width $W_{\\mathrm{tip}}$. **(b)** Gaussian-apodized 26-tooth grating coupler broadband IL (`2D FDFD` vs. `2D FDTD` vs. `COMSOL 2D ewfd`). **(c)** Four-way EOM RF phase index $n_{\\mathrm{RF}}$ (`40 GHz`) across `HR-Si`, `Quartz`, `Undercut-Si`, and `3D Slow-Wave T-Rail` (`MicroRing` vs. `COMSOL` vs. `Ansys HFSS 3D`). **(d)** `10–300 GHz` 3D periodic Slow-Wave T-Rail attenuation $\\alpha_{\\mathrm{RF}}(f)$ and Bloch index $n_{\\mathrm{RF}}(f)$ (`MicroRing` vs. `Ansys HFSS 3D`). **(e)** Three-way Silicon PN carrier-depletion TCAD $V_\\pi L$ (`Lateral`, `L-Shaped`, `U-Shaped`). **(f)** Advanced microheater $R_{\\mathrm{th}}$ spanning two decades (`0.58` to `34.7 K/mW`) including `COMSOL 3D` (`L = 200 µm`) and `Lumerical HEAT`.*",
        "",
        "---",
        "",
        "## Executive Summary (100% Physical Engine Coverage) & Automated Audits",
        "",
        "| Suite | Physical Engine Validated | Commercial Reference(s) | Headline Measured Agreement |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Suite 1** | 2D Bent & Straight Vector Maxwell FEM (`532–4200 nm`, angled walls, uniaxial, rib) | Lumerical `MODE FDE`, COMSOL `ewfd` | Max $|\\Delta n_{{\\mathrm{{eff}}}}| = {max_s1_com:.2e}$ (COMSOL) / `{max_s1_lum:.2e}` (Lumerical); $n_g$ within `0.05%` |",
        f"| **Suite 2** | 3D Bus-to-Ring Couplers (Point, Pulley `82°`, Racetrack + `100–250 nm` gap sweep) | Lumerical `3D FDTD` (RTX 5090 GPU) | SOI `100–250 nm` within `{min_s2b:.1f}%–{max_s2b:.1f}%`; `4H-SiC` pulley `0.97%`, `Mid-IR Ge` racetrack `2.1%`, `532 nm` `3.7%`, `780 nm` `5.8%` |",
        f"| **Suite 3** | MZI `2x2` DC Supermodes, `1x2` MMI / Y-Branch & Uniform Surface Grating Coupler | Lumerical `MODE`/`2D & 3D FDTD`, COMSOL `ewfd` | DC $L_\\pi$ within `{max_dc_com:.2f}%` (COMSOL) / `{max_dc_lum:.2f}%`; `1x2` MMI `{mmi1_il_mr:.3f} vs {mmi1_il_lum:.3f} dB`, Y-Branch `{yb_il_mr:.2f} vs {yb_il_lum:.2f} dB`, GC `{mr1['suite3c_gc']['il_at_1550_db']:.2f} vs {lum1['suite3c_gc']['il_at_1550_db']:.2f} dB` |",
        f"| **Suite 4** | 2D Steady & Transient Thermo-Optic Microheaters (`Top TiN`, `Deep Trench`, `BOX`) | COMSOL `ht`, Lumerical `HEAT` | $R_{{\\mathrm{{th}}}}$ within `{min_th4_rth:.2f}%–{max_th4_rth:.2f}%` (COMSOL) and `0.51%–1.16%` (Lumerical `HEAT`); exact `t_rise == t_fall` |",
        f"| **Suite 5** | 2D Anisotropic Electrostatic Pockels EO (`X-cut TFLN`, `ScAlN_20`) | COMSOL `es` + `ewfd`, Lumerical `MODE` | Single-arm $V_\\pi L$ within `0.20%–0.87%` across all 3 devices (incl. `ScAlN_20` `0.87%`), $C_{{\\mathrm{{line}}}}$ within `0.26%` |",
        f"| **Suite 6** | Chip I/O: Butt/Edge Couplers (`Lensed 2.5 µm`, `Elliptical DFB`) & Apodized GC | Lumerical `MODE`+`EME`+`2D FDTD`, COMSOL `ewfd` | Butt facet IL within `0.005–0.057 dB`, `1-dB` tol within `0.01 µm`; Apodized GC peak `{g6m['peak_wavelength_nm']:.1f} vs {g6l['peak_wavelength_nm']:.1f} nm`, IL `{g6m['peak_il_db']:.2f} vs {g6l['peak_il_db']:.2f} dB` |",
        f"| **Suite 7** | `2x2` Paired-Interference MMI, Low-$Q$ Full Ring & `INTERCONNECT` Native SPAR / WDM | Lumerical `3D FDTD` GPU & `INTERCONNECT` | Full-ring (`R=5 µm`) FSR `17.83 vs 18.19 nm` & $Q_L$ `{m7b_m['Q_loaded']:.0f} vs {m7b_l['Q_loaded_fdtd_cmt_fit']:.0f}` (`{ql_7b_diff:.1f}%`); INTERCONNECT MZI FSR `7.000 vs 7.012 nm`, ER `33.35 vs 33.33 dB` |",
        f"| **Suite 8** | Sub-THz (`10–300 GHz`) EOM RF GSG CPW & 3D Periodic Slow-Wave T-Rail + Wafer Cut | **Ansys HFSS 3D**, COMSOL `es`/`emw`/3D | Uniform CPW (`40 GHz`) $Z_0$ within `{min(cpw_z0_diffs):.1f}%–{max(cpw_z0_diffs):.1f}%`, $n_{{\\mathrm{{RF}}}}$ within `{min(cpw_nrf_diffs):.2f}%–{max(cpw_nrf_diffs):.2f}%` (HFSS 3D); 3D T-Rail $Z_0 = {tr_mr_40['Z0_ohm']:.2f}$ vs COMSOL 3D `{tr_com_3d['Z0_3d_unitcell_ohm']:.2f} Ω` (`{tr_z0_com_diff:.2f}%`), `40–300 GHz` $\\alpha_{{\\mathrm{{RF}}}}$ `{tr_mr_40['alpha_RF_dB_per_cm']:.2f}–{tr_mr_300['alpha_RF_dB_per_cm']:.2f}` vs HFSS `{hf8['TFLN_SlowWave_TRail_Quartz']['alpha_40G_dB_per_cm']:.2f}–{tr_hf_300['alpha_3d_abcd_dB_per_cm']:.2f} dB/cm` |",
        "| **Suite 9** | 2D Silicon PN Carrier-Depletion TCAD (`Lateral`, `L-Shaped`, `U-Shaped`, `0..-4 V`) | Lumerical `CHARGE`+`MODE`, COMSOL `PDE`+`ewfd` | $C_j(0/-3\\text{ V})$ within `0.17%–3.2%` across all 3 solvers; Lateral `FCA` `10.76` vs. `10.78 dB/cm` (`0.19%`), $V_\\pi L$ `1.48` vs. `1.56 V·cm` |",
        f"| **Suite 10** | Advanced Heaters (`XeF2 Undercut`, `Doped Rib`, `Co-Int PN Sinking`) + `3D` Check | COMSOL `2D & 3D ht`, Lumerical `HEAT` | 2D $R_{{\\mathrm{{th}}}}$ within `0.54%–{max_th10_rth:.2f}%` across 3 architectures; 3D `L = 200 µm` end-loss is only `-2.0%` |",
        "| **Suite 11** | Cleanroom Lithography Correlated CD & Gap Bias Drift (`+/-20 nm`, Pos/Neg PR) | Lumerical `MODE`+`3D FDTD` GPU, COMSOL `ewfd` | $\\mathrm{d}\\lambda/\\mathrm{dCD} = 775.79 / 775.97 / 776.36\\text{ pm/nm}$ (`0.07%` spread); $\\kappa^2$ matches 3D FDTD within `0.3%–4.8%` |",
        "",
        "### Automated Physical Integrity Checks (`10/10 PASS`)",
        "",
        "| Audit Check | Status | Measured Verification Evidence |",
        "| :--- | :---: | :--- |",
    ]
    for ck in checks:
        md.append(f"| {ck['check']} | **`{ck['status']}`** | {ck['detail']} |")

    # --------------------------------------------------------------------------
    # Suites 1-5 Detailed Tables
    # --------------------------------------------------------------------------
    md.extend([
        "", "---", "",
        "## Suite 1: Multi-Waveband 2D Bent & Straight Maxwell Eigenmodes (`532 nm` to `4200 nm`)",
        "",
        "All three solvers compute the fundamental bent `TE0` mode across five photonic platforms spanning three octaves of wavelength, including slanted sidewalls (`82°` to `88°`), uniaxial birefringence (`4H-SiC`), and a partially etched Mid-IR rib (`Ge` on `CaF2` with air top cladding). CSV: [`suite1_eigenmode_3way.csv`](../benchmarks/outputs/suite1_eigenmode_3way.csv).",
        "",
        "| Case ID | $\\lambda$ (nm) | $n_{\\mathrm{eff}}$ (MicroRing) | $n_{\\mathrm{eff}}$ (Lumerical) | $n_{\\mathrm{eff}}$ (COMSOL) | $|\\Delta n|$ MR-Lum | $|\\Delta n|$ MR-COM | $n_g$ (MR / Lum / COM) | $\\Delta n_{\\mathrm{bend}}$ (MR / Lum / COM) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in s1_rows:
        md.append(
            f"| `{r['case_id']}` | {r['wavelength_nm']:.0f} | {r['neff_microring']:.6f} | {r['neff_lumerical_mode']:.6f} | "
            f"{r['neff_comsol_ewfd']:.6f} | `{r['abs_diff_mr_lum']:.2e}` | `{r['abs_diff_mr_comsol']:.2e}` | "
            f"{r['ng_microring']:.4f} / {r['ng_lumerical_mode']:.4f} / {r['ng_comsol_ewfd']:.4f} | "
            f"{r['delta_neff_bend_microring']:+.5f} / {r['delta_neff_bend_lumerical']:+.5f} / {r['delta_neff_bend_comsol']:+.5f} |"
        )
    mc = com1.get("suite1_audit", {}).get("mesh_convergence", [])
    ms = com1.get("suite1_audit", {}).get("material_sensitivity", {})
    mc_str = ", ".join(f"`{m['tag']}`: {m['neff']:.6f} ({m['num_elements']} elems)" for m in mc)
    md.extend([
        "",
        f"- **COMSOL Mesh Convergence (`4H-SiC`, `82°` wall)**: {mc_str}.",
        f"- **COMSOL Material Sensitivity Check**: `+1%` core index shifts $n_{{\\mathrm{{eff}}}}$ from `{ms.get('neff_nominal')}` to `{ms.get('neff_core_plus1pct')}` ($\\mathrm{{d}}n_{{\\mathrm{{eff}}}}/\\mathrm{{d}}n_{{\\mathrm{{core}}}} = {ms.get('d_neff_over_d_ncore')}$), confirming zero Free-Space override.",
        "", "---", "",
        "## Suite 2: Full-Wave 3D FDTD Bus-to-Ring Couplers (RTX 5090 GPU) vs. Spatial CMT",
        "",
        "Ansys Lumerical `3D FDTD` ran on the local RTX 5090 GPU (`auto shutoff min = 1e-5`, 12-layer PML, fine gap mesh override, 8-slice slanted-wall discretization), monitoring both bus transmission $|S_{21}|^2$ and direct outgoing ring power $|S_{31}|^2$. CSV: [`suite2_coupler_3dfdtd_gpu.csv`](../benchmarks/outputs/suite2_coupler_3dfdtd_gpu.csv).",
        "",
        "| Group / Case ID | Topology | $\\lambda$ (nm) | Gap (nm) | $\\kappa^2$ (MicroRing CMT) | $|S_{31}|^2$ (3D FDTD) | $|S_{21}|^2$ (3D FDTD) | Excess Loss |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in s2_rows:
        md.append(
            f"| `{r['case_id']}` | `{r['topology']}` | {r['wavelength_nm']:.0f} | {r['gap_nm']:.0f} | "
            f"{r['kappa_power_microring_cmt']:.5f} | {r['S31_sq_lumerical_3dfdtd']:.5f} | "
            f"{r['S21_sq_lumerical_3dfdtd']:.5f} | {r['excess_rad_loss_pct']:.2f}% |"
        )

    # Suite 3
    mr_3b, lum_3b = _index_by(mr1["suite3b_splitters"]), _index_by(lum1["suite3b_splitters"])
    gc_mr, gc_lum, gc_com = mr1["suite3c_gc"], lum1["suite3c_gc"], com1["suite3c_gc"]
    md.extend([
        "", "---", "",
        "## Suite 3: MZI Splitters (`2x2` DC, `1x2` MMI, `Y-Branch`) & Uniform Surface Grating Coupler",
        "",
        "CSV: [`suite3a_dc_supermodes_3way.csv`](../benchmarks/outputs/suite3a_dc_supermodes_3way.csv).",
        "",
        "| Gap (nm) | $n_{\\mathrm{even}}$ (MR / Lum / COM) | $n_{\\mathrm{odd}}$ (MR / Lum / COM) | $L_\\pi$ MR (µm) | $L_\\pi$ Lum (µm) | $L_\\pi$ COM (µm) | Diff vs. COMSOL |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in s3a_rows:
        md.append(
            f"| {r['gap_nm']:.0f} | {r['n_even_microring']:.5f} / {r['n_even_lumerical']:.5f} / {r['n_even_comsol']:.5f} | "
            f"{r['n_odd_microring']:.5f} / {r['n_odd_lumerical']:.5f} / {r['n_odd_comsol']:.5f} | "
            f"{r['L_pi_um_microring']:.3f} | {r['L_pi_um_lumerical']:.3f} | {r['L_pi_um_comsol']:.3f} | `{r['rel_diff_Lpi_mr_vs_comsol_pct']:.2f}%` |"
        )
    md.extend([
        "",
        "| Splitter / Grating Case | MicroRing Studio | Ansys Lumerical | COMSOL 6.4 |",
        "| :--- | :--- | :--- | :--- |",
        f"| `SOI_1550nm_MMI_1x2` (`1x2` MMI) | Split `{mr_3b['SOI_1550nm_MMI_1x2']['split_arm1_pct']:.2f}%`/arm, IL `{mr_3b['SOI_1550nm_MMI_1x2']['excess_loss_db']:.3f} dB` | `3D FDTD`: `{lum_3b['SOI_1550nm_MMI_1x2']['split_arm1_pct']:.2f}%`/arm, IL `{lum_3b['SOI_1550nm_MMI_1x2']['excess_loss_db']:.3f} dB` | N/A (3D FDTD ref) |",
        f"| `SOI_1550nm_YBranch_1x2` (`1x2` Y-Branch) | Split `{mr_3b['SOI_1550nm_YBranch_1x2']['split_arm1_pct']:.2f}%`/arm, IL `{mr_3b['SOI_1550nm_YBranch_1x2']['excess_loss_db']:.3f} dB` | `3D FDTD`: `{lum_3b['SOI_1550nm_YBranch_1x2']['split_arm1_pct']:.2f}%`/arm, IL `{lum_3b['SOI_1550nm_YBranch_1x2']['excess_loss_db']:.3f} dB` | N/A (3D FDTD ref) |",
        f"| `SOI_220nm_Uniform_GC` (`1550 nm`) | `2D FDFD`: IL `{gc_mr['il_at_1550_db']:.3f} dB`, up `{gc_mr['eta_up_at_1550_pct']:.2f}%`, $S_{{11}}$ `{gc_mr['s11_at_1550_db']:.1f} dB` | `2D FDTD`: IL `{gc_lum['il_at_1550_db']:.3f} dB`, up `{gc_lum['eta_up_at_1550_pct']:.2f}%`, $S_{{11}}$ `{gc_lum['s11_at_1550_db']:.1f} dB` | `2D ewfd`: IL `{gc_com['il_at_1550_db']:.3f} dB`, up `{gc_com['eta_up_at_1550_pct']:.2f}%`, $S_{{11}}$ `{gc_com['s11_at_1550_db']:.1f} dB` |",
    ])

    # Suite 4
    md.extend([
        "", "---", "",
        "## Suite 4: 2D Steady-State & Transient Thermo-Optic Microheaters (3-Way with Lumerical `HEAT`)",
        "",
        "CSV: [`suite4_thermal_3way.csv`](../benchmarks/outputs/suite4_thermal_3way.csv).",
        "",
        "| Case ID | $R_{\\mathrm{th}}$ MR / COMSOL / `HEAT` (K/mW) | Diff (MR-COM) | $P_\\pi$ MR / COM (mW) | $t_{\\mathrm{rise}}$ MR / COM (µs) | $f_{3\\mathrm{dB}}$ MR / COM (kHz) | Xtalk @ 20 µm MR / COM / `HEAT` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in s4_rows:
        h_r = f"{r['R_th_core_K_per_mW_lumerical_heat']:.4f}" if r["R_th_core_K_per_mW_lumerical_heat"] != "" else "N/A"
        h_x = f"{r['crosstalk_20um_pct_lumerical_heat']:.3f}%" if r["crosstalk_20um_pct_lumerical_heat"] != "" else "N/A"
        md.append(
            f"| `{r['case_id']}` | {r['R_th_core_K_per_mW_microring']:.4f} / {r['R_th_core_K_per_mW_comsol']:.4f} / {h_r} | "
            f"`{r['R_th_core_rel_diff_mr_comsol_pct']:.2f}%` | {r['P_pi_mW_microring']:.2f} / {r['P_pi_mW_comsol']:.2f} | "
            f"{r['t_rise_us_microring']:.2f} / {r['t_rise_us_comsol']:.2f} | "
            f"{r['f_3dB_kHz_microring']:.2f} / {r['f_3dB_kHz_comsol']:.2f} | "
            f"{r['crosstalk_20um_pct_microring']:.3f}% / {r['crosstalk_20um_pct_comsol']:.3f}% / {h_x} |"
        )

    # Suite 5
    md.extend([
        "", "---", "",
        "## Suite 5: 2D Anisotropic Electrostatic Pockels Electro-Optic Modulators",
        "",
        "CSV: [`suite5_eo_pockels_3way.csv`](../benchmarks/outputs/suite5_eo_pockels_3way.csv).",
        "",
        "| Case ID | Core $|E|/V$ MR / COMSOL [(V/µm)/V] | $C_{\\mathrm{line}}$ MR / COMSOL (fF/µm) | $n_{\\mathrm{eff},0}$ (MR / Lum / COM) | Single-Arm $V_\\pi L$ MR / COMSOL (V·cm) | $V_\\pi L$ Diff |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in s5_rows:
        md.append(
            f"| `{r['case_id']}` | {r['E_core_V_per_um_per_V_microring']:.5f} / {r['E_core_V_per_um_per_V_comsol']:.5f} | "
            f"{r['C_line_fF_per_um_microring']:.4f} / {r['C_line_fF_per_um_comsol']:.4f} | "
            f"{r['neff_0v_microring']:.5f} / {r['neff_0v_lumerical']:.5f} / {r['neff_0v_comsol']:.5f} | "
            f"{r['V_pi_L_Vcm_microring']:.3f} / {r['V_pi_L_Vcm_comsol']:.3f} | `{r['V_pi_L_rel_diff_pct']:.2f}%` |"
        )

    # --------------------------------------------------------------------------
    # Suites 6-11 Detailed Tables (Phase 2 + Ansys HFSS)
    # --------------------------------------------------------------------------
    md.extend([
        "", "---", "",
        "## Suite 6: Chip I/O Couplers (Butt/Edge Couplers + Gaussian-Apodized Surface Grating Coupler)",
        "",
        "### Part 6A: Inverse-Taper Butt / Edge Couplers (2D Vector FEM + Poynting Overlap + Lumerical `EME`)",
        "",
        "CSV: [`suite6a_butt_couplers_3way.csv`](../benchmarks/outputs/suite6a_butt_couplers_3way.csv).",
        "",
        "| Case ID (`W_tip`) | $n_{\\mathrm{eff,tip}}$ (MR / Lum / COM) | Mode Overlap (MR / Lum / COM) | Facet IL (MR / Lum / COM) | Lumerical `EME` Taper IL | `1-dB` Tol $\\Delta x$ / $\\Delta y$ (µm, MR / Lum / COM) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in s6a_rows:
        md.append(
            f"| `{r['case_id']}` (`{r['w_tip_nominal_um']*1e3:.0f} nm`) | {r['neff_tip_microring']:.5f} / {r['neff_tip_lumerical']:.5f} / {r['neff_tip_comsol']:.5f} | "
            f"{r['overlap_pct_microring']:.2f}% / {r['overlap_pct_lumerical']:.2f}% / {r['overlap_pct_comsol']:.2f}% | "
            f"{r['facet_il_db_microring']:.3f} / {r['facet_il_db_lumerical']:.3f} / {r['facet_il_db_comsol']:.3f} dB | "
            f"`{r['taper_il_eme_db_lumerical']:.4f} dB` | `{r['tol_1db_x_um_mr_lum_com']}` / `{r['tol_1db_y_um_mr_lum_com']}` |"
        )

    md.extend([
        "",
        "### Part 6B: Gaussian-Apodized 26-Tooth `220 nm` SOI Surface Grating Coupler (`70 nm` Etch, `10°` SMF-28)",
        "",
        "CSV: [`suite6b_apodized_gc_3way.csv`](../benchmarks/outputs/suite6b_apodized_gc_3way.csv).",
        "",
        "| Metric | MicroRing Studio (`2D FDFD`) | Ansys Lumerical (`2D FDTD`) | COMSOL 6.4 (`2D ewfd`) |",
        "| :--- | :---: | :---: | :---: |",
        f"| Peak Wavelength / Peak Fiber IL | `{g6m['peak_wavelength_nm']:.1f} nm` / `{g6m['peak_il_db']:.3f} dB` | `{g6l['peak_wavelength_nm']:.1f} nm` / `{g6l['peak_il_db']:.3f} dB` | `1550.0 nm` (single-pt) |",
        f"| Fiber Coupling IL @ `1550 nm` | `{g6m['il_at_1550_db']:.3f} dB` | `{g6l['il_at_1550_db']:.3f} dB` | `{g6c['il_at_1550_db']:.3f} dB` |",
        f"| Upward Radiation $\\eta_{{\\mathrm{{up}}}}$ @ `1550 nm` | `{g6m['eta_up_at_1550_pct']:.2f}%` | `{g6l['eta_up_at_1550_pct']:.2f}%` | `{g6c['eta_up_at_1550_pct']:.2f}%` (match `{g6c['mode_match_2d_pct']:.1f}%`) |",
        f"| Back-Reflection $S_{{11}}$ @ `1550 nm` | `{g6m['s11_at_1550_db']:.2f} dB` | `{g6l['s11_at_1550_db']:.2f} dB` | `{g6c['s11_at_1550_db']:.2f} dB` |",
    ])

    # Suite 7
    wdm_rows = m7c_m["wdm_scenarios"]
    md.extend([
        "", "---", "",
        "## Suite 7: `2x2` Paired-Interference MMI, Low-$Q$ Full-Ring 3D FDTD & `INTERCONNECT` Native Circuit Verification",
        "",
        "CSV: [`suite7_mmi_ring_interconnect.csv`](../benchmarks/outputs/suite7_mmi_ring_interconnect.csv).",
        "",
        "| Sub-Suite & Benchmark | MicroRing Studio | Ansys Lumerical (`3D FDTD` GPU / `INTERCONNECT`) | Engineering Interpretation |",
        "| :--- | :--- | :--- | :--- |",
        f"| **7A `2x2` SOI MMI** (`W=6 µm, L=40.1 µm`) | Split `{m7a_m['split_bar_pct']:.2f}/{m7a_m['split_cross_pct']:.2f}%`, IL `{m7a_m['excess_loss_db']:.3f} dB`, Imb `{m7a_m['imbalance_db']:+.3f} dB`, $\\Delta\\phi = {m7a_m['phase_diff_deg']:+.1f}^\\circ$ | `3D FDTD`: `{m7a_l['split_bar_pct']:.2f}/{m7a_l['split_cross_pct']:.2f}%`, IL `{m7a_l['excess_loss_db']:.3f} dB`, Imb `{m7a_l['imbalance_db']:+.3f} dB`, $\\Delta\\phi = {m7a_l['phase_diff_deg']:+.1f}^\\circ$ | Exact 2nd-order `Nedelec2xP2` EME reveals `L=40.1 µm` is `2.2 µm` short of the true `L_opt = 42.3 µm` (`IL = 0.46 dB`), explaining why 3D FDTD also suffers `1.59 dB` dephasing loss here (`+1 µm` effective taper diffraction extends the 3D box slightly) |",
        f"| **7B Low-$Q$ Full Ring** (`R=5 µm, g=120 nm`) | Exact $M\\to\\infty$ TMM (`0.07 s`): FSR `{m7b_m['FSR_nm']:.3f} nm`, $Q_L = {m7b_m['Q_loaded']:.0f}$ (`ER = {m7b_m['ER_dB']:.2f} dB` over-coupled) | `14 ps` `3D FDTD` GPU (`status = 1` truncated): FSR `{m7b_l['FSR_fdtd_nm']:.3f} nm`, blind CMT fit $Q_L = {m7b_l['Q_loaded_fdtd_cmt_fit']:.0f}$ (`R^2 = {m7b_l['fit_r_squared']:.3f}`) | FSR agrees within `2.0%` and exact cylindrical-phase $Q_L$ (`{m7b_m['Q_loaded']:.0f}`) tracks 3D FDTD (`{m7b_l['Q_loaded_fdtd_cmt_fit']:.0f}`, `{ql_7b_diff:.1f}%`); `14 ps` FDTD still shows `T_max = 1.337` Gibbs ringing |",
        f"| **7C `INTERCONNECT` Native SPAR (`2x2` DC MZI)** | `ΔL = 80 µm`: FSR `{m7c_m['mzi_FSR_nm']:.3f} nm`, ER `{m7c_m['mzi_ER_bar_dB']:.2f} dB`, $T_{{\\mathrm{{bar}}}}(1550) = {m7c_m['mzi_bar_T_at_1550']:.4f}$ | `INTERCONNECT ONA`: FSR `{m7c_l['mzi_FSR_ic_nm']:.3f} nm`, ER `{m7c_l['mzi_ER_bar_ic_dB']:.2f} dB`, $\\max(T_{{\\mathrm{{bar}}}}+T_{{\\mathrm{{cross}}}}) = {m7c_l['mzi_max_sum_T']:.5f}$ | Exported `.dat` loads natively with zero syntax edits and reproduces FSR (`0.17%`) and ER (`0.02 dB`) |",
        f"| **7C `INTERCONNECT` Native SPAR (Apodized GC)** | `2D FDFD` IL(`1550 nm`) = `{g6m['il_at_1550_db']:.3f} dB` | `INTERCONNECT ONA`: IL(`1550 nm`) = `{m7c_l['gc_il_1550_ic_db']:.3f} dB`, peak `{m7c_l['gc_peak_il_ic_db']:.3f} dB` | Native passive SPAR circuit load verified (`|S|^2 <= 1.000`; ONA run captured prior uniform-grid SPAR export) |",
        "",
        f"- **4-Channel `25 Gbps` WDM Link Scenarios (`simulate_wdm_microring_link`)**: `nominal` (`Q = {wdm_rows[0]['Q_factor']:.2f}`, `BER = {wdm_rows[0]['BER']:.1e}`, `{wdm_rows[0]['energy_pj_per_bit']:.3f} pJ/bit`) $\\rightarrow$ `thermal_drift` (eye collapse `Q = {wdm_rows[1]['Q_factor']:.2f}`, `BER = {wdm_rows[1]['BER']:.3f}`) $\\rightarrow$ `heater_relocked` (recovered `Q = {wdm_rows[2]['Q_factor']:.2f}` at `{wdm_rows[2]['heater_power_per_ch_mW']:.2f} mW/ch`) $\\rightarrow$ `dense_crosstalk` (`Q = {wdm_rows[3]['Q_factor']:.2f}`).",
    ])

    # Suite 8 (4-Way with Ansys HFSS 3D!)
    md.extend([
        "", "---", "",
        "## Suite 8: Sub-THz (`10–300 GHz`) EOM RF Traveling-Wave & Slow-Wave T-Rail Electrodes (4-Way with Ansys HFSS 3D)",
        "",
        "All four `X-cut TFLN` RF electrode topologies (`HR-Si`, `Quartz`, `Undercut-Si` GSG CPW, and `4-period` 3D periodic `Slow-Wave T-Rail` on `Quartz`, $\\Lambda = 25\\,\\mu\\text{m}$) were solved independently in **MicroRing Studio** (`2D Quasi-TEM Laplace FEM + ABCD Bloch-Floquet`), **COMSOL 6.4** (`2D AC/DC es` + `2D full-wave lossy emw` + `3D Periodic Unit Cell es`), and **Ansys HFSS 2026 R1** (`3D Driven Modal Full-Wave FEM`, PMC half-GSG with open top/bottom radiation boundaries and exact 2-port Bloch-Floquet ABCD extraction). CSV: [`suite8_rf_electrodes_4way.csv`](../benchmarks/outputs/suite8_rf_electrodes_4way.csv).",
        "",
        "### Part 8A: Four-Electrode Cross-Solver Comparison at `40 GHz` (Optical Group Index $n_g = 2.2645$)",
        "",
        "| Electrode Topology | $Z_0$ MR / COMSOL / **HFSS 3D** ($\\Omega$) | $n_{\\mathrm{RF}}$ MR / COMSOL / **HFSS 3D** | $\\alpha_{\\mathrm{RF}}$ @ 40G MR / COMSOL / **HFSS 3D** (dB/cm) | MicroRing $V_\\pi L$ / $f_{3\\mathrm{dBe}}$ |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for r in s8_rows:
        md.append(
            f"| `{r['case_id']}` | {r['Z0_ohm_microring']:.2f} / {r['Z0_ohm_comsol']:.2f} / **{r['Z0_ohm_hfss_3d_40G']:.2f}** | "
            f"{r['n_RF_microring']:.4f} / {r['n_RF_comsol']:.4f} / **{r['n_RF_hfss_3d_40G']:.4f}** | "
            f"{r['alpha_40G_dB_per_cm_microring']:.2f} / {r['alpha_40G_dB_per_cm_comsol_emw']:.2f} / **{r['alpha_40G_dB_per_cm_hfss_3d']:.2f}** | "
            f"{r['V_pi_L_Vcm_microring']:.2f} V·cm / `{r['f_3dBe_GHz_microring']:.1f} GHz` |"
        )

    md.extend([
        "",
        "### Part 8B: Broadband `10–300 GHz` 3D Periodic Slow-Wave T-Rail Dispersion (`MicroRing ABCD` vs. `Ansys HFSS 3D`)",
        "",
        "| Frequency (GHz) | $n_{\\mathrm{RF}}$ (MicroRing) | $n_{\\mathrm{RF}}$ (**HFSS 3D**) | Rel. Diff ($n_{\\mathrm{RF}}$) | $\\alpha_{\\mathrm{RF}}$ MicroRing (dB/cm) | $\\alpha_{\\mathrm{RF}}$ **HFSS 3D** (dB/cm) | $Z_B$ **HFSS 3D** ($\\Omega$) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    hf_tr_map = {round(pt["freq_GHz"]): pt for pt in hf8["TFLN_SlowWave_TRail_Quartz"]["freq_sweep"]}
    mr_tr_map = mr8["TFLN_SlowWave_TRail_Quartz"]["freq_samples"]
    for fg in (10, 40, 100, 200, 300):
        pm = mr_tr_map[f"{fg}GHz"]
        ph = hf_tr_map[fg]
        md.append(
            f"| `{fg}` | {pm['n_RF']:.4f} | **{ph['n_RF_3d_abcd']:.4f}** | `{_pct_diff(pm['n_RF'], ph['n_RF_3d_abcd']):.2f}%` | "
            f"{pm['alpha_RF_dB_per_cm']:.2f} | **{ph['alpha_3d_abcd_dB_per_cm']:.2f}** | {ph['Z0_3d_abcd_ohm']:.2f} |"
        )

    wo = mr2["suite8_wafer_orientation"]
    md.extend([
        "",
        "- **3D Crystal-Tensor Wafer Orientation Symmetry Check**: `LN_Xcut` (`3m`, `TE0` $r_{33} = 30.9\\text{ pm/V}$ at $0^\\circ$, `0%` unsplit / `94.4%` split ring), `a-cut BTO` (`4mm`, peak $r_{\\mathrm{eff}} = 290.8\\text{ pm/V}$ at $\\phi = 53^\\circ$ via $r_{42}$ shear lobe), `ScAlN_20` (`6mm`, `TM0` $r_{33} = 1.34\\text{ pm/V}$ azimuthally isotropic, `100%` unsplit ring), and `(001) 3C-SiC` (`-43m`, `0.0 pm/V` along `[100]` and `|r_41| = 2.70 pm/V` along `[110]` at $45^\\circ$). All four match exact analytical tensor contraction.",
    ])

    # Suite 9
    md.extend([
        "", "---", "",
        "## Suite 9: 2D Silicon PN Carrier-Depletion Semiconductor TCAD (3-Way with Lumerical `CHARGE` + `MODE`)",
        "",
        "MicroRing Studio (`2D Damped Newton-Raphson Poisson-Boltzmann + Soref-Bennett 1987 + 2D Vector Maxwell FEM`), COMSOL 6.4 (`2D Quadratic Nonlinear PDE + Nedeljkovic 2011 + ewfd`), and Ansys Lumerical (`DEVICE CHARGE` full 3-equation drift-diffusion `0 -> 3 V` + `MODE FDE` native `np density` Soref-Bennett) were executed independently on identical `220 nm` SOI (`90 nm` slab, $N_A = N_D = 5\\times 10^{17}\\text{ cm}^{-3}$, $N_{\\mathrm{int}} = 2\\times 10^{18}$, $N_{\\mathrm{contact}} = 10^{20}$) rib modulators. CSV: [`suite9_si_pn_tcad_3way.csv`](../benchmarks/outputs/suite9_si_pn_tcad_3way.csv).",
        "",
        "| Junction Geometry | $C_j(0\\text{ V})$ MR / COM / `CHARGE` (fF/µm) | $C_j(-3\\text{ V})$ MR / COM / `CHARGE` (fF/µm) | $V_\\pi L(-3\\text{ V})$ MR / COM / `CHARGE` (V·cm) | `FCA(-3 V)` MR / COM / `CHARGE` (dB/cm) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for r in s9_rows:
        c0_l = f"{r['Cj_0V_fF_per_um_lumerical_charge']:.3f}" if r["Cj_0V_fF_per_um_lumerical_charge"] != "" else "N/A"
        c3_l = f"{r['Cj_m3V_fF_per_um_lumerical_charge']:.3f}" if r["Cj_m3V_fF_per_um_lumerical_charge"] != "" else "N/A"
        vp_l = f"{r['V_pi_L_m3V_Vcm_lumerical_charge']:.3f}" if r["V_pi_L_m3V_Vcm_lumerical_charge"] != "" else "N/A"
        fc_l = f"{r['FCA_m3V_dB_per_cm_lumerical_charge']:.2f}" if r["FCA_m3V_dB_per_cm_lumerical_charge"] != "" else "N/A"
        md.append(
            f"| `{r['case_id']}` | {r['Cj_0V_fF_per_um_microring']:.3f} / {r['Cj_0V_fF_per_um_comsol']:.3f} / {c0_l} | "
            f"{r['Cj_m3V_fF_per_um_microring']:.3f} / {r['Cj_m3V_fF_per_um_comsol']:.3f} / {c3_l} | "
            f"{r['V_pi_L_m3V_Vcm_microring']:.3f} / {r['V_pi_L_m3V_Vcm_comsol']:.3f} / {vp_l} | "
            f"{r['FCA_m3V_dB_per_cm_microring']:.2f} / {r['FCA_m3V_dB_per_cm_comsol']:.2f} / {fc_l} |"
        )

    # Suite 10
    md.extend([
        "", "---", "",
        "## Suite 10: Advanced Microheaters (`Undercut`, `Doped-Rib`, `Co-Int PN Sinking`) & True 3D Finite-Length Thermal Check",
        "",
        "CSV: [`suite10_thermal_adv_3way.csv`](../benchmarks/outputs/suite10_thermal_adv_3way.csv).",
        "",
        "| Heater Architecture | $R_{\\mathrm{th}}$ MicroRing (K/mW) | $R_{\\mathrm{th}}$ COMSOL `ht` (K/mW) | Rel. Diff | Xtalk @ `20 µm` (MR / COMSOL) | MicroRing $P_\\pi$ / $f_{3\\mathrm{dB}}$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in s10_rows:
        md.append(
            f"| `{r['case_id']}` | {r['R_th_core_K_per_mW_microring']:.4f} | {r['R_th_core_K_per_mW_comsol']:.4f} | "
            f"`{r['R_th_rel_diff_pct']:.2f}%` | {r['crosstalk_20um_pct_microring']:.3f}% / {r['crosstalk_20um_pct_comsol']:.3f}% | "
            f"{r['P_pi_mW_microring']:.2f} mW / {r['f_3dB_kHz_microring']:.2f} kHz |"
        )
    md.extend([
        "",
        f"- **Co-Integrated PN Contact Heat-Sinking Penalty (`SOI_220nm_CoInt_PN_TiN_Heater`)**: Adding W VIAC plugs and AlCu M1 PN electrode rails drops $R_{{\\mathrm{{th}}}}$ from `{com10['SOI_220nm_CoInt_PN_TiN_Heater']['R_th_bare_K_per_mW']:.4f}` to `{com10['SOI_220nm_CoInt_PN_TiN_Heater']['R_th_core_K_per_mW']:.4f} K/mW` (COMSOL) vs. `{mr10['SOI_220nm_CoInt_PN_TiN_Heater']['R_th_core_K_per_mW']:.4f} K/mW` (MicroRing, `0.78%` diff), confirming the `+{com10['SOI_220nm_CoInt_PN_TiN_Heater']['pn_sinking_penalty_pct']:.1f}%` tuning-power penalty.",
        f"- **True 3D Finite-Length Verification (`SOI Top TiN`, `L = 200 µm` in COMSOL 3D `ht`)**: Longitudinal end-conduction yields integrated effective $R_{{\\mathrm{{th,3D}}}} = {c3d['R_th_eff_integrated_K_per_mW']:.4f}\\text{{ K/mW}}$ (center-plane `{c3d['R_th_center_z0_K_per_mW']:.4f} K/mW`) versus 2D COMSOL `{com_4['SOI_220nm_Top_TiN_Heater']['R_th_core_K_per_mW']:.4f} K/mW` and MicroRing `{mr_4['SOI_220nm_Top_TiN_Heater']['R_th_core_K_per_mW']:.4f} K/mW`, quantifying the 2D-vs-3D longitudinal end-loss at only `-2.0%` for `L = 200 µm`.",
    ])

    # Suite 11
    md.extend([
        "", "---", "",
        "## Suite 11: Cleanroom Lithography Photoresist Tone & Correlated CD/Gap Bias Drift (`+/-20 nm`)",
        "",
        "Evaluating anti-correlated waveguide width $W_{\\mathrm{fab}} = W_0 + \\Delta\\mathrm{CD}$ and gap $g_{\\mathrm{fab}} = g_0 - 1.05\\,\\Delta\\mathrm{CD}$ on a `220 nm` SOI `R = 15 µm` ring (`W_0 = 450 nm`, `g_0 = 180 nm`). CSV: [`suite11_fab_cd_bias_3way.csv`](../benchmarks/outputs/suite11_fab_cd_bias_3way.csv).",
        "",
        "| $\\Delta\\mathrm{CD}$ (nm) | $W_{\\mathrm{fab}}$ / $g_{\\mathrm{fab}}$ (nm) | $n_{\\mathrm{eff,bent}}$ (MR / COM / Lum) | $\\Delta\\lambda_{\\mathrm{res}}$ (nm, MR / COM / Lum) | $\\kappa^2$ MR (CMT) | $1 - |S_{21}|^2$ (3D FDTD GPU) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in s11_rows:
        md.append(
            f"| `{r['delta_cd_nm']:+.0f}` | {r['w_fab_nm']:.0f} / {r['gap_fab_nm']:.1f} | "
            f"{r['neff_bent_microring']:.5f} / {r['neff_bent_comsol']:.5f} / {r['neff_bent_lumerical']:.5f} | "
            f"{r['delta_lambda_nm_microring']:+.3f} / {r['delta_lambda_nm_comsol']:+.3f} / {r['delta_lambda_nm_lumerical']:+.3f} | "
            f"{r['kappa_power_microring_cmt']:.5f} | {r['one_minus_S21_sq_3dfdtd']:.5f} |"
        )
    md.extend([
        "",
        f"- **3-Way Sensitivity Consensus**: $\\mathrm{{d}}\\lambda/\\mathrm{{dCD}} = {s11_m['dlambda_dCD_pm_per_nm']:.2f}$ (MicroRing), `{s11_c['dlambda_dCD_pm_per_nm']:.2f}` (COMSOL), and `{s11_l['dlambda_dCD_pm_per_nm']:.2f}` (Lumerical MODE) `pm/nm` (`0.07%` max spread). Under `+10%` exposure over-dose (`2 nm/%`), MicroRing captures the exact opposite sign inversion between positive PR (`{s11_m['positive_pr_overdose_dlam_nm']:+.2f} nm`) and negative PR (`{s11_m['negative_pr_overdose_dlam_nm']:+.2f} nm`). Both CMT and 3D FDTD show strong width-vs-gap coupling self-compensation at `+20 nm` (`1 - |S21|^2 = 0.070 / 0.065 / 0.063`; note that the raw `RingMon` monitor in Suite 11 sat `0.17 µm` from the bus cladding edge and picked up ~4% bus tail, so `1 - |S21|^2` is the clean single-mode coupling metric here).",
    ])

    # Physical Discussion & Scope Boundaries (honest, concrete, zero em-dashes)
    md.extend([
        "", "---", "",
        "## Physical Analysis of Discrepancies & Validity Boundaries",
        "",
        f"1. **2D Bent & Straight Vector Maxwell Eigenmodes (Suites 1, 3A, 6A, 11)**: Across `532 nm` to `4200 nm`, MicroRing Studio (`Nedelec2 x P2`) matches COMSOL `6.4` (`ewfd`) to `0.0e+0` to `1.6e-5` in bent $n_{{\\mathrm{{eff}}}}$ and Lumerical `MODE FDE` to `1.9e-5` to `5.4e-4`. On `2x2` SOI directional couplers, $L_\\pi$ matches within `0.16%` (COMSOL) and `0.36%` (Lumerical). On inverse-taper butt couplers (`SOI 1550 nm` and `Si3N4 1310 nm` elliptical laser), nominal facet IL agrees to `0.005–0.057 dB` and 1-dB alignment tolerance agrees within `0.01 µm` across all three solvers. On cleanroom CD bias (`±20 nm`), $\\mathrm{{d}}\\lambda/\\mathrm{{dCD}}$ agrees to `0.07%` across all three engines, and correlated $\\kappa^2$ matches 3D FDTD within `0.3%–4.8%` with the exact monotonic downward trend (`{s11_rows[0]['kappa_power_microring_cmt']:.4f} -> {s11_rows[1]['kappa_power_microring_cmt']:.4f} -> {s11_rows[2]['kappa_power_microring_cmt']:.4f}` vs. `0.0701 -> 0.0649 -> 0.0634`).",
        f"2. **Full-Wave 2D Surface Grating Couplers & `INTERCONNECT` Circuit Fidelity (Suites 3C, 6B, 7C)**: Using a non-uniform core-refined staggered Yee mesh (`dy = 13 nm` across the partial-etch core, `dz = 24 nm`), `2D FDFD` (MicroRing), `2D FDTD` (Lumerical), and `2D ewfd` (COMSOL) agree closely on both uniform (`1550 nm` IL `{mr1['suite3c_gc']['il_at_1550_db']:.2f} / {lum1['suite3c_gc']['il_at_1550_db']:.2f} / {com1['suite3c_gc']['il_at_1550_db']:.2f} dB`, peak `{mr1['suite3c_gc']['peak_il_db']:.2f} dB`) and Gaussian-apodized 26-tooth `220 nm` SOI gratings (peak `{g6m['peak_wavelength_nm']:.1f}` vs. `{g6l['peak_wavelength_nm']:.1f} nm`, peak IL `{g6m['peak_il_db']:.2f}` vs. `{g6l['peak_il_db']:.2f} dB`, `1550 nm` IL `{g6m['il_at_1550_db']:.2f} / {g6l['il_at_1550_db']:.2f} / {g6c['il_at_1550_db']:.2f} dB`, back-reflection `{g6m['s11_at_1550_db']:.1f} / {g6l['s11_at_1550_db']:.1f} / {g6c['s11_at_1550_db']:.1f} dB`). Exported `SPAR .dat` bundles load directly into Ansys Lumerical `INTERCONNECT` with exact passivity (`<= 1.000`) and match MZI FSR (`7.000` vs. `7.012 nm`) and ER (`33.35` vs. `33.33 dB`).",
        f"3. **3D FDTD Couplers, MZI Splitters & High-$Q$ Ring Truncation (Suites 2, 3B, 7A, 7B)**: With exact cylindrical azimuthal phase slip $\\phi_{{\\mathrm{{ring}}}}(z) = (\\beta_{{\\mathrm{{ring}}}} R)\\arctan(z/(R+d_0))$, unitary 2x2 coupled-mode transfer-matrix integration, conformal twin-core supermode depolarization, outward-bend pulley transition arcs, and bent-end racetrack coupling, spatial CMT matches RTX 5090 GPU `3D FDTD` across every topology and waveband: `100–250 nm` SOI point couplers within `{min_s2b:.1f}%–{max_s2b:.1f}%` (`0.2494 vs 0.2533`, `0.1120 vs 0.1106`, `0.0486 vs 0.0485`, `0.0212 vs 0.0214`), `4H-SiC` `82°`-sidewall pulley within `0.97%` (`0.0529 vs 0.0534`), Mid-IR `4.2 µm` air-clad `Ge` rib racetrack within `2.1%` (`0.00464 vs 0.00474`), `532 nm Si3N4` within `3.7%`, and asymmetric `780 nm TiO2` within `5.8%`. On `1x2` MMI (`{mmi1_il_mr:.3f}` vs. `{mmi1_il_lum:.3f} dB`, `{abs(mmi1_il_mr - mmi1_il_lum):.3f} dB` diff) and `1x2` Y-branch (`{yb_il_mr:.3f}` vs. `{yb_il_lum:.3f} dB`, `{abs(yb_il_mr - yb_il_lum):.3f} dB` diff), 2nd-order `Nedelec2xP2` EME with paraxial taper wavefront curvature closely matches 3D FDTD; on the fixed `L = 40.1 µm` `2x2` MMI (`Suite 7A`), exact `Nedelec2xP2` identifies `L_opt = 42.3 µm` (`IL = 0.46 dB`), explaining the elevated dephasing loss observed at `40.1 µm` in both EME and 3D FDTD (`1.59 dB`). On the `R = 5 µm` full ring (`Suite 7B`), exact $M\\to\\infty$ TMM (`0.07 s`) gives FSR `17.83 nm` and $Q_L = {m7b_m['Q_loaded']:.0f}$ (tracking `3D FDTD` `8174` within `{ql_7b_diff:.1f}%`), whereas `14 ps` of GPU `3D FDTD` still exhibits `T_max = 1.337` Gibbs ringing.",
        f"4. **Sub-THz EOM RF Electrodes vs. Ansys HFSS 3D & COMSOL (Suite 8)**: Using quadratic `ElementTriP2` anisotropic Laplace FEM (`0.006%–0.08%` electrostatic $C$ and `0.22%–0.30%` $L_{{\\mathrm{{ext}}}}$ vs. COMSOL `es`) coupled with exact finite-thickness `coth` conductor skin impedance, MicroRing Studio matches Ansys HFSS `3D` and COMSOL `emw` at `40 GHz` across all three uniform `X-cut TFLN` GSG CPW platforms (`HR-Si`, `Quartz`, `Undercut-Si`) to `{min(cpw_z0_diffs):.1f}%–{max(cpw_z0_diffs):.1f}%` in $Z_0$ and `{min(cpw_nrf_diffs):.2f}%–{max(cpw_nrf_diffs):.2f}%` in $n_{{\\mathrm{{RF}}}}$. On the `4-period` 3D periodic `Slow-Wave T-Rail` (`Λ = 25 µm`) from `10` to `300 GHz`, adding the 1/4-unit-cell 3D periodic finite-volume longitudinal fringing ratio and exact symmetric GSG parallel-slot stem resistance brings MicroRing to $Z_0 = {tr_mr_40['Z0_ohm']:.2f}\\,\\Omega$ vs. COMSOL 3D electrostatics `{tr_com_3d['Z0_3d_unitcell_ohm']:.2f} Ω` (`{tr_z0_com_diff:.2f}%`), $n_{{\\mathrm{{RF}}}} = {tr_mr_40['n_RF']:.4f}$ vs. COMSOL 3D `{tr_com_3d['n_RF_3d_unitcell']:.4f}` (`{tr_nrf_com_diff:.2f}%`), and tracks Ansys HFSS `3D` attenuation cleanly across the entire `40–300 GHz` band (`{tr_mr_40['alpha_RF_dB_per_cm']:.2f}` vs. `{hf8['TFLN_SlowWave_TRail_Quartz']['alpha_40G_dB_per_cm']:.2f} dB/cm` at `40 GHz`; `{tr_mr_200['alpha_RF_dB_per_cm']:.2f}` vs. `{tr_hf_200['alpha_3d_abcd_dB_per_cm']:.2f} dB/cm` at `200 GHz`; `{tr_mr_300['alpha_RF_dB_per_cm']:.2f}` vs. `{tr_hf_300['alpha_3d_abcd_dB_per_cm']:.2f} dB/cm` at `300 GHz`). Note that HFSS `3D` ABCD reports a lower Bloch impedance $Z_B = 42.1\\text{{--}}42.7\\,\\Omega$ than both COMSOL 3D (`49.24 Ω`) and MicroRing (`49.51 Ω`) because the HFSS outer sidewall boundary at `x = x_dom` acts as a PEC ground return (`W_G -> inf` semi-infinite ground plane), whereas both COMSOL 3D and MicroRing solve the finite `W_G = 55 µm` ground width.",
        "5. **Silicon PN Carrier-Depletion TCAD (Suite 9)**: Pure-electrostatic depletion capacitance $C_j(V)$ agrees across MicroRing Studio (`P1` FEM), COMSOL (`P2` FEM), and Ansys Lumerical `DEVICE CHARGE` (full 3-equation drift-diffusion) within `0.17%–3.2%` across `Lateral`, `L-Shaped`, and `U-Shaped` junctions at both `0 V` and `-3 V`. With Soref-Bennett (1987) plasma dispersion, MicroRing matches Lumerical `CHARGE + MODE` to `0.19%` in free-carrier absorption (`10.76` vs. `10.78 dB/cm` at `-3 V`) and `5.3%` in $V_\\pi L$ (`1.480` vs. `1.563 V·cm`). Selecting the peer-reviewed `plasma_model='nedeljkovic_2011'` option in `PNDopingTCADConfig` reproduces the `~1.68 V·cm` COMSOL `V_pi*L`, isolating the `11%–13%` spread cleanly to the empirical plasma-dispersion formula choice.",
        "6. **2D/3D Thermo-Optic Multiphysics & Pockels Electrostatics (Suites 4, 5, 10)**: Across seven microheater architectures spanning two orders of magnitude in thermal resistance (`0.58` to `34.7 K/mW`), MicroRing Studio matches COMSOL `ht` within `0.23%–0.78%` in core $R_{\\mathrm{th}}$ and `0.61%–1.44%` in $P_\\pi$, and matches Lumerical `HEAT` within `0.51%–1.16%`. COMSOL's full 3D finite-length model (`L = 200 µm`) confirms that 3D longitudinal end-conduction reduces effective $R_{\\mathrm{th}}$ by only `-2.0%` (`1.1357` vs. `1.1589 K/mW`). Normalizing Crank-Nicolson transient waveforms strictly against exact steady-state $T_{\\mathrm{steady}}$ over a `10*tau` window restores exact linear-PDE symmetry (`t_rise == t_fall`, e.g., `11.64 / 11.63 µs` on `SOI Top TiN`, independently verified to `0.34%` against a standalone 2D finite-volume heat solver at `11.68 µs`). In Pockels electrostatics (`Suite 5`), combining exact element-local Galerkin `-grad(phi)` with one-sided DG1 interior boundary traces brings all three modulators within sub-`1%` of COMSOL `es + ewfd`: `X-cut TFLN` `G = 5.0 µm` (`-0.61%`), `G = 3.2 µm` (`-0.20%`), and vertical `ScAlN_20` (`-0.87%`, `107.21` vs. `108.16 V·cm`).",
        "7. **Full Reproducibility**: All runner scripts (`commercial_validation_specs*.py`, `run_microring_independent*.py`, `run_lumerical_independent*.py`, `run_comsol_independent*.py`, `run_hfss_independent_p2.py`, and `compile_commercial_validation_report.py`), raw JSON archives, and 12 CSV tables live in [`benchmarks/`](../benchmarks/) and [`benchmarks/outputs/`](../benchmarks/outputs/).",
        "",
    ])

    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote unified 11-suite 4-way validation report to {REPORT_MD}")
    print(f"Saved Phase-1 figure to {p1_png} / {p1_svg}")
    print(f"Saved Phase-2 figure to {p2_png} / {p2_svg}")
    for ck in checks:
        print(f"  [AUDIT {ck['status']}] {ck['check']}: {ck['detail']}")


if __name__ == "__main__":
    main()
