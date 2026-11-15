"""Cleanroom Lithography Dose, Photoresist Tone & Etch Bias Tolerance Engine (`microring.fab_tolerance`).

Models realistic cleanroom and foundry fabrication errors on integrated photonic microring
resonators across **E-Beam Lithography (EBL)** and **DUV Photolithography (`248 nm` KrF / `193 nm` ArF)**:

1. **Photoresist Tone & Dose Polarity (`RESIST_TONE_CATALOG`)**:
   - **`positive_pr` (Positive Photoresist / EBL Trench Etch — `ZEP520A`, `CSAR 62`, `PMMA`, `KrF/ArF Positive DUV`)**:
     Exposed regions dissolve in developer while unexposed resist protects the waveguide core.
     * **Over-dose (`+ΔDose %`)** -> Core width **shrinks** (`ΔCD < 0`), coupling gap **widens** (`gap_fab = gap_drawn + η|ΔCD|`)
       -> Weaker coupling (`κ` drops toward under-coupling) & **blue-shifted** resonance (`Δλ_res < 0`).
     * **Under-dose (`-ΔDose %`)** -> Core width **widens** (`ΔCD > 0`), coupling gap **shrinks** (`gap_fab = gap_drawn - η·ΔCD`)
       -> Stronger coupling (`κ` rises toward over-coupling, risk of narrow-gap resist scumming/bridging) & **red-shifted** resonance (`Δλ_res > 0`).
   - **`negative_pr` (Negative Photoresist / Core Exposure or Lift-Off — `HSQ / FOx-16`, `ma-N 2403`, `SU-8`, `Negative DUV`)**:
     Exposed regions cross-link (or invert via metal hard-mask lift-off) to form the waveguide core mask.
     * **Over-dose (`+ΔDose %`)** -> Core width **blooms wider** (`ΔCD > 0`), coupling gap **shrinks** (`gap_fab = gap_drawn - η·ΔCD`)
       -> Stronger coupling (`κ` rises toward over-coupling, risk of HSQ backscatter gap bridging) & **red-shifted** resonance (`Δλ_res > 0`).
     * **Under-dose (`-ΔDose %`)** -> Core width **shrinks** (`ΔCD < 0`), coupling gap **widens** (`gap_fab = gap_drawn + η|ΔCD|`)
       -> Weaker coupling (`κ` drops toward under-coupling, risk of narrow-core delamination) & **blue-shifted** resonance (`Δλ_res < 0`).
   - **`direct_cd` (Direct Foundry CD Bias `ΔCD [nm]`)**:
     Directly sweeps fabricated waveguide width bias `ΔCD` (`nm`) with anti-correlated gap shift `gap_fab = gap_drawn - η·ΔCD`.

2. **Coupled 2D Vector FEM + Spatial CMT Corner Evaluation**:
   Unlike naive independent parameter sweeps, real lithography simultaneously perturbs `W_ring`,
   `W_bus`, and `gap` in opposite directions (plus optional RIE film-thickness `ΔH` and sidewall-angle
   `Δθ_sw` corners), shifting both evanescent decay, pulley phase-matching `Δβ`, cavity `n_eff`, and `ER`.

3. **Foundry Pre-Bias / OPC Dose-Bracket Array `.GDS` Generator (`export_fab_bias_array_gds`)**:
   Exports a multi-cell MPW test array `.gds` + KLayout `.lyp` with pre-compensated `(W - δ, gap + δ)`
   cells so at least one resonator on the fabricated chip lands at critical coupling despite dose drift.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import gdstk
import numpy as np

from .gds_bridge import write_klayout_layer_props
from .materials import T_REF_K, get_material_fn
from .mrr_topologies import MATERIAL_CATALOG, MRRDesignConfig, export_mrr_gds, simulate_mrr_topology

RESIST_TONE_CATALOG: dict[str, dict] = {
    "positive_pr": {
        "name_en": "Positive PR · Trench / Clear-Field Etch (ZEP520A, CSAR 62, PMMA, KrF/ArF DUV)",
        "name_zh": "正性光刻胶 · 沟槽曝光直接刻蚀 (ZEP520A, CSAR 62, PMMA, 248/193nm DUV 正胶)",
        "cd_sign_per_pos_dose": -1.0,  # Over-dose (+%) erodes unexposed resist core -> ΔCD < 0
        "default_slope_nm_per_pct": 2.0,
        "default_prox_gap_factor": 1.05,
        "overdose_effect_en": "Over-dose erodes resist edges: waveguide W shrinks (ΔCD < 0), coupling gap widens, coupling weakens (under-coupled), resonance blue-shifts.",
        "underdose_effect_en": "Under-dose leaves wider resist core: waveguide W widens (ΔCD > 0), coupling gap shrinks, coupling strengthens (over-coupled / scumming risk), resonance red-shifts.",
        "overdose_effect_zh": "过曝光（Over-dose）侵蚀胶边缘：波导宽度变窄（ΔCD < 0），耦合间距变宽，耦合减弱（偏向欠耦合），谐振峰蓝移。",
        "underdose_effect_zh": "欠曝光（Under-dose）残留更宽胶条：波导宽度变宽（ΔCD > 0），耦合间距变窄，耦合增强（偏向过耦合，窄缝易残胶粘连），谐振峰红移。",
    },
    "negative_pr": {
        "name_en": "Negative PR · Core Exposure / Lift-Off (HSQ, ma-N 2403, SU-8, Negative DUV)",
        "name_zh": "负性光刻胶 · 波导芯直接曝光 / 剥离反转 (HSQ, ma-N 2403, SU-8, 负胶)",
        "cd_sign_per_pos_dose": +1.0,  # Over-dose (+%) blooms cross-linked core -> ΔCD > 0
        "default_slope_nm_per_pct": 2.0,
        "default_prox_gap_factor": 1.15,
        "overdose_effect_en": "Over-dose blooms cross-linked resist: waveguide W widens (ΔCD > 0), coupling gap shrinks rapidly, coupling strengthens (over-coupled / HSQ gap-bridging risk), resonance red-shifts.",
        "underdose_effect_en": "Under-dose narrows cross-linked core: waveguide W shrinks (ΔCD < 0), coupling gap widens, coupling weakens (under-coupled / delamination risk), resonance blue-shifts.",
        "overdose_effect_zh": "过曝光（Over-dose）使交联区外扩：波导宽度变宽（ΔCD > 0），耦合间距急剧变窄，耦合增强（偏向过耦合，窄缝易受邻近效应桥接），谐振峰红移。",
        "underdose_effect_zh": "欠曝光（Under-dose）使交联芯变窄：波导宽度变窄（ΔCD < 0），耦合间距变宽，耦合减弱（偏向欠耦合，极窄波导易倒塌脱落），谐振峰蓝移。",
    },
    "direct_cd": {
        "name_en": "Direct Foundry CD Bias Sweep (ΔCD [nm], Resist-Independent)",
        "name_zh": "代工厂线宽偏置直接扫描 (ΔCD [nm]，不区分正负胶)",
        "cd_sign_per_pos_dose": +1.0,
        "default_slope_nm_per_pct": 1.0,
        "default_prox_gap_factor": 1.00,
        "overdose_effect_en": "Positive CD bias (+ΔCD > 0): waveguide W widens, coupling gap shrinks by η·ΔCD, resonance red-shifts.",
        "underdose_effect_en": "Negative CD bias (ΔCD < 0): waveguide W narrows, coupling gap widens by η·|ΔCD|, resonance blue-shifts.",
        "overdose_effect_zh": "正线宽偏置（+ΔCD > 0）：波导变宽，耦合间距同步缩小 η·ΔCD，谐振峰红移。",
        "underdose_effect_zh": "负线宽偏置（ΔCD < 0）：波导变窄，耦合间距同步增大 η·|ΔCD|，谐振峰蓝移。",
    },
}


def _interp_window_1d(x_vals: np.ndarray, y_vals: np.ndarray, threshold: float) -> tuple[float | None, float | None, float]:
    """Find contiguous interval around center where `y_vals >= threshold` via linear interpolation."""
    n = len(x_vals)
    if n == 0 or float(np.max(y_vals)) < threshold:
        return None, None, 0.0

    # Ensure ascending x order for clean interval extraction
    order = np.argsort(x_vals)
    xs = x_vals[order]
    ys = y_vals[order]

    above = ys >= threshold
    if not np.any(above):
        return None, None, 0.0

    # Pick the connected component closest to x = 0 (nominal)
    center_idx = int(np.argmin(np.abs(xs)))
    if not above[center_idx]:
        # Fall back to peak ER index if nominal design itself isn't critically coupled yet
        center_idx = int(np.argmax(ys))
        if not above[center_idx]:
            return None, None, 0.0

    i_left = center_idx
    while i_left > 0 and above[i_left - 1]:
        i_left -= 1
    if i_left > 0:
        x0, x1 = float(xs[i_left - 1]), float(xs[i_left])
        y0, y1 = float(ys[i_left - 1]), float(ys[i_left])
        frac = (threshold - y0) / max(abs(y1 - y0), 1e-9)
        x_low = x0 + np.clip(frac, 0.0, 1.0) * (x1 - x0)
    else:
        x_low = float(xs[0])

    i_right = center_idx
    while i_right < n - 1 and above[i_right + 1]:
        i_right += 1
    if i_right < n - 1:
        x0, x1 = float(xs[i_right]), float(xs[i_right + 1])
        y0, y1 = float(ys[i_right]), float(ys[i_right + 1])
        frac = (y0 - threshold) / max(abs(y0 - y1), 1e-9)
        x_high = x0 + np.clip(frac, 0.0, 1.0) * (x1 - x0)
    else:
        x_high = float(xs[-1])

    return round(float(x_low), 3), round(float(x_high), 3), round(float(x_high - x_low), 3)


def _extract_centered_dip_trace(sim_res: dict, half_span_nm: float = 0.65, n_interp: int = 161) -> dict:
    """Extract a clean resonance dip trace around the deepest resonance inside the simulated window."""
    wl = np.asarray(sim_res["spectrum"]["wavelength_nm"], dtype=float)
    thru_db = np.asarray(sim_res["spectrum"]["thru_dB"], dtype=float)
    idx_min = int(np.argmin(thru_db))
    wl_dip = float(wl[idx_min])
    fsr_nm = float(sim_res["metrics"]["FSR_nm"])
    h_span = min(max(0.12, 0.38 * fsr_nm), max(0.18, half_span_nm))
    detuning_pm = np.linspace(-h_span * 1e3, h_span * 1e3, n_interp)
    wl_query = wl_dip + detuning_pm * 1e-3
    thru_interp = np.interp(wl_query, wl, thru_db)
    return {
        "dip_wavelength_nm": round(wl_dip, 5),
        "detuning_pm": np.round(detuning_pm, 2).tolist(),
        "thru_dB": np.round(thru_interp, 4).tolist(),
    }


def run_fabrication_error_analysis(
    base_cfg: MRRDesignConfig,
    *,
    resist_tone: str = "positive_pr",
    max_dose_error_pct: float = 15.0,
    dose_slope_nm_per_pct: float | None = None,
    max_cd_bias_nm: float = 30.0,
    proximity_gap_factor: float | None = None,
    delta_h_film_nm: float = 0.0,
    delta_sidewall_deg: float = 0.0,
    roughness_loss_penalty_pct_per_10nm: float = 6.0,
    num_points: int = 7,
) -> dict:
    """Evaluate coupled lithography exposure / resist-tone / etch-bias errors on an MRR.

    Parameters
    ----------
    base_cfg : MRRDesignConfig
        Nominal (drawn mask) microring resonator configuration.
    resist_tone : {"positive_pr", "negative_pr", "direct_cd"}
        Photoresist / EBL process polarity.
    max_dose_error_pct : float
        Symmetric exposure dose error range `[-max_dose_error_pct, +max_dose_error_pct]` [%].
    dose_slope_nm_per_pct : float, optional
        Magnitude of core CD shift per 1% exposure dose change [nm/%] (default `2.0 nm/%`).
    max_cd_bias_nm : float
        Used when `resist_tone == "direct_cd"` (`[-max_cd_bias_nm, +max_cd_bias_nm]` [nm]).
    proximity_gap_factor : float, optional
        Multiplier `η` in `gap_fab = gap_drawn - η * ΔCD`. `1.0` = geometric sidewall shift;
        `>1.0` captures EBL electron backscatter / optical proximity inside narrow gaps.
    delta_h_film_nm : float
        Linearly correlated film-thickness / over-etch corner across the sweep (`-ΔH` to `+ΔH` [nm]).
    delta_sidewall_deg : float
        Linearly correlated sidewall-angle drift across the sweep (`-Δθ` to `+Δθ` [deg]).
    roughness_loss_penalty_pct_per_10nm : float
        Fractional propagation-loss change (%) when narrower/rougher core increases sidewall overlap.
    num_points : int
        Odd number of corner points (minimum 5, forced odd so exact `0` nominal is always included).
    """
    tone_key = resist_tone if resist_tone in RESIST_TONE_CATALOG else "positive_pr"
    tone_meta = RESIST_TONE_CATALOG[tone_key]

    n_pts = max(5, int(num_points))
    if n_pts % 2 == 0:
        n_pts += 1
    mid_idx = n_pts // 2

    slope = float(
        dose_slope_nm_per_pct
        if dose_slope_nm_per_pct is not None
        else tone_meta["default_slope_nm_per_pct"]
    )
    prox = float(
        proximity_gap_factor
        if proximity_gap_factor is not None
        else tone_meta["default_prox_gap_factor"]
    )
    cd_sign = float(tone_meta["cd_sign_per_pos_dose"])

    if tone_key == "direct_cd":
        x_axis_vals = np.linspace(-abs(float(max_cd_bias_nm)), abs(float(max_cd_bias_nm)), n_pts)
        cd_shifts_nm = x_axis_vals.copy()
        dose_errors_pct = x_axis_vals / max(slope, 0.1)
        x_label_en = "CD Bias ΔCD [nm]"
        x_label_zh = "线宽偏置 ΔCD [nm]"
    else:
        x_axis_vals = np.linspace(-abs(float(max_dose_error_pct)), abs(float(max_dose_error_pct)), n_pts)
        dose_errors_pct = x_axis_vals.copy()
        cd_shifts_nm = cd_sign * slope * dose_errors_pct
        x_label_en = "Exposure Dose Error ΔDose [%]"
        x_label_zh = "曝光剂量误差 ΔDose [%]"

    # Solve nominal center point first so we have exact reference n_eff(0), n_g(0), and Q_int(0)
    cfg_nom = replace(base_cfg, num_spectrum_points=min(max(base_cfg.num_spectrum_points, 601), 1001))
    nom_sim = simulate_mrr_topology(cfg_nom)
    nom_m = nom_sim["metrics"]
    neff_nom = float(nom_m["neff_ring"])
    ng_nom = float(nom_m["ng_ring"])
    wl0_nm = float(cfg_nom.wavelength_nm)
    fsr_nom_nm = float(nom_m["FSR_nm"])
    base_loss_db_cm = float(nom_m["alpha_loss_db_per_cm"])

    # Exact modal confinement-weighted thermo-optic tuning rate dλ/dT [pm/K] for heater trim budget
    dlambda_dt_pm_per_k = float(nom_sim["tuning"]["thermo_optic_pm_per_k"])

    corners: list[dict] = []
    drc_warnings: list[str] = []
    corner_spectra: dict[str, dict] = {}

    for idx in range(n_pts):
        dose_pct = float(dose_errors_pct[idx])
        dcd_nm = float(cd_shifts_nm[idx])
        dcd_um = dcd_nm * 1e-3
        frac_span = (idx - mid_idx) / max(mid_idx, 1)  # -1.0 .. +1.0

        w_ring_fab = max(0.10, round(cfg_nom.ring_width_um + dcd_um, 4))
        w_bus_fab = max(0.10, round(cfg_nom.bus_width_um + dcd_um, 4))

        # Anti-correlated gap shift: when waveguides widen (+ΔCD > 0), edge-to-edge gap shrinks!
        gap_raw_thru = cfg_nom.gap_thru_um - prox * dcd_um
        gap_raw_drop = cfg_nom.gap_drop_um - prox * dcd_um
        inter_gap_raw = cfg_nom.inter_ring_gap_um - prox * dcd_um

        gap_bridged = gap_raw_thru <= 0.045 or (
            cfg_nom.topology.endswith("_ad") and gap_raw_drop <= 0.045
        )
        gap_scum_risk = (not gap_bridged) and (gap_raw_thru < 0.085)
        narrow_pinch_risk = min(w_ring_fab, w_bus_fab) < 0.150

        gap_sim_thru = max(0.045, round(gap_raw_thru, 4))
        gap_sim_drop = max(0.045, round(gap_raw_drop, 4))
        inter_gap_sim = max(0.060, round(inter_gap_raw, 4))

        h_film_fab = max(0.05, round(cfg_nom.film_height_um + (frac_span * delta_h_film_nm * 1e-3), 4))
        h_slab_fab = (
            max(0.0, min(h_film_fab - 0.03, round(cfg_nom.slab_height_um - 0.5 * frac_span * delta_h_film_nm * 1e-3, 4)))
            if cfg_nom.slab_height_um > 0.0
            else 0.0
        )
        sw_fab = float(np.clip(cfg_nom.sidewall_deg + frac_span * delta_sidewall_deg, 65.0, 90.0))

        # Sidewall scattering scales inversely with width (~ (W_nom / W_fab)^2 blended with bulk loss)
        width_ratio = cfg_nom.ring_width_um / max(w_ring_fab, 0.12)
        loss_scale = max(
            0.65,
            1.0 + (roughness_loss_penalty_pct_per_10nm / 100.0) * (-dcd_nm / 10.0) * min(width_ratio, 1.8),
        )
        loss_fab = round(base_loss_db_cm * loss_scale, 4)

        if idx == mid_idx and abs(delta_h_film_nm) < 1e-6 and abs(delta_sidewall_deg) < 1e-6:
            sim_i = nom_sim
        else:
            cfg_i = replace(
                cfg_nom,
                ring_width_um=w_ring_fab,
                bus_width_um=w_bus_fab,
                gap_thru_um=gap_sim_thru,
                gap_drop_um=gap_sim_drop,
                inter_ring_gap_um=inter_gap_sim,
                film_height_um=h_film_fab,
                slab_height_um=h_slab_fab,
                sidewall_deg=sw_fab,
                alpha_loss_db_per_cm=loss_fab,
            )
            sim_i = simulate_mrr_topology(cfg_i)

        mi = sim_i["metrics"]
        neff_i = float(mi["neff_ring"])
        ng_i = float(mi["ng_ring"])

        # Exact unwrapped azimuthal mode wavelength shift Δλ_res = λ_0 * Δn_eff / n_g [nm]
        delta_wl_res_nm = wl0_nm * (neff_i - neff_nom) / max(0.5 * (ng_i + ng_nom), 1.0)
        er_i = 0.5 if gap_bridged else float(mi["ER_thru_dB"])
        regime_str = "GAP_BRIDGED" if gap_bridged else str(mi["coupling_regime"])

        aspect_ratio = (h_film_fab - h_slab_fab) / max(gap_raw_thru, 0.010)
        status_flags: list[str] = []
        if gap_bridged:
            status_flags.append("GAP_BRIDGED (<45 nm)")
        elif gap_scum_risk:
            status_flags.append("NARROW_GAP (<85 nm)")
        if aspect_ratio > 4.2:
            status_flags.append(f"RIE_LAG (AR={aspect_ratio:.1f}:1)")
        if narrow_pinch_risk:
            status_flags.append("PINCH_OFF (<150 nm)")

        corners.append({
            "index": idx,
            "x_value": round(float(x_axis_vals[idx]), 3),
            "dose_error_pct": round(dose_pct, 2),
            "delta_cd_nm": round(dcd_nm, 2),
            "w_ring_fab_um": round(w_ring_fab, 4),
            "w_bus_fab_um": round(w_bus_fab, 4),
            "gap_fab_um": round(max(0.0, gap_raw_thru), 4),
            "h_film_fab_um": round(h_film_fab, 4),
            "neff_ring": round(neff_i, 6),
            "ng_ring": round(ng_i, 6),
            "delta_wl_res_nm": round(delta_wl_res_nm, 4),
            "fsr_nm": round(float(mi["FSR_nm"]), 4),
            "alpha_loss_db_per_cm": round(loss_fab, 4),
            "Q_loaded": round(float(mi["Q_loaded"]), 1),
            "Q_ext_thru": round(float(mi["Q_ext_thru"]), 1),
            "kappa_power_thru": round(float(mi["kappa_power_thru"]), 5),
            "extinction_ratio_dB": round(er_i, 2),
            "coupling_regime": regime_str,
            "gap_bridged": bool(gap_bridged),
            "drc_flags": status_flags,
        })

        if idx == 0:
            corner_spectra["min_corner"] = _extract_centered_dip_trace(sim_i)
        elif idx == mid_idx:
            corner_spectra["nominal"] = _extract_centered_dip_trace(sim_i)
        elif idx == n_pts - 1:
            corner_spectra["max_corner"] = _extract_centered_dip_trace(sim_i)

    # Extract Yield Windows (ER >= 10 dB and ER >= 15 dB)
    cd_arr = np.array([c["delta_cd_nm"] for c in corners], dtype=float)
    dose_arr = np.array([c["dose_error_pct"] for c in corners], dtype=float)
    er_arr = np.array([c["extinction_ratio_dB"] for c in corners], dtype=float)
    dwl_arr = np.array([c["delta_wl_res_nm"] for c in corners], dtype=float)

    cd_lo_10, cd_hi_10, cd_span_10 = _interp_window_1d(cd_arr, er_arr, 10.0)
    cd_lo_15, cd_hi_15, cd_span_15 = _interp_window_1d(cd_arr, er_arr, 15.0)
    dose_lo_10, dose_hi_10, dose_span_10 = _interp_window_1d(dose_arr, er_arr, 10.0)
    dose_lo_15, dose_hi_15, dose_span_15 = _interp_window_1d(dose_arr, er_arr, 15.0)

    # Sensitivity slopes via central finite difference around nominal
    dcd_step = cd_arr[mid_idx + 1] - cd_arr[mid_idx - 1]
    dwl_step_pm = (dwl_arr[mid_idx + 1] - dwl_arr[mid_idx - 1]) * 1e3
    dl_dcd_pm_per_nm = round(float(dwl_step_pm / dcd_step), 2) if abs(dcd_step) > 1e-6 else 0.0
    dl_ddose_pm_per_pct = round(float(cd_sign * slope * dl_dcd_pm_per_nm), 2)

    # Optimal CD mask pre-bias where ER peaks
    idx_best = int(np.argmax(er_arr))
    best_cd_offset_nm = float(cd_arr[idx_best])
    # If ER peaks at +10 nm fabricated shift, drawn mask should pre-add +10 nm W / -10 nm gap (or compensate expected fab drift)
    recommended_opc_prebias_nm = round(-0.5 * ((cd_lo_15 or 0.0) + (cd_hi_15 or 0.0)), 1) if cd_span_15 > 0 else 0.0

    # Heater ΔT needed to recover from a -10 nm blue-shift vs +10 nm red-shift (modulo FSR)
    shift_10nm_pm = abs(dl_dcd_pm_per_nm) * 10.0
    wrapped_shift_nm = (shift_10nm_pm * 1e-3) % max(fsr_nom_nm, 1e-4)
    if abs(dlambda_dt_pm_per_k) > 0.2:
        if dlambda_dt_pm_per_k > 0:
            dt_recover_blueshift_k = round((wrapped_shift_nm * 1e3) / dlambda_dt_pm_per_k, 1)
            dt_recover_redshift_k = round(((fsr_nom_nm - wrapped_shift_nm) * 1e3) / dlambda_dt_pm_per_k, 1)
        else:
            dt_recover_blueshift_k = round(((fsr_nom_nm - wrapped_shift_nm) * 1e3) / abs(dlambda_dt_pm_per_k), 1)
            dt_recover_redshift_k = round((wrapped_shift_nm * 1e3) / abs(dlambda_dt_pm_per_k), 1)
    else:
        dt_recover_blueshift_k = None
        dt_recover_redshift_k = None

    # Cleanroom DRC & Process Warnings
    for c in corners:
        if c["gap_bridged"]:
            drc_warnings.append(
                f"Gap bridges (<45 nm) at Dose={c['dose_error_pct']:+.1f}% (ΔCD={c['delta_cd_nm']:+.1f} nm) — bus and ring merge in RIE!"
            )
            break
    if not drc_warnings and any("NARROW_GAP (<85 nm)" in c["drc_flags"] for c in corners):
        drc_warnings.append(
            "Fabricated gap drops below 85 nm at high-CD corners: risk of EBL proximity scumming or incomplete RIE trench clearing."
        )
    if any(any(f.startswith("RIE_LAG") for f in c["drc_flags"]) for c in corners):
        drc_warnings.append(
            "Coupler trench aspect ratio exceeds 4.2:1 at narrow-gap corners: expect RIE lag (partial slab remaining inside gap)."
        )
    if cd_span_15 < 12.0:
        drc_warnings.append(
            f"Tight 15-dB CD yield window ({cd_span_15:.1f} nm): consider widening bus/ring gap with a Pulley or Racetrack coupler, or exporting a 5-cell Pre-Biased Bracket .GDS array."
        )

    under_corner = corners[0]       # Most negative ΔDose% (Under-Dose) or negative ΔCD
    nominal_corner = corners[mid_idx]
    over_corner = corners[-1]       # Most positive ΔDose% (Over-Dose) or positive ΔCD

    return {
        "resist_tone": tone_key,
        "resist_name_en": tone_meta["name_en"],
        "resist_name_zh": tone_meta["name_zh"],
        "overdose_mechanism_en": tone_meta["overdose_effect_en"],
        "underdose_mechanism_en": tone_meta["underdose_effect_en"],
        "overdose_mechanism_zh": tone_meta["overdose_effect_zh"],
        "underdose_mechanism_zh": tone_meta["underdose_effect_zh"],
        "x_label_en": x_label_en,
        "x_label_zh": x_label_zh,
        "dose_slope_nm_per_pct": round(slope, 3),
        "proximity_gap_factor": round(prox, 3),
        "summary": {
            "nominal_er_dB": nominal_corner["extinction_ratio_dB"],
            "nominal_regime": nominal_corner["coupling_regime"],
            "under_dose_corner": under_corner,
            "over_dose_corner": over_corner,
            "cd_window_er10_nm": [cd_lo_10, cd_hi_10],
            "cd_window_er10_span_nm": cd_span_10,
            "cd_window_er15_nm": [cd_lo_15, cd_hi_15],
            "cd_window_er15_span_nm": cd_span_15,
            "dose_window_er10_pct": [dose_lo_10, dose_hi_10],
            "dose_window_er10_span_pct": dose_span_10,
            "dose_window_er15_pct": [dose_lo_15, dose_hi_15],
            "dose_window_er15_span_pct": dose_span_15,
            "sensitivity_dl_dcd_pm_per_nm": dl_dcd_pm_per_nm,
            "sensitivity_dl_ddose_pm_per_pct": dl_ddose_pm_per_pct,
            "dlambda_dt_pm_per_k": round(dlambda_dt_pm_per_k, 2),
            "heater_dt_for_10nm_blueshift_k": dt_recover_blueshift_k,
            "heater_dt_for_10nm_redshift_k": dt_recover_redshift_k,
            "peak_er_at_cd_nm": round(best_cd_offset_nm, 2),
            "recommended_mask_prebias_nm": recommended_opc_prebias_nm,
        },
        "drc_warnings": drc_warnings,
        "corners": corners,
        "corner_spectra": corner_spectra,
    }


def export_fab_bias_array_gds(
    base_cfg: MRRDesignConfig,
    output_path: str | Path,
    *,
    cd_prebias_list_nm: list[float] | None = None,
    proximity_gap_factor: float = 1.0,
    pitch_x_um: float | None = None,
) -> dict:
    """Export a foundry/cleanroom Dose & CD-Bias Compensation MPW Test Array `.GDS` + `.LYP`.

    Each cell `k` pre-biases the drawn polygons by `δ_mask = cd_prebias_list_nm[k]`:
      - `W_ring_drawn = W_ring_nom + δ_mask`
      - `W_bus_drawn  = W_bus_nom  + δ_mask`
      - `gap_drawn    = gap_nom    - η * δ_mask`
    If cleanroom exposure produces a fabrication CD shift of `-δ_mask`, cell `k` lands at exact
    nominal `(W_nom, gap_nom)` and hits critical coupling on the first tape-out!
    """
    biases_nm = cd_prebias_list_nm if cd_prebias_list_nm else [-20.0, -10.0, 0.0, +10.0, +20.0]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell("MicroRing_Fab_CD_Bias_Array")

    default_span = 2.6 * base_cfg.ring_radius_um + base_cfg.straight_length_um + 60.0
    dx_pitch = float(pitch_x_um) if pitch_x_um is not None else max(default_span, 145.0)

    cells_meta: list[dict] = []
    for idx, bias_nm in enumerate(biases_nm):
        b_um = float(bias_nm) * 1e-3
        w_r = max(0.12, round(base_cfg.ring_width_um + b_um, 4))
        w_b = max(0.12, round(base_cfg.bus_width_um + b_um, 4))
        g_t = max(0.080, round(base_cfg.gap_thru_um - proximity_gap_factor * b_um, 4))
        g_d = max(0.080, round(base_cfg.gap_drop_um - proximity_gap_factor * b_um, 4))
        g_i = max(0.090, round(base_cfg.inter_ring_gap_um - proximity_gap_factor * b_um, 4))

        cfg_i = replace(
            base_cfg,
            ring_width_um=w_r,
            bus_width_um=w_b,
            gap_thru_um=g_t,
            gap_drop_um=g_d,
            inter_ring_gap_um=g_i,
        )
        tmp_path = out.parent / f"._tmp_fabbias_{idx}.gds"
        try:
            export_mrr_gds(cfg_i, tmp_path)
            sub_lib = gdstk.read_gds(str(tmp_path))
            sub_cell = sub_lib.top_level()[0]
            sign_tag = f"p{int(round(bias_nm))}nm" if bias_nm >= 0 else f"m{abs(int(round(bias_nm)))}nm"
            sub_cell.name = f"MRR_OPC_{idx+1}_{sign_tag}"
            lib.add(sub_cell)
            top.add(gdstk.Reference(sub_cell, origin=(idx * dx_pitch, 0.0)))
            top.add(gdstk.Label(
                f"#{idx+1} DRAWN_CD{bias_nm:+.0f}nm W={w_r*1e3:.0f}nm G={g_t*1e3:.0f}nm",
                (idx * dx_pitch, -(base_cfg.ring_radius_um + 23.0)),
                layer=66,
            ))
            cells_meta.append({
                "cell": sub_cell.name,
                "drawn_cd_bias_nm": round(float(bias_nm), 1),
                "compensates_fab_cd_nm": round(-float(bias_nm), 1),
                "w_ring_drawn_nm": round(w_r * 1e3, 1),
                "w_bus_drawn_nm": round(w_b * 1e3, 1),
                "gap_drawn_nm": round(g_t * 1e3, 1),
            })
        finally:
            tmp_path.unlink(missing_ok=True)

    lib.write_gds(str(out))
    lyp_path = write_klayout_layer_props(out.with_suffix(".lyp"))
    return {
        "gds_path": str(out.resolve()),
        "lyp_path": lyp_path,
        "num_cells": len(cells_meta),
        "cells": cells_meta,
    }


def run_foundry_process_corners(
    base_cfg: MRRDesignConfig,
    *,
    pdk_key: str = "gsip_soi220",
    delta_w_3sigma_nm: float | None = None,
    delta_h_core_3sigma_nm: float | None = None,
    delta_h_slab_3sigma_nm: float | None = None,
    delta_sidewall_3sigma_deg: float | None = None,
    delta_doping_3sigma_pct: float | None = None,
    proximity_gap_factor: float | None = None,
    include_active_pn: bool = False,
    mc_trials: int = 128,
    seed: int = 2026,
) -> dict:
    """Evaluate Multi-Parameter Foundry Process Corners (`Nominal/SS/FF/SF/FS`) + Monte-Carlo Yield.

    Simultaneously perturbs correlated `pattern_variations` (`dW` + anti-correlated coupler gap `-eta*dW`)
    and `stack_variations` (`dH_core`, `dH_slab`, `d_sidewall`, `d_doping`) using MicroRing Studio's
    native 2D Vector Maxwell FEM + Spatial CMT (+ optional 2D Poisson-Boltzmann PN TCAD) solvers.
    """
    from .foundry_pdk import get_foundry_pdk

    pdk = get_foundry_pdk(pdk_key)
    vc = pdk["variation_corners"]

    dw_3s = float(delta_w_3sigma_nm) if delta_w_3sigma_nm is not None else float(vc["delta_w_3sigma_nm"])
    dh_c_3s = float(delta_h_core_3sigma_nm) if delta_h_core_3sigma_nm is not None else float(vc["delta_h_core_3sigma_nm"])
    dh_s_3s = (
        (float(delta_h_slab_3sigma_nm) if delta_h_slab_3sigma_nm is not None else float(vc["delta_h_slab_3sigma_nm"]))
        if base_cfg.slab_height_um > 1e-4
        else 0.0
    )
    dsw_3s = float(delta_sidewall_3sigma_deg) if delta_sidewall_3sigma_deg is not None else float(vc.get("delta_sidewall_3sigma_deg", 1.5))
    ddop_3s = float(delta_doping_3sigma_pct) if delta_doping_3sigma_pct is not None else float(vc.get("delta_doping_3sigma_pct", 12.0))
    eta_gap = float(proximity_gap_factor) if proximity_gap_factor is not None else float(vc.get("proximity_gap_factor", 1.08))

    def _eval_corner(name: str, dw_nm: float, dhc_nm: float, dhs_nm: float, dsw_deg: float, ddop_pct: float) -> tuple[dict, dict]:
        dw_u = dw_nm * 1e-3
        dhc_u = dhc_nm * 1e-3
        dhs_u = dhs_nm * 1e-3
        h_new = max(0.08, base_cfg.film_height_um + dhc_u)
        s_new = float(np.clip(base_cfg.slab_height_um + dhs_u, 0.0, h_new - 0.04)) if base_cfg.slab_height_um > 1e-4 else 0.0
        wr_new = max(0.14, base_cfg.ring_width_um + dw_u)
        wb_new = max(0.14, base_cfg.bus_width_um + dw_u)
        gt_new = max(0.06, base_cfg.gap_thru_um - eta_gap * dw_u)
        gd_new = max(0.06, base_cfg.gap_drop_um - eta_gap * dw_u)
        sw_new = float(np.clip(base_cfg.sidewall_deg + dsw_deg, 60.0, 90.0))

        base_loss_db_cm = (
            float(base_cfg.alpha_loss_db_per_cm)
            if base_cfg.alpha_loss_db_per_cm is not None
            else float(MATERIAL_CATALOG.get(base_cfg.core_material, {}).get("typical_loss_db_cm", 2.0))
        )
        loss_use = base_loss_db_cm
        vpil_vcm = None
        if include_active_pn and base_cfg.core_material == "Si":
            from .multiphysics_tcad import PNDopingTCADConfig, simulate_pn_doping_tcad

            dop_scale = max(0.25, 1.0 + 0.01 * ddop_pct)
            pn_r = simulate_pn_doping_tcad(PNDopingTCADConfig(
                wavelength_nm=base_cfg.wavelength_nm,
                film_height_um=h_new,
                slab_height_um=max(s_new, 0.06),
                sidewall_deg=sw_new,
                waveguide_width_um=wr_new,
                na_core_cm3=5.0e17 * dop_scale,
                nd_core_cm3=5.0e17 * dop_scale,
                bias_voltages_v=[0.0, -2.0],
                map_bias_v=-2.0,
            ))
            vpil_vcm = pn_r["metrics"]["V_pi_L_Vcm"]
            loss_use = loss_use + 0.68 * float(pn_r["metrics"]["alpha_FCA_bias_dB_per_cm"])

        cfg_c = replace(
            base_cfg,
            film_height_um=h_new,
            slab_height_um=s_new,
            ring_width_um=wr_new,
            bus_width_um=wb_new,
            gap_thru_um=gt_new,
            gap_drop_um=gd_new,
            sidewall_deg=sw_new,
            alpha_loss_db_per_cm=loss_use,
        )
        sim = simulate_mrr_topology(cfg_c)
        mm = sim["metrics"]
        sp = sim["spectrum"]
        wl_arr = np.asarray(sp["wavelength_nm"], dtype=float)
        th_arr = np.asarray(sp["thru_dB"], dtype=float)
        idx_dip = int(np.argmin(th_arr))
        lam_dip = float(wl_arr[idx_dip])

        # Unwrapped resonance wavelength shift from effective index change relative to Nominal
        rec = {
            "corner": name,
            "delta_w_nm": round(dw_nm, 2),
            "delta_h_core_nm": round(dhc_nm, 2),
            "delta_h_slab_nm": round(dhs_nm, 2),
            "gap_fab_nm": round(gt_new * 1e3, 2),
            "neff_ring": mm["neff_ring"],
            "ng_ring": mm["ng_ring"],
            "FSR_GHz": mm["FSR_GHz"],
            "FSR_nm": mm["FSR_nm"],
            "Q_loaded": mm["Q_loaded"],
            "ER_thru_dB": mm["ER_thru_dB"],
            "kappa_power_thru": mm["kappa_power_thru"],
            "coupling_regime": mm["coupling_regime"],
            "dip_wavelength_nm": round(lam_dip, 5),
        }
        if vpil_vcm is not None:
            rec["V_pi_L_Vcm"] = vpil_vcm
        spec_small = {
            "wavelength_nm": sp["wavelength_nm"][::4],
            "thru_dB": sp["thru_dB"][::4],
        }
        return rec, spec_small

    # 1. Deterministic 5 Foundry Corners + 2 Pure-Axial Jacobian Probes (`+dW_only`, `+dH_core_only`)
    corner_specs = [
        ("Nominal", "Nominal Target", 0.0, 0.0, 0.0, 0.0, 0.0),
        ("SS", "Narrow & Thin (-3σ W, -3σ H)", -dw_3s, -dh_c_3s, -dh_s_3s, -dsw_3s, -ddop_3s),
        ("FF", "Wide & Thick (+3σ W, +3σ H)", +dw_3s, +dh_c_3s, +dh_s_3s, +dsw_3s, +ddop_3s),
        ("SF", "Wide & Thin (+3σ W, -3σ H)", +dw_3s, -dh_c_3s, -dh_s_3s, 0.0, 0.0),
        ("FS", "Narrow & Thick (-3σ W, +3σ H)", -dw_3s, +dh_c_3s, +dh_s_3s, 0.0, 0.0),
    ]

    corners_out: list[dict] = []
    corner_spectra: dict[str, dict] = {}
    for cname, cdesc, dw_i, dhc_i, dhs_i, dsw_i, ddop_i in corner_specs:
        r_i, sp_i = _eval_corner(cname, dw_i, dhc_i, dhs_i, dsw_i, ddop_i)
        r_i["corner_description"] = cdesc
        corners_out.append(r_i)
        corner_spectra[cname] = sp_i

    nom = corners_out[0]
    neff0 = float(nom["neff_ring"])
    ng0 = float(nom["ng_ring"])
    wl0_nm = float(base_cfg.wavelength_nm)

    for r_i in corners_out:
        dneff = float(r_i["neff_ring"]) - neff0
        dlam_nm = round((dneff / max(ng0, 1.0)) * wl0_nm, 4)
        r_i["delta_lambda_nm"] = dlam_nm
        r_i["unwrapped_dlambda_nm"] = dlam_nm

    # Pure axial FEM probes to separate dW vs dH_core vs dH_slab sensitivities cleanly
    r_w_pos, _ = _eval_corner("_probe_w", max(dw_3s, 5.0), 0.0, 0.0, 0.0, 0.0)
    r_hc_pos, _ = _eval_corner("_probe_hc", 0.0, max(dh_c_3s, 3.0), 0.0, 0.0, 0.0)
    dl_dw_pm_nm = ((float(r_w_pos["neff_ring"]) - neff0) / max(ng0, 1.0) * wl0_nm * 1e3) / max(dw_3s, 5.0)
    dl_dhc_pm_nm = ((float(r_hc_pos["neff_ring"]) - neff0) / max(ng0, 1.0) * wl0_nm * 1e3) / max(dh_c_3s, 3.0)
    if dh_s_3s > 0.1:
        r_hs_pos, _ = _eval_corner("_probe_hs", 0.0, 0.0, max(dh_s_3s, 3.0), 0.0, 0.0)
        dl_dhs_pm_nm = ((float(r_hs_pos["neff_ring"]) - neff0) / max(ng0, 1.0) * wl0_nm * 1e3) / max(dh_s_3s, 3.0)
    else:
        dl_dhs_pm_nm = 0.0

    # Coupling log-sensitivity `d(ln kappa^2)/dW` & `d(ln kappa^2)/dH_core` for fast calibrated Monte-Carlo
    kp0 = max(float(nom["kappa_power_thru"]), 1e-6)
    dlnkp_dw = math.log(max(float(r_w_pos["kappa_power_thru"]), 1e-6) / kp0) / max(dw_3s, 5.0)
    dlnkp_dhc = math.log(max(float(r_hc_pos["kappa_power_thru"]), 1e-6) / kp0) / max(dh_c_3s, 3.0)

    # 2. Correlated Gaussian Wafer Monte-Carlo (`n_mc` trials calibrated to the exact 2D-FEM sensitivities)
    n_mc = int(np.clip(mc_trials, 16, 2000))
    rng = np.random.default_rng(seed)
    dw_mc = rng.normal(0.0, dw_3s / 3.0, size=n_mc)
    dhc_mc = rng.normal(0.0, dh_c_3s / 3.0, size=n_mc)
    dhs_mc = rng.normal(0.0, dh_s_3s / 3.0, size=n_mc)

    dlam_mc_pm = dl_dw_pm_nm * dw_mc + dl_dhc_pm_nm * dhc_mc + dl_dhs_pm_nm * dhs_mc
    kp_mc = np.clip(kp0 * np.exp(dlnkp_dw * dw_mc + dlnkp_dhc * dhc_mc), 1e-5, 0.995)

    # Exact CMT extinction ratio & loaded Q across Monte-Carlo ensemble
    l_rt_m = (2.0 * math.pi * base_cfg.ring_radius_um + 2.0 * base_cfg.straight_length_um) * 1e-6
    nom_alpha_db_cm = (
        float(base_cfg.alpha_loss_db_per_cm)
        if base_cfg.alpha_loss_db_per_cm is not None
        else float(MATERIAL_CATALOG.get(base_cfg.core_material, {}).get("typical_loss_db_cm", 2.0))
    )
    alpha_m = nom_alpha_db_cm * 100.0 * math.log(10.0) / 10.0
    a_rt = math.exp(-0.5 * alpha_m * l_rt_m)
    tau_mc = np.sqrt(1.0 - kp_mc)
    is_4p = base_cfg.topology.endswith("_ad")
    tau2 = math.sqrt(max(1.0 - (float(nom["kappa_power_thru"]) if is_4p else 0.0), 0.0))
    t_min_amp = np.abs(tau_mc - a_rt * tau2) / np.maximum(1.0 - tau_mc * a_rt * tau2, 1e-6)
    er_mc_db = -20.0 * np.log10(np.maximum(t_min_amp, 1e-4))
    q_mc = (2.0 * math.pi * ng0 * l_rt_m / (wl0_nm * 1e-9)) / np.maximum(
        alpha_m * l_rt_m - np.log1p(-kp_mc) - (math.log(max(tau2 ** 2, 1e-6)) if is_4p else 0.0),
        1e-9,
    )

    sigma_lam_pm = float(np.std(dlam_mc_pm))
    fsr_pm = float(nom["FSR_nm"]) * 1e3
    fsr_frac_3s_pct = min(100.0, (3.0 * sigma_lam_pm) / max(fsr_pm, 1.0) * 100.0)
    # Estimate thermo-optic trim power needed to cover 3-sigma random resonant spread
    mat_c = get_material_fn(base_cfg.core_material)(wl0_nm * 1e-9, T_REF_K)
    dlam_dt_pm_k = abs(float(mat_c.dn_dt_per_k)) / max(ng0, 1.0) * wl0_nm * 1e3
    dt_trim_3s_k = (3.0 * sigma_lam_pm) / max(dlam_dt_pm_k, 0.5)

    return {
        "pdk_key": pdk_key,
        "pdk_name": pdk["name_en"],
        "solver_engine": "100% MicroRing Native 2D Vector Maxwell FEM + Spatial CMT + Calibrated Wafer MC",
        "variation_inputs_3sigma": {
            "delta_w_3sigma_nm": round(dw_3s, 2),
            "delta_h_core_3sigma_nm": round(dh_c_3s, 2),
            "delta_h_slab_3sigma_nm": round(dh_s_3s, 2),
            "delta_sidewall_3sigma_deg": round(dsw_3s, 2),
            "proximity_gap_factor": round(eta_gap, 3),
        },
        "sensitivities": {
            "dlambda_dW_pm_per_nm": round(dl_dw_pm_nm, 2),
            "dlambda_dH_core_pm_per_nm": round(dl_dhc_pm_nm, 2),
            "dlambda_dH_slab_pm_per_nm": round(dl_dhs_pm_nm, 2),
        },
        "monte_carlo_summary": {
            "trials": n_mc,
            "lambda_res_1sigma_pm": round(sigma_lam_pm, 2),
            "lambda_res_3sigma_nm": round(3.0 * sigma_lam_pm * 1e-3, 4),
            "fsr_3sigma_spread_pct": round(fsr_frac_3s_pct, 2),
            "heater_trim_3sigma_K": round(dt_trim_3s_k, 2),
            "yield_ER_gt_10dB_pct": round(float(np.mean(er_mc_db >= 10.0) * 100.0), 2),
            "yield_ER_gt_15dB_pct": round(float(np.mean(er_mc_db >= 15.0) * 100.0), 2),
            "ER_dB_p05_p50_p95": [round(float(v), 2) for v in np.percentile(er_mc_db, [5, 50, 95])],
            "Q_loaded_p05_p50_p95": [round(float(v), 0) for v in np.percentile(q_mc, [5, 50, 95])],
        },
        "corners": corners_out,
        "corner_spectra": corner_spectra,
    }

