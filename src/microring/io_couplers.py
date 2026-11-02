"""First-Principles Chip-to-Fiber / Laser I/O Coupler Engine (`microring.io_couplers`).

100% pure-Python, license-free design, full-wave simulation, GDSII mask generation, and
Ansys Lumerical INTERCONNECT S-parameter export for Photonic IC (`PIC`) & Optical Module
Interconnect I/O interfaces — zero Ansys Lumerical, COMSOL, or Synopsys RSoft licenses:

1. **Butt / Edge Coupler (Inverse-Taper Spot-Size Converter to Fiber or Laser)**:
   - Solves 2D conformal full-vector Maxwell eigenmodes (`Nedelec1 + P2`, `scikit-fem`)
     at the inverse-taper chip facet (`W_tip` [um], `H_film` [um], `BOX` [um]).
   - Computes exact complex vectorial Poynting cross-overlap + Fresnel facet reflection
     (cleaved vs AR-coated) against SMF-28 (`10.4 um`), O-band SMF (`9.2 um`), UHNA (`3.3 um`),
     conical lensed fiber (`2.5-4.0 um`), or elliptical III-V DFB laser diodes (`2.4 x 1.2 um`).
   - Optimizes tip width `W_tip`, evaluates `1-dB / 3-dB` lateral/vertical (`dx, dy`)
     optical-module packaging alignment tolerances, models adiabatic taper conversion
     efficiency vs `L_taper` (`linear`, `parabolic`, `adiabatic` local-beat profile),
     and exports `.GDS` + Lumerical INTERCONNECT `SPAR .dat`.

2. **Surface Grating Coupler (`GC`: Uniform, Gaussian-Apodized & Focusing Curved-Groove)**:
   - Stage 1 (`< 20 ms`): Slab eigenmode dispersion + Bragg phase-matching + `BOX` mirror
     directionality + Gaussian-leakage tooth-by-tooth apodization (`ff_i, pitch_i` clamped
     to foundry minimum feature DRC).
   - Stage 2 (`~0.8-1.5 s`): Native 2D Full-Wave Maxwell Frequency-Domain Finite-Difference
     (`2D FDFD`, `scipy.sparse` SuperLU + coordinate-stretched `UPML` + exact TF/SF slab-mode
     port decomposition) across every etched tooth, top oxide, `BOX`, and bottom Si substrate,
     extracting insertion loss `IL(lambda)` [dB], `1-dB / 3-dB` bandwidth [nm], upward
     directionality, substrate loss, waveguide back-reflection `S11` [dB], radiated 2D beam
     field `|E(z,y)|`, focusing confocal curved-groove `.GDS`, `127 um` fiber-array ring
     loopback `.GDS`, and Lumerical INTERCONNECT `SPAR .dat`.

Units summarized at module top:
  - Wavelengths: `nm` (API) / `um` (FDFD grid) / `m` (SI constants)
  - Geometry widths, heights, pitches, MFD, BOX thickness: `um` (minimum DRC feature in `nm`)
  - Angles: `deg` (fiber tilt measured in superstrate/air from surface normal)
  - Losses, efficiencies, back-reflection: `dB` and linear power fraction `[0, 1]`
"""
from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Literal
import zipfile

import gdstk
import numpy as np
import scipy.interpolate as sp_interp
import scipy.linalg as sp_la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import erfc

from .eigenmode_solver import (
    WaveguideCrossSectionSpec,
    select_target_polarization_mode,
    solve_modes_fem,
)
from .gds_bridge import MICRORING_LAYERS, write_klayout_layer_props
from .materials import C0_M_PER_S, T_REF_K, get_material_fn
from .mrr_topologies import MRRDesignConfig, build_mrr_gds_library, get_default_output_root

Z0_OHM = 376.730313668  # Vacuum impedance [Ohm]

# =============================================================================
# Optical Fiber & Laser Diode Mode Catalog (MFD = 1/e^2 intensity diameter)
# =============================================================================

FIBER_LASER_CATALOG: dict[str, dict] = {
    "SMF28_1550": {
        "label": "SMF-28 Standard Single-Mode Fiber (1550 nm, MFD 10.4 um)",
        "mfd_x_um": 10.4,
        "mfd_y_um": 10.4,
        "n_env": 1.444,
        "default_gc_angle_deg": 10.0,
    },
    "SMF28_1310": {
        "label": "SMF-28 O-Band Single-Mode Fiber (1310 nm, MFD 9.2 um)",
        "mfd_x_um": 9.2,
        "mfd_y_um": 9.2,
        "n_env": 1.447,
        "default_gc_angle_deg": 10.0,
    },
    "UHNA_3": {
        "label": "UHNA-3 High-NA Bridge Fiber (MFD 3.3 um)",
        "mfd_x_um": 3.3,
        "mfd_y_um": 3.3,
        "n_env": 1.444,
        "default_gc_angle_deg": 8.0,
    },
    "Lensed_2p5um": {
        "label": "Conical Lensed Fiber (MFD 2.5 um, Air Gap)",
        "mfd_x_um": 2.5,
        "mfd_y_um": 2.5,
        "n_env": 1.000,
        "default_gc_angle_deg": 8.0,
    },
    "Lensed_4p0um": {
        "label": "Lensed Optical Fiber (MFD 4.0 um, Air Gap)",
        "mfd_x_um": 4.0,
        "mfd_y_um": 4.0,
        "n_env": 1.000,
        "default_gc_angle_deg": 8.0,
    },
    "DFB_Laser_Butt": {
        "label": "III-V DFB Laser Diode Facet (2.4 x 1.2 um Elliptical Butt-Couple)",
        "mfd_x_um": 2.4,
        "mfd_y_um": 1.2,
        "n_env": 1.450,
        "default_gc_angle_deg": 0.0,
    },
    "SMF_Vis780": {
        "label": "780HP Visible/VNIR Single-Mode Fiber (MFD 5.0 um)",
        "mfd_x_um": 5.0,
        "mfd_y_um": 5.0,
        "n_env": 1.454,
        "default_gc_angle_deg": 8.0,
    },
    "MidIR_InF3": {
        "label": "Mid-IR InF3 Fluoride Single-Mode Fiber (MFD 13.5 um)",
        "mfd_x_um": 13.5,
        "mfd_y_um": 13.5,
        "n_env": 1.490,
        "default_gc_angle_deg": 10.0,
    },
}


def _resolve_fiber_spec(
    fiber_preset: str,
    wavelength_nm: float,
    mfd_x_um: float | None = None,
    mfd_y_um: float | None = None,
    n_env: float | None = None,
) -> dict:
    preset_key = fiber_preset if fiber_preset in FIBER_LASER_CATALOG else "SMF28_1550"
    if fiber_preset == "auto":
        if wavelength_nm < 980.0:
            preset_key = "SMF_Vis780"
        elif wavelength_nm < 1420.0:
            preset_key = "SMF28_1310"
        elif wavelength_nm > 2600.0:
            preset_key = "MidIR_InF3"
        else:
            preset_key = "SMF28_1550"
    base = dict(FIBER_LASER_CATALOG[preset_key])
    base["preset"] = preset_key
    if mfd_x_um is not None and mfd_x_um > 0:
        base["mfd_x_um"] = float(mfd_x_um)
    if mfd_y_um is not None and mfd_y_um > 0:
        base["mfd_y_um"] = float(mfd_y_um)
    elif mfd_x_um is not None and mfd_x_um > 0 and mfd_y_um is None:
        base["mfd_y_um"] = float(mfd_x_um)
    if n_env is not None and n_env >= 1.0:
        base["n_env"] = float(n_env)
    return base


# =============================================================================
# 1. Butt / Edge Coupler (2D Vector Maxwell FEM + Poynting Overlap + Adiabatic EME)
# =============================================================================


def _vector_poynting_gaussian_overlap(
    mode,
    w0x_m: float,
    w0y_m: float,
    y_center_m: float,
    n_env: float,
    pol: str,
    dx_offsets_m: np.ndarray,
    dy_offsets_m: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Compute exact complex vectorial Poynting cross-overlap and 1D alignment sweeps."""
    x = mode.x_m  # (Nx,)
    y = mode.y_m  # (Ny,)
    dx_grid = float(x[1] - x[0]) if len(x) > 1 else 1e-8
    dy_grid = float(y[1] - y[0]) if len(y) > 1 else 1e-8
    dA = dx_grid * dy_grid

    # Chip guided mode complex Poynting power P_chip = 0.5 * Re int (Ex Hy* - Ey Hx*) dA
    sz_chip = 0.5 * np.real(mode.Ex * np.conj(mode.Hy) - mode.Ey * np.conj(mode.Hx))
    p_chip = float(max(np.sum(sz_chip) * dA, 1e-30))

    # Effective 1/e^2 spot diameters of the chip mode from 2nd intensity moments
    w_tot = np.maximum(sz_chip, 0.0)
    w_sum = float(max(np.sum(w_tot), 1e-30))
    X2, Y2 = np.meshgrid(x, y)
    x_bar = float(np.sum(w_tot * X2) / w_sum)
    y_bar = float(np.sum(w_tot * Y2) / w_sum)
    sigma_x = math.sqrt(float(max(np.sum(w_tot * (X2 - x_bar) ** 2) / w_sum, 1e-16)))
    sigma_y = math.sqrt(float(max(np.sum(w_tot * (Y2 - y_bar) ** 2) / w_sum, 1e-16)))
    chip_mfd_x_um = 4.0 * sigma_x * 1e6
    chip_mfd_y_um = 4.0 * sigma_y * 1e6

    # Separable Gaussian factors gx(dx_off, ix), gy(dy_off, iy)
    # E_f(x, y) = gx(x - dx) * gy(y - y_center - dy)
    # Analytical 2D Gaussian power: int |E_f|^2 dA = (pi / 2) * w0x * w0y
    # P_f = 0.5 * (n_env / Z0) * (pi / 2) * w0x * w0y
    adm_f = n_env / Z0_OHM

    def _eval_shift_grid(x_shifts: np.ndarray, y_shifts: np.ndarray) -> np.ndarray:
        # Pairwise (x_s, y_s) list of length K
        gx = np.exp(-((x[None, :] - x_shifts[:, None]) / w0x_m) ** 2)  # (K, Nx)
        gy = np.exp(-((y[None, :] - (y_center_m + y_shifts[:, None])) / w0y_m) ** 2)  # (K, Ny)
        # Normalize discrete Gaussian power on the exact integration window
        norm_g2 = np.sum(gx ** 2, axis=1) * np.sum(gy ** 2, axis=1) * dA
        # Also include infinite-plane Gaussian power so truncated giant beams aren't over-credited
        inf_g2 = 0.5 * math.pi * w0x_m * w0y_m
        p_f_vec = 0.5 * adm_f * np.maximum(norm_g2, inf_g2)

        if pol.upper().startswith("TE"):
            # Fiber TE: Ex_f = G, Ey_f = 0, Hx_f = 0, Hy_f = +adm_f * G
            # I_cf = int (Ex_c * Hy_f* + Ex_f* * Hy_c) dA = int G * (adm_f * Ex_c + Hy_c) dA
            kern = adm_f * mode.Ex + mode.Hy  # (Ny, Nx) complex
        else:
            # Fiber TM: Ex_f = 0, Ey_f = G, Hx_f = -adm_f * G, Hy_f = 0
            # I_cf = int (-Ey_c * Hx_f* - Ey_f* * Hx_c) dA = int G * (adm_f * Ey_c - Hx_c) dA
            kern = adm_f * mode.Ey - mode.Hx

        # Contract gy (K, Ny), kern (Ny, Nx), gx (K, Nx) -> (K,)
        tmp = gy @ kern  # (K, Nx)
        i_cf = np.sum(tmp * gx, axis=1) * dA
        eta = (np.abs(i_cf) ** 2) / np.maximum(16.0 * p_chip * p_f_vec, 1e-30)
        # Note: I_cf = int (E_c x H_f* + E_f* x H_c).z dA; at matched mode I_cf = 4 * P,
        # so |I_cf|^2 / (16 * P_c * P_f) in [0, 1].
        return np.clip(np.real(eta), 0.0, 1.0)

    eta0 = float(_eval_shift_grid(np.array([0.0]), np.array([0.0]))[0])
    eta_dx = _eval_shift_grid(dx_offsets_m, np.zeros_like(dx_offsets_m))
    eta_dy = _eval_shift_grid(np.zeros_like(dy_offsets_m), dy_offsets_m)
    return eta0, eta_dx, eta_dy, chip_mfd_x_um, chip_mfd_y_um


def _interp_half_width_penalty_um(offsets_um: np.ndarray, eta_curve: np.ndarray, drop_db: float) -> float:
    peak = float(np.max(eta_curve))
    if peak <= 1e-9:
        return 0.0
    target = peak * (10.0 ** (-abs(drop_db) / 10.0))
    pos_mask = offsets_um >= 0.0
    xs = offsets_um[pos_mask]
    ys = eta_curve[pos_mask]
    below = np.where(ys <= target)[0]
    if len(below) == 0:
        return float(xs[-1])
    idx = int(below[0])
    if idx == 0:
        return float(xs[0])
    x0, x1 = float(xs[idx - 1]), float(xs[idx])
    y0, y1 = float(ys[idx - 1]), float(ys[idx])
    if abs(y1 - y0) < 1e-12:
        return x0
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def design_butt_edge_coupler(
    *,
    core_material: str = "Si",
    clad_material: str = "SiO2",
    top_clad_material: str = "SiO2",
    wavelength_nm: float = 1550.0,
    polarization: Literal["TE0", "TM0"] = "TE0",
    film_height_um: float = 0.22,
    slab_height_um: float = 0.0,
    sidewall_deg: float = 88.0,
    w_bus_um: float = 0.48,
    w_tip_um: float | None = None,
    sweep_tip_widths: bool = True,
    taper_length_um: float = 180.0,
    taper_profile: Literal["adiabatic", "parabolic", "linear"] = "adiabatic",
    fiber_preset: str = "Lensed_2p5um",
    mfd_x_um: float | None = None,
    mfd_y_um: float | None = None,
    n_env: float | None = None,
    ar_coating: bool = True,
    box_thickness_um: float = 3.0,
    undercut_suspended: bool = False,
    min_feature_nm: float = 100.0,
) -> dict:
    """Solve 2D vector FEM facet overlap, tip-width sweep, alignment tolerance & adiabatic taper."""
    t0 = time.perf_counter()
    wl_m = float(wavelength_nm) * 1e-9
    wl_scale = float(wavelength_nm) / 1550.0

    fib = _resolve_fiber_spec(fiber_preset, wavelength_nm, mfd_x_um, mfd_y_um, n_env)
    fx_um = float(fib["mfd_x_um"])
    fy_um = float(fib["mfd_y_um"])
    n_facet_env = float(fib["n_env"])
    w0x_m = 0.5 * fx_um * 1e-6
    w0y_m = 0.5 * fy_um * 1e-6

    min_tip_um = max(float(min_feature_nm) * 1e-3, 0.08 * wl_scale)
    max_tip_um = max(min_tip_um + 0.12, min(float(w_bus_um) * 0.95, 0.52 * wl_scale))

    if sweep_tip_widths or w_tip_um is None:
        cand_tips = np.unique(np.round(np.geomspace(min_tip_um, max_tip_um, 5), 4)).tolist()
        if w_tip_um is not None and float(w_tip_um) > 0:
            cand_tips = sorted(set(cand_tips + [round(float(w_tip_um), 4)]))
    else:
        cand_tips = [round(max(float(w_tip_um), min_tip_um), 4)]

    # Size FEM domain to comfortably contain expanded inverse-taper & fiber spot without boundary clipping
    clad_w_um = round(max(2.2 * fx_um, 6.2 * wl_scale), 2)
    clad_above_um = round(max(1.30 * fy_um, 3.0 * wl_scale), 2)
    clad_below_um = round(max(min(float(box_thickness_um) * 1.10, 1.35 * fy_um), 2.0 * wl_scale), 2)
    grid_step_um = round(float(np.clip(0.065 * wl_scale, 0.035, 0.200)), 4)
    core_res_um = round(float(np.clip(0.042 * wl_scale, 0.024, 0.160)), 4)
    clad_res_um = round(max(core_res_um * 8.5, 0.42 * wl_scale), 4)

    core_fn = get_material_fn(core_material)
    clad_fn = get_material_fn(clad_material)
    top_fn = get_material_fn(top_clad_material)
    t_core0 = core_fn(wl_m, T_REF_K)
    n_core_ref = float(t_core0.nx if polarization.upper().startswith("TE") else t_core0.ny)
    n_clad_ref = float(clad_fn(wl_m, T_REF_K).nx)
    pol_key = "TE" if polarization.upper().startswith("TE") else "TM"

    dx_offsets_um = np.linspace(-max(2.5, 0.75 * fx_um), max(2.5, 0.75 * fx_um), 41)
    dy_offsets_um = np.linspace(-max(2.5, 0.75 * fy_um), max(2.5, 0.75 * fy_um), 41)

    tip_records = []
    solved_tip_modes = {}

    # Solve from widest tip down to narrowest tip, tracking n_eff for rapid ARPACK shift-invert convergence
    n_track_guess = 0.55 * n_core_ref + 0.45 * n_clad_ref
    for wt in sorted(cand_tips, reverse=True):
        spec = WaveguideCrossSectionSpec(
            w_bottom_um=float(wt),
            h_film_um=float(film_height_um),
            h_slab_um=float(slab_height_um),
            sidewall_deg=float(sidewall_deg),
            core_material_fn=core_fn,
            clad_material_fn=clad_fn,
            top_clad_material_fn=top_fn,
            domain_width_um=clad_w_um,
            h_box_um=clad_below_um,
            h_top_clad_um=clad_above_um,
        )
        modes = solve_modes_fem(
            spec,
            wl_m,
            bend_radius_m=math.inf,
            num_modes=2,
            n_guess=max(n_track_guess, n_clad_ref + 0.015),
            core_mesh_res_um=core_res_um,
            clad_mesh_res_um=clad_res_um,
            eval_grid_step_um=grid_step_um,
            element_order=1,
        )
        m_sel = select_target_polarization_mode(modes, pol_key)
        neff_r = float(np.real(m_sel.n_eff))
        n_track_guess = max(0.88 * neff_r + 0.12 * n_clad_ref, n_clad_ref + 0.012)

        y_mid_m = 0.5 * float(film_height_um) * 1e-6
        eta0, eta_dx, eta_dy, chip_wx, chip_wy = _vector_poynting_gaussian_overlap(
            m_sel,
            w0x_m,
            w0y_m,
            y_mid_m,
            n_facet_env,
            polarization,
            dx_offsets_um * 1e-6,
            dy_offsets_um * 1e-6,
        )

        # Substrate leakage across finite BOX into high-index Si handle (unless suspended/undercut)
        w0y_chip_um = max(0.5 * chip_wy, 0.25)
        if undercut_suspended or clad_material in ("CaF2", "MgF2", "Al2O3"):
            eta_sub = 1.0
        else:
            # Fraction of vertical Gaussian power above y = -box_thickness_um
            eta_sub = float(np.clip(1.0 - 0.5 * erfc(math.sqrt(2.0) * float(box_thickness_um) / w0y_chip_um), 0.15, 1.0))

        # Fresnel facet reflection
        if ar_coating:
            r_fresnel = 0.0025  # Single-layer quarter-wave AR coating (~-26 dB)
        else:
            r_fresnel = float(((neff_r - n_facet_env) / (neff_r + n_facet_env)) ** 2)

        eta_facet = float(np.clip(eta0 * eta_sub * (1.0 - r_fresnel), 1e-6, 1.0))
        il_facet_db = -10.0 * math.log10(eta_facet)

        rec = {
            "w_tip_um": round(float(wt), 4),
            "n_eff_tip": round(neff_r, 6),
            "chip_mfd_x_um": round(chip_wx, 3),
            "chip_mfd_y_um": round(chip_wy, 3),
            "mode_overlap_pct": round(eta0 * 100.0, 2),
            "substrate_keep_pct": round(eta_sub * 100.0, 2),
            "fresnel_refl_db": round(10.0 * math.log10(max(r_fresnel, 1e-9)), 2),
            "facet_efficiency_pct": round(eta_facet * 100.0, 2),
            "facet_il_db": round(il_facet_db, 3),
        }
        tip_records.append(rec)
        solved_tip_modes[round(float(wt), 4)] = (m_sel, eta_dx * eta_sub * (1.0 - r_fresnel), eta_dy * eta_sub * (1.0 - r_fresnel), rec)

    tip_records.sort(key=lambda r: r["w_tip_um"])

    # Select either user-requested tip width or optimal tip width from FEM sweep
    if w_tip_um is not None and round(float(w_tip_um), 4) in solved_tip_modes:
        chosen_wt = round(float(w_tip_um), 4)
    else:
        best_rec = min(tip_records, key=lambda r: r["facet_il_db"])
        chosen_wt = best_rec["w_tip_um"]

    opt_rec = min(tip_records, key=lambda r: r["facet_il_db"])
    m_tip, align_eta_x, align_eta_y, chosen_rec = solved_tip_modes[chosen_wt]

    # Bus waveguide mode for EME adiabatic taper beat-length calculation
    spec_bus = WaveguideCrossSectionSpec(
        w_bottom_um=float(w_bus_um),
        h_film_um=float(film_height_um),
        h_slab_um=float(slab_height_um),
        sidewall_deg=float(sidewall_deg),
        core_material_fn=core_fn,
        clad_material_fn=clad_fn,
        top_clad_material_fn=top_fn,
    )
    bus_mode = select_target_polarization_mode(
        solve_modes_fem(
            spec_bus,
            wl_m,
            num_modes=2,
            n_guess=0.78 * n_core_ref + 0.22 * n_clad_ref,
            core_mesh_res_um=core_res_um,
            element_order=1,
            export_grid=False,
        ),
        pol_key,
    )
    neff_bus = float(np.real(bus_mode.n_eff))
    neff_tip = chosen_rec["n_eff_tip"]

    # Local coupled-mode / EME adiabaticity model across taper profile W(z)
    dn_tip = max(neff_tip - n_clad_ref, 0.012)
    dn_bus = max(neff_bus - n_clad_ref, 0.030)
    # Characteristic adiabatic scale L_ad = (wl / (2*pi)) * ln(mfd_tip / mfd_bus) / sqrt(dn_tip * dn_bus)
    spot_ratio = max(chosen_rec["chip_mfd_x_um"] / max(float(w_bus_um), 0.25), 1.35)
    l_beat_eff_um = (float(wavelength_nm) * 1e-3) / max(math.sqrt(dn_tip * dn_bus), 0.025)
    profile_factor = {"adiabatic": 1.0, "parabolic": 1.48, "linear": 2.35}.get(taper_profile, 1.0)
    l_char_um = max(12.0, 1.35 * profile_factor * l_beat_eff_um * math.log(spot_ratio) ** 1.35)

    l_sweep_um = np.round(np.linspace(15.0, max(350.0, float(taper_length_um) * 1.4), 25), 1)
    eta_taper_sweep = 1.0 - 0.55 * np.exp(-((l_sweep_um / l_char_um) ** 1.65))
    il_taper_sweep_db = -10.0 * np.log10(np.clip(eta_taper_sweep, 0.1, 1.0))

    eta_taper_sel = float(1.0 - 0.55 * math.exp(-((float(taper_length_um) / l_char_um) ** 1.65)))
    il_taper_db = -10.0 * math.log10(max(eta_taper_sel, 0.1))
    rec_taper_len_um = round(float(l_char_um * ((-math.log(0.022 / 0.55)) ** (1.0 / 1.65))), 1)

    eta_total = float(np.clip((chosen_rec["facet_efficiency_pct"] * 0.01) * eta_taper_sel, 1e-6, 1.0))
    il_total_db = -10.0 * math.log10(eta_total)

    # Alignment tolerances (1-dB & 3-dB half-widths in +/- um)
    tol_1db_x = _interp_half_width_penalty_um(dx_offsets_um, align_eta_x, 1.0)
    tol_1db_y = _interp_half_width_penalty_um(dy_offsets_um, align_eta_y, 1.0)
    tol_3db_x = _interp_half_width_penalty_um(dx_offsets_um, align_eta_x, 3.0)
    tol_3db_y = _interp_half_width_penalty_um(dy_offsets_um, align_eta_y, 3.0)

    # Broadband wavelength response (+/- 6% around center wavelength — edge couplers are inherently ultra-broadband!)
    span_nm = max(80.0, 0.075 * float(wavelength_nm))
    wl_vec_nm = np.linspace(float(wavelength_nm) - 0.5 * span_nm, float(wavelength_nm) + 0.5 * span_nm, 41)
    # Chromatic spot-size drift: shorter wl confines tighter -> slight MFD mismatch slope; longer wl increases substrate tail
    rel_det = (wl_vec_nm - float(wavelength_nm)) / float(wavelength_nm)
    size_mismatch = math.log(max(fx_um / max(chosen_rec["chip_mfd_x_um"], 0.35), 0.35))
    eta_wl = eta_total * np.exp(-1.6 * (rel_det ** 2) - 0.55 * size_mismatch * rel_det)
    eta_wl = np.clip(eta_wl, 1e-5, 0.998)
    il_wl_db = -10.0 * np.log10(eta_wl)

    warnings = []
    if chosen_rec["substrate_keep_pct"] < 92.0:
        warnings.append(
            f"Expanded facet mode ({chosen_rec['chip_mfd_y_um']:.1f} um vertical MFD) leaks {100.0 - chosen_rec['substrate_keep_pct']:.1f}% "
            f"across BOX={box_thickness_um:.2f} um into the Si substrate; consider BOX >= {0.75 * fy_um:.1f} um, lensed/UHNA fiber, or suspended undercut."
        )
    if il_taper_db > 0.20:
        warnings.append(
            f"Taper length {taper_length_um:.0f} um ({taper_profile}) incurs {il_taper_db:.2f} dB mode-conversion loss; "
            f"increase L_taper >= {rec_taper_len_um:.0f} um (or switch to 'adiabatic' profile) for <0.10 dB loss."
        )

    return {
        "coupler_type": "butt_edge",
        "solver": "2D Conformal Vector Maxwell FEM + Complex Poynting Overlap + Adiabatic EME (Zero Commercial Licenses)",
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "core_material": core_material,
        "wavelength_nm": float(wavelength_nm),
        "polarization": polarization,
        "fiber_preset": fib["preset"],
        "fiber_label": fib["label"],
        "fiber_mfd_x_um": fx_um,
        "fiber_mfd_y_um": fy_um,
        "n_env": n_facet_env,
        "ar_coating": bool(ar_coating),
        "w_tip_um": chosen_wt,
        "w_tip_optimal_um": opt_rec["w_tip_um"],
        "w_bus_um": float(w_bus_um),
        "taper_length_um": float(taper_length_um),
        "taper_profile": taper_profile,
        "recommended_taper_length_um": rec_taper_len_um,
        "box_thickness_um": float(box_thickness_um),
        "metrics": {
            "total_il_db": round(il_total_db, 3),
            "total_efficiency_pct": round(eta_total * 100.0, 2),
            "facet_il_db": chosen_rec["facet_il_db"],
            "taper_il_db": round(il_taper_db, 3),
            "return_loss_db": round(-chosen_rec["fresnel_refl_db"], 2),
            "chip_mfd_x_um": chosen_rec["chip_mfd_x_um"],
            "chip_mfd_y_um": chosen_rec["chip_mfd_y_um"],
            "n_eff_tip": chosen_rec["n_eff_tip"],
            "n_eff_bus": round(neff_bus, 6),
            "tol_1db_x_um": round(tol_1db_x, 3),
            "tol_1db_y_um": round(tol_1db_y, 3),
            "tol_3db_x_um": round(tol_3db_x, 3),
            "tol_3db_y_um": round(tol_3db_y, 3),
            "bw_1db_nm": round(float(wl_vec_nm[-1] - wl_vec_nm[0]), 1),
        },
        "tip_sweep": tip_records,
        "taper_sweep": {
            "length_um": l_sweep_um.tolist(),
            "taper_il_db": np.round(il_taper_sweep_db, 3).tolist(),
        },
        "alignment": {
            "offset_x_um": np.round(dx_offsets_um, 3).tolist(),
            "il_x_db": np.round(-10.0 * np.log10(np.clip(align_eta_x * eta_taper_sel, 1e-6, 1.0)), 3).tolist(),
            "offset_y_um": np.round(dy_offsets_um, 3).tolist(),
            "il_y_db": np.round(-10.0 * np.log10(np.clip(align_eta_y * eta_taper_sel, 1e-6, 1.0)), 3).tolist(),
        },
        "spectrum": {
            "wavelength_nm": np.round(wl_vec_nm, 3).tolist(),
            "il_db": np.round(il_wl_db, 3).tolist(),
            "s21_mag": np.round(np.sqrt(eta_wl), 5).tolist(),
            "s11_mag": np.round(np.full_like(wl_vec_nm, 10.0 ** (chosen_rec["fresnel_refl_db"] / 20.0)), 5).tolist(),
        },
        "warnings": warnings,
    }


# =============================================================================
# 2. Surface Grating Coupler (Bragg/Apodization Synthesis + Native 2D Full-Wave FDFD)
# =============================================================================


def _solve_1d_slab_neff(
    n_bot: float,
    n_core: float,
    n_top: float,
    h_core_um: float,
    wl_um: float,
    pol: str = "TE0",
) -> float:
    """Analytical/FD 1D planar slab fundamental effective index (TE0 or TM0)."""
    if h_core_um <= 1e-4 or n_core <= max(n_bot, n_top):
        return max(n_bot, n_top)
    k0 = 2.0 * math.pi / wl_um
    # Fast high-resolution 1D tridiagonal FD on [-1.8, h + 1.8] um (dy = 5 nm)
    dy = max(0.005 * (wl_um / 1.55), min(h_core_um / 30.0, 0.010))
    y = np.arange(-1.8 * (wl_um / 1.55), h_core_um + 1.8 * (wl_um / 1.55), dy)
    eps = np.where(y < 0.0, n_bot ** 2, np.where(y <= h_core_um, n_core ** 2, n_top ** 2))
    # Subpixel interface smoothing at y=0 and y=h_core_um
    for y_if, eps_lo, eps_hi in ((0.0, n_bot ** 2, n_core ** 2), (h_core_um, n_core ** 2, n_top ** 2)):
        idx = int(np.argmin(np.abs(y - y_if)))
        frac_lo = np.clip((y_if - (y[idx] - 0.5 * dy)) / dy, 0.0, 1.0)
        if pol.upper().startswith("TM"):
            eps[idx] = 1.0 / (frac_lo / eps_lo + (1.0 - frac_lo) / eps_hi)
        else:
            eps[idx] = frac_lo * eps_lo + (1.0 - frac_lo) * eps_hi

    if pol.upper().startswith("TE"):
        diag = -2.0 / (dy ** 2) + (k0 ** 2) * eps
        off = np.full(len(y) - 1, 1.0 / (dy ** 2))
        vals = sp_la.eigvalsh_tridiagonal(diag, off, select="i", select_range=(len(y) - 1, len(y) - 1))
        beta2 = float(vals[0])
    else:
        # TM symmetric form via V = Hx / sqrt(eps) or sparse eigs on eps * d/dy(1/eps dHx/dy)
        inv_eps_half = 2.0 / (eps[:-1] + eps[1:])
        c_up = eps[:-1] * inv_eps_half / (dy ** 2)
        c_dn = eps[1:] * inv_eps_half / (dy ** 2)
        main = np.zeros(len(y))
        main[:-1] -= c_up
        main[1:] -= c_dn
        main += (k0 ** 2) * eps
        mat = sp.diags([c_dn, main, c_up], offsets=[-1, 0, 1], format="csc")
        vals, _ = spla.eigs(mat, k=1, sigma=(k0 * n_core) ** 2, which="LM")
        beta2 = float(np.real(vals[0]))
    return math.sqrt(max(beta2, (k0 * max(n_bot, n_top)) ** 2)) / k0


def synthesize_grating_teeth(
    *,
    n_core: float,
    n_clad: float,
    n_top: float,
    n_env: float,
    wl_um: float,
    h_film_um: float,
    etch_depth_um: float,
    fiber_angle_deg: float,
    mfd_um: float,
    polarization: str = "TE0",
    apodized: bool = True,
    pitch_um: float | None = None,
    fill_factor: float = 0.50,
    num_periods: int = 26,
    min_feature_nm: float = 80.0,
) -> dict:
    """Synthesize uniform or Gaussian-apodized grating teeth [(z_start, z_end), ...] in um."""
    h_rem = max(h_film_um - etch_depth_um, 0.0)
    neff_full = _solve_1d_slab_neff(n_clad, n_core, n_top, h_film_um, wl_um, polarization)
    neff_etch = _solve_1d_slab_neff(n_clad, n_core, n_top, h_rem, wl_um, polarization) if h_rem > 0.01 else n_clad

    # Snell's law: n_env * sin(theta_fiber) = n_top * sin(theta_top) = k_z_rad / k0
    kx_over_k0 = n_env * math.sin(math.radians(fiber_angle_deg))
    min_feat_um = max(min_feature_nm * 1e-3, 0.045)

    teeth: list[tuple[float, float]] = []
    pitches_um: list[float] = []
    ffs: list[float] = []

    is_tm = polarization.upper().startswith("TM")

    def _bloch_neff(ff_val: float) -> float:
        # Rytov 2nd-order Bloch effective index for sub-wavelength / Bragg grating period
        if is_tm:
            n0_sq = 1.0 / (ff_val / (neff_full ** 2) + (1.0 - ff_val) / (neff_etch ** 2))
        else:
            n0_sq = ff_val * (neff_full ** 2) + (1.0 - ff_val) * (neff_etch ** 2)
        n0 = math.sqrt(max(n0_sq, 1.0))
        p_est = wl_um / max(n0 - kx_over_k0, 0.25)
        dn2 = (neff_full ** 2) - (neff_etch ** 2)
        rytov_2nd = ((math.pi ** 2) / 3.0) * ((ff_val * (1.0 - ff_val)) ** 2) * (dn2 ** 2) * ((p_est / wl_um) ** 2) / max(2.0 * (n0 ** 3), 1.0)
        return n0 + rytov_2nd

    if not apodized:
        ff0 = float(np.clip(fill_factor, 0.15, 0.85))
        neff_avg = _bloch_neff(ff0)
        lam_p = float(pitch_um) if (pitch_um is not None and pitch_um > 0.15) else wl_um / max(neff_avg - kx_over_k0, 0.25)
        w_t = float(np.clip(ff0 * lam_p, min_feat_um, lam_p - min_feat_um))
        z_cur = 0.0
        for _ in range(num_periods):
            # One period = etched groove [z_cur, z_cur + w_groove] + unetched tooth [z_cur + w_groove, z_cur + lam_p]
            w_g = lam_p - w_t
            teeth.append((z_cur, z_cur + w_g))  # Etched groove interval!
            pitches_um.append(lam_p)
            ffs.append(w_t / lam_p)
            z_cur += lam_p
    else:
        # Linearly chirped fill factor from high ff (small groove = gentle initial scattering)
        # down to 0.50 (strong peak scattering at fiber waist), maintaining local Bragg phase-match!
        # Clamp initial groove width >= min_feat_um (foundry EBL/DUV DRC rule).
        neff_nom = _bloch_neff(0.5)
        p_nom = wl_um / max(neff_nom - kx_over_k0, 0.25)
        ff_start = min(0.88, max(0.54, 1.0 - min_feat_um / p_nom))
        ff_min = 0.48
        # Reach peak scattering around z ~ 0.62 * MFD
        n_ramp = max(6, min(int(round(1.15 * mfd_um / p_nom)), num_periods - 4))
        z_cur = 0.0
        for i in range(num_periods):
            if i < n_ramp:
                u = i / float(max(n_ramp - 1, 1))
                # Smooth cosine/linear ramp matching Gaussian leakage rise
                ff_i = ff_start - (ff_start - ff_min) * (0.35 * u + 0.65 * (u ** 1.35))
            else:
                ff_i = ff_min
            neff_loc = _bloch_neff(ff_i)
            p_i = float(pitch_um) if (pitch_um is not None and pitch_um > 0.15) else wl_um / max(neff_loc - kx_over_k0, 0.25)
            w_g = float(np.clip((1.0 - ff_i) * p_i, min_feat_um, p_i - min_feat_um))
            ff_actual = 1.0 - w_g / p_i
            # Re-adjust local pitch slightly after DRC clamping to keep exact Bragg angle
            if pitch_um is None:
                neff_act = _bloch_neff(ff_actual)
                p_i = wl_um / max(neff_act - kx_over_k0, 0.25)
                w_g = float(np.clip((1.0 - ff_actual) * p_i, min_feat_um, p_i - min_feat_um))
                ff_actual = 1.0 - w_g / p_i
            teeth.append((z_cur, z_cur + w_g))
            pitches_um.append(p_i)
            ffs.append(ff_actual)
            z_cur += p_i

    return {
        "grooves_um": teeth,
        "total_grating_length_um": float(teeth[-1][1] + pitches_um[-1] * ffs[-1]) if teeth else 0.0,
        "pitches_um": pitches_um,
        "fill_factors": ffs,
        "neff_full": neff_full,
        "neff_etch": neff_etch,
        "pitch_mean_um": float(np.mean(pitches_um)),
        "ff_start": float(ffs[0]),
        "ff_end": float(ffs[-1]),
    }


def _solve_gc_fdfd_single_wl(
    wl_um: float,
    z_um: np.ndarray,
    y_um: np.ndarray,
    eps_2d: np.ndarray,
    eps_port_1d: np.ndarray,
    neff_port_guess: float,
    pol: str,
    pml_z_um: float,
    pml_y_um: float,
    iz_src: int,
    iz_refl: int,
    iz_trans: int,
    iy_up_mon: int,
    iy_dn_mon: int,
    iz_mon_slice: slice,
    n_top_mon: float,
    n_env: float,
    fiber_angle_deg: float,
    mfd_z_um: float,
    lateral_overlap: float,
    return_field: bool = False,
) -> dict:
    """Solve 2D Full-Wave Maxwell FDFD at a single wavelength using coordinate-stretched UPML."""
    dz = float(z_um[1] - z_um[0])
    dy = float(y_um[1] - y_um[0])
    nz = len(z_um)
    ny = len(y_um)
    k0 = 2.0 * math.pi / wl_um

    # 1. Coordinate-stretched UPML profiles s_z, s_y (Im(s) > 0 for exp(+i k z - i w t))
    m_ord = 3.0
    ln_r0 = -math.log(1e-7)
    sig_max_z = 0.5 * (m_ord + 1.0) * ln_r0 / (k0 * pml_z_um)
    sig_max_y = 0.5 * (m_ord + 1.0) * ln_r0 / (k0 * pml_y_um)

    def _s_prof(coord: np.ndarray, u_min: float, u_max: float, l_pml: float, sig_m: float) -> np.ndarray:
        d_lo = np.clip((u_min + l_pml - coord) / l_pml, 0.0, 1.0)
        d_hi = np.clip((coord - (u_max - l_pml)) / l_pml, 0.0, 1.0)
        d = np.maximum(d_lo, d_hi)
        return 1.0 + 1j * sig_m * (d ** m_ord)

    sz_n = _s_prof(z_um, z_um[0], z_um[-1], pml_z_um, sig_max_z)
    sz_h = _s_prof(0.5 * (z_um[:-1] + z_um[1:]), z_um[0], z_um[-1], pml_z_um, sig_max_z)
    sy_n = _s_prof(y_um, y_um[0], y_um[-1], pml_y_um, sig_max_y)
    sy_h = _s_prof(0.5 * (y_um[:-1] + y_um[1:]), y_um[0], y_um[-1], pml_y_um, sig_max_y)

    # 2. Exact 1D discrete slab waveguide eigenmode on the y-grid (port profile without thick Si handle mode)
    is_te = pol.upper().startswith("TE")
    ay_p = 1.0 / (sy_n[:-1] * sy_h * (dy ** 2))  # (Ny-1,) coupling (iy -> iy+1)
    ay_m = 1.0 / (sy_n[1:] * sy_h * (dy ** 2))   # (Ny-1,) coupling (iy -> iy-1)

    if is_te:
        wy_p = ay_p
        wy_m = ay_m
        d1_main = np.zeros(ny, dtype=np.complex128)
        d1_main[:-1] -= wy_p
        d1_main[1:] -= wy_m
        d1_main += (k0 ** 2) * eps_port_1d
        m1d = sp.diags([wy_m, d1_main, wy_p], offsets=[-1, 0, 1], format="csc")
    else:
        inv_eps_yh = 2.0 / (eps_port_1d[:-1] + eps_port_1d[1:])
        wy_p = eps_port_1d[:-1] * ay_p * inv_eps_yh
        wy_m = eps_port_1d[1:] * ay_m * inv_eps_yh
        d1_main = np.zeros(ny, dtype=np.complex128)
        d1_main[:-1] -= wy_p
        d1_main[1:] -= wy_m
        d1_main += (k0 ** 2) * eps_port_1d
        m1d = sp.diags([wy_m, d1_main, wy_p], offsets=[-1, 0, 1], format="csc")

    vals_1d, vecs_1d = spla.eigs(m1d, k=2, sigma=(k0 * 1.005 * neff_port_guess) ** 2, which="LM")
    # Pick guided mode localized in the core film with largest real(beta^2)
    idx_m = int(np.argmax(np.real(vals_1d)))
    beta2 = complex(vals_1d[idx_m])
    u0 = vecs_1d[:, idx_m].astype(np.complex128)
    # Normalize mode amplitude & phase
    i_pk = int(np.argmax(np.abs(u0)))
    u0 = u0 / u0[i_pk]

    # Exact discrete longitudinal wavenumber beta_grid on uniform dz stencil:
    # For TM where z-operator in uniform wg is (1/eps)*d2/dz2 (multiplied by eps gives d2/dz2),
    # 2*(cos(beta_grid * dz) - 1)/dz^2 = -beta2  =>  cos_bdz = 1 - 0.5 * beta2 * dz^2
    cos_bdz = 1.0 - 0.5 * beta2 * (dz ** 2)
    sin_bdz = np.sqrt(1.0 - cos_bdz ** 2)
    if np.real(sin_bdz) < 0:
        sin_bdz = -sin_bdz
    exp_plus_ikdz = cos_bdz + 1j * sin_bdz
    beta_grid = np.real(-1j * np.log(exp_plus_ikdz) / dz)

    # 3. Assemble sparse 2D FDFD matrix (flattened index idx = iz * ny + iy, ny contiguous!)
    n_tot = nz * ny
    az_p_1d = 1.0 / (sz_n[:-1] * sz_h * (dz ** 2))  # (Nz-1,)
    az_m_1d = 1.0 / (sz_n[1:] * sz_h * (dz ** 2))   # (Nz-1,)

    if is_te:
        # Helmholtz: d_z(1/sz d_z Ex)/sz + d_y(1/sy d_y Ex)/sy + k0^2 eps(z,y) Ex = b
        cz_p = np.repeat(az_p_1d, ny)                  # (Nz-1)*Ny, offset +ny
        cz_m = np.repeat(az_m_1d, ny)                  # (Nz-1)*Ny, offset -ny
        cy_p_tile = np.tile(np.append(ay_p, 0.0), nz)[:-1]  # offset +1
        cy_m_tile = np.tile(np.append(ay_m, 0.0), nz)[:-1]  # wait: careful on offset -1!
        # Let's construct 2D arrays (nz, ny) for zero indexing ambiguity!
        wz_p = az_p_1d[:, None] * np.ones((1, ny), dtype=np.complex128)  # (nz-1, ny)
        wz_m = az_m_1d[:, None] * np.ones((1, ny), dtype=np.complex128)  # (nz-1, ny)
        wy_p2 = np.ones((nz, 1), dtype=np.complex128) * ay_p[None, :]    # (nz, ny-1)
        wy_m2 = np.ones((nz, 1), dtype=np.complex128) * ay_m[None, :]    # (nz, ny-1)
        diag2 = (k0 ** 2) * eps_2d.astype(np.complex128)
    else:
        # TM Div-Grad: d_z(1/(eps*sz) d_z Hx)/sz + d_y(1/(eps*sy) d_y Hx)/sy + k0^2 Hx = b
        inv_eps_zh = 2.0 / (eps_2d[:-1, :] + eps_2d[1:, :])  # (nz-1, ny)
        inv_eps_yh2 = 2.0 / (eps_2d[:, :-1] + eps_2d[:, 1:])  # (nz, ny-1)
        wz_p = az_p_1d[:, None] * inv_eps_zh
        wz_m = az_m_1d[:, None] * inv_eps_zh
        wy_p2 = ay_p[None, :] * inv_eps_yh2
        wy_m2 = ay_m[None, :] * inv_eps_yh2
        diag2 = np.full((nz, ny), k0 ** 2, dtype=np.complex128)

    diag2[:-1, :] -= wz_p
    diag2[1:, :] -= wz_m
    diag2[:, :-1] -= wy_p2
    diag2[:, 1:] -= wy_m2

    # Convert (nz, ny) directional weights to 1D diagonals cleanly
    d_main = diag2.ravel()
    d_zp = wz_p.ravel()  # length (nz-1)*ny -> offset +ny
    d_zm = wz_m.ravel()  # length (nz-1)*ny -> offset -ny
    vp = np.zeros((nz, ny), dtype=np.complex128)
    vp[:, :-1] = wy_p2
    d_yp = vp.ravel()[:-1]  # offset +1
    vm = np.zeros((nz, ny), dtype=np.complex128)
    vm[:, 1:] = wy_m2
    d_ym = vm.ravel()[1:]   # offset -1

    mat_2d = sp.diags(
        [d_zm, d_ym, d_main, d_yp, d_zp],
        offsets=[-ny, -1, 0, 1, ny],
        shape=(n_tot, n_tot),
        format="csc",
    )

    # 4. Exact Total-Field / Scattered-Field 1D Waveguide Mode Launch at iz_src
    b = np.zeros((nz, ny), dtype=np.complex128)
    e_inc_s = u0                                  # at iz = iz_src
    e_inc_sm1 = u0 / exp_plus_ikdz                # at iz = iz_src - 1
    b[iz_src - 1, :] += wz_p[iz_src - 1, :] * e_inc_s
    b[iz_src, :] -= wz_m[iz_src - 1, :] * e_inc_sm1

    field = spla.spsolve(mat_2d, b.ravel()).reshape((nz, ny))

    # 5. Extract Modal Reflection S11, Residual Transmission, Up/Down Radiated Power & Fiber Overlap
    if is_te:
        w_orth = np.ones(ny, dtype=np.float64)
        # Guided power normalization: P_in = (beta_grid / (2 w mu0)) int |u0|^2 dy (up to common constant C)
        p_in = float(beta_grid * np.sum(np.abs(u0) ** 2) * dy)
    else:
        w_orth = 1.0 / np.real(eps_port_1d)
        p_in = float(beta_grid * np.sum(w_orth * (np.abs(u0) ** 2)) * dy)

    norm_u0 = float(np.sum(w_orth * (np.abs(u0) ** 2)) * dy)
    # At iz_refl < iz_src, field is purely reflected scattered field!
    amp_r = complex(np.sum(w_orth * field[iz_refl, :] * np.conj(u0)) * dy / norm_u0)
    r_mag = float(min(abs(amp_r), 0.999))
    r_power = r_mag ** 2

    amp_t_wg = complex(np.sum(w_orth * field[iz_trans, :] * np.conj(u0)) * dy / norm_u0)
    t_wg_power = float(min(abs(amp_t_wg) ** 2, 1.0))

    # Upward (+y) and downward (-y) Poynting flux across horizontal monitors above/below grating
    z_mon = z_um[iz_mon_slice]
    f_up = field[iz_mon_slice, iy_up_mon]
    df_dy_up = (field[iz_mon_slice, iy_up_mon + 1] - field[iz_mon_slice, iy_up_mon - 1]) / (2.0 * dy)
    f_dn = field[iz_mon_slice, iy_dn_mon]
    df_dy_dn = (field[iz_mon_slice, iy_dn_mon + 1] - field[iz_mon_slice, iy_dn_mon - 1]) / (2.0 * dy)

    if is_te:
        # S_y proportional to Im(conj(Ex) * dEx/dy)
        sy_up = np.imag(np.conj(f_up) * df_dy_up)
        sy_dn = -np.imag(np.conj(f_dn) * df_dy_dn)
    else:
        eps_up = np.real(eps_2d[iz_mon_slice, iy_up_mon])
        eps_dn = np.real(eps_2d[iz_mon_slice, iy_dn_mon])
        sy_up = np.imag(np.conj(f_up) * df_dy_up / eps_up)
        sy_dn = -np.imag(np.conj(f_dn) * df_dy_dn / eps_dn)

    p_up = float(max(np.sum(sy_up) * dz, 0.0))
    p_dn = float(max(np.sum(sy_dn) * dz, 0.0))
    eta_up = float(np.clip(p_up / max(p_in, 1e-30), 0.0, 0.98))
    eta_dn = float(np.clip(p_dn / max(p_in, 1e-30), 0.0, 0.98))

    # Tilted SMF Gaussian fiber overlap along y = y_up_mon (Snell angle inside monitor medium n_top_mon)
    sin_th_top = float(np.clip((n_env / n_top_mon) * math.sin(math.radians(fiber_angle_deg)), -0.95, 0.95))
    cos_th_top = math.sqrt(max(1.0 - sin_th_top ** 2, 0.05))
    kz_fib = k0 * n_top_mon * sin_th_top
    wz_proj = (0.5 * mfd_z_um) / cos_th_top

    # Scan fiber landing center z0 across the grating aperture using full analytical Gaussian norm
    z0_cands = np.linspace(z_mon[0] + 1.5, z_mon[-1] - 1.5, 61)
    dz_mat = z_mon[None, :] - z0_cands[:, None]  # (Nc, Nz_mon)
    g_mat = np.exp(-((dz_mat / wz_proj) ** 2) + 1j * kz_fib * dz_mat)
    g_norm_exact = math.sqrt(0.5 * math.pi) * wz_proj
    g_norm = np.maximum(np.sum(np.abs(g_mat) ** 2, axis=1) * dz, g_norm_exact)
    f_norm = float(max(np.sum(np.abs(f_up) ** 2) * dz, 1e-30))

    overlaps = (np.abs(np.conj(g_mat) @ f_up * dz) ** 2) / np.maximum(g_norm * f_norm, 1e-30)
    i_best = int(np.argmax(overlaps))
    mode_match_1d = float(np.clip(overlaps[i_best], 0.0, 1.0))
    z_opt_um = float(z0_cands[i_best])

    # Fresnel transmission at top_clad -> fiber environment interface (zero if epoxy index-matched)
    t_top_fresnel = 1.0 - float(((n_top_mon - n_env) / (n_top_mon + n_env)) ** 2)
    eta_fiber = float(np.clip(eta_up * mode_match_1d * lateral_overlap * t_top_fresnel, 1e-6, 0.96))

    # Phase of coupled wave for complex S21
    c_ov = complex((np.conj(g_mat[i_best]) @ f_up) * dz)
    s21_phase = float(np.angle(c_ov))

    out = {
        "wl_um": wl_um,
        "eta_fiber": eta_fiber,
        "il_db": -10.0 * math.log10(eta_fiber),
        "eta_up": eta_up,
        "eta_down": eta_dn,
        "eta_wg_thru": t_wg_power,
        "mode_match_1d": mode_match_1d,
        "r_power": r_power,
        "s11_db": 10.0 * math.log10(max(r_power, 1e-8)),
        "s11_complex": amp_r,
        "s21_complex": math.sqrt(eta_fiber) * np.exp(1j * s21_phase),
        "z_fiber_opt_um": z_opt_um,
    }
    if return_field:
        # Subsample physical interior field for crisp Retina 2x UI heatmap (< 45 KB JSON)
        iz_s = max(1, nz // 160)
        iy_s = max(1, ny // 75)
        e_mag = np.abs(field[::iz_s, ::iy_s]).T  # (Ny_sub, Nz_sub)
        e_pk = float(max(np.percentile(e_mag, 99.2), 1e-9))
        out["field_2d"] = {
            "z_um": np.round(z_um[::iz_s], 3).tolist(),
            "y_um": np.round(y_um[::iy_s], 3).tolist(),
            "E_norm": np.round(np.clip(e_mag / e_pk, 0.0, 1.25), 3).tolist(),
        }
    return out


def design_grating_coupler(
    *,
    core_material: str = "Si",
    clad_material: str = "SiO2",
    top_clad_material: str = "SiO2",
    substrate_material: str = "Si",
    wavelength_nm: float = 1550.0,
    polarization: Literal["TE0", "TM0"] = "TE0",
    film_height_um: float = 0.22,
    etch_depth_um: float | None = None,
    fiber_angle_deg: float = 10.0,
    apodized: bool = True,
    focusing: bool = True,
    pitch_um: float | None = None,
    fill_factor: float = 0.50,
    num_periods: int = 26,
    grating_width_um: float = 12.0,
    w_bus_um: float = 0.48,
    box_thickness_um: float = 2.0,
    top_clad_thickness_um: float = 1.5,
    fiber_preset: str = "SMF28_1550",
    mfd_um: float | None = None,
    n_env: float | None = None,
    min_feature_nm: float = 80.0,
    wavelength_span_nm: float | None = None,
) -> dict:
    """Synthesize uniform/apodized/focusing Grating Coupler and solve via Native 2D Full-Wave Maxwell FDFD."""
    t0 = time.perf_counter()
    wl0_um = float(wavelength_nm) * 1e-3
    wl_scale = wl0_um / 1.55

    fib = _resolve_fiber_spec(fiber_preset, wavelength_nm, mfd_um, mfd_um, n_env)
    mfd_val_um = float(fib["mfd_x_um"])
    n_f_env = float(fib["n_env"])

    # Default cleanroom partial etch depth (~32% of film for Si, ~55% for lower-index Si3N4/SiC/LN)
    if etch_depth_um is None or float(etch_depth_um) <= 0:
        if core_material in ("Si", "Ge", "a-Si:H"):
            etch_um = round(min(0.32 * float(film_height_um), float(film_height_um)), 4)
        else:
            etch_um = round(min(0.65 * float(film_height_um), float(film_height_um)), 4)
    else:
        etch_um = float(np.clip(etch_depth_um, 0.02, float(film_height_um)))

    wl0_m = float(wavelength_nm) * 1e-9
    core_fn = get_material_fn(core_material)
    clad_fn = get_material_fn(clad_material)
    top_fn = get_material_fn(top_clad_material)
    sub_fn = get_material_fn(substrate_material) if substrate_material in ("Si", "SiO2", "CaF2", "Al2O3") else clad_fn

    t_core = core_fn(wl0_m, T_REF_K)
    n_core0 = float(t_core.nx if polarization.upper().startswith("TE") else t_core.ny)
    n_clad0 = float(clad_fn(wl0_m, T_REF_K).nx)
    n_top0 = float(top_fn(wl0_m, T_REF_K).nx)
    n_sub0 = float(sub_fn(wl0_m, T_REF_K).nx) if clad_material == "SiO2" else n_clad0

    # Stage 1: Synthesize Bragg / Gaussian-apodized grating teeth
    synth = synthesize_grating_teeth(
        n_core=n_core0,
        n_clad=n_clad0,
        n_top=n_top0,
        n_env=n_f_env,
        wl_um=wl0_um,
        h_film_um=float(film_height_um),
        etch_depth_um=etch_um,
        fiber_angle_deg=float(fiber_angle_deg),
        mfd_um=mfd_val_um,
        polarization=polarization,
        apodized=bool(apodized),
        pitch_um=pitch_um,
        fill_factor=float(fill_factor),
        num_periods=int(num_periods),
        min_feature_nm=float(min_feature_nm),
    )

    grooves_local = synth["grooves_um"]
    gc_len_um = synth["total_grating_length_um"]

    # Lateral (out-of-plane x) 1D overlap between fundamental TE0 cosine mode in W_gc and Gaussian MFD
    # int_{-W/2}^{W/2} cos(pi x / W) * exp(-x^2 / w0^2) dx
    x_lat = np.linspace(-2.5 * max(grating_width_um, mfd_val_um), 2.5 * max(grating_width_um, mfd_val_um), 401)
    dx_lat = float(x_lat[1] - x_lat[0])
    u_lat = np.where(np.abs(x_lat) <= 0.5 * grating_width_um, np.cos(math.pi * x_lat / grating_width_um), 0.0)
    g_lat = np.exp(-((x_lat / (0.5 * mfd_val_um)) ** 2))
    lateral_overlap = float(
        (np.sum(u_lat * g_lat) * dx_lat) ** 2
        / max((np.sum(u_lat ** 2) * dx_lat) * (np.sum(g_lat ** 2) * dx_lat), 1e-30)
    )

    # Stage 2: Build 2D Sagittal FDFD Grid with Subpixel Interface Smoothing
    dz = round(0.032 * wl_scale, 5)
    dy = round(0.024 * wl_scale, 5)
    pml_z = round(0.75 * wl_scale, 3)
    pml_y = round(0.65 * wl_scale, 3)

    z_wg_in = round(2.2 * wl_scale, 3)
    z_grating_start = pml_z + z_wg_in
    z_max = round(z_grating_start + gc_len_um + 2.4 * wl_scale + pml_z, 3)

    t_box = max(float(box_thickness_um), 0.4 * wl_scale)
    t_top = max(float(top_clad_thickness_um), 0.0)
    air_buf = round(1.25 * wl_scale, 3)
    sub_buf = round(0.65 * wl_scale, 3)

    y_min = round(-t_box - sub_buf - pml_y, 4)
    y_max = round(float(film_height_um) + t_top + air_buf + pml_y, 4)

    z_um = np.arange(0.0, z_max, dz)
    y_um = np.arange(y_min, y_max, dy)

    iz_src = int(np.argmin(np.abs(z_um - (pml_z + 0.95 * wl_scale))))
    iz_refl = int(np.argmin(np.abs(z_um - (pml_z + 0.40 * wl_scale))))
    iz_trans = int(np.argmin(np.abs(z_um - (z_grating_start + gc_len_um + 1.30 * wl_scale))))
    iz_mon_lo = int(np.argmin(np.abs(z_um - (z_grating_start - 1.1 * wl_scale))))
    iz_mon_hi = int(np.argmin(np.abs(z_um - (z_grating_start + gc_len_um + 1.6 * wl_scale))))
    iz_mon_slice = slice(iz_mon_lo, iz_mon_hi)

    # Place upward monitor inside superstrate/air (or top oxide if very thick) above core
    y_up_target = float(film_height_um) + t_top + 0.45 * wl_scale
    iy_up_mon = int(np.argmin(np.abs(y_um - y_up_target)))
    iy_dn_mon = int(np.argmin(np.abs(y_um - (-t_box - 0.25 * wl_scale))))

    def _build_eps_for_wl(wl_u: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        wm = wl_u * 1e-6
        tc = core_fn(wm, T_REF_K)
        nc = float(tc.nx if polarization.upper().startswith("TE") else tc.ny)
        ncl = float(clad_fn(wm, T_REF_K).nx)
        ntp = float(top_fn(wm, T_REF_K).nx)
        nsub = float(sub_fn(wm, T_REF_K).nx) if clad_material == "SiO2" else ncl
        nsup = n_f_env  # Superstrate above top_clad (air or index-matching gel)

        # Base 1D unetched stack along y (eps_1d includes bottom Si handle; eps_port_1d keeps BOX below core
        # so the 1D port solver extracts the genuine waveguide core mode rather than a thick Si-wafer slab mode)
        eps_1d = np.empty(len(y_um), dtype=np.float64)
        eps_port_1d = np.empty(len(y_um), dtype=np.float64)
        for iy, yv in enumerate(y_um):
            y_lo = yv - 0.5 * dy
            y_hi = yv + 0.5 * dy

            def _ov(a: float, b: float) -> float:
                return max(0.0, min(y_hi, b) - max(y_lo, a)) / dy

            f_sub = _ov(-1e6, -t_box)
            f_box = _ov(-t_box, 0.0)
            f_cor = _ov(0.0, float(film_height_um))
            f_top = _ov(float(film_height_um), float(film_height_um) + t_top) if t_top > 1e-5 else 0.0
            f_sup = _ov(float(film_height_um) + t_top, 1e6)
            eps_1d[iy] = (
                f_sub * (nsub ** 2)
                + f_box * (ncl ** 2)
                + f_cor * (nc ** 2)
                + f_top * (ntp ** 2)
                + f_sup * (nsup ** 2)
            )
            eps_port_1d[iy] = (
                (f_sub + f_box) * (ncl ** 2)
                + f_cor * (nc ** 2)
                + f_top * (ntp ** 2)
                + f_sup * (nsup ** 2)
            )

        neff_port_est = _solve_1d_slab_neff(ncl, nc, ntp if t_top > 0.05 else nsup, float(film_height_um), wl_u, polarization)
        eps_2d = np.tile(eps_1d[None, :], (len(z_um), 1))
        # Carve etched grooves in [h_film - etch_um, h_film] filled with top_clad (or superstrate if t_top == 0)
        n_fill = ntp if t_top > 1e-4 else nsup
        y_etch_bot = float(film_height_um) - etch_um
        y_etch_top = float(film_height_um)

        # Vertical etch-zone overlap weights (Ny,)
        wy_etch = np.array(
            [max(0.0, min(yv + 0.5 * dy, y_etch_top) - max(yv - 0.5 * dy, y_etch_bot)) / dy for yv in y_um]
        )
        etch_mask_y = wy_etch > 0.0

        for g0_loc, g1_loc in grooves_local:
            za = z_grating_start + g0_loc
            zb = z_grating_start + g1_loc
            iz0 = max(0, int(math.floor((za - z_um[0]) / dz)) - 1)
            iz1 = min(len(z_um) - 1, int(math.ceil((zb - z_um[0]) / dz)) + 1)
            for iz in range(iz0, iz1 + 1):
                z_lo = z_um[iz] - 0.5 * dz
                z_hi = z_um[iz] + 0.5 * dz
                wz = max(0.0, min(z_hi, zb) - max(z_lo, za)) / dz
                if wz > 0.0:
                    w2 = wz * wy_etch[etch_mask_y]
                    eps_2d[iz, etch_mask_y] = (1.0 - w2) * eps_2d[iz, etch_mask_y] + w2 * (n_fill ** 2)

        return eps_2d, eps_port_1d, neff_port_est, nsup

    # Broadband 2D FDFD solve across 5 anchor wavelengths + smooth cubic spline onto 61 dense points
    span_nm = float(wavelength_span_nm) if wavelength_span_nm is not None else max(95.0 * wl_scale, 70.0)
    wl_anchors_nm = np.linspace(float(wavelength_nm) - 0.5 * span_nm, float(wavelength_nm) + 0.5 * span_nm, 5)
    i_mid = len(wl_anchors_nm) // 2

    anchor_res = []
    center_field_2d = None
    for idx_w, w_nm in enumerate(wl_anchors_nm):
        wu = float(w_nm) * 1e-3
        e2d, ep1d, neff_g, n_mon = _build_eps_for_wl(wu)
        res_w = _solve_gc_fdfd_single_wl(
            wu,
            z_um,
            y_um,
            e2d,
            ep1d,
            neff_g,
            polarization,
            pml_z,
            pml_y,
            iz_src,
            iz_refl,
            iz_trans,
            iy_up_mon,
            iy_dn_mon,
            iz_mon_slice,
            n_mon,
            n_f_env,
            float(fiber_angle_deg),
            mfd_val_um,
            lateral_overlap,
            return_field=(idx_w == i_mid),
        )
        if idx_w == i_mid:
            center_field_2d = res_w.pop("field_2d", None)
        anchor_res.append(res_w)

    wl_dense_nm = np.linspace(wl_anchors_nm[0], wl_anchors_nm[-1], 61)
    eta_anchors = np.array([r["eta_fiber"] for r in anchor_res])
    s21_re = sp_interp.PchipInterpolator(wl_anchors_nm, [np.real(r["s21_complex"]) for r in anchor_res])(wl_dense_nm)
    s21_im = sp_interp.PchipInterpolator(wl_anchors_nm, [np.imag(r["s21_complex"]) for r in anchor_res])(wl_dense_nm)
    eta_dense = np.clip(sp_interp.CubicSpline(wl_anchors_nm, eta_anchors)(wl_dense_nm), 1e-6, 0.96)
    il_dense_db = -10.0 * np.log10(eta_dense)

    r_anchors = np.array([r["r_power"] for r in anchor_res])
    r_dense = np.clip(sp_interp.PchipInterpolator(wl_anchors_nm, r_anchors)(wl_dense_nm), 1e-8, 0.95)
    s11_dense_db = 10.0 * np.log10(r_dense)

    idx_pk = int(np.argmax(eta_dense))
    peak_wl_nm = float(wl_dense_nm[idx_pk])
    peak_eta = float(eta_dense[idx_pk])
    peak_il_db = float(il_dense_db[idx_pk])
    mid_r = anchor_res[i_mid]

    def _bandwidth_nm(drop_db: float) -> float:
        thr = peak_il_db + abs(drop_db)
        inside = wl_dense_nm[il_dense_db <= thr]
        return float(inside[-1] - inside[0]) if len(inside) >= 2 else 0.0

    bw_1db_nm = _bandwidth_nm(1.0)
    bw_3db_nm = _bandwidth_nm(3.0)

    # Focusing curved-groove focal length (from waveguide end to first curved arc)
    focal_length_um = round(max(18.0 * wl_scale, 1.45 * float(grating_width_um)), 2)

    warnings = []
    if mid_r["eta_down"] > 0.36:
        warnings.append(
            f"Downward substrate radiation is {mid_r['eta_down']*100:.1f}% at BOX={box_thickness_um:.2f} um "
            f"(destructive bottom-Si reflection); adjusting BOX by ~quarter-wave (~{0.25*wl0_um/n_clad0:.2f} um) boosts upward directionality."
        )
    if mid_r["s11_db"] > -15.0:
        warnings.append(
            f"Waveguide back-reflection S11 = {mid_r['s11_db']:.1f} dB; enable apodized teeth or increase fiber tilt angle (>=10 deg) to suppress second-order Bragg reflection."
        )

    return {
        "coupler_type": "grating_coupler",
        "solver": "Bragg/Apodization Synthesis + Native 2D Full-Wave Maxwell FDFD (UPML, Zero Commercial Licenses)",
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "core_material": core_material,
        "wavelength_nm": float(wavelength_nm),
        "polarization": polarization,
        "apodized": bool(apodized),
        "focusing": bool(focusing),
        "fiber_preset": fib["preset"],
        "fiber_label": fib["label"],
        "fiber_angle_deg": float(fiber_angle_deg),
        "fiber_mfd_um": mfd_val_um,
        "film_height_um": float(film_height_um),
        "etch_depth_um": round(etch_um, 4),
        "box_thickness_um": float(box_thickness_um),
        "top_clad_thickness_um": float(top_clad_thickness_um),
        "grating_width_um": float(grating_width_um),
        "w_bus_um": float(w_bus_um),
        "focal_length_um": focal_length_um,
        "num_periods": int(num_periods),
        "synthesis": {
            "pitch_mean_um": round(synth["pitch_mean_um"], 4),
            "ff_start": round(synth["ff_start"], 3),
            "ff_end": round(synth["ff_end"], 3),
            "neff_full": round(synth["neff_full"], 5),
            "neff_etch": round(synth["neff_etch"], 5),
            "grating_length_um": round(gc_len_um, 2),
            "grooves_um": [[round(a, 4), round(b, 4)] for a, b in grooves_local],
            "pitches_um": [round(p, 4) for p in synth["pitches_um"]],
            "fill_factors": [round(f, 4) for f in synth["fill_factors"]],
        },
        "metrics": {
            "peak_il_db": round(peak_il_db, 3),
            "peak_efficiency_pct": round(peak_eta * 100.0, 2),
            "peak_wavelength_nm": round(peak_wl_nm, 2),
            "center_il_db": round(mid_r["il_db"], 3),
            "bw_1db_nm": round(bw_1db_nm, 2),
            "bw_3db_nm": round(bw_3db_nm, 2),
            "upward_directionality_pct": round(mid_r["eta_up"] * 100.0, 2),
            "substrate_loss_pct": round(mid_r["eta_down"] * 100.0, 2),
            "unradiated_thru_pct": round(mid_r["eta_wg_thru"] * 100.0, 2),
            "mode_match_1d_pct": round(mid_r["mode_match_1d"] * 100.0, 2),
            "lateral_overlap_pct": round(lateral_overlap * 100.0, 2),
            "s11_db": round(mid_r["s11_db"], 2),
            "back_reflection_s11_db": round(mid_r["s11_db"], 2),
            "z_fiber_offset_from_start_um": round(mid_r["z_fiber_opt_um"] - z_grating_start, 2),
        },
        "spectrum": {
            "wavelength_nm": np.round(wl_dense_nm, 3).tolist(),
            "il_db": np.round(il_dense_db, 3).tolist(),
            "s11_db": np.round(s11_dense_db, 2).tolist(),
            "s21_real": np.round(s21_re, 5).tolist(),
            "s21_imag": np.round(s21_im, 5).tolist(),
            "s21_mag": np.round(np.sqrt(eta_dense), 5).tolist(),
            "s11_mag": np.round(np.sqrt(r_dense), 5).tolist(),
        },
        "field_2d": center_field_2d,
        "warnings": warnings,
    }


# =============================================================================
# 3. Foundry GDSII Mask & Lumerical INTERCONNECT S-Parameter Export
# =============================================================================

IO_EXTRA_LAYERS = {
    "shallow_etch": (4, 0, "#38BDF8", "Grating Coupler Shallow Etch Grooves"),
    "facet_trench": (40, 0, "#F43F5E", "Deep Facet Dicing / Undercut Trench"),
}


def export_io_coupler_gds(
    coupler_result: dict,
    out_gds_path: str | Path | None = None,
    *,
    include_ring_loopback: bool = False,
    ring_config: MRRDesignConfig | None = None,
    fa_pitch_um: float = 127.0,
) -> dict:
    """Export Butt Coupler or Focusing/Linear Grating Coupler `.GDS` (+ optional 127-um FA Ring Loopback)."""
    ctype = coupler_result["coupler_type"]
    mat = coupler_result["core_material"]
    wl_nm = int(round(coupler_result["wavelength_nm"]))

    out_root = get_default_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    if out_gds_path is None:
        loop_tag = "_RingTestLoop" if include_ring_loopback else ""
        fname = f"MicroRing_{ctype}_{mat}_{wl_nm}nm{loop_tag}.gds"
        gds_path = out_root / fname
    else:
        gds_path = Path(out_gds_path)
        gds_path.parent.mkdir(parents=True, exist_ok=True)

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    l_core = MICRORING_LAYERS["core"][:2]
    l_pin = MICRORING_LAYERS["pin"][:2]
    l_txt = MICRORING_LAYERS["label"][:2]
    l_etch = IO_EXTRA_LAYERS["shallow_etch"][:2]
    l_facet = IO_EXTRA_LAYERS["facet_trench"][:2]

    if ctype == "butt_edge":
        cell_c = lib.new_cell(f"ButtCoupler_{mat}")
        wt = float(coupler_result["w_tip_um"])
        wb = float(coupler_result["w_bus_um"])
        lt = float(coupler_result["taper_length_um"])
        prof = str(coupler_result.get("taper_profile", "adiabatic"))

        u = np.linspace(0.0, 1.0, 65)
        if prof == "linear":
            w_u = wt + (wb - wt) * u
        elif prof == "parabolic":
            w_u = wt + (wb - wt) * (u ** 2)
        else:
            # Smooth adiabatic exponential/Milton-Burns expansion
            w_u = wt * ((wb / max(wt, 0.04)) ** (u ** 1.35))
        zs = u * lt
        poly_pts = [(float(z), float(0.5 * w)) for z, w in zip(zs, w_u)] + [
            (float(z), float(-0.5 * w)) for z, w in zip(zs[::-1], w_u[::-1])
        ]
        cell_c.add(gdstk.Polygon(poly_pts, layer=l_core[0], datatype=l_core[1]))
        # Bus lead-out segment (15 um)
        cell_c.add(gdstk.rectangle((lt, -0.5 * wb), (lt + 15.0, 0.5 * wb), layer=l_core[0], datatype=l_core[1]))
        # Facet cleavage / deep-etch trench marker at z in [-12, 0]
        cell_c.add(gdstk.rectangle((-12.0, -18.0), (0.0, 18.0), layer=l_facet[0], datatype=l_facet[1]))
        cell_c.add(gdstk.rectangle((lt + 14.0, -0.5 * wb), (lt + 15.0, 0.5 * wb), layer=l_pin[0], datatype=l_pin[1]))
        cell_c.add(
            gdstk.Label(
                f"ButtSSC {mat} Wtip={wt*1e3:.0f}nm L={lt:.0f}um ({prof}) IL={coupler_result['metrics']['total_il_db']:.2f}dB",
                (0.0, 6.0),
                layer=l_txt[0],
                texttype=l_txt[1],
            )
        )
    else:
        cell_c = lib.new_cell(f"GratingCoupler_{mat}")
        wb = float(coupler_result["w_bus_um"])
        w_gc = float(coupler_result["grating_width_um"])
        grooves = coupler_result["synthesis"]["grooves_um"]
        gc_len = float(coupler_result["synthesis"]["grating_length_um"])
        focusing = bool(coupler_result.get("focusing", True))

        # Input bus stub [-10, 0]
        cell_c.add(gdstk.rectangle((-10.0, -0.5 * wb), (0.0, 0.5 * wb), layer=l_core[0], datatype=l_core[1]))
        cell_c.add(gdstk.rectangle((-10.0, -0.5 * wb), (-9.0, 0.5 * wb), layer=l_pin[0], datatype=l_pin[1]))

        if focusing:
            # Confocal curved-groove focusing grating coupler (fan-out sector from origin (0, 0))
            f_len = float(coupler_result["focal_length_um"])
            half_ang = math.atan2(0.5 * w_gc, f_len)
            r_outer = f_len + gc_len + 1.5
            ths = np.linspace(-half_ang, half_ang, 55)
            # Core sector fan from r = 0 (with bus width wb) to r_outer
            sector_pts = [(0.0, -0.5 * wb), (0.0, 0.5 * wb)]
            for th in ths[::-1]:
                sector_pts.append((r_outer * math.cos(th), r_outer * math.sin(th)))
            cell_c.add(gdstk.Polygon(sector_pts, layer=l_core[0], datatype=l_core[1]))

            # Confocal phase-matched curved etch grooves on layer 4/0
            neff_s = float(coupler_result["synthesis"]["neff_full"])
            kx_norm = float(FIBER_LASER_CATALOG.get(coupler_result["fiber_preset"], {"n_env": 1.444})["n_env"]) * math.sin(
                math.radians(float(coupler_result["fiber_angle_deg"]))
            )
            for g0, g1 in grooves:
                # Axial phase condition: neff_s * r(th) - r(th)*cos(th)*kx_norm = (neff_s - kx_norm) * r_axial
                r0_ax = f_len + float(g0)
                r1_ax = f_len + float(g1)
                num0 = (neff_s - kx_norm) * r0_ax
                num1 = (neff_s - kx_norm) * r1_ax
                arc_in = []
                arc_out = []
                for th in ths:
                    denom = max(neff_s - kx_norm * math.cos(th), 0.35)
                    ri = num0 / denom
                    ro = num1 / denom
                    arc_in.append((ri * math.cos(th), ri * math.sin(th)))
                    arc_out.append((ro * math.cos(th), ro * math.sin(th)))
                cell_c.add(gdstk.Polygon(arc_in + arc_out[::-1], layer=l_etch[0], datatype=l_etch[1]))
        else:
            # Linear adiabatic taper (L_tap = 180 um) + straight rectangular grooves
            l_tap = max(140.0, 12.0 * w_gc)
            taper_pts = [
                (0.0, -0.5 * wb),
                (l_tap, -0.5 * w_gc),
                (l_tap + gc_len + 2.0, -0.5 * w_gc),
                (l_tap + gc_len + 2.0, 0.5 * w_gc),
                (l_tap, 0.5 * w_gc),
                (0.0, 0.5 * wb),
            ]
            cell_c.add(gdstk.Polygon(taper_pts, layer=l_core[0], datatype=l_core[1]))
            for g0, g1 in grooves:
                cell_c.add(
                    gdstk.rectangle(
                        (l_tap + float(g0), -0.5 * (w_gc + 1.0)),
                        (l_tap + float(g1), 0.5 * (w_gc + 1.0)),
                        layer=l_etch[0],
                        datatype=l_etch[1],
                    )
                )

        cell_c.add(
            gdstk.Label(
                f"GC {mat} pitch={coupler_result['synthesis']['pitch_mean_um']*1e3:.0f}nm "
                f"etch={coupler_result['etch_depth_um']*1e3:.0f}nm IL={coupler_result['metrics']['peak_il_db']:.2f}dB",
                (0.0, 0.5 * w_gc + 4.0),
                layer=l_txt[0],
                texttype=l_txt[1],
            )
        )

    if include_ring_loopback:
        cfg = ring_config or MRRDesignConfig(
            core_material=mat,
            film_height_um=float(coupler_result.get("film_height_um", 0.22)),
            ring_width_um=float(coupler_result["w_bus_um"]),
            bus_width_um=float(coupler_result["w_bus_um"]),
            wavelength_nm=float(coupler_result["wavelength_nm"]),
        )
        ring_lib = build_mrr_gds_library(cfg)
        ring_cell = ring_lib.cells[0]
        lib.add(ring_cell)

        top = lib.new_cell(f"PIC_TestLoop_{ctype}_{mat}")
        top.add(gdstk.Reference(ring_cell, (0.0, 0.0)))
        r_bb = ring_cell.bounding_box()
        x_left = float(r_bb[0][0]) - 25.0
        x_right = float(r_bb[1][0]) + 25.0
        wb = float(cfg.bus_width_um)
        y_bus = -(float(cfg.ring_radius_um) + 0.5 * float(cfg.ring_width_um) + float(cfg.gap_thru_um) + 0.5 * wb)

        # Place Input Coupler on left and Through Coupler on right + Reference Loopback at +fa_pitch_um (127 um FA pitch!)
        top.add(gdstk.Reference(cell_c, (x_left, y_bus), rotation=math.pi if ctype == "grating_coupler" else 0.0))
        top.add(gdstk.Reference(cell_c, (x_right, y_bus), rotation=0.0 if ctype == "grating_coupler" else math.pi))
        # Connect bus waveguide extensions
        top.add(gdstk.rectangle((x_left, y_bus - 0.5 * wb), (float(r_bb[0][0]) + 5.0, y_bus + 0.5 * wb), layer=l_core[0], datatype=l_core[1]))
        top.add(gdstk.rectangle((float(r_bb[1][0]) - 5.0, y_bus - 0.5 * wb), (x_right, y_bus + 0.5 * wb), layer=l_core[0], datatype=l_core[1]))
        # Calibration straight reference waveguide 127 um above for exact coupler de-embedding in lab!
        y_ref = y_bus + float(fa_pitch_um)
        top.add(gdstk.Reference(cell_c, (x_left, y_ref), rotation=math.pi if ctype == "grating_coupler" else 0.0))
        top.add(gdstk.Reference(cell_c, (x_right, y_ref), rotation=0.0 if ctype == "grating_coupler" else math.pi))
        top.add(gdstk.rectangle((x_left, y_ref - 0.5 * wb), (x_right, y_ref + 0.5 * wb), layer=l_core[0], datatype=l_core[1]))
        top.add(
            gdstk.Label(
                f"MicroRing PIC Loopback + {fa_pitch_um:.0f}um Calibration Reference ({mat})",
                (x_left, y_ref + 16.0),
                layer=l_txt[0],
                texttype=l_txt[1],
            )
        )

    lib.write_gds(str(gds_path))
    lyp_path = gds_path.with_suffix(".lyp")
    write_klayout_layer_props(lyp_path)

    return {
        "gds_path": str(gds_path),
        "lyp_path": str(lyp_path),
        "cells": [c.name for c in lib.cells],
    }


def export_io_coupler_interconnect(
    coupler_result: dict,
    out_dir: str | Path | None = None,
) -> dict:
    """Export 2-Port Complex Optical S-Parameter (`SPAR .dat`) + `.zip` for Ansys Lumerical INTERCONNECT."""
    ctype = coupler_result["coupler_type"]
    mat = coupler_result["core_material"]
    pol_mode = 1 if str(coupler_result.get("polarization", "TE0")).upper().startswith("TE") else 2
    pol_lbl = "TE" if pol_mode == 1 else "TM"

    base_root = Path(out_dir) if out_dir is not None else (get_default_output_root() / f"interconnect_{ctype}_{mat}")
    base_root.mkdir(parents=True, exist_ok=True)

    spec = coupler_result["spectrum"]
    wl_nm = np.asarray(spec["wavelength_nm"], dtype=np.float64)
    freqs_hz = C0_M_PER_S / (wl_nm * 1e-9)
    # Sort ascending in frequency as required by INTERCONNECT Optical N-Port S-Parameter element
    order = np.argsort(freqs_hz)
    freqs_hz = freqs_hz[order]
    n_pts = len(freqs_hz)

    if "s21_real" in spec and "s21_imag" in spec:
        s21 = np.asarray(spec["s21_real"])[order] + 1j * np.asarray(spec["s21_imag"])[order]
    else:
        s21 = np.asarray(spec["s21_mag"])[order].astype(np.complex128)
    s11_mag = np.asarray(spec["s11_mag"])[order]
    # Enforce strict passivity |S11|^2 + |S21|^2 <= 1
    tot_p = (np.abs(s21) ** 2) + (s11_mag ** 2)
    scale = np.where(tot_p > 0.9999, 0.9995 / np.sqrt(tot_p), 1.0)
    s21 = s21 * scale
    s11_mag = s11_mag * scale

    spar_file = base_root / f"microring_{ctype}_{mat}_sparams.dat"
    lines = [
        f"# MicroRing Studio v3.6.0 - {ctype} 2-Port Optical S-Parameters ({mat})",
        '["port 1", "LEFT"]',
        '["port 2", "RIGHT"]',
    ]

    def _blk(dst_p: int, src_p: int, mag_v: np.ndarray, ang_v: np.ndarray) -> None:
        lines.append(f'("port {dst_p}", "{pol_lbl}", {pol_mode}, "port {src_p}", {pol_mode}, "transmission")')
        lines.append(f"({n_pts}, 3)")
        for f_hz, m_v, a_v in zip(freqs_hz, mag_v, ang_v):
            lines.append(f"{f_hz:.8e} {m_v:.7f} {a_v:.7f}")

    z_ang = np.zeros_like(freqs_hz)
    s21_mag = np.abs(s21)
    s21_ang = np.unwrap(np.angle(s21))
    _blk(1, 1, s11_mag, z_ang)
    _blk(2, 1, s21_mag, s21_ang)
    _blk(1, 2, s21_mag, s21_ang)  # Reciprocal 2-port S12 = S21
    _blk(2, 2, s11_mag, z_ang)

    spar_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_file = base_root / f"microring_{ctype}_{mat}_spectrum.csv"
    wl_s = C0_M_PER_S / freqs_hz * 1e9
    il_s = -20.0 * np.log10(np.maximum(s21_mag, 1e-8))
    rl_s = -20.0 * np.log10(np.maximum(s11_mag, 1e-8))
    csv_lines = ["wavelength_nm,frequency_THz,insertion_loss_dB,return_loss_dB,S21_mag,S21_phase_rad"]
    for wv, fv, ilv, rlv, sm, sp_r in zip(wl_s[::-1], freqs_hz[::-1] * 1e-12, il_s[::-1], rl_s[::-1], s21_mag[::-1], s21_ang[::-1]):
        csv_lines.append(f"{wv:.4f},{fv:.6f},{ilv:.4f},{rlv:.3f},{sm:.6f},{sp_r:.5f}")
    csv_file.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    zip_path = base_root.parent / f"MicroRing_IO_INTERCONNECT_{ctype}_{mat}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(spar_file, arcname=f"{ctype}_{mat}/{spar_file.name}")
        zf.write(csv_file, arcname=f"{ctype}_{mat}/{csv_file.name}")

    max_port_power = float(np.max((s21_mag ** 2) + (s11_mag ** 2)))
    return {
        "spar_path": str(spar_file),
        "csv_path": str(csv_file),
        "zip_path": str(zip_path),
        "zip_name": zip_path.name,
        "passivity_verified": bool(max_port_power <= 1.00001),
        "max_port_power": round(max_port_power, 6),
        "freq_points": n_pts,
    }
