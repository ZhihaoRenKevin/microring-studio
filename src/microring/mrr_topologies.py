"""Unified Multi-Topology MRR Synthesis, Coupler Engine, Active Tuning & GDSII Exporter.

Supports designing & analyzing 7 integrated photonic microring topologies from
first-principles Maxwell eigenmodes + spatial/temporal CMT (zero Lumerical,
RSoft, or COMSOL licenses required):
  1. `point_circular_ap` : Point-Coupled Circular All-Pass MRR (2-port)
  2. `point_circular_ad` : Point-Coupled Circular Add-Drop MRR (4-port)
  3. `racetrack_ap`      : Straight-Coupler Racetrack All-Pass MRR (incl. X-cut LN orientation averaging)
  4. `racetrack_ad`      : Straight-Coupler Racetrack Add-Drop MRR (4-port)
  5. `pulley_ap`         : Concentric Wrap-Around Pulley All-Pass MRR
  6. `pulley_ad`         : Concentric Wrap-Around Pulley Add-Drop / Dichroic MRR
  7. `vernier_crow_ad`   : Coupled Dual-Ring Vernier / 2nd-Order CROW Filter

Also computes:
  - Automatic Critical-Coupling Synthesis (`g_CC` [nm], `L_s,CC` [um], or `theta_CC` [deg]).
  - Thermo-Optic (`Delta_T`), Pockels Electro-Optic (`Voltage`, `pm/V`, `V_pi*L`), and PCM tuning.
  - Foundry-ready `.gds` mask layout export via `gdstk`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.special import airy, hankel2

from .eigenmode_solver import (
    MU0_H_PER_M,
    EigenmodeResult,
    WaveguideCrossSectionSpec,
    solve_modes_fde_tidy3d,
    solve_modes_fem,
)
from .hardware_accel import accelerated_tmm_spectrum, detect_hardware_profile
from .materials import (
    C0_M_PER_S,
    EPS0_F_PER_M,
    EXTRA_LITERATURE_MATERIALS,
    MATERIAL_CATALOG,
    T_REF_K,
    check_wavelength_compatibility,
    get_material_fn,
    n_linbo3_xcut,
)
from .mrr_cmt_spatial import (
    PulleyCMTChannel,
    coupling_coefficient_cylindrical,
    scan_pulley_single_bus,
)

TOPOLOGY_CATALOG: dict[str, dict] = {
    "point_circular_ap": {
        "name": "Point-Coupled Circular (All-Pass)",
        "ports": 2,
        "coupler_param": "gap_thru_um",
        "description": "Straight bus tangent to a circular ring of radius R; broadband FSR & compact footprint.",
    },
    "point_circular_ad": {
        "name": "Point-Coupled Circular (4-Port Add-Drop)",
        "ports": 4,
        "coupler_param": "gap_thru_um",
        "description": "Dual straight buses coupled to a circular ring; provides both Through notch and Drop passband.",
    },
    "racetrack_ap": {
        "name": "Racetrack Directional Coupler (All-Pass)",
        "ports": 2,
        "coupler_param": "straight_length_um",
        "description": "Two 180 deg bends + straight coupler L_s; strong coupling at relaxed lithography gaps & ideal for X-cut TFLN.",
    },
    "racetrack_ad": {
        "name": "Racetrack Directional Coupler (4-Port Add-Drop)",
        "ports": 4,
        "coupler_param": "straight_length_um",
        "description": "4-port racetrack resonator with straight coupling arms on Through and Drop buses.",
    },
    "pulley_ap": {
        "name": "Concentric Pulley Wrap-Around (All-Pass)",
        "ports": 2,
        "coupler_param": "pulley_angle_thru_deg",
        "description": "Concentric curved bus wrapping angle theta around ring; phase-matched / selective Q_ext control for thick SiN, SiC & AlN.",
    },
    "pulley_ad": {
        "name": "Concentric Pulley Wrap-Around (4-Port Add-Drop)",
        "ports": 4,
        "coupler_param": "pulley_angle_thru_deg",
        "description": "Dual concentric pulley couplers for wavelength-selective / dichroic add-drop & FWM/QFC routing.",
    },
    "vernier_crow_ad": {
        "name": "Dual-Ring Vernier / 2nd-Order CROW (Add-Drop)",
        "ports": 4,
        "coupler_param": "inter_ring_gap_um",
        "description": "Two series-coupled rings (R1, R2): flat-top 2nd-order CROW when R1=R2, or ultra-wide Vernier FSR when R1!=R2.",
    },
}


@dataclass
class MRRDesignConfig:
    """Complete physical specification for MRR eigenmode + coupler + spectrum synthesis."""
    topology: str = "point_circular_ap"
    # Material stack
    core_material: str = "Si3N4"
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    sc_fraction: float = 0.20               # used when core_material startswith 'ScAlN'
    custom_n: float | None = None
    custom_k: float = 0.0
    custom_clad_n: float | None = None
    custom_top_clad_n: float | None = None
    custom_dn_dt: float | None = None
    custom_r33_pm_v: float | None = None
    # Cross-section geometry
    ring_width_um: float = 1.20             # unit: um (bottom width)
    bus_width_um: float = 1.00              # unit: um (bottom width)
    film_height_um: float = 0.40            # unit: um
    slab_height_um: float = 0.0             # unit: um (0 = strip, >0 = rib)
    sidewall_deg: float = 85.0              # unit: deg
    # Optional bilayer / active cap
    interlayer_h_um: float = 0.0
    upper_core_h_um: float = 0.0
    upper_core_material: str | None = None
    cap_h_um: float = 0.0
    cap_w_um: float | None = None
    cap_material: str | None = None
    cap_length_um: float | None = None      # optional localized PCM/active patch length along ring (None = full ring)
    # Resonator & coupler layout parameters
    ring_radius_um: float = 40.0            # unit: um (R1)
    ring2_radius_um: float = 42.5           # unit: um (R2, used for vernier_crow_ad)
    gap_thru_um: float = 0.25               # unit: um (edge-to-edge bottom gap)
    gap_drop_um: float = 0.25               # unit: um
    inter_ring_gap_um: float = 0.35         # unit: um (used for vernier_crow_ad)
    straight_length_um: float = 12.0        # unit: um (L_s, used for racetrack_*)
    pulley_angle_thru_deg: float = 25.0     # unit: deg (theta_thru, used for pulley_*)
    pulley_angle_drop_deg: float = 25.0     # unit: deg
    # Optical band, polarization & loss
    wavelength_nm: float = 1550.0           # unit: nm
    span_nm: float = 12.0                   # unit: nm (spectrum window around wavelength_nm)
    num_spectrum_points: int = 2000
    polarization: str = "TE0"               # 'TE0' or 'TM0'
    alpha_loss_db_per_cm: float | None = None  # defaults to material typical_loss_db_cm
    sidewall_rms_roughness_nm: float = 0.0  # optional Payne-Lacey sidewall roughness sigma_rms [nm] (0 = off)
    sidewall_corr_length_nm: float = 50.0   # exponential roughness correlation length L_c [nm]
    broadband_coupler_dispersion: bool = False  # True (or span_nm >= 80 nm) evaluates dispersive kappa(lambda)
    # Active tuning inputs
    delta_temp_k: float = 0.0               # unit: K (thermo-optic shift relative to 300 K)
    voltage_v: float = 0.0                  # unit: V (electro-optic bias)
    electrode_gap_um: float = 5.0           # unit: um (effective EO electrode gap)
    # Numerical solver controls
    solver_backend: str = "fem"             # 'fem' (BSD conformal skfem) or 'fde' (tidy3d)
    core_mesh_res_um: float = 0.045         # unit: um (fast interactive default; 0.030 for high-precision)


def _select_target_polarization(modes: list[EigenmodeResult], pol: str) -> EigenmodeResult:
    want_te = pol.upper().startswith("TE")
    candidates = [m for m in modes if (m.te_fraction >= 0.5) == want_te]
    if not candidates:
        return modes[0]
    confined = [m for m in candidates if m.core_confinement >= 0.20]
    base_pool = confined if confined else candidates
    # Prefer fundamental (0 interior nodes) with highest Re(n_eff)
    fund = [m for m in base_pool if m.ex_core_nodes == 0]
    pool = fund if fund else base_pool
    return max(pool, key=lambda m: float(np.real(m.n_eff)))


def get_default_output_root() -> Path:
    """Return a writable output directory across repo clones, pip installs, and frozen .app/.exe apps."""
    env_dir = os.environ.get("MICRORING_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir)
    if not getattr(sys, "frozen", False):
        repo_candidate = Path(__file__).resolve().parents[2]
        if (repo_candidate / "pyproject.toml").is_file():
            return repo_candidate / "outputs"
    docs = Path.home() / "Documents"
    base = docs if docs.is_dir() else Path.home()
    return base / "MicroRing_Studio"


_MODE_MEM_CACHE: dict[str, EigenmodeResult] = {}
_CACHE_DIR = Path(
    os.environ.get("MICRORING_CACHE_DIR", str(get_default_output_root() / ".mode_cache"))
)


def _mode_cache_key(
    spec: WaveguideCrossSectionSpec,
    wavelength_m: float,
    bend_radius_m: float,
    pol: str,
    solver_backend: str,
    core_mesh_res_um: float,
    export_grid: bool,
) -> str:
    c_val = spec.core_material_fn(wavelength_m, spec.temperature_k) if spec.core_material_fn else None
    cl_val = spec.clad_material_fn(wavelength_m, spec.temperature_k) if spec.clad_material_fn else None
    tp_val = spec.top_clad_material_fn(wavelength_m, spec.temperature_k) if spec.top_clad_material_fn else cl_val
    up_val = (
        spec.upper_core_material_fn(wavelength_m, spec.temperature_k)
        if spec.upper_core_h_um > 0 and spec.upper_core_material_fn is not None
        else None
    )
    cp_val = (
        spec.cap_material_fn(wavelength_m, spec.temperature_k)
        if spec.cap_h_um > 0 and spec.cap_material_fn is not None
        else None
    )
    payload = {
        "v": "microring_v3_fem1",
        "geom": [
            round(spec.w_bottom_um, 6),
            round(spec.h_film_um, 6),
            round(spec.h_slab_um, 6),
            round(spec.sidewall_deg, 4),
            round(spec.interlayer_h_um, 6),
            round(spec.upper_core_h_um, 6),
            round(spec.cap_h_um, 6),
            [round(b, 4) for b in spec.x_bounds_um()],
            [round(b, 4) for b in spec.y_bounds_um()],
            round(spec.temperature_k, 3),
        ],
        "n_core": [round(c_val.nx, 7), round(c_val.ny, 7), round(c_val.nz, 7), round(c_val.kx, 8)] if c_val else None,
        "n_clad": round(cl_val.nx, 7) if cl_val else None,
        "n_top": round(tp_val.nx, 7) if tp_val else None,
        "wl_pm": round(wavelength_m * 1e12, 5),
        "r_nm": "inf" if not math.isfinite(bend_radius_m) else round(bend_radius_m * 1e9, 3),
        "pol": pol.upper()[:2],
        "backend": solver_backend,
        "res_um": round(core_mesh_res_um, 5),
        "grid": bool(export_grid),
    }
    if up_val is not None or spec.upper_w_bottom_um is not None:
        payload["n_upper"] = (
            [round(up_val.nx, 7), round(up_val.ny, 7), round(up_val.nz, 7), round(up_val.kx, 8)] if up_val else None
        )
        payload["upper_w_um"] = round(spec.upper_w_bottom_um, 6) if spec.upper_w_bottom_um is not None else None
    if cp_val is not None or spec.cap_w_um is not None:
        payload["n_cap"] = (
            [round(cp_val.nx, 7), round(cp_val.ny, 7), round(cp_val.nz, 7), round(cp_val.kx, 8)] if cp_val else None
        )
        payload["cap_w_um"] = round(spec.cap_w_um, 6) if spec.cap_w_um is not None else None
    if spec.twin_core_gap_um > 0.0:
        payload["twin_gap_um"] = round(spec.twin_core_gap_um, 6)
    if math.isfinite(bend_radius_m) and (bend_radius_m < 8e-6 or spec.h_slab_um > 0.0):
        payload["caustic_guard_v"] = 2
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _solve_single_mode(
    spec: WaveguideCrossSectionSpec,
    wavelength_m: float,
    bend_radius_m: float,
    pol: str,
    solver_backend: str,
    core_mesh_res_um: float,
    export_grid: bool = True,
) -> EigenmodeResult:
    key = _mode_cache_key(spec, wavelength_m, bend_radius_m, pol, solver_backend, core_mesh_res_um, export_grid)
    if key in _MODE_MEM_CACHE:
        return _MODE_MEM_CACHE[key]
    disk_file = _CACHE_DIR / f"{key}.pkl"
    if disk_file.is_file():
        try:
            cached_mode = pickle.loads(disk_file.read_bytes())
            _MODE_MEM_CACHE[key] = cached_mode
            return cached_mode
        except Exception:
            pass

    n_modes = 4
    n_guess_bent: float | None = None
    _caustic_risk = False
    if solver_backend != "fde" and math.isfinite(bend_radius_m):
        if bend_radius_m < 8e-6 or spec.h_slab_um > 0.0:
            _caustic_risk = True
        elif spec.core_material_fn is not None and spec.clad_material_fn is not None:
            _nc_c = float(spec.core_material_fn(wavelength_m, spec.temperature_k).nx)
            _nb_c = float(spec.clad_material_fn(wavelength_m, spec.temperature_k).nx)
            _xmax_c = spec.x_bounds_um()[1] * 1e-6
            if _nb_c * (1.0 + _xmax_c / max(bend_radius_m, 1e-9)) >= 0.90 * _nc_c:
                _caustic_risk = True
    if _caustic_risk:
        straight_ref = _solve_single_mode(
            spec, wavelength_m, math.inf, pol, solver_backend, core_mesh_res_um, export_grid=False
        )
        n_guess_bent = float(np.real(straight_ref.n_eff)) * 1.01
        n_modes = 6
    wl_scale_grid = float(np.clip((wavelength_m * 1e9) / 1550.0, 0.65, 2.80))
    grid_step_um = 0.025 if abs(wl_scale_grid - 1.0) < 0.05 else round(max(0.016, 0.025 * wl_scale_grid), 4)
    if solver_backend == "fde":
        modes = solve_modes_fde_tidy3d(
            spec,
            wavelength_m,
            bend_radius_m=bend_radius_m,
            num_modes=n_modes,
            core_dl_um=max(core_mesh_res_um * 0.65, 0.010),
        )
    else:
        modes = solve_modes_fem(
            spec,
            wavelength_m,
            bend_radius_m=bend_radius_m,
            num_modes=n_modes,
            n_guess=n_guess_bent,
            core_mesh_res_um=core_mesh_res_um,
            clad_mesh_res_um=max(core_mesh_res_um * 5.0, 0.14 * wl_scale_grid),
            eval_grid_step_um=grid_step_um,
            export_grid=export_grid,
            mesh_backend="bsd_conformal",
        )
    chosen = _select_target_polarization(modes, pol)
    _MODE_MEM_CACHE[key] = chosen
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        disk_file.write_bytes(pickle.dumps(chosen, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        pass
    return chosen


def _compute_group_index_fast(
    spec: WaveguideCrossSectionSpec,
    wavelength_m: float,
    bend_radius_m: float,
    pol: str,
    solver_backend: str,
    core_mesh_res_um: float,
    neff_center: float,
) -> float:
    """Fast 2-point centered finite-difference modal group index (grid export disabled)."""
    dl_m = max(wavelength_m * 4e-4, 0.5e-9)
    m_minus = _solve_single_mode(
        spec, wavelength_m - dl_m, bend_radius_m, pol, solver_backend, core_mesh_res_um, export_grid=False
    )
    m_plus = _solve_single_mode(
        spec, wavelength_m + dl_m, bend_radius_m, pol, solver_backend, core_mesh_res_um, export_grid=False
    )
    dn_dl = (float(np.real(m_plus.n_eff)) - float(np.real(m_minus.n_eff))) / (2.0 * dl_m)
    return float(neff_center - wavelength_m * dn_dl)


def _lateral_perturbation_kd_vs_gap(
    ring_mode: EigenmodeResult,
    bus_mode: EigenmodeResult,
    *,
    gaps_um: np.ndarray,
    ring_w_bottom_um: float,
    bus_w_bottom_um: float,
    film_h_um: float,
    slab_h_um: float,
    sidewall_deg: float,
    n_core_xyz: tuple[float, float, float],
    n_clad: float,
    wavelength_m: float,
    quad_step_um: float = 0.025,
) -> np.ndarray:
    """Evaluate exact 2D vector dielectric perturbation coupling rate k_d(g) [1/m] vs edge gap g [um]."""
    qs_m = quad_step_um * 1e-6
    wb_m = bus_w_bottom_um * 1e-6
    hf_m = film_h_um * 1e-6
    hs_m = slab_h_um * 1e-6

    x_loc = np.arange(-0.5 * wb_m + 0.5 * qs_m, 0.5 * wb_m, qs_m)
    y_loc = np.arange(hs_m + 0.5 * qs_m, hf_m, qs_m)
    xx, yy = np.meshgrid(x_loc, y_loc)
    inv_tan = 0.0 if sidewall_deg == 90.0 else 1.0 / math.tan(math.radians(sidewall_deg))
    hw = 0.5 * wb_m - (yy - hs_m) * inv_tan
    mask = np.abs(xx) < hw
    xr = xx[mask]
    yr = yy[mask]

    bus_pts = np.column_stack([yr, xr])
    eb_comps = []
    for arr in (bus_mode.Ex, bus_mode.Ey, bus_mode.Ez):
        itp = RegularGridInterpolator((bus_mode.y_m, bus_mode.x_m), arr, bounds_error=False, fill_value=0.0)
        eb_comps.append(itp(bus_pts))

    er_itps = [
        RegularGridInterpolator((ring_mode.y_m, ring_mode.x_m), arr, bounds_error=False, fill_value=0.0)
        for arr in (ring_mode.Ex, ring_mode.Ey, ring_mode.Ez)
    ]

    de = np.array([n_core_xyz[0] ** 2 - n_clad ** 2, n_core_xyz[1] ** 2 - n_clad ** 2, n_core_xyz[2] ** 2 - n_clad ** 2])
    omega = 2.0 * math.pi * C0_M_PER_S / wavelength_m
    prefac = EPS0_F_PER_M * omega * (qs_m ** 2) / (4.0 * math.sqrt(max(ring_mode.power_w * bus_mode.power_w, 1e-30)))

    kd_arr = np.empty(len(gaps_um), dtype=float)
    for idx, g_um in enumerate(gaps_um):
        dx_center_m = (0.5 * ring_w_bottom_um + float(g_um) + 0.5 * bus_w_bottom_um) * 1e-6
        r_pts = np.column_stack([yr, xr + dx_center_m])
        ov = sum(de[c] * np.vdot(eb_comps[c], er_itps[c](r_pts)) for c in range(3))
        kd_arr[idx] = abs(ov) * prefac
    return kd_arr


_SUPERMODE_MEM_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _twin_core_supermode_depol_ratio(
    *,
    ring_w_bottom_um: float,
    bus_w_bottom_um: float,
    film_h_um: float,
    slab_h_um: float,
    sidewall_deg: float,
    core_fn,
    clad_fn,
    top_fn,
    n_core_xyz: tuple[float, float, float],
    n_clad: float,
    wavelength_m: float,
    polarization: str,
    ring_mode_str: EigenmodeResult,
    bus_mode_str: EigenmodeResult,
    gaps_grid_um: np.ndarray,
    gap_focus_um: float,
    box_pad_um: float,
    eff_mesh_um: float,
    quad_step_um: float,
    n_top_clad: float | None = None,
) -> np.ndarray | None:
    """Compute exact 2D vector Maxwell twin-core supermode depolarization ratio `kappa_super(g) / kappa_pert(g)`.

    First-principles low-contrast perturbation CMT (`int Delta_eps E1* . E2 dA`) neglects the normal
    `D_x = eps * E_x` boundary jump across vertical sidewalls in high-index-contrast TE waveguides
    (Hardy & Streifer, IEEE JLT 1985; Huang, JOSA A 1994). Solving exact conformal 2D vector FEM
    twin-core supermodes at canonical gap anchors recovers exact vector coupling:
        kappa_super(g) = 0.5 * sqrt(max((beta_even - beta_odd)^2 - (beta_1,0 - beta_2,0)^2, 0))
    while preserving curved-ring centrifugal shift via the smooth ratio `kappa_super / kappa_pert_str`.
    """
    wl_um = wavelength_m * 1e6
    wl_scale = float(np.clip(wl_um / 1.55, 0.32, 4.50))
    c_val = core_fn(wavelength_m, T_REF_K)
    cl_val = clad_fn(wavelength_m, T_REF_K)
    tp_val = top_fn(wavelength_m, T_REF_K)
    n_bg_ridge = float(n_top_clad) if n_top_clad is not None else float(tp_val.nx)
    n_guided_floor = max(float(n_clad), n_bg_ridge)

    # Canonical waveguide-only gap anchors so gap sweeps on the same waveguide hit cache in < 1 ms
    g_lo = max(0.06, round(0.10 * wl_scale, 3))
    g_hi = round(0.24 * wl_scale, 3)
    anchors = [g_lo, g_hi]
    if gap_focus_um < g_lo - 0.02:
        anchors.insert(0, round(max(gap_focus_um, 0.05), 3))
    elif gap_focus_um > g_hi + 0.08:
        anchors.append(round(min(gap_focus_um, 0.55 * wl_scale), 3))
    anchors_arr = np.array(sorted(set(anchors)), dtype=float)

    # Quadratic Nedelec2xP2 twin-core splitting achieves O(h^4) accuracy at ~4-5 elements
    # across the core (~9-11 quadratic nodes), avoiding linear-element under-confinement bias.
    min_feat = max(min(ring_w_bottom_um, bus_w_bottom_um, film_h_um), 0.12)
    tw_mesh = round(float(np.clip(max(0.068 * wl_scale, min_feat / 4.2), 0.032, 0.240)), 4)
    cache_payload = {
        "v": "supermode_depol_v5",
        "w1": round(ring_w_bottom_um, 5),
        "w2": round(bus_w_bottom_um, 5),
        "hf": round(film_h_um, 5),
        "hs": round(slab_h_um, 5),
        "sw": round(sidewall_deg, 3),
        "nc": [round(c_val.nx, 6), round(c_val.ny, 6), round(c_val.nz, 6)],
        "ncl": round(cl_val.nx, 6),
        "ntp": round(tp_val.nx, 6),
        "wl_pm": round(wavelength_m * 1e12, 3),
        "pol": polarization.upper()[:2],
        "g": [round(float(x), 4) for x in anchors_arr],
        "res": tw_mesh,
    }
    ckey = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    cached = _SUPERMODE_MEM_CACHE.get(ckey)
    if cached is None:
        disk_f = _CACHE_DIR / f"sup_{ckey}.pkl"
        if disk_f.is_file():
            try:
                cached = pickle.loads(disk_f.read_bytes())
                _SUPERMODE_MEM_CACHE[ckey] = cached
            except Exception:
                cached = None

    if cached is None:
        want_te = polarization.upper().startswith("TE")
        k0 = 2.0 * math.pi / wavelength_m
        n1_0 = float(np.real(ring_mode_str.n_eff))
        n2_0 = float(np.real(bus_mode_str.n_eff))
        db0 = k0 * abs(n1_0 - n2_0)
        n_seed = max(n1_0, n2_0) * 1.035

        kd_p_str = _lateral_perturbation_kd_vs_gap(
            ring_mode_str,
            bus_mode_str,
            gaps_um=anchors_arr,
            ring_w_bottom_um=ring_w_bottom_um,
            bus_w_bottom_um=bus_w_bottom_um,
            film_h_um=film_h_um,
            slab_h_um=slab_h_um,
            sidewall_deg=sidewall_deg,
            n_core_xyz=n_core_xyz,
            n_clad=n_bg_ridge,
            wavelength_m=wavelength_m,
            quad_step_um=quad_step_um,
        )

        valid_g: list[float] = []
        valid_r: list[float] = []
        for idx_g, g_a in enumerate(anchors_arr):
            if kd_p_str[idx_g] <= 1e-3:
                continue
            dom_w = max(3.6 * wl_scale, ring_w_bottom_um + bus_w_bottom_um + float(g_a) + 1.8 * wl_um)
            tw_box = min(box_pad_um, 1.8 * wl_scale)
            sp_tw = WaveguideCrossSectionSpec(
                w_bottom_um=ring_w_bottom_um,
                h_film_um=film_h_um,
                h_slab_um=slab_h_um,
                sidewall_deg=sidewall_deg,
                core_material_fn=core_fn,
                clad_material_fn=clad_fn,
                top_clad_material_fn=top_fn,
                twin_core_gap_um=float(g_a),
                twin_core_w2_bottom_um=bus_w_bottom_um,
                domain_width_um=dom_w,
                h_box_um=tw_box,
                h_top_clad_um=tw_box,
            )
            try:
                ms = solve_modes_fem(
                    sp_tw,
                    wavelength_m,
                    num_modes=4,
                    n_guess=n_seed,
                    core_mesh_res_um=tw_mesh,
                    clad_mesh_res_um=max(tw_mesh * 6.5, 0.44 * wl_scale),
                    element_order=2,
                    export_grid=False,
                    mesh_backend="bsd_conformal",
                )
            except Exception:
                continue
            pol_ms = [
                m for m in ms
                if ((m.te_fraction >= 0.5) == want_te) and float(np.real(m.n_eff)) > n_guided_floor + 0.015
            ]
            pol_ms.sort(key=lambda m: -float(np.real(m.n_eff)))
            if len(pol_ms) < 2:
                continue
            ne1 = float(np.real(pol_ms[0].n_eff))
            ne2 = float(np.real(pol_ms[1].n_eff))
            db_sup = k0 * max(ne1 - ne2, 0.0)
            # Require clear supermode hybridization above isolated detuning floor
            if db_sup <= max(1.06 * db0, 80.0):
                continue
            kd_sup = 0.5 * math.sqrt(max(db_sup * db_sup - db0 * db0, 0.0))
            ratio = kd_sup / float(kd_p_str[idx_g])
            if 0.30 <= ratio <= 1.50:
                valid_g.append(float(g_a))
                valid_r.append(float(ratio))

        if len(valid_g) < 1:
            return None
        cached = (np.array(valid_g, dtype=float), np.array(valid_r, dtype=float))
        _SUPERMODE_MEM_CACHE[ckey] = cached
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            (_CACHE_DIR / f"sup_{ckey}.pkl").write_bytes(pickle.dumps(cached, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            pass

    vg, vr = cached
    if len(vg) == 1:
        return np.full_like(gaps_grid_um, float(vr[0]), dtype=float)
    return np.interp(gaps_grid_um, vg, vr, left=float(vr[0]), right=float(vr[-1]))


def _point_coupler_kappa_power_from_gap_curve(
    g0_um: float,
    radius_um: float,
    gaps_grid_um: np.ndarray,
    kd_grid_per_m: np.ndarray,
    beta_ring: float,
    beta_bus: float,
    ring2_radius_um: float | None = None,
    w_sum_half_um: float = 0.0,
) -> float:
    """Integrate spatial coupling k_d(g(z))*exp(i*Delta_phi(z)) along curved approach z.

    For a straight bus tangent to a circular ring of radius R_1 at center-to-center radial
    separation d_0 = (W_ring + W_bus)/2 + g_0 (Little et al., IEEE JLT 15(6), 998, 1997):
      - Radial distance to bus at longitudinal coordinate z: r(z) = sqrt((R_1 + d_0)^2 + z^2)
      - Local edge-to-edge gap: g(z) = g_0 + r(z) - (R_1 + d_0)
      - Exact ring azimuthal phase at bus: phi_ring(z) = (beta_ring * R_1) * atan(z / (R_1 + d_0))
      - Phase slip along bus: Delta_phi(z) = beta_bus * z - phi_ring(z)
    This exact cylindrical geometry captures both the angular-momentum lever-arm detuning
    beta_bus - beta_ring * R_1 / (R_1 + d_0) and all higher-order curvature phase dispersion.
    """
    r1_m = radius_um * 1e-6
    d0_m = max(g0_um + max(float(w_sum_half_um), 0.0), 0.0) * 1e-6
    if ring2_radius_um is None:
        r_bus0_m = r1_m + d0_m
        r_eff_m = r_bus0_m
    else:
        r2_m = ring2_radius_um * 1e-6
        r_eff_m = (r1_m * r2_m) / max(r1_m + r2_m, 1e-12)

    max_extra_gap_um = float(gaps_grid_um[-1] - g0_um)
    if max_extra_gap_um <= 0.02:
        return 0.0
    z_max_m = min(0.80 * r1_m, math.sqrt(2.0 * r_eff_m * (max_extra_gap_um * 1e-6)))
    z_m = np.linspace(-z_max_m, z_max_m, 201)
    dz_m = float(z_m[1] - z_m[0])

    if ring2_radius_um is None:
        r_z_m = np.sqrt(r_bus0_m ** 2 + z_m ** 2)
        g_z_um = g0_um + (r_z_m - r_bus0_m) * 1e6
        theta_z = np.arctan2(z_m, r_bus0_m)
        dphi_z = beta_bus * z_m - (beta_ring * r1_m) * theta_z
    else:
        r2_m = ring2_radius_um * 1e-6
        # Exact cylindrical coordinates of Ring 2 core (center at (0, r1 + r2 + d0)) as seen from Ring 1 (0, 0):
        #   y2(z) = (r1 + r2 + d0) - sqrt(r2^2 - z^2),  r1_to_c2(z) = sqrt(z^2 + y2(z)^2)
        #   Delta_phi(z) = (beta_2 * r2) * arcsin(z / r2) - (beta_1 * r1) * atan2(z, y2(z))
        # Captures the azimuthal lever-arm phase slip between Ring 1's outer tail and Ring 2's core even for r1 == r2,
        # and reduces identically to Little's straight-bus cylindrical formula in the limit r2 -> inf.
        y2_z_m = (r1_m + r2_m + d0_m) - np.sqrt(np.maximum(r2_m ** 2 - z_m ** 2, 0.0))
        r1_to_c2_m = np.sqrt(z_m ** 2 + y2_z_m ** 2)
        g_z_um = g0_um + (r1_to_c2_m - (r1_m + d0_m)) * 1e6
        dphi_z = (
            (beta_bus * r2_m) * np.arcsin(np.clip(z_m / max(r2_m, 1e-12), -0.9999, 0.9999))
            - (beta_ring * r1_m) * np.arctan2(z_m, y2_z_m)
        )

    kd_z = np.interp(g_z_um, gaps_grid_um, kd_grid_per_m, left=kd_grid_per_m[0], right=0.0)
    # Exact unitary 2x2 coupled-mode transfer-matrix integration along the curved approach:
    # d/dz [a1; a2] = -i [[0, k(z) exp(i dphi(z))]; [k(z) exp(-i dphi(z)), 0]] [a1; a2]
    # Preserves exact |a1|^2 + |a2|^2 = 1 across both weak and strong-coupling regimes.
    a1 = 1.0 + 0.0j
    a2 = 0.0 + 0.0j
    for k_v, ph_v in zip(kd_z, dphi_z):
        if k_v <= 0.0:
            continue
        ang = float(k_v) * dz_m
        c_v = math.cos(ang)
        s_v = math.sin(ang)
        ep = complex(math.cos(float(ph_v)), math.sin(float(ph_v)))
        na1 = c_v * a1 - 1j * s_v * ep * a2
        na2 = -1j * s_v * ep.conjugate() * a1 + c_v * a2
        a1, a2 = na1, na2
    return float(min(max(abs(a2) ** 2, 0.0), 1.0))


def _vernier_crow_spectra(
    wavelength_nm: np.ndarray,
    *,
    l1_um: float,
    l2_um: float,
    neff1_ref: float,
    ng1_ref: float,
    neff2_ref: float,
    ng2_ref: float,
    lambda_ref_nm: float,
    alpha_db_per_cm: float,
    kp_thru: float,
    kp_inter: float,
    kp_drop: float,
    delta_neff_tuning: complex = 0.0j,
) -> dict[str, np.ndarray]:
    """Exact 2nd-order series-coupled dual-ring (Vernier / CROW) transfer matrix."""
    wl_np = np.asarray(wavelength_nm, dtype=np.float64)
    wl_m = wl_np * 1e-9
    dlam = wl_np - lambda_ref_nm

    dn1 = -(ng1_ref - neff1_ref) / lambda_ref_nm
    dn2 = -(ng2_ref - neff2_ref) / lambda_ref_nm
    n1 = neff1_ref + dn1 * dlam + np.real(delta_neff_tuning)
    n2 = neff2_ref + dn2 * dlam + np.real(delta_neff_tuning)

    alpha_p = alpha_db_per_cm * 100.0 * (math.log(10.0) / 10.0) + (4.0 * math.pi * np.imag(delta_neff_tuning) / wl_m)
    l1_m, l2_m = l1_um * 1e-6, l2_um * 1e-6
    a1 = np.exp(-0.5 * alpha_p * l1_m)
    a2 = np.exp(-0.5 * alpha_p * l2_m)
    phi1 = (2.0 * math.pi * n1 / wl_m) * l1_m
    phi2 = (2.0 * math.pi * n2 / wl_m) * l2_m

    t1, k1 = math.sqrt(max(1.0 - kp_thru, 0.0)), math.sqrt(max(kp_thru, 0.0))
    t12, k12 = math.sqrt(max(1.0 - kp_inter, 0.0)), math.sqrt(max(kp_inter, 0.0))
    t2, k2 = math.sqrt(max(1.0 - kp_drop, 0.0)), math.sqrt(max(kp_drop, 0.0))

    # Effective reflection of Ring 2 loading Ring 1
    e1 = np.exp(1j * phi1)
    e2 = np.exp(1j * phi2)
    denom2 = 1.0 - t12 * t2 * a2 * e2
    tau_eff_ring2 = (t12 - t2 * a2 * e2) / denom2

    denom_tot = 1.0 - t1 * a1 * e1 * tau_eff_ring2
    thru_amp = (t1 - a1 * e1 * tau_eff_ring2) / denom_tot
    drop_amp = (-k1 * k12 * k2 * np.sqrt(a1 * a2) * np.exp(0.5j * (phi1 + phi2))) / (denom2 * denom_tot)

    return {
        "wavelength_nm": wl_np,
        "thru_T": np.abs(thru_amp) ** 2,
        "drop_T": np.abs(drop_amp) ** 2,
        "thru_phase_rad": np.unwrap(np.angle(thru_amp)),
    }


def _compute_active_tuning(
    cfg: MRRDesignConfig,
    mat_core_ref,
    mat_clad_ref,
    gamma_core: float,
    ng_eff: float,
    l_rt_um: float,
    is_te: bool,
) -> dict:
    """Compute Thermo-Optic (Delta_T), Pockels Electro-Optic (Voltage), and PCM shifts."""
    wl_nm = cfg.wavelength_nm

    # 1. Thermo-optic shift: Delta_n_eff = Delta_T * (Gamma * dn_core/dT + (1-Gamma) * dn_clad/dT)
    dndt_eff = gamma_core * mat_core_ref.dn_dt_per_k + (1.0 - gamma_core) * mat_clad_ref.dn_dt_per_k
    dneff_to = dndt_eff * cfg.delta_temp_k
    to_pm_per_k = (wl_nm * dndt_eff / max(ng_eff, 1e-6)) * 1e3

    # 2. Pockels electro-optic shift: Delta_n_EO = -0.5 * n^3 * r_eff * Gamma_EO * eta_duty * (V / d_elec)
    sym = mat_core_ref.crystal_symmetry
    if "Xcut" in sym:
        # X-cut TFLN: in-plane horizontal E-field couples to r33 for TE along Y-straight arms
        r_eff_pm_v = mat_core_ref.r33_pm_per_v if is_te else mat_core_ref.r13_pm_per_v
        n_active = mat_core_ref.nx if is_te else mat_core_ref.ny
        if cfg.topology.startswith("racetrack"):
            duty = (2.0 * cfg.straight_length_um) / max(l_rt_um, 1e-6)
        else:
            # In circular X-cut ring without segmented polarity reversal, push-pull covers ~2 straight-like quadrants
            duty = 0.50
    else:
        # c-axis vertical (Z-cut TFLN, AlN, ScAlN, 4H-SiC): vertical E_y couples to r33 (TM) and r13 (TE)
        r_eff_pm_v = mat_core_ref.r13_pm_per_v if is_te else mat_core_ref.r33_pm_per_v
        n_active = mat_core_ref.nx if is_te else mat_core_ref.ny
        duty = 1.0

    gamma_eo = 0.72 * gamma_core
    e_field_v_per_m = float(cfg.voltage_v) / max(cfg.electrode_gap_um * 1e-6, 1e-7)
    dneff_eo_per_v = -0.5 * (n_active ** 3) * (r_eff_pm_v * 1e-12) * gamma_eo * duty / max(cfg.electrode_gap_um * 1e-6, 1e-7)
    dneff_eo = dneff_eo_per_v * float(cfg.voltage_v)
    eo_pm_per_v = abs(wl_nm * dneff_eo_per_v / max(ng_eff, 1e-6)) * 1e3
    vpi_l_v_cm = (
        ((wl_nm * 1e-9) / max(abs(dneff_eo_per_v) * 2.0, 1e-30)) * 100.0
        if abs(r_eff_pm_v) > 1e-6 else None
    )

    total_dneff = complex(dneff_to + dneff_eo, 0.0)
    total_shift_pm = (wl_nm * np.real(total_dneff) / max(ng_eff, 1e-6)) * 1e3

    return {
        "dndt_modal_per_k": round(dndt_eff, 8),
        "thermo_optic_pm_per_k": round(to_pm_per_k, 3),
        "delta_temp_k": cfg.delta_temp_k,
        "eo_r_eff_pm_per_v": round(r_eff_pm_v, 3),
        "eo_duty_factor": round(duty, 3),
        "electro_optic_pm_per_v": round(eo_pm_per_v, 4),
        "Vpi_L_V_cm": round(vpi_l_v_cm, 3) if vpi_l_v_cm is not None else None,
        "voltage_v": cfg.voltage_v,
        "applied_electric_field_V_per_um": round(e_field_v_per_m * 1e-6, 4),
        "delta_neff_total_real": float(np.real(total_dneff)),
        "resonance_shift_pm": round(float(total_shift_pm), 3),
    }


def _log_abs_hankel2(m: float, z: np.ndarray) -> np.ndarray:
    """Overflow-safe exact `ln |H_m^(2)(z)|` for real azimuthal order `m > 0` and `z > 0`.

    Uses `scipy.special.hankel2` directly where finite (`m <= 120`) and seamlessly switches to
    Olver's uniform Airy / Debye asymptotic expansion (DLMF 10.20.4) at large `m`, avoiding
    IEEE-754 float64 overflow when `|Y_m(z)| > 1e308` on large-radius high-Q rings.
    """
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr)
    m_third = m ** (1.0 / 3.0)
    m_twothird = m ** (2.0 / 3.0)
    for i, zv in enumerate(z_arr):
        if zv <= 1e-12:
            out[i] = 1e6
            continue
        if m <= 120.0:
            hv = hankel2(m, zv)
            if np.isfinite(hv) and 1e-250 < abs(hv) < 1e250:
                out[i] = math.log(abs(hv))
                continue
        x = zv / m
        if x < 1.0 - 1e-7:
            sq = math.sqrt(max(1.0 - x * x, 0.0))
            xi = math.log((1.0 + sq) / x) - sq
            zeta = (1.5 * xi) ** (2.0 / 3.0)
            arg = m_twothird * zeta
            if arg > 18.0:
                out[i] = m * xi + 0.5 * math.log(2.0 / (math.pi * m * sq))
            else:
                ai, _, bi, _ = airy(arg)
                fac = (4.0 * zeta / max(1.0 - x * x, 1e-15)) ** 0.25
                out[i] = math.log(max(math.hypot(ai, bi) * fac / m_third, 1e-300))
        elif x > 1.0 + 1e-7:
            sq = math.sqrt(max(x * x - 1.0, 0.0))
            xi = sq - math.acos(1.0 / x)
            zeta = -((1.5 * xi) ** (2.0 / 3.0))
            arg = m_twothird * zeta
            if arg < -18.0:
                out[i] = 0.5 * math.log(2.0 / (math.pi * m * sq))
            else:
                ai, _, bi, _ = airy(arg)
                fac = (-4.0 * zeta / max(x * x - 1.0, 1e-15)) ** 0.25
                out[i] = math.log(max(math.hypot(ai, bi) * fac / m_third, 1e-300))
        else:
            ai0, _, bi0, _ = airy(0.0)
            out[i] = math.log(max(math.hypot(ai0, bi0) * (2.0 ** (1.0 / 3.0)) / m_third, 1e-300))
    return out


def _solve_1d_slab_fundamental(
    h_slab_m: float,
    wl_m: float,
    n_core: float,
    n_box: float,
    n_top: float,
    is_te: bool,
    y_m: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """Solve exact 1D 3-layer slab waveguide fundamental mode (`n_slab`, unit-L2 `phi(y)`)."""
    n_lo = max(n_box, n_top)
    if h_slab_m <= 1e-9 or n_core <= n_lo + 1e-4:
        return None
    k0 = 2.0 * math.pi / wl_m
    eta_b = 1.0 if is_te else (n_core / n_box) ** 2
    eta_t = 1.0 if is_te else (n_core / n_top) ** 2

    def _disp(neff: float) -> float:
        kap = k0 * math.sqrt(max(n_core * n_core - neff * neff, 1e-15))
        gb = k0 * math.sqrt(max(neff * neff - n_box * n_box, 0.0))
        gt = k0 * math.sqrt(max(neff * neff - n_top * n_top, 0.0))
        return kap * h_slab_m - math.atan(eta_b * gb / kap) - math.atan(eta_t * gt / kap)

    eps = 1e-6 * (n_core - n_lo)
    f_lo, f_hi = _disp(n_lo + eps), _disp(n_core - eps)
    if f_lo * f_hi > 0.0:
        return None
    n_slab = float(brentq(_disp, n_lo + eps, n_core - eps, xtol=1e-9))
    kap = k0 * math.sqrt(max(n_core * n_core - n_slab * n_slab, 1e-15))
    gb = k0 * math.sqrt(max(n_slab * n_slab - n_box * n_box, 0.0))
    gt = k0 * math.sqrt(max(n_slab * n_slab - n_top * n_top, 0.0))
    phi0 = math.atan(eta_b * gb / kap)
    phi_y = np.where(
        y_m < 0.0,
        math.cos(phi0) * np.exp(gb * y_m),
        np.where(
            y_m <= h_slab_m,
            np.cos(kap * y_m - phi0),
            math.cos(kap * h_slab_m - phi0) * np.exp(-gt * (y_m - h_slab_m)),
        ),
    )
    dy = float(np.mean(np.diff(y_m))) if len(y_m) > 1 else 1e-8
    norm = math.sqrt(max(float(np.sum(phi_y * phi_y) * dy), 1e-30))
    return n_slab, phi_y / norm


def _estimate_cylindrical_bend_radiation_loss(
    ring_bent_mode: EigenmodeResult,
    ring_spec: WaveguideCrossSectionSpec,
    *,
    radius_m: float,
    wavelength_m: float,
    w_ring_m: float,
    h_slab_m: float,
    n_core: float,
    n_box: float,
    n_top: float,
    polarization: str,
    solver_backend: str,
    eff_mesh_um: float,
    ng_rt: float,
) -> dict:
    """First-principles cylindrical bend radiation loss via exact Hankel-Wronskian outgoing flux.

    Evaluates outgoing cylindrical wave power `dP_rad / dtheta` by Fourier-decomposing the 2D bent
    eigenmode outer-cladding field (`Hy`, `Ey`) at reference radius `r0 = R + x0` onto exact
    Sommerfeld-outgoing cylindrical Hankel waves `H_m^(2)(k_r r)` (Marcuse 1976; Vassallo 1991),
    including exact 1D lateral slab-continuum leakage when `h_slab > 0`. Verified to `ratio = 1.0000`
    on exact 1D cylindrical Bessel slabs and within ~15-30% of Lumerical MODE FDE PML on tight SOI bends.
    """
    neff = float(np.real(ring_bent_mode.n_eff))
    k0 = 2.0 * math.pi / wavelength_m
    om = k0 * C0_M_PER_S
    is_te = polarization.upper().startswith("TE")

    # Identify slab-continuum mode (if rib waveguide) and dominant radiation caustic x_rad
    slab_info = (
        _solve_1d_slab_fundamental(h_slab_m, wavelength_m, n_core, n_box, n_top, is_te, ring_bent_mode.y_m)
        if h_slab_m > 1e-9 and len(ring_bent_mode.y_m) > 8
        else None
    )
    n_slab = slab_info[0] if (slab_info is not None and slab_info[0] < neff - 1e-4) else None
    n_clad_max = max(n_box, n_top)
    n_rad_dom = max(n_clad_max, n_slab or 0.0)
    if not math.isfinite(radius_m) or radius_m <= 0.0 or neff <= n_rad_dom + 1e-4:
        return {
            "alpha_bend_db_per_cm": 0.0,
            "bend_loss_db_per_90deg": 0.0,
            "Q_rad_bend": None,
            "caustic_offset_um": None,
            "radiation_channel": "cutoff_or_straight",
        }

    x_rad_m = radius_m * (neff / n_rad_dom - 1.0)
    r_um = radius_m * 1e6
    x_rad_um = x_rad_m * 1e6
    w_ring_um = w_ring_m * 1e6
    wl_scale = float(np.clip((wavelength_m * 1e9) / 1550.0, 0.35, 4.50))

    # On tight bends where the outer PEC wall lies beyond the radiation caustic, solve one fast
    # caustic-clamped reflection-free mode so `Hy(x0), Ey(x0)` is uncontaminated by PEC reflections.
    mode_eval = ring_bent_mode
    x_max_cur_um = ring_spec.x_bounds_um()[1]
    if x_rad_um < 1.05 * x_max_cur_um and x_rad_um > 0.5 * w_ring_um + 0.14 * wl_scale:
        xmax_clamp = min(x_max_cur_um, max(0.5 * w_ring_um + 0.45 * wl_scale, 0.88 * x_rad_um))
        if xmax_clamp < x_max_cur_um - 0.12:
            clamp_spec = WaveguideCrossSectionSpec(
                w_bottom_um=ring_spec.w_bottom_um,
                h_film_um=ring_spec.h_film_um,
                h_slab_um=ring_spec.h_slab_um,
                sidewall_deg=ring_spec.sidewall_deg,
                core_material_fn=ring_spec.core_material_fn,
                clad_material_fn=ring_spec.clad_material_fn,
                top_clad_material_fn=ring_spec.top_clad_material_fn,
                interlayer_h_um=ring_spec.interlayer_h_um,
                upper_core_h_um=ring_spec.upper_core_h_um,
                upper_w_bottom_um=ring_spec.upper_w_bottom_um,
                upper_sidewall_deg=ring_spec.upper_sidewall_deg,
                upper_core_material_fn=ring_spec.upper_core_material_fn,
                cap_h_um=ring_spec.cap_h_um,
                cap_w_um=ring_spec.cap_w_um,
                cap_material_fn=ring_spec.cap_material_fn,
                x_min_um=ring_spec.x_min_um,
                x_max_um=round(xmax_clamp, 3),
                h_box_um=ring_spec.h_box_um,
                h_top_clad_um=ring_spec.h_top_clad_um,
                temperature_k=ring_spec.temperature_k,
            )
            mode_eval = _solve_single_mode(
                clamp_spec, wavelength_m, radius_m, polarization, solver_backend, eff_mesh_um, export_grid=True
            )

    x_m = mode_eval.x_m
    y_m = mode_eval.y_m
    if len(x_m) < 8 or len(y_m) < 8:
        return {
            "alpha_bend_db_per_cm": 0.0,
            "bend_loss_db_per_90deg": 0.0,
            "Q_rad_bend": None,
            "caustic_offset_um": round(x_rad_um, 3),
            "radiation_channel": "none",
        }

    m_az = float(mode_eval.m_azimuthal)
    neff_ev = float(np.real(mode_eval.n_eff))
    dy = float(np.mean(np.diff(y_m)))
    x_rad_ev = radius_m * max(neff_ev / n_rad_dom - 1.0, 0.01)
    x0_target = min(
        0.5 * w_ring_m + 0.30e-6 * wl_scale,
        0.5 * (0.5 * w_ring_m + min(x_rad_ev, float(x_m[-1]) - 0.12e-6 * wl_scale)),
    )
    x0_target = max(x0_target, 0.5 * w_ring_m + 0.08e-6 * wl_scale)
    if x0_target >= float(x_m[-1]) - 0.04e-6:
        x0_target = float(x_m[-1]) - 0.08e-6
    ix = int(np.argmin(np.abs(x_m - x0_target)))
    x0 = float(x_m[ix])
    r0 = radius_m + x0

    hy_col = np.asarray(mode_eval.Hy[:, ix], dtype=complex)
    ey_col = np.asarray(mode_eval.Ey[:, ix], dtype=complex)

    def _hankel_bulk_channel(hy_vec: np.ndarray, ey_vec: np.ndarray, n_c: float) -> float:
        if neff_ev <= n_c + 1e-4:
            return 0.0
        n_fft = max(512, 4 * len(y_m))
        ky = 2.0 * math.pi * np.fft.fftfreq(n_fft, d=dy)
        dky = 2.0 * math.pi / (n_fft * dy)
        hy_ft = np.fft.fft(hy_vec, n=n_fft) * dy
        ey_ft = np.fft.fft(ey_vec, n=n_fft) * dy
        kr_sq = (k0 * n_c) ** 2 - ky ** 2
        mask = kr_sq > (0.05 * k0 * n_c) ** 2
        if not np.any(mask):
            return 0.0
        kr = np.sqrt(kr_sq[mask])
        lh2 = _log_abs_hankel2(m_az, kr * r0)
        inv_h2_sq = np.exp(np.clip(-2.0 * lh2, -700.0, 200.0))
        term_te = (om * MU0_H_PER_M / (math.pi * kr * kr)) * (np.abs(hy_ft[mask]) ** 2) * inv_h2_sq
        term_tm = (om * EPS0_F_PER_M * (n_c ** 2) / (math.pi * kr * kr)) * (np.abs(ey_ft[mask]) ** 2) * inv_h2_sq
        return float(np.sum(term_te + term_tm) * dky / (2.0 * math.pi))

    dp_dth_bulk = 0.0
    if abs(n_box - n_top) < 0.02:
        dp_dth_bulk = _hankel_bulk_channel(hy_col, ey_col, 0.5 * (n_box + n_top))
    else:
        lo_m = y_m <= 0.0
        hi_m = ~lo_m
        dp_dth_bulk += _hankel_bulk_channel(np.where(lo_m, hy_col, 0.0), np.where(lo_m, ey_col, 0.0), n_box)
        dp_dth_bulk += _hankel_bulk_channel(np.where(hi_m, hy_col, 0.0), np.where(hi_m, ey_col, 0.0), n_top)

    dp_dth_slab = 0.0
    if n_slab is not None and h_slab_m > 1e-9:
        slab_ev = _solve_1d_slab_fundamental(h_slab_m, wavelength_m, n_core, n_box, n_top, is_te, y_m)
        if slab_ev is not None and slab_ev[0] < neff_ev - 1e-4:
            ns_ev, phi_s = slab_ev
            kr_s = k0 * ns_ev
            lh2_s = float(_log_abs_hankel2(m_az, np.array([kr_s * r0]))[0])
            inv_h2_s = math.exp(max(min(-2.0 * lh2_s, 200.0), -700.0))
            if is_te:
                a_s = complex(np.sum(hy_col * phi_s) * dy)
                dp_dth_slab = (om * MU0_H_PER_M / (math.pi * kr_s * kr_s)) * (abs(a_s) ** 2) * inv_h2_s
            else:
                a_s = complex(np.sum(ey_col * phi_s) * dy)
                dp_dth_slab = (om * EPS0_F_PER_M * (ns_ev ** 2) / (math.pi * kr_s * kr_s)) * (abs(a_s) ** 2) * inv_h2_s

    dp_dth_tot = max(dp_dth_bulk + dp_dth_slab, 0.0)
    alpha_p_m = dp_dth_tot / max(radius_m * float(mode_eval.power_w), 1e-30)
    alpha_db_cm = alpha_p_m * 1e-2 * (10.0 / math.log(10.0))
    db_per_90 = alpha_p_m * (0.5 * math.pi * radius_m) * (10.0 / math.log(10.0))
    q_rad = (2.0 * math.pi * ng_rt) / (alpha_p_m * wavelength_m) if alpha_p_m > 1e-16 else None
    rad_chan = "slab_continuum" if dp_dth_slab > dp_dth_bulk else "bulk_cladding"
    return {
        "alpha_bend_db_per_cm": float(alpha_db_cm),
        "bend_loss_db_per_90deg": float(db_per_90),
        "Q_rad_bend": round(float(q_rad), 0) if (q_rad is not None and q_rad < 1e14) else None,
        "caustic_offset_um": round(x_rad_um, 3),
        "radiation_channel": rad_chan,
    }


def estimate_sidewall_scattering_payne_lacey(
    mode: EigenmodeResult,
    *,
    w_bottom_um: float,
    h_film_um: float,
    h_slab_um: float,
    sidewall_deg: float,
    n_core: float,
    n_clad_side: float,
    sigma_rms_nm: float,
    corr_length_nm: float = 50.0,
    roundtrip_length_um: float = 100.0,
    ng_ring: float = 2.1,
) -> dict[str, float]:
    """First-principles 3D Volume-Current Payne-Lacey / Barwicz-Haus sidewall scattering & backscattering.

    Computes sidewall roughness scattering attenuation `alpha_scattering_db_per_cm` and Lorentz-reciprocity
    RMS CW/CCW doublet splitting `backscatter_split_rms_mhz` directly from the 2D vector FEM sidewall
    electric field `(Ex, Ey, Ez)` for exponential autocorrelation `R(z) = sigma^2 * exp(-|z|/L_c)`
    (Payne & Lacey 1994; Barwicz & Haus JLT 2005; Morichetti et al. PRL 2010).
    """
    sig_m = max(float(sigma_rms_nm), 0.0) * 1e-9
    lc_m = max(float(corr_length_nm), 1.0) * 1e-9
    if sig_m <= 1e-13 or len(mode.x_m) < 8 or len(mode.y_m) < 8:
        return {"alpha_scattering_db_per_cm": 0.0, "backscatter_split_rms_mhz": 0.0}

    k0 = 2.0 * math.pi / mode.wavelength_m
    om = k0 * C0_M_PER_S
    beta = k0 * float(np.real(mode.n_eff))
    kc = k0 * n_clad_side
    deps = max(n_core * n_core - n_clad_side * n_clad_side, 0.0)
    if deps <= 1e-6:
        return {"alpha_scattering_db_per_cm": 0.0, "backscatter_split_rms_mhz": 0.0}

    x_m, y_m = mode.x_m, mode.y_m
    dy = float(np.mean(np.diff(y_m)))
    ymask = (y_m >= h_slab_um * 1e-6) & (y_m <= h_film_um * 1e-6)
    if int(np.count_nonzero(ymask)) < 2:
        return {"alpha_scattering_db_per_cm": 0.0, "backscatter_split_rms_mhz": 0.0}
    y_sub = y_m[ymask]

    inv_tan = 0.0 if abs(sidewall_deg - 90.0) < 1e-3 else 1.0 / math.tan(math.radians(sidewall_deg))
    # Average ridge half-width along the etched sidewall height
    half_w_mid_m = max(0.5 * (w_bottom_um - (h_film_um - h_slab_um) * inv_tan), 0.05) * 1e-6
    dx_sample = max(float(np.mean(np.diff(x_m))) * 1.2, 0.025e-6)

    # Exact Barwicz-Haus (2005) sidewall dielectric-interface Fresnel image screening:
    # tangential dipoles (py, pz) next to high-index core carry negative image r_s = (n_c - n_core)/(n_c + n_core),
    # giving 1 + r_s = 2*n_clad/(n_core + n_clad), while normal px uses core-screened E_{x,in} directly.
    t_par_img = (2.0 * n_clad_side) / max(n_core + n_clad_side, 1e-6)

    u_kz = np.linspace(-0.995, 0.995, 65)
    du = float(u_kz[1] - u_kz[0])
    kz = kc * u_kz
    krho = kc * np.sqrt(1.0 - u_kz ** 2)
    phi = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    r_psd = (2.0 * sig_m * sig_m * lc_m) / (1.0 + ((beta - kz) * lc_m) ** 2)
    ky_grid = krho[:, None] * np.sin(phi[None, :])
    phase_y = np.exp(-1j * ky_grid[:, :, None] * y_sub[None, None, :])

    dp_dz = 0.0
    back_sq_sum = 0.0
    for sgn in (-1.0, +1.0):
        xw = sgn * half_w_mid_m
        i_in = int(np.argmin(np.abs(x_m - (xw - sgn * dx_sample))))
        i_out = int(np.argmin(np.abs(x_m - (xw + sgn * dx_sample))))
        ex_in = mode.Ex[ymask, i_in]
        ex_out = mode.Ex[ymask, i_out]
        ey_w = 0.5 * (mode.Ey[ymask, i_in] + mode.Ey[ymask, i_out])
        ez_w = 0.5 * (mode.Ez[ymask, i_in] + mode.Ez[ymask, i_out])

        # Barwicz-Haus (2005) volume current radiating into cladding:
        # normal dipole uses core-screened E_{x,in}; tangential (py, pz) carry Fresnel image factor 1 + r_s = t_par_img
        px = EPS0_F_PER_M * deps * np.einsum("kpy,y->kp", phase_y, ex_in) * dy
        py = t_par_img * EPS0_F_PER_M * deps * np.einsum("kpy,y->kp", phase_y, ey_w) * dy
        pz = t_par_img * EPS0_F_PER_M * deps * np.einsum("kpy,y->kp", phase_y, ez_w) * dy
        p_perp_sq = np.mean(np.abs(px) ** 2 + np.abs(py) ** 2, axis=1)
        p_z_sq = np.mean(np.abs(pz) ** 2, axis=1)
        integrand = r_psd * (0.5 * (1.0 + u_kz ** 2) * p_perp_sq + (1.0 - u_kz ** 2) * p_z_sq)
        dp_dz += (om ** 4 * n_clad_side / (16.0 * math.pi * EPS0_F_PER_M * (C0_M_PER_S ** 3))) * float(np.sum(integrand) * du)

        # Lorentz-reciprocity CW<->CCW backscattering coupling (Johnson 2002; Ez flips sign on reversal)
        chi_b = (om * EPS0_F_PER_M * deps / (4.0 * max(mode.power_w, 1e-30))) * float(
            abs(np.sum((np.abs(ex_in * ex_out) + np.abs(ey_w) ** 2 - np.abs(ez_w) ** 2) * dy))
        )
        back_sq_sum += chi_b * chi_b

    alpha_p = dp_dz / max(mode.power_w, 1e-30)
    alpha_db_cm = alpha_p * 1e-2 * (10.0 / math.log(10.0))

    r_2b = (2.0 * sig_m * sig_m * lc_m) / (1.0 + ((2.0 * beta) * lc_m) ** 2)
    l_rt_m = max(roundtrip_length_um, 1.0) * 1e-6
    r_rt_rms = math.sqrt(max(l_rt_m * r_2b * back_sq_sum, 0.0))
    t_rt_s = max(ng_ring, 1.0) * l_rt_m / C0_M_PER_S
    split_mhz = (r_rt_rms / (math.pi * max(t_rt_s, 1e-18))) * 1e-6
    return {
        "alpha_scattering_db_per_cm": float(alpha_db_cm),
        "backscatter_split_rms_mhz": float(split_mhz),
    }


def simulate_mrr_topology(cfg: MRRDesignConfig) -> dict:
    """End-to-end Maxwell Eigenmode -> Spatial/Temporal CMT -> Broadband Spectrum synthesis."""
    t_sim_start = time.perf_counter()
    if cfg.topology not in TOPOLOGY_CATALOG:
        raise ValueError(f"Unknown topology '{cfg.topology}'. Supported: {list(TOPOLOGY_CATALOG)}")

    wl_m = cfg.wavelength_nm * 1e-9
    freq_hz = C0_M_PER_S / wl_m

    core_fn = get_material_fn(
        cfg.core_material,
        sc_fraction=cfg.sc_fraction if cfg.core_material.startswith("ScAlN") else None,
        custom_n=cfg.custom_n,
        custom_k=cfg.custom_k,
        custom_dn_dt=1.0e-5 if cfg.custom_dn_dt is None else float(cfg.custom_dn_dt),
        custom_r33_pm_v=0.0 if cfg.custom_r33_pm_v is None else float(cfg.custom_r33_pm_v),
    )
    clad_fn = get_material_fn(
        "Custom" if cfg.custom_clad_n is not None else cfg.clad_material,
        custom_n=cfg.custom_clad_n,
    )
    top_fn = get_material_fn(
        "Custom" if cfg.custom_top_clad_n is not None else cfg.top_clad_material,
        custom_n=cfg.custom_top_clad_n,
    )
    up_fn = get_material_fn(cfg.upper_core_material) if cfg.upper_core_material else None
    cap_fn = get_material_fn(cfg.cap_material) if cfg.cap_material else None

    alpha_db_cm = (
        float(cfg.alpha_loss_db_per_cm)
        if cfg.alpha_loss_db_per_cm is not None
        else float(
            MATERIAL_CATALOG.get(cfg.core_material, EXTRA_LITERATURE_MATERIALS.get(cfg.core_material, {})).get(
                "typical_loss_db_cm", 1.0
            )
        )
    )

    # Wavelength-adaptive domain box & mesh resolution across Visible (400 nm) -> O/C -> Mid-IR (12 um)
    wl_um = cfg.wavelength_nm * 1e-3
    wl_scale = float(np.clip(cfg.wavelength_nm / 1550.0, 0.32, 4.50))
    eff_mesh_um = (
        0.045
        if abs(cfg.core_mesh_res_um - 0.045) < 1e-6 and abs(wl_scale - 1.0) < 0.05
        else (
            round(float(np.clip(0.045 * wl_scale, 0.015, 0.150)), 4)
            if abs(cfg.core_mesh_res_um - 0.045) < 1e-6
            else float(cfg.core_mesh_res_um)
        )
    )
    min_core_dim_um = min(float(cfg.ring_width_um), float(cfg.film_height_um))
    if min_core_dim_um > 1.8 and abs(cfg.core_mesh_res_um - 0.045) < 1e-6:
        eff_mesh_um = max(eff_mesh_um, round(0.065 * min_core_dim_um, 4))
    uhq_gap_floor = round(2.20 * wl_scale, 3) if alpha_db_cm < 0.08 else 0.0
    box_pad_um = 2.5 if abs(wl_scale - 1.0) < 0.05 else round(max(1.8, 1.35 * wl_um), 3)
    max_gap_um = max(cfg.gap_thru_um, cfg.gap_drop_um, cfg.inter_ring_gap_um, round(0.72 * wl_um, 3), 0.70, uhq_gap_floor)
    x_max_ring_um = max(2.2 * wl_scale, 0.5 * cfg.ring_width_um + max_gap_um + cfg.bus_width_um + 0.65 * wl_um + 0.25)
    x_min_ring_um = -max(1.9 * wl_scale, 0.5 * cfg.ring_width_um + 1.15 * wl_um)
    if abs(wl_scale - 1.0) < 0.05:
        max_gap_um = max(cfg.gap_thru_um, cfg.gap_drop_um, cfg.inter_ring_gap_um, 1.10, uhq_gap_floor)
        x_max_ring_um = max(3.0, 0.5 * cfg.ring_width_um + max_gap_um + cfg.bus_width_um + 0.9)
        x_min_ring_um = -max(2.6, 0.5 * cfg.ring_width_um + 1.8)
    if cfg.ring_radius_um > 0:
        x_min_ring_um = max(x_min_ring_um, -0.82 * cfg.ring_radius_um)

    wavelength_warnings = check_wavelength_compatibility(
        cfg.core_material, cfg.clad_material, cfg.top_clad_material, cfg.wavelength_nm
    )

    has_partial_cap = (
        cfg.cap_h_um > 0.0
        and cap_fn is not None
        and cfg.cap_length_um is not None
        and float(cfg.cap_length_um) > 0.0
    )

    ring_spec = WaveguideCrossSectionSpec(
        w_bottom_um=cfg.ring_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        core_material_fn=core_fn,
        clad_material_fn=clad_fn,
        top_clad_material_fn=top_fn,
        interlayer_h_um=cfg.interlayer_h_um,
        upper_core_h_um=cfg.upper_core_h_um,
        upper_core_material_fn=up_fn,
        cap_h_um=0.0 if has_partial_cap else cfg.cap_h_um,
        cap_w_um=None if has_partial_cap else cfg.cap_w_um,
        cap_material_fn=None if has_partial_cap else cap_fn,
        x_min_um=x_min_ring_um,
        x_max_um=x_max_ring_um,
        h_box_um=box_pad_um,
        h_top_clad_um=box_pad_um,
        temperature_k=T_REF_K,
    )

    bus_dom_w = 5.2 if abs(wl_scale - 1.0) < 0.05 else max(3.4 * wl_scale, cfg.bus_width_um + 2.4 * wl_um)
    if uhq_gap_floor > 0.0:
        bus_dom_w = max(bus_dom_w, cfg.bus_width_um + 2.0 * max_gap_um + 1.6 * wl_scale)
    bus_spec = WaveguideCrossSectionSpec(
        w_bottom_um=cfg.bus_width_um,
        h_film_um=cfg.film_height_um,
        h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg,
        core_material_fn=core_fn,
        clad_material_fn=clad_fn,
        top_clad_material_fn=top_fn,
        interlayer_h_um=cfg.interlayer_h_um,
        upper_core_h_um=cfg.upper_core_h_um,
        upper_core_material_fn=up_fn,
        domain_width_um=max(bus_dom_w, cfg.bus_width_um + (3.6 if abs(wl_scale - 1.0) < 0.05 else 2.4 * wl_um)),
        h_box_um=box_pad_um,
        h_top_clad_um=box_pad_um,
        temperature_k=T_REF_K,
    )

    r1_m = cfg.ring_radius_um * 1e-6
    ring_bent_mode = _solve_single_mode(
        ring_spec, wl_m, r1_m, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=True
    )
    neff_bent = float(np.real(ring_bent_mode.n_eff))
    ng_bent = _compute_group_index_fast(
        ring_spec, wl_m, r1_m, cfg.polarization, cfg.solver_backend, eff_mesh_um, neff_bent
    )

    mat_core_val = core_fn(wl_m, T_REF_K)
    mat_clad_val = clad_fn(wl_m, T_REF_K)
    mat_top_val = top_fn(wl_m, T_REF_K)
    n_core_xyz = (mat_core_val.nx, mat_core_val.ny, mat_core_val.nz)
    # Etched waveguide ridges (y in [h_slab, h_film]) displace the upper cladding medium
    n_clad_scalar = mat_top_val.nx

    # Handle X-cut LiNbO3 azimuthal bend averaging & Racetrack straight sections
    if cfg.topology.startswith("racetrack"):
        ring_straight_mode = _solve_single_mode(
            ring_spec, wl_m, math.inf, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=True
        )
        neff_str = float(np.real(ring_straight_mode.n_eff))
        ng_str = _compute_group_index_fast(
            ring_spec, wl_m, math.inf, cfg.polarization, cfg.solver_backend, eff_mesh_um, neff_str
        )
        if cfg.core_material == "LN_Xcut":
            # Average Y-propagating bend start (phi=0) and X-propagating bend apex (phi=90 deg)
            apex_spec = WaveguideCrossSectionSpec(
                w_bottom_um=cfg.ring_width_um,
                h_film_um=cfg.film_height_um,
                h_slab_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg,
                core_material_fn=lambda w, t=T_REF_K: n_linbo3_xcut(w, t, inplane_propagation_angle_deg=90.0),
                clad_material_fn=clad_fn,
                top_clad_material_fn=top_fn,
                domain_width_um=max(5.5, 3.6 * wl_scale),
                h_box_um=box_pad_um,
                h_top_clad_um=box_pad_um,
            )
            m_apex = _solve_single_mode(
                apex_spec, wl_m, r1_m, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=False
            )
            neff_bent_avg = 0.5 * (neff_bent + float(np.real(m_apex.n_eff)))
            ng_bent_avg = ng_bent + (neff_bent_avg - neff_bent)
        else:
            neff_bent_avg = neff_bent
            ng_bent_avg = ng_bent

        l_arc_um = 2.0 * math.pi * cfg.ring_radius_um
        l_str_tot_um = 2.0 * cfg.straight_length_um
        l_rt_um = l_arc_um + l_str_tot_um
        neff_rt = (neff_bent_avg * l_arc_um + neff_str * l_str_tot_um) / l_rt_um
        ng_rt = (ng_bent_avg * l_arc_um + ng_str * l_str_tot_um) / l_rt_um
    else:
        ring_straight_mode = ring_bent_mode
        l_rt_um = 2.0 * math.pi * cfg.ring_radius_um
        neff_rt = neff_bent
        ng_rt = ng_bent

    # Intrinsic cavity decay & Q_int from propagation loss + core/cladding absorption + cylindrical bend radiation
    neff_imag_modal = float(abs(np.imag(ring_bent_mode.n_eff)))
    k_core_bulk = ring_bent_mode.core_confinement * max(mat_core_val.kx, 0.0)
    k_clad_bulk = (1.0 - ring_bent_mode.core_confinement) * 0.5 * (max(mat_clad_val.kx, 0.0) + max(mat_top_val.kx, 0.0))
    k_eff_mat = max(neff_imag_modal, k_core_bulk + k_clad_bulk)

    bend_rad = _estimate_cylindrical_bend_radiation_loss(
        ring_bent_mode,
        ring_spec,
        radius_m=r1_m,
        wavelength_m=wl_m,
        w_ring_m=cfg.ring_width_um * 1e-6,
        h_slab_m=cfg.slab_height_um * 1e-6,
        n_core=float(mat_core_val.nx if cfg.polarization.upper().startswith("TE") else mat_core_val.ny),
        n_box=float(mat_clad_val.nx),
        n_top=float(mat_top_val.nx),
        polarization=cfg.polarization,
        solver_backend=cfg.solver_backend,
        eff_mesh_um=eff_mesh_um,
        ng_rt=ng_rt,
    )
    alpha_bend_db_cm = float(bend_rad["alpha_bend_db_per_cm"])
    bend_loss_db_90 = float(bend_rad["bend_loss_db_per_90deg"])
    f_arc_bend = (2.0 * math.pi * cfg.ring_radius_um) / max(l_rt_um, 1e-9)
    alpha_bend_rt_per_m = f_arc_bend * alpha_bend_db_cm * 100.0 * (math.log(10.0) / 10.0)
    if alpha_bend_db_cm > 0.5:
        wavelength_warnings.append(
            f"Tight bend radiation loss: alpha_bend = {alpha_bend_db_cm:.2f} dB/cm "
            f"({bend_loss_db_90:.4g} dB/90deg via {bend_rad['radiation_channel']}) at R = {cfg.ring_radius_um:.2f} um; "
            f"increase ring_radius_um to raise radiation-limited Q_rad."
        )

    # Optional first-principles Payne-Lacey / Barwicz-Haus sidewall roughness scattering loss
    if float(cfg.sidewall_rms_roughness_nm) > 0.0:
        scat_res = estimate_sidewall_scattering_payne_lacey(
            ring_bent_mode,
            w_bottom_um=cfg.ring_width_um,
            h_film_um=cfg.film_height_um,
            h_slab_um=cfg.slab_height_um,
            sidewall_deg=cfg.sidewall_deg,
            n_core=float(mat_core_val.nx if cfg.polarization.upper().startswith("TE") else mat_core_val.ny),
            n_clad_side=float(mat_top_val.nx),
            sigma_rms_nm=cfg.sidewall_rms_roughness_nm,
            corr_length_nm=cfg.sidewall_corr_length_nm,
            roundtrip_length_um=l_rt_um,
            ng_ring=ng_rt,
        )
        alpha_scat_db_cm = float(scat_res["alpha_scattering_db_per_cm"])
        backscatter_split_mhz = float(scat_res["backscatter_split_rms_mhz"])
    else:
        alpha_scat_db_cm = 0.0
        backscatter_split_mhz = 0.0

    alpha_power_per_m = (
        (alpha_db_cm + alpha_scat_db_cm) * 100.0 * (math.log(10.0) / 10.0)
        + (4.0 * math.pi * k_eff_mat / wl_m)
        + alpha_bend_rt_per_m
    )

    cap_patch_summary: dict | None = None
    if has_partial_cap:
        l_cap_um = min(max(float(cfg.cap_length_um), 0.0), l_rt_um)
        duty_cap = l_cap_um / max(l_rt_um, 1e-9)
        cap_ring_spec = WaveguideCrossSectionSpec(
            w_bottom_um=cfg.ring_width_um,
            h_film_um=cfg.film_height_um,
            h_slab_um=cfg.slab_height_um,
            sidewall_deg=cfg.sidewall_deg,
            core_material_fn=core_fn,
            clad_material_fn=clad_fn,
            top_clad_material_fn=top_fn,
            interlayer_h_um=cfg.interlayer_h_um,
            upper_core_h_um=cfg.upper_core_h_um,
            upper_core_material_fn=up_fn,
            cap_h_um=cfg.cap_h_um,
            cap_w_um=cfg.cap_w_um,
            cap_material_fn=cap_fn,
            domain_width_um=max(5.2, cfg.ring_width_um + 3.6 * wl_scale),
            h_box_um=box_pad_um,
            h_top_clad_um=box_pad_um,
            temperature_k=T_REF_K,
        )
        r_cap_m = math.inf if cfg.topology.startswith("racetrack") else r1_m
        cap_mode = _solve_single_mode(
            cap_ring_spec, wl_m, r_cap_m, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=False
        )
        neff_cap_real = float(np.real(cap_mode.n_eff))
        neff_cap_imag = float(abs(np.imag(cap_mode.n_eff)))
        ng_cap = _compute_group_index_fast(
            cap_ring_spec, wl_m, r_cap_m, cfg.polarization, cfg.solver_backend, eff_mesh_um, neff_cap_real
        )
        ref_bare_neff = float(np.real(ring_straight_mode.n_eff)) if cfg.topology.startswith("racetrack") else neff_bent
        neff_rt = neff_rt + duty_cap * (neff_cap_real - ref_bare_neff)
        ng_rt = (1.0 - duty_cap) * ng_rt + duty_cap * ng_cap
        alpha_cap_excess_per_m = 4.0 * math.pi * max(neff_cap_imag - neff_imag_modal, 0.0) / wl_m
        alpha_power_per_m = alpha_power_per_m + duty_cap * alpha_cap_excess_per_m
        loss_db_per_um = (4.0 * math.pi * neff_cap_imag / wl_m) * 1e-6 * (10.0 / math.log(10.0))
        cap_patch_summary = {
            "cap_material": cfg.cap_material,
            "cap_h_um": cfg.cap_h_um,
            "cap_w_um": cfg.cap_w_um if cfg.cap_w_um is not None else cfg.ring_width_um,
            "cap_length_um": round(l_cap_um, 4),
            "duty_fraction": round(duty_cap, 6),
            "n_eff_cap_real": round(neff_cap_real, 6),
            "n_eff_cap_imag": round(neff_cap_imag, 7),
            "delta_neff_vs_bare": round(neff_cap_real - ref_bare_neff, 6),
            "modal_loss_db_per_um": round(loss_db_per_um, 5),
            "patch_insertion_loss_db": round(loss_db_per_um * l_cap_um, 4),
        }

    a_rt = math.exp(-0.5 * alpha_power_per_m * (l_rt_um * 1e-6))
    kp_intrinsic_loss = 1.0 - (a_rt ** 2)
    q_int = (2.0 * math.pi * ng_rt) / max(alpha_power_per_m * wl_m, 1e-18)

    is_4port = TOPOLOGY_CATALOG[cfg.topology]["ports"] == 4
    coupler_sweep: dict = {}
    vernier_summary: dict | None = None

    # ------------------------------------------------------------------
    # Synthesize Couplers per Topology
    # ------------------------------------------------------------------
    quad_cyl_m = float(np.clip(0.020e-6 * wl_scale, 0.008e-6, 0.055e-6))
    quad_lat_um = float(np.clip(0.025 * wl_scale, 0.010, 0.060))
    n_top_scalar = float(mat_top_val.nx)

    def _build_pulley_3seg_evaluator(
        gap_um_val: float,
        bus_mode_obj: EigenmodeResult,
        bus_r_m: float,
        k_cyl_abs: float,
    ):
        """Exact 3-segment continuous cylindrical unitary pulley coupler evaluator (`Entry Bend + Arc + Exit Bend`).

        Uses O(1) 2x2 unitary matrix factorization U_tot(theta) = U_ex(theta) @ U_arc(theta) @ U_en(theta)
        with exact polar radial separation r_bus(s) and ring azimuthal phase projection theta_ring(s)
        along outward return bends of radius R_out = bus_r_m, plus conformal twin-core supermode
        depolarization correction on high-contrast vertical sidewalls.
        """
        k_eff = float(k_cyl_abs)
        if cfg.solver_backend != "fde" and cfg.upper_core_h_um <= 0.0 and cfg.cap_h_um <= 0.0:
            ring_str_p = _solve_single_mode(
                ring_spec, wl_m, math.inf, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=True
            )
            bus_str_p = (
                ring_str_p
                if abs(cfg.ring_width_um - cfg.bus_width_um) <= 1e-5
                else _solve_single_mode(
                    bus_spec, wl_m, math.inf, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=True
                )
            )
            dr_p = _twin_core_supermode_depol_ratio(
                ring_w_bottom_um=cfg.ring_width_um,
                bus_w_bottom_um=cfg.bus_width_um,
                film_h_um=cfg.film_height_um,
                slab_h_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg,
                core_fn=core_fn,
                clad_fn=clad_fn,
                top_fn=top_fn,
                n_core_xyz=n_core_xyz,
                n_clad=n_clad_scalar,
                n_top_clad=n_top_scalar,
                wavelength_m=wl_m,
                polarization=cfg.polarization,
                ring_mode_str=ring_str_p,
                bus_mode_str=bus_str_p,
                gaps_grid_um=np.array([gap_um_val], dtype=float),
                gap_focus_um=gap_um_val,
                box_pad_um=box_pad_um,
                eff_mesh_um=eff_mesh_um,
                quad_step_um=quad_lat_um,
            )
            if dr_p is not None and len(dr_p) == 1:
                k_eff = k_eff * float(dr_p[0])

        k0_p = 2.0 * math.pi / wl_m
        beta_b_p = k0_p * float(np.real(bus_mode_obj.n_eff))
        beta_r_R_p = float(ring_bent_mode.m_azimuthal)
        db_arc = beta_b_p - (beta_r_R_p / max(bus_r_m, 1e-12))
        r_out_m = bus_r_m

        g_span = max(0.85 * wl_scale, gap_um_val + 0.55 * wl_scale)
        g_loc = np.linspace(max(0.05, min(gap_um_val - 0.02, 0.08 * wl_scale)), gap_um_val + g_span, 40)
        kd_loc = _lateral_perturbation_kd_vs_gap(
            ring_bent_mode,
            bus_mode_obj,
            gaps_um=g_loc,
            ring_w_bottom_um=cfg.ring_width_um,
            bus_w_bottom_um=cfg.bus_width_um,
            film_h_um=cfg.film_height_um,
            slab_h_um=cfg.slab_height_um,
            sidewall_deg=cfg.sidewall_deg,
            n_core_xyz=n_core_xyz,
            n_clad=n_top_scalar,
            wavelength_m=wl_m,
            quad_step_um=quad_lat_um,
        )
        kd_at_g0 = float(np.interp(gap_um_val, g_loc, kd_loc))
        u_en0 = np.eye(2, dtype=complex)
        u_ex0 = np.eye(2, dtype=complex)
        if k_eff > 1e-6 and kd_at_g0 > 1e-6 and bus_r_m > 1e-9:
            kd_scaled = kd_loc * (k_eff / kd_at_g0)
            r_eff_m = (bus_r_m * r_out_m) / max(bus_r_m + r_out_m, 1e-12)
            s_ext_m = min(0.75 * r_out_m, math.sqrt(2.0 * r_eff_m * (g_span * 1e-6)))
            if s_ext_m > 1e-9:
                n_seg = 160
                dso = s_ext_m / n_seg
                so_mid = (np.arange(n_seg, dtype=float) + 0.5) * dso
                po = so_mid / r_out_m
                r_cur = np.sqrt((bus_r_m + r_out_m) ** 2 + r_out_m ** 2 - 2.0 * (bus_r_m + r_out_m) * r_out_m * np.cos(po))
                g_out_um = gap_um_val + (r_cur - bus_r_m) * 1e6
                kd_out = np.interp(g_out_um, g_loc, kd_scaled, left=kd_scaled[0], right=0.0)
                dth_out = np.arctan2(r_out_m * np.sin(po), bus_r_m + r_out_m * (1.0 - np.cos(po)))
                dphi_exit = beta_b_p * so_mid - beta_r_R_p * dth_out
                for kv, phv in zip(kd_out, dphi_exit):
                    if kv <= 0.0:
                        continue
                    ang = float(kv) * dso
                    c_v, s_v = math.cos(ang), math.sin(ang)
                    ep = complex(math.cos(float(phv)), math.sin(float(phv)))
                    u_ex0 = np.array([[c_v, -1j * s_v * ep], [-1j * s_v * ep.conjugate(), c_v]], dtype=complex) @ u_ex0
                for kv, phv in zip(kd_out[::-1], -dphi_exit[::-1]):
                    if kv <= 0.0:
                        continue
                    ang = float(kv) * dso
                    c_v, s_v = math.cos(ang), math.sin(ang)
                    ep = complex(math.cos(float(phv)), math.sin(float(phv)))
                    u_en0 = np.array([[c_v, -1j * s_v * ep], [-1j * s_v * ep.conjugate(), c_v]], dtype=complex) @ u_en0

        psi_arc = math.hypot(k_eff, 0.5 * db_arc)

        def _eval_kp(theta_deg_val: float) -> float:
            l_arc_m = bus_r_m * math.radians(max(float(theta_deg_val), 0.0))
            ph_h = 0.5 * db_arc * l_arc_m
            ep_h = complex(math.cos(0.5 * ph_h), math.sin(0.5 * ph_h))
            em_h = ep_h.conjugate()
            d_p = np.array([[ep_h, 0.0], [0.0, em_h]], dtype=complex)
            d_m = np.array([[em_h, 0.0], [0.0, ep_h]], dtype=complex)
            if psi_arc > 1e-12 and l_arc_m > 0.0:
                c_a = math.cos(psi_arc * l_arc_m)
                s_a = math.sin(psi_arc * l_arc_m)
                m_rot = np.array(
                    [
                        [c_a - 1j * (0.5 * db_arc / psi_arc) * s_a, -1j * (k_eff / psi_arc) * s_a],
                        [-1j * (k_eff / psi_arc) * s_a, c_a + 1j * (0.5 * db_arc / psi_arc) * s_a],
                    ],
                    dtype=complex,
                )
            else:
                m_rot = np.eye(2, dtype=complex)
            u_arc = d_p @ m_rot @ d_p
            u_tot = (d_p @ u_ex0 @ d_m) @ u_arc @ (d_m @ u_en0 @ d_p)
            return float(min(max(abs(u_tot[1, 0]) ** 2, 0.0), 0.999999))

        return _eval_kp, k_eff

    if cfg.topology in ("pulley_ap", "pulley_ad"):
        bus_r_thru_m = (cfg.ring_radius_um + 0.5 * cfg.ring_width_um + cfg.gap_thru_um + 0.5 * cfg.bus_width_um) * 1e-6
        bus_mode_thru = _solve_single_mode(
            bus_spec, wl_m, bus_r_thru_m, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=True
        )
        cc_dict = coupling_coefficient_cylindrical(
            ring_bent_mode.to_spatial_cmt_field_dict(),
            bus_mode_thru.to_spatial_cmt_field_dict(),
            wavelength_m=wl_m,
            ring_radius_m=r1_m,
            bus_radius_m=bus_r_thru_m,
            bus_bottom_width_m=cfg.bus_width_um * 1e-6,
            film_height_m=cfg.film_height_um * 1e-6,
            slab_height_m=cfg.slab_height_um * 1e-6,
            sidewall_deg=cfg.sidewall_deg,
            n_core_xyz=n_core_xyz,
            n_clad=n_clad_scalar,
            n_top_clad=n_top_scalar,
            quad_step_m=quad_cyl_m,
            ring_azimuthal_m=ring_bent_mode.m_azimuthal,
            bus_neff=float(np.real(bus_mode_thru.n_eff)),
        )
        eval_thru_kp, k_eff_thru = _build_pulley_3seg_evaluator(
            cfg.gap_thru_um, bus_mode_thru, bus_r_thru_m, abs(cc_dict["k_per_m"])
        )
        kp_thru = eval_thru_kp(cfg.pulley_angle_thru_deg)

        if is_4port:
            bus_r_drop_m = (cfg.ring_radius_um + 0.5 * cfg.ring_width_um + cfg.gap_drop_um + 0.5 * cfg.bus_width_um) * 1e-6
            if abs(cfg.gap_drop_um - cfg.gap_thru_um) < 1e-6:
                eval_drop_kp = eval_thru_kp
            else:
                bus_mode_drop = _solve_single_mode(
                    bus_spec, wl_m, bus_r_drop_m, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=True
                )
                cc_d = coupling_coefficient_cylindrical(
                    ring_bent_mode.to_spatial_cmt_field_dict(),
                    bus_mode_drop.to_spatial_cmt_field_dict(),
                    wavelength_m=wl_m,
                    ring_radius_m=r1_m,
                    bus_radius_m=bus_r_drop_m,
                    bus_bottom_width_m=cfg.bus_width_um * 1e-6,
                    film_height_m=cfg.film_height_um * 1e-6,
                    slab_height_m=cfg.slab_height_um * 1e-6,
                    sidewall_deg=cfg.sidewall_deg,
                    n_core_xyz=n_core_xyz,
                    n_clad=n_clad_scalar,
                    n_top_clad=n_top_scalar,
                    quad_step_m=quad_cyl_m,
                    ring_azimuthal_m=ring_bent_mode.m_azimuthal,
                    bus_neff=float(np.real(bus_mode_drop.n_eff)),
                )
                eval_drop_kp, _ = _build_pulley_3seg_evaluator(
                    cfg.gap_drop_um, bus_mode_drop, bus_r_drop_m, abs(cc_d["k_per_m"])
                )
            kp_drop = eval_drop_kp(cfg.pulley_angle_drop_deg)
        else:
            kp_drop = 0.0

        # Exact 3-segment sweep across drawn wrap angles theta_drawn in [0.5, 180.0] deg
        ang_sweep = np.arange(0.5, 180.01, 0.5, dtype=float)
        kp_sweep_p = np.array([eval_thru_kp(float(th)) for th in ang_sweep], dtype=float)
        target_kp_p = 1.0 - (a_rt ** 2) * (1.0 - kp_drop)
        q_ext_sweep_p = (2.0 * math.pi * ng_rt * (l_rt_um * 1e-6) / wl_m) / np.maximum(
            -np.log1p(-np.clip(kp_sweep_p, 1e-12, 0.9999)), 1e-12
        )
        cp_sweep_p = kp_sweep_p / max(target_kp_p, 1e-12)
        diff_p = kp_sweep_p - target_kp_p
        cc_found_p = bool(np.any(diff_p[:-1] * diff_p[1:] <= 0.0))
        if cc_found_p:
            cc_val = float(ang_sweep[int(np.argmin(np.abs(diff_p)))])
            for j_p in range(len(ang_sweep) - 1):
                if diff_p[j_p] * diff_p[j_p + 1] <= 0.0:
                    cc_val = float(brentq(lambda th: eval_thru_kp(th) - target_kp_p, float(ang_sweep[j_p]), float(ang_sweep[j_p + 1]), xtol=1e-3))
                    break
        else:
            cc_val = float(ang_sweep[int(np.argmax(kp_sweep_p))])
        coupler_sweep = {
            "param_name": "pulley_angle_thru_deg",
            "param_unit": "deg",
            "x_values": np.round(ang_sweep, 3).tolist(),
            "Q_ext": np.round(np.minimum(q_ext_sweep_p, 1e11), 1).tolist(),
            "coupling_ratio_Cp": np.round(cp_sweep_p, 4).tolist(),
            "critical_coupling_value": round(cc_val, 3),
            "critical_coupling_found": cc_found_p,
            "k_per_m_abs": round(k_eff_thru, 2),
            "delta_beta_per_m": round(float(cc_dict["delta_beta_per_m"]), 2),
        }

    else:
        # Straight bus eigenmode + lateral gap perturbation curve k_d(g)
        bus_mode_str = _solve_single_mode(
            bus_spec, wl_m, math.inf, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=True
        )
        g_min_um = max(0.05, round(0.08 * min(wl_scale, 1.5), 3))
        g_max_um = max(max_gap_um + 0.50 * wl_scale, 0.95 * wl_scale, 0.70)
        gaps_grid = np.linspace(g_min_um, g_max_um, 48)
        ref_ring_for_coupler = ring_straight_mode if cfg.topology.startswith("racetrack") else ring_bent_mode
        kd_grid = _lateral_perturbation_kd_vs_gap(
            ref_ring_for_coupler,
            bus_mode_str,
            gaps_um=gaps_grid,
            ring_w_bottom_um=cfg.ring_width_um,
            bus_w_bottom_um=cfg.bus_width_um,
            film_h_um=cfg.film_height_um,
            slab_h_um=cfg.slab_height_um,
            sidewall_deg=cfg.sidewall_deg,
            n_core_xyz=n_core_xyz,
            n_clad=n_top_scalar,
            wavelength_m=wl_m,
            quad_step_um=quad_lat_um,
        )
        kd_bent_ends_grid: np.ndarray | None = None
        if cfg.topology.startswith("racetrack"):
            kd_bent_ends_grid = _lateral_perturbation_kd_vs_gap(
                ring_bent_mode,
                bus_mode_str,
                gaps_um=gaps_grid,
                ring_w_bottom_um=cfg.ring_width_um,
                bus_w_bottom_um=cfg.bus_width_um,
                film_h_um=cfg.film_height_um,
                slab_h_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg,
                n_core_xyz=n_core_xyz,
                n_clad=n_top_scalar,
                wavelength_m=wl_m,
                quad_step_um=quad_lat_um,
            )
        # Apply exact conformal twin-core 2D vector FEM supermode depolarization correction
        # for single-layer strip/rib/trapezoid waveguides (resolves normal D_x jump in high-contrast TE).
        if cfg.solver_backend != "fde" and cfg.upper_core_h_um <= 0.0 and cfg.cap_h_um <= 0.0:
            ring_str_for_depol = (
                bus_mode_str
                if abs(cfg.ring_width_um - cfg.bus_width_um) <= 1e-5
                else (
                    ring_straight_mode
                    if cfg.topology.startswith("racetrack")
                    else _solve_single_mode(
                        ring_spec, wl_m, math.inf, cfg.polarization, cfg.solver_backend, eff_mesh_um, export_grid=True
                    )
                )
            )
            depol_vec = _twin_core_supermode_depol_ratio(
                ring_w_bottom_um=cfg.ring_width_um,
                bus_w_bottom_um=cfg.bus_width_um,
                film_h_um=cfg.film_height_um,
                slab_h_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg,
                core_fn=core_fn,
                clad_fn=clad_fn,
                top_fn=top_fn,
                n_core_xyz=n_core_xyz,
                n_clad=n_clad_scalar,
                n_top_clad=n_top_scalar,
                wavelength_m=wl_m,
                polarization=cfg.polarization,
                ring_mode_str=ring_str_for_depol,
                bus_mode_str=bus_mode_str,
                gaps_grid_um=gaps_grid,
                gap_focus_um=cfg.gap_thru_um,
                box_pad_um=box_pad_um,
                eff_mesh_um=eff_mesh_um,
                quad_step_um=quad_lat_um,
            )
            if depol_vec is not None:
                kd_grid = kd_grid * depol_vec
                if kd_bent_ends_grid is not None:
                    kd_bent_ends_grid = kd_bent_ends_grid * depol_vec
        k0_wave = 2.0 * math.pi / wl_m
        beta_ring_c = k0_wave * float(np.real(ref_ring_for_coupler.n_eff))
        beta_ring_bent_c = k0_wave * float(np.real(ring_bent_mode.n_eff))
        beta_bus_c = k0_wave * float(np.real(bus_mode_str.n_eff))

        w_half_rb = 0.5 * (cfg.ring_width_um + cfg.bus_width_um)
        if cfg.topology in ("point_circular_ap", "point_circular_ad"):
            kp_thru = _point_coupler_kappa_power_from_gap_curve(
                cfg.gap_thru_um, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c, w_sum_half_um=w_half_rb
            )
            kp_drop = (
                _point_coupler_kappa_power_from_gap_curve(
                    cfg.gap_drop_um, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c, w_sum_half_um=w_half_rb
                )
                if is_4port else 0.0
            )
            g_sweep = np.linspace(g_min_um, g_max_um * 0.92, 48)
            kp_sweep = np.array([
                _point_coupler_kappa_power_from_gap_curve(float(g), cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c, w_sum_half_um=w_half_rb)
                for g in g_sweep
            ])
            target_kp = 1.0 - (a_rt ** 2) * (1.0 - kp_drop)
            q_ext_sweep = (2.0 * math.pi * ng_rt * (l_rt_um * 1e-6) / wl_m) / np.maximum(-np.log1p(-np.clip(kp_sweep, 1e-12, 0.9999)), 1e-12)
            cp_sweep = kp_sweep / max(target_kp, 1e-12)
            idx_cc = int(np.argmin(np.abs(kp_sweep - target_kp)))
            cc_found = bool((kp_sweep.min() <= target_kp <= kp_sweep.max()))
            if cc_found:
                # Refine via brentq
                for j in range(len(g_sweep) - 1):
                    if (kp_sweep[j] - target_kp) * (kp_sweep[j + 1] - target_kp) <= 0:
                        cc_gap = float(brentq(
                            lambda g: _point_coupler_kappa_power_from_gap_curve(g, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c, w_sum_half_um=w_half_rb) - target_kp,
                            float(g_sweep[j]), float(g_sweep[j + 1]), xtol=1e-4,
                        ))
                        break
                else:
                    cc_gap = float(g_sweep[idx_cc])
            else:
                cc_gap = float(g_sweep[idx_cc])
            coupler_sweep = {
                "param_name": "gap_thru_um",
                "param_unit": "um",
                "x_values": np.round(g_sweep, 4).tolist(),
                "Q_ext": np.round(np.minimum(q_ext_sweep, 1e11), 1).tolist(),
                "coupling_ratio_Cp": np.round(cp_sweep, 4).tolist(),
                "critical_coupling_value": round(cc_gap, 4),
                "critical_coupling_found": cc_found,
            }

        elif cfg.topology in ("racetrack_ap", "racetrack_ad"):
            def _rt_kp(ls_um: float, gap_um: float) -> float:
                kd = float(np.interp(gap_um, gaps_grid, kd_grid))
                kd_end_curve = kd_bent_ends_grid if kd_bent_ends_grid is not None else kd_grid
                kp_end = _point_coupler_kappa_power_from_gap_curve(
                    gap_um, cfg.ring_radius_um, gaps_grid, kd_end_curve, beta_ring_bent_c, beta_bus_c, w_sum_half_um=w_half_rb
                )
                theta_end = math.asin(math.sqrt(min(max(kp_end, 0.0), 0.9999)))
                db = 0.5 * (beta_bus_c - beta_ring_c)
                psi = math.hypot(kd, db)
                if psi < 1e-9:
                    return 0.0
                l_eff_m = (ls_um * 1e-6) + (theta_end / max(kd, 1e-6))
                return float(min((kd / psi) ** 2 * (math.sin(psi * l_eff_m) ** 2), 0.9999))

            kp_thru = _rt_kp(cfg.straight_length_um, cfg.gap_thru_um)
            kp_drop = _rt_kp(cfg.straight_length_um, cfg.gap_drop_um) if is_4port else 0.0

            ls_sweep = np.linspace(0.5, max(cfg.straight_length_um * 2.5, 45.0 * wl_scale), 60)
            kp_sweep = np.array([_rt_kp(float(ls), cfg.gap_thru_um) for ls in ls_sweep])
            l_rt_sweep_m = (2.0 * math.pi * cfg.ring_radius_um + 2.0 * ls_sweep) * 1e-6
            target_kp_vec = 1.0 - np.exp(-alpha_power_per_m * l_rt_sweep_m) * (1.0 - kp_drop)
            q_ext_sweep = (2.0 * math.pi * ng_rt * l_rt_sweep_m / wl_m) / np.maximum(-np.log1p(-np.clip(kp_sweep, 1e-12, 0.9999)), 1e-12)
            cp_sweep = kp_sweep / np.maximum(target_kp_vec, 1e-12)
            diff_vec = kp_sweep - target_kp_vec
            cc_found = bool(np.any(diff_vec[:-1] * diff_vec[1:] <= 0.0))
            cc_ls = float(ls_sweep[int(np.argmin(np.abs(diff_vec)))])
            if cc_found:
                for j in range(len(ls_sweep) - 1):
                    if diff_vec[j] * diff_vec[j + 1] <= 0.0:
                        cc_ls = float(brentq(
                            lambda ls: _rt_kp(ls, cfg.gap_thru_um) - (1.0 - math.exp(-alpha_power_per_m * (2.0 * math.pi * cfg.ring_radius_um + 2.0 * ls) * 1e-6) * (1.0 - kp_drop)),
                            float(ls_sweep[j]), float(ls_sweep[j + 1]), xtol=1e-3,
                        ))
                        break
            coupler_sweep = {
                "param_name": "straight_length_um",
                "param_unit": "um",
                "x_values": np.round(ls_sweep, 3).tolist(),
                "Q_ext": np.round(np.minimum(q_ext_sweep, 1e11), 1).tolist(),
                "coupling_ratio_Cp": np.round(cp_sweep, 4).tolist(),
                "critical_coupling_value": round(cc_ls, 3),
                "critical_coupling_found": cc_found,
            }

        else:
            # Dual-Ring Vernier / CROW (`vernier_crow_ad`)
            r2_m = cfg.ring2_radius_um * 1e-6
            if abs(cfg.ring2_radius_um - cfg.ring_radius_um) < 1e-3:
                ring2_mode = ring_bent_mode
                neff_r2, ng_r2 = neff_bent, ng_bent
            else:
                ring2_mode = _solve_single_mode(
                    ring_spec, wl_m, r2_m, cfg.polarization, cfg.solver_backend, cfg.core_mesh_res_um, export_grid=True
                )
                neff_r2 = float(np.real(ring2_mode.n_eff))
                ng_r2 = _compute_group_index_fast(
                    ring_spec, wl_m, r2_m, cfg.polarization, cfg.solver_backend, cfg.core_mesh_res_um, neff_r2
                )
            # Mirror Ring 2 laterally (x -> -x) so its centrifugally shifted outer tail physically faces Ring 1's outer edge
            import copy
            ring2_facing = copy.copy(ring2_mode)
            ring2_facing.x_m = -ring2_mode.x_m[::-1]
            ring2_facing.Ex = -ring2_mode.Ex[:, ::-1]
            ring2_facing.Ey = ring2_mode.Ey[:, ::-1]
            ring2_facing.Ez = ring2_mode.Ez[:, ::-1]
            kd_inter_grid = _lateral_perturbation_kd_vs_gap(
                ring_bent_mode,
                ring2_facing,
                gaps_um=gaps_grid,
                ring_w_bottom_um=cfg.ring_width_um,
                bus_w_bottom_um=cfg.ring_width_um,
                film_h_um=cfg.film_height_um,
                slab_h_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg,
                n_core_xyz=n_core_xyz,
                n_clad=n_top_scalar,
                wavelength_m=wl_m,
                quad_step_um=quad_lat_um,
            )
            if depol_vec is not None and abs(cfg.ring_width_um - cfg.bus_width_um) <= 1e-5:
                kd_inter_grid = kd_inter_grid * depol_vec
            kp_thru = _point_coupler_kappa_power_from_gap_curve(
                cfg.gap_thru_um, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c, w_sum_half_um=w_half_rb
            )
            kp_drop = _point_coupler_kappa_power_from_gap_curve(
                cfg.gap_drop_um, cfg.ring2_radius_um, gaps_grid, kd_grid, k0_wave * neff_r2, beta_bus_c, w_sum_half_um=w_half_rb
            )
            kp_inter = _point_coupler_kappa_power_from_gap_curve(
                cfg.inter_ring_gap_um,
                cfg.ring_radius_um,
                gaps_grid,
                kd_inter_grid,
                beta_ring_c,
                k0_wave * neff_r2,
                ring2_radius_um=cfg.ring2_radius_um,
                w_sum_half_um=cfg.ring_width_um,
            )
            # Optimal maximally-flat 2nd-order CROW condition: kp_inter_opt ~ (kp_thru * kp_drop) / 4 + loss
            kp_inter_opt = min(0.28 * (kp_thru + kp_drop) ** 2 + kp_intrinsic_loss ** 2, 0.5)
            g12_sweep = np.linspace(0.10, 0.95, 40)
            kp12_sweep = np.array([
                _point_coupler_kappa_power_from_gap_curve(
                    float(g), cfg.ring_radius_um, gaps_grid, kd_inter_grid, beta_ring_c, k0_wave * neff_r2, ring2_radius_um=cfg.ring2_radius_um, w_sum_half_um=cfg.ring_width_um
                )
                for g in g12_sweep
            ])
            opt_g12 = float(g12_sweep[int(np.argmin(np.abs(kp12_sweep - kp_inter_opt)))])
            l2_um = 2.0 * math.pi * cfg.ring2_radius_um
            fsr1_nm = ((wl_m ** 2) / (ng_rt * (l_rt_um * 1e-6))) * 1e9
            fsr2_nm = ((wl_m ** 2) / (ng_r2 * (l2_um * 1e-6))) * 1e9
            fsr_vernier_nm = (fsr1_nm * fsr2_nm / abs(fsr1_nm - fsr2_nm)) if abs(fsr1_nm - fsr2_nm) > 1e-4 else fsr1_nm
            vernier_summary = {
                "R1_um": cfg.ring_radius_um,
                "R2_um": cfg.ring2_radius_um,
                "FSR1_nm": round(fsr1_nm, 4),
                "FSR2_nm": round(fsr2_nm, 4),
                "FSR_Vernier_nm": round(fsr_vernier_nm, 3),
                "kappa_inter_power": round(kp_inter, 6),
            }
            coupler_sweep = {
                "param_name": "inter_ring_gap_um",
                "param_unit": "um",
                "x_values": np.round(g12_sweep, 4).tolist(),
                "Q_ext": np.round(1.0 / np.maximum(kp12_sweep, 1e-8), 1).tolist(),
                "coupling_ratio_Cp": np.round(kp12_sweep / max(kp_inter_opt, 1e-9), 4).tolist(),
                "critical_coupling_value": round(opt_g12, 4),
                "critical_coupling_found": True,
            }

    # ------------------------------------------------------------------
    # Active Tuning + Broadband Spectrum Computation (GPU / AMX / CPU)
    # ------------------------------------------------------------------
    is_te = ring_bent_mode.te_fraction >= 0.5
    tuning_info = _compute_active_tuning(
        cfg,
        mat_core_val,
        mat_clad_val,
        ring_bent_mode.core_confinement,
        ng_rt,
        l_rt_um,
        is_te,
    )
    dneff_complex = complex(tuning_info["delta_neff_total_real"], 0.0)

    # Compute analytical Q, FSR, Finesse & FWHM first so high-Q dips get adaptive sub-pm grid points
    fsr_nm = ((wl_m ** 2) / (ng_rt * (l_rt_um * 1e-6))) * 1e9
    fsr_ghz = (C0_M_PER_S / (ng_rt * (l_rt_um * 1e-6))) * 1e-9
    q_ext_thru = (2.0 * math.pi * ng_rt * (l_rt_um * 1e-6) / wl_m) / max(-math.log1p(-min(kp_thru, 0.999999)), 1e-15)
    q_ext_drop = (
        (2.0 * math.pi * ng_rt * (l_rt_um * 1e-6) / wl_m) / max(-math.log1p(-min(kp_drop, 0.999999)), 1e-15)
        if kp_drop > 1e-12 else float("inf")
    )
    inv_q_tot = (1.0 / max(q_int, 1.0)) + (1.0 / max(q_ext_thru, 1.0)) + (0.0 if not math.isfinite(q_ext_drop) else 1.0 / q_ext_drop)
    q_loaded = 1.0 / inv_q_tot
    fwhm_pm = (cfg.wavelength_nm / q_loaded) * 1e3
    fwhm_nm = fwhm_pm * 1e-3
    finesse = fsr_nm / max(fwhm_nm, 1e-12)

    wl_min_nm = cfg.wavelength_nm - 0.5 * cfg.span_nm
    wl_max_nm = cfg.wavelength_nm + 0.5 * cfg.span_nm
    wl_base = np.linspace(wl_min_nm, wl_max_nm, int(cfg.num_spectrum_points))

    # Add dense sub-FWHM points around every exact resonance inside [wl_min_nm, wl_max_nm]
    dn_dlam_nm = -(ng_rt - neff_rt) / cfg.wavelength_nm
    n_at_min = neff_rt + dn_dlam_nm * (wl_min_nm - cfg.wavelength_nm) + np.real(dneff_complex)
    n_at_max = neff_rt + dn_dlam_nm * (wl_max_nm - cfg.wavelength_nm) + np.real(dneff_complex)
    m_hi = int(math.ceil(n_at_min * (l_rt_um * 1e3) / wl_min_nm))
    m_lo = int(math.floor(n_at_max * (l_rt_um * 1e3) / wl_max_nm))
    dense_clusters = [wl_base]
    l_rt_nm = l_rt_um * 1e3
    for m_int in range(min(m_lo, m_hi), max(m_lo, m_hi) + 1):
        if m_int <= 0:
            continue
        # Linearized exact root of (neff_rt + dn_dlam*(lam - lam0) + dn_tune) * L_rt = m_int * lam
        num_r = (neff_rt - dn_dlam_nm * cfg.wavelength_nm + np.real(dneff_complex)) * l_rt_nm
        den_r = float(m_int) - dn_dlam_nm * l_rt_nm
        if den_r <= 0:
            continue
        lam_res = num_r / den_r
        if wl_min_nm <= lam_res <= wl_max_nm:
            hw_dense = max(min(5.0 * fwhm_nm, 0.25 * fsr_nm), 1e-7)
            # Lorentzian-graded sinh cluster around lam_res resolves Q up to 1e8+ (sub-femtometer FWHM)
            u_lin = np.linspace(-1.0, 1.0, 129)
            cluster = lam_res + hw_dense * np.sinh(2.2 * u_lin) / math.sinh(2.2)
            cluster = np.clip(cluster, wl_min_nm, wl_max_nm)
            dense_clusters.append(np.append(cluster, lam_res))
    wl_vec = np.unique(np.concatenate(dense_clusters))

    # Optional broadband coupler dispersion kappa(lambda) across wide spectral scans (>= 80 nm or explicit flag)
    kp_thru_spec: float | np.ndarray = kp_thru
    kp_drop_spec: float | np.ndarray = kp_drop
    use_bb_coupler = (
        (bool(cfg.broadband_coupler_dispersion) or float(cfg.span_nm) >= 80.0)
        and cfg.topology != "vernier_crow_ad"
        and kp_thru > 1e-7
    )
    if use_bb_coupler:
        wl_anc = np.array([wl_min_nm, cfg.wavelength_nm, wl_max_nm], dtype=float)
        kp1_anc = []
        kp2_anc = []
        for wa in wl_anc:
            if abs(wa - cfg.wavelength_nm) < 1e-4:
                kp1_anc.append(kp_thru)
                kp2_anc.append(kp_drop)
            else:
                try:
                    sub_cfg = MRRDesignConfig(
                        topology=cfg.topology,
                        core_material=cfg.core_material,
                        clad_material=cfg.clad_material,
                        top_clad_material=cfg.top_clad_material,
                        sc_fraction=cfg.sc_fraction,
                        custom_n=cfg.custom_n,
                        custom_k=cfg.custom_k,
                        custom_clad_n=cfg.custom_clad_n,
                        custom_top_clad_n=cfg.custom_top_clad_n,
                        ring_width_um=cfg.ring_width_um,
                        bus_width_um=cfg.bus_width_um,
                        film_height_um=cfg.film_height_um,
                        slab_height_um=cfg.slab_height_um,
                        sidewall_deg=cfg.sidewall_deg,
                        ring_radius_um=cfg.ring_radius_um,
                        gap_thru_um=cfg.gap_thru_um,
                        gap_drop_um=cfg.gap_drop_um,
                        straight_length_um=cfg.straight_length_um,
                        pulley_angle_thru_deg=cfg.pulley_angle_thru_deg,
                        pulley_angle_drop_deg=cfg.pulley_angle_drop_deg,
                        wavelength_nm=float(wa),
                        span_nm=4.0,
                        num_spectrum_points=80,
                        polarization=cfg.polarization,
                        alpha_loss_db_per_cm=alpha_db_cm,
                        solver_backend=cfg.solver_backend,
                        core_mesh_res_um=cfg.core_mesh_res_um,
                        broadband_coupler_dispersion=False,
                    )
                    sub_m = simulate_mrr_topology(sub_cfg)["metrics"]
                    kp1_anc.append(max(float(sub_m["kappa_power_thru"]), 1e-9))
                    kp2_anc.append(max(float(sub_m["kappa_power_drop"]), 0.0))
                except Exception:
                    kp1_anc.append(kp_thru)
                    kp2_anc.append(kp_drop)
        # Log-linear interpolation captures exact evanescent coupling dispersion exp(-gamma(wl)*gap)
        kp_thru_spec = np.clip(np.exp(np.interp(wl_vec, wl_anc, np.log(np.maximum(kp1_anc, 1e-9)))), 1e-9, 0.9999)
        if is_4port and kp_drop > 1e-7:
            kp_drop_spec = np.clip(np.exp(np.interp(wl_vec, wl_anc, np.log(np.maximum(kp2_anc, 1e-9)))), 1e-9, 0.9999)

    alpha_eff_db_cm = alpha_power_per_m / (100.0 * math.log(10.0) / 10.0)
    if cfg.topology == "vernier_crow_ad" and vernier_summary is not None:
        spec_out = _vernier_crow_spectra(
            wl_vec,
            l1_um=l_rt_um,
            l2_um=2.0 * math.pi * cfg.ring2_radius_um,
            neff1_ref=neff_rt,
            ng1_ref=ng_rt,
            neff2_ref=neff_r2,
            ng2_ref=ng_r2,
            lambda_ref_nm=cfg.wavelength_nm,
            alpha_db_per_cm=alpha_eff_db_cm,
            kp_thru=kp_thru,
            kp_inter=vernier_summary["kappa_inter_power"],
            kp_drop=kp_drop,
            delta_neff_tuning=dneff_complex,
        )
        comp_backend = detect_hardware_profile().selected_spectrum_backend
    else:
        spec_out = accelerated_tmm_spectrum(
            wl_vec,
            roundtrip_length_um=l_rt_um,
            neff_at_ref=neff_rt,
            ng_at_ref=ng_rt,
            lambda_ref_nm=cfg.wavelength_nm,
            alpha_loss_db_per_cm=alpha_eff_db_cm,
            kappa_power_thru=kp_thru_spec,
            kappa_power_drop=kp_drop_spec,
            delta_neff_tuning=dneff_complex,
        )
        comp_backend = str(spec_out["compute_backend"])

    thru_arr = np.asarray(spec_out["thru_T"], dtype=float)
    drop_arr = np.asarray(spec_out["drop_T"], dtype=float)
    er_thru_db = -10.0 * math.log10(max(float(np.min(thru_arr)) / max(float(np.max(thru_arr)), 1e-12), 1e-12))
    il_drop_db = -10.0 * math.log10(max(float(np.max(drop_arr)), 1e-12)) if is_4port else None

    cp_ratio = kp_thru / max(1.0 - (a_rt ** 2) * (1.0 - kp_drop), 1e-12)
    if abs(cp_ratio - 1.0) <= 0.05:
        regime_str = "Critically Coupled"
    elif cp_ratio > 1.0:
        regime_str = "Over-Coupled"
    else:
        regime_str = "Under-Coupled"

    thru_db_arr = 10.0 * np.log10(np.maximum(thru_arr, 1e-12))
    drop_db_arr = (
        10.0 * np.log10(np.maximum(drop_arr, 1e-6))
        if is_4port else np.full_like(thru_arr, -60.0)
    )
    # High-Q cavity photon-lifetime & FDTD time-domain ring-down diagnostics
    omega0_rad_s = 2.0 * math.pi * C0_M_PER_S / (cfg.wavelength_nm * 1e-9)
    tau_ph_ps = (q_loaded / omega0_rad_s) * 1e12
    t_rt_ps = (ng_rt * (l_rt_um * 1e-6) / C0_M_PER_S) * 1e12
    # Energy decay exp(-t / tau_ph) <= 1e-5 requires t >= ln(1e5) * tau_ph = 11.5129 * tau_ph
    fdtd_shutoff_1e5_ps = math.log(1e5) * tau_ph_ps
    fdtd_roundtrips_1e5 = fdtd_shutoff_1e5_ps / max(t_rt_ps, 1e-6)

    return {
        "elapsed_ms": round((time.perf_counter() - t_sim_start) * 1e3, 1),
        "config": asdict(cfg),
        "topology_info": TOPOLOGY_CATALOG[cfg.topology],
        "eigenmode": {
            "solver_backend": ring_bent_mode.solver_backend,
            "n_eff_bent": round(neff_bent, 6),
            "n_eff_roundtrip": round(neff_rt, 6),
            "n_g_roundtrip": round(ng_rt, 6),
            "te_fraction": round(ring_bent_mode.te_fraction, 4),
            "a_eff_um2": round(ring_bent_mode.a_eff_um2, 4),
            "core_confinement": round(ring_bent_mode.core_confinement, 4),
            "m_azimuthal": round(ring_bent_mode.m_azimuthal, 3) if math.isfinite(ring_bent_mode.m_azimuthal) else None,
        },
        "metrics": {
            "roundtrip_length_um": round(l_rt_um, 3),
            "neff_ring": round(neff_rt, 6),
            "ng_ring": round(ng_rt, 6),
            "FSR_nm": round(fsr_nm, 4),
            "FSR_GHz": round(fsr_ghz, 3),
            "FWHM_pm": round(fwhm_pm, 3),
            "finesse": round(finesse, 2),
            "Q_int": round(q_int, 0),
            "Q_ext_thru": round(min(q_ext_thru, 1e11), 0),
            "Q_ext_drop": round(min(q_ext_drop, 1e11), 0) if math.isfinite(q_ext_drop) else None,
            "Q_loaded": round(q_loaded, 0),
            "kappa_power_thru": round(kp_thru, 6),
            "kappa_power_drop": round(kp_drop, 6),
            "extinction_ratio_thru_dB": round(er_thru_db, 2),
            "ER_thru_dB": round(er_thru_db, 2),
            "insertion_loss_drop_dB": round(il_drop_db, 2) if il_drop_db is not None else None,
            "coupling_regime": regime_str,
            "coupling_ratio_Cp": round(cp_ratio, 4),
            "alpha_loss_db_per_cm": alpha_db_cm,
            "alpha_bend_db_per_cm": round(alpha_bend_db_cm, 6),
            "bend_loss_db_per_90deg": round(bend_loss_db_90, 6),
            "Q_rad_bend": bend_rad["Q_rad_bend"],
            "alpha_scattering_db_per_cm": round(alpha_scat_db_cm, 5),
            "backscatter_split_rms_mhz": round(backscatter_split_mhz, 2),
            "broadband_coupler_dispersive": bool(use_bb_coupler),
            "tuning_shift_pm": tuning_info["resonance_shift_pm"],
            "eo_tunability_pm_per_v": tuning_info["electro_optic_pm_per_v"],
            "photon_lifetime_ps": round(tau_ph_ps, 3),
            "roundtrip_time_ps": round(t_rt_ps, 4),
            "fdtd_shutoff_1e5_ps": round(fdtd_shutoff_1e5_ps, 2),
            "fdtd_roundtrips_1e5": round(fdtd_roundtrips_1e5, 1),
            "compute_backend": comp_backend,
        },
        "critical_coupling": {
            "parameter": coupler_sweep["param_name"],
            "unit": coupler_sweep["param_unit"],
            "recommended_value": coupler_sweep["critical_coupling_value"],
            "found": coupler_sweep["critical_coupling_found"],
            "regime": regime_str,
        },
        "coupler_curve": {
            "x_param": coupler_sweep["param_name"],
            "x_unit": coupler_sweep["param_unit"],
            "x_values": coupler_sweep["x_values"],
            "log10_Q_ext": np.round(np.log10(np.maximum(np.asarray(coupler_sweep["Q_ext"], dtype=float), 1.0)), 4).tolist(),
        },
        "vernier": vernier_summary,
        "cap_patch": cap_patch_summary,
        "tuning": tuning_info,
        "wavelength_warnings": wavelength_warnings,
        "coupler_sweep": coupler_sweep,
        "spectrum": {
            "wavelength_nm": np.round(wl_vec, 7).tolist(),
            "thru_T": np.round(thru_arr, 6).tolist(),
            "drop_T": np.round(drop_arr, 6).tolist(),
            "thru_dB": np.round(thru_db_arr, 3).tolist(),
            "drop_dB": np.round(drop_db_arr, 3).tolist(),
            "thru_phase_rad": np.round(np.asarray(spec_out["thru_phase_rad"], dtype=float), 5).tolist(),
        },
    }


def build_mrr_gds_library(cfg: MRRDesignConfig):
    """Build in-memory `gdstk.Library` for any of the 7 MRR topologies."""
    import gdstk

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(f"MicroRing_{cfg.topology}_{cfg.core_material}")

    L_CORE = 1
    L_SLAB = 2
    L_HEATER = 10
    L_TEXT = 66

    r1 = cfg.ring_radius_um
    wr = cfg.ring_width_um
    wb = cfg.bus_width_um
    g1 = cfg.gap_thru_um
    g2 = cfg.gap_drop_um
    tol = 0.002

    if cfg.topology.startswith("racetrack"):
        ls = cfg.straight_length_um
        rp = gdstk.RobustPath((-0.5 * ls, r1), wr, layer=L_CORE, datatype=0, tolerance=tol)
        rp.segment((0.5 * ls, r1))
        rp.arc(r1, 0.5 * math.pi, -0.5 * math.pi)
        rp.segment((-0.5 * ls, -r1))
        rp.arc(r1, 1.5 * math.pi, 0.5 * math.pi)
        cell.add(rp)

        span_x = ls + 2.0 * r1 + 30.0
        y_thru = r1 + 0.5 * wr + g1 + 0.5 * wb
        bus1 = gdstk.RobustPath((-0.5 * span_x, y_thru), wb, layer=L_CORE, datatype=0, tolerance=tol)
        bus1.segment((0.5 * span_x, y_thru))
        cell.add(bus1)

        if TOPOLOGY_CATALOG[cfg.topology]["ports"] == 4:
            y_drop = -(r1 + 0.5 * wr + g2 + 0.5 * wb)
            bus2 = gdstk.RobustPath((-0.5 * span_x, y_drop), wb, layer=L_CORE, datatype=0, tolerance=tol)
            bus2.segment((0.5 * span_x, y_drop))
            cell.add(bus2)

    elif cfg.topology.startswith("pulley"):
        ring = gdstk.ellipse((0.0, 0.0), r1 + 0.5 * wr, inner_radius=r1 - 0.5 * wr, layer=L_CORE, datatype=0, tolerance=tol)
        cell.add(ring)
        rb1 = r1 + 0.5 * wr + g1 + 0.5 * wb
        th1 = math.radians(cfg.pulley_angle_thru_deg)
        a_start = 0.5 * math.pi + 0.5 * th1
        a_end = 0.5 * math.pi - 0.5 * th1
        x_s, y_s = rb1 * math.cos(a_start), rb1 * math.sin(a_start)
        p_bus = gdstk.RobustPath((x_s - 20.0, y_s + 6.0), wb, layer=L_CORE, datatype=0, tolerance=tol)
        p_bus.segment((x_s, y_s))
        p_bus.arc(rb1, a_start, a_end)
        x_e, y_e = rb1 * math.cos(a_end), rb1 * math.sin(a_end)
        p_bus.segment((x_e + 20.0, y_e + 6.0))
        cell.add(p_bus)

        if TOPOLOGY_CATALOG[cfg.topology]["ports"] == 4:
            rb2 = r1 + 0.5 * wr + g2 + 0.5 * wb
            th2 = math.radians(cfg.pulley_angle_drop_deg)
            b_start = -0.5 * math.pi - 0.5 * th2
            b_end = -0.5 * math.pi + 0.5 * th2
            xs2, ys2 = rb2 * math.cos(b_start), rb2 * math.sin(b_start)
            p_drop = gdstk.RobustPath((xs2 - 20.0, ys2 - 6.0), wb, layer=L_CORE, datatype=0, tolerance=tol)
            p_drop.segment((xs2, ys2))
            p_drop.arc(rb2, b_start, b_end)
            xe2, ye2 = rb2 * math.cos(b_end), rb2 * math.sin(b_end)
            p_drop.segment((xe2 + 20.0, ye2 - 6.0))
            cell.add(p_drop)

    elif cfg.topology == "vernier_crow_ad":
        r2 = cfg.ring2_radius_um
        dy_c = r1 + r2 + wr + cfg.inter_ring_gap_um
        cell.add(gdstk.ellipse((0.0, 0.0), r1 + 0.5 * wr, inner_radius=r1 - 0.5 * wr, layer=L_CORE, datatype=0, tolerance=tol))
        cell.add(gdstk.ellipse((0.0, -dy_c), r2 + 0.5 * wr, inner_radius=r2 - 0.5 * wr, layer=L_CORE, datatype=0, tolerance=tol))
        span_x = 2.0 * max(r1, r2) + 30.0
        y_thru = r1 + 0.5 * wr + g1 + 0.5 * wb
        y_drop = -dy_c - (r2 + 0.5 * wr + g2 + 0.5 * wb)
        b1 = gdstk.RobustPath((-0.5 * span_x, y_thru), wb, layer=L_CORE, datatype=0, tolerance=tol)
        b1.segment((0.5 * span_x, y_thru))
        b2 = gdstk.RobustPath((-0.5 * span_x, y_drop), wb, layer=L_CORE, datatype=0, tolerance=tol)
        b2.segment((0.5 * span_x, y_drop))
        cell.add(b1, b2)

    else:
        # Point-coupled circular AP / AD
        cell.add(gdstk.ellipse((0.0, 0.0), r1 + 0.5 * wr, inner_radius=r1 - 0.5 * wr, layer=L_CORE, datatype=0, tolerance=tol))
        span_x = 2.0 * r1 + 30.0
        y_thru = r1 + 0.5 * wr + g1 + 0.5 * wb
        b1 = gdstk.RobustPath((-0.5 * span_x, y_thru), wb, layer=L_CORE, datatype=0, tolerance=tol)
        b1.segment((0.5 * span_x, y_thru))
        cell.add(b1)
        if TOPOLOGY_CATALOG[cfg.topology]["ports"] == 4:
            y_drop = -(r1 + 0.5 * wr + g2 + 0.5 * wb)
            b2 = gdstk.RobustPath((-0.5 * span_x, y_drop), wb, layer=L_CORE, datatype=0, tolerance=tol)
            b2.segment((0.5 * span_x, y_drop))
            cell.add(b2)

    # Add micro-heater / EO tuning arc on Layer 10
    heater_arc = gdstk.RobustPath(
        (r1 * math.cos(math.radians(210)), r1 * math.sin(math.radians(210))),
        max(wr * 1.6, 2.0),
        layer=L_HEATER,
        datatype=0,
        tolerance=tol,
    )
    heater_arc.arc(r1, math.radians(210), math.radians(330))
    cell.add(heater_arc)

    if cfg.slab_height_um > 0:
        bbox = cell.bounding_box()
        if bbox is not None:
            cell.add(gdstk.rectangle(
                (bbox[0][0] - 4.0, bbox[0][1] - 4.0),
                (bbox[1][0] + 4.0, bbox[1][1] + 4.0),
                layer=L_SLAB,
                datatype=0,
            ))

    cell.add(gdstk.Label(
        f"MicroRing {cfg.topology} | {cfg.core_material} | R={r1:.1f}um W={wr:.2f}um g={g1*1e3:.0f}nm",
        (0.0, r1 + 8.0),
        layer=L_TEXT,
    ))
    return lib


def export_mrr_gds(cfg: MRRDesignConfig, output_path: str | Path) -> str:
    """Export foundry-ready GDSII layout (`.gds`) for any of the 7 MRR topologies via `gdstk`."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lib = build_mrr_gds_library(cfg)
    lib.write_gds(str(out))
    return str(out.resolve())


def analyze_high_q_fdtd_convergence(
    cfg: MRRDesignConfig,
    sim_times_ps: list[float] | None = None,
) -> dict:
    """Quantify why full-ring 3D FDTD fails to converge at High-Q and compare truncated FDTD vs exact infinite-roundtrip TMM.

    Computes the exact finite-M round-trip impulse-response sum experienced when a time-domain
    FDTD solver terminates at `t_sim` (e.g., 10 ps, 50 ps, 250 ps) before `auto shutoff min = 1e-5`,
    overlaying the resulting sinc-ringing & collapsed extinction ratio against MicroRing Studio's
    exact infinite-roundtrip frequency-domain solution (`M -> infinity`, solved in ~1 s for any Q).
    """
    res = simulate_mrr_topology(cfg)
    m = res["metrics"]
    q_l = float(m["Q_loaded"])
    tau_ph_ps = float(m["photon_lifetime_ps"])
    t_rt_ps = float(m["roundtrip_time_ps"])
    t_1e5_ps = float(m["fdtd_shutoff_1e5_ps"])
    l_rt_um = float(m["roundtrip_length_um"])
    neff_rt = float(m["neff_ring"])
    ng_rt = float(m["ng_ring"])
    kp1 = float(m["kappa_power_thru"])
    kp2 = float(m["kappa_power_drop"])
    alpha_db_cm = float(m["alpha_loss_db_per_cm"])

    tau1 = math.sqrt(max(1.0 - kp1, 0.0))
    tau2 = math.sqrt(max(1.0 - kp2, 0.0))
    alpha_m = alpha_db_cm * 100.0 * math.log(10.0) / 10.0
    a_rt = math.exp(-0.5 * alpha_m * (l_rt_um * 1e-6))
    fb_gain = tau1 * tau2 * a_rt

    # Center zoom window (+/- 8 FWHM around nearest resonance to cfg.wavelength_nm)
    fwhm_nm = max(cfg.wavelength_nm / max(q_l, 100.0), 1e-7)
    dn_dlam = -(ng_rt - neff_rt) / cfg.wavelength_nm
    m_ord = max(1, int(round(neff_rt * (l_rt_um * 1e3) / cfg.wavelength_nm)))
    lam_res = ((neff_rt - dn_dlam * cfg.wavelength_nm) * (l_rt_um * 1e3)) / (m_ord - dn_dlam * (l_rt_um * 1e3))

    det_pm = np.linspace(-8.0 * fwhm_nm * 1e3, 8.0 * fwhm_nm * 1e3, 301)
    wl_zoom_nm = lam_res + det_pm * 1e-3
    neff_w = neff_rt + dn_dlam * (wl_zoom_nm - cfg.wavelength_nm)
    phi = 2.0 * math.pi * neff_w * (l_rt_um * 1e3) / wl_zoom_nm
    z_fb = fb_gain * np.exp(1j * phi)

    # Exact infinite-roundtrip converged transmission (M -> inf):
    # t_inf = tau1 - kp1 * tau2 * a_rt * exp(i*phi) / (1 - fb_gain * exp(i*phi))
    t_inf = tau1 - (kp1 * tau2 * a_rt * np.exp(1j * phi)) / (1.0 - z_fb)
    thru_inf_db = 20.0 * np.log10(np.maximum(np.abs(t_inf), 1e-6))
    er_inf_db = float(np.max(thru_inf_db) - np.min(thru_inf_db))

    if sim_times_ps is None:
        # Pick 3 illustrative FDTD cutoff times showing under-converged vs partially-converged behavior
        sim_times_ps = [
            round(max(1.0, 0.02 * t_1e5_ps), 2),
            round(max(10.0, 0.12 * t_1e5_ps), 2),
            round(max(50.0, 0.35 * t_1e5_ps), 2),
        ]

    truncated_traces = []
    for t_ps in sim_times_ps:
        n_rt = max(1, int(math.floor(float(t_ps) / max(t_rt_ps, 1e-5))))
        # Exact geometric partial sum sum_{k=1}^{n_rt} z_fb^k = z_fb * (1 - z_fb^n_rt) / (1 - z_fb)
        geom_sum = z_fb * (1.0 - np.power(z_fb, n_rt)) / (1.0 - z_fb)
        t_tr = tau1 - (kp1 / max(tau1, 1e-9)) * geom_sum
        tr_db = 20.0 * np.log10(np.maximum(np.abs(t_tr), 1e-6))
        residual_shutoff = float(math.exp(-float(t_ps) / max(tau_ph_ps, 1e-6)))
        er_apparent_db = float(np.max(tr_db) - np.min(tr_db))
        truncated_traces.append({
            "fdtd_sim_time_ps": float(t_ps),
            "completed_roundtrips": n_rt,
            "residual_energy_shutoff": round(residual_shutoff, 6),
            "converged_1e5": bool(residual_shutoff <= 1e-5),
            "apparent_er_db": round(er_apparent_db, 2),
            "er_error_db": round(abs(er_inf_db - er_apparent_db), 2),
            "thru_dB": np.round(tr_db, 3).tolist(),
        })

    # Single-pass coupler transit time for hybrid coupler-only 3D FDTD (typically ~25 um -> ~0.25 ps!)
    coupler_len_um = max(20.0, cfg.straight_length_um + 16.0)
    t_coupler_only_ps = round(3.5 * (ng_rt * coupler_len_um * 1e-6 / C0_M_PER_S) * 1e12, 3)
    speedup_vs_full_ring = round(t_1e5_ps / max(t_coupler_only_ps, 0.05), 0)

    return {
        "Q_loaded": round(q_l, 0),
        "Q_int": m["Q_int"],
        "FWHM_pm": m["FWHM_pm"],
        "resonance_wavelength_nm": round(float(lam_res), 6),
        "photon_lifetime_ps": round(tau_ph_ps, 3),
        "roundtrip_time_ps": round(t_rt_ps, 4),
        "required_fdtd_sim_time_1e5_ps": round(t_1e5_ps, 2),
        "required_fdtd_sim_time_1e5_ns": round(t_1e5_ps * 1e-3, 4),
        "required_fdtd_roundtrips_1e5": round(float(m["fdtd_roundtrips_1e5"]), 1),
        "coupler_only_fdtd_time_ps": t_coupler_only_ps,
        "time_domain_speedup_factor": speedup_vs_full_ring,
        "true_converged_er_db": round(er_inf_db, 2),
        "detuning_pm": np.round(det_pm, 5).tolist(),
        "converged_infinite_rt_thru_dB": np.round(thru_inf_db, 3).tolist(),
        "truncated_fdtd_traces": truncated_traces,
    }


def export_coupler_cutout_gds(cfg: MRRDesignConfig, out_dir: str | Path | None = None) -> dict:
    """Export non-resonant Coupler-Only Cutout `.GDS` + Lumerical 3D-FDTD `.lsf` (`~0.8 ps` vs `20 ns` full ring)."""
    import gdstk
    from .gds_bridge import write_klayout_layer_props

    if out_dir is None:
        root = get_default_output_root()
        root.mkdir(parents=True, exist_ok=True)
        gds_path = root / f"MicroRing_CouplerCutout_{cfg.topology}_{cfg.core_material}.gds"
    else:
        p_in = Path(out_dir)
        if p_in.suffix.lower() == ".gds":
            p_in.parent.mkdir(parents=True, exist_ok=True)
            gds_path = p_in
            root = p_in.parent
        else:
            p_in.mkdir(parents=True, exist_ok=True)
            root = p_in
            gds_path = root / f"MicroRing_CouplerCutout_{cfg.topology}_{cfg.core_material}.gds"
    lsf_path = root / f"run_coupler_only_3dfdtd_{cfg.core_material}.lsf"
    lyp_path = gds_path.with_suffix(".lyp")

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell(f"CouplerCutout_{cfg.core_material}")
    r1 = cfg.ring_radius_um
    wr = cfg.ring_width_um
    wb = cfg.bus_width_um
    g1 = cfg.gap_thru_um
    ls = cfg.straight_length_um if cfg.topology.startswith("racetrack") else 0.0
    half_span_x = max(12.0, 0.5 * ls + min(r1 * 0.65, 16.0))

    # Straight bus segment across coupling zone
    y_bus = r1 + 0.5 * wr + g1 + 0.5 * wb
    b1 = gdstk.RobustPath((-half_span_x, y_bus), wb, layer=1, datatype=0, tolerance=0.002)
    b1.segment((half_span_x, y_bus))
    cell.add(b1)

    # Upper ring arc / straight section only (terminated cleanly inside FDTD PML so zero cavity resonance!)
    ang_span = min(math.pi * 0.42, math.asin(min(0.95, half_span_x / max(r1, 2.0))))
    a0 = 0.5 * math.pi + ang_span
    a1 = 0.5 * math.pi - ang_span
    if ls > 0:
        rp = gdstk.RobustPath((-0.5 * ls - r1 * math.sin(ang_span), r1 * math.cos(ang_span)), wr, layer=1, datatype=0, tolerance=0.002)
        rp.arc(r1, a0, 0.5 * math.pi)
        rp.segment((0.5 * ls, r1))
        rp.arc(r1, 0.5 * math.pi, a1)
    else:
        rp = gdstk.RobustPath((r1 * math.cos(a0), r1 * math.sin(a0)), wr, layer=1, datatype=0, tolerance=0.002)
        rp.arc(r1, a0, a1)
    cell.add(rp)
    lib.write_gds(str(gds_path))

    lsf_txt = f"""# MicroRing Studio - Non-Resonant Single-Pass Coupler 3D-FDTD Script
# Solves kappa^2 in ~0.5 ps (zero High-Q ring-down stall!) and plugs directly back into MicroRing Studio.
newproject; redrawoff;
gdsimport("{gds_path.name}", "{cell.name}", "1:0", "{cfg.core_material}", 0, {cfg.film_height_um}e-6);
addfdtd;
set("x", 0); set("x span", {2.0*half_span_x - 2.0:.2f}e-6);
set("y", {(r1 + y_bus)*0.5:.3f}e-6); set("y span", {max(8.0, wr + wb + g1 + 6.0):.2f}e-6);
set("z", {0.5*cfg.film_height_um:.3f}e-6); set("z span", 3.2e-6);
set("simulation time", 800e-15); # Only 0.8 ps needed (single pass, no resonant ring-down!)
set("auto shutoff min", 1e-6);
?"Coupler-only FDTD region configured (0.8 ps simulation time, zero High-Q ring-down stall).";
"""
    lsf_path.write_text(lsf_txt, encoding="utf-8")
    write_klayout_layer_props(lyp_path)
    return {
        "gds_path": str(gds_path),
        "lyp_path": str(lyp_path),
        "lsf_path": str(lsf_path),
        "span_um": round(2.0 * half_span_x, 2),
        "recommended_fdtd_time_ps": 0.8,
    }
