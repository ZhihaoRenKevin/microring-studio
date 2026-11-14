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

from .eigenmode_solver import (
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
    if solver_backend != "fde" and math.isfinite(bend_radius_m) and (bend_radius_m < 8e-6 or spec.h_slab_um > 0.0):
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


def _point_coupler_kappa_power_from_gap_curve(
    g0_um: float,
    radius_um: float,
    gaps_grid_um: np.ndarray,
    kd_grid_per_m: np.ndarray,
    beta_ring: float,
    beta_bus: float,
    ring2_radius_um: float | None = None,
) -> float:
    """Integrate spatial coupling k_d(g(z))*exp(i*Delta_phi(z)) along curved approach z."""
    r1_m = radius_um * 1e-6
    if ring2_radius_um is None:
        r_eff_m = r1_m
    else:
        r2_m = ring2_radius_um * 1e-6
        r_eff_m = (r1_m * r2_m) / max(r1_m + r2_m, 1e-12)

    max_extra_gap_um = float(gaps_grid_um[-1] - g0_um)
    if max_extra_gap_um <= 0.02:
        return 0.0
    z_max_m = min(0.80 * r1_m, math.sqrt(2.0 * r_eff_m * (max_extra_gap_um * 1e-6)))
    z_m = np.linspace(-z_max_m, z_max_m, 161)
    dz_m = float(z_m[1] - z_m[0])

    if ring2_radius_um is None:
        g_z_um = g0_um + (r1_m - np.sqrt(np.maximum(r1_m ** 2 - z_m ** 2, 0.0))) * 1e6
    else:
        r2_m = ring2_radius_um * 1e-6
        g_z_um = g0_um + (
            r1_m + r2_m
            - np.sqrt(np.maximum(r1_m ** 2 - z_m ** 2, 0.0))
            - np.sqrt(np.maximum(r2_m ** 2 - z_m ** 2, 0.0))
        ) * 1e6

    kd_z = np.interp(g_z_um, gaps_grid_um, kd_grid_per_m, left=kd_grid_per_m[0], right=0.0)
    dphi_z = (beta_bus - beta_ring) * z_m - (beta_ring * (z_m ** 3)) / (6.0 * max(r_eff_m ** 2, 1e-18))
    theta_eff = abs(np.sum(kd_z * np.exp(1j * dphi_z)) * dz_m)
    return float(math.sin(min(theta_eff, 0.5 * math.pi)) ** 2)


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
    )
    clad_fn = get_material_fn(cfg.clad_material)
    top_fn = get_material_fn(cfg.top_clad_material)
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

    # Intrinsic cavity decay & Q_int from propagation loss + core/cladding material absorption k
    neff_imag_modal = float(abs(np.imag(ring_bent_mode.n_eff)))
    k_core_bulk = ring_bent_mode.core_confinement * max(mat_core_val.kx, 0.0)
    k_clad_bulk = (1.0 - ring_bent_mode.core_confinement) * 0.5 * (max(mat_clad_val.kx, 0.0) + max(mat_top_val.kx, 0.0))
    k_eff_mat = max(neff_imag_modal, k_core_bulk + k_clad_bulk)
    alpha_power_per_m = alpha_db_cm * 100.0 * (math.log(10.0) / 10.0) + (4.0 * math.pi * k_eff_mat / wl_m)

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
            quad_step_m=quad_cyl_m,
            ring_azimuthal_m=ring_bent_mode.m_azimuthal,
            bus_neff=float(np.real(bus_mode_thru.n_eff)),
        )
        tau_rt_s = (l_rt_um * 1e-6) * ng_rt / C0_M_PER_S
        ch_thru = PulleyCMTChannel.from_couplings(
            [cc_dict["k_per_m"]],
            [cc_dict["delta_beta_per_m"]],
            None,
            frequency_hz=freq_hz,
            roundtrip_time_s=tau_rt_s,
            bus_radius_m=bus_r_thru_m,
            label="pulley_thru",
        )
        scan = scan_pulley_single_bus(ch_thru, q_int=q_int, low_deg=0.5, high_deg=180.0, step_deg=0.5)
        rates_thru = ch_thru.coupling_rates([cfg.pulley_angle_thru_deg])
        kp_thru = float(rates_thru["full_power"][0])

        if is_4port:
            bus_r_drop_m = (cfg.ring_radius_um + 0.5 * cfg.ring_width_um + cfg.gap_drop_um + 0.5 * cfg.bus_width_um) * 1e-6
            if abs(cfg.gap_drop_um - cfg.gap_thru_um) < 1e-6:
                ch_drop = ch_thru
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
                    quad_step_m=quad_cyl_m,
                    ring_azimuthal_m=ring_bent_mode.m_azimuthal,
                    bus_neff=float(np.real(bus_mode_drop.n_eff)),
                )
                ch_drop = PulleyCMTChannel.from_couplings(
                    [cc_d["k_per_m"]],
                    [cc_d["delta_beta_per_m"]],
                    None,
                    frequency_hz=freq_hz,
                    roundtrip_time_s=tau_rt_s,
                    bus_radius_m=bus_r_drop_m,
                    label="pulley_drop",
                )
            kp_drop = float(ch_drop.coupling_rates([cfg.pulley_angle_drop_deg])["full_power"][0])
        else:
            kp_drop = 0.0

        cc_val = float(scan["roots"][0]["angle_deg"]) if scan["roots"] else float(scan["Cp_max_angle_deg"])
        coupler_sweep = {
            "param_name": "pulley_angle_thru_deg",
            "param_unit": "deg",
            "x_values": np.round(scan["angle_deg"], 3).tolist(),
            "Q_ext": np.round(np.minimum(scan["pump_Qext"], 1e11), 1).tolist(),
            "coupling_ratio_Cp": np.round(scan["pump_Cp"], 4).tolist(),
            "critical_coupling_value": round(cc_val, 3),
            "critical_coupling_found": len(scan["roots"]) > 0,
            "k_per_m_abs": round(abs(cc_dict["k_per_m"]), 2),
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
            n_clad=n_clad_scalar,
            wavelength_m=wl_m,
            quad_step_um=quad_lat_um,
        )
        k0_wave = 2.0 * math.pi / wl_m
        beta_ring_c = k0_wave * float(np.real(ref_ring_for_coupler.n_eff))
        beta_bus_c = k0_wave * float(np.real(bus_mode_str.n_eff))

        if cfg.topology in ("point_circular_ap", "point_circular_ad"):
            kp_thru = _point_coupler_kappa_power_from_gap_curve(
                cfg.gap_thru_um, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c
            )
            kp_drop = (
                _point_coupler_kappa_power_from_gap_curve(
                    cfg.gap_drop_um, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c
                )
                if is_4port else 0.0
            )
            g_sweep = np.linspace(g_min_um, g_max_um * 0.92, 48)
            kp_sweep = np.array([
                _point_coupler_kappa_power_from_gap_curve(float(g), cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c)
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
                            lambda g: _point_coupler_kappa_power_from_gap_curve(g, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c) - target_kp,
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
                kp_end = _point_coupler_kappa_power_from_gap_curve(
                    gap_um, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c
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
            kd_inter_grid = _lateral_perturbation_kd_vs_gap(
                ring_bent_mode,
                ring2_mode,
                gaps_um=gaps_grid,
                ring_w_bottom_um=cfg.ring_width_um,
                bus_w_bottom_um=cfg.ring_width_um,
                film_h_um=cfg.film_height_um,
                slab_h_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg,
                n_core_xyz=n_core_xyz,
                n_clad=n_clad_scalar,
                wavelength_m=wl_m,
            )
            kp_thru = _point_coupler_kappa_power_from_gap_curve(
                cfg.gap_thru_um, cfg.ring_radius_um, gaps_grid, kd_grid, beta_ring_c, beta_bus_c
            )
            kp_drop = _point_coupler_kappa_power_from_gap_curve(
                cfg.gap_drop_um, cfg.ring2_radius_um, gaps_grid, kd_grid, k0_wave * neff_r2, beta_bus_c
            )
            kp_inter = _point_coupler_kappa_power_from_gap_curve(
                cfg.inter_ring_gap_um,
                cfg.ring_radius_um,
                gaps_grid,
                kd_inter_grid,
                beta_ring_c,
                k0_wave * neff_r2,
                ring2_radius_um=cfg.ring2_radius_um,
            )
            # Optimal maximally-flat 2nd-order CROW condition: kp_inter_opt ~ (kp_thru * kp_drop) / 4 + loss
            kp_inter_opt = min(0.28 * (kp_thru + kp_drop) ** 2 + kp_intrinsic_loss ** 2, 0.5)
            g12_sweep = np.linspace(0.10, 0.95, 40)
            kp12_sweep = np.array([
                _point_coupler_kappa_power_from_gap_curve(
                    float(g), cfg.ring_radius_um, gaps_grid, kd_inter_grid, beta_ring_c, k0_wave * neff_r2, ring2_radius_um=cfg.ring2_radius_um
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
            kappa_power_thru=kp_thru,
            kappa_power_drop=kp_drop,
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
