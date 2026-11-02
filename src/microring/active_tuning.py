"""First-Principles 2D-FEM Active Tuning & Lumerical INTERCONNECT Compact Model Engine (`microring.active_tuning`).

100% self-contained pure-Python multiphysics solver (`scikit-fem` + `scipy` + `numpy`,
requiring ZERO COMSOL or commercial TCAD/FDTD licenses) linking material thermo-optic
(`dn/dT`), anisotropic Pockels electro-optic (`r_33`, `r_13`), and Silicon semiconductor
plasma-dispersion coefficients directly to 2D waveguide/electrode/heater cross-section
geometry, perturbed 2D vector Maxwell eigenmodes (`n_eff(V, I, T, E)`), multi-bias
transmission spectra, and Ansys Lumerical INTERCONNECT compact models.

Supported Input Drive Modes (`drive_type`):
  - `"delta_temp_k"`   : Direct temperature shift `Delta_T [K]` across waveguide stack
  - `"e_field_v_per_um"`: Direct electric field `E [V/um]` inside electro-optic crystal
  - `"voltage_v"`      : Electrical voltage `V [V]` linked to device geometry (Si PN depletion,
                         2D-FEM electrostatic Pockels electrodes, or Joule microheater `V^2/R`)
  - `"current_ma"`     : Electrical current `I [mA]` linked to device geometry (2D-FEM Joule
                         microheater `I^2*R` or forward-biased Si PIN carrier injection)
  - `"power_mw"`       : Electrical heater power `P [mW]` linked to 2D-FEM thermal geometry

Supported Physical Mechanisms (`mechanism`):
  1. `"thermo_optic"`    : 2D-FEM Fourier heat conduction (`-div(k_th grad T) = Q_vol`) on actual
                           BOX/core/cladding/microheater geometry + material `dn/dT` in 2D FEM.
  2. `"pockels_eo"`      : 2D-FEM anisotropic electrostatic Laplace (`-div(eps_DC grad phi) = 0`)
                           on coplanar GSG or vertical electrodes + crystal-cut `r_33, r_13` in 2D FEM.
  3. `"si_pn_depletion"` : Lateral/offset Silicon rib PN junction Poisson depletion `W_dep(V)`,
                           Debye-screened `p(x,V), n(x,V)`, `p++/n++` slab contact wings, slab `R_s`,
                           `C_j(V)`, and Soref-Bennett `Delta_n + i*Delta_k` in 2D FEM.
  4. `"si_pin_injection"`: Forward-biased Silicon rib PIN diode ambipolar carrier injection
                           `Delta_N = Delta_P` + Soref-Bennett complex plasma dispersion in 2D FEM.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
import time
import zipfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erfc
import shapely
from shapely.geometry import Polygon, box

from skfem import (
    Basis,
    BilinearForm,
    ElementTriP0,
    ElementTriP1,
    LinearForm,
    MeshTri,
    condense,
    solve,
)

from .eigenmode_solver import WaveguideCrossSectionSpec, solve_perturbed_neff_sweep_fem
from .hardware_accel import accelerated_tmm_spectrum
from .materials import C0_M_PER_S, EPS0_F_PER_M, T_REF_K, get_material_fn, n_linbo3_xcut
from .mrr_topologies import TOPOLOGY_CATALOG, MRRDesignConfig, simulate_mrr_topology

# Fundamental semiconductor constants (SI)
Q_E_C = 1.602176634e-19        # unit: C
KB_J_PER_K = 1.380649e-23      # unit: J/K
NI_SI_300K_CM3 = 1.00e10       # unit: cm^-3

TUNING_MECHANISM_CATALOG: dict[str, dict] = {
    "auto": {
        "name_en": "Auto-Detect from Core Material (Si PN / Pockels EO / TO Heater)",
        "name_zh": "按波导材料自动匹配物理机制 (硅 PN 结 / Pockels 电光 / 热光微加热器)",
        "default_drive": "auto",
    },
    "si_pn_depletion": {
        "name_en": "Silicon PN Junction Reverse Depletion (2D Plasma Dispersion + FCA)",
        "name_zh": "硅横向/偏置 PN 结反偏耗尽调制 (2D 载流子色散 + 自由载流子吸收 FCA)",
        "default_drive": "voltage_v",
    },
    "pockels_eo": {
        "name_en": "Anisotropic Pockels Electro-Optic (2D Electrostatic Laplace + r33/r13)",
        "name_zh": "各向异性 Pockels 线性电光调制 (2D 静电场有限元 + r33/r13 张量)",
        "default_drive": "voltage_v",
    },
    "thermo_optic": {
        "name_en": "Thermo-Optic Microheater / Temperature (2D Heat Conduction + dn/dT)",
        "name_zh": "热光微加热器 / 温度场调谐 (2D 热传导有限元 + 材料热光系数 dn/dT)",
        "default_drive": "current_ma",
    },
    "si_pin_injection": {
        "name_en": "Silicon PIN Diode Forward Carrier Injection (Current-Driven Plasma Shift)",
        "name_zh": "硅 PIN 二极管正偏载流子注入调谐 (电流驱动等离子体蓝移 + FCA)",
        "default_drive": "current_ma",
    },
}

# Room-temperature thermal conductivity k_th [W/(m*K)] (peer-reviewed literature values)
THERMAL_CONDUCTIVITY_W_MK: dict[str, float] = {
    "Si": 148.0,
    "a-Si": 1.80,
    "Ge": 60.0,
    "Si3N4": 20.0,
    "SiRich_SiNx": 12.0,
    "AlN": 180.0,
    "ScAlN_10": 55.0,
    "ScAlN_20": 35.0,
    "ScAlN_30": 24.0,
    "4H-SiC": 370.0,
    "3C-SiC": 320.0,
    "LN_Zcut": 4.60,
    "LN_Xcut": 4.60,
    "SiO2": 1.38,
    "CaF2": 9.70,
    "MgF2": 11.6,
    "Al2O3": 30.0,
    "TiO2": 8.50,
    "Ta2O5": 3.50,
    "GaN": 130.0,
    "Diamond": 2000.0,
    "GSST_amorphous": 0.45,
    "GSST_crystalline": 1.20,
    "Air": 0.026,
}

# Volumetric heat capacity rho * C_p [J/(m^3*K)] for thermal time-constant estimation
VOLUMETRIC_HEAT_CAPACITY_J_M3K: dict[str, float] = {
    "SiO2": 1.63e6,
    "Si": 1.63e6,
    "Si3N4": 2.20e6,
    "AlN": 2.40e6,
    "4H-SiC": 2.21e6,
    "LN_Xcut": 2.91e6,
    "LN_Zcut": 2.91e6,
    "CaF2": 2.71e6,
    "Al2O3": 3.08e6,
}

# Low-frequency / RF clamped relative permittivity (eps_xx_DC, eps_yy_DC)
STATIC_PERMITTIVITY_DC: dict[str, tuple[float, float]] = {
    "Si": (11.7, 11.7),
    "a-Si": (11.5, 11.5),
    "Ge": (16.2, 16.2),
    "Si3N4": (7.5, 7.5),
    "SiRich_SiNx": (8.5, 8.5),
    "AlN": (9.0, 10.5),
    "ScAlN_10": (11.0, 12.8),
    "ScAlN_20": (13.8, 15.6),
    "ScAlN_30": (17.5, 20.5),
    "4H-SiC": (9.7, 10.0),
    "3C-SiC": (9.7, 9.7),
    "LN_Zcut": (44.0, 28.0),
    "LN_Xcut": (28.0, 44.0),
    "SiO2": (3.9, 3.9),
    "CaF2": (6.8, 6.8),
    "MgF2": (5.3, 5.3),
    "Al2O3": (9.4, 9.4),
    "TiO2": (40.0, 40.0),
    "Ta2O5": (22.0, 22.0),
    "GaN": (8.9, 10.4),
    "Diamond": (5.7, 5.7),
    "GSST_amorphous": (14.0, 14.0),
    "GSST_crystalline": (28.0, 28.0),
    "Air": (1.0, 1.0),
}

HEATER_METAL_CATALOG: dict[str, dict] = {
    "TiN": {"name": "Foundry Sputtered TiN", "rho_ohm_m": 1.60e-6, "kappa_th_w_mk": 28.0},
    "Pt": {"name": "E-Beam / Lift-Off Platinum (Pt)", "rho_ohm_m": 1.06e-7, "kappa_th_w_mk": 71.6},
    "W": {"name": "CMOS Tungsten (W)", "rho_ohm_m": 1.20e-7, "kappa_th_w_mk": 170.0},
    "NiCr": {"name": "Nichrome (Ni80Cr20)", "rho_ohm_m": 1.10e-6, "kappa_th_w_mk": 11.3},
    "doped_Si": {"name": "Heavily Doped Silicon (n++/p++)", "rho_ohm_m": 8.50e-6, "kappa_th_w_mk": 95.0},
}


# ==============================================================================
# 1. Soref-Bennett / Nedeljkovic Silicon Plasma Dispersion & Carrier Mobility
# ==============================================================================

def soref_bennett_si_dn_dk(
    delta_ne_cm3: np.ndarray | float,
    delta_nh_cm3: np.ndarray | float,
    wavelength_nm: float = 1550.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Silicon free-carrier index shift `Delta_n` and extinction `Delta_k`.

    Uses empirical Soref & Bennett (IEEE JQE 23, 123, 1987) coefficients at 1310 nm and
    1550 nm with smooth interpolation across O/C/L bands and Drude `lambda^2` scaling
    into the 2-um / Mid-IR Silicon window (Nedeljkovic et al., IEEE Photonics J. 3, 1171, 2011).

    Args:
      delta_ne_cm3: Free electron concentration `N_e` [cm^-3] (`>= 0`).
      delta_nh_cm3: Free hole concentration `N_h` [cm^-3] (`>= 0`).
      wavelength_nm: Optical vacuum wavelength [nm].

    Returns:
      Tuple `(dn_real, dk_imag)` where `dn_real <= 0` and `dk_imag >= 0`.
    """
    ne = np.maximum(np.asarray(delta_ne_cm3, dtype=float), 0.0)
    nh = np.maximum(np.asarray(delta_nh_cm3, dtype=float), 0.0)
    wl_um = max(float(wavelength_nm) * 1e-3, 1.10)

    # Interpolate exact Soref-Bennett 1.31 um & 1.55 um anchors, then scale ~ wl^2
    t = float(np.clip((wl_um - 1.31) / (1.55 - 1.31), 0.0, 1.0))
    ce_n = (1.0 - t) * 6.2e-22 + t * 8.8e-22
    ch_n = (1.0 - t) * 6.0e-18 + t * 8.5e-18
    ce_a = (1.0 - t) * 6.0e-18 + t * 8.5e-18
    ch_a = (1.0 - t) * 4.0e-18 + t * 6.0e-18

    if wl_um > 1.55:
        drude_scale = (wl_um / 1.55) ** 2
        ce_n *= drude_scale
        ch_n *= drude_scale ** 0.92
        ce_a *= drude_scale ** 1.08
        ch_a *= drude_scale ** 1.05
    elif wl_um < 1.31:
        drude_scale = (wl_um / 1.31) ** 2
        ce_n *= drude_scale
        ch_n *= drude_scale
        ce_a *= drude_scale
        ch_a *= drude_scale

    dn_real = -(ce_n * ne + ch_n * np.power(nh, 0.80))
    alpha_cm_inv = ce_a * ne + ch_a * nh
    wl_m = wl_um * 1e-6
    dk_imag = (wl_m / (4.0 * math.pi)) * (alpha_cm_inv * 100.0)
    return dn_real, dk_imag


def _si_carrier_mobilities_cm2_vs(na_cm3: float, nd_cm3: float) -> tuple[float, float]:
    """Caughey-Thomas room-temperature hole & electron mobilities in doped c-Si [cm^2/(V*s)]."""
    mu_p = 49.7 + (470.5 - 49.7) / (1.0 + (max(na_cm3, 1e14) / 1.60e17) ** 0.70)
    mu_n = 92.0 + (1360.0 - 92.0) / (1.0 + (max(nd_cm3, 1e14) / 1.30e17) ** 0.91)
    return float(mu_p), float(mu_n)


# ==============================================================================
# 2. Pure-Python 2D Structured-Mesh Builder & Solvers (`scikit-fem`, Zero COMSOL)
# ==============================================================================

def _snapped_1d_axis(knots: list[float], h_fine: float, h_coarse: float) -> np.ndarray:
    """Build strictly ascending 1D FEM coordinates snapped to exact geometry knot locations."""
    uniq = sorted(set(round(float(k), 6) for k in knots))
    segs: list[np.ndarray] = []
    x_center_span = 3.5
    for i in range(len(uniq) - 1):
        a, b = uniq[i], uniq[i + 1]
        span = b - a
        if span <= 1e-7:
            continue
        mid = 0.5 * (a + b)
        local_h = h_fine if abs(mid) <= x_center_span else min(h_coarse, h_fine + 0.12 * (abs(mid) - x_center_span))
        n_div = max(2, int(math.ceil(span / max(local_h, 0.02))))
        seg = np.linspace(a, b, n_div + 1)
        segs.append(seg if not segs else seg[1:])
    return np.concatenate(segs)


def _rectilinear_mesh_tri(x_um: np.ndarray, y_um: np.ndarray) -> tuple[MeshTri, np.ndarray, np.ndarray]:
    """Create a mirror-symmetric `skfem.MeshTri` (in SI metres) from 1D `(x_um, y_um)` axes."""
    nx, ny = len(x_um), len(y_um)
    xx, yy = np.meshgrid(x_um, y_um)
    p_m = np.ascontiguousarray(np.vstack([xx.ravel(), yy.ravel()]) * 1e-6)

    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    v00 = (jj * nx + ii).ravel()
    v10 = v00 + 1
    v01 = v00 + nx
    v11 = v01 + 1

    tris = np.empty((2 * len(v00), 3), dtype=np.int64)
    tris[0::2] = np.column_stack([v00, v10, v11])
    tris[1::2] = np.column_stack([v00, v11, v01])
    t_mat = np.ascontiguousarray(tris.T)

    xc_um = (xx.ravel()[t_mat[0]] + xx.ravel()[t_mat[1]] + xx.ravel()[t_mat[2]]) / 3.0
    yc_um = (yy.ravel()[t_mat[0]] + yy.ravel()[t_mat[1]] + yy.ravel()[t_mat[2]]) / 3.0
    return MeshTri(p_m, t_mat), xc_um, yc_um


def solve_2d_heater_thermal_fem(
    cfg: MRRDesignConfig,
    *,
    roundtrip_length_um: float,
    active_fraction: float = 0.75,
    box_thickness_um: float = 2.0,
    top_clad_thickness_um: float = 1.8,
    heater_material: str = "TiN",
    heater_width_um: float = 1.6,
    heater_thickness_um: float = 0.12,
    heater_gap_um: float = 0.90,
    heater_offset_um: float = 0.0,
    heater_resistance_ohm: float | None = None,
    lead_resistance_ohm: float = 12.0,
) -> dict:
    """Solve 2D steady-state Fourier heat conduction `-div(k_th grad T) = Q_vol` in pure Python (`skfem`).

    Links microheater dimensions (`heater_width_um`, `heater_thickness_um`, `heater_gap_um`,
    `heater_offset_um`), buried oxide thickness (`box_thickness_um` above isothermal Si substrate),
    and active heater arc length `L_heat = active_fraction * L_rt` to the 2D temperature field
    `Delta_T(x, y)` per mW of electrical power.
    """
    h_metal = HEATER_METAL_CATALOG.get(heater_material, HEATER_METAL_CATALOG["TiN"])
    l_heat_um = max(float(active_fraction) * float(roundtrip_length_um), 5.0)
    w_heat_um = max(float(heater_width_um), 0.25)
    t_heat_um = max(float(heater_thickness_um), 0.02)
    g_heat_um = max(float(heater_gap_um), 0.15)
    h_box_um = max(float(box_thickness_um), 0.40)

    y_core_top = cfg.film_height_um + cfg.interlayer_h_um + cfg.upper_core_h_um + cfg.cap_h_um
    y_heat0 = y_core_top + g_heat_um
    y_heat1 = y_heat0 + t_heat_um
    y_oxide_top = max(float(top_clad_thickness_um), y_heat1 + 0.35)
    y_air_top = y_oxide_top + 1.20
    x_dom = max(14.0, abs(heater_offset_um) + 0.5 * w_heat_um + 8.0, 0.5 * cfg.ring_width_um + 8.0)

    # Geometry-derived heater resistance R = rho * L / (w * t) + R_lead
    r_geom_ohm = float(h_metal["rho_ohm_m"]) * (l_heat_um * 1e-6) / ((w_heat_um * 1e-6) * (t_heat_um * 1e-6)) + float(lead_resistance_ohm)
    r_eff_ohm = float(heater_resistance_ohm) if (heater_resistance_ohm is not None and heater_resistance_ohm > 0) else r_geom_ohm

    inv_tan = 0.0 if cfg.sidewall_deg == 90.0 else 1.0 / math.tan(math.radians(cfg.sidewall_deg))
    w_top = max(0.08, cfg.ring_width_um - 2.0 * (cfg.film_height_um - cfg.slab_height_um) * inv_tan)

    x_knots = [
        -x_dom,
        -0.5 * cfg.ring_width_um,
        -0.5 * w_top,
        0.0,
        0.5 * w_top,
        0.5 * cfg.ring_width_um,
        heater_offset_um - 0.5 * w_heat_um,
        heater_offset_um,
        heater_offset_um + 0.5 * w_heat_um,
        x_dom,
    ]
    y_knots = [-h_box_um, 0.0, y_core_top, y_heat0, y_heat1, y_oxide_top, y_air_top]
    if cfg.slab_height_um > 1e-6:
        y_knots.append(cfg.slab_height_um)
    if cfg.upper_core_h_um > 1e-6:
        y_knots.extend([cfg.film_height_um, cfg.film_height_um + cfg.interlayer_h_um])

    x_axis_um = _snapped_1d_axis(x_knots, h_fine=0.14, h_coarse=0.85)
    y_axis_um = _snapped_1d_axis(y_knots, h_fine=0.10, h_coarse=0.45)
    mesh, xc_um, yc_um = _rectilinear_mesh_tri(x_axis_um, y_axis_um)

    # Ordered polygon regions (earlier overrides later)
    poly_heater = box(heater_offset_um - 0.5 * w_heat_um, y_heat0, heater_offset_um + 0.5 * w_heat_um, y_heat1)
    poly_core = Polygon([
        (-0.5 * cfg.ring_width_um, cfg.slab_height_um),
        (0.5 * cfg.ring_width_um, cfg.slab_height_um),
        (0.5 * w_top, cfg.film_height_um),
        (-0.5 * w_top, cfg.film_height_um),
    ])
    k_core = THERMAL_CONDUCTIVITY_W_MK.get(cfg.core_material, 20.0)
    k_clad = THERMAL_CONDUCTIVITY_W_MK.get(cfg.clad_material, 1.38)
    k_top = THERMAL_CONDUCTIVITY_W_MK.get(cfg.top_clad_material, 1.38)
    k_heat = float(h_metal["kappa_th_w_mk"])

    basis = Basis(mesh, ElementTriP1())
    basis0 = basis.with_element(ElementTriP0())
    k_th_elems = np.full(mesh.nelements, k_top, dtype=float)
    q_src_elems = np.zeros(mesh.nelements, dtype=float)

    k_th_elems[yc_um < 0.0] = k_clad
    k_th_elems[yc_um > y_oxide_top] = THERMAL_CONDUCTIVITY_W_MK["Air"]
    if cfg.slab_height_um > 1e-6:
        in_slab = (yc_um >= 0.0) & (yc_um <= cfg.slab_height_um)
        k_th_elems[in_slab] = k_core
    in_core = shapely.contains_xy(poly_core, xc_um, yc_um)
    k_th_elems[in_core] = k_core
    if cfg.upper_core_h_um > 1e-6 and cfg.upper_core_material:
        y_u0 = cfg.film_height_um + cfg.interlayer_h_um
        in_up = (yc_um >= y_u0) & (yc_um <= y_u0 + cfg.upper_core_h_um) & (np.abs(xc_um) <= 0.5 * w_top)
        k_th_elems[in_up] = THERMAL_CONDUCTIVITY_W_MK.get(cfg.upper_core_material, k_core)

    in_heater = shapely.contains_xy(poly_heater, xc_um, yc_um)
    k_th_elems[in_heater] = k_heat

    # Volumetric heat density Q_vol [W/m^3] for P = 1 mW = 1e-3 W dissipated over L_heat
    heater_area_m2 = (w_heat_um * 1e-6) * (t_heat_um * 1e-6)
    q_vol_1mw = 1e-3 / (heater_area_m2 * (l_heat_um * 1e-6))
    q_src_elems[in_heater] = q_vol_1mw

    k_dofs = basis0.zeros(dtype=float)
    q_dofs = basis0.zeros(dtype=float)
    k_dofs[:] = k_th_elems
    q_dofs[:] = q_src_elems

    @BilinearForm
    def laplace_th(u, v, w):
        return w["k_th"] * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

    @LinearForm
    def rhs_th(v, w):
        return w["q_vol"] * v

    A_th = laplace_th.assemble(basis, k_th=basis0.interpolate(k_dofs))
    b_th = rhs_th.assemble(basis, q_vol=basis0.interpolate(q_dofs))

    # Isothermal Si handle substrate at y = -h_box_um and far lateral heat-sink boundaries |x| = x_dom
    px_um = mesh.p[0] * 1e6
    py_um = mesh.p[1] * 1e6
    sink_nodes = np.flatnonzero(
        (py_um <= -h_box_um + 1e-5)
        | (px_um <= -x_dom + 1e-5)
        | (px_um >= x_dom - 1e-5)
    )
    dt_1mw_nodes = solve(*condense(A_th, b_th, D=sink_nodes, x=np.zeros(basis.N, dtype=float)))
    dt_2d = dt_1mw_nodes.reshape(len(y_axis_um), len(x_axis_um))

    interp_dt = RegularGridInterpolator(
        (y_axis_um, x_axis_um),
        dt_2d,
        bounds_error=False,
        fill_value=0.0,
    )

    # Extract core average temperature rise per mW (= effective modal thermal resistance R_th [K/mW])
    xc_core = xc_um[in_core] if np.any(in_core) else np.array([0.0])
    yc_core = yc_um[in_core] if np.any(in_core) else np.array([0.5 * cfg.film_height_um])
    dt_core_per_mw = float(np.mean(interp_dt(np.column_stack([yc_core, xc_core]))))
    dt_heater_peak_per_mw = float(np.max(dt_1mw_nodes))

    # Effective thermal time constant tau_th ~ C_th_eff * R_th from heated zone integral
    rho_cp_clad = VOLUMETRIC_HEAT_CAPACITY_J_M3K.get(cfg.clad_material, 1.63e6)
    dx_grid = np.gradient(x_axis_um * 1e-6)
    dy_grid = np.gradient(y_axis_um * 1e-6)
    da_grid = np.outer(dy_grid, dx_grid)
    oxide_mask = (y_axis_um[:, None] <= y_oxide_top)
    eth_per_mw_j = float(np.sum(rho_cp_clad * dt_2d * da_grid * oxide_mask)) * (l_heat_um * 1e-6)
    # tau_th [s] = stored thermal energy at steady-state P=1mW divided by P=1e-3 W
    tau_th_us = (eth_per_mw_j / 1e-3) * 1e6
    f_3db_khz = 1.0 / (2.0 * math.pi * max(tau_th_us * 1e-6, 1e-9)) * 1e-3

    metal_loss_warn_db_cm = 0.0
    if g_heat_um < 0.55 and abs(heater_offset_um) < 0.5 * (cfg.ring_width_um + w_heat_um):
        # Evanescent metal proximity absorption warning when heater sits < 550 nm directly over core
        metal_loss_warn_db_cm = float(4.5 * math.exp(-(g_heat_um - 0.25) / 0.14))

    return {
        "interp_dt_per_mw": interp_dt,
        "heater_material": heater_material,
        "heater_length_um": round(l_heat_um, 2),
        "heater_width_um": round(w_heat_um, 3),
        "heater_thickness_um": round(t_heat_um, 3),
        "heater_gap_um": round(g_heat_um, 3),
        "heater_offset_um": round(float(heater_offset_um), 3),
        "box_thickness_um": round(h_box_um, 3),
        "heater_resistance_ohm": round(r_eff_ohm, 2),
        "heater_resistance_geom_ohm": round(r_geom_ohm, 2),
        "r_th_core_k_per_mw": round(dt_core_per_mw, 5),
        "dt_heater_peak_k_per_mw": round(dt_heater_peak_per_mw, 4),
        "tau_thermal_us": round(tau_th_us, 2),
        "f_3db_to_khz": round(f_3db_khz, 2),
        "metal_proximity_loss_db_per_cm": round(metal_loss_warn_db_cm, 4),
    }


def solve_2d_electrostatic_eo_fem(
    cfg: MRRDesignConfig,
    *,
    roundtrip_length_um: float,
    active_fraction: float = 0.78,
    electrode_config: str = "auto",
    electrode_gap_um: float | None = None,
    electrode_width_um: float = 4.0,
    electrode_thickness_um: float = 0.45,
    electrode_voffset_um: float = 0.75,
    box_thickness_um: float = 2.0,
    top_clad_thickness_um: float = 1.8,
    driver_impedance_ohm: float = 50.0,
) -> dict:
    """Solve 2D anisotropic electrostatic Laplace PDE `-div(eps_0 eps_DC grad phi) = 0` in pure Python (`skfem`).

    Links coplanar GSG (`inplane_gsg`) or vertical parallel-plate (`vertical_parallel`)
    metal electrode geometry and dielectric permittivity discontinuities (`eps_LN` vs `eps_SiO2`)
    directly to the 2D vector electric field `(E_x(x,y), E_y(x,y))` per applied Volt and
    electrostatic line capacitance `C_elec [fF/um]`.
    """
    eo_mat_name = cfg.upper_core_material if (cfg.upper_core_h_um > 0 and cfg.upper_core_material in ("AlN", "ScAlN_10", "ScAlN_20", "ScAlN_30", "LN_Xcut", "LN_Zcut")) else cfg.core_material
    inplane_crystals = {"LN_Xcut"}
    if electrode_config == "auto":
        elec_cfg = "inplane_gsg" if eo_mat_name in inplane_crystals else "vertical_parallel"
    else:
        elec_cfg = electrode_config

    g_elec = float(electrode_gap_um) if (electrode_gap_um is not None and electrode_gap_um > 0) else max(float(cfg.electrode_gap_um), cfg.ring_width_um + 1.2)
    w_elec = max(float(electrode_width_um), 0.8)
    t_elec = max(float(electrode_thickness_um), 0.08)
    h_box = max(float(box_thickness_um), 0.8)
    y_core_top = cfg.film_height_um + cfg.interlayer_h_um + cfg.upper_core_h_um + cfg.cap_h_um

    inv_tan = 0.0 if cfg.sidewall_deg == 90.0 else 1.0 / math.tan(math.radians(cfg.sidewall_deg))
    w_top = max(0.08, cfg.ring_width_um - 2.0 * (cfg.film_height_um - cfg.slab_height_um) * inv_tan)

    if elec_cfg == "inplane_gsg":
        g_eff = max(g_elec, cfg.ring_width_um + 0.30)
        y_el0 = cfg.slab_height_um
        y_el1 = y_el0 + t_elec
        y_ox_top = max(float(top_clad_thickness_um), y_core_top + 0.8, y_el1 + 0.4)
        y_dom_top = y_ox_top + 2.0
        x_dom = max(14.0, 0.5 * g_eff + w_elec + 5.0)
        sig_poly = box(-0.5 * g_eff - w_elec, y_el0, -0.5 * g_eff, y_el1)
        gnd_poly = box(0.5 * g_eff, y_el0, 0.5 * g_eff + w_elec, y_el1)
        x_knots = [
            -x_dom, -0.5 * g_eff - w_elec, -0.5 * g_eff,
            -0.5 * cfg.ring_width_um, -0.5 * w_top, 0.0, 0.5 * w_top, 0.5 * cfg.ring_width_um,
            0.5 * g_eff, 0.5 * g_eff + w_elec, x_dom,
        ]
        y_knots = [-h_box, 0.0, y_el0, y_core_top, y_el1, y_ox_top, y_dom_top]
    else:
        # Vertical electrode pair across crystal c-axis (top electrode above oxide buffer, bottom conductive plane)
        v_buf = max(float(electrode_voffset_um), 0.25)
        y_el0 = y_core_top + v_buf
        y_el1 = y_el0 + t_elec
        d_bot = min(h_box, 0.65)
        g_eff = y_el0 + d_bot
        y_ox_top = max(float(top_clad_thickness_um), y_el1 + 0.35)
        y_dom_top = y_ox_top + 1.5
        x_dom = max(10.0, 0.5 * w_elec + 5.0, 0.5 * cfg.ring_width_um + 5.0)
        # Put +1 V on bottom electrode and 0 V on top electrode so +V gives upward +E_y > 0
        sig_poly = box(-0.5 * max(w_elec * 1.5, cfg.ring_width_um + 2.0), -h_box, 0.5 * max(w_elec * 1.5, cfg.ring_width_um + 2.0), -d_bot)
        gnd_poly = box(-0.5 * w_elec, y_el0, 0.5 * w_elec, y_el1)
        x_knots = [
            -x_dom, -0.5 * w_elec, -0.5 * cfg.ring_width_um, -0.5 * w_top,
            0.0, 0.5 * w_top, 0.5 * cfg.ring_width_um, 0.5 * w_elec, x_dom,
        ]
        y_knots = [-h_box, -d_bot, 0.0, y_core_top, y_el0, y_el1, y_ox_top, y_dom_top]

    if cfg.slab_height_um > 1e-6:
        y_knots.append(cfg.slab_height_um)
    if cfg.upper_core_h_um > 1e-6:
        y_knots.extend([cfg.film_height_um, cfg.film_height_um + cfg.interlayer_h_um])

    x_axis_um = _snapped_1d_axis(x_knots, h_fine=0.10, h_coarse=0.75)
    y_axis_um = _snapped_1d_axis(y_knots, h_fine=0.07, h_coarse=0.40)
    mesh, xc_um, yc_um = _rectilinear_mesh_tri(x_axis_um, y_axis_um)

    eps_core = STATIC_PERMITTIVITY_DC.get(cfg.core_material, (10.0, 10.0))
    eps_clad = STATIC_PERMITTIVITY_DC.get(cfg.clad_material, (3.9, 3.9))
    eps_top = STATIC_PERMITTIVITY_DC.get(cfg.top_clad_material, (3.9, 3.9))

    exx_el = np.full(mesh.nelements, eps_top[0], dtype=float)
    eyy_el = np.full(mesh.nelements, eps_top[1], dtype=float)
    exx_el[yc_um < 0.0] = eps_clad[0]
    eyy_el[yc_um < 0.0] = eps_clad[1]
    exx_el[yc_um > y_ox_top] = 1.0
    eyy_el[yc_um > y_ox_top] = 1.0

    if cfg.slab_height_um > 1e-6:
        in_slab = (yc_um >= 0.0) & (yc_um <= cfg.slab_height_um)
        exx_el[in_slab] = eps_core[0]
        eyy_el[in_slab] = eps_core[1]

    poly_core = Polygon([
        (-0.5 * cfg.ring_width_um, cfg.slab_height_um),
        (0.5 * cfg.ring_width_um, cfg.slab_height_um),
        (0.5 * w_top, cfg.film_height_um),
        (-0.5 * w_top, cfg.film_height_um),
    ])
    in_core = shapely.contains_xy(poly_core, xc_um, yc_um)
    exx_el[in_core] = eps_core[0]
    eyy_el[in_core] = eps_core[1]

    if cfg.upper_core_h_um > 1e-6 and cfg.upper_core_material:
        y_u0 = cfg.film_height_um + cfg.interlayer_h_um
        in_up = (yc_um >= y_u0) & (yc_um <= y_u0 + cfg.upper_core_h_um) & (np.abs(xc_um) <= 0.5 * w_top)
        eps_up = STATIC_PERMITTIVITY_DC.get(cfg.upper_core_material, eps_core)
        exx_el[in_up] = eps_up[0]
        eyy_el[in_up] = eps_up[1]

    basis = Basis(mesh, ElementTriP1())
    basis0 = basis.with_element(ElementTriP0())
    exx_dofs = basis0.zeros(dtype=float)
    eyy_dofs = basis0.zeros(dtype=float)
    exx_dofs[:] = exx_el
    eyy_dofs[:] = eyy_el

    @BilinearForm
    def laplace_eo(u, v, w):
        return EPS0_F_PER_M * (w["exx"] * u.grad[0] * v.grad[0] + w["eyy"] * u.grad[1] * v.grad[1])

    A_eo = laplace_eo.assemble(
        basis,
        exx=basis0.interpolate(exx_dofs),
        eyy=basis0.interpolate(eyy_dofs),
    )
    b_eo = np.zeros(basis.N, dtype=float)

    px_um = mesh.p[0] * 1e6
    py_um = mesh.p[1] * 1e6
    sig_nodes = np.flatnonzero(shapely.covers(sig_poly, shapely.points(px_um, py_um)))
    gnd_nodes = np.flatnonzero(shapely.covers(gnd_poly, shapely.points(px_um, py_um)))
    dir_nodes = np.unique(np.concatenate([sig_nodes, gnd_nodes]))
    phi_bc = np.zeros(basis.N, dtype=float)
    phi_bc[sig_nodes] = 1.0
    phi_bc[gnd_nodes] = 0.0

    phi_1v = solve(*condense(A_eo, b_eo, D=dir_nodes, x=phi_bc))
    phi_2d = phi_1v.reshape(len(y_axis_um), len(x_axis_um))

    # Electric field E = -grad(phi) in [V/um] per 1 V applied
    ey_1v_v_per_um, ex_1v_v_per_um = np.gradient(-phi_2d, y_axis_um, x_axis_um)

    interp_ex = RegularGridInterpolator((y_axis_um, x_axis_um), ex_1v_v_per_um, bounds_error=False, fill_value=0.0)
    interp_ey = RegularGridInterpolator((y_axis_um, x_axis_um), ey_1v_v_per_um, bounds_error=False, fill_value=0.0)

    # Electrostatic capacitance per unit length C = phi^T A_eo phi [F/m] -> [fF/um] (* 1e9)
    c_line_f_per_m = float(phi_1v @ (A_eo @ phi_1v))
    c_line_ff_per_um = max(c_line_f_per_m * 1e9, 0.02)
    l_act_um = float(active_fraction) * float(roundtrip_length_um)
    c_tot_ff = c_line_ff_per_um * l_act_um + 8.0  # incl. ~8 fF pad parasitic
    f_rc_ghz = 1.0 / (2.0 * math.pi * max(float(driver_impedance_ohm), 1.0) * (c_tot_ff * 1e-15)) * 1e-9

    # Core-averaged electric field components per 1 V
    xc_c = xc_um[in_core] if np.any(in_core) else np.array([0.0])
    yc_c = yc_um[in_core] if np.any(in_core) else np.array([0.5 * cfg.film_height_um])
    pts_c = np.column_stack([yc_c, xc_c])
    ex_core_1v = float(np.mean(interp_ex(pts_c)))
    ey_core_1v = float(np.mean(interp_ey(pts_c)))
    e_mag_core_1v = math.hypot(ex_core_1v, ey_core_1v)
    screen_factor = e_mag_core_1v / max(1.0 / g_eff, 1e-9)

    # Azimuthal crystal-axis orientation factor along ring/racetrack circumference
    if eo_mat_name == "LN_Xcut":
        if cfg.topology.startswith("racetrack"):
            l_str = 2.0 * cfg.straight_length_um
            # Push-pull straight Y-arms enjoy 100% alignment; curved arcs with quadrant electrodes give 2/pi
            orient_factor = min(1.0, (l_str + 0.6366 * max(l_act_um - l_str, 0.0)) / max(l_act_um, 1e-6))
        else:
            # Split push-pull electrodes on opposite halves of an X-cut circular ring yield <cos(phi)> = 2/pi
            orient_factor = 2.0 / math.pi
    else:
        orient_factor = 1.0

    return {
        "interp_ex_1v_v_per_um": interp_ex,
        "interp_ey_1v_v_per_um": interp_ey,
        "electrode_config": elec_cfg,
        "electrode_gap_um": round(g_eff, 3),
        "electrode_width_um": round(w_elec, 3),
        "electrode_thickness_um": round(t_elec, 3),
        "active_length_um": round(l_act_um, 2),
        "ex_core_v_per_um_per_v": round(ex_core_1v, 5),
        "ey_core_v_per_um_per_v": round(ey_core_1v, 5),
        "dielectric_screening_factor": round(screen_factor, 4),
        "crystal_azimuthal_factor": round(orient_factor, 4),
        "c_elec_ff_per_um": round(c_line_ff_per_um, 4),
        "c_total_ff": round(c_tot_ff, 2),
        "f_rc_ghz": round(f_rc_ghz, 2),
    }


def _pockels_dn_xyz(
    mat_tensor,
    ex_v_per_um: np.ndarray,
    ey_v_per_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate anisotropic Pockels index perturbation `(dn_x, dn_y, dn_z)` from `(E_x, E_y) [V/um]`.

    Note unit conversion: `r_ij [pm/V] * E [V/um] = 1e-12 [m/V] * 1e6 [V/m] = 1e-6` dimensionless!
    """
    r33 = float(mat_tensor.r33_pm_per_v)
    r13 = float(mat_tensor.r13_pm_per_v)
    if abs(r33) < 1e-9 and abs(r13) < 1e-9:
        z = np.zeros_like(ex_v_per_um, dtype=complex)
        return z, z, z

    sym = str(mat_tensor.crystal_symmetry)
    if "Xcut" in sym:
        # Optic c/Z-axis is horizontal x: E_x couples to r_33 along x and r_13 along y, z
        dn_x = -0.5 * (mat_tensor.nx ** 3) * (r33 * 1e-6) * ex_v_per_um
        dn_y = -0.5 * (mat_tensor.ny ** 3) * (r13 * 1e-6) * ex_v_per_um
        dn_z = -0.5 * (mat_tensor.nz ** 3) * (r13 * 1e-6) * ex_v_per_um
    elif "zincblende" in sym:
        # Cubic 43m (3C-SiC): r_41 stored in r33_pm_per_v couples vertical E_y into in-plane birefringence
        dn_x = -0.5 * (mat_tensor.nx ** 3) * (r33 * 1e-6) * ey_v_per_um
        dn_y = np.zeros_like(ey_v_per_um)
        dn_z = +0.5 * (mat_tensor.nz ** 3) * (r33 * 1e-6) * ey_v_per_um
    else:
        # Uniaxial vertical c/Z-axis (LN_Zcut, AlN, ScAlN, 4H-SiC, GaN): E_y couples to r_33 (y) & r_13 (x, z)
        dn_y = -0.5 * (mat_tensor.ny ** 3) * (r33 * 1e-6) * ey_v_per_um
        dn_x = -0.5 * (mat_tensor.nx ** 3) * (r13 * 1e-6) * ey_v_per_um
        dn_z = -0.5 * (mat_tensor.nz ** 3) * (r13 * 1e-6) * ey_v_per_um

    return dn_x.astype(complex), dn_y.astype(complex), dn_z.astype(complex)


# ==============================================================================
# 3. Unified 2D-FEM Active Tuning + Multi-State Spectrum Synthesis
# ==============================================================================

def _resolve_mechanism_and_drive(
    cfg: MRRDesignConfig,
    mechanism: str,
    drive_type: str,
    bias_values: list[float] | None,
) -> tuple[str, str, list[float]]:
    core_fn = get_material_fn(
        cfg.core_material,
        sc_fraction=cfg.sc_fraction if cfg.core_material.startswith("ScAlN") else None,
        custom_n=cfg.custom_n,
        custom_k=cfg.custom_k,
    )
    mat_c = core_fn(cfg.wavelength_nm * 1e-9, T_REF_K)
    has_pockels = abs(mat_c.r33_pm_per_v) > 0.05
    if not has_pockels and cfg.upper_core_h_um > 0 and cfg.upper_core_material:
        up_mat = get_material_fn(cfg.upper_core_material)(cfg.wavelength_nm * 1e-9, T_REF_K)
        has_pockels = abs(up_mat.r33_pm_per_v) > 0.05

    mech = mechanism.lower().strip()
    drv = drive_type.lower().strip()

    if drv == "delta_temp_k":
        mech = "thermo_optic"
    elif drv == "e_field_v_per_um":
        mech = "pockels_eo"

    if mech == "auto":
        if cfg.core_material == "Si" and drv in ("auto", "voltage_v"):
            mech = "si_pn_depletion"
        elif cfg.core_material == "Si" and drv == "current_ma":
            mech = "si_pin_injection"
        elif has_pockels and drv in ("auto", "voltage_v", "e_field_v_per_um"):
            mech = "pockels_eo"
        else:
            mech = "thermo_optic"

    if drv == "auto":
        drv = TUNING_MECHANISM_CATALOG[mech]["default_drive"]

    if bias_values is not None and len(bias_values) > 0:
        vals = [float(v) for v in bias_values]
    else:
        if drv == "delta_temp_k":
            vals = [0.0, 10.0, 25.0, 50.0, 75.0]
        elif drv == "e_field_v_per_um":
            vals = [-10.0, -5.0, 0.0, 5.0, 10.0]
        elif mech == "si_pn_depletion":
            vals = [0.0, -1.0, -2.0, -4.0, -6.0]
        elif mech == "si_pin_injection":
            vals = [0.0, 0.25, 0.50, 1.00, 2.00]
        elif mech == "pockels_eo":
            vals = [-15.0, -7.5, 0.0, 7.5, 15.0]
        elif drv == "voltage_v":
            vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        elif drv == "power_mw":
            vals = [0.0, 5.0, 10.0, 20.0, 30.0]
        else:
            vals = [0.0, 2.0, 4.0, 6.0, 8.0]

    return mech, drv, vals


def _drive_unit_label(drive_type: str) -> tuple[str, str]:
    mapping = {
        "voltage_v": ("V", "Voltage"),
        "current_ma": ("mA", "Current"),
        "power_mw": ("mW", "Heater Power"),
        "delta_temp_k": ("K", "Temperature Shift ΔT"),
        "e_field_v_per_um": ("V/um", "Electric Field E"),
    }
    return mapping.get(drive_type, ("V", "Bias"))


def run_active_tuning_analysis(
    base_cfg: MRRDesignConfig,
    *,
    mechanism: str = "auto",
    drive_type: str = "auto",
    bias_values: list[float] | None = None,
    active_fraction: float = 0.78,
    # Thermo-optic & microheater geometry parameters
    heater_material: str = "TiN",
    heater_width_um: float = 1.60,
    heater_thickness_um: float = 0.12,
    heater_gap_um: float = 0.90,
    heater_offset_um: float = 0.0,
    heater_resistance_ohm: float | None = None,
    box_thickness_um: float = 2.00,
    top_clad_thickness_um: float = 1.80,
    # Pockels EO electrode & direct E-field parameters
    electrode_config: str = "auto",
    electrode_gap_um: float | None = None,
    electrode_width_um: float = 4.00,
    electrode_thickness_um: float = 0.45,
    electrode_voffset_um: float = 0.75,
    e_field_axis: str = "auto",
    # Silicon PN depletion & PIN injection geometry/doping parameters
    na_p_cm3: float = 5.0e17,
    nd_n_cm3: float = 5.0e17,
    pn_offset_nm: float = 25.0,
    pn_slab_height_um: float | None = None,
    heavy_contact_offset_um: float = 0.85,
    carrier_lifetime_ns: float = 1.0,
    driver_impedance_ohm: float = 50.0,
    precomputed_passive_metrics: dict | None = None,
) -> dict:
    """Compute device tuning efficiency & multi-state spectra via perturbed 2D-FEM eigenmodes (Zero COMSOL).

    Pipeline:
      1. Evaluates baseline passive MRR (`simulate_mrr_topology`) for spatial-CMT coupler
         `kappa_thru`, `kappa_drop`, roundtrip length `L_rt`, and group index `n_g`.
      2. Solves the geometry-linked 2D cross-section multiphysics problem in pure Python:
         - 2D FEM Fourier heat conduction (`thermo_optic`),
         - 2D FEM anisotropic Laplace electrostatics (`pockels_eo`), or
         - 2D Debye-screened Silicon PN depletion / PIN injection (`si_pn_depletion` / `si_pin_injection`).
      3. Perturbs the spatial complex refractive index `n(x, y) + i*k(x, y)` inside the 2D conformal
         vector Maxwell eigenmode solver (`solve_perturbed_neff_sweep_fem`) at every state, extracting
         exact `Re(n_eff)` and `Im(n_eff)` (modal FCA/absorption).
      4. Synthesizes broadband transmission spectra (`thru_dB`, `drop_dB`, `thru_phase_rad`) at each
         voltage / current / temperature / E-field state and extracts full device KPIs.
    """
    t0 = time.perf_counter()
    mech, drv, vals = _resolve_mechanism_and_drive(base_cfg, mechanism, drive_type, bias_values)
    unit_str, drive_label = _drive_unit_label(drv)
    f_act = float(np.clip(active_fraction, 0.05, 1.0))

    # For Silicon lateral PN / PIN junctions, ensure a physical doped Si slab exists for electrical contact
    work_cfg = replace(base_cfg, delta_temp_k=0.0, voltage_v=0.0)
    slab_auto_promoted = False
    if mech in ("si_pn_depletion", "si_pin_injection"):
        if work_cfg.core_material != "Si":
            raise ValueError(
                f"Mechanism '{mech}' requires core_material='Si' (got '{work_cfg.core_material}'). "
                "Use 'pockels_eo' or 'thermo_optic' for non-Silicon waveguides."
            )
        if pn_slab_height_um is not None and pn_slab_height_um > 0:
            work_cfg = replace(work_cfg, slab_height_um=min(float(pn_slab_height_um), 0.75 * work_cfg.film_height_um))
        elif work_cfg.slab_height_um <= 0.01:
            auto_slab = round(min(0.09, 0.40 * work_cfg.film_height_um), 3)
            work_cfg = replace(work_cfg, slab_height_um=auto_slab)
            slab_auto_promoted = True

    # 1. Baseline passive MRR synthesis (spatial CMT couplers + cavity roundtrip), or fast straight-arm metrics
    if precomputed_passive_metrics is not None:
        pm = precomputed_passive_metrics
    else:
        passive_sim = simulate_mrr_topology(work_cfg)
        pm = passive_sim["metrics"]
    l_rt_um = float(pm["roundtrip_length_um"])
    l_act_um = f_act * l_rt_um
    neff_0_rt = float(pm["neff_ring"])
    ng_0_rt = float(pm["ng_ring"])
    kp_thru = float(pm["kappa_power_thru"])
    kp_drop = float(pm["kappa_power_drop"])
    alpha_passive_db_cm = float(pm["alpha_loss_db_per_cm"])
    q_loaded_passive = float(pm["Q_loaded"])
    is_4port = TOPOLOGY_CATALOG[work_cfg.topology]["ports"] == 4

    wl_nm = float(work_cfg.wavelength_nm)
    wl_m = wl_nm * 1e-9
    wl_um = wl_nm * 1e-3
    wl_scale = float(np.clip(wl_nm / 1550.0, 0.35, 4.50))

    core_fn = get_material_fn(
        work_cfg.core_material,
        sc_fraction=work_cfg.sc_fraction if work_cfg.core_material.startswith("ScAlN") else None,
        custom_n=work_cfg.custom_n,
        custom_k=work_cfg.custom_k,
    )
    clad_fn = get_material_fn(work_cfg.clad_material)
    top_fn = get_material_fn(work_cfg.top_clad_material)
    up_fn = get_material_fn(work_cfg.upper_core_material) if work_cfg.upper_core_material else None
    cap_fn = get_material_fn(work_cfg.cap_material) if work_cfg.cap_material else None

    mat_core = core_fn(wl_m, T_REF_K)
    mat_clad = clad_fn(wl_m, T_REF_K)
    mat_top = top_fn(wl_m, T_REF_K)
    mat_up = up_fn(wl_m, T_REF_K) if up_fn else None

    if mech == "pockels_eo":
        has_r = (abs(mat_core.r33_pm_per_v) > 0.05) or (mat_up is not None and abs(mat_up.r33_pm_per_v) > 0.05)
        if not has_r:
            raise ValueError(
                f"Material '{work_cfg.core_material}' is centrosymmetric/amorphous with r33 = 0 pm/V. "
                "Select LN_Xcut, LN_Zcut, AlN, ScAlN_20, 4H-SiC, 3C-SiC, or GaN for Pockels EO, "
                "or use 'si_pn_depletion' / 'thermo_optic'."
            )

    box_pad_um = max(1.85, 1.35 * wl_um) if abs(wl_scale - 1.0) >= 0.05 else 1.85
    dom_w_um = max(5.2, work_cfg.ring_width_um + 2.8 * max(wl_um, 1.0))
    if mech in ("si_pn_depletion", "si_pin_injection"):
        dom_w_um = max(dom_w_um, work_cfg.ring_width_um + 2.0 * float(heavy_contact_offset_um) + 1.8)

    ring_spec = WaveguideCrossSectionSpec(
        w_bottom_um=work_cfg.ring_width_um,
        h_film_um=work_cfg.film_height_um,
        h_slab_um=work_cfg.slab_height_um,
        sidewall_deg=work_cfg.sidewall_deg,
        core_material_fn=core_fn,
        clad_material_fn=clad_fn,
        top_clad_material_fn=top_fn,
        interlayer_h_um=work_cfg.interlayer_h_um,
        upper_core_h_um=work_cfg.upper_core_h_um,
        upper_core_material_fn=up_fn,
        cap_h_um=work_cfg.cap_h_um,
        cap_w_um=work_cfg.cap_w_um,
        cap_material_fn=cap_fn,
        domain_width_um=dom_w_um,
        h_box_um=box_pad_um,
        h_top_clad_um=box_pad_um,
        temperature_k=T_REF_K,
    )
    bend_r_m = math.inf if work_cfg.topology.startswith("racetrack") else (work_cfg.ring_radius_um * 1e-6)
    mesh_res_um = float(np.clip(min(work_cfg.core_mesh_res_um, 0.042) * wl_scale, 0.018, 0.160))

    # 2. Build spatial perturbation callbacks per state (plus state 0 reference = 0 bias)
    all_states_vals = [0.0] + vals
    perturbation_fns = []
    geometry_physics: dict = {
        "mechanism": mech,
        "mechanism_name_en": TUNING_MECHANISM_CATALOG[mech]["name_en"],
        "mechanism_name_zh": TUNING_MECHANISM_CATALOG[mech]["name_zh"],
        "drive_type": drv,
        "drive_unit": unit_str,
        "active_fraction": round(f_act, 4),
        "active_length_um": round(l_act_um, 2),
        "dn_dt_core_per_k": mat_core.dn_dt_per_k,
    }
    state_aux_meta: list[dict] = []

    reg_mat_lookup = {
        "core": mat_core,
        "slab": mat_core,
        "box": mat_clad,
        "top_clad": mat_top,
    }
    if mat_up is not None:
        reg_mat_lookup["upper_core"] = mat_up

    if mech == "thermo_optic":
        if drv == "delta_temp_k":
            r_heat = 250.0
            geometry_physics.update({
                "solver": "Direct Cross-Section Temperature Field + 2D Vector FEM Eigenmode",
                "core_dn_dt_per_k": mat_core.dn_dt_per_k,
                "clad_dn_dt_per_k": mat_clad.dn_dt_per_k,
            })
            for v_val in all_states_vals:
                dt_k = float(v_val)
                state_aux_meta.append({
                    "delta_t_core_k": round(dt_k, 4),
                    "delta_T_core_K": round(dt_k, 4),
                    "e_core_v_per_um": 0.0,
                })
                if abs(dt_k) < 1e-12:
                    perturbation_fns.append(None)
                else:
                    def _mk_dt_fn(dt_local: float):
                        def _fn(x_um: np.ndarray, y_um: np.ndarray, reg_name: str):
                            _ = y_um
                            m_reg = reg_mat_lookup.get(reg_name, mat_top)
                            dn = np.full_like(x_um, m_reg.dn_dt_per_k * dt_local, dtype=complex)
                            return dn, dn, dn
                        return _fn
                    perturbation_fns.append(_mk_dt_fn(dt_k))
        else:
            th_fem = solve_2d_heater_thermal_fem(
                work_cfg,
                roundtrip_length_um=l_rt_um,
                active_fraction=f_act,
                box_thickness_um=box_thickness_um,
                top_clad_thickness_um=top_clad_thickness_um,
                heater_material=heater_material,
                heater_width_um=heater_width_um,
                heater_thickness_um=heater_thickness_um,
                heater_gap_um=heater_gap_um,
                heater_offset_um=heater_offset_um,
                heater_resistance_ohm=heater_resistance_ohm,
            )
            r_heat = float(th_fem["heater_resistance_ohm"])
            interp_dt = th_fem["interp_dt_per_mw"]
            r_th_core = float(th_fem["r_th_core_k_per_mw"])
            geometry_physics.update({
                "solver": "2D scikit-fem Steady-State Heat Conduction + 2D Vector FEM Eigenmode (Zero COMSOL)",
                **{k: v for k, v in th_fem.items() if k != "interp_dt_per_mw"},
                "core_dn_dt_per_k": mat_core.dn_dt_per_k,
                "clad_dn_dt_per_k": mat_clad.dn_dt_per_k,
            })
            for v_val in all_states_vals:
                if drv == "current_ma":
                    p_mw = ((float(v_val) * 1e-3) ** 2) * r_heat * 1e3
                elif drv == "voltage_v":
                    p_mw = ((float(v_val) ** 2) / max(r_heat, 1e-3)) * 1e3
                else:
                    p_mw = max(float(v_val), 0.0)
                state_aux_meta.append({
                    "power_mw": round(p_mw, 4),
                    "delta_t_core_k": round(p_mw * r_th_core, 4),
                    "delta_T_core_K": round(p_mw * r_th_core, 4),
                    "e_core_v_per_um": 0.0,
                })
                if abs(p_mw) < 1e-12:
                    perturbation_fns.append(None)
                else:
                    def _mk_p_fn(p_local: float):
                        def _fn(x_um: np.ndarray, y_um: np.ndarray, reg_name: str):
                            m_reg = reg_mat_lookup.get(reg_name, mat_top)
                            dt_loc = p_local * interp_dt(np.column_stack([y_um, x_um]))
                            dn = (m_reg.dn_dt_per_k * dt_loc).astype(complex)
                            return dn, dn, dn
                        return _fn
                    perturbation_fns.append(_mk_p_fn(p_mw))

    elif mech == "pockels_eo":
        f_opt_ghz = (C0_M_PER_S / (wl_m * max(q_loaded_passive, 1.0))) * 1e-9
        if drv == "e_field_v_per_um":
            eo_orient = 1.0
            inplane = "Xcut" in str(mat_core.crystal_symmetry)
            use_ex = (e_field_axis == "x") or (e_field_axis == "auto" and inplane)
            geometry_physics.update({
                "solver": "Direct Anisotropic Pockels Tensor + 2D Vector FEM Eigenmode",
                "e_field_component": "Ex (horizontal)" if use_ex else "Ey (vertical)",
                "r33_pm_per_v": mat_core.r33_pm_per_v if abs(mat_core.r33_pm_per_v) > 0 else (mat_up.r33_pm_per_v if mat_up else 0.0),
                "r13_pm_per_v": mat_core.r13_pm_per_v if abs(mat_core.r13_pm_per_v) > 0 else (mat_up.r13_pm_per_v if mat_up else 0.0),
                "f_photon_ghz": round(f_opt_ghz, 2),
            })
            for e_val in all_states_vals:
                ev = float(e_val)
                state_aux_meta.append({
                    "power_mw": 0.0,
                    "e_core_v_per_um": round(ev, 4),
                    "Ex_core_v_per_um": round(ev if use_ex else 0.0, 4),
                    "Ey_core_v_per_um": round(0.0 if use_ex else ev, 4),
                })
                if abs(ev) < 1e-12:
                    perturbation_fns.append(None)
                else:
                    def _mk_e_fn(e_loc: float):
                        def _fn(x_um: np.ndarray, y_um: np.ndarray, reg_name: str):
                            _ = y_um
                            m_reg = reg_mat_lookup.get(reg_name, mat_top)
                            ex_arr = np.full_like(x_um, e_loc if use_ex else 0.0)
                            ey_arr = np.full_like(x_um, 0.0 if use_ex else e_loc)
                            return _pockels_dn_xyz(m_reg, ex_arr, ey_arr)
                        return _fn
                    perturbation_fns.append(_mk_e_fn(ev))
        else:
            eo_fem = solve_2d_electrostatic_eo_fem(
                work_cfg,
                roundtrip_length_um=l_rt_um,
                active_fraction=f_act,
                electrode_config=electrode_config,
                electrode_gap_um=electrode_gap_um,
                electrode_width_um=electrode_width_um,
                electrode_thickness_um=electrode_thickness_um,
                electrode_voffset_um=electrode_voffset_um,
                box_thickness_um=box_thickness_um,
                top_clad_thickness_um=top_clad_thickness_um,
                driver_impedance_ohm=driver_impedance_ohm,
            )
            interp_ex = eo_fem["interp_ex_1v_v_per_um"]
            interp_ey = eo_fem["interp_ey_1v_v_per_um"]
            eo_orient = float(eo_fem["crystal_azimuthal_factor"])
            f_rc_ghz = float(eo_fem["f_rc_ghz"])
            f_eo_3db = 1.0 / math.hypot(1.0 / max(f_rc_ghz, 1e-6), 1.0 / max(f_opt_ghz, 1e-6))
            e_per_v = math.hypot(float(eo_fem["ex_core_v_per_um_per_v"]), float(eo_fem["ey_core_v_per_um_per_v"]))
            geometry_physics.update({
                "solver": "2D scikit-fem Anisotropic Electrostatic Laplace + 2D Vector FEM Eigenmode (Zero COMSOL)",
                **{k: v for k, v in eo_fem.items() if not k.startswith("interp_")},
                "f_photon_ghz": round(f_opt_ghz, 2),
                "f_3db_eo_ghz": round(f_eo_3db, 2),
                "r33_pm_per_v": mat_core.r33_pm_per_v if abs(mat_core.r33_pm_per_v) > 0 else (mat_up.r33_pm_per_v if mat_up else 0.0),
                "r13_pm_per_v": mat_core.r13_pm_per_v if abs(mat_core.r13_pm_per_v) > 0 else (mat_up.r13_pm_per_v if mat_up else 0.0),
            })
            for v_val in all_states_vals:
                vv = float(v_val)
                state_aux_meta.append({
                    "power_mw": 0.0,
                    "e_core_v_per_um": round(vv * e_per_v, 4),
                    "Ex_core_v_per_um": round(vv * float(eo_fem["ex_core_v_per_um_per_v"]), 4),
                    "Ey_core_v_per_um": round(vv * float(eo_fem["ey_core_v_per_um_per_v"]), 4),
                })
                if abs(vv) < 1e-12:
                    perturbation_fns.append(None)
                else:
                    def _mk_v_eo_fn(v_loc: float):
                        def _fn(x_um: np.ndarray, y_um: np.ndarray, reg_name: str):
                            m_reg = reg_mat_lookup.get(reg_name, mat_top)
                            if abs(m_reg.r33_pm_per_v) < 1e-9 and abs(m_reg.r13_pm_per_v) < 1e-9:
                                z = np.zeros_like(x_um, dtype=complex)
                                return z, z, z
                            pts = np.column_stack([y_um, x_um])
                            ex_loc = (v_loc * eo_orient) * interp_ex(pts)
                            ey_loc = (v_loc * eo_orient) * interp_ey(pts)
                            return _pockels_dn_xyz(m_reg, ex_loc, ey_loc)
                        return _fn
                    perturbation_fns.append(_mk_v_eo_fn(vv))

    elif mech == "si_pn_depletion":
        na = max(float(na_p_cm3), 1e15)
        nd = max(float(nd_n_cm3), 1e15)
        vt_v = (KB_J_PER_K * T_REF_K) / Q_E_C
        v_bi = vt_v * math.log((na * nd) / (NI_SI_300K_CM3 ** 2))
        eps_si = EPS0_F_PER_M * 11.7
        na_m3 = na * 1e6
        nd_m3 = nd * 1e6
        ld_p_um = math.sqrt((eps_si * KB_J_PER_K * T_REF_K) / ((Q_E_C ** 2) * na_m3)) * 1e6
        ld_n_um = math.sqrt((eps_si * KB_J_PER_K * T_REF_K) / ((Q_E_C ** 2) * nd_m3)) * 1e6
        xj_um = float(pn_offset_nm) * 1e-3
        d_cont_um = max(float(heavy_contact_offset_um), 0.15)
        x_pp_edge = -0.5 * work_cfg.ring_width_um - d_cont_um
        x_nn_edge = +0.5 * work_cfg.ring_width_um + d_cont_um

        def _w_dep_um(v_bias: float) -> float:
            v_eff = min(float(v_bias), v_bi - 0.08)
            w_m = math.sqrt((2.0 * eps_si / Q_E_C) * ((1.0 / na_m3) + (1.0 / nd_m3)) * (v_bi - v_eff))
            return w_m * 1e6

        w_dep_0_um = _w_dep_um(0.0)
        v_min_rev = min(vals) if vals else -4.0
        w_dep_max_um = _w_dep_um(v_min_rev)

        # Geometry-derived slab access series resistance R_s [Ohm] & junction capacitance C_j [fF/um]
        mu_p, mu_n = _si_carrier_mobilities_cm2_vs(na, nd)
        rho_p_ohm_m = 0.01 / (Q_E_C * mu_p * na)
        rho_n_ohm_m = 0.01 / (Q_E_C * mu_n * nd)
        h_slab_m = max(work_cfg.slab_height_um, 0.05) * 1e-6
        l_act_m = max(l_act_um, 5.0) * 1e-6
        r_slab_p = rho_p_ohm_m * ((d_cont_um + 0.25 * work_cfg.ring_width_um) * 1e-6) / (h_slab_m * l_act_m)
        r_slab_n = rho_n_ohm_m * ((d_cont_um + 0.25 * work_cfg.ring_width_um) * 1e-6) / (h_slab_m * l_act_m)
        r_series_ohm = r_slab_p + r_slab_n + 4.5  # + ohmic silicide contact resistance

        cj_0v_ff_per_um = (eps_si * (work_cfg.film_height_um * 1e-6) / max(w_dep_0_um * 1e-6, 15e-9) + 0.06e-9) * 1e9
        v_mid_bias = 0.5 * v_min_rev
        cj_mid_ff_per_um = (eps_si * (work_cfg.film_height_um * 1e-6) / max(_w_dep_um(v_mid_bias) * 1e-6, 15e-9) + 0.06e-9) * 1e9
        c_tot_ff = cj_mid_ff_per_um * l_act_um + 8.0
        f_rc_ghz = 1.0 / (2.0 * math.pi * (r_series_ohm + float(driver_impedance_ohm)) * (c_tot_ff * 1e-15)) * 1e-9
        f_opt_ghz = (C0_M_PER_S / (wl_m * max(q_loaded_passive, 1.0))) * 1e-9
        f_eo_3db = 1.0 / math.hypot(1.0 / max(f_rc_ghz, 1e-6), 1.0 / max(f_opt_ghz, 1e-6))

        geometry_physics.update({
            "solver": "2D Debye-Screened Silicon PN Poisson Depletion + 2D Vector FEM Eigenmode (Zero COMSOL)",
            "slab_height_um": work_cfg.slab_height_um,
            "slab_auto_promoted_from_strip": slab_auto_promoted,
            "na_p_cm3": na,
            "nd_n_cm3": nd,
            "pn_offset_nm": round(float(pn_offset_nm), 2),
            "heavy_contact_offset_um": round(d_cont_um, 3),
            "v_bi_v": round(v_bi, 4),
            "w_dep_0v_nm": round(w_dep_0_um * 1e3, 2),
            "w_dep_max_nm": round(w_dep_max_um * 1e3, 2),
            "r_series_ohm": round(r_series_ohm, 2),
            "cj_0v_ff_per_um": round(cj_0v_ff_per_um, 4),
            "cj_bias_ff_per_um": round(cj_mid_ff_per_um, 4),
            "c_total_ff": round(c_tot_ff, 2),
            "f_rc_ghz": round(f_rc_ghz, 2),
            "f_photon_ghz": round(f_opt_ghz, 2),
            "f_3db_eo_ghz": round(f_eo_3db, 2),
        })

        for v_val in all_states_vals:
            vv = float(v_val)
            w_d = _w_dep_um(vv)
            e_pk = (2.0 * max(v_bi - min(vv, v_bi - 0.08), 0.08)) / max(w_d, 0.02)
            cj_v_ff = (eps_si * (work_cfg.film_height_um * 1e-6) / max(w_d * 1e-6, 15e-9) + 0.06e-9) * 1e9
            state_aux_meta.append({
                "power_mw": 0.0,
                "e_core_v_per_um": round(e_pk, 3),
                "w_dep_nm": round(w_d * 1e3, 2),
                "cj_ff_per_um": round(cj_v_ff, 4),
            })

            def _mk_pn_fn(v_loc: float):
                wd_loc = _w_dep_um(v_loc)
                xp_edge = xj_um - wd_loc * (nd / (na + nd))
                xn_edge = xj_um + wd_loc * (na / (na + nd))

                def _fn(x_um: np.ndarray, y_um: np.ndarray, reg_name: str):
                    _ = y_um
                    if reg_name not in ("core", "slab"):
                        z = np.zeros_like(x_um, dtype=complex)
                        return z, z, z
                    # Debye-screened majority hole & electron spatial profiles
                    p_loc = 0.5 * na * erfc((x_um - xp_edge) / (math.sqrt(2.0) * max(ld_p_um, 0.002)))
                    n_loc = 0.5 * nd * erfc((xn_edge - x_um) / (math.sqrt(2.0) * max(ld_n_um, 0.002)))
                    if reg_name == "slab":
                        # Heavy p++ / n++ (5e19 cm^-3) ohmic contact wings in outer slab
                        p_wing = 5.0e19 * 0.5 * erfc((x_um - x_pp_edge) / 0.035)
                        n_wing = 5.0e19 * 0.5 * erfc((x_nn_edge - x_um) / 0.035)
                        p_loc = np.maximum(p_loc, p_wing)
                        n_loc = np.maximum(n_loc, n_wing)
                    dn_r, dk_i = soref_bennett_si_dn_dk(n_loc, p_loc, wl_nm)
                    c_pert = dn_r + 1j * dk_i
                    return c_pert, c_pert, c_pert
                return _fn

            perturbation_fns.append(_mk_pn_fn(vv))

    elif mech == "si_pin_injection":
        tau_s = max(float(carrier_lifetime_ns), 0.05) * 1e-9
        d_cont_um = max(float(heavy_contact_offset_um), 0.25)
        w_i_um = work_cfg.ring_width_um + 2.0 * d_cont_um
        # Ambipolar diffusion length L_a ~ sqrt(D_a * tau_c) in c-Si (D_a ~ 18 cm^2/s)
        l_amb_um = math.sqrt(18.0e-4 * tau_s) * 1e6
        v_eff_cm3 = (
            (work_cfg.ring_width_um * work_cfg.film_height_um + 2.0 * d_cont_um * work_cfg.slab_height_um)
            * 1e-8
            * (l_act_um * 1e-4)
        )
        f_pin_ghz = 1.0 / (2.0 * math.pi * tau_s) * 1e-9
        f_opt_ghz = (C0_M_PER_S / (wl_m * max(q_loaded_passive, 1.0))) * 1e-9
        f_eo_3db = 1.0 / math.hypot(1.0 / max(f_pin_ghz, 1e-6), 1.0 / max(f_opt_ghz, 1e-6))

        geometry_physics.update({
            "solver": "2D Silicon PIN Ambipolar Injection + 2D Vector FEM Eigenmode (Zero COMSOL)",
            "slab_height_um": work_cfg.slab_height_um,
            "slab_auto_promoted_from_strip": slab_auto_promoted,
            "intrinsic_width_um": round(w_i_um, 3),
            "heavy_contact_offset_um": round(d_cont_um, 3),
            "carrier_lifetime_ns": round(float(carrier_lifetime_ns), 3),
            "ambipolar_diffusion_length_um": round(l_amb_um, 2),
            "f_carrier_ghz": round(f_pin_ghz, 3),
            "f_3db_eo_ghz": round(f_eo_3db, 3),
        })

        x_pp_edge = -0.5 * work_cfg.ring_width_um - d_cont_um
        x_nn_edge = +0.5 * work_cfg.ring_width_um + d_cont_um
        for i_val in all_states_vals:
            i_ma = max(float(i_val), 0.0)
            v_diode = 0.72 + 0.052 * math.log1p(i_ma / 0.05) if i_ma > 1e-6 else 0.0
            n_inj_est = ((i_ma * 1e-3) * tau_s) / (Q_E_C * max(v_eff_cm3, 1e-18))
            state_aux_meta.append({
                "power_mw": round(i_ma * v_diode, 4),
                "delta_N_cm3": float(f"{n_inj_est:.3e}"),
                "e_core_v_per_um": 0.0,
            })

            def _mk_pin_fn(curr_ma: float):
                n_inj_peak = ((curr_ma * 1e-3) * tau_s) / (Q_E_C * max(v_eff_cm3, 1e-18))

                def _fn(x_um: np.ndarray, y_um: np.ndarray, reg_name: str):
                    _ = y_um
                    if reg_name not in ("core", "slab"):
                        z = np.zeros_like(x_um, dtype=complex)
                        return z, z, z
                    profile = np.cosh(np.clip(x_um / max(l_amb_um, 0.2), -4.0, 4.0)) / math.cosh(
                        min(0.5 * w_i_um / max(l_amb_um, 0.2), 4.0)
                    )
                    n_inj = n_inj_peak * profile
                    p_loc = n_inj.copy()
                    n_loc = n_inj.copy()
                    if reg_name == "slab":
                        p_loc += 5.0e19 * 0.5 * erfc((x_um - x_pp_edge) / 0.035)
                        n_loc += 5.0e19 * 0.5 * erfc((x_nn_edge - x_um) / 0.035)
                    dn_r, dk_i = soref_bennett_si_dn_dk(n_loc, p_loc, wl_nm)
                    c_pert = dn_r + 1j * dk_i
                    return c_pert, c_pert, c_pert
                return _fn

            perturbation_fns.append(_mk_pin_fn(i_ma))
    else:
        raise ValueError(f"Unknown tuning mechanism '{mech}'")

    # 3. Solve perturbed 2D vector Maxwell eigenmodes across intrinsic + 0-bias + all bias states
    fem_rows = solve_perturbed_neff_sweep_fem(
        ring_spec,
        wl_m,
        [None] + perturbation_fns,
        bend_radius_m=bend_r_m,
        polarization=work_cfg.polarization,
        core_mesh_res_um=mesh_res_um,
    )

    neff_intrinsic_fem = float(fem_rows[0]["neff_real"])
    ref_fem = fem_rows[1]
    state_fem_rows = fem_rows[2:]
    neff_ref_active = float(ref_fem["neff_real"])

    # Roundtrip baseline effective index including 0-bias PN/PIN implant offset
    dneff_implant_0 = f_act * (neff_ref_active - neff_intrinsic_fem)
    neff_rt_0 = neff_0_rt + dneff_implant_0

    # Shared adaptive high-resolution wavelength grid covering all shifted resonances cleanly
    fsr_nm = float(pm["FSR_nm"])
    fsr_ghz = float(pm["FSR_GHz"])
    fwhm_nm = max(float(pm["FWHM_pm"]) * 1e-3, 1e-4)

    max_shift_est_nm = max(
        abs(f_act * (float(r["neff_real"]) - neff_ref_active) * wl_nm / ng_0_rt)
        for r in state_fem_rows
    )
    span_nm = max(float(work_cfg.span_nm), min(2.2 * fsr_nm, max(4.0 * max_shift_est_nm + 8.0 * fwhm_nm, 1.2)))
    wl_min_nm = wl_nm - 0.5 * span_nm
    wl_max_nm = wl_nm + 0.5 * span_nm
    wl_base = np.linspace(wl_min_nm, wl_max_nm, 1600)

    dn_dlam_nm = -(ng_0_rt - neff_rt_0) / wl_nm
    l_rt_nm = l_rt_um * 1e3
    # Pick integer azimuthal order M closest to nominal center wavelength at 0-bias
    m_target = max(1, int(round(neff_rt_0 * l_rt_nm / wl_nm)))

    def _exact_res_wl(dneff_rt: float) -> float:
        num_r = (neff_rt_0 - dn_dlam_nm * wl_nm + dneff_rt) * l_rt_nm
        den_r = float(m_target) - dn_dlam_nm * l_rt_nm
        return num_r / max(den_r, 1e-9)

    lam_res_0 = _exact_res_wl(0.0)
    # Re-center window around tracked resonance M if needed so every shifted curve is clearly visible
    if not (wl_min_nm + 0.1 * span_nm <= lam_res_0 <= wl_max_nm - 0.1 * span_nm):
        wl_min_nm = lam_res_0 - 0.5 * span_nm
        wl_max_nm = lam_res_0 + 0.5 * span_nm
        wl_base = np.linspace(wl_min_nm, wl_max_nm, 1600)

    clusters = [wl_base]
    for r in [ref_fem] + state_fem_rows:
        dneff_rt = f_act * (float(r["neff_real"]) - neff_ref_active)
        for dm in (-1, 0, +1):
            m_use = m_target + dm
            if m_use <= 0:
                continue
            num_r = (neff_rt_0 - dn_dlam_nm * wl_nm + dneff_rt) * l_rt_nm
            den_r = float(m_use) - dn_dlam_nm * l_rt_nm
            lam_c = num_r / max(den_r, 1e-9)
            if wl_min_nm <= lam_c <= wl_max_nm:
                hw = max(min(3.5 * fwhm_nm, 0.20 * fsr_nm), 2e-5)
                clusters.append(np.linspace(max(wl_min_nm, lam_c - hw), min(wl_max_nm, lam_c + hw), 55))
    wl_grid = np.unique(np.concatenate(clusters))

    # 4. Synthesize multi-bias spectra & per-state resonance KPIs
    state_records: list[dict] = []
    spectra_curves: list[dict] = []
    extra_metal_loss = float(geometry_physics.get("metal_proximity_loss_db_per_cm", 0.0))

    for idx_s, (b_val, fem_r, aux) in enumerate(zip(vals, state_fem_rows, state_aux_meta[1:])):
        dneff_active_re = float(fem_r["neff_real"]) - neff_ref_active
        dneff_rt_re = f_act * dneff_active_re

        alpha_active_modal = float(fem_r["alpha_modal_db_per_cm"]) + extra_metal_loss
        alpha_rt_db_cm = alpha_passive_db_cm + f_act * alpha_active_modal

        sp = accelerated_tmm_spectrum(
            wl_grid,
            roundtrip_length_um=l_rt_um,
            neff_at_ref=neff_rt_0,
            ng_at_ref=ng_0_rt,
            lambda_ref_nm=wl_nm,
            alpha_loss_db_per_cm=alpha_rt_db_cm,
            kappa_power_thru=kp_thru,
            kappa_power_drop=kp_drop,
            delta_neff_tuning=complex(dneff_rt_re, 0.0),
        )
        thru_t = np.asarray(sp["thru_T"], dtype=float)
        drop_t = np.asarray(sp["drop_T"], dtype=float)
        thru_db = 10.0 * np.log10(np.maximum(thru_t, 1e-12))
        drop_db = 10.0 * np.log10(np.maximum(drop_t, 1e-7)) if is_4port else np.full_like(thru_t, -60.0)

        lam_res_k = _exact_res_wl(dneff_rt_re)
        delta_wl_pm = (lam_res_k - lam_res_0) * 1e3
        delta_f_ghz = -(C0_M_PER_S / ((wl_m ** 2))) * (delta_wl_pm * 1e-12) * 1e-9

        # Local ER & Q around tracked resonance (avoid mixing adjacent FSRs)
        local_mask = np.abs(wl_grid - lam_res_k) <= 0.45 * fsr_nm
        loc_thru = thru_t[local_mask] if np.any(local_mask) else thru_t
        er_db_k = -10.0 * math.log10(max(float(np.min(loc_thru)) / max(float(np.max(thru_t)), 1e-12), 1e-12))

        alpha_pow_m = alpha_rt_db_cm * 100.0 * (math.log(10.0) / 10.0)
        q_int_k = (2.0 * math.pi * ng_0_rt) / max(alpha_pow_m * wl_m, 1e-18)
        q_ext_thru = float(pm["Q_ext_thru"])
        q_ext_drop = float(pm["Q_ext_drop"]) if pm.get("Q_ext_drop") else float("inf")
        inv_ql = (1.0 / max(q_int_k, 1.0)) + (1.0 / max(q_ext_thru, 1.0)) + (0.0 if not math.isfinite(q_ext_drop) else 1.0 / q_ext_drop)
        q_loaded_k = 1.0 / inv_ql

        lbl = f"{b_val:+g} {unit_str}" if (b_val < 0 or min(vals) < 0) else f"{b_val:g} {unit_str}"
        state_records.append({
            "index": idx_s,
            "bias_value": round(float(b_val), 5),
            "bias_unit": unit_str,
            "label": lbl,
            "power_mw": float(aux.get("power_mw", 0.0)),
            "delta_t_core_k": float(aux.get("delta_t_core_k", 0.0)),
            "e_core_v_per_um": float(aux.get("e_core_v_per_um", 0.0)),
            "neff_active_real": round(float(fem_r["neff_real"]), 7),
            "neff_active_imag": float(f"{float(fem_r['neff_imag']):.4e}"),
            "delta_neff_active": float(f"{dneff_active_re:+.4e}"),
            "delta_neff_active_real": float(f"{dneff_active_re:+.4e}"),
            "delta_neff_roundtrip_real": float(f"{dneff_rt_re:+.4e}"),
            "alpha_active_db_per_cm": round(alpha_active_modal, 4),
            "alpha_active_only_db_per_cm": round(alpha_active_modal, 4),
            "alpha_roundtrip_db_per_cm": round(alpha_rt_db_cm, 4),
            "alpha_total_roundtrip_db_per_cm": round(alpha_rt_db_cm, 4),
            "resonance_wl_nm": round(lam_res_k, 5),
            "delta_wl_pm": round(delta_wl_pm, 3),
            "resonance_shift_pm": round(delta_wl_pm, 3),
            "delta_freq_ghz": round(delta_f_ghz, 3),
            "ER_thru_dB": round(er_db_k, 2),
            "Q_int": round(q_int_k, 0),
            "Q_loaded": round(q_loaded_k, 0),
            "mode_overlap": round(float(fem_r["mode_overlap"]), 5),
            **aux,
        })
        spectra_curves.append({
            "bias_value": round(float(b_val), 5),
            "label": lbl,
            "thru_dB": np.round(thru_db, 3).tolist(),
            "drop_dB": np.round(drop_db, 3).tolist(),
            "thru_phase_rad": np.round(np.asarray(sp["thru_phase_rad"], dtype=float), 4).tolist(),
        })

    # 5. Extract Device Tuning Efficiency KPIs
    b_arr = np.array([r["bias_value"] for r in state_records], dtype=float)
    dwl_arr = np.array([r["delta_wl_pm"] for r in state_records], dtype=float)
    df_arr = np.array([r["delta_freq_ghz"] for r in state_records], dtype=float)
    dneff_act_arr = np.array([r["delta_neff_active_real"] for r in state_records], dtype=float)
    p_arr = np.array([r["power_mw"] for r in state_records], dtype=float)

    fsr_pm = fsr_nm * 1e3
    pi_shift_pm = 0.5 * fsr_pm  # pi roundtrip phase shift = half FSR

    efficiency: dict = {
        "mechanism": mech,
        "mechanism_name_en": TUNING_MECHANISM_CATALOG[mech]["name_en"],
        "mechanism_name_zh": TUNING_MECHANISM_CATALOG[mech]["name_zh"],
        "drive_type": drv,
        "drive_unit": unit_str,
        "drive_label": drive_label,
        "fsr_pm": round(fsr_pm, 2),
        "fsr_ghz": round(fsr_ghz, 3),
    }

    if mech == "thermo_optic":
        if drv == "delta_temp_k":
            slope_pm_k = float(np.polyfit(b_arr, dwl_arr, 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
            slope_ghz_k = float(np.polyfit(b_arr, df_arr, 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
            dt_pi = pi_shift_pm / max(abs(slope_pm_k), 1e-9)
            efficiency.update({
                "primary_slope": round(slope_pm_k, 3),
                "primary_slope_unit": "pm/K",
                "freq_slope": round(slope_ghz_k, 3),
                "freq_slope_unit": "GHz/K",
                "delta_T_pi_k": round(dt_pi, 2),
                "pi_figure": round(dt_pi, 2),
                "pi_figure_label": "ΔT_π",
                "pi_figure_unit": "K",
                "bandwidth_3db": None,
                "bandwidth_unit": "kHz",
            })
        else:
            slope_pm_mw = float(np.polyfit(p_arr, dwl_arr, 1)[0]) if len(p_arr) >= 2 and np.ptp(p_arr) > 1e-9 else 0.0
            slope_ghz_mw = float(np.polyfit(p_arr, df_arr, 1)[0]) if len(p_arr) >= 2 and np.ptp(p_arr) > 1e-9 else 0.0
            p_pi_mw = pi_shift_pm / max(abs(slope_pm_mw), 1e-9)
            r_h = float(geometry_physics.get("heater_resistance_ohm", 250.0))
            i_pi_ma = math.sqrt(max(p_pi_mw * 1e-3, 0.0) / max(r_h, 1e-3)) * 1e3
            v_pi_v = (i_pi_ma * 1e-3) * r_h
            efficiency.update({
                "primary_slope": round(slope_pm_mw, 3),
                "primary_slope_unit": "pm/mW",
                "freq_slope": round(slope_ghz_mw, 3),
                "freq_slope_unit": "GHz/mW",
                "P_pi_mw": round(p_pi_mw, 3),
                "P_2pi_mw": round(2.0 * p_pi_mw, 3),
                "I_pi_ma": round(i_pi_ma, 3),
                "V_pi_v": round(v_pi_v, 3),
                "pi_figure": round(p_pi_mw, 3),
                "pi_figure_label": "P_π",
                "pi_figure_unit": "mW",
                "bandwidth_3db": geometry_physics.get("f_3db_to_khz"),
                "bandwidth_unit": "kHz",
            })

    elif mech == "pockels_eo":
        slope_pm = float(np.polyfit(b_arr, dwl_arr, 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
        slope_ghz = float(np.polyfit(b_arr, df_arr, 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
        slope_dneff = float(np.polyfit(b_arr, dneff_act_arr, 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
        if drv == "e_field_v_per_um":
            e_pi = pi_shift_pm / max(abs(slope_pm), 1e-9)
            efficiency.update({
                "primary_slope": round(slope_pm, 4),
                "primary_slope_unit": "pm/(V/um)",
                "freq_slope": round(slope_ghz, 4),
                "freq_slope_unit": "GHz/(V/um)",
                "E_pi_v_per_um": round(e_pi, 3),
                "pi_figure": round(e_pi, 3),
                "pi_figure_label": "E_π",
                "pi_figure_unit": "V/um",
                "bandwidth_3db": geometry_physics.get("f_photon_ghz"),
                "bandwidth_unit": "GHz",
            })
        else:
            # V_pi * L [V*cm]: phase shift Delta_phi = (2*pi/lambda) * |d(neff_active)/dV| * V * L = pi
            vpi_l_v_cm = ((wl_m / (2.0 * max(abs(slope_dneff), 1e-15))) * 100.0) if abs(slope_dneff) > 1e-14 else None
            v_pi_ring = pi_shift_pm / max(abs(slope_pm), 1e-9)
            efficiency.update({
                "primary_slope": round(slope_pm, 4),
                "primary_slope_unit": "pm/V",
                "freq_slope": round(slope_ghz, 4),
                "freq_slope_unit": "GHz/V",
                "V_pi_L_v_cm": round(vpi_l_v_cm, 3) if vpi_l_v_cm is not None else None,
                "V_pi_ring_v": round(v_pi_ring, 2),
                "pi_figure": round(vpi_l_v_cm, 3) if vpi_l_v_cm is not None else None,
                "pi_figure_label": "V_π·L",
                "pi_figure_unit": "V·cm",
                "bandwidth_3db": geometry_physics.get("f_3db_eo_ghz"),
                "bandwidth_unit": "GHz",
            })

    elif mech == "si_pn_depletion":
        # Re-evaluate optical photon-lifetime cutoff using actual 2D-FEM doped PN loaded Q
        q_loaded_0v = float(state_records[0]["Q_loaded"]) if state_records else q_loaded_passive
        f_opt_pn_ghz = (C0_M_PER_S / (wl_m * max(q_loaded_0v, 1.0))) * 1e-9
        f_rc_pn_ghz = float(geometry_physics.get("f_rc_ghz", 30.0))
        f_eo_pn_3db = 1.0 / math.hypot(1.0 / max(f_rc_pn_ghz, 1e-6), 1.0 / max(f_opt_pn_ghz, 1e-6))
        geometry_physics["f_photon_ghz"] = round(f_opt_pn_ghz, 2)
        geometry_physics["f_3db_eo_ghz"] = round(f_eo_pn_3db, 2)

        # Note: reverse bias has b_arr <= 0; report magnitude |d(lambda)/dV| in pm/V and V_pi*L [V*cm]
        slope_pm_v = float(np.polyfit(np.abs(b_arr), np.abs(dwl_arr), 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
        slope_ghz_v = float(np.polyfit(np.abs(b_arr), np.abs(df_arr), 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
        slope_dneff_v = float(np.polyfit(np.abs(b_arr), np.abs(dneff_act_arr), 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
        vpi_l_v_cm = ((wl_m / (2.0 * max(abs(slope_dneff_v), 1e-15))) * 100.0) if abs(slope_dneff_v) > 1e-14 else None
        fca_0v = float(ref_fem["alpha_modal_db_per_cm"])
        fca_min = min(float(r["alpha_active_db_per_cm"]) for r in state_records)
        efficiency.update({
            "primary_slope": round(slope_pm_v, 3),
            "primary_slope_unit": "pm/V",
            "freq_slope": round(slope_ghz_v, 3),
            "freq_slope_unit": "GHz/V",
            "V_pi_L_v_cm": round(vpi_l_v_cm, 3) if vpi_l_v_cm is not None else None,
            "fca_0v_db_per_cm": round(fca_0v, 3),
            "fca_min_rev_db_per_cm": round(fca_min, 3),
            "fca_relief_db_per_cm": round(fca_0v - fca_min, 3),
            "pi_figure": round(vpi_l_v_cm, 3) if vpi_l_v_cm is not None else None,
            "pi_figure_label": "V_π·L",
            "pi_figure_unit": "V·cm",
            "bandwidth_3db": geometry_physics.get("f_3db_eo_ghz"),
            "bandwidth_unit": "GHz",
        })

    else:  # si_pin_injection
        q_loaded_0v = float(state_records[0]["Q_loaded"]) if state_records else q_loaded_passive
        f_opt_pin_ghz = (C0_M_PER_S / (wl_m * max(q_loaded_0v, 1.0))) * 1e-9
        f_car_ghz = float(geometry_physics.get("f_carrier_ghz", 0.16))
        f_eo_pin_3db = 1.0 / math.hypot(1.0 / max(f_car_ghz, 1e-6), 1.0 / max(f_opt_pin_ghz, 1e-6))
        geometry_physics["f_photon_ghz"] = round(f_opt_pin_ghz, 3)
        geometry_physics["f_3db_eo_ghz"] = round(f_eo_pin_3db, 3)

        slope_pm_ma = float(np.polyfit(b_arr, dwl_arr, 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
        slope_ghz_ma = float(np.polyfit(b_arr, df_arr, 1)[0]) if len(b_arr) >= 2 and np.ptp(b_arr) > 0 else 0.0
        i_pi_ma = pi_shift_pm / max(abs(slope_pm_ma), 1e-9)
        fca_max = max(float(r["alpha_active_db_per_cm"]) for r in state_records)
        efficiency.update({
            "primary_slope": round(slope_pm_ma, 2),
            "primary_slope_unit": "pm/mA",
            "freq_slope": round(slope_ghz_ma, 2),
            "freq_slope_unit": "GHz/mA",
            "I_pi_ma": round(i_pi_ma, 4),
            "fca_max_db_per_cm": round(fca_max, 2),
            "pi_figure": round(i_pi_ma, 4),
            "pi_figure_label": "I_π",
            "pi_figure_unit": "mA",
            "bandwidth_3db": geometry_physics.get("f_3db_eo_ghz"),
            "bandwidth_unit": "GHz",
        })

    efficiency["wavelength_slope"] = efficiency["primary_slope"]
    efficiency["wavelength_slope_unit"] = efficiency["primary_slope_unit"]

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(work_cfg),
        "passive_metrics": {
            "roundtrip_length_um": round(l_rt_um, 3),
            "neff_0_roundtrip": round(neff_rt_0, 6),
            "ng_0_roundtrip": round(ng_0_rt, 6),
            "kappa_power_thru": round(kp_thru, 6),
            "kappa_power_drop": round(kp_drop, 6),
            "alpha_passive_db_per_cm": round(alpha_passive_db_cm, 4),
            "FSR_nm": round(fsr_nm, 4),
            "FSR_GHz": round(fsr_ghz, 3),
            "Q_loaded_0": round(q_loaded_passive, 0),
            "tracked_resonance_0_nm": round(lam_res_0, 5),
            "is_4port": is_4port,
        },
        "geometry_physics": geometry_physics,
        "efficiency": efficiency,
        "states": state_records,
        "spectra": {
            "wavelength_nm": np.round(wl_grid, 5).tolist(),
            "curves": spectra_curves,
        },
    }


# ==============================================================================
# 4. Ansys Lumerical INTERCONNECT Compact Model Bundle Exporter
# ==============================================================================

def export_lumerical_interconnect_bundle(
    base_cfg: MRRDesignConfig,
    tuning_result: dict,
    output_dir: str | Path,
    *,
    make_zip: bool = True,
) -> dict:
    """Export 2D-FEM passive S-parameters + active `n_eff` table + `.lsf` circuit for Lumerical INTERCONNECT.

    Generates:
      1. `microring_passive_sparams.dat` : Exact INTERCONNECT `Optical N-Port S-Parameter` (`SPAR`)
                                         file with strictly ascending frequency `[Hz]`, `(abs_S, unwrap_phase_rad)`,
                                         reciprocal port symmetry, and verified passivity (`sum |S|^2 <= 1`).
      2. `microring_active_neff_table.txt`: Exact INTERCONNECT `Optical Modulator Measured` (`OMM`)
                                         table `(bias_SI, Re(delta_neff), Im(neff_active))` sorted in
                                         strictly ascending bias order with non-negative `Im(neff) >= 0`.
      3. `build_microring_interconnect.lsf`: Turnkey INTERCONNECT circuit builder placing `Waveguide Coupler`,
                                         passive `Straight Waveguide`, active `Optical Modulator Measured`,
                                         `DC Source`, `SPAR` comparator, and `Optical Network Analyzer`.
      4. `run_interconnect_lumapi.py`  : Automated Python `lumapi.INTERCONNECT()` multi-bias runner.
      5. `microring_active_neff_loss.csv` & `microring_multibias_spectra.csv`: Unit-labeled reference CSVs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pm = tuning_result["passive_metrics"]
    gp = tuning_result["geometry_physics"]
    eff = tuning_result["efficiency"]
    states = tuning_result["states"]
    wl_nm_arr = np.asarray(tuning_result["spectra"]["wavelength_nm"], dtype=float)

    l_rt_um = float(pm["roundtrip_length_um"])
    f_act = float(gp["active_fraction"])
    l_act_um = float(gp["active_length_um"])
    l_pas_um = max(l_rt_um - l_act_um, 0.0)
    neff_0 = float(pm["neff_0_roundtrip"])
    ng_0 = float(pm["ng_0_roundtrip"])
    kp_thru = float(pm["kappa_power_thru"])
    kp_drop = float(pm["kappa_power_drop"])
    alpha_pas_db_cm = float(pm["alpha_passive_db_per_cm"])
    is_4port = bool(pm["is_4port"])

    pol_label = "TM" if str(base_cfg.polarization).upper().startswith("TM") else "TE"
    mode_id = 2 if pol_label == "TM" else 1

    # 1. Compute complex zero-bias S-parameters on strictly ascending frequency grid [Hz]
    # Include 0-bias active modal absorption so SPAR_0 matches active circuit at bias = 0
    zero_idx = int(np.argmin(np.abs([float(s["bias_value"]) for s in states])))
    alpha_0_tot_db_cm = float(states[zero_idx]["alpha_roundtrip_db_per_cm"])
    dneff_0_rt = float(states[zero_idx]["delta_neff_roundtrip_real"])

    wl_asc_nm = np.sort(wl_nm_arr)
    wl_desc_nm = wl_asc_nm[::-1]  # strictly ascending frequency f = c / lambda!
    freq_asc_hz = C0_M_PER_S / (wl_desc_nm * 1e-9)

    wl_m_vec = wl_desc_nm * 1e-9
    wl0_m = float(base_cfg.wavelength_nm) * 1e-9
    dn_dl_m = -(ng_0 - neff_0) / wl0_m
    neff_disp = neff_0 + dneff_0_rt + dn_dl_m * (wl_m_vec - wl0_m)

    alpha_pow_m = alpha_0_tot_db_cm * 100.0 * (math.log(10.0) / 10.0)
    a_rt = math.exp(-0.5 * alpha_pow_m * (l_rt_um * 1e-6))
    t1 = math.sqrt(max(1.0 - kp_thru, 0.0))
    k1 = math.sqrt(max(kp_thru, 0.0))
    t2 = math.sqrt(max(1.0 - kp_drop, 0.0)) if is_4port else 1.0
    k2 = math.sqrt(max(kp_drop, 0.0)) if is_4port else 0.0

    phi_rt = (2.0 * math.pi / wl_m_vec) * neff_disp * (l_rt_um * 1e-6)
    exp_iphi = np.exp(1j * phi_rt)
    denom = 1.0 - t1 * t2 * a_rt * exp_iphi

    s21 = (t1 - t2 * a_rt * exp_iphi) / denom
    s31 = (-k1 * k2 * math.sqrt(a_rt) * np.exp(0.5j * phi_rt)) / denom if is_4port else np.zeros_like(s21)

    # Enforce strict numerical passivity |S21|^2 + |S31|^2 <= 1.0 - 1e-12
    p_tot = (np.abs(s21) ** 2) + (np.abs(s31) ** 2)
    over_mask = p_tot > (1.0 - 1e-12)
    if np.any(over_mask):
        scale = np.sqrt((1.0 - 1e-12) / p_tot[over_mask])
        s21[over_mask] *= scale
        s31[over_mask] *= scale

    abs_s21 = np.abs(s21)
    ang_s21 = np.unwrap(np.angle(s21))
    abs_s31 = np.abs(s31)
    ang_s31 = np.unwrap(np.angle(s31))

    spar_path = out_dir / "microring_passive_sparams.dat"
    spar_lines: list[str] = []

    def _append_spar_block(dst_p: int, src_p: int, mag_v: np.ndarray, ph_v: np.ndarray) -> None:
        spar_lines.append(f'("port {dst_p}","{pol_label}",{mode_id},"port {src_p}",{mode_id},"transmission")')
        spar_lines.append(f"({len(freq_asc_hz)}, 3)")
        for f_hz, m_val, p_val in zip(freq_asc_hz, mag_v, ph_v):
            spar_lines.append(f"{f_hz:.8e} {m_val:.8e} {p_val:.8e}")

    _append_spar_block(2, 1, abs_s21, ang_s21)
    _append_spar_block(1, 2, abs_s21, ang_s21)
    if is_4port:
        _append_spar_block(3, 1, abs_s31, ang_s31)
        _append_spar_block(1, 3, abs_s31, ang_s31)
        _append_spar_block(4, 2, abs_s31, ang_s31)
        _append_spar_block(2, 4, abs_s31, ang_s31)
        _append_spar_block(4, 3, abs_s21, ang_s21)
        _append_spar_block(3, 4, abs_s21, ang_s21)

    spar_path.write_text("\n".join(spar_lines) + "\n", encoding="utf-8")

    # 2. Write INTERCONNECT `Optical Modulator Measured` table (strictly ascending SI bias!)
    drv = str(eff["drive_type"])
    bias_si_scale = 1e-3 if drv in ("current_ma", "power_mw") else 1.0  # mA -> A, mW -> W, V -> V
    bias_si_unit = {"current_ma": "A", "power_mw": "W", "voltage_v": "V", "delta_temp_k": "K", "e_field_v_per_um": "V_per_um"}.get(drv, "V")

    sorted_states = sorted(states, key=lambda r: float(r["bias_value"]))
    # Ensure strictly ascending unique bias entries for INTERCONNECT table parser
    omm_path = out_dir / "microring_active_neff_table.txt"
    omm_lines: list[str] = []
    seen_b: list[float] = []
    for st in sorted_states:
        b_si = float(st["bias_value"]) * bias_si_scale
        if seen_b and abs(b_si - seen_b[-1]) < 1e-12:
            continue
        seen_b.append(b_si)
        re_dn = float(st["delta_neff_active_real"])
        # Non-negative active modal extinction k_eff >= 0 guarantees strict passivity at every bias
        alpha_act_m = max(float(st["alpha_active_db_per_cm"]), 0.0) * 100.0 * (math.log(10.0) / 10.0)
        im_k = (alpha_act_m * wl0_m) / (4.0 * math.pi)
        omm_lines.append(f"{b_si:.8e} {re_dn:+.8e} {im_k:.8e}")
    omm_path.write_text("\n".join(omm_lines) + "\n", encoding="utf-8")

    # 3. Write unit-labeled summary CSV & multi-bias spectra CSV
    neff_csv_path = out_dir / "microring_active_neff_loss.csv"
    csv_hdr = (
        f"bias_{eff['drive_unit']},bias_SI_{bias_si_unit},heater_power_mW,core_temp_rise_K,"
        "core_E_field_V_per_um,neff_active_real,delta_neff_active_real,alpha_active_dB_per_cm,"
        "resonance_wl_nm,delta_wl_pm,ER_thru_dB,Q_loaded"
    )
    csv_rows = [csv_hdr]
    for st in sorted_states:
        csv_rows.append(
            f"{st['bias_value']},{float(st['bias_value'])*bias_si_scale:.6e},{st['power_mw']},"
            f"{st['delta_t_core_k']},{st['e_core_v_per_um']},{st['neff_active_real']},"
            f"{st['delta_neff_active_real']:.6e},{st['alpha_active_db_per_cm']},"
            f"{st['resonance_wl_nm']},{st['delta_wl_pm']},{st['ER_thru_dB']},{int(st['Q_loaded'])}"
        )
    neff_csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    spec_csv_path = out_dir / "microring_multibias_spectra.csv"
    curves = tuning_result["spectra"]["curves"]
    freq_thz = (C0_M_PER_S / (wl_nm_arr * 1e-9)) * 1e-12
    col_names = ["wavelength_nm", "frequency_THz"]
    cols = [wl_nm_arr, freq_thz]
    for c in curves:
        tag = str(c["label"]).replace(" ", "_").replace("/", "p").replace("+", "pos").replace("-", "neg")
        col_names.append(f"thru_dB_{tag}")
        cols.append(np.asarray(c["thru_dB"], dtype=float))
        if is_4port:
            col_names.append(f"drop_dB_{tag}")
            cols.append(np.asarray(c["drop_dB"], dtype=float))
    np.savetxt(spec_csv_path, np.column_stack(cols), delimiter=",", header=",".join(col_names), comments="", fmt="%.6f")

    # 4. Write turnkey Lumerical INTERCONNECT `.lsf` schematic builder
    f_center_hz = C0_M_PER_S / wl0_m
    span_nm_val = float(wl_nm_arr[-1] - wl_nm_arr[0])
    f_span_hz = (C0_M_PER_S / ((wl_nm_arr[0] * 1e-9))) - (C0_M_PER_S / ((wl_nm_arr[-1] * 1e-9)))
    alpha_pas_db_m = alpha_pas_db_cm * 100.0
    default_bias_si = float(sorted_states[-1]["bias_value"]) * bias_si_scale

    lsf_path = out_dir / "build_microring_interconnect.lsf"
    lsf_code = f"""# ==============================================================================
# MicroRing Studio v3.4.0 -> Ansys Lumerical INTERCONNECT Turnkey Circuit Script
# Solver Source : MicroRing 2D Conformal Vector FEM + Spatial CMT (Zero COMSOL)
# Core Material : {base_cfg.core_material} ({base_cfg.polarization}) | Mechanism: {eff['mechanism']}
# ==============================================================================
switchtolayout;
deleteall;

# --- 1. Centralized Device & 2D-FEM Extracted Parameters (with SI units) ---
f0_hz          = {f_center_hz:.8e};      # Center optical frequency [Hz] ({base_cfg.wavelength_nm:.2f} nm)
f_span_hz      = {f_span_hz:.8e};      # ONA sweep frequency span [Hz] ({span_nm_val:.3f} nm)
L_rt_m         = {l_rt_um * 1e-6:.8e};      # Total ring round-trip length [m] ({l_rt_um:.3f} um)
L_active_m     = {l_act_um * 1e-6:.8e};      # Active modulated arc length [m] ({l_act_um:.3f} um)
L_passive_m    = {max(l_pas_um, 0.5) * 1e-6:.8e};      # Passive unmodulated arc length [m] ({l_pas_um:.3f} um)
neff_0         = {neff_0:.7f};              # 2D-FEM round-trip effective index at f0 [-]
ng_0           = {ng_0:.7f};              # 2D-FEM round-trip group index at f0 [-]
loss_pas_dB_m  = {alpha_pas_db_m:.4f};              # Passive waveguide propagation loss [dB/m]
kp_thru        = {kp_thru:.6f};              # Spatial-CMT Through power coupling kappa^2 [-]
kp_drop        = {kp_drop:.6f};              # Spatial-CMT Drop power coupling kappa^2 [-]
bias_init_SI   = {default_bias_si:.6e};      # Initial DC bias [{bias_si_unit}]

spar_file      = pwd + "/microring_passive_sparams.dat";
neff_tbl_file  = pwd + "/microring_active_neff_table.txt";

# --- 2. Place Passive N-Port S-Parameter Reference Element (0-Bias Verification) ---
addelement("Optical N-Port S-Parameter");
set("name", "SPAR_0Bias_Ref");
set("x position", 250);
set("y position", 420);
set("load from file", 1);
set("s parameters filename", spar_file);

# --- 3. Build Active Modulated Microring Subcircuit ---
addelement("Waveguide Coupler");
set("name", "C_thru");
set("x position", 250);
set("y position", 180);
set("coupling coefficient 1", kp_thru);

addelement("Straight Waveguide");
set("name", "WG_ring_passive");
set("x position", 140);
set("y position", 60);
set("frequency", f0_hz);
set("length", L_passive_m);
set("effective index 1", neff_0);
set("group index 1", ng_0);
set("loss 1", loss_pas_dB_m);

addelement("Optical Modulator Measured");
set("name", "MOD_ring_active");
set("x position", 360);
set("y position", 60);
set("frequency", f0_hz);
set("length", L_active_m);
set("effective index", neff_0);
set("group index", ng_0);
set("loss", loss_pas_dB_m);
set("measurement type", "effective index");
set("load from file", 1);
set("filename", neff_tbl_file);

addelement("DC Source");
set("name", "BIAS_1");
set("x position", 360);
set("y position", -60);
set("amplitude", bias_init_SI);

addelement("Optical Network Analyzer");
set("name", "ONA_1");
set("x position", -40);
set("y position", 240);
set("input parameter", "center and span");
set("center frequency", f0_hz);
set("frequency span", f_span_hz);
set("number of points", {len(wl_nm_arr)});
set("number of input ports", {"3" if is_4port else "2"});

# --- 4. Wire Ring Feedback Loop, DC Drive & ONA Probes ---
connect("BIAS_1", "output", "MOD_ring_active", "modulation");
connect("ONA_1", "output", "C_thru", "port 1");
connect("C_thru", "port 2", "ONA_1", "input 1");
"""
    if is_4port:
        lsf_code += """
addelement("Waveguide Coupler");
set("name", "C_drop");
set("x position", 250);
set("y position", -40);
set("coupling coefficient 1", kp_drop);

connect("C_thru", "port 4", "MOD_ring_active", "port 1");
connect("MOD_ring_active", "port 2", "C_drop", "port 1");
connect("C_drop", "port 3", "WG_ring_passive", "port 1");
connect("WG_ring_passive", "port 2", "C_thru", "port 3");
connect("C_drop", "port 2", "ONA_1", "input 2");
connect("SPAR_0Bias_Ref", "port 2", "ONA_1", "input 3");
"""
    else:
        lsf_code += """
connect("C_thru", "port 4", "MOD_ring_active", "port 1");
connect("MOD_ring_active", "port 2", "WG_ring_passive", "port 1");
connect("WG_ring_passive", "port 2", "C_thru", "port 3");
connect("SPAR_0Bias_Ref", "port 2", "ONA_1", "input 2");
"""
    lsf_code += """
?"MicroRing INTERCONNECT circuit built successfully. Running ONA simulation...";
run;
"""
    lsf_path.write_text(lsf_code, encoding="utf-8")

    # 5. Write automated Python `lumapi.INTERCONNECT()` multi-bias sweep runner
    py_path = out_dir / "run_interconnect_lumapi.py"
    bias_si_list = [round(float(st["bias_value"]) * bias_si_scale, 8) for st in sorted_states]
    py_code = f'''"""Automated Lumerical INTERCONNECT Multi-Bias Runner for MicroRing Compact Model.

Requires Ansys Lumerical INTERCONNECT (`lumapi`) on the workstation running INTERCONNECT.
All 2D-FEM effective-index perturbations and passive S-parameters in this folder were
pre-computed license-free by MicroRing Studio v3.4.0.
"""
from pathlib import Path
import numpy as np

BUNDLE_DIR = Path(__file__).resolve().parent
BIAS_VALUES_SI = {bias_si_list!r}  # unit: {bias_si_unit}


def run_in_interconnect() -> None:
    import lumapi  # Provided by Ansys Lumerical Python API

    with lumapi.INTERCONNECT(hide=False) as ic:
        ic.cd(str(BUNDLE_DIR))
        ic.feval("build_microring_interconnect.lsf")

        traces = []
        wl_nm = None
        for b_val in BIAS_VALUES_SI:
            ic.switchtolayout()
            ic.setnamed("BIAS_1", "amplitude", float(b_val))
            ic.run()
            res = ic.getresult("ONA_1", "input 1/mode 1/gain")
            if wl_nm is None:
                wl_nm = np.asarray(res["wavelength"]).squeeze() * 1e9
            gain_db = np.asarray(res["TE gain (dB)" if "TE gain (dB)" in res else list(res.keys())[0]]).squeeze()
            traces.append(gain_db)
            print(f"Completed INTERCONNECT bias = {{b_val:+.4g}} {bias_si_unit}: min Through = {{np.min(gain_db):.2f}} dB")

        out_cmp = BUNDLE_DIR / "interconnect_ona_results.csv"
        cols = [wl_nm] + traces
        hdr = "wavelength_nm," + ",".join(f"IC_thru_dB_{{b:g}}_{bias_si_unit}" for b in BIAS_VALUES_SI)
        np.savetxt(out_cmp, np.column_stack(cols), delimiter=",", header=hdr, comments="", fmt="%.6f")
        print(f"Saved Lumerical INTERCONNECT ONA sweep to {{out_cmp}}")


if __name__ == "__main__":
    run_in_interconnect()
'''
    py_path.write_text(py_code, encoding="utf-8")

    # 6. Optional single-click `.zip` bundle for UI download
    zip_path: Path | None = None
    exported_files = [spar_path, omm_path, lsf_path, py_path, neff_csv_path, spec_csv_path]
    if make_zip:
        safe_mat = str(base_cfg.core_material).replace("-", "")
        zip_path = out_dir.parent / f"MicroRing_INTERCONNECT_{safe_mat}_{eff['mechanism']}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fp in exported_files:
                zf.write(fp, arcname=f"microring_interconnect_{safe_mat}/{fp.name}")

    return {
        "bundle_dir": str(out_dir),
        "zip_path": str(zip_path) if zip_path else None,
        "zip_name": zip_path.name if zip_path else None,
        "spar_dat": str(spar_path),
        "active_neff_table": str(omm_path),
        "lsf_script": str(lsf_path),
        "lumapi_runner": str(py_path),
        "neff_loss_csv": str(neff_csv_path),
        "spectra_csv": str(spec_csv_path),
        "num_freq_points": len(freq_asc_hz),
        "num_bias_states": len(seen_b),
        "passivity_verified": bool(np.max(p_tot) <= 1.0 + 1e-9),
    }
