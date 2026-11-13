"""First-Principles 2D Semiconductor PN/PIN Doping TCAD & Steady/Transient Thermal FEM Suite (`microring.multiphysics_tcad`).

Replaces commercial `Lumerical CHARGE` and `Lumerical HEAT` / `COMSOL` workflows with 100%
license-free pure-Python 2D Finite-Element Method (`scikit-fem` + `scipy` + `gdstk`):

1. **2D Silicon PN / PIN Doping TCAD (`simulate_pn_doping_tcad`, `export_pn_doping_gds`)**:
   - Supports `"lateral_pn"` (standard self-aligned or offset lateral rib junction),
     `"l_shaped_pn"` (vertical-lateral L-junction maximizing optical mode overlap for low `V_pi*L`),
     `"u_shaped_pn"` (wrap-around ridge junction), and `"pin_diode"` (`P++ / i / N++` carrier-injection).
   - Solves the **exact 2D Nonlinear Poisson-Boltzmann PDE** via globally convergent damped
     Newton-Raphson on `skfem.MeshTri`:
       `-div(eps(x,y) grad psi) = q * (p(psi, phi_Fp) - n(psi, phi_Fn) + N_D+(x,y) - N_A-(x,y))`
     across Silicon rib + slab + `SiO2` cladding at each reverse bias `V`, extracting 2D electrostatic
     potential `psi(x,y,V)`, 2D electron/hole distributions `n(x,y,V), p(x,y,V)`, differential junction
     capacitance `C_j(V) = dQ/dV [fF/um]`, Caughey-Thomas 2D doped-slab access resistance `R_s [Ohm*cm]`,
     electrical `RC` cutoff `f_3dB_RC(V) [GHz]`, and coupled 2D Vector Maxwell Soref-Bennett
     `V_pi*L(V) [V*cm]` & free-carrier absorption `alpha_FCA(V) [dB/cm]`.
   - Exports multi-implant Cleanroom `.GDS` + `.lyp` (`WG_CORE`, `WG_SLAB`, `P_CORE`, `N_CORE`,
     `P++_CONTACT`, `N++_CONTACT`, `VIA`, `RF_METAL`).

2. **2D Steady-State & Crank-Nicolson Transient Thermal FEM (`simulate_thermal_phase_shifter`, `export_thermal_shifter_gds`)**:
   - Supports `"top_heater"` (standard cladded `TiN`/`Pt`/`W`/`NiCr` heater),
     `"deep_trench"` (deep oxide/substrate air-isolation trenches flanking the waveguide),
     `"undercut_suspended"` (`XeF2` isotropic silicon undercut cavity beneath `BOX` for sub-3 mW `P_pi`),
     and `"doped_si_rib"` (direct in-rib `N++/N/N++` doped-silicon microheater for sub-5 us switching).
   - Assembles exact 2D FEM conductivity stiffness `K_th` and heat-capacity mass `M_th` matrices:
     * Steady-state `-div(k_th grad T) = Q_Joule`: 2D temperature map `Delta_T(x,y)`, `R_th [K/mW]`,
       heater peak temperature at `P_pi`, and **lateral thermal crosstalk profile `Delta_T(x)`** across
       adjacent MZI arms / neighboring waveguides (`5–50 um`).
     * Implicit **Crank-Nicolson time-stepping** `(M + dt/2 K) T^{n+1} = (M - dt/2 K) T^n + dt b`:
       exact heating & cooling pulse waveform `T_core(t)`, 10%-90% `t_rise [us]` & 90%-10% `t_fall [us]`.
     * Exact **frequency-domain complex harmonic FEM** `(K + j*omega*M) T(omega) = b`: exact multi-pole
       electro-thermal transfer `|H_th(f)| [dB]` and `f_3dB_thermal [kHz]`.
   - Feeds spatial `Delta_T(x,y)` into 2D Vector Maxwell FEM for `P_pi [mW]`, `V_pi [V]`, `I_pi [mA]`,
     and `P_pi * tau_th [mW*us]` FOM, and exports Cleanroom Heater + Air-Trench `.GDS` + `.lyp`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
import zipfile

import gdstk
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import erfc
import shapely
from shapely.geometry import Polygon, box

from skfem import (
    Basis,
    BilinearForm,
    ElementTriP0,
    ElementTriP1,
    LinearForm,
    condense,
    solve,
)

from .active_tuning import (
    HEATER_METAL_CATALOG,
    KB_J_PER_K,
    NI_SI_300K_CM3,
    Q_E_C,
    THERMAL_CONDUCTIVITY_W_MK,
    VOLUMETRIC_HEAT_CAPACITY_J_M3K,
    _rectilinear_mesh_tri,
    _si_carrier_mobilities_cm2_vs,
    _snapped_1d_axis,
    soref_bennett_si_dn_dk,
)
from .eigenmode_solver import WaveguideCrossSectionSpec, solve_perturbed_neff_sweep_fem
from .gds_bridge import MICRORING_LAYERS, write_klayout_layer_props
from .materials import EPS0_F_PER_M, T_REF_K, get_material_fn


PN_JUNCTION_CATALOG: dict[str, dict] = {
    "lateral_pn": {
        "name_en": "Lateral Self-Aligned / Offset Silicon Rib PN Junction",
        "name_zh": "横向偏置硅脊波导 PN 结 (标准 CMOS 硅光耗尽调制)",
    },
    "l_shaped_pn": {
        "name_en": "L-Shaped Vertical-Lateral Hybrid PN Junction (High Overlap, Low V_π·L)",
        "name_zh": "L 型垂直+横向复合 PN 结 (高模场交叠 · 超低 V_π·L)",
    },
    "u_shaped_pn": {
        "name_en": "U-Shaped Wrap-Around Ridge PN Junction",
        "name_zh": "U 型包覆脊波导 PN 结 (双侧耗尽层交叠)",
    },
    "pin_diode": {
        "name_en": "Lateral Silicon P++ / Intrinsic / N++ (PIN) Carrier-Injection Diode",
        "name_zh": "硅基横向 P++ / i / N++ (PIN) 载流子二极管结构",
    },
}

THERMAL_STRUCTURE_CATALOG: dict[str, dict] = {
    "top_heater": {
        "name_en": "Top Cladded Metal Microheater (TiN / Pt / W / NiCr)",
        "name_zh": "顶层包层金属微加热器 (标准流片 TiN / Pt / W)",
    },
    "deep_trench": {
        "name_en": "Deep Air-Trench Isolated Microheater (Etched Cladding + BOX Trenches)",
        "name_zh": "深空气隔离槽微加热器 (两侧刻穿包层+BOX 抑制横向漏热与串扰)",
    },
    "undercut_suspended": {
        "name_en": "XeF₂ Isotropic Silicon-Undercut Suspended Microheater (Sub-3 mW P_π)",
        "name_zh": "XeF₂ 硅衬底掏空悬臂微加热器 (超低功耗 P_π < 3 mW)",
    },
    "doped_si_rib": {
        "name_en": "Direct In-Rib Doped-Silicon (N++ / N / N++) Microheater (Fast µs Switching)",
        "name_zh": "脊波导内嵌重掺杂硅自热移相器 (微秒级超快热光响应)",
    },
}


@dataclass
class PNDopingTCADConfig:
    """Configuration for 2D Nonlinear Poisson-Boltzmann Silicon PN/PIN Doping TCAD."""
    junction_type: str = "lateral_pn"       # "lateral_pn" | "l_shaped_pn" | "u_shaped_pn" | "pin_diode"
    wavelength_nm: float = 1550.0           # unit: nm (1310 nm O-band or 1550 nm C-band or 2000 nm)
    polarization: str = "TE0"               # "TE0" | "TM0"
    film_height_um: float = 0.22            # unit: um (220 nm SOI standard)
    slab_height_um: float = 0.09            # unit: um (90 nm partial etch rib slab)
    sidewall_deg: float = 86.0              # unit: deg
    waveguide_width_um: float = 0.45        # unit: um
    active_length_um: float = 1500.0        # unit: um (1.5 mm phase shifter)
    drive_scheme: str = "push_pull"         # "push_pull" | "single_arm"
    # Multi-stage implant concentrations & geometry
    na_core_cm3: float = 5.0e17             # unit: cm^-3 (core p-type boron implant)
    nd_core_cm3: float = 5.0e17             # unit: cm^-3 (core n-type phosphorus/arsenic implant)
    na_inter_cm3: float = 2.0e18            # unit: cm^-3 (intermediate p+ slab implant)
    nd_inter_cm3: float = 2.0e18            # unit: cm^-3 (intermediate n+ slab implant)
    n_heavy_contact_cm3: float = 1.0e20     # unit: cm^-3 (p++ / n++ ohmic silicide contact implant)
    pn_offset_nm: float = 25.0              # unit: nm (lateral junction offset from x = 0 toward +x)
    l_vertical_y_nm: float = 110.0          # unit: nm (horizontal cap junction height for l_shaped_pn)
    implant_straggle_nm: float = 18.0       # unit: nm (Gaussian lateral/vertical ion-implant straggle)
    inter_implant_offset_um: float = 0.35   # unit: um (edge of p+/n+ intermediate slab wing from core center)
    heavy_contact_offset_um: float = 0.90   # unit: um (edge of p++/n++ silicide contact from core center)
    bias_voltages_v: list[float] | None = None  # e.g. [0.0, -1.0, -2.0, -3.0, -4.0, -5.0]
    map_bias_v: float = -3.0                # unit: V (reverse voltage at which 2D carrier/potential map is exported)
    driver_impedance_ohm: float = 50.0      # unit: Ohm


@dataclass
class ThermalSimConfig:
    """Configuration for 2D Steady-State & Crank-Nicolson Transient Thermal FEM."""
    thermal_structure: str = "top_heater"   # "top_heater" | "deep_trench" | "undercut_suspended" | "doped_si_rib"
    core_material: str = "Si"               # any of 29 materials: "Si", "Si3N4", "LN_Xcut", "4H-SiC", "AlN", etc.
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"
    wavelength_nm: float = 1550.0           # unit: nm
    polarization: str = "TE0"
    film_height_um: float = 0.22            # unit: um
    slab_height_um: float = 0.0             # unit: um (auto-set to 0.09 um if doped_si_rib)
    sidewall_deg: float = 86.0              # unit: deg
    waveguide_width_um: float = 0.45        # unit: um
    active_length_um: float = 200.0         # unit: um (heater length along MZI arm or ring arc)
    box_thickness_um: float = 2.00          # unit: um
    top_clad_thickness_um: float = 1.80     # unit: um
    # Heater parameters
    heater_material: str = "TiN"            # "TiN" | "Pt" | "W" | "NiCr" | "doped_Si"
    heater_width_um: float = 1.80           # unit: um
    heater_thickness_um: float = 0.12       # unit: um
    heater_gap_um: float = 0.95             # unit: um (vertical oxide gap above core top)
    heater_offset_um: float = 0.0           # unit: um (lateral offset from core center)
    # Air-trench & undercut isolation geometry
    trench_edge_offset_um: float = 2.80     # unit: um (distance from core center to inner wall of DRIE isolation trench)
    trench_width_um: float = 6.00           # unit: um
    trench_depth_um: float = 42.0           # unit: um (deep Bosch DRIE trench cutting through top clad + BOX + Si handle)
    undercut_width_um: float = 16.0         # unit: um (width of XeF2 air cavity beneath BOX)
    undercut_depth_um: float = 8.0          # unit: um (height of XeF2 air cavity beneath BOX)
    neighbor_pitch_um: float = 20.0         # unit: um (distance to adjacent MZI arm / neighbor ring for crosstalk)
    # Optional user-supplied custom thermal / thermo-optic / heater resistivity overrides (None -> literature catalog)
    custom_dn_dt_per_k: float | None = None      # unit: 1/K (user's core thermo-optic coefficient dn/dT)
    custom_core_kappa_w_mk: float | None = None  # unit: W/(m*K) (user's core thermal conductivity)
    custom_heater_rho_ohm_m: float | None = None # unit: Ohm*m (user's heater bulk resistivity)


# ==============================================================================
# 1. 2D Nonlinear Poisson-Boltzmann Silicon PN / PIN Doping TCAD Solver
# ==============================================================================

@BilinearForm
def _poisson_stiff_form(u, v, w):
    return w["eps_si_ox"] * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])


@LinearForm
def _node_lumped_vol_form(v, _w):
    return 1.0 * v


def _build_2d_doping_profile_cm3(
    x_um: np.ndarray,
    y_um: np.ndarray,
    cfg: PNDopingTCADConfig,
    in_si_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate 2D acceptor `N_A(x,y)` and donor `N_D(x,y)` [cm^-3] with Gaussian implant straggle."""
    sig_um = max(float(cfg.implant_straggle_nm) * 1e-3, 0.004)
    s2 = math.sqrt(2.0) * sig_um
    xj = float(cfg.pn_offset_nm) * 1e-3
    yj = float(cfg.l_vertical_y_nm) * 1e-3

    na = np.zeros_like(x_um, dtype=float)
    nd = np.zeros_like(x_um, dtype=float)

    jtype = cfg.junction_type.lower().strip()
    if jtype == "pin_diode":
        # Intrinsic core (`1e15` background p-) flanked by heavy P++ (left) & N++ (right) wings
        na_core = np.full_like(x_um, 1.0e15)
        nd_core = np.zeros_like(x_um)
    elif jtype == "l_shaped_pn":
        # L-shaped junction: vertical leg at x_leg = xj - 0.095 um + horizontal cap at y = yj across core peak!
        x_leg = xj - 0.095
        x_cap_end = max(xj + 0.16, 0.5 * cfg.waveguide_width_um - 0.03)
        f_left = 0.5 * erfc((x_um - x_leg) / s2)
        f_top_cap = (0.5 * erfc(-(y_um - yj) / s2)) * (0.5 * erfc((x_um - x_cap_end) / s2))
        p_frac = np.clip(np.maximum(f_left, f_top_cap), 0.0, 1.0)
        na_core = cfg.na_core_cm3 * p_frac
        nd_core = cfg.nd_core_cm3 * (1.0 - p_frac)
    elif jtype == "u_shaped_pn":
        # U-shaped wrap-around: n-type inner pillar (`|x - xj| < 0.11` and `y < 0.15`), p-type wrapping top & outer walls
        in_inner_n = (0.5 * erfc((np.abs(x_um - xj) - 0.11) / s2)) * (0.5 * erfc((y_um - 0.155) / s2))
        nd_core = cfg.nd_core_cm3 * in_inner_n
        na_core = cfg.na_core_cm3 * (1.0 - in_inner_n)
    else:
        # Standard lateral PN junction at x = xj
        p_frac = 0.5 * erfc((x_um - xj) / s2)
        na_core = cfg.na_core_cm3 * p_frac
        nd_core = cfg.nd_core_cm3 * (1.0 - p_frac)

    # Intermediate p+ / n+ slab wings (lowering series resistance without overlapping optical core peak)
    x_pi = -max(float(cfg.inter_implant_offset_um), 0.5 * cfg.waveguide_width_um + 0.04)
    x_ni = +max(float(cfg.inter_implant_offset_um), 0.5 * cfg.waveguide_width_um + 0.04)
    in_slab_y = 0.5 * erfc((y_um - (cfg.slab_height_um + 0.015)) / s2)

    if jtype != "pin_diode":
        na_inter = cfg.na_inter_cm3 * 0.5 * erfc((x_um - x_pi) / (1.5 * s2)) * in_slab_y
        nd_inter = cfg.nd_inter_cm3 * 0.5 * erfc(-(x_um - x_ni) / (1.5 * s2)) * in_slab_y
    else:
        na_inter = np.zeros_like(x_um)
        nd_inter = np.zeros_like(x_um)

    # Heavy P++ / N++ ohmic contact wings at +- heavy_contact_offset_um
    x_pp = -max(float(cfg.heavy_contact_offset_um), 0.5 * cfg.waveguide_width_um + 0.18)
    x_nn = +max(float(cfg.heavy_contact_offset_um), 0.5 * cfg.waveguide_width_um + 0.18)
    na_heavy = cfg.n_heavy_contact_cm3 * 0.5 * erfc((x_um - x_pp) / (1.8 * s2)) * in_slab_y
    nd_heavy = cfg.n_heavy_contact_cm3 * 0.5 * erfc(-(x_um - x_nn) / (1.8 * s2)) * in_slab_y

    na = np.where(in_si_nodes, np.maximum.reduce([na_core, na_inter, na_heavy]), 0.0)
    nd = np.where(in_si_nodes, np.maximum.reduce([nd_core, nd_inter, nd_heavy]), 0.0)
    return na, nd


def simulate_pn_doping_tcad(cfg: PNDopingTCADConfig) -> dict:
    """Solve 2D Nonlinear Poisson-Boltzmann PN/PIN TCAD + Coupled 2D Vector Maxwell Eigenmodes."""
    t0 = time.perf_counter()
    biases = [float(v) for v in cfg.bias_voltages_v] if cfg.bias_voltages_v else [0.0, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0]
    if not any(abs(v) < 1e-9 for v in biases):
        biases = [0.0] + biases

    h_film = max(float(cfg.film_height_um), 0.12)
    h_slab = float(np.clip(cfg.slab_height_um, 0.03, 0.80 * h_film))
    w_wg = max(float(cfg.waveguide_width_um), 0.22)
    inv_tan = 0.0 if cfg.sidewall_deg == 90.0 else 1.0 / math.tan(math.radians(cfg.sidewall_deg))
    w_top = max(0.08, w_wg - 2.0 * (h_film - h_slab) * inv_tan)

    d_cont = max(float(cfg.heavy_contact_offset_um), 0.5 * w_wg + 0.20)
    x_dom = d_cont + 0.55
    xj = float(cfg.pn_offset_nm) * 1e-3

    # High-resolution 2D Semiconductor TCAD mesh (12 nm spacing inside rib core/junction!)
    x_knots = sorted(set([
        -x_dom, -d_cont, -float(cfg.inter_implant_offset_um),
        -0.5 * w_wg, -0.5 * w_top, xj - 0.08, xj, xj + 0.08,
        0.5 * w_top, 0.5 * w_wg,
        +float(cfg.inter_implant_offset_um), +d_cont, +x_dom,
    ]))
    y_knots = sorted(set([-0.45, 0.0, 0.5 * h_slab, h_slab, 0.5 * (h_slab + h_film), h_film, h_film + 0.45]))
    if cfg.junction_type == "l_shaped_pn":
        y_knots.append(float(np.clip(cfg.l_vertical_y_nm * 1e-3, h_slab + 0.01, h_film - 0.01)))

    # Fine 14 nm mesh across [-0.35, +0.35] um core, 45 nm across slab wings
    segs_x: list[np.ndarray] = []
    uniq_x = sorted(set(round(k, 5) for k in x_knots))
    for i in range(len(uniq_x) - 1):
        a, b = uniq_x[i], uniq_x[i + 1]
        if b - a <= 1e-6:
            continue
        mid = 0.5 * (a + b)
        hx = 0.014 if abs(mid) <= 0.5 * w_wg + 0.12 else 0.045
        seg = np.linspace(a, b, max(2, int(math.ceil((b - a) / hx))) + 1)
        segs_x.append(seg if not segs_x else seg[1:])
    x_axis_um = np.concatenate(segs_x)

    segs_y: list[np.ndarray] = []
    uniq_y = sorted(set(round(k, 5) for k in y_knots))
    for i in range(len(uniq_y) - 1):
        a, b = uniq_y[i], uniq_y[i + 1]
        if b - a <= 1e-6:
            continue
        mid = 0.5 * (a + b)
        hy = 0.014 if -0.02 <= mid <= h_film + 0.02 else 0.055
        seg = np.linspace(a, b, max(2, int(math.ceil((b - a) / hy))) + 1)
        segs_y.append(seg if not segs_y else seg[1:])
    y_axis_um = np.concatenate(segs_y)

    mesh, xc_um, yc_um = _rectilinear_mesh_tri(x_axis_um, y_axis_um)
    basis = Basis(mesh, ElementTriP1())
    basis0 = basis.with_element(ElementTriP0())

    poly_rib = Polygon([
        (-0.5 * w_wg, h_slab),
        (+0.5 * w_wg, h_slab),
        (+0.5 * w_top, h_film),
        (-0.5 * w_top, h_film),
    ])
    poly_slab = box(-x_dom - 0.01, 0.0, +x_dom + 0.01, h_slab)
    poly_si = poly_rib.union(poly_slab)

    in_si_el = shapely.contains_xy(poly_si.buffer(1e-5), xc_um, yc_um)
    eps_el = np.where(in_si_el, 11.7 * EPS0_F_PER_M, 3.9 * EPS0_F_PER_M)
    eps_dofs = basis0.zeros(dtype=float)
    eps_dofs[:] = eps_el
    K_pois = _poisson_stiff_form.assemble(basis, eps_si_ox=basis0.interpolate(eps_dofs))

    # Lumped nodal control-volume weights w_vol [m^2] in Silicon ONLY (0 in SiO2!)
    mesh_si_mask = Basis(mesh, ElementTriP0())
    si_indicator = mesh_si_mask.zeros(dtype=float)
    si_indicator[:] = np.where(in_si_el, 1.0, 0.0)

    @LinearForm
    def _si_vol_form(v, w):
        return w["ind"] * v

    vol_si_nodes_m2 = _si_vol_form.assemble(basis, ind=mesh_si_mask.interpolate(si_indicator))
    in_si_nodes = vol_si_nodes_m2 > 1e-18

    px_um = mesh.p[0] * 1e6
    py_um = mesh.p[1] * 1e6
    na_nodes_cm3, nd_nodes_cm3 = _build_2d_doping_profile_cm3(px_um, py_um, cfg, in_si_nodes)
    net_doping_m3 = (nd_nodes_cm3 - na_nodes_cm3) * 1e6

    vt = (KB_J_PER_K * T_REF_K) / Q_E_C
    ni_m3 = NI_SI_300K_CM3 * 1e6

    # Ohmic silicide contact Dirichlet nodes at left P++ edge and right N++ edge on slab top
    p_cont_nodes = np.flatnonzero(in_si_nodes & (px_um <= -d_cont - 0.05) & (py_um <= h_slab + 1e-4))
    n_cont_nodes = np.flatnonzero(in_si_nodes & (px_um >= +d_cont + 0.05) & (py_um <= h_slab + 1e-4))
    dir_nodes = np.unique(np.concatenate([p_cont_nodes, n_cont_nodes]))
    free_mask = np.ones(basis.N, dtype=bool)
    free_mask[dir_nodes] = False
    free_idx = np.flatnonzero(free_mask)

    # Charge-neutral initial guess at 0 V
    psi_eq = np.where(in_si_nodes, vt * np.arcsinh(net_doping_m3 / (2.0 * ni_m3)), 0.0)

    def _solve_poisson_boltzmann_at_bias(
        v_bias_v: float,
        psi_init: np.ndarray,
        v_init_v: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Damped Newton-Raphson solver for 2D Nonlinear Poisson-Boltzmann PDE + Gauss-Law terminal charge."""
        # Keep p-contact at reference 0 V and apply reverse bias V_bias <= 0 by raising n-side quasi-Fermi phi_Fn = -V_bias
        v_clamped = min(float(v_bias_v), 0.55)
        phi_fp = 0.0
        phi_fn = -v_clamped

        # Smooth warm-start across both Si and cladding so heavy 1e20 N++ wing starts near charge neutrality
        dv_shift = -(v_clamped - min(float(v_init_v), 0.55))
        n_side_w = 0.5 * (1.0 + np.tanh((px_um - xj) / 0.08))
        psi = psi_init + dv_shift * n_side_w
        psi[p_cont_nodes] = phi_fp + vt * np.arcsinh(net_doping_m3[p_cont_nodes] / (2.0 * ni_m3))
        psi[n_cont_nodes] = phi_fn + vt * np.arcsinh(net_doping_m3[n_cont_nodes] / (2.0 * ni_m3))

        for _it in range(95):
            arg_p = np.clip((phi_fp - psi) / vt, -60.0, 60.0)
            arg_n = np.clip((psi - phi_fn) / vt, -60.0, 60.0)
            p_m3 = np.where(in_si_nodes, ni_m3 * np.exp(arg_p), 0.0)
            n_m3 = np.where(in_si_nodes, ni_m3 * np.exp(arg_n), 0.0)

            rho_q = Q_E_C * (p_m3 - n_m3 + net_doping_m3) * vol_si_nodes_m2
            res_vec = (K_pois @ psi) - rho_q
            res_free = res_vec[free_idx]
            if _it >= 2 and float(np.max(np.abs(res_free))) < 1e-20:
                break

            # Exact analytical Jacobian J = K_pois + diag(q * (p + n) / V_T * vol_si) (strictly SPD!)
            jac_diag = Q_E_C * ((p_m3 + n_m3) / vt) * vol_si_nodes_m2
            J_tot = K_pois + diags(jac_diag, 0, format="csc")
            J_ff = J_tot[free_idx, :][:, free_idx]
            dpsi_f = spsolve(J_ff, -res_free)

            # Limit maximum step to 6.0 * V_T (~155 mV) per Newton iteration so exponentials never overflow
            max_step = float(np.max(np.abs(dpsi_f)))
            alpha_damp = min(1.0, (6.0 * vt) / max(max_step, 1e-15))
            psi[free_idx] += alpha_damp * dpsi_f
            if alpha_damp >= 0.999999 and max_step < 1e-11:
                break

        arg_p = np.clip((phi_fp - psi) / vt, -60.0, 60.0)
        arg_n = np.clip((psi - phi_fn) / vt, -60.0, 60.0)
        p_m3 = np.where(in_si_nodes, ni_m3 * np.exp(arg_p), 0.0)
        n_m3 = np.where(in_si_nodes, ni_m3 * np.exp(arg_n), 0.0)
        # Active junction electron sheet charge Q_n [C/m] inside |x| <= d_cont - 0.12 um (excluding 1e20 ohmic pad)
        junc_mask = in_si_nodes & (np.abs(px_um) <= d_cont - 0.12)
        q_n_junc_c_per_m = float(np.sum((n_m3 * vol_si_nodes_m2)[junc_mask])) * Q_E_C
        return psi, n_m3 * 1e-6, p_m3 * 1e-6, q_n_junc_c_per_m

    ny_g, nx_g = len(y_axis_um), len(x_axis_um)
    psi_prev = psi_eq.copy()
    v_prev = 0.0

    # Spatially varying Caughey-Thomas Silicon hole & electron mobilities [cm^2/(V*s)] across core/slab/heavy-contact!
    ntot_cm3 = np.maximum(na_nodes_cm3 + nd_nodes_cm3, 1.0e14)
    mu_p_nodes = 44.9 + (470.5 - 44.9) / (1.0 + (ntot_cm3 / 2.23e17) ** 0.719)
    mu_n_nodes = 68.5 + (1414.0 - 68.5) / (1.0 + (ntot_cm3 / 9.20e16) ** 0.711)

    # Solve across all requested bias voltages + central +/-25 mV finite difference for exact 2D differential C_j(V) = dQ_n / dV!
    bias_solutions: list[dict] = []
    perturbation_fns = []

    for v_b in biases:
        psi_v, n_v, p_v, _ = _solve_poisson_boltzmann_at_bias(v_b, psi_prev, v_prev)
        psi_prev = psi_v
        v_prev = v_b
        dv = 0.025
        _, _, _, q_n_plus = _solve_poisson_boltzmann_at_bias(v_b + dv, psi_v, v_b)
        _, _, _, q_n_minus = _solve_poisson_boltzmann_at_bias(v_b - dv, psi_v, v_b)
        cj_f_per_m = abs(q_n_plus - q_n_minus) / (2.0 * dv) + 0.035e-9  # 2D space-charge dQ/dV + 0.035 fF/um SiO2 fringing cap
        cj_ff_per_um = cj_f_per_m * 1e9

        # 2D slab series resistance R_s [Ohm*cm] integrated across undepleted p/p+/p++ and n/n+/n++ wings up to depletion edges!
        sigma_s_per_m = Q_E_C * (mu_p_nodes * p_v + mu_n_nodes * n_v) * 100.0
        sigma_2d = sigma_s_per_m.reshape(ny_g, nx_g)
        slab_rows = (y_axis_um >= -1e-5) & (y_axis_um <= h_slab + 1e-5)
        dy_m = np.gradient(y_axis_um[slab_rows]) * 1e-6
        dx_m = np.gradient(x_axis_um) * 1e-6
        g_sheet_s = np.sum(sigma_2d[slab_rows, :] * dy_m[:, None], axis=0)  # sheet conductance [S] along x
        # Exclude only the carrier-depleted space-charge core (where sheet conductance drops below 22% of core neutral slab)
        g_core_ref = Q_E_C * (160.0 * max(cfg.na_core_cm3, 1e17)) * 100.0 * (h_slab * 1e-6)
        in_access = (np.abs(x_axis_um) <= d_cont) & (g_sheet_s >= 0.22 * g_core_ref)
        r_unit_len_ohm_m = float(np.sum(dx_m[in_access] / np.maximum(g_sheet_s[in_access], 1e-5))) + 0.0040
        r_series_ohm_cm = r_unit_len_ohm_m * 100.0  # [Ohm * cm] for 1 cm length (R_device = R_ohm_cm / L_cm)

        n_2d = n_v.reshape(ny_g, nx_g)
        p_2d = p_v.reshape(ny_g, nx_g)
        psi_2d = psi_v.reshape(ny_g, nx_g)
        in_n = RegularGridInterpolator((y_axis_um, x_axis_um), n_2d, bounds_error=False, fill_value=0.0)
        in_p = RegularGridInterpolator((y_axis_um, x_axis_um), p_2d, bounds_error=False, fill_value=0.0)

        def _mk_fn(interp_n_loc, interp_p_loc):
            def _fn(xu: np.ndarray, yu: np.ndarray, reg: str):
                if reg not in ("core", "slab"):
                    z = np.zeros_like(xu, dtype=complex)
                    return z, z, z
                pts = np.column_stack([yu, xu])
                nl = np.maximum(interp_n_loc(pts), 0.0)
                pl = np.maximum(interp_p_loc(pts), 0.0)
                dn_r, dk_i = soref_bennett_si_dn_dk(nl, pl, cfg.wavelength_nm)
                c = dn_r + 1j * dk_i
                return c, c, c
            return _fn

        perturbation_fns.append(_mk_fn(in_n, in_p))
        bias_solutions.append({
            "voltage_v": round(v_b, 4),
            "cj_ff_per_um": round(cj_ff_per_um, 4),
            "r_series_ohm_cm": round(r_series_ohm_cm, 4),
            "psi_2d": psi_2d,
            "n_2d": n_2d,
            "p_2d": p_2d,
        })

    # Solve 2D Vector Maxwell Optical Eigenmode across intrinsic Si + all TCAD bias states!
    wl_m = cfg.wavelength_nm * 1e-9
    wl_scale = float(np.clip(cfg.wavelength_nm / 1550.0, 0.35, 4.50))
    spec = WaveguideCrossSectionSpec(
        w_bottom_um=w_wg,
        h_film_um=h_film,
        h_slab_um=h_slab,
        sidewall_deg=cfg.sidewall_deg,
        core_material_fn=get_material_fn("Si"),
        clad_material_fn=get_material_fn("SiO2"),
        top_clad_material_fn=get_material_fn("SiO2"),
        domain_width_um=max(4.6, 2.0 * d_cont + 1.4),
        h_box_um=1.85,
        h_top_clad_um=1.85,
    )
    opt_rows = solve_perturbed_neff_sweep_fem(
        spec, wl_m, [None] + perturbation_fns,
        polarization=cfg.polarization,
        core_mesh_res_um=float(np.clip(0.032 * wl_scale, 0.016, 0.08)),
        element_order=1,
    )
    ref_0v_idx = int(np.argmin(np.abs([b["voltage_v"] for b in bias_solutions])))
    neff_0v = float(opt_rows[1 + ref_0v_idx]["neff_real"])
    push_pull_mult = 2.0 if cfg.drive_scheme == "push_pull" else 1.0
    l_cm = max(cfg.active_length_um * 1e-4, 1e-4)

    states_table: list[dict] = []
    for idx_b, b_sol in enumerate(bias_solutions):
        orow = opt_rows[1 + idx_b]
        vb = float(b_sol["voltage_v"])
        dneff_re = float(orow["neff_real"]) - neff_0v
        alpha_fca = float(orow["alpha_modal_db_per_cm"])
        dphi_pi = (2.0 * dneff_re * push_pull_mult * (cfg.active_length_um * 1e-6)) / wl_m

        # Secant V_pi*L from 0 V to vb (or local tangent at 0 V)
        if abs(vb) > 0.08 and abs(dneff_re) > 1e-9:
            vpil = (wl_m * abs(vb) / (2.0 * abs(dneff_re) * push_pull_mult)) * 100.0
        else:
            vpil = None

        r_dev_ohm = float(b_sol["r_series_ohm_cm"]) / l_cm + 2.5
        c_dev_ff = (float(b_sol["cj_ff_per_um"]) * cfg.active_length_um) / (2.0 if cfg.drive_scheme == "push_pull" else 1.0) + 10.0
        f_rc_ghz = 1.0 / (2.0 * math.pi * (r_dev_ohm + cfg.driver_impedance_ohm) * (c_dev_ff * 1e-15)) * 1e-9
        f_rc_intr_ghz = 1.0 / (2.0 * math.pi * max(float(b_sol["r_series_ohm_cm"]), 1e-4) * (max(float(b_sol["cj_ff_per_um"]), 1e-4) * 1e-11)) * 1e-9

        # 50-Ohm Traveling-Wave MZM Electro-Optic Bandwidth loaded by user's PN junction C_j(V) & R_s(V)
        f_tw_vec = np.linspace(0.2, 250.0, 320)
        om_tw = 2.0 * math.pi * f_tw_vec * 1e9
        cj_m = (float(b_sol["cj_ff_per_um"]) * 1e-9) / (2.0 if cfg.drive_scheme == "push_pull" else 1.0)
        rs_m = (float(b_sol["r_series_ohm_cm"]) * 1e-2) * (2.0 if cfg.drive_scheme == "push_pull" else 1.0)
        rc_tw = om_tw * rs_m * cj_m
        g_pn_m = (om_tw ** 2 * (cj_m ** 2) * rs_m) / (1.0 + rc_tw ** 2)
        z0_tw = max(float(cfg.driver_impedance_ohm), 25.0)
        # Conductor skin loss (~0.32 dB/cm/sqrt(GHz) -> Np/m) + distributed PN shunt conductance 0.5 * Z0 * G_pn + walk-off |n_RF - n_g| ~ 0.18
        al_np_m = 0.037 * np.sqrt(f_tw_vec) * 100.0 + 0.5 * z0_tw * g_pn_m
        u_tw = (-al_np_m + 1j * om_tw * 0.18 / 299792458.0) * (cfg.active_length_um * 1e-6)
        h_tw = np.abs((np.exp(u_tw) - 1.0) / np.where(np.abs(u_tw) < 1e-9, 1e-9 + 0j, u_tw))
        h_tw_db = 20.0 * np.log10(np.maximum(h_tw / max(float(h_tw[0]), 1e-12), 1e-5))
        idx_3dbe = np.flatnonzero(h_tw_db <= -3.0103)
        idx_3dbo = np.flatnonzero(h_tw_db <= -6.0206)
        f_tw_3dbe = float(f_tw_vec[int(idx_3dbe[0])]) if len(idx_3dbe) else 250.0
        f_tw_3dbo = float(f_tw_vec[int(idx_3dbo[0])]) if len(idx_3dbo) else 250.0
        if cfg.junction_type == "pin_diode" and vb > 0.4:
            # Forward-injected PIN diode is governed by ambipolar carrier recombination lifetime tau_eff ~ 0.45 ns
            f_eff_mod = round(1.0 / (2.0 * math.pi * 0.45e-9) * 1e-9, 3)
        else:
            f_eff_mod = round(f_tw_3dbo if cfg.active_length_um >= 500.0 else f_rc_ghz, 2)

        states_table.append({
            "voltage_v": vb,
            "cj_ff_per_um": b_sol["cj_ff_per_um"],
            "r_series_ohm": round(r_dev_ohm, 2),
            "r_series_ohm_cm": b_sol["r_series_ohm_cm"],
            "f_rc_lumped_ghz": round(f_rc_ghz, 2),
            "f_rc_intrinsic_ghz": round(f_rc_intr_ghz, 2),
            "f_tw_3dbe_ghz": round(f_tw_3dbe, 2),
            "f_tw_3dbo_ghz": round(f_tw_3dbo, 2),
            "f_modulator_3db_ghz": f_eff_mod,
            "neff_real": round(float(orow["neff_real"]), 6),
            "delta_neff_real": float(f"{dneff_re:+.4e}"),
            "alpha_fca_db_per_cm": round(alpha_fca, 3),
            "phase_shift_pi": round(dphi_pi, 4),
            "v_pi_l_secant_vcm": round(vpil, 3) if vpil is not None else None,
        })

    # Fill 0V V_pi*L using nearest non-zero bias state
    nonzero_vpil = [s["v_pi_l_secant_vcm"] for s in states_table if s["v_pi_l_secant_vcm"] is not None]
    default_vpil = nonzero_vpil[min(1, len(nonzero_vpil) - 1)] if nonzero_vpil else 1.50
    for s in states_table:
        if s["v_pi_l_secant_vcm"] is None:
            s["v_pi_l_secant_vcm"] = default_vpil

    # Export 2D TCAD Carrier + Electrostatic Potential Map at map_bias_v
    idx_map = int(np.argmin(np.abs([b["voltage_v"] - cfg.map_bias_v for b in bias_solutions])))
    bs_map = bias_solutions[idx_map]
    # Subsample cleanly onto 128 x 88 visualization grid
    xv = np.linspace(-min(x_dom, 1.35), +min(x_dom, 1.35), 132)
    yv = np.linspace(-0.20, h_film + 0.22, 88)
    yy_v, xx_v = np.meshgrid(yv, xv, indexing="ij")
    pts_v = np.column_stack([yy_v.ravel(), xx_v.ravel()])
    ip_n = RegularGridInterpolator((y_axis_um, x_axis_um), bs_map["n_2d"], bounds_error=False, fill_value=0.0)
    ip_p = RegularGridInterpolator((y_axis_um, x_axis_um), bs_map["p_2d"], bounds_error=False, fill_value=0.0)
    ip_psi = RegularGridInterpolator((y_axis_um, x_axis_um), bs_map["psi_2d"], bounds_error=False, fill_value=0.0)

    nv_g = ip_n(pts_v).reshape(len(yv), len(xv))
    pv_g = ip_p(pts_v).reshape(len(yv), len(xv))
    psiv_g = ip_psi(pts_v).reshape(len(yv), len(xv))
    # Signed log10 carrier map: +log10(n/1e15) in n-region, -log10(p/1e15) in p-region, ~0 inside depleted space-charge zone!
    signed_log_carrier = np.log10(np.maximum(nv_g, 1e14) / 1e14) - np.log10(np.maximum(pv_g, 1e14) / 1e14)
    in_si_vis = shapely.contains_xy(poly_si.buffer(1e-4), xx_v, yy_v)
    signed_log_carrier = np.where(in_si_vis, signed_log_carrier, 0.0)

    op_row = states_table[idx_map]
    loss_fom = round(float(op_row["v_pi_l_secant_vcm"]) * float(op_row["alpha_fca_db_per_cm"]), 2)

    return {
        "mode": "pn_doping_tcad",
        "config": asdict(cfg),
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "junction_name_en": PN_JUNCTION_CATALOG.get(cfg.junction_type, PN_JUNCTION_CATALOG["lateral_pn"])["name_en"],
        "junction_name_zh": PN_JUNCTION_CATALOG.get(cfg.junction_type, PN_JUNCTION_CATALOG["lateral_pn"])["name_zh"],
        "metrics": {
            "operating_bias_V": bs_map["voltage_v"],
            "V_pi_L_Vcm": op_row["v_pi_l_secant_vcm"],
            "V_pi_V": round(op_row["v_pi_l_secant_vcm"] / l_cm, 3),
            "alpha_FCA_0V_dB_per_cm": states_table[ref_0v_idx]["alpha_fca_db_per_cm"],
            "alpha_FCA_bias_dB_per_cm": op_row["alpha_fca_db_per_cm"],
            "V_pi_L_alpha_V_dB": loss_fom,
            "Cj_0V_fF_per_um": states_table[ref_0v_idx]["cj_ff_per_um"],
            "Cj_bias_fF_per_um": op_row["cj_ff_per_um"],
            "R_series_ohm_cm": op_row["r_series_ohm_cm"],
            "f_RC_GHz": op_row["f_rc_lumped_ghz"],
            "f_RC_intrinsic_GHz": op_row["f_rc_intrinsic_ghz"],
            "f_TW_3dBe_GHz": op_row["f_tw_3dbe_ghz"],
            "f_TW_3dBo_GHz": op_row["f_tw_3dbo_ghz"],
            "f_modulator_3dB_GHz": op_row["f_modulator_3db_ghz"],
            "neff_intrinsic": round(float(opt_rows[0]["neff_real"]), 6),
        },
        "states": states_table,
        "field_map": {
            "bias_v": bs_map["voltage_v"],
            "x_um": np.round(xv, 4).tolist(),
            "y_um": np.round(yv, 4).tolist(),
            "signed_log_carrier": np.round(signed_log_carrier, 3).tolist(),
            "psi_v": np.round(psiv_g, 4).tolist(),
            "rib_outline_um": [[round(x, 4), round(y, 4)] for x, y in poly_rib.exterior.coords],
            "slab_height_um": round(h_slab, 4),
            "film_height_um": round(h_film, 4),
            "pn_offset_um": round(xj, 4),
        },
    }


def export_pn_doping_gds(pn_result: dict, out_path: str | Path) -> dict:
    """Export multi-implant Silicon PN/PIN Modulator `.GDS` + KLayout `.lyp`."""
    out_p = Path(out_path)
    if out_p.suffix.lower() != ".gds":
        out_p = out_p / "MicroRing_Si_PN_Doping_EOM.gds"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cfg = pn_result["config"]
    mm = pn_result["metrics"]
    w_wg = float(cfg["waveguide_width_um"])
    xj = float(cfg["pn_offset_nm"]) * 1e-3
    d_cont = float(cfg["heavy_contact_offset_um"])
    l_vis = float(np.clip(cfg["active_length_um"], 160.0, 520.0))
    y_sep = 28.0

    lyr_core = MICRORING_LAYERS["core"]
    lyr_slab = MICRORING_LAYERS["slab"]
    lyr_p = MICRORING_LAYERS["p_implant"]
    lyr_n = MICRORING_LAYERS["n_implant"]
    lyr_via = MICRORING_LAYERS["via"]
    lyr_m1 = MICRORING_LAYERS["electrode"]
    lyr_txt = MICRORING_LAYERS["label"]

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell_name = f"EOM_SI_{str(cfg['junction_type']).upper()}"
    cell = lib.new_cell(cell_name)

    x0, x1 = 65.0, 65.0 + l_vis
    # MZI Splitter + 2 Arms + Combiner + Slab + Multi-Implant P/N Windows + Via Arrays + GSG Electrodes
    cell.add(gdstk.rectangle((0.0, -0.5 * w_wg), (22.0, +0.5 * w_wg), layer=lyr_core[0], datatype=lyr_core[1]))
    cell.add(gdstk.rectangle((22.0, -3.0), (40.0, +3.0), layer=lyr_core[0], datatype=lyr_core[1]))
    for sgn in (+1.0, -1.0):
        ya = sgn * 0.5 * y_sep
        u = np.linspace(0.0, 1.0, 24)
        pts_in = [(40.0 + 25.0 * t, sgn * 1.5 + (ya - sgn * 1.5) * 0.5 * (1.0 - np.cos(math.pi * t))) for t in u]
        pts_out = [(x1 + 25.0 * t, ya + (sgn * 1.5 - ya) * 0.5 * (1.0 - np.cos(math.pi * t))) for t in u]
        for p in gdstk.FlexPath(pts_in, w_wg, layer=lyr_core[0], datatype=lyr_core[1]).to_polygons():
            cell.add(p)
        cell.add(gdstk.rectangle((x0, ya - 0.5 * w_wg), (x1, ya + 0.5 * w_wg), layer=lyr_core[0], datatype=lyr_core[1]))
        for p in gdstk.FlexPath(pts_out, w_wg, layer=lyr_core[0], datatype=lyr_core[1]).to_polygons():
            cell.add(p)

        # Rib slab & P / N implant masks straddling junction offset xj on each arm!
        cell.add(gdstk.rectangle((x0 - 4.0, ya - (d_cont + 3.5)), (x1 + 4.0, ya + (d_cont + 3.5)), layer=lyr_slab[0], datatype=lyr_slab[1]))
        # Mirror P/N between Arm 1 and Arm 2 so inner center Signal touches N++ on both arms (back-to-back push-pull diode!)
        if sgn > 0:
            cell.add(gdstk.rectangle((x0, ya + xj), (x1, ya + d_cont + 3.0), layer=lyr_p[0], datatype=lyr_p[1]))
            cell.add(gdstk.rectangle((x0, ya - (d_cont + 3.0)), (x1, ya + xj), layer=lyr_n[0], datatype=lyr_n[1]))
        else:
            cell.add(gdstk.rectangle((x0, ya - xj), (x1, ya + (d_cont + 3.0)), layer=lyr_n[0], datatype=lyr_n[1]))
            cell.add(gdstk.rectangle((x0, ya - (d_cont + 3.0)), (x1, ya - xj), layer=lyr_p[0], datatype=lyr_p[1]))

        for xv in np.linspace(x0 + 5.0, x1 - 5.0, max(6, int(l_vis // 14.0))):
            cell.add(gdstk.rectangle((float(xv) - 0.5, ya + d_cont + 0.5), (float(xv) + 0.5, ya + d_cont + 1.8), layer=lyr_via[0], datatype=lyr_via[1]))
            cell.add(gdstk.rectangle((float(xv) - 0.5, ya - d_cont - 1.8), (float(xv) + 0.5, ya - d_cont - 0.5), layer=lyr_via[0], datatype=lyr_via[1]))

    x_end = x1 + 65.0
    cell.add(gdstk.rectangle((x1 + 25.0, -3.0), (x1 + 43.0, +3.0), layer=lyr_core[0], datatype=lyr_core[1]))
    cell.add(gdstk.rectangle((x1 + 43.0, -0.5 * w_wg), (x_end, +0.5 * w_wg), layer=lyr_core[0], datatype=lyr_core[1]))

    # Push-Pull GSG Metal-1 Rails + North GSG Probe Pads
    y_sig_half = 0.5 * y_sep - d_cont - 0.3
    cell.add(gdstk.rectangle((x0, -y_sig_half), (x1, +y_sig_half), layer=lyr_m1[0], datatype=lyr_m1[1]))
    cell.add(gdstk.rectangle((x0, 0.5 * y_sep + d_cont + 0.3), (x1, 0.5 * y_sep + d_cont + 36.0), layer=lyr_m1[0], datatype=lyr_m1[1]))
    cell.add(gdstk.rectangle((x0, -(0.5 * y_sep + d_cont + 36.0)), (x1, -(0.5 * y_sep + d_cont + 0.3)), layer=lyr_m1[0], datatype=lyr_m1[1]))

    y_pad = 0.5 * y_sep + d_cont + 58.0
    for idx_p, lbl in enumerate(("GND", "SIG", "GND")):
        xp = x0 + 40.0 + idx_p * 125.0
        cell.add(gdstk.rectangle((xp - 40.0, y_pad), (xp + 40.0, y_pad + 80.0), layer=lyr_m1[0], datatype=lyr_m1[1]))
        cell.add(gdstk.Label(lbl, (xp - 14.0, y_pad + 36.0), layer=lyr_txt[0], texttype=lyr_txt[1]))

    cell.add(gdstk.Label(
        f"Si PN TCAD ({cfg['junction_type']}) VpiL={mm['V_pi_L_Vcm']:.2f}V*cm FCA={mm['alpha_FCA_bias_dB_per_cm']:.2f}dB/cm Cj={mm['Cj_bias_fF_per_um']:.2f}fF/um",
        (x0, -(0.5 * y_sep + d_cont + 48.0)),
        layer=lyr_txt[0], texttype=lyr_txt[1],
    ))

    lib.write_gds(str(out_p))
    lyp_p = out_p.with_suffix(".lyp")
    write_klayout_layer_props(lyp_p)
    used_layers = sorted({int(poly.layer) for poly in cell.polygons})
    return {
        "gds_path": str(out_p),
        "lyp_path": str(lyp_p),
        "cell_name": cell_name,
        "layers": used_layers,
        "footprint_um": [round(x_end, 2), round(y_pad + 85.0 + 0.5 * y_sep + 50.0, 2)],
    }


# ==============================================================================
# 2. 2D Steady-State + Crank-Nicolson Transient Thermal FEM + Trench Isolation
# ==============================================================================

@BilinearForm
def _thermal_stiff_form(u, v, w):
    return w["k_th"] * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])


@BilinearForm
def _thermal_mass_form(u, v, w):
    return w["rho_cp"] * u * v


@LinearForm
def _thermal_src_form(v, w):
    return w["q_vol"] * v


def simulate_thermal_phase_shifter(cfg: ThermalSimConfig) -> dict:
    """Solve 2D Steady-State, Crank-Nicolson Transient, Harmonic & Crosstalk Thermal FEM + Optical P_pi."""
    t0 = time.perf_counter()
    tstruct = cfg.thermal_structure.lower().strip()
    h_film = max(float(cfg.film_height_um), 0.12)
    h_slab = max(float(cfg.slab_height_um), 0.08 if tstruct == "doped_si_rib" else 0.0)
    w_wg = max(float(cfg.waveguide_width_um), 0.22)
    h_box = max(float(cfg.box_thickness_um), 0.50)
    l_heat = max(float(cfg.active_length_um), 10.0)

    h_mat_key = "doped_Si" if tstruct == "doped_si_rib" else cfg.heater_material
    h_metal = HEATER_METAL_CATALOG.get(h_mat_key, HEATER_METAL_CATALOG["TiN"])
    w_heat = max(float(cfg.heater_width_um), 0.35)
    t_heat = h_slab if tstruct == "doped_si_rib" else max(float(cfg.heater_thickness_um), 0.03)
    g_heat = 0.0 if tstruct == "doped_si_rib" else max(float(cfg.heater_gap_um), 0.20)

    y_heat0 = 0.0 if tstruct == "doped_si_rib" else (h_film + g_heat)
    y_heat1 = y_heat0 + t_heat
    y_ox_top = max(float(cfg.top_clad_thickness_um), y_heat1 + 0.45)
    y_air_top = y_ox_top + 4.0
    h_sub = max(26.0, float(cfg.trench_depth_um) + 14.0) if tstruct == "deep_trench" else 26.0
    y_sub_bot = -h_box - h_sub
    x_dom = max(float(cfg.neighbor_pitch_um) + 14.0, 36.0)

    inv_tan = 0.0 if cfg.sidewall_deg == 90.0 else 1.0 / math.tan(math.radians(cfg.sidewall_deg))
    w_top = max(0.08, w_wg - 2.0 * max(h_film - h_slab, 0.0) * inv_tan)

    tr_x0 = max(float(cfg.trench_edge_offset_um), 0.5 * max(w_wg, w_heat) + 0.35)
    tr_x1 = tr_x0 + max(float(cfg.trench_width_um), 1.0)
    tr_y_bot = max(y_ox_top - max(float(cfg.trench_depth_um), 1.5), y_sub_bot + 1.5)
    uc_half_w = 0.5 * max(float(cfg.undercut_width_um), 4.0)
    uc_y_bot = -h_box - max(float(cfg.undercut_depth_um), 2.0)

    clr_rib = 0.40
    w_wing_rib = float(np.clip(0.42 * w_heat, 0.60, 0.85))
    x_via_in = 0.5 * w_wg + clr_rib + w_wing_rib
    x_via_out = x_via_in + 0.45
    x_m1_sink = max(x_via_out + 9.5, 11.5)
    x_knots = [
        -x_dom, -float(cfg.neighbor_pitch_um), -uc_half_w, -tr_x1, -tr_x0,
        -0.5 * w_heat + cfg.heater_offset_um, -0.5 * w_wg, -0.5 * w_top, 0.0,
        +0.5 * w_top, +0.5 * w_wg, +0.5 * w_heat + cfg.heater_offset_um,
        +tr_x0, +tr_x1, +uc_half_w, +float(cfg.neighbor_pitch_um), +x_dom,
    ]
    if tstruct == "doped_si_rib":
        x_knots.extend([
            -x_m1_sink, -x_via_out, -x_via_in, -0.5 * w_wg - clr_rib,
            +0.5 * w_wg + clr_rib, +x_via_in, +x_via_out, +x_m1_sink,
        ])
    y_knots = [y_sub_bot, uc_y_bot, tr_y_bot, -h_box, 0.0, h_film, y_heat0, y_heat1, y_ox_top, y_air_top]
    if tstruct == "doped_si_rib":
        y_knots.extend([0.90, 1.45])
    if h_slab > 1e-5:
        y_knots.append(h_slab)

    x_axis_um = _snapped_1d_axis(x_knots, h_fine=0.15, h_coarse=1.10)
    y_axis_um = _snapped_1d_axis(y_knots, h_fine=0.11, h_coarse=0.85)
    mesh, xc_um, yc_um = _rectilinear_mesh_tri(x_axis_um, y_axis_um)

    k_core = float(cfg.custom_core_kappa_w_mk) if cfg.custom_core_kappa_w_mk is not None else THERMAL_CONDUCTIVITY_W_MK.get(cfg.core_material, 20.0)
    k_box = THERMAL_CONDUCTIVITY_W_MK.get(cfg.clad_material, 1.38)
    k_top = THERMAL_CONDUCTIVITY_W_MK.get(cfg.top_clad_material, 1.38)
    k_si_sub = 148.0
    k_air = THERMAL_CONDUCTIVITY_W_MK["Air"]

    cv_core = VOLUMETRIC_HEAT_CAPACITY_J_M3K.get(cfg.core_material, 1.85e6)
    cv_ox = VOLUMETRIC_HEAT_CAPACITY_J_M3K.get(cfg.clad_material, 1.63e6)
    cv_si = 1.63e6
    cv_air = 1.20e3  # dry air rho * Cp [J/(m^3*K)]

    k_el = np.full(mesh.nelements, k_top, dtype=float)
    cv_el = np.full(mesh.nelements, cv_ox, dtype=float)
    q_el = np.zeros(mesh.nelements, dtype=float)

    # Substrate, BOX, Top Cladding, Air above chip
    in_sub = yc_um < -h_box
    in_box = (yc_um >= -h_box) & (yc_um < 0.0)
    in_air = yc_um > y_ox_top
    k_el[in_sub], cv_el[in_sub] = k_si_sub, cv_si
    k_el[in_box], cv_el[in_box] = k_box, cv_ox
    k_el[in_air], cv_el[in_air] = k_air, cv_air

    if h_slab > 1e-5:
        slab_half_span = (x_via_in + 0.15) if tstruct == "doped_si_rib" else tr_x0
        in_slab = (yc_um >= 0.0) & (yc_um <= h_slab) & (np.abs(xc_um) <= slab_half_span)
        # Sub-100 nm heavily doped SOI slab phonon boundary + impurity scattering (Asheghi et al., J. Heat Transfer 120, 30, 1998)
        k_slab_eff = (
            min(k_core, 62.0)
            if (tstruct == "doped_si_rib" and cfg.core_material == "Si" and cfg.custom_core_kappa_w_mk is None)
            else k_core
        )
        k_el[in_slab], cv_el[in_slab] = k_slab_eff, cv_core

    poly_core = Polygon([
        (-0.5 * w_wg, h_slab), (+0.5 * w_wg, h_slab),
        (+0.5 * w_top, h_film), (-0.5 * w_top, h_film),
    ])
    in_core = shapely.contains_xy(poly_core, xc_um, yc_um)
    k_el[in_core], cv_el[in_core] = k_core, cv_core

    # Isolation features: Deep Air Trenches or XeF2 Isotropic Substrate Undercut
    trench_boxes: list[list[float]] = []
    if tstruct in ("deep_trench", "undercut_suspended"):
        for sgn in (-1.0, +1.0):
            bx0, bx1 = sorted([sgn * tr_x0, sgn * tr_x1])
            p_tr = box(bx0, tr_y_bot, bx1, y_ox_top + 0.05)
            m_tr = shapely.contains_xy(p_tr, xc_um, yc_um)
            k_el[m_tr], cv_el[m_tr] = k_air, cv_air
            trench_boxes.append([round(bx0, 3), round(tr_y_bot, 3), round(bx1, 3), round(y_ox_top, 3)])

    if tstruct == "undercut_suspended":
        # Out-of-plane anchor thermal conductance along suspended bridge between support tethers (~min(l_heat, 95 um))
        # Modeled in 2D by effective air+longitudinal-bridge conductance so 2D does not overestimate 1D infinity!
        l_susp_m = min(l_heat, 95.0) * 1e-6
        k_bridge_eff = k_air + (12.0 * (h_box * 1e-6) * k_box * (uc_half_w * 1e-6)) / (l_susp_m ** 2)
        p_uc = box(-uc_half_w, uc_y_bot, +uc_half_w, -h_box)
        m_uc = shapely.contains_xy(p_uc, xc_um, yc_um)
        k_el[m_uc], cv_el[m_uc] = k_bridge_eff, cv_air
        trench_boxes.append([round(-uc_half_w, 3), round(uc_y_bot, 3), round(+uc_half_w, 3), round(-h_box, 3)])

    # Heater geometry & Joule volumetric source for P = 1 mW
    y_m1_top = min(1.45, y_ox_top - 0.10)
    if tstruct == "doped_si_rib":
        # Longitudinal N++ doped-silicon slab strips alongside the rib + end-pad W vias & M1 leads (Jacques OE 27, 10456, 2019, Fig. 1b):
        # 550 nm clearance keeps optical mode loss-free while 90 nm SOI slab conducts heat directly into the core above the 2 um BOX.
        w_wing = w_wing_rib
        p_h1 = box(-0.5 * w_wg - clr_rib - w_wing, 0.0, -0.5 * w_wg - clr_rib, h_slab)
        p_h2 = box(+0.5 * w_wg + clr_rib, 0.0, +0.5 * w_wg + clr_rib + w_wing, h_slab)
        in_heat = shapely.contains_xy(p_h1, xc_um, yc_um) | shapely.contains_xy(p_h2, xc_um, yc_um)
        area_heat_m2 = 2.0 * (w_wing * 1e-6) * (h_slab * 1e-6)
        heater_boxes = [[round(b, 3) for b in p_h1.bounds], [round(b, 3) for b in p_h2.bounds]]
        # Periodic/end W contact vias & AlCu M1 pads (~10% longitudinal z-fill along L_heat)
        fz_end = float(np.clip(20.0 / max(l_heat, 40.0), 0.06, 0.22))
        k_via_eff = (1.0 - fz_end) * k_top + fz_end * 165.0
        k_m1_eff = (1.0 - fz_end) * k_top + fz_end * 205.0
        for sgn in (-1.0, +1.0):
            vx0, vx1 = sorted([sgn * x_via_in, sgn * x_via_out])
            mx0, mx1 = sorted([sgn * x_via_in, sgn * x_dom])
            m_via = shapely.contains_xy(box(vx0, h_slab, vx1, 0.90), xc_um, yc_um)
            m_m1 = shapely.contains_xy(box(mx0, 0.90, mx1, y_m1_top), xc_um, yc_um)
            k_el[m_via], cv_el[m_via] = k_via_eff, 1.70e6
            k_el[m_m1], cv_el[m_m1] = k_m1_eff, 1.75e6
    else:
        p_h = box(cfg.heater_offset_um - 0.5 * w_heat, y_heat0, cfg.heater_offset_um + 0.5 * w_heat, y_heat1)
        in_heat = shapely.contains_xy(p_h, xc_um, yc_um)
        k_el[in_heat], cv_el[in_heat] = float(h_metal["kappa_th_w_mk"]), 2.45e6
        area_heat_m2 = (w_heat * 1e-6) * (t_heat * 1e-6)
        heater_boxes = [[round(b, 3) for b in p_h.bounds]]

    rho_h_used = float(cfg.custom_heater_rho_ohm_m) if cfg.custom_heater_rho_ohm_m is not None else float(h_metal["rho_ohm_m"])
    r_heater_ohm = rho_h_used * (l_heat * 1e-6) / max(area_heat_m2, 1e-16) + 10.0
    q_vol_1mw = 1e-3 / (max(area_heat_m2, 1e-16) * (l_heat * 1e-6))
    q_el[in_heat] = q_vol_1mw

    basis = Basis(mesh, ElementTriP1())
    basis0 = basis.with_element(ElementTriP0())
    k_dofs = basis0.zeros(dtype=float); k_dofs[:] = k_el
    cv_dofs = basis0.zeros(dtype=float); cv_dofs[:] = cv_el
    q_dofs = basis0.zeros(dtype=float); q_dofs[:] = q_el

    K_th = _thermal_stiff_form.assemble(basis, k_th=basis0.interpolate(k_dofs))
    M_th = _thermal_mass_form.assemble(basis, rho_cp=basis0.interpolate(cv_dofs))
    b_1mw = _thermal_src_form.assemble(basis, q_vol=basis0.interpolate(q_vol_dofs if False else q_dofs))

    px_um = mesh.p[0] * 1e6
    py_um = mesh.p[1] * 1e6
    sink_mask = (py_um <= y_sub_bot + 1e-4) | (np.abs(px_um) >= x_dom - 1e-4)
    if tstruct == "doped_si_rib":
        sink_mask = sink_mask | ((np.abs(px_um) >= x_m1_sink - 1e-4) & (py_um >= 0.90 - 1e-4) & (py_um <= y_m1_top + 1e-4))
    sink_nodes = np.flatnonzero(sink_mask)
    free_mask = np.ones(basis.N, dtype=bool)
    free_mask[sink_nodes] = False
    free_idx = np.flatnonzero(free_mask)

    # 1. Steady-State 2D FEM Solution per 1 mW
    dt_1mw = solve(*condense(K_th, b_1mw, D=sink_nodes, x=np.zeros(basis.N)))
    dt_2d = dt_1mw.reshape(len(y_axis_um), len(x_axis_um))
    interp_dt = RegularGridInterpolator((y_axis_um, x_axis_um), dt_2d, bounds_error=False, fill_value=0.0)

    core_nodes = np.flatnonzero(shapely.contains_xy(poly_core.buffer(0.03), px_um, py_um))
    if len(core_nodes) == 0:
        core_nodes = np.array([int(np.argmin(px_um ** 2 + (py_um - 0.5 * h_film) ** 2))])

    r_th_core_k_mw = float(np.mean(dt_1mw[core_nodes]))
    r_th_peak_k_mw = float(np.max(dt_1mw))

    # 2. Coupled 2D Vector Maxwell Optical Eigenmode Solve -> Exact P_pi, V_pi, I_pi!
    wl_m = cfg.wavelength_nm * 1e-9
    wl_scale = float(np.clip(cfg.wavelength_nm / 1550.0, 0.35, 4.50))
    core_fn = get_material_fn(cfg.core_material)
    clad_fn = get_material_fn(cfg.clad_material)
    top_fn = get_material_fn(cfg.top_clad_material)
    mat_c = core_fn(wl_m, T_REF_K)
    mat_b = clad_fn(wl_m, T_REF_K)
    mat_t = top_fn(wl_m, T_REF_K)
    reg_m = {"core": mat_c, "slab": mat_c, "box": mat_b, "top_clad": mat_t}

    spec = WaveguideCrossSectionSpec(
        w_bottom_um=w_wg,
        h_film_um=h_film,
        h_slab_um=h_slab,
        sidewall_deg=cfg.sidewall_deg,
        core_material_fn=core_fn,
        clad_material_fn=clad_fn,
        top_clad_material_fn=top_fn,
        domain_width_um=max(5.0 * wl_scale, w_wg + 3.0 * max(wl_m * 1e6, 1.0)),
        h_box_um=max(1.85, 1.35 * wl_m * 1e6),
        h_top_clad_um=max(1.85, 1.35 * wl_m * 1e6),
    )
    p_test_mw = 10.0

    def _th_pert(xu: np.ndarray, yu: np.ndarray, reg: str):
        mr = reg_m.get(reg, mat_t)
        dt_l = p_test_mw * interp_dt(np.column_stack([yu, xu]))
        dndt_val = float(cfg.custom_dn_dt_per_k) if (cfg.custom_dn_dt_per_k is not None and reg in ("core", "slab")) else float(mr.dn_dt_per_k)
        dn = (dndt_val * dt_l).astype(complex)
        return dn, dn, dn

    opt_rows = solve_perturbed_neff_sweep_fem(
        spec, wl_m, [None, _th_pert],
        polarization=cfg.polarization,
        core_mesh_res_um=float(np.clip(0.040 * wl_scale, 0.018, 0.14)),
        element_order=1,
    )
    dneff_per_mw = abs(float(opt_rows[1]["neff_real"]) - float(opt_rows[0]["neff_real"])) / p_test_mw
    # Phase shift Delta_phi = 2 * pi / lambda * dneff_per_mw * P_mw * L = pi  =>  P_pi = lambda / (2 * L * dneff_per_mw)
    p_pi_mw = (wl_m / (2.0 * (l_heat * 1e-6) * max(dneff_per_mw, 1e-12)))
    v_pi_v = math.sqrt((p_pi_mw * 1e-3) * r_heater_ohm)
    i_pi_ma = ((p_pi_mw * 1e-3) / max(v_pi_v, 1e-6)) * 1e3
    dt_core_at_ppi = p_pi_mw * r_th_core_k_mw
    dt_heater_at_ppi = p_pi_mw * r_th_peak_k_mw

    # 3. Exact 2D Harmonic Frequency-Domain FEM `(K_ff + j*omega*M_ff) T = b_f` -> True Multi-Pole f_3dB_thermal!
    K_ff = K_th[free_idx, :][:, free_idx].tocsc()
    M_ff = M_th[free_idx, :][:, free_idx].tocsc()
    b_f = b_1mw[free_idx]
    core_free_pos = np.searchsorted(free_idx, core_nodes)

    f_khz_grid = np.logspace(-1.0, 3.0, 28)  # 0.1 kHz to 1000 kHz
    h_th_mag: list[float] = []
    for fk in f_khz_grid:
        w_rad = 2.0 * math.pi * (fk * 1e3)
        sol_c = spsolve(K_ff + (1j * w_rad) * M_ff, b_f)
        h_th_mag.append(float(np.abs(np.mean(sol_c[core_free_pos]))) / max(r_th_core_k_mw, 1e-12))

    h_th_arr = np.asarray(h_th_mag, dtype=float) / max(h_th_mag[0], 1e-12)
    h_th_db = 20.0 * np.log10(np.maximum(h_th_arr, 1e-5))
    below_3db = np.flatnonzero(h_th_db <= -3.0103)
    if len(below_3db) > 0 and below_3db[0] > 0:
        i0 = int(below_3db[0])
        f_3db_khz = float(np.interp(-3.0103, [h_th_db[i0], h_th_db[i0 - 1]], [f_khz_grid[i0], f_khz_grid[i0 - 1]]))
    else:
        f_3db_khz = float(f_khz_grid[-1])
    tau_eff_us = 1.0 / (2.0 * math.pi * max(f_3db_khz * 1e3, 1.0)) * 1e6

    # 4. Exact 2D Crank-Nicolson Implicit Transient Step Heating (0 -> t_pulse) & Cooling (t_pulse -> 2*t_pulse)
    t_pulse_us = max(4.2 * tau_eff_us, 8.0)
    n_half = 28
    dt_step_s = (t_pulse_us * 1e-6) / n_half
    A_cn = (M_ff + 0.5 * dt_step_s * K_ff).tocsc()
    B_cn = (M_ff - 0.5 * dt_step_s * K_ff).tocsc()

    t_vec_us = [0.0]
    t_core_wave_k = [0.0]
    u_cur = np.zeros(len(free_idx), dtype=float)
    rhs_on = dt_step_s * (p_pi_mw * b_f)

    for step in range(1, 2 * n_half + 1):
        rhs = (B_cn @ u_cur) + (rhs_on if step <= n_half else 0.0)
        u_cur = spsolve(A_cn, rhs)
        t_vec_us.append(round(step * dt_step_s * 1e6, 3))
        t_core_wave_k.append(round(float(np.mean(u_cur[core_free_pos])), 4))

    heat_seg_t = np.asarray(t_vec_us[: n_half + 1])
    heat_seg_y = np.asarray(t_core_wave_k[: n_half + 1])
    y_ss = max(float(heat_seg_y[-1]), 1e-9)
    t10 = float(np.interp(0.10 * y_ss, heat_seg_y, heat_seg_t))
    t90 = float(np.interp(0.90 * y_ss, heat_seg_y, heat_seg_t))
    t_rise_10_90_us = max(t90 - t10, 0.05)

    cool_seg_t = np.asarray(t_vec_us[n_half:]) - t_pulse_us
    cool_seg_y = np.asarray(t_core_wave_k[n_half:])
    t90_c = float(np.interp(0.90 * y_ss, cool_seg_y[::-1], cool_seg_t[::-1]))
    t10_c = float(np.interp(0.10 * y_ss, cool_seg_y[::-1], cool_seg_t[::-1]))
    t_fall_90_10_us = max(t10_c - t90_c, 0.05)

    # 5. Lateral Thermal Crosstalk Profile Delta_T(x) across waveguide plane y = 0.5 * h_film at P = P_pi!
    x_xtalk = np.linspace(0.0, min(x_dom - 1.0, max(cfg.neighbor_pitch_um * 1.6, 32.0)), 80)
    dt_xtalk_k = p_pi_mw * interp_dt(np.column_stack([np.full_like(x_xtalk, 0.5 * h_film), x_xtalk]))
    xtalk_at_neighbor_pct = float(np.interp(cfg.neighbor_pitch_um, x_xtalk, dt_xtalk_k)) / max(dt_core_at_ppi, 1e-9) * 100.0

    # 6. Export 2D Steady-State Temperature Map at P = P_pi
    x_vis = min(x_dom - 2.0, max(tr_x1 + 4.0, uc_half_w + 4.0, 14.0))
    xv = np.linspace(-x_vis, +x_vis, 136)
    yv = np.linspace(-min(h_box + 9.0, 14.0), y_ox_top + 1.5, 96)
    yy_v, xx_v = np.meshgrid(yv, xv, indexing="ij")
    dt_map_ppi = (p_pi_mw * interp_dt(np.column_stack([yy_v.ravel(), xx_v.ravel()]))).reshape(len(yv), len(xv))

    return {
        "mode": "thermal_fem",
        "config": asdict(cfg),
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "structure_name_en": THERMAL_STRUCTURE_CATALOG.get(tstruct, THERMAL_STRUCTURE_CATALOG["top_heater"])["name_en"],
        "structure_name_zh": THERMAL_STRUCTURE_CATALOG.get(tstruct, THERMAL_STRUCTURE_CATALOG["top_heater"])["name_zh"],
        "metrics": {
            "P_pi_mW": round(p_pi_mw, 3),
            "V_pi_V": round(v_pi_v, 3),
            "I_pi_mA": round(i_pi_ma, 3),
            "R_heater_ohm": round(r_heater_ohm, 2),
            "R_th_core_K_per_mW": round(r_th_core_k_mw, 4),
            "Delta_T_core_at_Ppi_K": round(dt_core_at_ppi, 2),
            "Delta_T_heater_at_Ppi_K": round(dt_heater_at_ppi, 1),
            "f_3dB_thermal_kHz": round(f_3db_khz, 2),
            "tau_thermal_us": round(tau_eff_us, 2),
            "t_rise_10_90_us": round(t_rise_10_90_us, 2),
            "t_fall_90_10_us": round(t_fall_90_10_us, 2),
            "FOM_Ppi_tau_mW_us": round(p_pi_mw * tau_eff_us, 2),
            "crosstalk_at_neighbor_pct": round(xtalk_at_neighbor_pct, 3),
            "neighbor_pitch_um": round(cfg.neighbor_pitch_um, 1),
        },
        "transient": {
            "time_us": t_vec_us,
            "delta_T_core_K": t_core_wave_k,
            "pulse_off_us": round(t_pulse_us, 2),
            "freq_kHz": np.round(f_khz_grid, 3).tolist(),
            "H_thermal_dB": np.round(h_th_db, 3).tolist(),
        },
        "crosstalk": {
            "distance_um": np.round(x_xtalk, 3).tolist(),
            "delta_T_K_at_Ppi": np.round(dt_xtalk_k, 4).tolist(),
            "normalized_pct": np.round(dt_xtalk_k / max(dt_core_at_ppi, 1e-9) * 100.0, 3).tolist(),
        },
        "field_map": {
            "x_um": np.round(xv, 3).tolist(),
            "y_um": np.round(yv, 3).tolist(),
            "delta_T_K": np.round(dt_map_ppi, 3).tolist(),
            "core_box_um": [round(-0.5 * w_wg, 3), round(h_slab, 3), round(+0.5 * w_wg, 3), round(h_film, 3)],
            "heater_boxes_um": heater_boxes,
            "trench_boxes_um": trench_boxes,
            "y_box_bottom_um": round(-h_box, 3),
            "y_oxide_top_um": round(y_ox_top, 3),
        },
    }


def export_thermal_shifter_gds(th_result: dict, out_path: str | Path) -> dict:
    """Export Cleanroom Thermo-Optic Phase-Shifter + Air-Isolation Trenches `.GDS` & `.lyp`."""
    out_p = Path(out_path)
    if out_p.suffix.lower() != ".gds":
        out_p = out_p / "MicroRing_Thermal_PhaseShifter.gds"
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cfg = th_result["config"]
    mm = th_result["metrics"]
    tstruct = str(cfg["thermal_structure"])
    w_wg = float(cfg["waveguide_width_um"])
    w_heat = float(cfg["heater_width_um"])
    l_vis = max(float(cfg["active_length_um"]), 80.0)
    y_sep = max(float(cfg["neighbor_pitch_um"]), 20.0)

    lyr_core = MICRORING_LAYERS["core"]
    lyr_slab = MICRORING_LAYERS["slab"]
    lyr_heat = MICRORING_LAYERS["heater"]
    lyr_m1 = MICRORING_LAYERS["electrode"]
    lyr_via = MICRORING_LAYERS["via"]
    lyr_tr = MICRORING_LAYERS["facet_trench"]
    lyr_txt = MICRORING_LAYERS["label"]

    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    safe_mat = str(cfg["core_material"]).replace("-", "")
    cell_name = f"EOM_THERMAL_{tstruct.upper()}_{safe_mat}"
    cell = lib.new_cell(cell_name)

    x0, x1 = 55.0, 55.0 + l_vis
    # MZI arms at y = +y_sep/2 (heated Arm 1) and y = -y_sep/2 (reference Arm 2)
    cell.add(gdstk.rectangle((0.0, -0.5 * w_wg), (20.0, +0.5 * w_wg), layer=lyr_core[0], datatype=lyr_core[1]))
    cell.add(gdstk.rectangle((20.0, -3.0), (36.0, +3.0), layer=lyr_core[0], datatype=lyr_core[1]))
    for sgn in (+1.0, -1.0):
        ya = sgn * 0.5 * y_sep
        u = np.linspace(0.0, 1.0, 24)
        pts_in = [(36.0 + 19.0 * t, sgn * 1.5 + (ya - sgn * 1.5) * 0.5 * (1.0 - np.cos(math.pi * t))) for t in u]
        pts_out = [(x1 + 19.0 * t, ya + (sgn * 1.5 - ya) * 0.5 * (1.0 - np.cos(math.pi * t))) for t in u]
        for p in gdstk.FlexPath(pts_in, w_wg, layer=lyr_core[0], datatype=lyr_core[1]).to_polygons():
            cell.add(p)
        cell.add(gdstk.rectangle((x0, ya - 0.5 * w_wg), (x1, ya + 0.5 * w_wg), layer=lyr_core[0], datatype=lyr_core[1]))
        for p in gdstk.FlexPath(pts_out, w_wg, layer=lyr_core[0], datatype=lyr_core[1]).to_polygons():
            cell.add(p)

    x_end = x1 + 55.0
    cell.add(gdstk.rectangle((x1 + 19.0, -3.0), (x1 + 35.0, +3.0), layer=lyr_core[0], datatype=lyr_core[1]))
    cell.add(gdstk.rectangle((x1 + 35.0, -0.5 * w_wg), (x_end, +0.5 * w_wg), layer=lyr_core[0], datatype=lyr_core[1]))

    ya1 = 0.5 * y_sep
    if tstruct == "doped_si_rib":
        cell.add(gdstk.rectangle((x0, ya1 - 3.0), (x1, ya1 + 3.0), layer=lyr_slab[0], datatype=lyr_slab[1]))
        cell.add(gdstk.rectangle((x0, ya1 + 0.5 * w_wg + 0.15), (x1, ya1 + 0.5 * w_wg + 1.2), layer=lyr_heat[0], datatype=lyr_heat[1]))
        cell.add(gdstk.rectangle((x0, ya1 - 0.5 * w_wg - 1.2), (x1, ya1 - 0.5 * w_wg - 0.15), layer=lyr_heat[0], datatype=lyr_heat[1]))
    else:
        yc_h = ya1 + float(cfg["heater_offset_um"])
        cell.add(gdstk.rectangle((x0, yc_h - 0.5 * w_heat), (x1, yc_h + 0.5 * w_heat), layer=lyr_heat[0], datatype=lyr_heat[1]))

    # Deep Isolation Air Trenches / XeF2 Release Openings (Layer 40/0) flanking heated Arm 1!
    if tstruct in ("deep_trench", "undercut_suspended"):
        off_t = float(cfg["trench_edge_offset_um"])
        wt_t = min(float(cfg["trench_width_um"]), max(y_sep - off_t - 2.5, 2.0))
        cell.add(gdstk.rectangle((x0 + 4.0, ya1 + off_t), (x1 - 4.0, ya1 + off_t + wt_t), layer=lyr_tr[0], datatype=lyr_tr[1]))
        cell.add(gdstk.rectangle((x0 + 4.0, ya1 - off_t - wt_t), (x1 - 4.0, ya1 - off_t), layer=lyr_tr[0], datatype=lyr_tr[1]))

    # Heater Contact Vias + Low-Resistance Au/Al Contact Pads (`11/0`)
    y_pad = ya1 + 32.0
    for idx_end, xv_c in enumerate((x0 + 3.0, x1 - 3.0)):
        cell.add(gdstk.rectangle((xv_c - 3.0, ya1 - 2.0), (xv_c + 3.0, y_pad), layer=lyr_m1[0], datatype=lyr_m1[1]))
        cell.add(gdstk.rectangle((xv_c - 1.5, ya1 - 1.0), (xv_c + 1.5, ya1 + 1.0), layer=lyr_via[0], datatype=lyr_via[1]))
        xp = x0 + idx_end * max(l_vis, 110.0)
        cell.add(gdstk.rectangle((xp - 38.0, y_pad), (xp + 38.0, y_pad + 76.0), layer=lyr_m1[0], datatype=lyr_m1[1]))
        cell.add(gdstk.Label(f"HEAT_{'POS' if idx_end == 0 else 'GND'}", (xp - 24.0, y_pad + 34.0), layer=lyr_txt[0], texttype=lyr_txt[1]))

    cell.add(gdstk.Label(
        f"Thermal ({tstruct}) P_pi={mm['P_pi_mW']:.2f}mW R={mm['R_heater_ohm']:.0f}Ohm t_rise={mm['t_rise_10_90_us']:.1f}us Xtalk@{mm['neighbor_pitch_um']:.0f}um={mm['crosstalk_at_neighbor_pct']:.1f}%",
        (x0 - 20.0, -0.5 * y_sep - 18.0),
        layer=lyr_txt[0], texttype=lyr_txt[1],
    ))

    lib.write_gds(str(out_p))
    lyp_p = out_p.with_suffix(".lyp")
    write_klayout_layer_props(lyp_p)
    used_layers = sorted({int(poly.layer) for poly in cell.polygons})
    return {
        "gds_path": str(out_p),
        "lyp_path": str(lyp_p),
        "cell_name": cell_name,
        "layers": used_layers,
        "footprint_um": [round(max(x_end, x0 + l_vis + 60.0), 2), round(y_pad + 85.0 + 0.5 * y_sep + 25.0, 2)],
    }


def export_tcad_data_bundle(tcad_result: dict, out_dir: str | Path) -> dict:
    """Export CSV curves + Lumerical INTERCONNECT compact script (`.zip`) for PN TCAD or Thermal FEM."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    mode = str(tcad_result.get("mode", "pn_doping_tcad"))

    if mode == "pn_doping_tcad":
        st = tcad_result["states"]
        csv_path = out_root / "MicroRing_Si_PN_TCAD_Sweep.csv"
        arr = np.column_stack([
            [r["voltage_v"] for r in st],
            [r["cj_ff_per_um"] for r in st],
            [r["r_series_ohm_cm"] for r in st],
            [r["delta_neff_real"] for r in st],
            [r["alpha_fca_db_per_cm"] for r in st],
            [r["v_pi_l_secant_vcm"] for r in st],
            [r["f_rc_lumped_ghz"] for r in st],
        ])
        np.savetxt(
            csv_path, arr, delimiter=",",
            header="voltage_V,Cj_fF_per_um,R_series_ohm_cm,delta_neff_real,alpha_FCA_dB_per_cm,V_pi_L_Vcm,f_RC_GHz",
            comments="", fmt="%.6e",
        )
        omm_path = out_root / "MicroRing_Si_PN_OMM.txt"
        omm_lines = ["# Voltage_V   Delta_neff_real   Attenuation_dB_per_m"] + [
            f"{r['voltage_v']:.4f}   {r['delta_neff_real']:+.7e}   {r['alpha_fca_db_per_cm']*100.0:.4f}" for r in st
        ]
        omm_path.write_text("\n".join(omm_lines) + "\n", encoding="utf-8")
        zip_path = (out_root.parent if out_root.name.startswith("tcad_") else out_root) / "MicroRing_Si_PN_TCAD_Bundle.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
            zf.write(omm_path, arcname=omm_path.name)
        return {"csv_path": str(csv_path), "zip_path": str(zip_path), "zip_name": zip_path.name}

    tr = tcad_result["transient"]
    xt = tcad_result["crosstalk"]
    csv_tr = out_root / "MicroRing_Thermal_Transient_Step.csv"
    np.savetxt(
        csv_tr, np.column_stack([tr["time_us"], tr["delta_T_core_K"]]),
        delimiter=",", header="time_us,delta_T_core_K", comments="", fmt="%.5f",
    )
    csv_xt = out_root / "MicroRing_Thermal_Lateral_Crosstalk.csv"
    np.savetxt(
        csv_xt, np.column_stack([xt["distance_um"], xt["delta_T_K_at_Ppi"], xt["normalized_pct"]]),
        delimiter=",", header="lateral_distance_um,delta_T_K_at_Ppi,crosstalk_pct", comments="", fmt="%.5f",
    )
    zip_path = (out_root.parent if out_root.name.startswith("tcad_") else out_root) / "MicroRing_Thermal_FEM_Bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_tr, arcname=csv_tr.name)
        zf.write(csv_xt, arcname=csv_xt.name)
    return {"csv_path": str(csv_tr), "zip_path": str(zip_path), "zip_name": zip_path.name}
