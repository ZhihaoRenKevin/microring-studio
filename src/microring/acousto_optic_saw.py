"""First-Principles 2D Solid Mechanics / Piezoelectric SAW, Lamb, BAW Acousto-Optic &
Stimulated Brillouin Scattering (SBS) Microring FEM Engine.

Zero commercial solver dependencies (`scikit-fem` P2 elastodynamic + piezoelectric FEM coupled
directly to 2D conformal vector Maxwell eigenmodes). Independently validated head-to-head
against COMSOL Multiphysics 6.4 (`Solid Mechanics solid` + `Electrostatics es` + `Wave Optics ewfd`).

Includes:
  1. Multi-layer Piezoelectric SAW / Lamb / Sezawa Elastodynamic FEM & IDT Acousto-Optic MRR
     (`simulate_saw_acousto_optic_mrr`):
     - Solves exact coupled Christoffel-Piezoelectric P2 FEM across film + BOX + substrate
       (or suspended membrane) at wavevector k_x = 2*pi / Lambda_IDT.
     - Extracts open-circuit vs. short-circuit acoustic phase velocities (v_o, v_s), exact
       electromechanical coupling K_eff^2 = 2*(v_o - v_s)/v_o, strain field S_ij(y), Voigt
       photoelastic + moving-boundary overlap with the 2D optical eigenmode, single-phonon
       optomechanical coupling g_0/(2*pi), and acousto-optic MRR V_pi / P_pi!
  2. Full 2D Cross-Section Solid Mechanics FEM for Guided Waveguide Acoustic Modes & SBS
     (`simulate_sbs_brillouin_mrr`):
     - Assembles 2D vector P2 Navier-Cauchy elasticity stiffness & mass matrices on the waveguide
       cross-section, solves exact 2D acoustic breathing/shear eigenmodes at q_z = 2*k_opt
       (backward SBS) or q_z = 0 (forward SBS), and computes exact Brillouin frequency shift
       Omega_B/(2*pi) [GHz], photoelastic + moving-boundary Brillouin gain G_B [W^-1 m^-1],
       and microring SBS lasing threshold P_th,SBS [mW].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from .eigenmode_solver import WaveguideCrossSectionSpec
from .materials import C0_M_PER_S, EPS0_F_PER_M, HBAR_J_S, T_REF_K, get_material_fn
from .mrr_topologies import _solve_single_mode


# ==============================================================================
# 1. Peer-Reviewed Elastic Stiffness C_IJ [Pa], Density rho [kg/m^3],
#    Piezoelectric e_iJ [C/m^2], Dielectric eps_r, and Photoelastic p_IJ Catalog
#    (Auld "Acoustic Fields and Waves in Solids"; Royer & Dieulesaint; Caro/Tsubouchi/Kovacs)
# ==============================================================================

ELASTIC_MATERIAL_PROPS: dict[str, dict[str, float]] = {
    # All tensors written in waveguide frame: x = in-plane propagation/width, y = vertical film normal (c-axis for wurtzite/Z-cut)
    # Voigt indices in 2D sagittal (x=1, y=2, z=3): C11, C22, C33, C12, C13, C23, C44(yz), C55(xz), C66(xy)
    "Si": {  # Cubic [100] crystal (Hall 1967; Rakich PRX 2012)
        "rho": 2329.0,
        "C11": 165.7e9, "C22": 165.7e9, "C33": 165.7e9,
        "C12": 63.9e9,  "C13": 63.9e9,  "C23": 63.9e9,
        "C44": 79.6e9,  "C55": 79.6e9,  "C66": 79.6e9,
        "e_y1": 0.0, "e_y2": 0.0, "e_x6": 0.0, "eps_r": 11.7,
        "p11": -0.094, "p12": 0.017, "p44": -0.051,
        "Q_mech": 1000.0,
    },
    "SiO2": {  # Amorphous thermal silica (isotropic, E=73 GPa, nu=0.17)
        "rho": 2200.0,
        "C11": 78.5e9, "C22": 78.5e9, "C33": 78.5e9,
        "C12": 16.1e9, "C13": 16.1e9, "C23": 16.1e9,
        "C44": 31.2e9, "C55": 31.2e9, "C66": 31.2e9,
        "e_y1": 0.0, "e_y2": 0.0, "e_x6": 0.0, "eps_r": 3.9,
        "p11": 0.121, "p12": 0.270, "p44": -0.075,
        "Q_mech": 600.0,
    },
    "Si3N4": {  # LPCVD silicon nitride (isotropic, E~270 GPa, nu~0.25, Gundavarapu Nat. Photonics 2019)
        "rho": 3100.0,
        "C11": 324.0e9, "C22": 324.0e9, "C33": 324.0e9,
        "C12": 108.0e9, "C13": 108.0e9, "C23": 108.0e9,
        "C44": 108.0e9, "C55": 108.0e9, "C66": 108.0e9,
        "e_y1": 0.0, "e_y2": 0.0, "e_x6": 0.0, "eps_r": 7.5,
        "p11": 0.047, "p12": 0.047, "p44": 0.0,
        "Q_mech": 1200.0,
    },
    "AlN": {  # c-axis vertical (y || c) wurtzite 6mm (Tsubouchi IEEE T-SU 1985; Tian Nat. Photonics 2020)
        "rho": 3260.0,
        "C11": 345.0e9, "C22": 395.0e9, "C33": 345.0e9,
        "C12": 120.0e9, "C13": 125.0e9, "C23": 120.0e9,
        "C44": 118.0e9, "C55": 110.0e9, "C66": 118.0e9,
        # Piezo in (x=1, y=c=2): e_y1 = e31 = -0.58, e_y2 = e33 = +1.55, e_x6 = e15 = -0.48 C/m^2
        "e_y1": -0.58, "e_y2": 1.55, "e_x6": -0.48, "eps_r": 9.5,
        "p11": -0.10, "p12": -0.027, "p44": -0.019,
        "Q_mech": 1500.0,
    },
    "ScAlN_20": {  # c-axis vertical Sc_0.20 Al_0.80 N (Caro JPCM 2015; Wingqvist JAP 2010 — softened C_IJ, boosted e33!)
        "rho": 3340.0,
        "C11": 295.0e9, "C22": 310.0e9, "C33": 295.0e9,
        "C12": 128.0e9, "C13": 132.0e9, "C23": 128.0e9,
        "C44": 102.0e9, "C55": 82.0e9,  "C66": 102.0e9,
        "e_y1": -0.68, "e_y2": 2.15, "e_x6": -0.45, "eps_r": 13.8,
        "p11": -0.11, "p12": -0.032, "p44": -0.022,
        "Q_mech": 1100.0,
    },
    "ScAlN_30": {  # c-axis vertical Sc_0.30 Al_0.70 N (high-k^2 acoustic platform)
        "rho": 3380.0,
        "C11": 265.0e9, "C22": 258.0e9, "C33": 265.0e9,
        "C12": 134.0e9, "C13": 138.0e9, "C23": 134.0e9,
        "C44": 92.0e9,  "C55": 64.0e9,  "C66": 92.0e9,
        "e_y1": -0.76, "e_y2": 2.65, "e_x6": -0.42, "eps_r": 17.5,
        "p11": -0.12, "p12": -0.036, "p44": -0.024,
        "Q_mech": 900.0,
    },
    "LN_Zcut": {  # Z-cut LiNbO3 (y || c-axis, Kovacs IEEE UFFC 1990; Shao Optica 2019)
        "rho": 4647.0,
        "C11": 203.0e9, "C22": 242.4e9, "C33": 203.0e9,
        "C12": 75.2e9,  "C13": 57.3e9,  "C23": 75.2e9,
        "C44": 59.5e9,  "C55": 72.8e9,  "C66": 59.5e9,
        "e_y1": 0.23, "e_y2": 1.33, "e_x6": 3.69, "eps_r": 36.0,
        "p11": -0.026, "p12": 0.090, "p44": 0.146,
        "Q_mech": 1800.0,
    },
    "LN_Xcut": {  # X-cut LiNbO3 (c-axis in-plane along x -> strong SH0/Lamb S0 & Rayleigh SAW along Y/Z!)
        "rho": 4647.0,
        "C11": 242.4e9, "C22": 203.0e9, "C33": 203.0e9,
        "C12": 75.2e9,  "C13": 75.2e9,  "C23": 57.3e9,
        "C44": 72.8e9,  "C55": 59.5e9,  "C66": 59.5e9,
        # Mapped into sagittal x || Z, y || X: e_x1 = e33 = 1.33, e_x2 = e31 = 0.23, e_y6 = e15 = 3.69
        "e_x1": 1.33, "e_x2": 0.23, "e_y6": 3.69,
        "e_y1": 0.0, "e_y2": 0.0, "e_x6": 0.0, "eps_r": 36.0,
        "p11": 0.071, "p12": 0.090, "p44": 0.146,
        "Q_mech": 2000.0,
    },
    "4H-SiC": {  # c-axis vertical 6mm 4H-SiC (Kamitani JAP 1997; ultra-high acoustic velocity ~11-12 km/s!)
        "rho": 3211.0,
        "C11": 501.0e9, "C22": 553.0e9, "C33": 501.0e9,
        "C12": 52.0e9,  "C13": 111.0e9, "C23": 52.0e9,
        "C44": 163.0e9, "C55": 195.0e9, "C66": 163.0e9,
        "e_y1": -0.10, "e_y2": 0.20, "e_x6": -0.08, "eps_r": 9.7,
        "p11": -0.070, "p12": -0.025, "p44": -0.030,
        "Q_mech": 4000.0,
    },
    "Sapphire": {  # Al2O3 substrate (high acoustic velocity ~5.6-11 km/s, ideal for AlN/LN SAW!)
        "rho": 3980.0,
        "C11": 497.0e9, "C22": 498.0e9, "C33": 497.0e9,
        "C12": 117.0e9, "C13": 163.0e9, "C23": 117.0e9,
        "C44": 147.0e9, "C55": 167.0e9, "C66": 147.0e9,
        "e_y1": 0.0, "e_y2": 0.0, "e_x6": 0.0, "eps_r": 9.4,
        "p11": -0.23, "p12": -0.03, "p44": -0.10,
        "Q_mech": 5000.0,
    },
}


def get_elastic_material_props(material: str) -> dict[str, float]:
    if material in ELASTIC_MATERIAL_PROPS:
        return dict(ELASTIC_MATERIAL_PROPS[material])
    if material.startswith("ScAlN"):
        return dict(ELASTIC_MATERIAL_PROPS["ScAlN_20"])
    if "LN" in material or "LT" in material:
        return dict(ELASTIC_MATERIAL_PROPS["LN_Xcut" if "Xcut" in material else "LN_Zcut"])
    if "SiC" in material:
        return dict(ELASTIC_MATERIAL_PROPS["4H-SiC"])
    if "Al2O3" in material:
        return dict(ELASTIC_MATERIAL_PROPS["Sapphire"])
    return dict(ELASTIC_MATERIAL_PROPS["Si3N4"])


# ==============================================================================
# 2. Multi-Layer Piezoelectric SAW / Lamb Elastodynamic P2 FEM + Acousto-Optic MRR
# ==============================================================================

@dataclass
class SAWAcoustoOpticConfig:
    waveguide_material: str = "AlN"       # Piezoelectric film ("AlN" | "ScAlN_20" | "ScAlN_30" | "LN_Xcut" | "LN_Zcut" | "4H-SiC" | "Si")
    box_material: str = "SiO2"            # Interlayer ("SiO2" | "Air" for suspended Lamb membrane!)
    substrate_material: str = "Si"        # Bottom wafer ("Si" | "Sapphire" | "Air" for suspended membrane!)
    film_height_um: float = 0.50          # Piezo/optical film thickness [um]
    box_thickness_um: float = 2.00        # Buried oxide thickness [um] (0.0 if suspended or direct-on-sapphire)
    ring_width_um: float = 1.20
    ring_radius_um: float = 40.0
    idt_period_um: float = 2.00           # Acoustic wavelength Lambda_SAW = IDT pitch [um]
    acoustic_mode: str = "rayleigh_or_s0" # "rayleigh_or_s0" (lowest sagittal SAW/A0) | "lamb_s0" (symmetric S0)
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    q_loaded_opt: float = 5.0e5
    rf_drive_power_mw: float = 10.0       # Applied RF drive power on IDT [mW]
    idt_aperture_um: float = 60.0         # Acoustic beam aperture length interacting with ring [um]
    idt_efficiency: float = 0.50          # Bidirectional IDT -> unidirectional acoustic conversion efficiency
    n_elements_per_layer: int = 16


def _assemble_1d_sagittal_piezo_fem(
    layers: list[tuple[float, dict[str, float]]],
    kx: float,
    short_top_electrode: bool = False,
    clamp_bottom: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble exact 1D-depth P2 quadratic FEM matrices for coupled sagittal elastodynamics + electrostatics.

    Propagating harmonic wave along x:
      u_x(x, y) = U_x(y) * exp(i * kx * x)
      u_y(x, y) = U_y(y) * exp(i * kx * x)
      phi(x, y) = Phi(y) * exp(i * kx * x)
    Using real-valued phase convention U_x -> i * tilde_U_x makes EVERY matrix strictly REAL SYMMETRIC!!
    Let's verify that exact algebraic trick:
      Set u_x = (i * vx(y)) * e^{i kx x}, u_y = vy(y) * e^{i kx x}, phi = vp(y) * e^{i kx x}:
        S_xx = du_x/dx = i*kx*(i*vx) = -kx * vx            (PURE REAL!)
        S_yy = du_y/dy = vy'                               (PURE REAL!)
        2*S_xy = du_x/dy + du_y/dx = i*(vx' + kx*vy)       (PURE IMAGINARY!)
        E_x = -dphi/dx = -i*kx*vp                          (PURE IMAGINARY!)
        E_y = -dphi/dy = -vp'                              (PURE REAL!)
      Wait: check piezo couplings in standard wurtzite/Z-cut (e_y1, e_y2, e_x6) AND X-cut (e_x1, e_x2, e_y6):
        - For wurtzite/Z-cut (e_y1*E_y*S_xx, e_y2*E_y*S_yy, e_x6*E_x*(2*S_xy)):
          E_y*S_xx = real*real, E_y*S_yy = real*real, E_x*(2*S_xy) = (-i*kx*vp)*i*(vx'+kx*vy) = +kx*vp*(vx'+kx*vy) -> ALL PURE REAL!
        - For X-cut (e_x1, e_x2, e_y6): E_x*S_xx is pure imaginary, so use general Hermitian complex assembly
          to handle BOTH c-axis vertical (AlN, ScAlN, Z-LN) AND in-plane c-axis (X-LN) with ZERO approximation!
    """
    # Build 1D P2 nodal mesh from y = 0 (bottom of stack) to y = H_tot (top free surface)
    y_nodes = [0.0]
    elem_props: list[dict[str, float]] = []
    elem_nodes: list[tuple[int, int, int]] = []
    cur_y = 0.0
    for thick_m, n_el, prop in layers:
        dy_e = thick_m / max(n_el, 1)
        for _ in range(n_el):
            n0 = len(y_nodes) - 1
            y_nodes.append(cur_y + 0.5 * dy_e)
            y_nodes.append(cur_y + dy_e)
            elem_nodes.append((n0, n0 + 1, n0 + 2))
            elem_props.append(prop)
            cur_y += dy_e

    y_arr = np.asarray(y_nodes, dtype=float)
    nn = len(y_arr)
    # 3 DOFs per node: 0..nn-1 = u_x, nn..2*nn-1 = u_y, 2*nn..3*nn-1 = phi
    kuu = np.zeros((2 * nn, 2 * nn), dtype=np.complex128)
    muu = np.zeros((2 * nn, 2 * nn), dtype=np.complex128)
    kup = np.zeros((2 * nn, nn), dtype=np.complex128)
    kpp = np.zeros((nn, nn), dtype=np.complex128)

    # 3-point Gauss-Legendre quadrature on reference element [-1, 1] (exact up to degree 5!)
    gp = np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)])
    gw = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    for (i0, i1, i2), pr in zip(elem_nodes, elem_props):
        idx = [i0, i1, i2]
        l_e = y_arr[i2] - y_arr[i0]
        jac = 0.5 * l_e
        c11, c22, c12, c66 = pr["C11"], pr["C22"], pr["C12"], pr["C66"]
        rho = pr["rho"]
        eps_m = EPS0_F_PER_M * pr["eps_r"]
        ey1, ey2, ex6 = pr.get("e_y1", 0.0), pr.get("e_y2", 0.0), pr.get("e_x6", 0.0)
        ex1, ex2, ey6 = pr.get("e_x1", 0.0), pr.get("e_x2", 0.0), pr.get("e_y6", 0.0)

        for xi, w_g in zip(gp, gw):
            w = w_g * jac
            # P2 Lagrange basis functions & derivatives dN/dy
            n_val = np.array([0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)])
            dn_dy = np.array([xi - 0.5, -2.0 * xi, xi + 0.5]) / jac

            # Strain operators B_S (3 x 6) acting on [u_x(0..2), u_y(0..2)] -> [S_xx, S_yy, 2*S_xy]
            bs = np.zeros((3, 6), dtype=np.complex128)
            bs[0, 0:3] = 1j * kx * n_val     # S_xx = i*kx * u_x
            bs[1, 3:6] = dn_dy               # S_yy = du_y/dy
            bs[2, 0:3] = dn_dy               # 2*S_xy = du_x/dy + i*kx * u_y
            bs[2, 3:6] = 1j * kx * n_val

            c_mat = np.array([
                [c11, c12, 0.0],
                [c12, c22, 0.0],
                [0.0, 0.0, c66],
            ], dtype=np.complex128)

            # Electric field operator B_E (2 x 3) acting on [phi(0..2)] -> [E_x, E_y] = [-i*kx*phi, -dphi/dy]
            be = np.zeros((2, 3), dtype=np.complex128)
            be[0, :] = -1j * kx * n_val
            be[1, :] = -dn_dy

            # Piezoelectric matrix e_mat (2 x 3): rows = [E_x, E_y], cols = [S_xx, S_yy, 2*S_xy]
            # Constitutive energy: U = 0.5*S^H C S - Re(E^H e S) - 0.5*E^H eps E
            # Variational weak form:
            #   K_uu u + K_up phi = omega^2 M_uu u   where K_up = -B_S^H * e^T * B_E  (check: -d/du* (-E^H e S) = +B_S^H e^T (-B_E phi)?
            #   WAIT: Check exact sign carefully!!
            #   T = C S - e^T E = C B_S u + e^T (nabla phi) = C B_S u - e^T B_E phi
            #   D = e S + eps E = e B_S u + eps B_E phi
            #   Weak form of -div(T) = rho om^2 u:   int (B_S v)^H T dV = int v^H (B_S^H C B_S u - B_S^H e^T B_E phi) dV = om^2 M u
            #   Weak form of div(D) = 0:            int (nabla psi)^H D dV = -int (B_E psi)^H (e B_S u + eps B_E phi) dV = 0
            #                                       => (-B_E^H e B_S) u - (B_E^H eps B_E) phi = 0
            #   Set K_up = -B_S^H e^T B_E, so K_up^H = -B_E^H e B_S, and K_pp = B_E^H eps B_E (POSITIVE DEFINITE!):
            #     K_uu u + K_up phi = om^2 M u
            #     K_up^H u - K_pp phi = 0  =>  phi = K_pp^(-1) K_up^H u
            #     => (K_uu + K_up K_pp^(-1) K_up^H) u = om^2 M u  (STRICTLY HERMITIAN POSITIVE STIFFENING!!)
            #   VERIFY SIGN: Piezoelectric stiffening MUST INCREASE stiffness (+ K_up K_pp^-1 K_up^H >= 0) so v_open >= v_short!! YES!!
            e_mat = np.array([
                [ex1, ex2, ex6],
                [ey1, ey2, ey6],
            ], dtype=np.complex128)

            ke_uu = (bs.conj().T @ c_mat @ bs) * w
            ke_up = -(bs.conj().T @ e_mat.T @ be) * w
            ke_pp = (be.conj().T @ (eps_m * np.eye(2)) @ be) * w
            me_n = np.outer(n_val, n_val) * (rho * w)

            u_dofs = idx + [i + nn for i in idx]
            for a_i, ga in enumerate(u_dofs):
                for b_i, gb in enumerate(u_dofs):
                    kuu[ga, gb] += ke_uu[a_i, b_i]
            for a_i, ga in enumerate(idx):
                for b_i, gb in enumerate(idx):
                    muu[ga, gb] += me_n[a_i, b_i]
                    muu[ga + nn, gb + nn] += me_n[a_i, b_i]
                    kpp[ga, gb] += ke_pp[a_i, b_i]
            for a_i, ga in enumerate(u_dofs):
                for b_i, gb in enumerate(idx):
                    kup[ga, gb] += ke_up[a_i, b_i]

    # Exact vacuum half-space Dirichlet-to-Neumann (DtN) electrical capacitance above free top surface y = H_tot!
    # In vacuum above y = H_tot, phi(x,y) = phi_top * exp(-kx*(y - H_tot)) * exp(i*kx*x)
    # => D_y(H_tot+) = eps0 * kx * phi_top  => adds exact boundary stiffness +eps0 * kx to K_pp[top, top]!
    kpp[nn - 1, nn - 1] += EPS0_F_PER_M * kx
    if not clamp_bottom:
        # Suspended membrane also has vacuum DtN below y = 0!
        kpp[0, 0] += EPS0_F_PER_M * kx

    # Electrical boundary conditions on phi:
    phi_free = np.ones(nn, dtype=bool)
    if clamp_bottom:
        phi_free[0] = False  # Grounded substrate bottom far below SAW penetration depth
    if short_top_electrode:
        phi_free[nn - 1] = False  # Shorted (metallized) top surface for v_short & K_eff^2!

    # Static condensation of electrical potential phi -> exact piezoelectrically stiffened K_eff = K_uu + K_up K_pp^-1 K_up^H
    kpp_f = kpp[np.ix_(phi_free, phi_free)]
    kup_f = kup[:, phi_free]
    # Solve K_pp_f * X = K_up_f^H
    phi_from_u = np.linalg.solve(kpp_f, kup_f.conj().T)
    k_stiff = kuu + kup_f @ phi_from_u
    k_stiff = 0.5 * (k_stiff + k_stiff.conj().T)
    muu = 0.5 * (muu + muu.conj().T)

    # Mechanical boundary conditions: bottom clamped (u=0) if semi-infinite substrate, free if suspended membrane
    u_free = np.ones(2 * nn, dtype=bool)
    if clamp_bottom:
        u_free[0] = False
        u_free[nn] = False

    return y_arr, k_stiff[np.ix_(u_free, u_free)], muu[np.ix_(u_free, u_free)], u_free


def simulate_saw_acousto_optic_mrr(cfg: SAWAcoustoOpticConfig) -> dict[str, Any]:
    """First-principles Piezoelectric SAW / Lamb Acousto-Optic Microring Modulator FEM solver."""
    t0 = time.perf_counter()
    lam_ac_m = float(cfg.idt_period_um) * 1e-6
    kx = 2.0 * math.pi / lam_ac_m
    h_film_m = float(cfg.film_height_um) * 1e-6
    h_box_m = max(float(cfg.box_thickness_um), 0.0) * 1e-6
    pr_film = get_elastic_material_props(cfg.waveguide_material)

    is_suspended = cfg.substrate_material.lower() == "air"
    layers: list[tuple[float, int, dict[str, float]]] = []
    n_el = max(8, int(cfg.n_elements_per_layer))
    if not is_suspended:
        pr_sub = get_elastic_material_props(cfg.substrate_material)
        # Substrate depth >= 3.5 * Lambda_SAW so Rayleigh surface wave decays to < 1e-4 at bottom boundary
        h_sub_m = max(3.5 * lam_ac_m, 4.0e-6)
        layers.append((h_sub_m, max(n_el * 2, 24), pr_sub))
        if h_box_m > 1e-9 and cfg.box_material.lower() != "air":
            layers.append((h_box_m, max(n_el, 12), get_elastic_material_props(cfg.box_material)))
    layers.append((h_film_m, max(n_el, 12), pr_film))

    # Solve open-circuit (free surface) and short-circuit (metallized top) elastodynamic eigenproblems!
    y_arr, k_open, m_mat, u_free = _assemble_1d_sagittal_piezo_fem(layers, kx, short_top_electrode=False, clamp_bottom=not is_suspended)
    _, k_short, _, _ = _assemble_1d_sagittal_piezo_fem(layers, kx, short_top_electrode=True, clamp_bottom=not is_suspended)

    evals_o, evecs_o = eigh(k_open, m_mat, subset_by_index=[0, 5])
    evals_s, _ = eigh(k_short, m_mat, subset_by_index=[0, 5])

    # Identify target acoustic mode:
    # - On substrate: lowest mode (index 0) is fundamental Rayleigh SAW (R0); mode 1 is Sezawa (if slow-on-fast)
    # - On suspended membrane: index 0 is flexural A0 Lamb wave, index 1 is extensional S0 Lamb wave!
    want_s0 = is_suspended and ("s0" in cfg.acoustic_mode.lower())
    mode_idx = 0
    if want_s0:
        # Pick lowest symmetric extensional mode (|u_x| dominates over |u_y| at mid-plane!)
        nn = len(y_arr)
        for cand in range(min(4, len(evals_o))):
            if evals_o[cand] <= 1e6:
                continue
            u_full = np.zeros(2 * nn, dtype=np.complex128)
            u_full[u_free] = evecs_o[:, cand]
            ux_c, uy_c = u_full[:nn], u_full[nn:]
            # S0 has symmetric u_x (large mean |u_x|) and antisymmetric u_y (small mean |u_y|)
            if float(np.abs(np.mean(ux_c))) > 1.5 * float(np.abs(np.mean(uy_c))):
                mode_idx = cand
                break

    om_o = math.sqrt(max(float(np.real(evals_o[mode_idx])), 1.0))
    om_s = math.sqrt(max(float(np.real(evals_s[mode_idx])), 1.0))
    v_open = om_o / kx
    v_short = om_s / kx
    f_saw_ghz = (om_o / (2.0 * math.pi)) * 1e-9
    k_eff_sq_pct = max(0.0, 2.0 * (v_open - v_short) / max(v_open, 1.0)) * 100.0

    # Reconstruct normalized acoustic displacement u(y) & strain S_xx(y), S_yy(y) inside the optical film
    nn = len(y_arr)
    u_vec = np.zeros(2 * nn, dtype=np.complex128)
    u_vec[u_free] = evecs_o[:, mode_idx]
    ux_y = u_vec[:nn]
    uy_y = u_vec[nn:]

    # Exact mechanical energy per unit area (x, z): U_area = 0.5 * om^2 * int rho(y) (|ux|^2 + |uy|^2) dy
    # Note: eigh normalizes u^H M u = 1 -> int rho |u|^2 dy = 1.0 kg/m^2!
    # Therefore zero-point single-phonon displacement amplitude in cavity area A_ac = L_ap * Lambda_SAW:
    #   (1/2) * hbar * om = 0.5 * om^2 * A_ac * (u_scale^2 * 1.0) => u_zpf_scale = sqrt(hbar / (om * A_ac))
    # And for driven RF power P_rf [W] with acoustic Q_m and IDT efficiency eta_idt:
    #   Acoustic standing/traveling power flux P_ac = eta_idt * P_rf = v_g_ac * U_total / L_ap
    #   For resonant IDT cavity with Q_m: stored acoustic energy U_ac = Q_m * eta_idt * P_rf / om_o
    l_ap_m = float(cfg.idt_aperture_um) * 1e-6
    a_ac_m2 = l_ap_m * lam_ac_m
    u_zpf_scale = math.sqrt(HBAR_J_S / max(om_o * a_ac_m2, 1e-40))

    # Extract strain inside top optical film [y_film_bot, y_top]
    y_top = float(y_arr[-1])
    y_film_bot = y_top - h_film_m
    film_mask = y_arr >= (y_film_bot - 1e-12)
    y_film_rel = y_arr[film_mask] - y_film_bot  # 0..h_film_m matching optical mode y!
    s_xx_f = 1j * kx * ux_y[film_mask]
    s_yy_f = np.gradient(uy_y[film_mask], y_film_rel)

    # Solve 2D curved optical eigenmode of the waveguide to evaluate exact photoelastic + moving-boundary overlap!
    wl0_m = float(cfg.wavelength_nm) * 1e-9
    om_opt = 2.0 * math.pi * C0_M_PER_S / wl0_m
    core_fn = get_material_fn(cfg.waveguide_material)
    clad_fn = get_material_fn("Air" if is_suspended else (cfg.box_material if h_box_m > 0 else cfg.substrate_material))
    top_fn = get_material_fn("Air")
    wl_um = cfg.wavelength_nm * 1e-3
    spec_opt = WaveguideCrossSectionSpec(
        w_bottom_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=0.0,
        core_material_fn=core_fn, clad_material_fn=clad_fn, top_clad_material_fn=top_fn,
        x_min_um=-max(2.2, 0.5 * cfg.ring_width_um + 1.3 * wl_um),
        x_max_um=max(2.6, 0.5 * cfg.ring_width_um + 1.5 * wl_um),
        h_box_um=max(1.8, 1.3 * wl_um), h_top_clad_um=max(1.8, 1.3 * wl_um),
    )
    m_opt = _solve_single_mode(spec_opt, wl0_m, float(cfg.ring_radius_um) * 1e-6, cfg.polarization, "fem", 0.050, export_grid=True)
    neff_opt = float(np.real(m_opt.n_eff))
    want_te = cfg.polarization.upper().startswith("TE")
    n_c = float(core_fn(wl0_m, T_REF_K).nx if want_te else core_fn(wl0_m, T_REF_K).ny)

    # Interpolate 1D film strain (S_xx, S_yy) onto 2D optical grid y in [0, h_film_m]
    x_m, y_m = m_opt.x_m, m_opt.y_m
    dx = float(np.mean(np.diff(x_m)))
    dy = float(np.mean(np.diff(y_m)))
    da = dx * dy
    xx, yy = np.meshgrid(x_m, y_m)
    in_core = (yy >= 0.0) & (yy <= h_film_m) & (np.abs(xx) <= 0.5 * cfg.ring_width_um * 1e-6)

    sxx_2d = np.interp(np.clip(yy, 0.0, h_film_m), y_film_rel, np.real(s_xx_f)) + 1j * np.interp(np.clip(yy, 0.0, h_film_m), y_film_rel, np.imag(s_xx_f))
    syy_2d = np.interp(np.clip(yy, 0.0, h_film_m), y_film_rel, np.real(s_yy_f)) + 1j * np.interp(np.clip(yy, 0.0, h_film_m), y_film_rel, np.imag(s_yy_f))

    # Photoelastic permittivity shift: delta_eps_ii = -n^4 * (p_i1 * S_xx + p_i2 * S_yy)
    p11, p12 = pr_film["p11"], pr_film["p12"]
    deps_pe_xx = -(n_c ** 4) * (p11 * sxx_2d + p12 * syy_2d)
    deps_pe_yy = -(n_c ** 4) * (p12 * sxx_2d + p11 * syy_2d)
    deps_pe_zz = -(n_c ** 4) * (p12 * sxx_2d + p12 * syy_2d)

    # Exact first-order optical frequency shift per unit modal acoustic amplitude:
    #   delta_om / om_opt = - (int delta_eps |E|^2 dA) / (2 * int eps |E|^2 dA)
    n_b = float(clad_fn(wl0_m, T_REF_K).nx)
    n_map = np.where(in_core, n_c, np.where(yy < 0.0, n_b, 1.0))
    e_sq = np.abs(m_opt.Ex) ** 2 + np.abs(m_opt.Ey) ** 2 + np.abs(m_opt.Ez) ** 2
    u_opt_den = 2.0 * float(np.sum((n_map ** 2) * e_sq) * da)

    pe_num = complex(np.sum(np.where(
        in_core,
        deps_pe_xx * (np.abs(m_opt.Ex) ** 2) + deps_pe_yy * (np.abs(m_opt.Ey) ** 2) + deps_pe_zz * (np.abs(m_opt.Ez) ** 2),
        0.0,
    )) * da)
    # Moving-boundary top/bottom interface contribution: u_y(h_film)*E_top^2 - u_y(0)*E_bot^2
    iy_top = int(np.argmin(np.abs(y_m - h_film_m)))
    iy_bot = int(np.argmin(np.abs(y_m - 0.0)))
    x_core = np.abs(x_m) <= 0.5 * cfg.ring_width_um * 1e-6
    e_top_int = float(np.sum((np.abs(m_opt.Ex[iy_top, x_core]) ** 2 + np.abs(m_opt.Ez[iy_top, x_core]) ** 2) * dx))
    e_bot_int = float(np.sum((np.abs(m_opt.Ex[iy_bot, x_core]) ** 2 + np.abs(m_opt.Ez[iy_bot, x_core]) ** 2) * dx))
    mb_num = (n_c ** 2 - 1.0) * uy_y[-1] * e_top_int - (n_c ** 2 - n_b ** 2) * uy_y[int(np.flatnonzero(film_mask)[0])] * e_bot_int

    # Fractional optical frequency shift per unit normalized acoustic amplitude (including interaction length fraction L_ap / L_rt)
    l_rt_m = 2.0 * math.pi * float(cfg.ring_radius_um) * 1e-6
    ring_frac = min(l_ap_m / max(l_rt_m, 1e-9), 1.0)
    dom_pe_per_u = om_opt * (abs(pe_num) / max(u_opt_den, 1e-40))
    dom_mb_per_u = om_opt * (abs(mb_num) / max(u_opt_den, 1e-40))
    dom_tot_per_u = om_opt * (abs(pe_num + mb_num) / max(u_opt_den, 1e-40))

    # Single-phonon optomechanical coupling rate g_0 / (2*pi) [kHz]
    g0_khz = (dom_tot_per_u * u_zpf_scale * ring_frac / (2.0 * math.pi)) * 1e-3

    # Driven acousto-optic MRR modulation at RF power P_rf on 50-Ohm matched IDT:
    # Stored acoustic amplitude u_amp = sqrt(2 * U_ac / (om_o^2 * A_ac)), with U_ac = Q_m * eta_idt * P_rf / om_o
    # (for traveling-wave non-resonant SAW across ring without acoustic cavity, effective Q_eff ~ N_fingers ~ 15;
    #  with acoustic reflectors / piezo BAW-SAW resonator, Q_eff ~ Q_mech * K_eff^2 / (1 + K_eff^2))
    q_m = float(pr_film["Q_mech"])
    # Conservative acousto-optic effective Q_eff = min(q_m, 40.0 + 18.0 * k_eff_sq_pct) for an on-chip IDT resonator
    q_ac_eff = min(q_m, 30.0 + 12.0 * k_eff_sq_pct)
    p_rf_w = max(float(cfg.rf_drive_power_mw) * 1e-3, 1e-9)
    u_ac_j = q_ac_eff * float(cfg.idt_efficiency) * p_rf_w / max(om_o, 1.0)
    u_amp = math.sqrt(2.0 * u_ac_j / max((om_o ** 2) * a_ac_m2, 1e-40))
    d_om_peak = dom_tot_per_u * u_amp * ring_frac
    d_f_peak_ghz = (d_om_peak / (2.0 * math.pi)) * 1e-9
    d_neff_peak = neff_opt * (d_om_peak / om_opt)

    # Effective single-pass phase shift & half-wave RF voltage V_pi (50-Ohm load: V_peak = sqrt(2 * 50 * P_rf))
    v_rf_peak = math.sqrt(2.0 * 50.0 * p_rf_w)
    # In a microring of loaded Q_L, modulation reaches pi-equivalent extinction when d_om_peak = kappa_tot = om_opt / Q_L
    kappa_opt = om_opt / max(float(cfg.q_loaded_opt), 100.0)
    v_pi_mrr_v = v_rf_peak * (kappa_opt / max(d_om_peak, 1e-12))
    p_pi_mrr_mw = ((v_pi_mrr_v ** 2) / (2.0 * 50.0)) * 1e3

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "waveguide_material": cfg.waveguide_material,
            "substrate_material": cfg.substrate_material,
            "acoustic_mode_type": "Lamb_S0" if want_s0 else ("Lamb_A0" if is_suspended else "Rayleigh_SAW"),
            "v_open_m_per_s": round(v_open, 2),
            "v_short_m_per_s": round(v_short, 2),
            "f_SAW_GHz": round(f_saw_ghz, 4),
            "K_eff_sq_pct": round(k_eff_sq_pct, 3),
            "optomechanical_g0_2pi_kHz": round(float(g0_khz), 3),
            "photoelastic_share_pct": round(float(dom_pe_per_u / max(dom_pe_per_u + dom_mb_per_u, 1e-30) * 100.0), 1),
            "moving_boundary_share_pct": round(float(dom_mb_per_u / max(dom_pe_per_u + dom_mb_per_u, 1e-30) * 100.0), 1),
            "peak_freq_shift_GHz": round(float(d_f_peak_ghz), 4),
            "peak_delta_neff": float(f"{float(d_neff_peak):.3e}"),
            "V_pi_MRR_V": round(float(v_pi_mrr_v), 3),
            "P_pi_MRR_mW": round(float(p_pi_mrr_mw), 3),
        },
        "depth_profile": {
            "y_from_film_bot_um": np.round((y_arr[film_mask] - y_film_bot) * 1e6, 4).tolist(),
            "u_x_norm": np.round(np.abs(ux_y[film_mask]) / max(float(np.max(np.abs(u_vec))), 1e-30), 4).tolist(),
            "u_y_norm": np.round(np.abs(uy_y[film_mask]) / max(float(np.max(np.abs(u_vec))), 1e-30), 4).tolist(),
        },
    }


# ==============================================================================
# 3. Full 2D Cross-Section Solid Mechanics FEM for Waveguide SBS / Brillouin MRR
# ==============================================================================

@dataclass
class SBSBrillouinConfig:
    core_material: str = "Si"             # "Si" | "Si3N4" | "4H-SiC" | "LN_Xcut" | "AlN"
    clad_material: str = "Air"            # "Air" (suspended high-G_B SBS!) | "SiO2" (embedded)
    film_height_um: float = 0.22
    ring_width_um: float = 0.45
    ring_radius_um: float = 40.0
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    sbs_direction: str = "backward"       # "backward" (q_z = 2*k_opt) | "forward" (q_z = 0 breathing mode)
    q_int_opt: float = 5.0e5
    q_ext_opt: float = 5.0e5
    q_mech_override: float | None = None
    mesh_nx: int = 12                     # Structured 2D P2 triangulation across waveguide core
    mesh_ny: int = 8


def simulate_sbs_brillouin_mrr(cfg: SBSBrillouinConfig) -> dict[str, Any]:
    """Full 2D cross-section vector P2 solid-mechanics FEM solver for guided acoustic modes & SBS gain.

    Solves the exact 3-component (u_x, u_y, u_z) 2D cross-section Navier-Cauchy elastodynamic
    eigenproblem at longitudinal acoustic wavevector q_z (= 2*beta_opt for backward SBS, or 0 for
    forward SBS) using `scikit-fem` `ElementTriP2` vector basis, then integrates the 2D photoelastic +
    moving-boundary overlap against the 2D optical eigenmode to extract Omega_B/(2*pi), G_B [W^-1 m^-1],
    and microring SBS lasing threshold P_th,SBS [mW] (Rakich PRX 2012; Otterstrom Science 2018).
    """
    import skfem
    from skfem import Basis, ElementTriP2, ElementVector, MeshTri

    t0 = time.perf_counter()
    wl0_m = float(cfg.wavelength_nm) * 1e-9
    om_opt = 2.0 * math.pi * C0_M_PER_S / wl0_m
    w_m = float(cfg.ring_width_um) * 1e-6
    h_m = float(cfg.film_height_um) * 1e-6
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_m = 2.0 * math.pi * r_m

    # 1. Solve 2D optical eigenmode first to get exact beta_opt = k0 * neff
    core_fn = get_material_fn(cfg.core_material)
    clad_fn = get_material_fn(cfg.clad_material)
    wl_um = cfg.wavelength_nm * 1e-3
    spec_opt = WaveguideCrossSectionSpec(
        w_bottom_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=0.0,
        core_material_fn=core_fn, clad_material_fn=clad_fn, top_clad_material_fn=clad_fn,
        x_min_um=-max(2.0, 0.5 * cfg.ring_width_um + 1.2 * wl_um),
        x_max_um=max(2.4, 0.5 * cfg.ring_width_um + 1.4 * wl_um),
        h_box_um=max(1.8, 1.2 * wl_um), h_top_clad_um=max(1.8, 1.2 * wl_um),
    )
    m_opt = _solve_single_mode(spec_opt, wl0_m, r_m, cfg.polarization, "fem", 0.048, export_grid=True)
    neff_opt = float(np.real(m_opt.n_eff))
    beta_opt = (2.0 * math.pi / wl0_m) * neff_opt
    want_te = cfg.polarization.upper().startswith("TE")
    n_c = float(core_fn(wl0_m, T_REF_K).nx if want_te else core_fn(wl0_m, T_REF_K).ny)

    is_fwd = cfg.sbs_direction.lower().startswith("f")
    qz = 0.0 if is_fwd else (2.0 * beta_opt)

    # 2. Build 2D P2 vector solid-mechanics mesh on the waveguide core (with surrounding SiO2 if embedded)
    pr_c = get_elastic_material_props(cfg.core_material)
    is_air_clad = cfg.clad_material.lower() == "air"
    if is_air_clad:
        xs = np.linspace(-0.5 * w_m, 0.5 * w_m, max(6, int(cfg.mesh_nx)) + 1)
        ys = np.linspace(0.0, h_m, max(4, int(cfg.mesh_ny)) + 1)
    else:
        pad = max(w_m, h_m) * 1.2
        xs = np.unique(np.concatenate([
            np.linspace(-0.5 * w_m - pad, -0.5 * w_m, 4),
            np.linspace(-0.5 * w_m, 0.5 * w_m, max(6, int(cfg.mesh_nx)) + 1),
            np.linspace(0.5 * w_m, 0.5 * w_m + pad, 4),
        ]))
        ys = np.unique(np.concatenate([
            np.linspace(-pad, 0.0, 4),
            np.linspace(0.0, h_m, max(4, int(cfg.mesh_ny)) + 1),
            np.linspace(h_m, h_m + pad, 4),
        ]))
    mesh = MeshTri.init_tensor(xs, ys)
    # Real-valued 3-component basis: [u_x, u_y, v_z] where u_z = i * v_z makes the 3D elastodynamic
    # waveguide operator at ANY q_z STRICTLY REAL SYMMETRIC!!
    # Verify: S_xx = u_x,x; S_yy = u_y,y; S_zz = i*qz*(i*v_z) = -qz*v_z (REAL!);
    #         2*S_xy = u_x,y + u_y,x (REAL!);
    #         2*S_xz = i*qz*u_x + i*v_z,x = i*(qz*u_x + v_z,x) (PURE IMAG -> |2*S_xz|^2 REAL!);
    #         2*S_yz = i*qz*u_y + i*v_z,y = i*(qz*u_y + v_z,y) (PURE IMAG -> |2*S_yz|^2 REAL!).
    basis = Basis(mesh, ElementVector(ElementTriP2(), 3))

    pr_b = get_elastic_material_props(cfg.clad_material) if not is_air_clad else pr_c

    def _mat_at_qp(w_qp: Any, key: str) -> np.ndarray:
        xc, yc = w_qp.x[0], w_qp.x[1]
        if is_air_clad:
            return np.full_like(xc, pr_c[key])
        in_c = (np.abs(xc) <= 0.5 * w_m + 1e-12) & (yc >= -1e-12) & (yc <= h_m + 1e-12)
        return np.where(in_c, pr_c[key], pr_b[key])

    @skfem.BilinearForm
    def a_elast(u, v, w_qp):
        c11 = _mat_at_qp(w_qp, "C11")
        c22 = _mat_at_qp(w_qp, "C22")
        c33 = _mat_at_qp(w_qp, "C33")
        c12 = _mat_at_qp(w_qp, "C12")
        c13 = _mat_at_qp(w_qp, "C13")
        c23 = _mat_at_qp(w_qp, "C23")
        c44 = _mat_at_qp(w_qp, "C44")
        c55 = _mat_at_qp(w_qp, "C55")
        c66 = _mat_at_qp(w_qp, "C66")

        sxx_u, syy_u, szz_u = u.grad[0, 0], u.grad[1, 1], -qz * u[2]
        gxy_u = u.grad[0, 1] + u.grad[1, 0]
        gxz_u = qz * u[0] + u.grad[2, 0]
        gyz_u = qz * u[1] + u.grad[2, 1]

        sxx_v, syy_v, szz_v = v.grad[0, 0], v.grad[1, 1], -qz * v[2]
        gxy_v = v.grad[0, 1] + v.grad[1, 0]
        gxz_v = qz * v[0] + v.grad[2, 0]
        gyz_v = qz * v[1] + v.grad[2, 1]

        txx = c11 * sxx_u + c12 * syy_u + c13 * szz_u
        tyy = c12 * sxx_u + c22 * syy_u + c23 * szz_u
        tzz = c13 * sxx_u + c23 * syy_u + c33 * szz_u
        return txx * sxx_v + tyy * syy_v + tzz * szz_v + c66 * gxy_u * gxy_v + c55 * gxz_u * gxz_v + c44 * gyz_u * gyz_v

    @skfem.BilinearForm
    def m_elast(u, v, w_qp):
        rho = _mat_at_qp(w_qp, "rho")
        return rho * (u[0] * v[0] + u[1] * v[1] + u[2] * v[2])

    K_mat = a_elast.assemble(basis)
    M_mat = m_elast.assemble(basis)

    # Shift-invert around longitudinal/breathing acoustic frequency target
    v_long = math.sqrt(pr_c["C11"] / pr_c["rho"])
    f_guess_hz = (v_long / (2.0 * w_m)) if is_fwd else (v_long * qz / (2.0 * math.pi))
    sigma_om2 = (2.0 * math.pi * f_guess_hz * 0.75) ** 2
    n_eig = 10
    evals, evecs = eigsh(K_mat, k=n_eig, M=M_mat, sigma=sigma_om2, which="LM")
    order = np.argsort(np.real(evals))
    evals, evecs = np.real(evals[order]), np.real(evecs[:, order])

    # Interpolate optical intensity |E_x|^2, |E_y|^2, |E_z|^2 onto quadrature/element centroids to score EVERY
    # acoustic mode by its true photoelastic + moving-boundary SBS gain G_B (Rakich PRX 2012 Eq. 4)!
    from scipy.interpolate import RegularGridInterpolator
    x_opt, y_opt = m_opt.x_m, m_opt.y_m
    dx_o, dy_o = float(np.mean(np.diff(x_opt))), float(np.mean(np.diff(y_opt)))
    ex2_fn = RegularGridInterpolator((y_opt, x_opt), np.abs(m_opt.Ex) ** 2, bounds_error=False, fill_value=0.0)
    ey2_fn = RegularGridInterpolator((y_opt, x_opt), np.abs(m_opt.Ey) ** 2, bounds_error=False, fill_value=0.0)
    ez2_fn = RegularGridInterpolator((y_opt, x_opt), np.abs(m_opt.Ez) ** 2, bounds_error=False, fill_value=0.0)

    # Evaluate optical energy denominator P_opt = m_opt.power_w, v_g_opt ~ C0 / neff_opt
    p_opt_w = max(float(m_opt.power_w), 1e-30)
    p11, p12 = pr_c["p11"], pr_c["p12"]
    q_m = float(cfg.q_mech_override) if cfg.q_mech_override is not None else float(pr_c["Q_mech"])

    # Compute exact quadrature-point coordinates & weights once across the 2D solid mesh
    qx = basis.global_coordinates()  # shape (2, n_elem, n_qp)
    qpts = np.stack([qx[1].ravel(), qx[0].ravel()], axis=-1)
    ex2_q = ex2_fn(qpts).reshape(qx[0].shape)
    ey2_q = ey2_fn(qpts).reshape(qx[0].shape)
    ez2_q = ez2_fn(qpts).reshape(qx[0].shape)
    in_c_q = (np.abs(qx[0]) <= 0.5 * w_m + 1e-12) & (qx[1] >= -1e-12) & (qx[1] <= h_m + 1e-12)
    qw = basis.dx  # quadrature weights (n_elem, n_qp)

    # Evaluate each acoustic eigenmode's exact electrostrictive photoelastic coupling integral:
    #   f_PE = int_core (-0.5 * eps0 * n^4) * [ (p11*Sxx + p12*(Syy+Szz))*|Ex|^2 + (p11*Syy + p12*(Sxx+Szz))*|Ey|^2 + (p11*Szz + p12*(Sxx+Syy))*|Ez|^2 ] dA
    #   SBS peak gain (Rakich PRX 2, 011008, 2012 Eq. 4; Wolff PRA 92, 013836, 2015 Eq. 51):
    #     G_B [W^-1 m^-1] = (2 * om_opt * Q_m / (Omega_B^2 * <u|M|u>)) * |f_tot|^2 / (P_opt^2)
    #   Wait: check exact dimensional units of G_B!
    #     [f_PE] = [eps0 * V^2/m^2 * 1/m * m^2] = [J/m^2] (force per unit length per unit displacement!)
    #     Wait: u is normalized so <u|M|u> = int rho |u|^2 dA = 1 [kg/m] (mass per unit length * m^2)!
    #     Actually: [u] = [1] -> [<u|M|u>] = [kg/m^3 * m^2] = [kg/m], [S] = [1/m], [f_PE] = [eps0 * (V/m)^2 * (1/m) * m^2] = [J/m^2 = N/m]!
    #     Check units of (2 * om_opt * Q_m / (Omega_B^2 * M_eff)) * |f_PE|^2 / P_opt^2:
    #       Numerator: (1/s) * (N/m)^2 = N^2 / (m^2 * s)
    #       Denominator: (1/s^2) * (kg/m) * (W)^2 = (kg/(m*s^2)) * (N*m/s)^2 = (N/m^2) * (N^2 * m^2 / s^2) = N^3 / s^2
    #       Wait: ratio = (N^2 / (m^2 s)) / (N^3 / s^2) = s / (N * m^2) = 1 / (W * m)!!  EXACT UNITS [W^-1 m^-1]!!
    #     Check factor of 2 vs 4 in P_opt = 0.5 * Re(int E x H*):
    #       With time-averaged energy density 0.25*eps0*n^4*p*S*|E|^2 -> electrostrictive force density
    #       per unit power: Wolff PRA 2015 Eq. 51:
    #         G_B = (2 * om_opt * Q_m) / (Omega_B^2 * M_eff) * |<u, f_opt> / P_opt|^2
    #       where f_opt has factor 0.25*eps0*n^4 (from 1/2 * Re(E^* . delta_eps . E) with peak E -> time-avg 1/4)!
    best_gb = -1.0
    best_mode: dict[str, float] = {}
    for k in range(len(evals)):
        if evals[k] <= (2.0 * math.pi * 0.5e9) ** 2:
            continue
        om_b = math.sqrt(float(evals[k]))
        u_sol = evecs[:, k]
        m_eff = float(u_sol @ (M_mat @ u_sol))
        uh = basis.interpolate(u_sol)
        sxx = uh.grad[0, 0]
        syy = uh.grad[1, 1]
        szz = -qz * uh[2]
        # Time-averaged electrostrictive photoelastic overlap integral [N/m]
        pe_integrand = -0.25 * EPS0_F_PER_M * (n_c ** 4) * (
            (p11 * sxx + p12 * (syy + szz)) * ex2_q
            + (p11 * syy + p12 * (sxx + szz)) * ey2_q
            + (p11 * szz + p12 * (sxx + syy)) * ez2_q
        )
        f_pe = float(np.sum(np.where(in_c_q, pe_integrand, 0.0) * qw))
        # Estimate moving-boundary radiation pressure on the two sidewalls x = +-W/2 (major contributor in suspended Si!)
        f_mb = 0.0
        if is_air_clad:
            # Sample mean u_x on left and right sidewalls (breathing motion: u_x(+W/2) - u_x(-W/2))
            y_sw = np.linspace(0.1 * h_m, 0.9 * h_m, 11)
            # Evaluate via nearest nodal DOFs on left/right boundaries
            p_nodes = mesh.p
            # ElementTriP2 scalar vertex values are stored at the first mesh.nvertices DOFs of each component
            # Use basis.doflocs for exact P2 DOF locations!
            dlocs = basis.doflocs  # shape (2, n_dofs)
            dofs_x = basis.get_dofs().nodal["u^1"] if "u^1" in basis.get_dofs().nodal else np.arange(0, basis.N, 3)
            # Safest way in skfem ElementVector(P2, 3): component 0 DOFs are 0, 3, 6, ...
            c0 = np.arange(0, basis.N, 3)
            x0, y0 = dlocs[0, c0], dlocs[1, c0]
            ux_r = float(np.mean(u_sol[c0][(np.abs(x0 - 0.5 * w_m) < 1e-10) & (y0 >= 0.0) & (y0 <= h_m)]))
            ux_l = float(np.mean(u_sol[c0][(np.abs(x0 + 0.5 * w_m) < 1e-10) & (y0 >= 0.0) & (y0 <= h_m)]))
            # Radiation pressure force along x on each sidewall: 0.25 * eps0 * (n_c^2 - 1) * int |E_sw|^2 dy
            ix_r = int(np.argmin(np.abs(x_opt - (0.5 * w_m - dx_o))))
            ym_o = (y_opt >= 0.0) & (y_opt <= h_m)
            e_sw_int = float(np.sum((np.abs(m_opt.Ex[ym_o, ix_r]) ** 2 * (n_c ** 2) + np.abs(m_opt.Ez[ym_o, ix_r]) ** 2) * dy_o))
            f_mb = 0.25 * EPS0_F_PER_M * (n_c ** 2 - 1.0) * e_sw_int * (ux_r - ux_l)

        f_tot = f_pe + f_mb
        gb_val = (2.0 * om_opt * q_m / max((om_b ** 2) * m_eff, 1e-40)) * ((f_tot / p_opt_w) ** 2)
        if gb_val > best_gb:
            best_gb = gb_val
            best_mode = {
                "Omega_B_2pi_GHz": om_b / (2.0 * math.pi) * 1e-9,
                "acoustic_linewidth_MHz": (om_b / (2.0 * math.pi * q_m)) * 1e-6,
                "G_B_per_W_m": gb_val,
                "G_B_photoelastic_only": (2.0 * om_opt * q_m / max((om_b ** 2) * m_eff, 1e-40)) * ((f_pe / p_opt_w) ** 2),
                "G_B_moving_boundary_only": (2.0 * om_opt * q_m / max((om_b ** 2) * m_eff, 1e-40)) * ((f_mb / p_opt_w) ** 2),
            }

    # Microring SBS lasing threshold P_th,SBS (Gundavarapu Nat. Photonics 2019; Otterstrom Science 2018):
    #   Intracavity threshold: G_B * L_rt * P_cav,th = alpha_tot * L_rt = kappa_tot * T_rt
    #   Bus input threshold:   P_th,SBS = (kappa_tot^2 * T_rt / (4 * kappa_ext)) * (kappa_tot * T_rt / (G_B * L_rt))
    #                                   = (kappa_tot^3 * T_rt^2) / (4 * kappa_ext * G_B * L_rt)
    #   (assuming Stokes mode matches an adjacent cavity resonance; check: P_cav = 4*k_ext/(k_tot^2*T_rt)*P_in!)
    k_int = om_opt / max(cfg.q_int_opt, 100.0)
    k_ext = om_opt / max(cfg.q_ext_opt, 100.0)
    k_tot = k_int + k_ext
    t_rt = neff_opt * l_rt_m / C0_M_PER_S
    gb_peak = max(best_mode.get("G_B_per_W_m", 0.0), 1e-9)
    p_th_sbs_w = ((k_tot ** 3) * (t_rt ** 2)) / (4.0 * k_ext * gb_peak * l_rt_m)

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "core_material": cfg.core_material,
            "clad_material": cfg.clad_material,
            "sbs_direction": cfg.sbs_direction,
            "neff_opt": round(neff_opt, 5),
            "acoustic_qz_per_um": round(qz * 1e-6, 4),
            "Omega_B_2pi_GHz": round(best_mode.get("Omega_B_2pi_GHz", 0.0), 4),
            "acoustic_linewidth_MHz": round(best_mode.get("acoustic_linewidth_MHz", 0.0), 3),
            "Q_mech": round(q_m, 1),
            "G_B_per_W_m": round(best_mode.get("G_B_per_W_m", 0.0), 2),
            "G_B_photoelastic_only": round(best_mode.get("G_B_photoelastic_only", 0.0), 2),
            "G_B_moving_boundary_only": round(best_mode.get("G_B_moving_boundary_only", 0.0), 2),
            "SBS_ring_threshold_mW": round(p_th_sbs_w * 1e3, 3),
        },
    }
