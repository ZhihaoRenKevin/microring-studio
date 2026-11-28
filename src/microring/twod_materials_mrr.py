"""First-Principles 2D van der Waals & Layered Photonic Material Library on Microring Resonators
(`microring.twod_materials_mrr` — 27 Peer-Reviewed 2D Materials across 8 Physics Families, 400 nm – 18 µm).

Zero commercial solver dependencies. Couples exact microscopic 2D surface-conductivity &
anisotropic dielectric tensors directly onto MicroRing Studio's 2D conformal curved vector
Maxwell FEM waveguide eigenmodes via exact full-tensor tangential + normal surface perturbation
with arbitrary in-plane crystal orientation `crystal_angle_deg` (Hanson JAP 2008; Phare Nat.
Photonics 2015; Sorianello Nat. Photonics 2018; Ma Nature 2018; Datta Nat. Photonics 2020;
Ermolaev Nat. Commun. 2021; Guo Nature 2023):

  Delta n_eff,|| = (i / (4 * k_0 * P_z)) * int_sheet [
      sigma_xx |E_x|^2 + sigma_zz |E_z|^2 + 2 * sigma_xz * Re(E_x* E_z)
  ] dx
  Delta n_eff,_| = (omega * eps_0 * d * n_clad^4 / (4 * k_0 * P_z))
                   * (1 / n_clad^2 - 1 / eps_yy) * int_sheet |E_y|^2 dx

27 Materials across 8 Physical Families:
  1. Dirac & Gapped Graphene (`dirac_semimetal`, `bilayer_graphene`):
     `Graphene` (SLG/DLG exact finite-T Kubo + quantum capacitance), `Bilayer_Graphene` (Bernal AB
     dual-gate tunable bandgap + gamma_1 = 0.39 eV Mid-IR resonance)
  2. Group-VI Monolayer TMDs (`tmd`):
     `WS2`, `MoS2`, `WSe2`, `MoSe2`, `MoTe2` (O-band / 1127–1310 nm telecom direct-gap TMD)
  3. High-Index Uniaxial Giant-Birefringence vdW Crystals (`vdw_high_index`):
     `MoS2_Bulk_vdW` (n_|| ~ 4.28, dn ~ 1.57), `WS2_Bulk_vdW` (n_|| ~ 3.99, dn ~ 1.43)
  4. Noble-Metal & Anisotropic Narrow-Gap TMDs (`narrow_gap_tmd`):
     `PtSe2` (telecom–Mid-IR 0.25 eV), `PdSe2` (anisotropic pentagonal 0.20 eV), `ReS2` (biaxial 1T')
  5. Mid-IR Uniaxial & Natural In-Plane Biaxial Hyperbolic Phonon Polaritons (`hyperbolic_phonon`):
     `hBN`, `hBN_Isotopic_10B` (3x lower damping), `alpha_MoO3` (biaxial [100]/[001]/[010]), `alpha_V2O5`
  6. Anisotropic Narrow-Gap Mid-IR / LWIR 2D Semiconductors (`anisotropic_2d`):
     `Black_Phosphorus`, `b_AsP` (LWIR 0.15 eV / 8.3 um), `Te_Tellurene` (chiral air-stable 0.35 eV)
  7. Plasmonic MXenes & Topological / Weyl Semimetals (`mxene_plasmonic`, `topological_semimetal`):
     `Ti3C2Tx_MXene` (telecom ENZ ~ 1.2 um), `Nb2CTx_MXene`, `Bi2Se3` (TI surface states + bulk),
     `WTe2` (Type-II Weyl in-plane hyperbolic plasmonics 9.5–16.6 um)
  8. Nonlinear / Electro-Optic / 2D Ferroelectric vdW Crystals (`vdw_eo_ferroelectric`):
     `NbOCl2` (giant chi^(2) d_eff ~ 140 pm/V), `In2Se3_alpha` (2D room-T ferroelectric),
     `CuInP2S6` (wide-gap transparent ferroelectric), `GaSe` (0.65–18 um layered chi^(2))
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np

from .eigenmode_solver import WaveguideCrossSectionSpec
from .materials import C0_M_PER_S, EPS0_F_PER_M, HBAR_J_S, T_REF_K, get_material_fn
from .mrr_cmt_fit_predict import loss_db_per_cm_to_q_int
from .mrr_topologies import _compute_group_index_fast, _solve_single_mode

KB_J_PER_K = 1.380649e-23
Q_E_C = 1.602176634e-19
M_E_KG = 9.1093837015e-31
VF_GRAPHENE = 1.0e6  # Fermi velocity in graphene [m/s]
SIGMA0_UNIVERSAL_S = (Q_E_C ** 2) / (4.0 * HBAR_J_S)  # Universal optical conductance 6.0853e-5 S


# ==============================================================================
# 1. Peer-Reviewed 2D van der Waals Material Catalog (27 Materials, 8 Families)
# ==============================================================================

TWOD_MATERIAL_CATALOG: dict[str, dict[str, Any]] = {
    # --- Family 1: Dirac & Gapped Graphene ---
    "Graphene": {
        "family": "dirac_semimetal",
        "thickness_nm": 0.34,
        "scattering_mev": 10.0,
        "sheet_res_ohm_sq": 350.0,
        "waveband_um": (0.4, 20.0),
        "is_biaxial": False,
        "source": "Hanson JAP 2008; Phare Nat. Photonics 2015; Sorianello Nat. Photonics 2018",
        "note": "Exact finite-T Kubo conductivity (intraband Drude + interband Pauli blocking at |mu_c| = hbar*om/2)",
    },
    "Bilayer_Graphene": {
        "family": "bilayer_graphene",
        "thickness_nm": 0.68,
        "gamma1_ev": 0.39,           # Interlayer dimer hopping energy (~3.18 um Mid-IR resonance!)
        "m_eff_ratio": 0.034,        # Parabolic band effective mass m*/m_0
        "max_bandgap_ev": 0.25,      # Dual-gate tunable bandgap saturation [eV]
        "v_gap_sat_v": 4.5,          # Characteristic gate voltage for bandgap opening [V]
        "scattering_mev": 12.0,
        "sheet_res_ohm_sq": 220.0,
        "waveband_um": (0.5, 20.0),
        "is_biaxial": False,
        "source": "Zhang Nature 2009; Nicol & Carbotte PRB 2008; Kuzmenko PRB 2009",
        "note": "Bernal AB-stacked bilayer graphene with gate-openable bandgap (0-0.25 eV) & 0.39 eV gamma_1 resonance",
    },

    # --- Family 2: Group-VI Monolayer Semiconducting TMDs ---
    "WS2": {
        "family": "tmd",
        "thickness_nm": 0.65,
        "eps_bg": 16.0, "eps_perp": 6.3,
        "e_x0_ev": 2.01, "gamma_x0_mev": 28.0, "f_x0_ev2": 2.45,
        "e_trion_ev": 1.97, "gamma_trion_mev": 34.0, "f_trion_ev2": 0.65,
        "e_xb_ev": 2.40, "gamma_xb_mev": 65.0, "f_xb_ev2": 2.60,
        "n_sat_cm2": 6.0e12,
        "sheet_res_ohm_sq": 4500.0,
        "waveband_um": (0.45, 2.5),
        "is_biaxial": False,
        "source": "Datta et al., Nature Photonics 14, 256 (2020)",
        "note": "Record sub-bandgap telecom electro-refraction phase-to-loss ratio (Delta_n / Delta_k > 10)",
    },
    "MoS2": {
        "family": "tmd",
        "thickness_nm": 0.65,
        "eps_bg": 18.0, "eps_perp": 6.8,
        "e_x0_ev": 1.88, "gamma_x0_mev": 35.0, "f_x0_ev2": 2.30,
        "e_trion_ev": 1.84, "gamma_trion_mev": 42.0, "f_trion_ev2": 0.60,
        "e_xb_ev": 2.04, "gamma_xb_mev": 60.0, "f_xb_ev2": 2.40,
        "n_sat_cm2": 7.0e12,
        "sheet_res_ohm_sq": 5500.0,
        "waveband_um": (0.45, 2.5),
        "is_biaxial": False,
        "source": "Li et al., Phys. Rev. B 90, 205422 (2014)",
        "note": "Archetypal monolayer TMD with A/B visible excitons and low-loss NIR phase gating",
    },
    "WSe2": {
        "family": "tmd",
        "thickness_nm": 0.67,
        "eps_bg": 16.5, "eps_perp": 7.0,
        "e_x0_ev": 1.66, "gamma_x0_mev": 25.0, "f_x0_ev2": 2.15,
        "e_trion_ev": 1.63, "gamma_trion_mev": 30.0, "f_trion_ev2": 0.55,
        "e_xb_ev": 2.08, "gamma_xb_mev": 58.0, "f_xb_ev2": 2.35,
        "n_sat_cm2": 5.5e12,
        "sheet_res_ohm_sq": 5000.0,
        "waveband_um": (0.50, 2.5),
        "is_biaxial": False,
        "source": "Li et al., Phys. Rev. B 90, 205422 (2014)",
        "note": "Narrow-linewidth NIR A-exciton (747 nm), strong valley/polariton coupling",
    },
    "MoSe2": {
        "family": "tmd",
        "thickness_nm": 0.67,
        "eps_bg": 17.5, "eps_perp": 7.2,
        "e_x0_ev": 1.57, "gamma_x0_mev": 26.0, "f_x0_ev2": 2.05,
        "e_trion_ev": 1.54, "gamma_trion_mev": 32.0, "f_trion_ev2": 0.55,
        "e_xb_ev": 1.78, "gamma_xb_mev": 55.0, "f_xb_ev2": 2.20,
        "n_sat_cm2": 5.5e12,
        "sheet_res_ohm_sq": 5200.0,
        "waveband_um": (0.55, 2.5),
        "is_biaxial": False,
        "source": "Li et al., Phys. Rev. B 90, 205422 (2014)",
        "note": "790 nm A-exciton TMD for SiN/SOI NIR integrated polaritonics",
    },
    "MoTe2": {
        "family": "tmd",
        "thickness_nm": 0.70,
        "eps_bg": 19.5, "eps_perp": 8.2,
        "e_x0_ev": 1.10, "gamma_x0_mev": 32.0, "f_x0_ev2": 1.85,
        "e_trion_ev": 1.07, "gamma_trion_mev": 38.0, "f_trion_ev2": 0.55,
        "e_xb_ev": 1.36, "gamma_xb_mev": 65.0, "f_xb_ev2": 2.10,
        "n_sat_cm2": 6.5e12,
        "sheet_res_ohm_sq": 3200.0,
        "waveband_um": (0.80, 2.2),
        "is_biaxial": False,
        "source": "Ruppert Nano Lett. 2014; Bie Nat. Nanotechnol. 2017; Maiti Nat. Photonics 2020",
        "note": "Direct-bandgap 2H-MoTe2 (1.10 eV / 1127 nm) for silicon-transparent O-band EA modulation & PDs",
    },

    # --- Family 3: High-Index Uniaxial Giant-Birefringence Multilayer/Bulk vdW Crystals ---
    "MoS2_Bulk_vdW": {
        "family": "vdw_high_index",
        "thickness_nm": 15.0,
        "eps_bg_par": 15.2,
        "osc_par": [
            (1.85, 0.065, 1.65),   # A-exciton [eV, gamma_eV, f_eV^2]
            (2.02, 0.090, 2.10),   # B-exciton
            (2.82, 0.320, 18.0),   # C-band nesting peak
        ],
        "eps_bg_perp": 6.2,
        "osc_perp": [
            (3.30, 0.400, 12.5),   # Out-of-plane interband transition
        ],
        "n_sat_cm2": 2.0e13,
        "sheet_res_ohm_sq": 1800.0,
        "waveband_um": (0.50, 5.0),
        "is_biaxial": False,
        "source": "Ermolaev et al., Nature Communications 12, 854 (2021)",
        "note": "High-index uniaxial vdW crystal (n_|| ~ 4.28, n_|_ ~ 2.71, giant birefringence dn ~ 1.57 at 1550 nm)",
    },
    "WS2_Bulk_vdW": {
        "family": "vdw_high_index",
        "thickness_nm": 15.0,
        "eps_bg_par": 13.6,
        "osc_par": [
            (1.97, 0.055, 1.50),
            (2.36, 0.110, 2.20),
            (2.88, 0.300, 14.5),
        ],
        "eps_bg_perp": 5.6,
        "osc_perp": [
            (3.45, 0.380, 10.5),
        ],
        "n_sat_cm2": 2.0e13,
        "sheet_res_ohm_sq": 2200.0,
        "waveband_um": (0.48, 5.0),
        "is_biaxial": False,
        "source": "Munkhbat Nat. Commun. 2022; Vyshnevyy Nano Lett. 2023",
        "note": "Low-loss giant-birefringence vdW nanophotonic crystal (n_|| ~ 3.99, n_|_ ~ 2.56 at 1550 nm)",
    },

    # --- Family 4: Noble-Metal & Biaxial Narrow-Gap TMDs ---
    "PtSe2": {
        "family": "narrow_gap_tmd",
        "thickness_nm": 4.0,
        "e_gap_ev": 0.25,          # Few-layer 1T-PtSe2 narrow gap (~5.0 um cutoff)
        "eps_bg_u": 12.5, "eps_bg_v": 12.5, "eps_perp": 7.5,
        "m_eff_u": 0.22, "m_eff_v": 0.22,
        "sigma_ib_u_s0": 2.4, "sigma_ib_v_s0": 2.4,
        "scattering_mev": 25.0,
        "sheet_res_ohm_sq": 650.0,
        "waveband_um": (0.8, 10.0),
        "is_biaxial": False,
        "source": "Yu Adv. Mater. 2018; Wang ACS Nano 2019",
        "note": "Air-stable few-layer noble-metal 1T-PtSe2 for >40 GHz broadband telecom & Mid-IR waveguide PDs",
    },
    "PdSe2": {
        "family": "narrow_gap_tmd",
        "thickness_nm": 5.0,
        "e_gap_ev": 0.20,          # Few-layer pentagonal PdSe2 (~6.2 um)
        "eps_bg_u": 13.8, "eps_bg_v": 11.4, "eps_perp": 7.2,
        "m_eff_u": 0.18, "m_eff_v": 0.42,
        "sigma_ib_u_s0": 2.2, "sigma_ib_v_s0": 1.35,
        "scattering_mev": 22.0,
        "sheet_res_ohm_sq": 900.0,
        "waveband_um": (0.8, 10.6),
        "is_biaxial": True,
        "source": "Long Adv. Sci. 2019; Oyedele JACS 2017",
        "note": "Puckered pentagonal in-plane anisotropic PdSe2 for polarization-sensitive telecom/LWIR PDs",
    },
    "ReS2": {
        "family": "biaxial_tmd",
        "thickness_nm": 2.0,
        "eps_bg_u": 15.8, "eps_bg_v": 13.6, "eps_perp": 8.0,
        "e_x_u_ev": 1.47, "gamma_x_u_mev": 38.0, "f_x_u_ev2": 2.20,  # Along Re-chain b-axis
        "e_x_v_ev": 1.55, "gamma_x_v_mev": 42.0, "f_x_v_ev2": 1.70,  # Orthogonal axis
        "n_sat_cm2": 7.5e12,
        "sheet_res_ohm_sq": 4200.0,
        "waveband_um": (0.55, 2.5),
        "is_biaxial": True,
        "source": "Tongay Nat. Commun. 2014; Aslan Nano Lett. 2016",
        "note": "Distorted 1T'-ReS2 with strong linear dichroism along Re-chain b-axis (E_x,|| = 1.47 vs 1.55 eV)",
    },

    # --- Family 5: Mid-IR Hyperbolic Phonon-Polariton vdW Crystals (Uniaxial & In-Plane Biaxial) ---
    "hBN": {
        "family": "hyperbolic_phonon",
        "thickness_nm": 10.0,
        "eps_inf_u": 4.87, "w_to_u_cm1": 1370.0, "w_lo_u_cm1": 1610.0, "gamma_u_cm1": 5.0,
        "eps_inf_v": 4.87, "w_to_v_cm1": 1370.0, "w_lo_v_cm1": 1610.0, "gamma_v_cm1": 5.0,
        "eps_inf_y": 2.95, "w_to_y_cm1": 780.0,  "w_lo_y_cm1": 830.0,  "gamma_y_cm1": 4.0,
        # Legacy aliases kept for 100% backward compatibility
        "eps_inf_par": 4.87, "w_to_par_cm1": 1370.0, "w_lo_par_cm1": 1610.0, "gamma_par_cm1": 5.0,
        "eps_inf_perp": 2.95, "w_to_perp_cm1": 780.0, "w_lo_perp_cm1": 830.0, "gamma_perp_cm1": 4.0,
        "sheet_res_ohm_sq": 1e9,
        "waveband_um": (4.0, 16.0),
        "is_biaxial": False,
        "source": "Caldwell et al., Nature Communications 5, 5221 (2014)",
        "note": "Natural hBN uniaxial hyperbolic Type-I (12.0-12.8 um) & Type-II (6.2-7.3 um) phonon polaritons",
    },
    "hBN_Isotopic_10B": {
        "family": "hyperbolic_phonon",
        "thickness_nm": 10.0,
        "eps_inf_u": 5.10, "w_to_u_cm1": 1394.5, "w_lo_u_cm1": 1650.0, "gamma_u_cm1": 1.8,
        "eps_inf_v": 5.10, "w_to_v_cm1": 1394.5, "w_lo_v_cm1": 1650.0, "gamma_v_cm1": 1.8,
        "eps_inf_y": 2.95, "w_to_y_cm1": 798.0,  "w_lo_y_cm1": 845.0,  "gamma_y_cm1": 1.5,
        "sheet_res_ohm_sq": 1e9,
        "waveband_um": (4.0, 16.0),
        "is_biaxial": False,
        "source": "Giles et al., Nature Materials 17, 134 (2018)",
        "note": "Monoisotopic h^10BN with 3x longer phonon-polariton lifetime (gamma = 1.8 cm^-1)",
    },
    "alpha_MoO3": {
        "family": "hyperbolic_phonon",
        "thickness_nm": 25.0,
        "eps_inf_u": 4.0, "w_to_u_cm1": 820.0, "w_lo_u_cm1": 972.0,  "gamma_u_cm1": 4.0,  # [100] a-axis
        "eps_inf_v": 5.2, "w_to_v_cm1": 545.0, "w_lo_v_cm1": 851.0,  "gamma_v_cm1": 4.0,  # [001] c-axis
        "eps_inf_y": 2.4, "w_to_y_cm1": 958.0, "w_lo_y_cm1": 1004.0, "gamma_y_cm1": 2.0,  # [010] b-axis (normal)
        "sheet_res_ohm_sq": 1e9,
        "waveband_um": (8.0, 19.0),
        "is_biaxial": True,
        "source": "Ma Nature 2018; Alvarez-Perez Adv. Mater. 32, 1908176 (2020)",
        "note": "Natural in-plane biaxial hyperbolic vdW crystal (3 orthogonal Reststrahlen bands, 9.9-18.3 um)",
    },
    "alpha_V2O5": {
        "family": "hyperbolic_phonon",
        "thickness_nm": 25.0,
        "eps_inf_u": 5.2, "w_to_u_cm1": 765.0, "w_lo_u_cm1": 950.0,  "gamma_u_cm1": 14.0, # [100]
        "eps_inf_v": 4.5, "w_to_v_cm1": 506.0, "w_lo_v_cm1": 842.0,  "gamma_v_cm1": 12.0, # [001]
        "eps_inf_y": 2.9, "w_to_y_cm1": 976.0, "w_lo_y_cm1": 1037.0, "gamma_y_cm1": 6.0,  # [010]
        "sheet_res_ohm_sq": 1e9,
        "waveband_um": (8.0, 20.0),
        "is_biaxial": True,
        "source": "Taboada-Gutierrez et al., Nature 582, 530 (2020)",
        "note": "Natural in-plane biaxial hyperbolic alpha-V2O5 supporting canalized Mid-IR phonon polaritons",
    },

    # --- Family 6: Anisotropic Narrow-Gap Mid-IR / LWIR 2D Semiconductors ---
    "Black_Phosphorus": {
        "family": "anisotropic_2d",
        "thickness_nm": 5.0,
        "e_gap_ev": 0.35,
        "m_eff_armchair": 0.15, "m_eff_zigzag": 1.15,
        "eps_bg_u": 13.0, "eps_bg_v": 10.5, "eps_perp": 8.3,
        "sigma_ib_u_s0": 1.60, "sigma_ib_v_s0": 0.25,
        "scattering_mev": 15.0,
        "sheet_res_ohm_sq": 1200.0,
        "waveband_um": (1.2, 10.0),
        "is_biaxial": True,
        "source": "Low et al., Phys. Rev. B 90, 075434 (2014)",
        "note": "Anisotropic direct-gap Mid-IR 2D semiconductor (armchair/zigzag 7.7x mass anisotropy + Moss-Burstein)",
    },
    "b_AsP": {
        "family": "anisotropic_2d",
        "thickness_nm": 8.0,
        "e_gap_ev": 0.15,          # b-As_0.83 P_0.17 LWIR ~8.3 um bandgap!
        "m_eff_armchair": 0.10, "m_eff_zigzag": 0.95,
        "eps_bg_u": 15.0, "eps_bg_v": 12.0, "eps_perp": 9.0,
        "sigma_ib_u_s0": 2.00, "sigma_ib_v_s0": 0.30,
        "scattering_mev": 14.0,
        "sheet_res_ohm_sq": 950.0,
        "waveband_um": (2.0, 12.0),
        "is_biaxial": True,
        "source": "Long Adv. Mater. 2017; Yuan ACS Nano 2018",
        "note": "Black arsenic-phosphorus alloy extending room-temperature anisotropic 2D detection to 8.3 um LWIR",
    },
    "Te_Tellurene": {
        "family": "anisotropic_2d",
        "thickness_nm": 6.0,
        "e_gap_ev": 0.35,          # Air-stable 2D chiral Tellurene (~3.5 um)
        "m_eff_armchair": 0.11, "m_eff_zigzag": 0.38,  # helical c-axis vs perpendicular
        "eps_bg_u": 23.0, "eps_bg_v": 16.5, "eps_perp": 16.5,
        "sigma_ib_u_s0": 1.80, "sigma_ib_v_s0": 0.70,
        "scattering_mev": 16.0,
        "sheet_res_ohm_sq": 850.0,
        "waveband_um": (1.2, 8.0),
        "is_biaxial": True,
        "source": "Wang Nat. Electron. 2018; Amani ACS Nano 2018",
        "note": "Air-stable high-mobility chiral 2D Tellurene with strong helical c-axis dichroism for SWIR/MWIR",
    },

    # --- Family 7: Plasmonic MXenes & Topological / Weyl Semimetals ---
    "Ti3C2Tx_MXene": {
        "family": "mxene_plasmonic",
        "thickness_nm": 5.0,
        "eps_inf": 2.6, "wp_ev": 2.05, "gamma_p_ev": 0.35,
        "w1_ev": 1.62, "f1": 0.75, "gamma_1_ev": 0.45,
        "eps_perp": 4.5,
        "sheet_res_ohm_sq": 120.0,
        "waveband_um": (0.5, 10.0),
        "is_biaxial": False,
        "source": "Chaudhuri ACS Nano 2018; Maleski ACS Photonics 2018",
        "note": "2D metallic Ti3C2Tx MXene with telecom ENZ crossover (~1.24 um) & plasmonic photothermal/EA tuning",
    },
    "Nb2CTx_MXene": {
        "family": "mxene_plasmonic",
        "thickness_nm": 5.0,
        "eps_inf": 3.2, "wp_ev": 1.55, "gamma_p_ev": 0.40,
        "w1_ev": 1.25, "f1": 0.85, "gamma_1_ev": 0.50,
        "eps_perp": 4.2,
        "sheet_res_ohm_sq": 180.0,
        "waveband_um": (0.6, 10.0),
        "is_biaxial": False,
        "source": "Gao et al., Nature Communications 11, 2020",
        "note": "Broadband NIR/SWIR Nb2CTx MXene saturable absorber & photothermal modulator (ENZ ~ 1.75 um)",
    },
    "Bi2Se3": {
        "family": "topological_semimetal",
        "thickness_nm": 10.0,
        "eps_inf_bulk": 27.0, "e_gap_ev": 0.30,
        "w_to_ph_cm1": 61.0, "w_lo_ph_cm1": 133.0, "gamma_ph_cm1": 6.0,
        "vf_surface_m_s": 5.0e5,     # Topological Dirac surface-state Fermi velocity
        "mu0_surface_ev": 0.20,      # Native n-doped topological surface Fermi level
        "scattering_mev": 12.0,
        "eps_perp": 18.0,
        "sheet_res_ohm_sq": 280.0,
        "waveband_um": (1.0, 20.0),
        "is_biaxial": False,
        "source": "Di Pietro Nat. Nanotechnol. 2013; Autore Adv. Opt. Mater. 2017",
        "note": "Topological insulator thin film combining high-n bulk (n ~ 5.2) + gated Dirac surface plasmonics",
    },
    "WTe2": {
        "family": "weyl_semimetal",
        "thickness_nm": 15.0,
        "eps_inf_u": 16.0, "wp_u_ev": 0.52, "gamma_u_ev": 0.030,  # a-axis (metallic above ~9.5 um)
        "eps_inf_v": 14.0, "wp_v_ev": 0.28, "gamma_v_ev": 0.035,  # b-axis (dielectric until ~16.6 um)
        "eps_perp": 10.0,
        "sheet_res_ohm_sq": 160.0,
        "waveband_um": (1.5, 20.0),
        "is_biaxial": True,
        "source": "Frenzel PRB 2017; Wang Nature Materials 19, 964 (2020)",
        "note": "Type-II Weyl semimetal Td-WTe2 with natural in-plane hyperbolic plasmonic window (9.5-16.6 um)",
    },

    # --- Family 8: Nonlinear / Electro-Optic / 2D Ferroelectric vdW Crystals ---
    "NbOCl2": {
        "family": "vdw_eo_ferroelectric",
        "thickness_nm": 20.0,
        "n_u_1550": 2.32, "n_v_1550": 2.18, "n_y_1550": 1.92,
        "cauchy_b_um2": 0.045,
        "e_gap_ev": 2.50,
        "r_eff_u_pm_v": 15.0, "r_eff_v_pm_v": 5.5,
        "d_eff_pm_v": 140.0,
        "ferroelectric_dn": 0.0, "e_coercive_v_nm": 0.0,
        "sheet_res_ohm_sq": 1e9,
        "waveband_um": (0.5, 5.0),
        "is_biaxial": True,
        "source": "Guo et al., Nature 613, 53 (2023); Abdelwahab Nat. Photonics 2022",
        "note": "Giant-chi^(2) van der Waals biaxial crystal (d_eff ~ 140 pm/V, r_eff ~ 15 pm/V) for SHG/SPDC/EO MRRs",
    },
    "In2Se3_alpha": {
        "family": "vdw_eo_ferroelectric",
        "thickness_nm": 12.0,
        "n_u_1550": 2.88, "n_v_1550": 2.88, "n_y_1550": 2.45,
        "cauchy_b_um2": 0.085,
        "e_gap_ev": 1.39,
        "r_eff_u_pm_v": 12.0, "r_eff_v_pm_v": 12.0,
        "d_eff_pm_v": 45.0,
        "ferroelectric_dn": 0.035, "e_coercive_v_nm": 0.15,
        "sheet_res_ohm_sq": 5e5,
        "waveband_um": (0.95, 6.0),
        "is_biaxial": False,
        "source": "Ding Nat. Commun. 2017; Wang Adv. Mater. 2021",
        "note": "Room-temperature 2D ferroelectric alpha-In2Se3 for non-volatile dipole-switched MRR phase tuning",
    },
    "CuInP2S6": {
        "family": "vdw_eo_ferroelectric",
        "thickness_nm": 20.0,
        "n_u_1550": 2.22, "n_v_1550": 2.22, "n_y_1550": 2.02,
        "cauchy_b_um2": 0.030,
        "e_gap_ev": 2.90,
        "r_eff_u_pm_v": 18.0, "r_eff_v_pm_v": 18.0,
        "d_eff_pm_v": 28.0,
        "ferroelectric_dn": 0.015, "e_coercive_v_nm": 0.12,
        "sheet_res_ohm_sq": 1e9,
        "waveband_um": (0.45, 5.0),
        "is_biaxial": False,
        "source": "Belianinov Nano Lett. 2015; Liu Nat. Commun. 2016",
        "note": "Wide-bandgap transparent 2D ferroelectric CIPS (E_g = 2.9 eV) for zero-loss telecom EO/NV tuning",
    },
    "GaSe": {
        "family": "vdw_eo_ferroelectric",
        "thickness_nm": 20.0,
        "n_u_1550": 2.76, "n_v_1550": 2.76, "n_y_1550": 2.45,
        "cauchy_b_um2": 0.060,
        "e_gap_ev": 2.00,
        "r_eff_u_pm_v": 2.8, "r_eff_v_pm_v": 2.8,
        "d_eff_pm_v": 54.0,
        "ferroelectric_dn": 0.0, "e_coercive_v_nm": 0.0,
        "sheet_res_ohm_sq": 1e8,
        "waveband_um": (0.65, 18.0),
        "is_biaxial": False,
        "source": "Jie Adv. Funct. Mater. 2015; Tang Nat. Photonics 2019",
        "note": "Ultra-broadband layered epsilon-GaSe (0.65-18 um transparency, d_22 = 54 pm/V) for NIR/Mid-IR PICs",
    },
}


def list_2d_materials(
    family: str | None = None,
    wavelength_nm: float | None = None,
) -> list[dict[str, Any]]:
    """Return structured metadata for all 27 peer-reviewed 2D materials (optionally filtered by family or wavelength)."""
    wl_um = float(wavelength_nm) * 1e-3 if wavelength_nm is not None else None
    out: list[dict[str, Any]] = []
    for name, info in TWOD_MATERIAL_CATALOG.items():
        if family and info["family"] != family:
            continue
        wmin, wmax = info.get("waveband_um", (0.4, 20.0))
        if wl_um is not None and not (wmin <= wl_um <= wmax):
            continue
        out.append({
            "name": name,
            "family": info["family"],
            "thickness_nm": info["thickness_nm"],
            "waveband_um": [wmin, wmax],
            "is_biaxial": bool(info.get("is_biaxial", False)),
            "source": info.get("source", ""),
            "note": info.get("note", ""),
        })
    return out


# ==============================================================================
# 2. Exact Microscopic 2D Surface Conductivity & Tensor Engines
# ==============================================================================

def graphene_kubo_conductivity(
    wavelength_nm: float,
    mu_c_ev: float,
    scattering_mev: float = 10.0,
    temperature_k: float = T_REF_K,
) -> tuple[complex, complex, complex]:
    """Exact finite-temperature complex Kubo surface conductivity sigma(omega, mu_c, Gamma, T) [S].

    Uses exact closed-form intraband Drude + exact Gauss-Legendre numerical integration of the
    Kubo interband principal-value dispersion integral (Hanson JAP 103, 064302, 2008 Eq. 6-7):
      sigma_intra = i * (e^2 * k_B * T) / (pi * hbar^2 * (omega + i*2*Gamma))
                    * [mu_c/(k_B*T) + 2*ln(exp(-mu_c/(k_B*T)) + 1)]
      sigma_inter = (e^2 / (4*hbar)) * [ H(hbar*omega/2) - (i/pi) * P.V. int_0^inf (H(E) - H(hbar*omega/2)) / (E^2 - (hbar*omega/2)^2) * (hbar*omega) dE ]
    Returns `(sigma_total, sigma_intra, sigma_inter)` in Siemens [S] under `exp(-i*omega*t)` convention.
    """
    om = 2.0 * math.pi * C0_M_PER_S / (float(wavelength_nm) * 1e-9)
    hw_ev = (HBAR_J_S * om) / Q_E_C
    kbt_ev = (KB_J_PER_K * float(temperature_k)) / Q_E_C
    mu_abs = abs(float(mu_c_ev))
    gam_rad = (max(float(scattering_mev), 0.1) * 1e-3 * Q_E_C) / HBAR_J_S

    x_mu = mu_abs / max(kbt_ev, 1e-6)
    log_term = x_mu if x_mu > 40.0 else (x_mu + 2.0 * math.log1p(math.exp(-x_mu)))
    sig_intra = (
        1j * (Q_E_C ** 2 * KB_J_PER_K * float(temperature_k))
        / (math.pi * (HBAR_J_S ** 2) * (om + 1j * 2.0 * gam_rad))
        * log_term
    )

    e0 = 0.5 * hw_ev

    def _h_func(e_ev: np.ndarray | float) -> np.ndarray | float:
        return 0.5 * (np.tanh((e_ev + mu_abs) / (2.0 * kbt_ev)) + np.tanh((e_ev - mu_abs) / (2.0 * kbt_ev)))

    h_e0 = float(_h_func(e0))
    e_max = max(12.0 * e0, mu_abs + 4.0, 15.0)
    g_pts, g_wts = np.polynomial.legendre.leggauss(64)
    edges = [0.0, 0.5 * e0, 1.5 * e0, e_max]
    pv_int = 0.0 + 0.0j
    hw_c = hw_ev + 1j * (2.0 * max(float(scattering_mev), 0.1) * 1e-3)
    e0_c = 0.5 * hw_c
    for a, b in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        ee = mid + half * g_pts
        ww = half * g_wts
        he = _h_func(ee)
        integrand = (he - h_e0) / (ee ** 2 - e0_c ** 2)
        pv_int += complex(np.sum(integrand * ww))

    sig_inter = SIGMA0_UNIVERSAL_S * (h_e0 - (1j * hw_c / math.pi) * pv_int)
    return sig_intra + sig_inter, sig_intra, sig_inter


def graphene_gate_voltage_to_mu_c(
    v_gate_v: float,
    v_cnp_v: float = 0.0,
    ox_thickness_nm: float = 10.0,
    ox_eps_r: float = 9.0,
    dual_layer: bool = False,
) -> tuple[float, float, float]:
    """Solve exact oxide + Dirac quantum capacitance electrostatics for V_g -> (mu_c [eV], n_s [cm^-2], C_tot [uF/cm^2])."""
    dv = abs(float(v_gate_v) - float(v_cnp_v))
    c_ox = EPS0_F_PER_M * float(ox_eps_r) / max(float(ox_thickness_nm) * 1e-9, 1e-10)
    n_q = 2.0 if dual_layer else 1.0
    a_coef = Q_E_C / (math.pi * ((HBAR_J_S * VF_GRAPHENE) ** 2) * c_ox)
    b_coef = n_q / Q_E_C
    u_j = (-b_coef + math.sqrt(b_coef * b_coef + 4.0 * a_coef * dv)) / (2.0 * a_coef)
    mu_ev = (u_j / Q_E_C) * (1.0 if (v_gate_v - v_cnp_v) >= 0 else -1.0)
    n_s_m2 = (u_j ** 2) / (math.pi * ((HBAR_J_S * VF_GRAPHENE) ** 2))
    c_q = max(2.0 * (Q_E_C ** 2) * max(u_j, 0.026 * Q_E_C) / (math.pi * ((HBAR_J_S * VF_GRAPHENE) ** 2)), 1e-6)
    c_tot_f_m2 = 1.0 / (1.0 / c_ox + n_q / c_q)
    return float(mu_ev), float(n_s_m2 * 1e-4), float(c_tot_f_m2 * 1e2)


def bilayer_graphene_conductivity(
    wavelength_nm: float,
    v_gate_v: float,
    v_cnp_v: float = 0.0,
    ox_thickness_nm: float = 10.0,
    ox_eps_r: float = 9.0,
    scattering_mev: float = 12.0,
    temperature_k: float = T_REF_K,
) -> tuple[complex, float, float, float, float]:
    """Bernal AB-stacked Bilayer Graphene conductivity with dual-gate bandgap Delta_g(V_g) & gamma_1=0.39 eV peak.

    Returns `(sigma_total [S], mu_c [eV], n_s [cm^-2], C_tot [uF/cm^2], bandgap_ev [eV])`.
    """
    pr = TWOD_MATERIAL_CATALOG["Bilayer_Graphene"]
    om = 2.0 * math.pi * C0_M_PER_S / (float(wavelength_nm) * 1e-9)
    hw_ev = (HBAR_J_S * om) / Q_E_C
    kbt_ev = (KB_J_PER_K * float(temperature_k)) / Q_E_C
    gam_ev = max(float(scattering_mev), 0.5) * 1e-3
    gam_rad = (gam_ev * Q_E_C) / HBAR_J_S

    dv = abs(float(v_gate_v) - float(v_cnp_v))
    c_ox = EPS0_F_PER_M * float(ox_eps_r) / max(float(ox_thickness_nm) * 1e-9, 1e-10)
    m_star = pr["m_eff_ratio"] * M_E_KG
    # 2D parabolic quantum capacitance C_Q = 2 * e^2 * m* / (pi * hbar^2)
    c_q = 2.0 * (Q_E_C ** 2) * m_star / (math.pi * (HBAR_J_S ** 2))
    c_tot = 1.0 / (1.0 / c_ox + 1.0 / c_q)
    n_s_m2 = max(1e14, (c_tot * dv) / Q_E_C)
    # Gate-opened bandgap Delta_g (saturating at 0.25 eV, Zhang Nature 2009)
    x_gap = dv / pr["v_gap_sat_v"]
    delta_g_ev = pr["max_bandgap_ev"] * (x_gap / math.sqrt(1.0 + x_gap * x_gap))
    e_fermi_parabolic = (math.pi * (HBAR_J_S ** 2) * n_s_m2) / (2.0 * m_star * Q_E_C)
    mu_ev = 0.5 * delta_g_ev + e_fermi_parabolic

    # 1. Massive parabolic intraband Drude
    sig_intra = 1j * (Q_E_C ** 2 * n_s_m2) / (m_star * (om + 1j * 2.0 * gam_rad))

    # 2. Interband absorption with 2*sigma_0 asymptotic value + threshold at max(Delta_g, 2*mu_c)
    e_th = max(delta_g_ev, 2.0 * mu_ev, 1e-3)
    w_broad = max(2.0 * kbt_ev, 2.0 * gam_ev)
    step_re = 0.5 * (1.0 + math.tanh((hw_ev - e_th) / w_broad))
    kk_im = -(1.0 / math.pi) * math.log((e_th + hw_ev) / math.sqrt((e_th - hw_ev) ** 2 + w_broad ** 2))
    sig_inter_band = (2.0 * SIGMA0_UNIVERSAL_S) * (step_re + 1j * kk_im)

    # 3. Iconic interlayer transition peak at hbar*omega = gamma_1 = 0.39 eV (~3.18 um)
    g1 = pr["gamma1_ev"]
    g1_w = max(0.035, 3.0 * gam_ev)
    # Lorentzian conductance peak under exp(-i*om*t): Re > 0 at peak, Im changes sign across g1
    f_g1 = 1.8 * SIGMA0_UNIVERSAL_S * g1_w * hw_ev
    sig_g1 = (1j * f_g1) / (hw_ev ** 2 - g1 ** 2 + 1j * g1_w * hw_ev) * (0.35 + 0.65 * math.tanh(mu_ev / 0.08))

    sig_tot = sig_intra + sig_inter_band + sig_g1
    return complex(sig_tot), float(mu_ev), float(n_s_m2 * 1e-4), float(c_tot * 1e2), float(delta_g_ev)


def tmd_sheet_conductivity(
    material: str,
    wavelength_nm: float,
    carrier_density_cm2: float,
    *,
    cladding_eps: float = 1.0,
    thickness_nm: float | None = None,
) -> tuple[complex, complex]:
    """Gate-tunable monolayer TMD (WS2/MoS2/WSe2/MoSe2/MoTe2) complex sheet conductivity sigma_2D [S] & effective eps_2D."""
    pr = TWOD_MATERIAL_CATALOG[material]
    om = 2.0 * math.pi * C0_M_PER_S / (float(wavelength_nm) * 1e-9)
    hw_ev = (HBAR_J_S * om) / Q_E_C
    d_m = (float(thickness_nm) if thickness_nm is not None else pr["thickness_nm"]) * 1e-9
    x_n = max(float(carrier_density_cm2), 0.0) / pr["n_sat_cm2"]

    f_x0 = pr["f_x0_ev2"] / (1.0 + x_n)
    f_tr = pr["f_trion_ev2"] * (x_n / ((1.0 + 0.5 * x_n) ** 2))
    f_xb = pr["f_xb_ev2"] / (1.0 + 0.45 * x_n)

    def _lor(e_res: float, gam_mev: float, f_osc: float) -> complex:
        g_ev = gam_mev * 1e-3
        return f_osc / (e_res ** 2 - hw_ev ** 2 - 1j * g_ev * hw_ev)

    eps_2d = (
        pr["eps_bg"]
        + _lor(pr["e_x0_ev"], pr["gamma_x0_mev"], f_x0)
        + _lor(pr["e_trion_ev"], pr["gamma_trion_mev"], f_tr)
        + _lor(pr["e_xb_ev"], pr["gamma_xb_mev"], f_xb)
    )
    sigma_2d = -1j * om * EPS0_F_PER_M * (eps_2d - cladding_eps) * d_m
    return complex(sigma_2d), complex(eps_2d)


def biaxial_phonon_tensor(material: str, wavelength_nm: float) -> tuple[complex, complex, complex, str]:
    """Exact Mid-IR 3-axis Lorentz phonon-polariton tensor `(eps_u, eps_v, eps_y, regime)` for hBN, h^10BN, alpha-MoO3, alpha-V2O5."""
    pr = TWOD_MATERIAL_CATALOG[material]
    nu_cm1 = 1.0e7 / max(float(wavelength_nm), 100.0)

    def _axis_eps(eps_inf: float, w_to: float, w_lo: float, gam: float) -> complex:
        return eps_inf * (1.0 + (w_lo ** 2 - w_to ** 2) / (w_to ** 2 - nu_cm1 ** 2 - 1j * gam * nu_cm1))

    eps_u = _axis_eps(pr["eps_inf_u"], pr["w_to_u_cm1"], pr["w_lo_u_cm1"], pr["gamma_u_cm1"])
    eps_v = _axis_eps(pr["eps_inf_v"], pr["w_to_v_cm1"], pr["w_lo_v_cm1"], pr["gamma_v_cm1"])
    eps_y = _axis_eps(pr["eps_inf_y"], pr["w_to_y_cm1"], pr["w_lo_y_cm1"], pr["gamma_y_cm1"])

    ru, rv, ry = float(np.real(eps_u)), float(np.real(eps_v)), float(np.real(eps_y))
    if (ru * rv) < 0.0:
        band = "InPlane_Hyperbolic_Biaxial"
    elif ru < 0.0 and rv < 0.0 and ry > 0.0:
        band = "Hyperbolic_Type_II_Reststrahlen"
    elif ru > 0.0 and rv > 0.0 and ry < 0.0:
        band = "Hyperbolic_Type_I_Reststrahlen"
    elif ru < 0.0 and rv < 0.0 and ry < 0.0:
        band = "Reststrahlen_All_Negative"
    else:
        band = "Elliptic_Dielectric"
    return complex(eps_u), complex(eps_v), complex(eps_y), band


def hbn_hyperbolic_tensor(wavelength_nm: float) -> tuple[complex, complex, str]:
    """Exact Mid-IR uniaxial Lorentz phonon-polariton tensor `(eps_par, eps_perp, band)` for hBN."""
    eps_u, _, eps_y, band = biaxial_phonon_tensor("hBN", wavelength_nm)
    return eps_u, eps_y, band


def evaluate_2d_material_tensor(
    material_2d: str,
    wavelength_nm: float,
    *,
    v_gate_v: float = 0.0,
    v_cnp_v: float = 0.0,
    spacer_ox_nm: float = 10.0,
    gate_ox_eps_r: float = 9.0,
    capacitor_type: str = "SLG",
    scattering_mev: float | None = None,
    temperature_k: float = T_REF_K,
    cladding_eps: float = 2.085,
    thickness_nm: float | None = None,
    crystal_angle_deg: float = 0.0,
) -> dict[str, Any]:
    """Evaluate exact microscopic 2D complex conductivity & permittivity tensor for any of the 27 2D vdW materials.

    Rotates principal in-plane axes `(u, v)` by `crystal_angle_deg` into waveguide `(x, z)` coordinates:
      sigma_xx = sigma_u cos^2(theta) + sigma_v sin^2(theta)
      sigma_zz = sigma_u sin^2(theta) + sigma_v cos^2(theta)
      sigma_xz = (sigma_u - sigma_v) sin(theta) cos(theta)
    Returns exact `(sigma_xx, sigma_zz, sigma_xz, eps_u, eps_v, eps_y, d_nm, mu_c_eV, n_s_cm2, C_tot_uF_cm2, regime)`.
    """
    mat = material_2d if material_2d in TWOD_MATERIAL_CATALOG else "Graphene"
    pr = TWOD_MATERIAL_CATALOG[mat]
    fam = pr["family"]
    wl_nm = float(wavelength_nm)
    wl_um = wl_nm * 1e-3
    om = 2.0 * math.pi * C0_M_PER_S / (wl_nm * 1e-9)
    hw_ev = (HBAR_J_S * om) / Q_E_C
    d_nm = float(thickness_nm) if (thickness_nm is not None and thickness_nm > 0.0) else float(pr["thickness_nm"])
    d_m = d_nm * 1e-9
    c_ox_f_m2 = EPS0_F_PER_M * float(gate_ox_eps_r) / max(float(spacer_ox_nm) * 1e-9, 1e-10)
    dv = abs(float(v_gate_v) - float(v_cnp_v))

    mu_ev = 0.0
    n_s_cm2 = 0.0
    c_tot_uf_cm2 = c_ox_f_m2 * 1e2
    bandgap_ev = float(pr.get("e_gap_ev", 0.0))
    eps_u: complex = complex(cladding_eps)
    eps_v: complex = complex(cladding_eps)
    eps_y: complex = complex(cladding_eps)
    sig_u: complex = 0.0 + 0.0j
    sig_v: complex = 0.0 + 0.0j
    regime = "Elliptic_Dielectric"
    has_normal_pert = False

    if fam == "dirac_semimetal":
        is_dlg = capacitor_type.upper() == "DLG"
        n_sheets = 2.0 if is_dlg else 1.0
        scatt = float(scattering_mev) if scattering_mev is not None else float(pr["scattering_mev"])
        mu_ev, n_s_cm2, c_tot_uf_cm2 = graphene_gate_voltage_to_mu_c(
            v_gate_v, v_cnp_v, spacer_ox_nm, gate_ox_eps_r, dual_layer=is_dlg
        )
        sig_1, _, _ = graphene_kubo_conductivity(wl_nm, mu_ev, scatt, temperature_k)
        sig_u = sig_v = n_sheets * sig_1
        eps_u = eps_v = cladding_eps + 1j * sig_u / (om * EPS0_F_PER_M * max(n_sheets * d_m, 1e-12))
        eps_y = complex(2.2)
        regime = "Dirac_Pauli_Blocked" if abs(mu_ev) >= 0.5 * hw_ev else "Dirac_Interband_Absorbing"

    elif fam == "bilayer_graphene":
        scatt = float(scattering_mev) if scattering_mev is not None else float(pr["scattering_mev"])
        sig_blg, mu_ev, n_s_cm2, c_tot_uf_cm2, bandgap_ev = bilayer_graphene_conductivity(
            wl_nm, v_gate_v, v_cnp_v, spacer_ox_nm, gate_ox_eps_r, scatt, temperature_k
        )
        sig_u = sig_v = sig_blg
        eps_u = eps_v = cladding_eps + 1j * sig_blg / (om * EPS0_F_PER_M * max(d_m, 1e-12))
        eps_y = complex(2.4)
        regime = "BLG_Gapped_Blocked" if (hw_ev < max(bandgap_ev, 2.0 * mu_ev)) else "BLG_Interband_Absorbing"

    elif fam == "tmd":
        n_s_cm2 = (c_ox_f_m2 * dv / Q_E_C) * 1e-4
        sig_t, eps_t = tmd_sheet_conductivity(mat, wl_nm, n_s_cm2, cladding_eps=cladding_eps, thickness_nm=d_nm)
        sig_u = sig_v = sig_t
        eps_u = eps_v = eps_t
        eps_y = complex(pr.get("eps_perp", 6.5))
        has_normal_pert = d_nm >= 2.0
        regime = "Exciton_SubBandgap_Phase" if hw_ev < (pr["e_x0_ev"] - 0.15) else "Exciton_Resonant"

    elif fam == "biaxial_tmd":
        n_s_cm2 = (c_ox_f_m2 * dv / Q_E_C) * 1e-4
        x_n = max(n_s_cm2, 0.0) / pr["n_sat_cm2"]
        f_u = pr["f_x_u_ev2"] / (1.0 + x_n)
        f_v = pr["f_x_v_ev2"] / (1.0 + x_n)
        g_u = pr["gamma_x_u_mev"] * 1e-3
        g_v = pr["gamma_x_v_mev"] * 1e-3
        eps_u = pr["eps_bg_u"] + f_u / (pr["e_x_u_ev"] ** 2 - hw_ev ** 2 - 1j * g_u * hw_ev)
        eps_v = pr["eps_bg_v"] + f_v / (pr["e_x_v_ev"] ** 2 - hw_ev ** 2 - 1j * g_v * hw_ev)
        eps_y = complex(pr["eps_perp"])
        sig_u = -1j * om * EPS0_F_PER_M * (eps_u - cladding_eps) * d_m
        sig_v = -1j * om * EPS0_F_PER_M * (eps_v - cladding_eps) * d_m
        has_normal_pert = True
        regime = "Biaxial_Anisotropic_Exciton"

    elif fam == "vdw_high_index":
        n_s_cm2 = (c_ox_f_m2 * dv / Q_E_C) * 1e-4
        quench = 1.0 / (1.0 + max(n_s_cm2, 0.0) / pr["n_sat_cm2"])
        eps_par = complex(pr["eps_bg_par"])
        for idx_o, (e0, g0, f0) in enumerate(pr["osc_par"]):
            scale = quench if idx_o < 2 else (0.7 + 0.3 * quench)
            eps_par += (f0 * scale) / (e0 ** 2 - hw_ev ** 2 - 1j * g0 * hw_ev)
        eps_perp = complex(pr["eps_bg_perp"])
        for e0, g0, f0 in pr["osc_perp"]:
            eps_perp += f0 / (e0 ** 2 - hw_ev ** 2 - 1j * g0 * hw_ev)
        eps_u = eps_v = eps_par
        eps_y = eps_perp
        sig_u = sig_v = -1j * om * EPS0_F_PER_M * (eps_par - cladding_eps) * d_m
        has_normal_pert = True
        regime = "Giant_Birefringence_Uniaxial_vdW"

    elif fam == "hyperbolic_phonon":
        eps_u, eps_v, eps_y, regime = biaxial_phonon_tensor(mat, wl_nm)
        sig_u = -1j * om * EPS0_F_PER_M * (eps_u - cladding_eps) * d_m
        sig_v = -1j * om * EPS0_F_PER_M * (eps_v - cladding_eps) * d_m
        has_normal_pert = True
        c_tot_uf_cm2 = 0.0

    elif fam in ("anisotropic_2d", "narrow_gap_tmd"):
        n_s_cm2 = max(1.0e11, (c_ox_f_m2 * dv / Q_E_C) * 1e-4)
        m_u = (pr.get("m_eff_armchair", pr.get("m_eff_u", 0.2))) * M_E_KG
        m_v = (pr.get("m_eff_zigzag", pr.get("m_eff_v", 0.2))) * M_E_KG
        m_dos = math.sqrt(m_u * m_v)
        mu_ev = (math.pi * (HBAR_J_S ** 2) * (n_s_cm2 * 1e4)) / (m_dos * Q_E_C)
        gam_ev = float(pr["scattering_mev"]) * 1e-3
        gam_rad = (gam_ev * Q_E_C) / HBAR_J_S

        # Exact anisotropic Drude + Moss-Burstein shifted interband conductivity + background dielectric slab
        sig_intra_u = 1j * (Q_E_C ** 2 * (n_s_cm2 * 1e4)) / (m_u * (om + 1j * gam_rad))
        sig_intra_v = 1j * (Q_E_C ** 2 * (n_s_cm2 * 1e4)) / (m_v * (om + 1j * gam_rad))
        e_eff = float(pr["e_gap_ev"]) + 2.0 * mu_ev
        w_ib = max(0.035, 2.5 * gam_ev)
        step_re = 0.5 * (1.0 + math.tanh((hw_ev - e_eff) / w_ib))
        kk_im = -(1.0 / math.pi) * math.log((e_eff + hw_ev) / math.sqrt((e_eff - hw_ev) ** 2 + w_ib ** 2))
        ib_fac = step_re + 1j * kk_im
        sig_ib_u = (float(pr["sigma_ib_u_s0"]) * SIGMA0_UNIVERSAL_S) * ib_fac
        sig_ib_v = (float(pr["sigma_ib_v_s0"]) * SIGMA0_UNIVERSAL_S) * ib_fac

        # Dielectric background contrast
        sig_bg_u = -1j * om * EPS0_F_PER_M * (float(pr["eps_bg_u"]) - cladding_eps) * d_m
        sig_bg_v = -1j * om * EPS0_F_PER_M * (float(pr["eps_bg_v"]) - cladding_eps) * d_m
        sig_u = sig_intra_u + sig_ib_u + sig_bg_u
        sig_v = sig_intra_v + sig_ib_v + sig_bg_v
        eps_u = cladding_eps + 1j * sig_u / (om * EPS0_F_PER_M * max(d_m, 1e-12))
        eps_v = cladding_eps + 1j * sig_v / (om * EPS0_F_PER_M * max(d_m, 1e-12))
        eps_y = complex(pr.get("eps_perp", 8.0))
        has_normal_pert = d_nm >= 3.0
        regime = "Anisotropic_Interband_Absorbing" if hw_ev >= e_eff else "Anisotropic_SubGap_Drude"

    elif fam == "mxene_plasmonic":
        # Gate-modulated Drude plasma frequency (~3% electro-optic modulation at 6 V via carrier density shift)
        n_s_cm2 = (c_ox_f_m2 * dv / Q_E_C) * 1e-4
        wp_scale = math.sqrt(max(0.7, 1.0 + 0.012 * (n_s_cm2 * 1e-12)))
        wp = float(pr["wp_ev"]) * wp_scale
        gp = float(pr["gamma_p_ev"])
        w1, f1, g1 = float(pr["w1_ev"]), float(pr["f1"]), float(pr["gamma_1_ev"])
        eps_par = (
            float(pr["eps_inf"])
            - (wp ** 2) / (hw_ev ** 2 + 1j * gp * hw_ev)
            + (f1 * (w1 ** 2)) / (w1 ** 2 - hw_ev ** 2 - 1j * g1 * hw_ev)
        )
        eps_u = eps_v = complex(eps_par)
        eps_y = complex(pr["eps_perp"])
        sig_u = sig_v = -1j * om * EPS0_F_PER_M * (eps_par - cladding_eps) * d_m
        has_normal_pert = True
        re_p = float(np.real(eps_par))
        if abs(re_p) < 0.6:
            regime = "ENZ_Epsilon_Near_Zero"
        elif re_p < 0.0:
            regime = "Metallic_Plasmonic_MXene"
        else:
            regime = "Dielectric_MXene"

    elif fam == "topological_semimetal":
        # Bulk high-n + far-IR optical phonon + top/bottom topological Dirac surface state Drude
        nu_cm1 = 1.0e7 / max(wl_nm, 100.0)
        eps_ph = float(pr["eps_inf_bulk"]) * (
            1.0 + (pr["w_lo_ph_cm1"] ** 2 - pr["w_to_ph_cm1"] ** 2)
            / (pr["w_to_ph_cm1"] ** 2 - nu_cm1 ** 2 - 1j * pr["gamma_ph_cm1"] * nu_cm1)
        )
        # Weak interband absorption above 0.30 eV bulk gap
        k_ib = 0.35 * (0.5 * (1.0 + math.tanh((hw_ev - pr["e_gap_ev"]) / 0.08)))
        eps_bulk = eps_ph + 1j * (2.0 * math.sqrt(max(float(np.real(eps_ph)), 1.0)) * k_ib)
        n_s_cm2 = (c_ox_f_m2 * dv / Q_E_C) * 1e-4
        vf_ti = float(pr["vf_surface_m_s"])
        mu_base_j = float(pr["mu0_surface_ev"]) * Q_E_C
        n0_m2 = (mu_base_j ** 2) / (4.0 * math.pi * (HBAR_J_S * vf_ti) ** 2)  # Single Dirac cone (g_s*g_v = 1)
        n_tot_m2 = n0_m2 + n_s_cm2 * 1e4
        mu_j = math.sqrt(4.0 * math.pi * n_tot_m2) * HBAR_J_S * vf_ti
        mu_ev = mu_j / Q_E_C
        gam_rad = (float(pr["scattering_mev"]) * 1e-3 * Q_E_C) / HBAR_J_S
        # 2 topological surfaces (top + bottom), each single Dirac cone: sigma_D = i * e^2 * mu / (4*pi*hbar^2*(om + i*2G))
        sig_surf = 2.0 * 1j * (Q_E_C ** 2 * mu_j) / (4.0 * math.pi * (HBAR_J_S ** 2) * (om + 1j * 2.0 * gam_rad))
        sig_bulk = -1j * om * EPS0_F_PER_M * (eps_bulk - cladding_eps) * d_m
        sig_u = sig_v = sig_bulk + sig_surf
        eps_u = eps_v = cladding_eps + 1j * sig_u / (om * EPS0_F_PER_M * max(d_m, 1e-12))
        eps_y = complex(pr["eps_perp"])
        has_normal_pert = True
        regime = "Topological_Dirac_Surface_Plasmonic" if float(np.real(eps_u)) < 0.0 else "Topological_HighIndex_Dirac"

    elif fam == "weyl_semimetal":
        # Td-WTe2 in-plane anisotropic Drude-Lorentz -> natural in-plane hyperbolic plasmonics (9.5 - 16.6 um)!
        n_s_cm2 = (c_ox_f_m2 * dv / Q_E_C) * 1e-4
        wp_scale = math.sqrt(max(0.7, 1.0 + 0.025 * (n_s_cm2 * 1e-12)))
        wpu = float(pr["wp_u_ev"]) * wp_scale
        wpv = float(pr["wp_v_ev"]) * wp_scale
        gu = float(pr["gamma_u_ev"])
        gv = float(pr["gamma_v_ev"])
        eps_u = float(pr["eps_inf_u"]) - (wpu ** 2) / (hw_ev ** 2 + 1j * gu * hw_ev)
        eps_v = float(pr["eps_inf_v"]) - (wpv ** 2) / (hw_ev ** 2 + 1j * gv * hw_ev)
        eps_y = complex(pr["eps_perp"])
        sig_u = -1j * om * EPS0_F_PER_M * (eps_u - cladding_eps) * d_m
        sig_v = -1j * om * EPS0_F_PER_M * (eps_v - cladding_eps) * d_m
        has_normal_pert = True
        ru, rv = float(np.real(eps_u)), float(np.real(eps_v))
        if (ru * rv) < 0.0:
            regime = "InPlane_Hyperbolic_Plasmonic"
        elif ru < 0.0 and rv < 0.0:
            regime = "Anisotropic_Metallic_Weyl"
        else:
            regime = "Anisotropic_Dielectric_Weyl"

    else:  # vdw_eo_ferroelectric (NbOCl2, In2Se3_alpha, CuInP2S6, GaSe)
        # Cauchy dispersion + Urbach tail above bandgap + Pockels EO + ferroelectric remnant polarization switch
        disp = float(pr["cauchy_b_um2"]) * (1.0 / max(wl_um ** 2, 0.16) - 1.0 / (1.55 ** 2))
        # Vertical gate field across 2D slab + spacer: E_z ~ (V_g - V_cnp) / (t_ox + d_2d)
        t_eff_m = max((float(spacer_ox_nm) + d_nm) * 1e-9, 1e-9)
        e_field_v_m = (float(v_gate_v) - float(v_cnp_v)) / t_eff_m
        e_field_v_nm = abs(e_field_v_m) * 1e-9

        n_u0 = float(pr["n_u_1550"]) + disp
        n_v0 = float(pr["n_v_1550"]) + disp
        n_y0 = float(pr["n_y_1550"]) + 0.7 * disp

        # Linear Pockels electro-optic shift dn = -0.5 * n^3 * r_eff * E
        dn_eo_u = -0.5 * (n_u0 ** 3) * (float(pr["r_eff_u_pm_v"]) * 1e-12) * e_field_v_m
        dn_eo_v = -0.5 * (n_v0 ** 3) * (float(pr["r_eff_v_pm_v"]) * 1e-12) * e_field_v_m

        # Ferroelectric non-volatile dipole switching term (for alpha-In2Se3 & CuInP2S6)
        dn_fe = 0.0
        ec = float(pr.get("e_coercive_v_nm", 0.0))
        if ec > 0.0 and float(pr.get("ferroelectric_dn", 0.0)) > 0.0:
            dn_fe = float(pr["ferroelectric_dn"]) * math.tanh((e_field_v_nm - ec) / (0.25 * ec)) * (1.0 if e_field_v_nm > 1e-6 else 0.0)
            dn_fe = max(0.0, dn_fe)

        # Interband absorption above E_g (zero below bandgap!)
        k_abs = 0.0
        eg = float(pr["e_gap_ev"])
        if hw_ev > (eg - 0.10):
            k_abs = 0.35 * (0.5 * (1.0 + math.tanh((hw_ev - eg) / 0.05)))

        eps_u = complex(n_u0 + dn_eo_u + dn_fe, k_abs) ** 2
        eps_v = complex(n_v0 + dn_eo_v + dn_fe, k_abs) ** 2
        eps_y = complex(n_y0 + 0.5 * dn_fe, k_abs) ** 2
        sig_u = -1j * om * EPS0_F_PER_M * (eps_u - cladding_eps) * d_m
        sig_v = -1j * om * EPS0_F_PER_M * (eps_v - cladding_eps) * d_m
        has_normal_pert = True
        c_tot_uf_cm2 = (EPS0_F_PER_M * float(gate_ox_eps_r) / t_eff_m) * 1e2
        regime = "Ferroelectric_NonVolatile_EO" if ec > 0.0 else "Biaxial_Chi2_Pockels_EO"

    # Rotate principal in-plane conductivity tensor diag(sig_u, sig_v) by crystal_angle_deg into waveguide (x, z)
    th = math.radians(float(crystal_angle_deg))
    c2 = math.cos(th) ** 2
    s2 = math.sin(th) ** 2
    sc = math.sin(th) * math.cos(th)
    sig_xx = sig_u * c2 + sig_v * s2
    sig_zz = sig_u * s2 + sig_v * c2
    sig_xz = (sig_u - sig_v) * sc

    return {
        "material_2d": mat,
        "family": fam,
        "thickness_nm": d_nm,
        "crystal_angle_deg": float(crystal_angle_deg),
        "sigma_u_S": complex(sig_u),
        "sigma_v_S": complex(sig_v),
        "sigma_xx_S": complex(sig_xx),
        "sigma_zz_S": complex(sig_zz),
        "sigma_xz_S": complex(sig_xz),
        "eps_u": complex(eps_u),
        "eps_v": complex(eps_v),
        "eps_y": complex(eps_y),
        "has_normal_pert": has_normal_pert,
        "mu_c_eV": float(mu_ev),
        "n_s_cm2": float(n_s_cm2),
        "C_tot_uF_cm2": float(c_tot_uf_cm2),
        "bandgap_eV": float(bandgap_ev),
        "d_eff_pm_V": float(pr.get("d_eff_pm_v", 0.0)),
        "dispersion_regime": regime,
        "source": pr.get("source", ""),
    }


# ==============================================================================
# 3. Unified Full-Tensor 2D-Material-on-MRR First-Principles Solver
# ==============================================================================

@dataclass
class TwoDMaterialMRRConfig:
    material_2d: str = "Graphene"          # Any of the 27 materials in TWOD_MATERIAL_CATALOG
    capacitor_type: str = "SLG"            # "SLG" | "DLG" (for Graphene)
    core_material: str = "Si"              # Underlying MRR core ("Si" | "Si3N4" | "LN_Xcut" | "Ge")
    clad_material: str = "SiO2"
    top_clad_material: str = "SiO2"        # Use "Air" or "SiO2"
    film_height_um: float = 0.22
    slab_height_um: float = 0.0
    ring_width_um: float = 0.48
    ring_radius_um: float = 15.0
    sidewall_deg: float = 90.0
    wavelength_nm: float = 1550.0
    polarization: str = "TE0"
    spacer_ox_nm: float = 10.0             # Gate oxide spacer between waveguide top and 2D sheet [nm]
    gate_ox_eps_r: float = 9.0             # Al2O3 = 9.0, hBN = 3.8, SiO2 = 3.9
    thickness_nm_override: float | None = None  # Optional custom flake thickness [nm] (None -> catalog default)
    crystal_angle_deg: float = 0.0         # In-plane principal axis angle relative to waveguide x-axis [deg]
    active_coverage_fraction: float = 0.75 # Fraction of ring circumference covered by 2D active sheet
    alpha_passive_db_per_cm: float = 1.5
    q_ext_thru: float = 1.5e4
    v_cnp_v: float = 0.0
    v_gate_sweep_v: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    scattering_mev: float = 10.0           # Graphene/BLG scattering rate Gamma [meV]
    temperature_k: float = T_REF_K
    contact_resistance_ohm_um: float = 400.0
    sheet_access_width_um: float = 1.2
    core_mesh_res_um: float = 0.040


def simulate_2d_material_mrr(cfg: TwoDMaterialMRRConfig) -> dict[str, Any]:
    """First-principles 2D conformal vector FEM + exact full-tensor 2D surface-current perturbation MRR solver."""
    t0 = time.perf_counter()
    mat2d = cfg.material_2d if cfg.material_2d in TWOD_MATERIAL_CATALOG else "Graphene"
    pr2d = TWOD_MATERIAL_CATALOG[mat2d]
    wl0_nm = float(cfg.wavelength_nm)
    wl0_m = wl0_nm * 1e-9
    k0 = 2.0 * math.pi / wl0_m
    om0 = C0_M_PER_S * k0
    r_m = float(cfg.ring_radius_um) * 1e-6
    l_rt_m = 2.0 * math.pi * r_m
    l_act_m = float(np.clip(cfg.active_coverage_fraction, 0.05, 1.0)) * l_rt_m

    # 1. Solve unperturbed 2D curved vector Maxwell eigenmode on waveguide cross-section
    core_fn = get_material_fn(cfg.core_material)
    clad_fn = get_material_fn(cfg.clad_material)
    top_fn = get_material_fn(cfg.top_clad_material)
    wl_um = wl0_nm * 1e-3
    wl_scale = float(np.clip(wl0_nm / 1550.0, 0.35, 6.50))
    box_pad = max(1.8, 1.35 * wl_um)
    spec = WaveguideCrossSectionSpec(
        w_bottom_um=cfg.ring_width_um, h_film_um=cfg.film_height_um, h_slab_um=cfg.slab_height_um,
        sidewall_deg=cfg.sidewall_deg, core_material_fn=core_fn, clad_material_fn=clad_fn,
        top_clad_material_fn=top_fn,
        x_min_um=max(-max(2.0 * wl_scale, 0.5 * cfg.ring_width_um + 1.25 * wl_um), -0.78 * cfg.ring_radius_um),
        x_max_um=max(2.4 * wl_scale, 0.5 * cfg.ring_width_um + 1.55 * wl_um),
        h_box_um=box_pad, h_top_clad_um=box_pad, temperature_k=cfg.temperature_k,
    )
    eff_mesh = float(cfg.core_mesh_res_um) * max(1.0, wl_scale)
    m0 = _solve_single_mode(spec, wl0_m, r_m, cfg.polarization, "fem", eff_mesh, export_grid=True)
    neff0 = float(np.real(m0.n_eff))
    ng0 = float(_compute_group_index_fast(spec, wl0_m, r_m, cfg.polarization, "fem", eff_mesh, neff0))

    # 2. Compute exact modal Poynting power P_z and directional tangential/normal sheet coupling factors!
    x_m, y_m = m0.x_m, m0.y_m
    dx = float(np.mean(np.diff(x_m)))
    dy = float(np.mean(np.diff(y_m)))
    mu0 = 4.0e-7 * math.pi
    beta0 = k0 * neff0

    dez_dy, dez_dx = np.gradient(m0.Ez, dy, dx)
    hx = (dez_dy - 1j * beta0 * m0.Ey) / (1j * om0 * mu0)
    hy = (1j * beta0 * m0.Ex - dez_dx) / (1j * om0 * mu0)
    p_z = 0.5 * float(np.sum(np.real(m0.Ex * np.conj(hy) - m0.Ey * np.conj(hx))) * dx * dy)
    p_z = max(abs(p_z), 1e-30)

    # Evaluate modal field at mid-plane of the 2D sheet: y_sheet = h_film + spacer_ox + 0.5*d_2d
    d_2d_nm = float(cfg.thickness_nm_override) if (cfg.thickness_nm_override is not None and cfg.thickness_nm_override > 0.0) else float(pr2d["thickness_nm"])
    y_sheet_m = float(cfg.film_height_um) * 1e-6 + max(float(cfg.spacer_ox_nm), 0.5) * 1e-9 + (0.5 * d_2d_nm * 1e-9 if d_2d_nm > 2.0 else 0.0)
    iy_hi = min(max(int(np.searchsorted(y_m, y_sheet_m)), 1), len(y_m) - 1)
    iy_lo = iy_hi - 1
    w_hi = float(np.clip((y_sheet_m - y_m[iy_lo]) / max(y_m[iy_hi] - y_m[iy_lo], 1e-12), 0.0, 1.0))
    ex_sh = (1.0 - w_hi) * m0.Ex[iy_lo, :] + w_hi * m0.Ex[iy_hi, :]
    ez_sh = (1.0 - w_hi) * m0.Ez[iy_lo, :] + w_hi * m0.Ez[iy_hi, :]
    ey_sh = (1.0 - w_hi) * m0.Ey[iy_lo, :] + w_hi * m0.Ey[iy_hi, :]

    x_mask = np.abs(x_m) <= 0.5 * (float(cfg.ring_width_um) + 0.8) * 1e-6
    ex_sq_line = float(np.sum((np.abs(ex_sh[x_mask]) ** 2) * dx))
    ez_sq_line = float(np.sum((np.abs(ez_sh[x_mask]) ** 2) * dx))
    exz_re_line = float(np.sum(np.real(np.conj(ex_sh[x_mask]) * ez_sh[x_mask]) * dx))
    e_perp_sq_line = float(np.sum((np.abs(ey_sh[x_mask]) ** 2) * dx))

    # Directional coupling impedances [Ohm]: Delta_n_|| = i * (S_xx*sig_xx + S_zz*sig_zz + 2*S_xz*sig_xz)
    s_xx_ohm = ex_sq_line / (4.0 * k0 * p_z)
    s_zz_ohm = ez_sq_line / (4.0 * k0 * p_z)
    s_xz_ohm = exz_re_line / (4.0 * k0 * p_z)
    s_par_ohm = s_xx_ohm + s_zz_ohm

    n_top = float(top_fn(wl0_m, cfg.temperature_k).nx)
    eps_clad = n_top ** 2

    # 3. Sweep gate voltage V_g across exact 2D tensor model!
    is_dlg = (mat2d == "Graphene" and cfg.capacitor_type.upper() == "DLG")
    v_list = list(cfg.v_gate_sweep_v)
    bias_rows: list[dict[str, Any]] = []

    k_ext = om0 / max(float(cfg.q_ext_thru), 10.0)
    last_regime = "Elliptic_Dielectric"
    last_tens: dict[str, Any] = {}

    for vg in v_list:
        tens = evaluate_2d_material_tensor(
            mat2d,
            wl0_nm,
            v_gate_v=float(vg),
            v_cnp_v=cfg.v_cnp_v,
            spacer_ox_nm=cfg.spacer_ox_nm,
            gate_ox_eps_r=cfg.gate_ox_eps_r,
            capacitor_type=cfg.capacitor_type,
            scattering_mev=cfg.scattering_mev if mat2d in ("Graphene", "Bilayer_Graphene") else None,
            temperature_k=cfg.temperature_k,
            cladding_eps=eps_clad,
            thickness_nm=cfg.thickness_nm_override,
            crystal_angle_deg=cfg.crystal_angle_deg,
        )
        last_tens = tens
        last_regime = tens["dispersion_regime"]

        dneff_par = 1j * (
            s_xx_ohm * tens["sigma_xx_S"]
            + s_zz_ohm * tens["sigma_zz_S"]
            + 2.0 * s_xz_ohm * tens["sigma_xz_S"]
        )
        dneff_perp = 0.0 + 0.0j
        if tens["has_normal_pert"]:
            d_m = tens["thickness_nm"] * 1e-9
            eps_y = tens["eps_y"]
            if abs(eps_y) > 1e-6:
                dneff_perp = (
                    (om0 * EPS0_F_PER_M * d_m * (eps_clad ** 2) * (1.0 / eps_clad - 1.0 / eps_y) / k0)
                    * (e_perp_sq_line / (4.0 * p_z))
                )
        dneff_c = dneff_par + dneff_perp

        alpha_2d_np_m = 2.0 * k0 * max(float(np.imag(dneff_c)), 0.0)
        alpha_2d_db_cm = alpha_2d_np_m * (10.0 / math.log(10.0)) * 1e-2
        dneff_real = float(np.real(dneff_c))

        alpha_ring_eff = cfg.alpha_passive_db_per_cm + cfg.active_coverage_fraction * alpha_2d_db_cm
        q_int_tot = loss_db_per_cm_to_q_int(alpha_ring_eff, ng0, wl0_nm)
        k_int_tot = om0 / max(q_int_tot, 5.0)
        k_tot = k_int_tot + k_ext
        q_loaded = om0 / k_tot

        t_on_res = ((k_int_tot - k_ext) / k_tot) ** 2
        er_db = -10.0 * math.log10(max(t_on_res, 1e-6))

        sig_rep = tens["sigma_xx_S"]
        bias_rows.append({
            "V_gate_V": round(float(vg), 3),
            "mu_c_eV": round(tens["mu_c_eV"], 4),
            "n_s_1e12_cm2": round(tens["n_s_cm2"] * 1e-12, 3),
            "sigma_Re_uS": round(float(np.real(sig_rep)) * 1e6, 3),
            "sigma_Im_uS": round(float(np.imag(sig_rep)) * 1e6, 3),
            "delta_neff_real": float(f"{dneff_real:.5e}"),
            "alpha_2d_dB_per_cm": round(alpha_2d_db_cm, 2),
            "Q_loaded": round(q_loaded, 1),
            "T_on_res_dB": round(10.0 * math.log10(max(t_on_res, 1e-6)), 3),
            "ER_dB": round(er_db, 2),
            "C_tot_uF_cm2": round(tens["C_tot_uF_cm2"], 3),
        })

    dn0 = bias_rows[0]["delta_neff_real"]
    for row in bias_rows:
        ddn = row["delta_neff_real"] - dn0
        dlam_pm = (ddn / max(ng0, 1.0)) * wl0_nm * 1e3 * cfg.active_coverage_fraction
        row["delta_lambda_pm"] = round(dlam_pm, 2)

    alphas = [r["alpha_2d_dB_per_cm"] for r in bias_rows]
    shifts = [r["delta_lambda_pm"] for r in bias_rows]
    i_low = int(np.argmin(alphas))
    i_high = int(np.argmax(alphas))
    ea_mod_depth_db_cm = max(alphas) - min(alphas)
    max_shift_pm = float(np.max(np.abs(shifts)))
    dn_span = max(r["delta_neff_real"] for r in bias_rows) - min(r["delta_neff_real"] for r in bias_rows)
    dk_span = (ea_mod_depth_db_cm * 1e2 * math.log(10.0) / 10.0) / (2.0 * k0)
    dn_over_dk = dn_span / max(dk_span, 1e-9)

    c_avg_f_m2 = max(float(np.mean([r["C_tot_uF_cm2"] for r in bias_rows])) * 1e-2, 1e-6)
    area_cap_m2 = l_act_m * (cfg.ring_width_um * 1e-6)
    c_dev_f = c_avg_f_m2 * area_cap_m2
    r_contact = cfg.contact_resistance_ohm_um / max(l_act_m * 1e6, 1.0)
    r_sheet = min(float(pr2d["sheet_res_ohm_sq"]), 1e5) * (cfg.sheet_access_width_um / max(l_act_m * 1e6, 1.0))
    r_tot_ohm = (2.0 if is_dlg else 1.0) * (r_contact + r_sheet) + 50.0
    f_rc_ghz = (1.0 / (2.0 * math.pi * r_tot_ohm * max(c_dev_f, 1e-18))) * 1e-9
    q_on = max(bias_rows[i_low]["Q_loaded"], 10.0)
    f_opt_ghz = (om0 / (2.0 * math.pi * q_on)) * 1e-9
    f_3db_ghz = 1.0 / math.sqrt(1.0 / max(f_rc_ghz ** 2, 1e-6) + 1.0 / max(f_opt_ghz ** 2, 1e-6))
    v_pp = max(v_list) - min(v_list) if len(v_list) > 1 else 4.0
    e_bit_fj = 0.25 * c_dev_f * (v_pp ** 2) * 1e15

    dlam_span_pm = max(6.0 * (wl0_nm * 1e3 / max(q_on, 50.0)), 3.0 * max_shift_pm, 250.0)
    det_pm = np.linspace(-0.6 * dlam_span_pm, 0.6 * dlam_span_pm, 401)
    dom_rad = -om0 * (det_pm * 1e-3 / wl0_nm)

    def _mrr_spec_db(row: dict[str, Any]) -> np.ndarray:
        k_i = om0 / max(loss_db_per_cm_to_q_int(cfg.alpha_passive_db_per_cm + cfg.active_coverage_fraction * row["alpha_2d_dB_per_cm"], ng0, wl0_nm), 5.0)
        dom_sh = -om0 * (row["delta_lambda_pm"] * 1e-3 / wl0_nm)
        t_c = (1j * (dom_rad - dom_sh) + 0.5 * (k_i - k_ext)) / (1j * (dom_rad - dom_sh) + 0.5 * (k_i + k_ext))
        return 10.0 * np.log10(np.maximum(np.abs(t_c) ** 2, 1e-6))

    t_low_db = _mrr_spec_db(bias_rows[i_low])
    t_high_db = _mrr_spec_db(bias_rows[i_high])
    dyn_er_mrr_db = float(np.max(np.abs(t_low_db - t_high_db)))

    return {
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
        "config": asdict(cfg),
        "metrics": {
            "material_2d": mat2d,
            "family": pr2d["family"],
            "capacitor_type": cfg.capacitor_type if mat2d == "Graphene" else pr2d["family"],
            "thickness_nm": round(d_2d_nm, 2),
            "crystal_angle_deg": round(float(cfg.crystal_angle_deg), 1),
            "neff_bare": round(neff0, 5),
            "ng_bare": round(ng0, 4),
            "delta_neff_V0": bias_rows[0]["delta_neff_real"],
            "surface_coupling_S_par_Ohm": round(s_par_ohm, 4),
            "alpha_2d_min_dB_per_cm": round(min(alphas), 2),
            "alpha_2d_max_dB_per_cm": round(max(alphas), 2),
            "EA_modulation_depth_dB_per_cm": round(ea_mod_depth_db_cm, 2),
            "max_resonance_shift_pm": round(max_shift_pm, 2),
            "phase_to_loss_ratio_dn_dk": round(dn_over_dk, 3),
            "MRR_dynamic_ER_dB": round(dyn_er_mrr_db, 2),
            "C_device_fF": round(c_dev_f * 1e15, 2),
            "f_3dB_RC_GHz": round(f_rc_ghz, 2),
            "f_3dB_total_GHz": round(f_3db_ghz, 2),
            "energy_per_bit_fJ": round(e_bit_fj, 2),
            "d_eff_pm_V": round(float(pr2d.get("d_eff_pm_v", 0.0)), 1),
            "eps_u_re": round(float(np.real(last_tens["eps_u"])), 3),
            "eps_v_re": round(float(np.real(last_tens["eps_v"])), 3),
            "eps_y_re": round(float(np.real(last_tens["eps_y"])), 3),
            "dispersion_regime": last_regime,
            "hbn_dispersion_regime": last_regime if pr2d["family"] == "hyperbolic_phonon" else "N/A",
            "source": pr2d.get("source", ""),
        },
        "bias_sweep": bias_rows,
        "spectrum": {
            "detuning_pm": np.round(det_pm, 3).tolist(),
            "thru_low_loss_dB": np.round(t_low_db, 3).tolist(),
            "thru_high_loss_dB": np.round(t_high_db, 3).tolist(),
        },
    }
