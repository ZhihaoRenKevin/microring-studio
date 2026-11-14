"""Comprehensive Integrated Photonics Material Optical Property Library.

Clean-room implementation from peer-reviewed literature (no proprietary vendor
material files used). Covers 14+ core waveguide, cladding, and active materials:
  - Group IV: Crystalline Si (`Si`), Hydrogenated Amorphous Si (`a-Si`), Germanium (`Ge`)
  - Nitrides: Stoichiometric LPCVD `Si3N4` (`SiN`), Silicon-Rich `SiNx`, Wurtzite `AlN`,
    Scandium-doped `ScAlN` (`Sc_x Al_{1-x} N`, continuous x in [0, 0.40])
  - Carbides: Uniaxial `4H-SiC` (c-cut), Cubic `3C-SiC`
  - Ferroelectrics: Thin-Film Lithium Niobate `LiNbO3` (`LN_Zcut` & `LN_Xcut`)
  - Dielectrics & Oxides: Fused Silica (`SiO2`), Sapphire/ALD Alumina (`Al2O3`), `Air`
  - Phase-Change Materials: Low-loss `GSST` (`Ge2Sb2Se4Te1`, amorphous & crystalline states)

All wavelengths are in SI metres (m) unless noted; temperatures in Kelvin (K).
Cross-section axis convention:
  x = radial / horizontal in-plane, y = vertical (wafer normal), z = propagation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np

# Fundamental physical constants (SI units)
C0_M_PER_S = 299792458.0          # unit: m/s
H_PLANCK_J_S = 6.62607015e-34     # unit: J*s
HBAR_J_S = H_PLANCK_J_S / (2.0 * math.pi)
EPS0_F_PER_M = 8.8541878128e-12   # unit: F/m
Z0_OHM = 376.730313668            # unit: Ohm
T_REF_K = 300.0                   # unit: K


@dataclass(frozen=True)
class MaterialIndexTensor:
    """Diagonal complex refractive index + thermo/electro-optic & Kerr properties."""
    nx: float
    ny: float
    nz: float
    kx: float = 0.0
    ky: float = 0.0
    kz: float = 0.0
    dn_dt_per_k: float = 0.0       # unit: K^-1 (dominant thermo-optic coefficient)
    r33_pm_per_v: float = 0.0      # unit: pm/V (Pockels r_33 along crystal c/Z axis)
    r13_pm_per_v: float = 0.0      # unit: pm/V (Pockels r_13 perpendicular to c/Z axis)
    n2_m2_per_w: float = 0.0       # unit: m^2/W (Kerr nonlinear index)
    crystal_symmetry: str = "isotropic"
    validity_um: tuple[float, float] = (0.4, 5.0)
    source: str = ""
    is_anisotropic: bool = False

    def complex_eps_diag(self) -> tuple[complex, complex, complex]:
        ex = (self.nx + 1j * self.kx) ** 2
        ey = (self.ny + 1j * self.ky) ** 2
        ez = (self.nz + 1j * self.kz) ** 2
        return ex, ey, ez

    def to_dict(self) -> dict:
        return asdict(self)


def _check_wl(wavelength_m: float) -> float:
    if not (math.isfinite(wavelength_m) and wavelength_m > 0.0):
        raise ValueError("wavelength_m must be finite and strictly positive")
    return wavelength_m * 1e6  # returns wavelength in um


# ==============================================================================
# 1. Claddings & Oxides (Air, SiO2, Al2O3)
# ==============================================================================

def n_air(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Air / vacuum reference index (n = 1.0)."""
    _ = (_check_wl(wavelength_m), temperature_k)
    return MaterialIndexTensor(
        nx=1.0, ny=1.0, nz=1.0,
        dn_dt_per_k=0.0,
        validity_um=(0.1, 50.0),
        source="Vacuum/Air nominal n=1.0",
        is_anisotropic=False,
    )


def n_sio2_sellmeier(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Fused silica (SiO2) Sellmeier + thermo-optic shift + Mid-IR multiphonon k(lambda).

    Source: Malitson, JOSA 55, 1205 (1965); Leviton & Frey (2006) dn/dT = 8.5e-6 K^-1;
    Kitamura, Pilon & Jonasz, Appl. Opt. 46, 8118 (2007) Mid-IR multiphonon onset > 3.6 um.
    """
    u = _check_wl(wavelength_m)
    x = min(u, 6.8) ** 2
    n0 = math.sqrt(
        max(
            1.0
            + 0.6961663 * x / (x - 0.068043 ** 2)
            + 0.4079426 * x / (x - 0.1162414 ** 2)
            + 0.8974794 * x / (x - 9.896161 ** 2),
            1.02,
        )
    )
    # Multiphonon absorption onset above ~3.6 um (critical for Mid-IR cladding selection)
    k_mir = 0.0
    if u > 3.60:
        k_mir = float(min(1.8e-4 * math.exp((u - 3.60) / 0.42), 0.25))
    dn_dt = 8.5e-6
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        kx=k_mir, ky=k_mir, kz=k_mir,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=2.6e-20,
        validity_um=(0.21, 3.65),
        source="Malitson JOSA 1965 + Kitamura Appl. Opt. 2007 (transparent 0.21-3.65 um; absorbs in Mid-IR)",
        is_anisotropic=False,
    )


def n_caf2_fluoride(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Calcium Fluoride (CaF2) UV-to-Mid-IR low-index substrate/cladding (0.15 - 9.5 um).

    Source: Malitson, Appl. Opt. 2, 1103 (1963); Tropf, Opt. Eng. 34 (1995) dn/dT = -1.06e-5 K^-1.
    """
    u = min(max(_check_wl(wavelength_m), 0.15), 11.0)
    x = u * u
    n0 = math.sqrt(
        max(
            1.0
            + 0.5675888 * x / (x - 0.050263605 ** 2)
            + 0.4710914 * x / (x - 0.1003909 ** 2)
            + 3.8484723 * x / (x - 34.649040 ** 2),
            1.05,
        )
    )
    dn_dt = -1.06e-5
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    k_lwir = float(min(1e-4 * math.exp((u - 9.5) / 0.6), 0.1)) if u > 9.5 else 0.0
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        kx=k_lwir, ky=k_lwir, kz=k_lwir,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=1.9e-20,
        validity_um=(0.15, 9.5),
        source="Malitson Appl. Opt. 1963 (CaF2 UV-to-Mid-IR cladding, 0.15-9.5 um)",
        is_anisotropic=False,
    )


def n_mgf2_fluoride(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Magnesium Fluoride (MgF2) UV-to-Mid-IR low-index cladding (0.14 - 7.5 um).

    Source: Dodge, Appl. Opt. 23, 1980 (1984).
    """
    u = min(max(_check_wl(wavelength_m), 0.14), 9.0)
    x = u * u
    n0 = math.sqrt(
        max(
            1.0
            + 0.48755108 * x / (x - 0.04338408 ** 2)
            + 0.39875031 * x / (x - 0.09461442 ** 2)
            + 2.3120353 * x / (x - 23.793604 ** 2),
            1.05,
        )
    )
    dn_dt = 1.9e-6
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    k_lwir = float(min(1e-4 * math.exp((u - 7.5) / 0.5), 0.1)) if u > 7.5 else 0.0
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        kx=k_lwir, ky=k_lwir, kz=k_lwir,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=1.0e-20,
        validity_um=(0.14, 7.5),
        source="Dodge Appl. Opt. 1984 (MgF2 UV-to-Mid-IR cladding, 0.14-7.5 um)",
        is_anisotropic=False,
    )


def n_al2o3_sapphire(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """ALD Alumina / C-plane Sapphire (Al2O3) Sellmeier dispersion.

    Source: Malitson, JOSA 52, 1377 (1962).
    """
    u = _check_wl(wavelength_m)
    x = u * u
    no = math.sqrt(
        max(
            1.0
            + 1.4313493 * x / (x - 0.0726631 ** 2)
            + 0.65054713 * x / (x - 0.1193242 ** 2)
            + 5.3414021 * x / (x - 18.028251 ** 2),
            1.0,
        )
    )
    dn_dt = 1.2e-5
    n = no + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=3.0e-20,
        validity_um=(0.25, 5.5),
        source="Malitson JOSA 1962 (Al2O3 / Sapphire)",
        is_anisotropic=False,
    )


# ==============================================================================
# 2. Group IV Semiconductors (Si, a-Si:H, Ge)
# ==============================================================================

def n_si_crystalline(
    wavelength_m: float,
    temperature_k: float = T_REF_K,
    fixed_1550_index: bool = False,
) -> MaterialIndexTensor:
    """Crystalline Silicon (Si) dispersion, thermo-optic shift & interband absorption below 1.12 um.

    Source: Salzberg & Villa, JOSA 47, 244 (1957); Green, Sol. Energy Mater. 92, 1305 (2008)
    visible/VNIR interband k(lambda); Cocorullo & Della Corte (1999) dn/dT = 1.86e-4 K^-1.
    """
    u = _check_wl(wavelength_m)
    dn_dt = 1.86e-4
    k_ib = 0.0
    if u < 1.12:
        # Realistic indirect-to-direct interband absorption below 1.12 eV bandgap
        k_ib = float(min(1.5e-5 * math.exp((1.12 - u) / 0.068), 0.35))
    if fixed_1550_index:
        n = 3.476 + dn_dt * (temperature_k - T_REF_K)
    else:
        x = max(u, 1.12) ** 2
        n0 = math.sqrt(
            1.0
            + 10.6684293 * x / (x - 0.301516485 ** 2)
            + 0.0030434748 * x / (x - 1.13475115 ** 2)
            + 1.54133408 * x / (x - 1104.0 ** 2)
        )
        if u < 1.12:
            # Normal dispersion rise into VNIR/Visible (Green 2008: n~3.71@780nm, ~4.15@532nm)
            n0 += 0.52 * ((1.12 / max(u, 0.42)) ** 1.35 - 1.0)
        n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        kx=k_ib, ky=k_ib, kz=k_ib,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=4.5e-18,
        crystal_symmetry="cubic_m3m",
        validity_um=(1.15, 8.5),
        source="Salzberg & Villa JOSA 1957 + Green 2008 + Cocorullo 1999 (transparent 1.15-8.5 um)",
        is_anisotropic=False,
    )


def n_asi_hydrogenated(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Hydrogenated Amorphous Silicon (a-Si:H) PECVD waveguide film.

    Source: Della Corte et al., JAP 88, 7115 (2000); Kuyken et al., Opt. Express (2011).
    """
    raw_u = _check_wl(wavelength_m)
    u = max(raw_u, 0.85)
    x = u * u
    n0 = math.sqrt(1.0 + 10.90 * x / (x - 0.365 ** 2))
    k_ib = float(min(2.0e-4 * math.exp((0.85 - raw_u) / 0.06), 0.40)) if raw_u < 0.85 else 0.0
    dn_dt = 2.30e-4
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        kx=k_ib, ky=k_ib, kz=k_ib,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=1.3e-17,
        validity_um=(0.95, 5.5),
        source="PECVD a-Si:H Cauchy/Sellmeier (Della Corte 2000; dn/dT=2.3e-4 K^-1)",
        is_anisotropic=False,
    )


def n_ge_crystalline(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Crystalline Germanium (Ge) Mid-IR waveguide dispersion (transparent 2.0 - 14.0 um).

    Source: Barnes & Piltch, JOSA 69, 178 (1979).
    """
    raw_u = _check_wl(wavelength_m)
    u = max(raw_u, 1.95)
    x = u * u
    n0 = math.sqrt(
        9.28156
        + 6.72880 * x / (x - 0.44105)
        + 0.21307 * x / (x - 3870.1)
    )
    k_ib = float(min(3.0e-4 * math.exp((1.90 - raw_u) / 0.09), 0.45)) if raw_u < 1.90 else 0.0
    dn_dt = 4.02e-4
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        kx=k_ib, ky=k_ib, kz=k_ib,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=2.5e-17,
        crystal_symmetry="cubic_m3m",
        validity_um=(2.0, 14.0),
        source="Barnes & Piltch JOSA 1979 (Crystalline Ge Mid-IR, transparent 2.0-14.0 um)",
        is_anisotropic=False,
    )


# ==============================================================================
# 3. Silicon Nitrides & III-Nitrides (Si3N4, SiRich-SiNx, AlN, ScAlN)
# ==============================================================================

def n_sin_lpcvd(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Stoichiometric LPCVD Silicon Nitride (Si3N4 / SiN).

    Source: Luke et al., Opt. Lett. 40, 4823 (2015); Arbabi & Goddard, Opt. Lett. 38,
    3878 (2013) dn/dT = 2.45e-5 K^-1; Ikeda et al. (2008) n2 = 2.4e-19 m^2/W.
    """
    u = _check_wl(wavelength_m)
    x = u * u
    n0 = math.sqrt(
        max(
            1.0
            + 3.0249 * x / (x - 0.1353406 ** 2)
            + 40314.0 * x / (x - 1239.842 ** 2),
            1.0,
        )
    )
    dn_dt = 2.45e-5
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=2.4e-19,
        validity_um=(0.31, 5.5),
        source="Luke et al. Opt. Lett. 2015 (LPCVD Si3N4) + Arbabi 2013 (dn/dT=2.45e-5 K^-1)",
        is_anisotropic=False,
    )


def n_sirich_sinx(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Low-stress Silicon-Rich Silicon Nitride (SRN / SiNx, n ~ 2.10 @ 1550 nm).

    Source: Lacava et al., Photonics Res. 5, 151 (2017); Krückel et al., Opt. Express (2015).
    """
    u = max(_check_wl(wavelength_m), 0.55)
    x = u * u
    n0 = math.sqrt(1.0 + 3.320 * x / (x - 0.185 ** 2))
    dn_dt = 3.20e-5
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=1.4e-18,
        validity_um=(0.60, 5.0),
        source="Silicon-Rich SiNx (Lacava Photonics Res. 2017; n~2.10@1550nm)",
        is_anisotropic=False,
    )


def n_aln_film(
    wavelength_m: float,
    temperature_k: float = T_REF_K,
    nominal_n_override: float | None = None,
) -> MaterialIndexTensor:
    """Wurtzite c-axis vertical Aluminum Nitride (AlN) uniaxial Sellmeier `[no, ne, no]`.

    Source: Pastrnak & Roskovcova, Phys. Status Solidi 14, K5 (1966);
    Watanabe et al., JAP 104, 103531 (2008) dn/dT = 2.32e-5 K^-1;
    Yang et al., Nat. Commun. 15, 6168 (2024) measured 1550 nm Pockels r33 = +1.50 pm/V,
    r13 = -0.92 pm/V (opposite signs in pure wurtzite AlN; Xiong et al., Nano Lett. 2012 ~1.0 pm/V).
    """
    u = max(_check_wl(wavelength_m), 0.25)
    dn_dt = 2.32e-5
    if nominal_n_override is not None:
        n = float(nominal_n_override) + dn_dt * (temperature_k - T_REF_K)
        return MaterialIndexTensor(
            nx=n, ny=n, nz=n,
            dn_dt_per_k=dn_dt,
            r33_pm_per_v=1.50,
            r13_pm_per_v=-0.92,
            n2_m2_per_w=2.3e-19,
            crystal_symmetry="wurtzite_6mm_scalar_override",
            validity_um=(0.30, 5.2),
            source=f"AlN scalar calibration override n={nominal_n_override:.4f}",
            is_anisotropic=False,
        )
    x = u * u
    no = math.sqrt(1.0 + 3.1399 * x / (x - 0.1362 ** 2)) + dn_dt * (temperature_k - T_REF_K)
    ne = math.sqrt(1.0 + 3.2792 * x / (x - 0.1416 ** 2)) + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=no, ny=ne, nz=no,
        dn_dt_per_k=dn_dt,
        r33_pm_per_v=1.50,
        r13_pm_per_v=-0.92,
        n2_m2_per_w=2.3e-19,
        crystal_symmetry="wurtzite_6mm_c_vertical",
        validity_um=(0.26, 5.2),
        source="Pastrnak & Roskovcova 1966 + Yang et al. Nat. Commun. 2024 (c-axis AlN, r33=+1.50, r13=-0.92 pm/V)",
        is_anisotropic=True,
    )


def n_scaln_film(
    wavelength_m: float,
    temperature_k: float = T_REF_K,
    sc_fraction: float = 0.20,
) -> MaterialIndexTensor:
    """Wurtzite c-axis Scandium-Doped Aluminum Nitride (Sc_x Al_{1-x} N, 0 <= x <= 0.43).

    Experimental calibration (actual sputtered film index & r_ij depend on Sc fraction x,
    residual stress, polarity purity, and XRD (0002) rocking-curve FWHM):
      - Optical index increase with Sc content x: Baeumler et al., J. Appl. Phys. 126,
        115105 (2019); Wingqvist et al., J. Appl. Phys. 107, 084909 (2010).
      - Measured 1550 nm Pockels tensor vs Sc content x: Yang et al., Nature Communications
        15, 6168 (2024) & Wang & Van de Walle, Appl. Phys. Lett. 126 (2025): no EO gain over
        pure AlN for x <= 0.20 (r33 ~ 1.22-1.34 pm/V), r13(x) crosses zero near x ~ 0.16
        (-0.92 pm/V at x=0 -> -0.36 pm/V at x=0.10 -> +0.32 pm/V at x=0.20 -> +1.29 pm/V at
        x=0.319), and r33 reaches 2.07 +/- 0.45 pm/V at x=0.319 before steeper lattice-softening
        enhancement at x = 0.36-0.40.
    """
    if not (0.0 <= sc_fraction <= 0.43):
        raise ValueError("Wurtzite Sc_x Al_{1-x} N sc_fraction x must be in [0.0, 0.43]")
    base = n_aln_film(wavelength_m, temperature_k=T_REF_K)
    u = max(_check_wl(wavelength_m), 0.35)
    # Bandgap redshift with Sc content increases both baseline index and UV pole wavelength
    disp_factor = 1.0 + 0.045 / (u * u)
    delta_no = (0.48 * sc_fraction + 0.22 * (sc_fraction ** 2)) * disp_factor
    # Birefringence (n_e - n_o) softens gradually as c/a ratio decreases with Sc doping
    biref_0 = base.ny - base.nx
    biref_x = biref_0 * max(1.0 - 1.45 * sc_fraction, 0.15)
    dn_dt = 2.32e-5 * (1.0 + 0.65 * sc_fraction)
    dt = temperature_k - T_REF_K

    no = base.nx + delta_no + dn_dt * dt
    ne = no + biref_x
    # Calibrated to Yang et al. Nat. Commun. 15, 6168 (2024) measured microring Pockels data
    x2 = sc_fraction * sc_fraction
    x3 = x2 * sc_fraction
    r33_est = 1.50 - 4.50 * sc_fraction + 16.50 * x2 + 10.00 * x3
    r13_est = -0.92 + 5.00 * sc_fraction + 6.00 * x2
    return MaterialIndexTensor(
        nx=no, ny=ne, nz=no,
        dn_dt_per_k=dn_dt,
        r33_pm_per_v=r33_est,
        r13_pm_per_v=r13_est,
        n2_m2_per_w=3.5e-19,
        crystal_symmetry=f"wurtzite_6mm_c_vertical_Sc{int(round(sc_fraction*100)):02d}",
        validity_um=(0.38, 5.0),
        source=f"Baeumler JAP 2019 + Yang Nat. Commun. 2024 measured Sc_{sc_fraction:.2f}Al_{1-sc_fraction:.2f}N",
        is_anisotropic=True,
    )


# ==============================================================================
# 4. Silicon Carbides (4H-SiC Uniaxial & 3C-SiC Cubic)
# ==============================================================================

def n_4h_sic_anisotropic(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """4H-SiC uniaxial Sellmeier dispersion in c-axis-vertical local coordinates `[no, ne, no]`.

    Source: S. Wang et al., Opt. Lett. 38, 23 (2013) [Sellmeier];
    J. Li et al., Phys. Rev. Appl. 2023 (n2 = 8.9e-19 m^2/W);
    R. Wang, J. Li, L. Cai, & Q. Li, Opt. Lett. 48, 1482-1485 (2023)
    (|r13| = 0.3-0.7 pm/V ~ 0.50 pm/V, |r33| = 0-0.03 pm/V in high-Q 4H-SiCOI rings).
    """
    u = max(_check_wl(wavelength_m), 0.38)
    x = u * u
    no = math.sqrt(
        1.0
        + 0.20075 * x / (x + 12.07224)
        + 5.54861 * x / (x - 0.02641)
        + 35.65066 * x / (x - 1268.24708)
    )
    ne = math.sqrt(6.79485 + 0.15558 / (x - 0.03535) - 0.02296 * x)
    dn_dt = 3.9e-5
    if abs(temperature_k - T_REF_K) > 1e-9:
        no += dn_dt * (temperature_k - T_REF_K)
        ne += dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=no, ny=ne, nz=no,
        dn_dt_per_k=dn_dt,
        r33_pm_per_v=0.03,
        r13_pm_per_v=-0.50,
        n2_m2_per_w=8.9e-19,
        crystal_symmetry="hexagonal_6mm_c_vertical",
        validity_um=(0.40, 5.2),
        source="S. Wang Opt. Lett. 38, 23 (2013) Sellmeier; R. Wang & Q. Li Opt. Lett. 48, 1482 (2023) Pockels",
        is_anisotropic=True,
    )


def n_3c_sic(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Cubic 3C-SiC isotropic Sellmeier dispersion.

    Source: Shaffer, Appl. Opt. 10, 1034 (1971); Wang et al., Opt. Lett. 38 (2013).
    """
    u = max(_check_wl(wavelength_m), 0.46)
    x = u * u
    n0 = math.sqrt(1.0 + 5.5515 * x / (x - 0.1625 ** 2))
    dn_dt = 3.8e-5
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        r33_pm_per_v=2.7,  # r_41 in zincblende 3C-SiC
        n2_m2_per_w=6.0e-19,
        crystal_symmetry="zincblende_43m",
        validity_um=(0.48, 5.0),
        source="Shaffer 1971 / Wang Opt. Lett. 2013 (Cubic 3C-SiC)",
        is_anisotropic=False,
    )


# ==============================================================================
# 5. Thin-Film Lithium Niobate (LiNbO3 / TFLN: Z-cut & X-cut)
# ==============================================================================

def _linbo3_no_ne(wavelength_m: float, temperature_k: float = T_REF_K) -> tuple[float, float]:
    """Congruent Lithium Niobate (LiNbO3) ordinary & extraordinary indices.

    Source: Zelmon, Small & Jundt, JOSA B 14, 3319 (1997) (valid 0.4 - 5.0 um).
    """
    u = max(_check_wl(wavelength_m), 0.38)
    x = u * u
    no = math.sqrt(
        max(
            1.0
            + 2.6734 * x / (x - 0.01764)
            + 1.2290 * x / (x - 0.05914)
            + 12.614 * x / (x - 474.60),
            1.0,
        )
    )
    ne = math.sqrt(
        max(
            1.0
            + 2.9804 * x / (x - 0.02047)
            + 0.5981 * x / (x - 0.0666)
            + 8.9543 * x / (x - 416.08),
            1.0,
        )
    )
    dt = temperature_k - T_REF_K
    no += 0.40e-5 * dt
    ne += 3.70e-5 * dt
    return no, ne


def n_linbo3_zcut(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Z-cut Thin-Film Lithium Niobate (TFLN): optic Z-axis vertical `[no, ne, no]`.

    In Z-cut TFLN rings, cylindrical symmetry is preserved around 360 deg:
      - TM modes see n_e and strong vertical Pockels r_33 = 30.9 pm/V
      - TE modes see n_o and vertical Pockels r_13 = 9.6 pm/V.
    """
    no, ne = _linbo3_no_ne(wavelength_m, temperature_k)
    return MaterialIndexTensor(
        nx=no, ny=ne, nz=no,
        dn_dt_per_k=3.70e-5,
        r33_pm_per_v=30.9,
        r13_pm_per_v=9.6,
        n2_m2_per_w=1.8e-19,
        crystal_symmetry="trigonal_3m_Zcut_vertical",
        validity_um=(0.40, 5.0),
        source="Zelmon et al. JOSA B 1997 (Congruent Z-cut TFLN [no, ne, no])",
        is_anisotropic=True,
    )


def n_linbo3_xcut(
    wavelength_m: float,
    temperature_k: float = T_REF_K,
    inplane_propagation_angle_deg: float = 0.0,
) -> MaterialIndexTensor:
    """X-cut Thin-Film Lithium Niobate (TFLN): optic Z-axis in wafer plane.

    Convention (`inplane_propagation_angle_deg = 0` = Y-propagating straight section,
    standard for X-cut TFLN push-pull modulators & racetrack straight arms):
      - Horizontal x-axis is crystal Z-axis (`n_x = n_e`, enabling TE mode `r_33 = 30.9 pm/V`!)
      - Vertical y-axis is crystal X-axis (`n_y = n_o`)
      - Propagation z-axis is crystal Y-axis (`n_z = n_o`).
    For arbitrary in-plane propagation angle `phi` (e.g. along a bend), the in-plane
    transverse extraordinary-ordinary index ellipse gives:
      1 / n_horiz(phi)^2 = cos^2(phi) / n_e^2 + sin^2(phi) / n_o^2.
    """
    no, ne = _linbo3_no_ne(wavelength_m, temperature_k)
    phi = math.radians(inplane_propagation_angle_deg)
    c2 = math.cos(phi) ** 2
    s2 = math.sin(phi) ** 2
    nx_eff = 1.0 / math.sqrt(c2 / (ne * ne) + s2 / (no * no))
    nz_eff = 1.0 / math.sqrt(s2 / (ne * ne) + c2 / (no * no))
    return MaterialIndexTensor(
        nx=nx_eff, ny=no, nz=nz_eff,
        dn_dt_per_k=3.70e-5 * c2 + 0.40e-5 * s2,
        r33_pm_per_v=30.9,
        r13_pm_per_v=9.6,
        n2_m2_per_w=1.8e-19,
        crystal_symmetry="trigonal_3m_Xcut_inplane",
        validity_um=(0.40, 5.0),
        source="Zelmon et al. JOSA B 1997 (Congruent X-cut TFLN, TE sees r33 along straight Y-arm)",
        is_anisotropic=True,
    )


def _litao3_no_ne(wavelength_m: float, temperature_k: float = T_REF_K) -> tuple[float, float]:
    """Optical-grade Lithium Tantalate (LiTaO3 / TFLT) Sellmeier dispersion (0.28 - 5.5 um).

    Sources: Bruner et al., J. Appl. Phys. 94, 4950 (2003); Casson et al., Appl. Opt. 43, 6412 (2004);
    Wang et al., Nature 629, 788-794 (2024) (Ultra-low birefringence n_e - n_o = +0.0038 at 1550 nm).
    """
    u = min(max(_check_wl(wavelength_m), 0.28), 5.6)
    x = u * u
    dt = temperature_k - T_REF_K
    no = math.sqrt(max(1.0 + 3.4839 * x / (x - 0.1338 ** 2) + 0.0884 * x / (x - 0.2950 ** 2) - 0.0235 * x, 1.5))
    ne = math.sqrt(max(1.0 + 3.5020 * x / (x - 0.1342 ** 2) + 0.0865 * x / (x - 0.2980 ** 2) - 0.0242 * x, 1.5))
    return no + 0.70e-5 * dt, ne + 4.20e-5 * dt


def n_litao3_zcut(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Z-cut Thin-Film Lithium Tantalate (TFLT): optic Z-axis vertical (y)."""
    no, ne = _litao3_no_ne(wavelength_m, temperature_k)
    return MaterialIndexTensor(
        nx=no, ny=ne, nz=no,
        dn_dt_per_k=4.20e-5,
        r33_pm_per_v=30.5,
        r13_pm_per_v=8.4,
        n2_m2_per_w=1.5e-19,
        crystal_symmetry="trigonal_3m_Zcut_vertical",
        validity_um=(0.28, 5.5),
        source="Bruner et al. JAP 2003 + Wang et al. Nature 2024 (Z-cut TFLT, r33=30.5 pm/V)",
        is_anisotropic=True,
    )


def n_litao3_xcut(
    wavelength_m: float,
    temperature_k: float = T_REF_K,
    inplane_propagation_angle_deg: float = 0.0,
) -> MaterialIndexTensor:
    """X-cut Thin-Film Lithium Tantalate (TFLT): optic Z-axis horizontal in wafer plane.

    Ultra-low birefringence (`Delta_n = +0.0038`, 19x lower than LiNbO3!) eliminates leaky
    TE0-TM0 slab hybridization across tight bends while preserving `r_33 = 30.5 pm/V`.
    """
    no, ne = _litao3_no_ne(wavelength_m, temperature_k)
    phi = math.radians(inplane_propagation_angle_deg)
    c2 = math.cos(phi) ** 2
    s2 = math.sin(phi) ** 2
    nx_eff = 1.0 / math.sqrt(c2 / (ne * ne) + s2 / (no * no))
    nz_eff = 1.0 / math.sqrt(s2 / (ne * ne) + c2 / (no * no))
    return MaterialIndexTensor(
        nx=nx_eff, ny=no, nz=nz_eff,
        dn_dt_per_k=4.20e-5 * c2 + 0.70e-5 * s2,
        r33_pm_per_v=30.5,
        r13_pm_per_v=8.4,
        n2_m2_per_w=1.5e-19,
        crystal_symmetry="trigonal_3m_Xcut_inplane",
        validity_um=(0.28, 5.5),
        source="Wang et al. Nature 629, 788 (2024) + Bruner JAP 2003 (X-cut TFLT, ultra-low birefringence)",
        is_anisotropic=True,
    )


def _bto_no_ne(wavelength_m: float, temperature_k: float = T_REF_K) -> tuple[float, float]:
    """Tetragonal Ferroelectric Barium Titanate (BaTiO3 / BTO, 4mm) Sellmeier (0.42 - 6.5 um).

    Sources: Wemple, DiDomenico & Camlibel, J. Phys. Chem. Solids 29, 1797 (1968);
    Abel et al., Nature Materials 18, 42-47 (2019); Eltes et al., Nature Materials 19, 1164 (2020).
    """
    u = min(max(_check_wl(wavelength_m), 0.42), 6.6)
    x = u * u
    dt = temperature_k - T_REF_K
    no = math.sqrt(max(1.0 + 4.239 * x / (x - 0.2228 ** 2) - 0.0155 * x, 1.8))
    ne = math.sqrt(max(1.0 + 4.095 * x / (x - 0.2155 ** 2) - 0.0150 * x, 1.8))
    return no - 4.20e-5 * dt, ne - 4.20e-5 * dt


def n_bto_acut(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Epitaxial a-oriented Tetragonal BaTiO3 (BTO-on-Si via SrTiO3 buffer, 4mm).

    Along the optimal Si [110] propagation azimuth (~45-51 deg to in-plane BTO c-axis),
    coplanar GSG TE0 waveguides unlock the giant shear Pockels tensor `r_42`:
      - Clamped high-speed RF (>5 GHz) `r_42^S ~ 360 pm/V` -> `r_eff ~ 285 pm/V`
      - Low-frequency unclamped DC `r_42^T ~ 923 pm/V` -> `r_eff ~ 730 pm/V` (Abel et al. 2019).
    Here we report the conservative clamped high-speed RF `r_eff = 285 pm/V` for EOM design.
    """
    no, ne = _bto_no_ne(wavelength_m, temperature_k)
    n_inplane = 0.5 * (no + ne)  # nanoscale 90-deg a1/a2 twin-domain / 45-deg [110] effective index
    return MaterialIndexTensor(
        nx=n_inplane, ny=no, nz=n_inplane,
        dn_dt_per_k=-4.20e-5,
        r33_pm_per_v=285.0,
        r13_pm_per_v=45.0,
        n2_m2_per_w=8.0e-19,
        crystal_symmetry="tetragonal_4mm_BTO_inplane",
        validity_um=(0.42, 6.5),
        source="Abel et al. Nat. Mater. 2019 + Eltes et al. Nat. Mater. 2020 (Epitaxial a-BTO on Si, clamped RF r42)",
        is_anisotropic=True,
    )


def n_bto_ccut(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Out-of-plane c-oriented Tetragonal BaTiO3 (4mm, vertical c-axis)."""
    no, ne = _bto_no_ne(wavelength_m, temperature_k)
    return MaterialIndexTensor(
        nx=no, ny=ne, nz=no,
        dn_dt_per_k=-4.20e-5,
        r33_pm_per_v=105.0,
        r13_pm_per_v=19.5,
        n2_m2_per_w=8.0e-19,
        crystal_symmetry="tetragonal_4mm_Zcut_vertical",
        validity_um=(0.42, 6.5),
        source="Abel et al. Nat. Mater. 2019 + Wemple 1968 (c-oriented tetragonal BTO)",
        is_anisotropic=True,
    )


def n_pzt_poled(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Lanthanide-seeded Thin-Film Lead Zirconate Titanate (PbZr0.52Ti0.48O3 / PZT, 0.45 - 6.5 um).

    Source: Alexander et al., Nature Communications 9, 3444 (2018) (Ghent / Van Thourhout
    CSD PZT on SiO2/SiN: post-fabrication electrical poling across coplanar GSG / ring electrodes
    aligns remnant polarization P_s || E_RF, yielding clamped high-speed EO r_eff ~ 67 pm/V up to >33 GHz).
    """
    u = min(max(_check_wl(wavelength_m), 0.45), 6.6)
    x = u * u
    dt = temperature_k - T_REF_K
    no = math.sqrt(max(1.0 + 4.695 * x / (x - 0.238 ** 2) - 0.012 * x, 1.8)) + 3.50e-5 * dt
    ne = math.sqrt(max(1.0 + 4.598 * x / (x - 0.235 ** 2) - 0.012 * x, 1.8)) + 3.50e-5 * dt
    return MaterialIndexTensor(
        nx=ne, ny=no, nz=no,
        dn_dt_per_k=3.50e-5,
        r33_pm_per_v=67.0,
        r13_pm_per_v=20.0,
        n2_m2_per_w=6.0e-19,
        crystal_symmetry="poled_4mm_PZT_inplane",
        validity_um=(0.45, 6.5),
        source="Alexander et al. Nat. Commun. 9, 3444 (2018) (Clamped high-speed poled PZT film, r_eff=67 pm/V)",
        is_anisotropic=True,
    )


# ==============================================================================
# 6. Active Phase-Change Materials (Low-Loss Chalcogenide GSST)
# ==============================================================================

def n_gsst_pcm(wavelength_m: float, state: str = "amorphous") -> MaterialIndexTensor:
    """Published Ge2Sb2Se4Te1 (GSST) low-loss optical phase-change material model.

    Source: Zhang et al., Nature Communications 10, 4279 (2019).
    """
    u = _check_wl(wavelength_m)
    n_amorphous = 3.32 - 0.025 * np.clip(u - 1.55, 0.0, 4.0)
    if state == "amorphous":
        return MaterialIndexTensor(
            nx=float(n_amorphous), ny=float(n_amorphous), nz=float(n_amorphous),
            kx=0.0, ky=0.0, kz=0.0,
            dn_dt_per_k=5.0e-5,
            validity_um=(1.5, 10.0),
            source="Zhang et al. Nat. Commun. 2019 (Amorphous GSST low-loss PCM)",
            is_anisotropic=False,
        )
    if state == "crystalline":
        # Crystalline GSST: n_c ~ 5.08 (k_c ~ 0.35 at 1.55 um, dropping to ~0.015 in Mid-IR)
        nc = 5.08 - 0.04 * np.clip(u - 1.55, 0.0, 4.0)
        kc = 0.35 * math.exp(-max(u - 1.55, 0.0) / 0.85) + 0.015
        return MaterialIndexTensor(
            nx=float(nc), ny=float(nc), nz=float(nc),
            kx=float(kc), ky=float(kc), kz=float(kc),
            dn_dt_per_k=8.0e-5,
            validity_um=(1.5, 10.0),
            source="Zhang et al. Nat. Commun. 2019 (Crystalline GSST PCM)",
            is_anisotropic=False,
        )
    raise ValueError(f"Unknown PCM state '{state}'; expected 'amorphous' or 'crystalline'")


def n_gst_pcm(wavelength_m: float, state: str = "amorphous") -> MaterialIndexTensor:
    """Published Ge2Sb2Te5 (GST-225) chalcogenide phase-change material model (1.10 - 2.50 um).

    Source: Shportko et al., Nat. Mater. 7, 653 (2008); Rios et al., Nat. Photon. 9, 725 (2015);
    Stegmaier et al., Adv. Opt. Mater. 5, 1600787 (2017).
    """
    u = max(_check_wl(wavelength_m), 1.05)
    if state == "amorphous":
        na = 3.880 + 0.145 / (u * u)
        ka = 0.024 * math.exp(min(max((1.55 - u) / 0.42, -2.5), 2.5))
        return MaterialIndexTensor(
            nx=float(na), ny=float(na), nz=float(na),
            kx=float(ka), ky=float(ka), kz=float(ka),
            dn_dt_per_k=8.0e-5,
            validity_um=(1.10, 2.50),
            source="Rios et al. Nat. Photon. 9, 725 (2015) / Shportko Nat. Mater. 2008 (Amorphous GST-225)",
            is_anisotropic=False,
        )
    if state == "crystalline":
        nc = 5.860 + 0.600 / (u * u)
        kc = 1.090 * math.exp(min(max((1.55 - u) / 0.65, -2.0), 2.0))
        return MaterialIndexTensor(
            nx=float(nc), ny=float(nc), nz=float(nc),
            kx=float(kc), ky=float(kc), kz=float(kc),
            dn_dt_per_k=1.2e-4,
            validity_um=(1.10, 2.50),
            source="Rios et al. Nat. Photon. 9, 725 (2015) / Shportko Nat. Mater. 2008 (Crystalline GST-225)",
            is_anisotropic=False,
        )
    raise ValueError(f"Unknown PCM state '{state}'; expected 'amorphous' or 'crystalline'")


def n_sb2se3_pcm(wavelength_m: float, state: str = "amorphous") -> MaterialIndexTensor:
    """Published Sb2Se3 ultra-low-loss phase-change material model (0.98 - 2.20 um).

    Source: Delaney et al., Adv. Funct. Mater. 30, 2002447 (2020) Table 1 (ellipsometry Cauchy fit).
    """
    u = max(_check_wl(wavelength_m), 0.90)
    if state == "amorphous":
        na = 3.226 + 0.142 / (u * u)
        ka = 0.0 if u >= 1.02 else float(0.015 * math.exp((1.02 - u) / 0.06))
        return MaterialIndexTensor(
            nx=float(na), ny=float(na), nz=float(na),
            kx=float(ka), ky=float(ka), kz=float(ka),
            dn_dt_per_k=4.5e-5,
            validity_um=(0.98, 2.20),
            source="Delaney et al. Adv. Funct. Mater. 30, 2002447 (2020) (Amorphous Sb2Se3 low-loss PCM)",
            is_anisotropic=False,
        )
    if state == "crystalline":
        nc = 3.922 + 0.308 / (u * u)
        kc = 1.2e-4 if u >= 1.20 else float(0.018 * math.exp((1.20 - u) / 0.12))
        return MaterialIndexTensor(
            nx=float(nc), ny=float(nc), nz=float(nc),
            kx=float(kc), ky=float(kc), kz=float(kc),
            dn_dt_per_k=6.5e-5,
            validity_um=(1.05, 2.20),
            source="Delaney et al. Adv. Funct. Mater. 30, 2002447 (2020) (Crystalline Sb2Se3 low-loss PCM)",
            is_anisotropic=False,
        )
    raise ValueError(f"Unknown PCM state '{state}'; expected 'amorphous' or 'crystalline'")


def n_sb2s3_pcm(wavelength_m: float, state: str = "amorphous") -> MaterialIndexTensor:
    """Published Sb2S3 wide-bandgap visible-to-NIR low-loss phase-change material model (0.60 - 2.20 um).

    Source: Dong et al., Adv. Funct. Mater. 29, 1806181 (2019); Delaney et al., Adv. Funct. Mater. 30, 2002447 (2020).
    """
    u = max(_check_wl(wavelength_m), 0.58)
    u2 = u * u
    if state == "amorphous":
        na = 2.652 + 0.144 / u2 + 0.026 / (u2 * u2)
        ka = 0.0 if u >= 0.61 else float(0.020 * math.exp((0.61 - u) / 0.04))
        return MaterialIndexTensor(
            nx=float(na), ny=float(na), nz=float(na),
            kx=float(ka), ky=float(ka), kz=float(ka),
            dn_dt_per_k=3.8e-5,
            validity_um=(0.60, 2.20),
            source="Dong Adv. Funct. Mater. 2019 / Delaney Adv. Funct. Mater. 2020 (Amorphous Sb2S3 PCM)",
            is_anisotropic=False,
        )
    if state == "crystalline":
        nc = 3.218 + 0.216 / u2 + 0.038 / (u2 * u2)
        kc = 0.0 if u >= 0.74 else float(0.015 * math.exp((0.74 - u) / 0.06))
        return MaterialIndexTensor(
            nx=float(nc), ny=float(nc), nz=float(nc),
            kx=float(kc), ky=float(kc), kz=float(kc),
            dn_dt_per_k=5.2e-5,
            validity_um=(0.72, 2.20),
            source="Dong Adv. Funct. Mater. 2019 / Delaney Adv. Funct. Mater. 2020 (Crystalline Sb2S3 PCM)",
            is_anisotropic=False,
        )
    raise ValueError(f"Unknown PCM state '{state}'; expected 'amorphous' or 'crystalline'")


def n_hydex_glass(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """High-index doped silica glass (Hydex, n ~ 1.70 at 1550 nm) for low-loss Kerr microcombs.

    Source: Moss et al., Nat. Photon. 7, 597 (2013); Razzari et al., Nat. Photon. 4, 41 (2010).
    """
    u = max(_check_wl(wavelength_m), 0.35)
    x = u * u
    n0 = math.sqrt(max(1.0 + 1.855 * x / (x - 0.115 ** 2) + 0.85 * x / (x - 9.896 ** 2), 1.45))
    dn_dt = 1.15e-5
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=1.15e-19,
        validity_um=(0.40, 2.60),
        source="Moss et al. Nat. Photon. 7, 597 (2013) (High-index doped silica Hydex glass)",
        is_anisotropic=False,
    )


def n_algaas_epi(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Al0.20Ga0.80As-on-insulator nonlinear III-V waveguide core (0.90 - 4.00 um).

    Source: Gehrsitz et al., J. Appl. Phys. 87, 7825 (2000); Pu et al., Optica 3, 823 (2016); Chang et al., Nat. Commun. 11, 1331 (2020).
    """
    u = max(_check_wl(wavelength_m), 0.88)
    x = u * u
    n0 = math.sqrt(max(7.10 + 3.42 * x / (x - 0.395 ** 2), 2.5))
    dn_dt = 2.30e-4
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        r33_pm_per_v=1.43,
        n2_m2_per_w=2.6e-17,
        crystal_symmetry="zincblende_43m",
        validity_um=(0.90, 4.00),
        source="Gehrsitz JAP 2000 / Pu Optica 2016 / Chang Nat. Commun. 2020 (Al0.2Ga0.8As-OI)",
        is_anisotropic=False,
    )


def n_sio2_plc(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """GeO2-doped silica planar lightwave circuit core (1.5% Delta, n ~ 1.466 at 1550 nm).

    Source: Carolan et al., Science 349, 711 (2015); Okamoto, Fundamentals of Optical Waveguides (2021).
    """
    base = n_sio2_sellmeier(wavelength_m, temperature_k)
    n_plc = base.nx * 1.0152
    return MaterialIndexTensor(
        nx=n_plc, ny=n_plc, nz=n_plc,
        dn_dt_per_k=1.05e-5,
        n2_m2_per_w=2.8e-20,
        validity_um=(0.35, 3.65),
        source="Carolan Science 2015 / Okamoto 2021 (1.5%-Delta Ge-doped SiO2 PLC core)",
        is_anisotropic=False,
    )


# Published GSST phase-change material model above


# ==============================================================================
# 6B. Visible / VNIR / Broadband Materials (TiO2, Ta2O5, GaN, Diamond)
# ==============================================================================

def n_tio2_ald(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """ALD / Sputtered Anatase Titanium Dioxide (TiO2) for Visible/VNIR/NIR (0.40 - 4.5 um).

    Source: Siefke et al., Adv. Opt. Mater. 4, 1780 (2016); Raghunathan et al.
    dn/dT = -1.0e-4 K^-1 (negative thermo-optic coefficient, useful for athermal PICs!).
    """
    raw_u = _check_wl(wavelength_m)
    u = max(raw_u, 0.39)
    x = u * u
    n0 = math.sqrt(max(1.0 + 4.022 * x / (x - 0.244 ** 2), 1.5))
    dn_dt = -1.0e-4
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    k_uv = float(min(1e-3 * math.exp((0.39 - raw_u) / 0.02), 0.3)) if raw_u < 0.39 else 0.0
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        kx=k_uv, ky=k_uv, kz=k_uv,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=9.4e-19,
        validity_um=(0.40, 4.5),
        source="Siefke Adv. Opt. Mater. 2016 (ALD TiO2 Visible/NIR, dn/dT=-1.0e-4 K^-1)",
        is_anisotropic=False,
    )


def n_ta2o5_tantala(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Sputtered Tantalum Pentoxide (Ta2O5 / Tantala) ultra-low-loss Visible/NIR waveguide.

    Source: Bright et al., J. Appl. Phys. 114, 083515 (2013); Jung et al., Optica (2021).
    """
    raw_u = _check_wl(wavelength_m)
    u = max(raw_u, 0.35)
    x = u * u
    n0 = math.sqrt(max(1.0 + 3.276 * x / (x - 0.168 ** 2), 1.4))
    dn_dt = 0.88e-5
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=7.2e-19,
        validity_um=(0.38, 4.5),
        source="Bright JAP 2013 / Jung Optica 2021 (Ta2O5 Visible-to-MIR low-loss waveguide)",
        is_anisotropic=False,
    )


def n_gan_wurtzite(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Wurtzite c-axis Gallium Nitride (GaN) uniaxial `[no, ne, no]` (Visible 0.39 um to MIR 6.5 um).

    Source: Barker & Ilegems, Phys. Rev. B 7, 743 (1973); Watanabe et al., JAP (2008).
    """
    raw_u = _check_wl(wavelength_m)
    u = max(raw_u, 0.38)
    x = u * u
    no = math.sqrt(max(3.60 + 1.75 * x / (x - 0.256 ** 2), 1.5))
    ne = math.sqrt(max(3.85 + 1.80 * x / (x - 0.262 ** 2), 1.5))
    dn_dt = 6.8e-5
    dt = temperature_k - T_REF_K
    no += dn_dt * dt
    ne += dn_dt * dt
    return MaterialIndexTensor(
        nx=no, ny=ne, nz=no,
        dn_dt_per_k=dn_dt,
        r33_pm_per_v=1.9,
        r13_pm_per_v=0.6,
        n2_m2_per_w=1.4e-18,
        crystal_symmetry="wurtzite_6mm_c_vertical",
        validity_um=(0.39, 6.5),
        source="Barker & Ilegems PRB 1973 (Wurtzite GaN Visible-to-Mid-IR [no, ne, no])",
        is_anisotropic=True,
    )


def n_diamond_cvd(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Single-Crystal CVD Diamond (UV 0.24 um -> Visible NV/SiV -> Mid-IR/LWIR 20 um).

    Source: Peter, Z. Phys. 15, 358 (1923); Hausmann et al., Nat. Nanotechnol. 9 (2014).
    """
    u = max(_check_wl(wavelength_m), 0.24)
    x = u * u
    n0 = math.sqrt(
        max(
            1.0
            + 0.3306 * x / (x - 0.1750 ** 2)
            + 4.3356 * x / (x - 0.1060 ** 2),
            1.5,
        )
    )
    dn_dt = 1.0e-5
    n = n0 + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=1.3e-19,
        crystal_symmetry="cubic_Fd3m",
        validity_um=(0.24, 20.0),
        source="Peter 1923 / Hausmann Nat. Nanotechnol. 2014 (CVD Diamond UV-Visible-MIR)",
        is_anisotropic=False,
    )


# ==============================================================================
# 7. Unified 24-Material Catalog & Multi-Waveband Presets (Visible -> 1310 -> 1550 -> MIR)
# ==============================================================================

WAVEBAND_PRESETS: dict[str, dict] = {
    "visible_532": {
        "label": "Visible Green (532 nm)",
        "wavelength_nm": 532.0,
        "recommended_cores": ["Si3N4", "TiO2", "Ta2O5", "AlN", "GaN", "Diamond", "LN_Xcut"],
        "recommended_clads": ["SiO2", "Al2O3", "CaF2", "Air"],
    },
    "vnir_780": {
        "label": "VNIR / Quantum Rb & Color Centers (780 nm)",
        "wavelength_nm": 780.0,
        "recommended_cores": ["Si3N4", "4H-SiC", "AlN", "TiO2", "Ta2O5", "LN_Xcut", "Diamond"],
        "recommended_clads": ["SiO2", "Al2O3", "Air"],
    },
    "oband_1310": {
        "label": "Datacom O-Band (1310 nm)",
        "wavelength_nm": 1310.0,
        "recommended_cores": ["Si", "Si3N4", "LN_Xcut", "LT_Xcut", "BTO", "PZT", "4H-SiC", "AlN", "ScAlN_20", "a-Si"],
        "recommended_clads": ["SiO2", "Air"],
    },
    "cband_1550": {
        "label": "Telecom C-Band (1550 nm)",
        "wavelength_nm": 1550.0,
        "recommended_cores": ["Si", "Si3N4", "LN_Xcut", "LT_Xcut", "BTO", "PZT", "4H-SiC", "LN_Zcut", "AlN", "ScAlN_20"],
        "recommended_clads": ["SiO2", "Air"],
    },
    "swir_2000": {
        "label": "2-um SWIR / Thulium Band (2000 nm)",
        "wavelength_nm": 2000.0,
        "recommended_cores": ["Si", "Si3N4", "Ge", "4H-SiC", "AlN", "LN_Xcut", "GaN"],
        "recommended_clads": ["SiO2", "Al2O3", "CaF2", "Air"],
    },
    "midir_4200": {
        "label": "Mid-IR Gas Sensing CO2/CH4 (4200 nm)",
        "wavelength_nm": 4200.0,
        "recommended_cores": ["Ge", "Si", "Si3N4", "AlN", "GaN", "4H-SiC", "Diamond", "GSST_amorphous"],
        "recommended_clads": ["CaF2", "MgF2", "Al2O3", "Si", "Air"],
    },
    "lwir_7500": {
        "label": "Long-Wave Mid-IR Fingerprint (7500 nm)",
        "wavelength_nm": 7500.0,
        "recommended_cores": ["Ge", "Diamond"],
        "recommended_clads": ["Si", "CaF2", "Air"],
    },
}


MATERIAL_CATALOG: dict[str, dict] = {
    "Si": {
        "name": "Silicon (c-Si, O/C-band to MIR 1.15-8.5 um)",
        "category": "Group IV",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 1.5,
        "fn": lambda wl, t=T_REF_K: n_si_crystalline(wl, t, fixed_1550_index=False),
    },
    "a-Si": {
        "name": "Amorphous Silicon (a-Si:H, 0.95-5.5 um)",
        "category": "Group IV",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 3.0,
        "fn": n_asi_hydrogenated,
    },
    "Ge": {
        "name": "Germanium (c-Ge Mid-IR, 2.0-14.0 um)",
        "category": "Group IV",
        "default_band_nm": 4200.0,
        "typical_loss_db_cm": 2.0,
        "fn": n_ge_crystalline,
    },
    "Si3N4": {
        "name": "Silicon Nitride (LPCVD Si3N4, Vis-MIR 0.31-5.0 um)",
        "category": "Nitride",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.15,
        "fn": n_sin_lpcvd,
    },
    "SiRich_SiNx": {
        "name": "Silicon-Rich Nitride (SRN SiNx, 0.60-5.0 um)",
        "category": "Nitride",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.80,
        "fn": n_sirich_sinx,
    },
    "TiO2": {
        "name": "Titanium Dioxide (ALD TiO2, Visible-NIR 0.40-4.5 um)",
        "category": "Visible Oxide",
        "default_band_nm": 633.0,
        "typical_loss_db_cm": 0.80,
        "fn": n_tio2_ald,
    },
    "Ta2O5": {
        "name": "Tantalum Pentoxide (Ta2O5, Visible-NIR 0.38-4.5 um)",
        "category": "Visible Oxide",
        "default_band_nm": 780.0,
        "typical_loss_db_cm": 0.25,
        "fn": n_ta2o5_tantala,
    },
    "4H-SiC": {
        "name": "4H-Silicon Carbide (4H-SiC c-cut, Vis-MIR 0.40-5.2 um)",
        "category": "Carbide",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.30,
        "fn": n_4h_sic_anisotropic,
    },
    "3C-SiC": {
        "name": "3C-Silicon Carbide (Cubic 3C-SiC, 0.48-5.0 um)",
        "category": "Carbide",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.80,
        "fn": n_3c_sic,
    },
    "Diamond": {
        "name": "CVD Diamond (UV-Visible NV/SiV to LWIR 0.24-20 um)",
        "category": "Carbon/Quantum",
        "default_band_nm": 737.0,
        "typical_loss_db_cm": 0.35,
        "fn": n_diamond_cvd,
    },
    "AlN": {
        "name": "Aluminum Nitride (c-axis AlN, UV-Vis-MIR 0.26-5.2 um)",
        "category": "III-Nitride",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.60,
        "fn": lambda wl, t=T_REF_K: n_aln_film(wl, t),
    },
    "GaN": {
        "name": "Gallium Nitride (c-axis GaN, Vis-MIR 0.39-6.5 um)",
        "category": "III-Nitride",
        "default_band_nm": 780.0,
        "typical_loss_db_cm": 0.90,
        "fn": n_gan_wurtzite,
    },
    "ScAlN_10": {
        "name": "Sc0.10Al0.90N (10% Sc-doped AlN, 0.38-5.0 um)",
        "category": "III-Nitride",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 1.20,
        "fn": lambda wl, t=T_REF_K: n_scaln_film(wl, t, sc_fraction=0.10),
    },
    "ScAlN_20": {
        "name": "Sc0.20Al0.80N (20% Sc-doped AlN, 0.38-5.0 um)",
        "category": "III-Nitride",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 1.80,
        "fn": lambda wl, t=T_REF_K: n_scaln_film(wl, t, sc_fraction=0.20),
    },
    "ScAlN_30": {
        "name": "Sc0.30Al0.70N (30% Sc-doped AlN, 0.38-5.0 um)",
        "category": "III-Nitride",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 2.50,
        "fn": lambda wl, t=T_REF_K: n_scaln_film(wl, t, sc_fraction=0.30),
    },
    "LN_Zcut": {
        "name": "Lithium Niobate (TFLN Z-cut, Vis-MIR 0.40-5.0 um)",
        "category": "Ferroelectric",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.10,
        "fn": n_linbo3_zcut,
    },
    "LN_Xcut": {
        "name": "Lithium Niobate (TFLN X-cut, Vis-MIR 0.40-5.0 um)",
        "category": "Ferroelectric",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.10,
        "fn": lambda wl, t=T_REF_K: n_linbo3_xcut(wl, t, inplane_propagation_angle_deg=0.0),
    },
    "LT_Xcut": {
        "name": "Lithium Tantalate (TFLT X-cut, Low-Biref 0.28-5.5 um)",
        "category": "Ferroelectric",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.08,
        "fn": lambda wl, t=T_REF_K: n_litao3_xcut(wl, t, inplane_propagation_angle_deg=0.0),
    },
    "LT_Zcut": {
        "name": "Lithium Tantalate (TFLT Z-cut, UV-MIR 0.28-5.5 um)",
        "category": "Ferroelectric",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.08,
        "fn": n_litao3_zcut,
    },
    "BTO": {
        "name": "Barium Titanate (Epitaxial a-BTO on Si, Giant r42, 0.42-6.5 um)",
        "category": "Ferroelectric",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 1.50,
        "fn": n_bto_acut,
    },
    "BTO_ccut": {
        "name": "Barium Titanate (c-Oriented Tetragonal BTO, 0.42-6.5 um)",
        "category": "Ferroelectric",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 1.50,
        "fn": n_bto_ccut,
    },
    "PZT": {
        "name": "Lead Zirconate Titanate (Poled Thin-Film PZT, 0.45-6.5 um)",
        "category": "Ferroelectric",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 1.20,
        "fn": n_pzt_poled,
    },
    "Al2O3": {
        "name": "Alumina / Sapphire (Al2O3, 0.25-5.5 um)",
        "category": "Dielectric/Clad",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.20,
        "fn": n_al2o3_sapphire,
    },
    "CaF2": {
        "name": "Calcium Fluoride (CaF2 Mid-IR Clad, 0.15-9.5 um)",
        "category": "Mid-IR Cladding",
        "default_band_nm": 4200.0,
        "typical_loss_db_cm": 0.05,
        "fn": n_caf2_fluoride,
    },
    "MgF2": {
        "name": "Magnesium Fluoride (MgF2 Mid-IR Clad, 0.14-7.5 um)",
        "category": "Mid-IR Cladding",
        "default_band_nm": 4200.0,
        "typical_loss_db_cm": 0.08,
        "fn": n_mgf2_fluoride,
    },
    "SiO2": {
        "name": "Fused Silica (SiO2 BOX/Clad, 0.21-3.65 um)",
        "category": "Dielectric/Clad",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.01,
        "fn": n_sio2_sellmeier,
    },
    "GSST_amorphous": {
        "name": "GSST PCM (Amorphous Low-Loss, 1.5-10.0 um)",
        "category": "Phase-Change",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.50,
        "fn": lambda wl, t=T_REF_K: n_gsst_pcm(wl, state="amorphous"),
    },
    "GSST_crystalline": {
        "name": "GSST PCM (Crystalline High-Index, 1.5-10.0 um)",
        "category": "Phase-Change",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 150.0,
        "fn": lambda wl, t=T_REF_K: n_gsst_pcm(wl, state="crystalline"),
    },
    "Air": {
        "name": "Air / Vacuum (n = 1.0, Broadband)",
        "category": "Cladding",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.0,
        "fn": n_air,
    },
}


EXTRA_LITERATURE_MATERIALS: dict[str, dict] = {
    "GST_amorphous": {
        "name": "GST-225 PCM (Amorphous Ge2Sb2Te5, 1.1-2.5 um)",
        "category": "Phase-Change",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 8400.0,
        "fn": lambda wl, t=T_REF_K: n_gst_pcm(wl, state="amorphous"),
    },
    "GST_crystalline": {
        "name": "GST-225 PCM (Crystalline Ge2Sb2Te5, 1.1-2.5 um)",
        "category": "Phase-Change",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 380000.0,
        "fn": lambda wl, t=T_REF_K: n_gst_pcm(wl, state="crystalline"),
    },
    "Sb2Se3_amorphous": {
        "name": "Sb2Se3 PCM (Amorphous Ultra-Low-Loss, 0.98-2.2 um)",
        "category": "Phase-Change",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.20,
        "fn": lambda wl, t=T_REF_K: n_sb2se3_pcm(wl, state="amorphous"),
    },
    "Sb2Se3_crystalline": {
        "name": "Sb2Se3 PCM (Crystalline Low-Loss, 1.05-2.2 um)",
        "category": "Phase-Change",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 40.0,
        "fn": lambda wl, t=T_REF_K: n_sb2se3_pcm(wl, state="crystalline"),
    },
    "Sb2S3_amorphous": {
        "name": "Sb2S3 PCM (Amorphous Visible/NIR Low-Loss, 0.60-2.2 um)",
        "category": "Phase-Change",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.15,
        "fn": lambda wl, t=T_REF_K: n_sb2s3_pcm(wl, state="amorphous"),
    },
    "Sb2S3_crystalline": {
        "name": "Sb2S3 PCM (Crystalline Visible/NIR Low-Loss, 0.72-2.2 um)",
        "category": "Phase-Change",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.50,
        "fn": lambda wl, t=T_REF_K: n_sb2s3_pcm(wl, state="crystalline"),
    },
    "Hydex": {
        "name": "Hydex High-Index Doped Silica Glass (n~1.70, 0.40-2.6 um)",
        "category": "Dielectric/Glass",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.06,
        "fn": n_hydex_glass,
    },
    "AlGaAs": {
        "name": "Al0.2Ga0.8As-on-Insulator (0.90-4.0 um)",
        "category": "III-V",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.40,
        "fn": n_algaas_epi,
    },
    "SiO2_PLC": {
        "name": "GeO2-Doped Silica PLC Core (1.5% Delta, 0.35-3.65 um)",
        "category": "Dielectric/PLC",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.03,
        "fn": n_sio2_plc,
    },
    "ChG_As2S3": {
        "name": "Chalcogenide As2S3 Glass (0.62-11.0 um)",
        "category": "Chalcogenide",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.25,
        "fn": lambda wl, t=T_REF_K: _n_chg_as2s3(wl, t),
    },
    "GaP": {
        "name": "Gallium Phosphide GaP-on-Insulator (0.56-10.0 um)",
        "category": "III-V",
        "default_band_nm": 1550.0,
        "typical_loss_db_cm": 0.80,
        "fn": lambda wl, t=T_REF_K: _n_gap_epi(wl, t),
    },
}


def _n_chg_as2s3(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Arsenic trisulfide (As2S3) chalcogenide glass 5-term Sellmeier (0.62-11.0 um).

    Source: Rodney, Malitson & King, J. Opt. Soc. Am. 48, 633 (1958).
    """
    u = max(_check_wl(wavelength_m), 0.62)
    x = u * u
    eps = (
        1.0
        + 1.8983678 * x / (x - 0.150 ** 2)
        + 1.9222979 * x / (x - 0.250 ** 2)
        + 0.8765134 * x / (x - 0.350 ** 2)
        + 0.1188704 * x / (x - 0.450 ** 2)
        + 0.9569903 * x / (x - 27.3861 ** 2)
    )
    dn_dt = 0.9e-5
    n = math.sqrt(max(eps, 2.0)) + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        n2_m2_per_w=3.0e-18,
        validity_um=(0.62, 11.0),
        source="Rodney, Malitson & King JOSA 1958 (As2S3 Chalcogenide Glass)",
        is_anisotropic=False,
    )


def _n_gap_epi(wavelength_m: float, temperature_k: float = T_REF_K) -> MaterialIndexTensor:
    """Zincblende cubic Gallium Phosphide (GaP) Sellmeier (0.56-10.0 um).

    Source: Bond, J. Appl. Phys. 36, 1674 (1965); Parsons & Coleman, Appl. Opt. 19, 248 (1980).
    """
    u = max(_check_wl(wavelength_m), 0.56)
    x = u * u
    eps = (
        4.1705
        + 4.9113 * x / (x - 0.2904 ** 2)
        + 1.9928 * x / (x - 27.525 ** 2)
    )
    dn_dt = 1.10e-4
    n = math.sqrt(max(eps, 2.5)) + dn_dt * (temperature_k - T_REF_K)
    return MaterialIndexTensor(
        nx=n, ny=n, nz=n,
        dn_dt_per_k=dn_dt,
        r33_pm_per_v=1.07,
        n2_m2_per_w=1.1e-17,
        crystal_symmetry="zincblende_-43m",
        validity_um=(0.56, 10.0),
        source="Bond JAP 1965 / Wilson Nat. Photonics 2020 (GaP-on-Insulator)",
        is_anisotropic=False,
    )


def check_wavelength_compatibility(
    core_material: str,
    clad_material: str,
    top_clad_material: str,
    wavelength_nm: float,
) -> list[str]:
    """Return physical transparency & index-contrast warnings across UV/Vis/O/C/MIR."""
    wl_um = float(wavelength_nm) * 1e-3
    wl_m = float(wavelength_nm) * 1e-9
    warnings: list[str] = []
    try:
        core_t = get_material_fn(core_material)(wl_m, T_REF_K)
        clad_t = get_material_fn(clad_material)(wl_m, T_REF_K)
        top_t = get_material_fn(top_clad_material)(wl_m, T_REF_K)
    except Exception:
        return warnings

    lo_c, hi_c = core_t.validity_um
    if wl_um < lo_c * 0.98:
        warnings.append(
            f"Core '{core_material}' absorbs strongly at {wavelength_nm:.0f} nm (transparency >= {lo_c*1000:.0f} nm, k={core_t.kx:.2e}). "
            "Consider Si3N4, TiO2, Ta2O5, AlN, GaN, 4H-SiC, LN, or Diamond."
        )
    elif wl_um > hi_c * 1.02:
        warnings.append(
            f"Core '{core_material}' exceeds multiphonon transparency limit ({hi_c:.1f} um) at {wl_um:.2f} um. "
            "Consider Ge, Si (<=8.5 um), or Diamond."
        )

    for role, cname, ct in (("Bottom cladding", clad_material, clad_t), ("Top cladding", top_clad_material, top_t)):
        if cname == "Air":
            continue
        lo_cl, hi_cl = ct.validity_um
        if wl_um > hi_cl * 1.02 or ct.kx > 1e-4:
            warnings.append(
                f"{role} '{cname}' has Mid-IR absorption at {wl_um:.2f} um (limit ~{hi_cl:.2f} um, k={ct.kx:.2e}). "
                "Switch cladding to CaF2, MgF2, Al2O3 (<=5.5 um), Si (for Ge-on-Si), or Air."
            )

    if max(core_t.nx, core_t.ny) <= max(clad_t.nx, top_t.nx) + 0.02:
        warnings.append(
            f"Total internal reflection violated: core index ({max(core_t.nx, core_t.ny):.3f}) <= cladding index ({max(clad_t.nx, top_t.nx):.3f})."
        )
    return warnings


def get_material_fn(
    material_key: str,
    *,
    sc_fraction: float | None = None,
    custom_n: float | None = None,
    custom_k: float = 0.0,
    custom_dn_dt: float = 1.0e-5,
    custom_r33_pm_v: float = 0.0,
) -> Callable[[float, float], MaterialIndexTensor]:
    """Return `(wavelength_m, temperature_k) -> MaterialIndexTensor` callable for any catalog or custom material."""
    if material_key == "Custom" or custom_n is not None:
        n_base = float(2.0 if custom_n is None else custom_n)

        def _custom_fn(wl_m: float, temp_k: float = T_REF_K) -> MaterialIndexTensor:
            _ = _check_wl(wl_m)
            n_val = n_base + custom_dn_dt * (temp_k - T_REF_K)
            return MaterialIndexTensor(
                nx=n_val, ny=n_val, nz=n_val,
                kx=custom_k, ky=custom_k, kz=custom_k,
                dn_dt_per_k=custom_dn_dt,
                r33_pm_per_v=custom_r33_pm_v,
                source=f"User Custom Material (n={n_base:.4f}, k={custom_k:.4g})",
                is_anisotropic=False,
            )

        return _custom_fn

    if material_key.startswith("ScAlN") and sc_fraction is not None:
        x_sc = float(sc_fraction)
        return lambda wl_m, temp_k=T_REF_K: n_scaln_film(wl_m, temp_k, sc_fraction=x_sc)

    if material_key in MATERIAL_CATALOG:
        return MATERIAL_CATALOG[material_key]["fn"]
    if material_key in EXTRA_LITERATURE_MATERIALS:
        return EXTRA_LITERATURE_MATERIALS[material_key]["fn"]
    raise KeyError(
        f"Unknown material '{material_key}'. Available: {list(MATERIAL_CATALOG) + list(EXTRA_LITERATURE_MATERIALS)}"
    )


def evaluate_material_dispersion_table(
    material_key: str,
    wavelength_min_nm: float = 450.0,
    wavelength_max_nm: float = 4500.0,
    num_points: int = 120,
    temperature_k: float = T_REF_K,
    sc_fraction: float | None = None,
) -> dict:
    """Evaluate dispersion curves (nx, ny, nz, ng_x, ng_y) across a wavelength span."""
    fn = get_material_fn(material_key, sc_fraction=sc_fraction)
    # Clip range safely to material validity window
    probe = fn(1.55e-6, temperature_k)
    vmin_nm = max(wavelength_min_nm, probe.validity_um[0] * 1e3 * 1.02)
    vmax_nm = min(wavelength_max_nm, probe.validity_um[1] * 1e3 * 0.99)
    if vmax_nm <= vmin_nm:
        vmin_nm, vmax_nm = probe.validity_um[0] * 1e3 * 1.05, probe.validity_um[1] * 1e3 * 0.98

    wls_nm = np.linspace(vmin_nm, vmax_nm, num_points)
    nx_arr, ny_arr, nz_arr, kx_arr = [], [], [], []
    for w_nm in wls_nm:
        t = fn(float(w_nm) * 1e-9, temperature_k)
        nx_arr.append(t.nx)
        ny_arr.append(t.ny)
        nz_arr.append(t.nz)
        kx_arr.append(t.kx)

    nx_np = np.asarray(nx_arr, dtype=float)
    ny_np = np.asarray(ny_arr, dtype=float)
    dnx_dl = np.gradient(nx_np, wls_nm)
    dny_dl = np.gradient(ny_np, wls_nm)
    ngx_np = nx_np - wls_nm * dnx_dl
    ngy_np = ny_np - wls_nm * dny_dl

    ref_1550 = fn(1.55e-6 if probe.validity_um[0] <= 1.55 <= probe.validity_um[1] else (0.5 * (vmin_nm + vmax_nm) * 1e-9), temperature_k)
    return {
        "material_key": material_key,
        "display_name": MATERIAL_CATALOG.get(material_key, {}).get("name", material_key),
        "wavelength_nm": np.round(wls_nm, 3).tolist(),
        "nx": np.round(nx_np, 6).tolist(),
        "ny": np.round(ny_np, 6).tolist(),
        "nz": np.round(nz_arr, 6).tolist(),
        "kx": np.round(kx_arr, 6).tolist(),
        "ng_x": np.round(ngx_np, 6).tolist(),
        "ng_y": np.round(ngy_np, 6).tolist(),
        "properties_at_ref": ref_1550.to_dict(),
    }
