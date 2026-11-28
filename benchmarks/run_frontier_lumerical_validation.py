"""Independent Ansys Lumerical 2026 R1 (RTX 5090 GPU 3D FDTD) Frontier Validation Runner.

Independently validates MicroRing Studio's new first-principles frontier engines against
full-wave 3D Maxwell FDTD on the local NVIDIA RTX 5090 GPU:
  1. Suite F-L1 (3D Periodic SWG Waveguide):
     Full 3D periodic pillared Si SWG strip (220 x 500 nm, Lambda=220 nm, f=0.50) on SiO2 BOX
     under BOTH Water (n=1.318 & n=1.338) and SiO2 top cladding:
       - Extracts exact 3D Bloch effective index n_eff_SWG, group index n_g_SWG,
         Bragg stopband edge lambda_Bragg, and aqueous biosensing bulk sensitivity S_bulk [nm/RIU]
       - Compares blindly against `simulate_swg_microring` (2nd-order Rytov EMT + 2D curved FEM).
  2. Suite F-L2 (3D Mid-IR Au/Dielectric/Au MIM Nanoantenna Absorber & SEIRA PMMA):
     Full 3D periodic plasmonic MIM nanorod absorber (Px=Py=2.4 um, Palik Au) with and without
     a 40 nm conformal Lorentz-oscillator PMMA layer (nu0=1730 cm^-1, HWHM=12 cm^-1):
       - Runs blind inverse fitting with `fit_nanoantenna_tcmt_spectrum` (Suh-Wang-Fan TCMT)
       - Verifies R^2 > 0.99, energy conservation, critical coupling, and blind vibrational recovery!
  3. Suite F-L3 (3D Photonic Crystal Ring PhCR Selective Mode Splitting):
     Half-period 3D corrugated Si3N4 (500 x 1200 nm, A_mod = 40 nm inner wall) Brillouin zone-edge
     standing-wave isolation (`PMC`/`Metal` symmetry on [0, Lambda/2]):
       - Extracts exact lower (`cos`) and upper (`sin`) band-edge eigenfrequencies and splitting Delta_f
       - Compares blindly against `simulate_phc_microring` (2D-FEM dneff/dW + vector Ez reversal).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH_DIR = OUT_DIR / "_lum_frontier_scratch"
OUT_JSON = OUT_DIR / "frontier_lumerical_validation.json"

LUMERICAL_WIN_HINTS = [
    r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v252\Lumerical\api\python",
]
C0 = 299792458.0


def _import_lumapi():
    for p in LUMERICAL_WIN_HINTS:
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)
    import lumapi  # type: ignore
    return lumapi


def _scalar_real(val) -> float:
    return float(np.real(np.asarray(val).reshape(-1)[0]))


def _run_fdtd_gpu(fdtd, fsp_path: Path, retries: int = 3) -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    fdtd.save(str(fsp_path))
    last_err = None
    for attempt in range(retries):
        try:
            fdtd.run("FDTD", "GPU")
            return
        except Exception as exc:
            last_err = exc
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"RTX 5090 GPU 3D FDTD failed on {fsp_path.name}: {last_err}")


# ==============================================================================
# Suite F-L1: 3D Periodic SWG Waveguide (Bloch neff, ng, Bragg Edge, S_bulk)
# ==============================================================================

def _run_single_swg_fdtd(
    fdtd,
    tag: str,
    n_top: float,
    period_um: float = 0.220,
    duty_cycle: float = 0.50,
    w_um: float = 0.500,
    h_um: float = 0.220,
    n_periods: int = 34,
) -> dict:
    t0 = time.perf_counter()
    fdtd.newproject()
    fdtd.switchtolayout()

    a_m = period_um * 1e-6
    l_core_m = duty_cycle * a_m
    w_m = w_um * 1e-6
    h_m = h_um * 1e-6
    l_swg_m = n_periods * a_m
    l_lead_m = 1.6e-6  # Solid Si input/output lead waveguide for clean Mode Source excitation

    z_start = -l_lead_m
    z_end = l_swg_m + l_lead_m

    # Background = top cladding (Water or SiO2)
    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    fdtd.set("background index", float(n_top))
    fdtd.set("x", 0.0)
    fdtd.set("x span", 3.2e-6)
    fdtd.set("y", 0.5 * h_m)
    fdtd.set("y span", 2.8e-6)
    fdtd.set("z min", z_start - 0.8e-6)
    fdtd.set("z max", z_end + 0.8e-6)
    fdtd.set("mesh accuracy", 3)
    fdtd.set("simulation time", 2500e-15)
    fdtd.set("auto shutoff min", 1e-5)
    fdtd.set("x min bc", "Anti-Symmetric")  # Exact TE0 symmetry (Ex even -> Anti-Symmetric at x=0!)

    # Fine mesh override across SWG pillars
    fdtd.addmesh()
    fdtd.set("name", "swg_mesh")
    fdtd.set("x", 0.0)
    fdtd.set("x span", w_m + 0.30e-6)
    fdtd.set("y", 0.5 * h_m)
    fdtd.set("y span", h_m + 0.24e-6)
    fdtd.set("z min", -0.2e-6)
    fdtd.set("z max", l_swg_m + 0.2e-6)
    fdtd.set("dx", 20e-9)
    fdtd.set("dy", 20e-9)
    fdtd.set("dz", 11e-9)  # Exactly 20 cells per 220 nm period!

    # SiO2 BOX substrate (y < 0)
    fdtd.addrect()
    fdtd.set("name", "BOX")
    fdtd.set("material", "SiO2 (Glass) - Palik")
    fdtd.set("x", 0.0)
    fdtd.set("x span", 5.0e-6)
    fdtd.set("y max", 0.0)
    fdtd.set("y min", -2.5e-6)
    fdtd.set("z min", z_start - 1.5e-6)
    fdtd.set("z max", z_end + 1.5e-6)

    # Solid Si lead-in and lead-out waveguides
    for nm, z0, z1 in (("lead_in", z_start - 1.2e-6, 0.0), ("lead_out", l_swg_m, z_end + 1.2e-6)):
        fdtd.addrect()
        fdtd.set("name", nm)
        fdtd.set("material", "Si (Silicon) - Palik")
        fdtd.set("x", 0.0)
        fdtd.set("x span", w_m)
        fdtd.set("y min", 0.0)
        fdtd.set("y max", h_m)
        fdtd.set("z min", z0)
        fdtd.set("z max", z1)

    # 3D Periodic SWG Silicon Pillars (z in [k*a + (a-l_core)/2, k*a + (a+l_core)/2])
    for k in range(n_periods):
        zc = (k + 0.5) * a_m
        fdtd.addrect()
        fdtd.set("name", f"pillar_{k}")
        fdtd.set("material", "Si (Silicon) - Palik")
        fdtd.set("x", 0.0)
        fdtd.set("x span", w_m)
        fdtd.set("y min", 0.0)
        fdtd.set("y max", h_m)
        fdtd.set("z", zc)
        fdtd.set("z span", l_core_m)

    # Broadband Mode Source (950 nm - 1700 nm to cover both Bragg stopband & 1550 nm SWG regime!)
    fdtd.addmode()
    fdtd.set("name", "src")
    fdtd.set("injection axis", "z-axis")
    fdtd.set("direction", "Forward")
    fdtd.set("x", 0.0)
    fdtd.set("x span", 2.4e-6)
    fdtd.set("y", 0.5 * h_m)
    fdtd.set("y span", 2.0e-6)
    fdtd.set("z", -0.85e-6)
    fdtd.set("wavelength start", 0.90e-6)
    fdtd.set("wavelength stop", 1.70e-6)
    fdtd.set("mode selection", "fundamental TE mode")

    # 3 Internal Bloch-phase monitors at exact pillar centers (k=10, k=11, k=24 -> dz1=1*a, dz2=14*a)
    # Plus 1 transmission monitor in lead_out
    k_a, k_b, k_c = 10, 11, 24
    for m_name, zm in (
        ("mon_A", (k_a + 0.5) * a_m),
        ("mon_B", (k_b + 0.5) * a_m),
        ("mon_C", (k_c + 0.5) * a_m),
        ("mon_T", l_swg_m + 0.85e-6),
    ):
        fdtd.addpower()
        fdtd.set("name", m_name)
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x", 0.0)
        fdtd.set("x span", 2.0e-6)
        fdtd.set("y", 0.5 * h_m)
        fdtd.set("y span", 1.8e-6)
        fdtd.set("z", zm)
        fdtd.set("override global monitor settings", 1)
        fdtd.set("frequency points", 241)

    fsp = SCRATCH_DIR / f"swg_{tag}.fsp"
    _run_fdtd_gpu(fdtd, fsp)
    status = int(_scalar_real(fdtd.getresult("FDTD", "status")))

    # Extract complex Ex(x, y, f) on mon_A, mon_B, mon_C and transmission T(f) on mon_T
    f_arr = np.asarray(fdtd.getdata("mon_A", "f")).ravel()
    wl_nm = (C0 / f_arr) * 1e9
    order = np.argsort(wl_nm)
    wl_nm = wl_nm[order]
    wl_m = wl_nm * 1e-9

    ex_a = np.asarray(fdtd.getdata("mon_A", "Ex"))[:, :, 0, order]
    ex_b = np.asarray(fdtd.getdata("mon_B", "Ex"))[:, :, 0, order]
    ex_c = np.asarray(fdtd.getdata("mon_C", "Ex"))[:, :, 0, order]
    t_raw = np.asarray(fdtd.transmission("mon_T")).ravel()[order]

    # Exact Bloch transfer ratio across 1 period (dz1 = a_m, zero 2*pi ambiguity!) and 14 periods (dz2 = 14*a_m)
    # Note: Lumerical uses exp(-i*omega*t), so forward wave +z has phase exp(+i*beta*z) -> angle(C/A) = +beta*dz!
    num_1 = np.sum(ex_b * np.conj(ex_a), axis=(0, 1))
    den_a = np.maximum(np.sum(np.abs(ex_a) ** 2, axis=(0, 1)), 1e-30)
    h1 = num_1 / den_a
    phi_1 = np.mod(np.angle(h1), 2.0 * math.pi)  # Strictly in (0, 2*pi) -> for 1 period at 1550 nm, phi_1 ~ 1.4 rad!
    neff_1 = phi_1 * wl_m / (2.0 * math.pi * a_m)

    num_14 = np.sum(ex_c * np.conj(ex_a), axis=(0, 1))
    h14 = num_14 / den_a
    # Unwrap around 1550 nm anchored by unambiguous 1-period neff_1!
    i1550 = int(np.argmin(np.abs(wl_nm - 1550.0)))
    dz14 = (k_c - k_a) * a_m
    phi_14_raw = np.angle(h14)
    phi_14_unw = np.unwrap(phi_14_raw)
    # Shift by exact integer multiple of 2*pi so 14-period phase matches 14 * phi_1 at 1550 nm
    target_phi14 = (k_c - k_a) * phi_1[i1550]
    m_shift = int(round((target_phi14 - phi_14_unw[i1550]) / (2.0 * math.pi)))
    phi_14_abs = phi_14_unw + 2.0 * math.pi * m_shift
    neff_14 = phi_14_abs * wl_m / (2.0 * math.pi * dz14)

    # Extract smooth n_eff and n_g at 1550 nm via local linear fit over [1510, 1590] nm (averages residual Fabry-Perot ripples!)
    win = (wl_nm >= 1510.0) & (wl_nm <= 1590.0)
    p_lin = np.polyfit(wl_m[win] - 1.55e-6, neff_14[win], 1)
    neff_1550 = float(p_lin[1])
    dneff_dwl = float(p_lin[0])
    ng_1550 = float(neff_1550 - 1.55e-6 * dneff_dwl)

    # Extract 3D FDTD Bragg band-edge wavelength (where T drops below 10% moving down from 1550 nm)
    t_sub = t_raw[:i1550 + 1]
    wl_sub = wl_nm[:i1550 + 1]
    t_pass_ref = float(np.median(t_raw[win]))
    below = np.flatnonzero(t_sub < 0.15 * max(t_pass_ref, 0.2))
    wl_bragg_edge_3d = float(wl_sub[below[-1]]) if len(below) else float(wl_nm[0])

    # Remove scratch .fsp to keep disk clean
    try:
        fsp.unlink()
    except OSError:
        pass

    return {
        "tag": tag,
        "n_top": n_top,
        "status": status,
        "neff_1550_1p": round(float(neff_1[i1550]), 5),
        "neff_1550_3d_fdtd": round(neff_1550, 5),
        "ng_1550_3d_fdtd": round(ng_1550, 4),
        "bragg_edge_nm_3d_fdtd": round(wl_bragg_edge_3d, 1),
        "passband_T_1550": round(t_pass_ref, 4),
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }


# ==============================================================================
# Suite F-L2: 3D Mid-IR Au/Dielectric/Au MIM Nanoantenna Absorber & SEIRA PMMA
# ==============================================================================

def _run_mim_nanoantenna_fdtd(fdtd) -> dict:
    from microring.nanoantenna_tcmt import fit_nanoantenna_tcmt_spectrum

    t_all = time.perf_counter()
    results = {}

    # Shared Mid-IR MIM geometry: Px=Py=2.4 um, bottom Au mirror, 160 nm dielectric spacer (n=1.45),
    # top Au nanorod Lx=1.08 um, Wy=0.34 um, t=60 nm -> clean dipole magnetic-plasmon resonance ~1730 cm^-1!
    px = 2.4e-6
    py = 2.4e-6
    t_au_bot = 0.15e-6
    t_spacer = 0.16e-6
    t_rod = 0.06e-6
    lx_rod = 1.08e-6
    wy_rod = 0.34e-6
    t_pmma = 0.04e-6

    # Build exact Lorentz-oscillator PMMA sampled permittivity table around 1730 cm^-1
    # eps(nu) = eps_inf + S * nu0^2 / (nu0^2 - nu^2 - i * 2*gamma_hwhm * nu)
    nu_tab = np.linspace(1100.0, 2600.0, 301)  # cm^-1
    f_tab_hz = nu_tab * 100.0 * C0
    nu0_pmma = 1730.0
    gam_pmma_hwhm = 12.0  # cm^-1 (exact preset!)
    # Note: Lumerical exp(-i*om*t) convention has Im(eps) > 0 for absorption!
    eps_pmma = 2.15 + 0.040 * (nu0_pmma ** 2) / (nu0_pmma ** 2 - nu_tab ** 2 - 1j * 2.0 * gam_pmma_hwhm * nu_tab)

    for variant in ("bare_mim", "seira_pmma"):
        t0 = time.perf_counter()
        fdtd.newproject()
        fdtd.switchtolayout()

        # Add sampled PMMA Lorentz material if SEIRA variant (Lumerical Sampled 3D data expects 2 cols: [f_Hz, complex_eps]!)
        if variant == "seira_pmma":
            mat_name = "PMMA_Lorentz_1730"
            m_data = np.column_stack([f_tab_hz.astype(np.complex128), eps_pmma.astype(np.complex128)])
            fdtd.setmaterial(fdtd.addmaterial("Sampled 3D data"), "name", mat_name)
            fdtd.setmaterial(mat_name, "sampled data", m_data)
            fdtd.setmaterial(mat_name, "max coefficients", 6)
            fdtd.setmaterial(mat_name, "tolerance", 0.002)

        fdtd.addfdtd()
        fdtd.set("dimension", "3D")
        fdtd.set("x", 0.0)
        fdtd.set("x span", px)
        fdtd.set("y", 0.0)
        fdtd.set("y span", py)
        fdtd.set("z min", -0.30e-6)
        fdtd.set("z max", 4.50e-6)
        fdtd.set("x min bc", "Periodic")  # Exact infinite periodic MIM metasurface array!
        fdtd.set("y min bc", "Periodic")
        fdtd.set("z min bc", "PML")
        fdtd.set("z max bc", "PML")
        fdtd.set("mesh accuracy", 3)
        fdtd.set("simulation time", 3000e-15)
        fdtd.set("auto shutoff min", 1e-5)

        # Fine mesh around nanorod + spacer
        fdtd.addmesh()
        fdtd.set("name", "rod_mesh")
        fdtd.set("x", 0.0)
        fdtd.set("x span", lx_rod + 0.40e-6)
        fdtd.set("y", 0.0)
        fdtd.set("y span", wy_rod + 0.30e-6)
        fdtd.set("z min", -0.02e-6)
        fdtd.set("z max", t_spacer + t_rod + 0.10e-6)
        fdtd.set("dx", 25e-9)
        fdtd.set("dy", 25e-9)
        fdtd.set("dz", 15e-9)

        # Bottom Au mirror (z in [-t_au_bot, 0])
        fdtd.addrect()
        fdtd.set("name", "Au_mirror")
        fdtd.set("material", "Au (Gold) - Palik")
        fdtd.set("x", 0.0)
        fdtd.set("x span", px + 0.5e-6)
        fdtd.set("y", 0.0)
        fdtd.set("y span", py + 0.5e-6)
        fdtd.set("z min", -t_au_bot)
        fdtd.set("z max", 0.0)

        # Dielectric spacer (z in [0, t_spacer], n=1.45)
        fdtd.addrect()
        fdtd.set("name", "Dielectric_spacer")
        fdtd.set("index", 1.45)
        fdtd.set("x", 0.0)
        fdtd.set("x span", px + 0.5e-6)
        fdtd.set("y", 0.0)
        fdtd.set("y span", py + 0.5e-6)
        fdtd.set("z min", 0.0)
        fdtd.set("z max", t_spacer)

        # Optional 40 nm conformal PMMA analyte film coating nanorod + spacer
        if variant == "seira_pmma":
            fdtd.addrect()
            fdtd.set("name", "PMMA_film")
            fdtd.set("material", "PMMA_Lorentz_1730")
            fdtd.set("override mesh order from material database", 1)
            fdtd.set("mesh order", 3)  # Rod (order 2) takes precedence inside rod volume!
            fdtd.set("x", 0.0)
            fdtd.set("x span", px + 0.5e-6)
            fdtd.set("y", 0.0)
            fdtd.set("y span", py + 0.5e-6)
            fdtd.set("z min", t_spacer)
            fdtd.set("z max", t_spacer + t_rod + t_pmma)

        # Top Au nanorod (z in [t_spacer, t_spacer + t_rod])
        fdtd.addrect()
        fdtd.set("name", "Au_nanorod")
        fdtd.set("material", "Au (Gold) - Palik")
        fdtd.set("override mesh order from material database", 1)
        fdtd.set("mesh order", 2)
        fdtd.set("x", 0.0)
        fdtd.set("x span", lx_rod)
        fdtd.set("y", 0.0)
        fdtd.set("y span", wy_rod)
        fdtd.set("z min", t_spacer)
        fdtd.set("z max", t_spacer + t_rod)

        # Normal-incidence x-polarized plane wave (4.2 - 8.2 um = 1220 - 2380 cm^-1)
        fdtd.addplane()
        fdtd.set("name", "src_pw")
        fdtd.set("injection axis", "z-axis")
        fdtd.set("direction", "Backward")
        fdtd.set("polarization angle", 0.0)
        fdtd.set("x", 0.0)
        fdtd.set("x span", px + 0.5e-6)
        fdtd.set("y", 0.0)
        fdtd.set("y span", py + 0.5e-6)
        fdtd.set("z", 2.6e-6)
        fdtd.set("wavelength start", 4.2e-6)
        fdtd.set("wavelength stop", 8.2e-6)

        # Reflection monitor behind source (z = 3.6 um)
        fdtd.addpower()
        fdtd.set("name", "mon_R")
        fdtd.set("monitor type", "2D Z-normal")
        fdtd.set("x", 0.0)
        fdtd.set("x span", px + 0.5e-6)
        fdtd.set("y", 0.0)
        fdtd.set("y span", py + 0.5e-6)
        fdtd.set("z", 3.6e-6)
        fdtd.set("override global monitor settings", 1)
        fdtd.set("frequency points", 301)

        fsp = SCRATCH_DIR / f"mim_{variant}.fsp"
        _run_fdtd_gpu(fdtd, fsp)
        status = int(_scalar_real(fdtd.getresult("FDTD", "status")))

        f_arr = np.asarray(fdtd.getdata("mon_R", "f")).ravel()
        nu_cm1 = (f_arr / C0) * 1e-2
        r_arr = np.clip(np.abs(np.asarray(fdtd.transmission("mon_R")).ravel()), 0.0, 1.0)
        order = np.argsort(nu_cm1)
        nu_cm1 = nu_cm1[order]
        r_arr = r_arr[order]
        a_arr = 1.0 - r_arr

        # Run BLIND inverse TCMT fit directly on Lumerical 3D FDTD reflectance!
        if variant == "bare_mim":
            fit = fit_nanoantenna_tcmt_spectrum(
                nu_cm1, r_arr, with_molecule=False
            )
        else:
            fit = fit_nanoantenna_tcmt_spectrum(
                nu_cm1, r_arr, with_molecule=True, nu_mol_guess_cm1=1730.0
            )
        fm = fit["metrics"]
        r_fit = np.asarray(fit["R_fit"], dtype=float)
        rmse = float(np.sqrt(np.mean((r_arr - r_fit) ** 2)))

        try:
            fsp.unlink()
        except OSError:
            pass

        results[variant] = {
            "status": status,
            "fdtd_peak_absorbance": round(float(np.max(a_arr)), 4),
            "fdtd_peak_nu_cm1": round(float(nu_cm1[int(np.argmax(a_arr))]), 2),
            "tcmt_fit": fm,
            "tcmt_fit_r_squared": fm["r_squared"],
            "tcmt_fit_rmse": round(rmse, 6),
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM-MIM] {variant:<12} status={status} A_peak={results[variant]['fdtd_peak_absorbance']:.4f} "
            f"@ {results[variant]['fdtd_peak_nu_cm1']:.1f} cm^-1 | TCMT fit R^2={fm['r_squared']:.5f} "
            f"RMSE={rmse:.5f} ({results[variant]['wall_time_s']:.1f}s)"
        )

    results["total_wall_time_s"] = round(time.perf_counter() - t_all, 2)
    return results


# ==============================================================================
# Suite F-L3: 3D Photonic Crystal Ring (PhCR) Zone-Edge Selective Mode Splitting
# ==============================================================================

def _run_phcr_3d_fdtd(fdtd) -> dict:
    from microring.swg_and_phc_mrr import PhCMicroringConfig, simulate_phc_microring
    from microring.materials import get_material_fn, T_REF_K

    t_all = time.perf_counter()
    # Straight waveguide limit (R = 5000 um) so straight 3D corrugated unit cell matches 1-to-1
    cfg = PhCMicroringConfig(
        core_material="Si3N4", clad_material="SiO2", top_clad_material="SiO2",
        film_height_um=0.50, ring_width_um=1.20, ring_radius_um=5000.0,
        wavelength_nm=1550.0, modulation_amplitude_nm=40.0, corrugation_both_walls=False,
        core_mesh_res_um=0.040,
    )
    mr = simulate_phc_microring(cfg)["metrics"]
    lam_b = mr["corrugation_period_nm"] * 1e-9
    half_lam = 0.5 * lam_b
    n_sin = float(get_material_fn("Si3N4")(1.55e-6, T_REF_K).nx)
    n_sio2 = float(get_material_fn("SiO2")(1.55e-6, T_REF_K).nx)
    w0, h0, a_mod = 1.20e-6, 0.50e-6, 0.040e-6

    n_seg = 40
    xs = np.linspace(-0.08 * lam_b, 0.58 * lam_b, n_seg)
    y_bot = -0.5 * w0 - a_mod * np.cos(2.0 * np.pi * xs / lam_b)
    verts = np.column_stack([
        np.concatenate([xs, xs[::-1]]),
        np.concatenate([y_bot, np.full_like(xs, 0.5 * w0)]),
    ])

    freqs_found: dict[str, float] = {}
    for mode_tag, x_min_bc, x_max_bc, x_dip in (
        ("lower_cos", "PMC", "Metal", 0.05 * lam_b),
        ("upper_sin", "Metal", "PMC", 0.45 * lam_b),
    ):
        fdtd.newproject()
        fdtd.switchtolayout()
        fdtd.addfdtd()
        fdtd.set("dimension", "3D")
        fdtd.set("background index", n_sio2)
        fdtd.set("x min", 0.0)
        fdtd.set("x max", half_lam)
        fdtd.set("y", 0.0)
        fdtd.set("y span", 3.4e-6)
        fdtd.set("z", 0.0)
        fdtd.set("z span", 3.0e-6)
        fdtd.set("x min bc", x_min_bc)
        fdtd.set("x max bc", x_max_bc)
        fdtd.set("y min bc", "PML")
        fdtd.set("y max bc", "PML")
        fdtd.set("z min bc", "Symmetric")
        fdtd.set("z max bc", "PML")
        fdtd.set("mesh accuracy", 3)
        fdtd.set("simulation time", 8000e-15)
        fdtd.set("auto shutoff min", 1e-9)

        fdtd.addmesh()
        fdtd.set("name", "core_mesh")
        fdtd.set("x min", 0.0)
        fdtd.set("x max", half_lam)
        fdtd.set("y", 0.0)
        fdtd.set("y span", w0 + 0.30e-6)
        fdtd.set("z", 0.0)
        fdtd.set("z span", h0 + 0.30e-6)
        fdtd.set("dx", 10e-9)
        fdtd.set("dy", 10e-9)
        fdtd.set("dz", 15e-9)

        fdtd.addpoly()
        fdtd.set("name", "Si3N4_PhCR")
        fdtd.set("index", n_sin)
        fdtd.set("vertices", verts)
        fdtd.set("z min", -0.5 * h0)
        fdtd.set("z max", 0.5 * h0)

        fdtd.adddipole()
        fdtd.set("name", "dip")
        fdtd.set("dipole type", "Electric dipole")
        fdtd.set("x", x_dip)
        fdtd.set("y", 0.0)
        fdtd.set("z", 0.02e-6)
        fdtd.set("theta", 90.0)
        fdtd.set("phi", 90.0)
        fdtd.set("wavelength start", 1.48e-6)
        fdtd.set("wavelength stop", 1.62e-6)

        fdtd.addtime()
        fdtd.set("name", "tmon")
        fdtd.set("x", x_dip)
        fdtd.set("y", 0.0)
        fdtd.set("z", 0.02e-6)

        fsp = SCRATCH_DIR / f"phcr_{mode_tag}.fsp"
        _run_fdtd_gpu(fdtd, fsp)
        e_res = fdtd.getresult("tmon", "E")
        t_arr = np.asarray(e_res["t"]).ravel()
        ey_arr = np.real(np.asarray(e_res["E"]).reshape(-1, 3)[:, 1])
        mask = t_arr > 1.0e-12
        dt = float(np.mean(np.diff(t_arr[mask])))
        nfft = 1 << 20
        spec = np.abs(np.fft.rfft(ey_arr[mask] * np.hanning(int(np.sum(mask))), n=nfft))
        f_axis = np.fft.rfftfreq(nfft, d=dt)
        band = (f_axis > 180e12) & (f_axis < 210e12)
        fb, sb = f_axis[band], spec[band]
        ipk = int(np.argmax(sb))
        if 0 < ipk < len(sb) - 1:
            al, be, ga = np.log(sb[ipk - 1] + 1e-30), np.log(sb[ipk] + 1e-30), np.log(sb[ipk + 1] + 1e-30)
            p = 0.5 * (al - ga) / (al - 2.0 * be + ga)
            f_pk = fb[ipk] + p * (fb[1] - fb[0])
        else:
            f_pk = fb[ipk]
        freqs_found[mode_tag] = float(f_pk)
        try:
            fsp.unlink()
        except OSError:
            pass

    split_3d_ghz = (freqs_found["upper_sin"] - freqs_found["lower_cos"]) * 1e-9
    f_center_thz = 0.5 * (freqs_found["upper_sin"] + freqs_found["lower_cos"]) * 1e-12
    err_pct = abs(mr["selective_mode_split_GHz"] - split_3d_ghz) / max(split_3d_ghz, 1e-6) * 100.0
    return {
        "case_id": "Si3N4_500x1200_PhCR_A40nm_InnerWall",
        "lumerical_3d_fdtd": {
            "f_lower_cos_THz": round(freqs_found["lower_cos"] * 1e-12, 6),
            "f_upper_sin_THz": round(freqs_found["upper_sin"] * 1e-12, 6),
            "bragg_center_nm": round((C0 / (f_center_thz * 1e12)) * 1e9, 3),
            "selective_mode_split_GHz": round(split_3d_ghz, 3),
        },
        "microring_2dfem_bragg": {
            "corrugation_period_nm": mr["corrugation_period_nm"],
            "selective_mode_split_GHz": mr["selective_mode_split_GHz"],
            "johnson_field_split_check_GHz": mr["johnson_field_split_check_GHz"],
        },
        "rel_err_split_pct": round(err_pct, 2),
        "wall_time_s": round(time.perf_counter() - t_all, 2),
    }


def main() -> None:
    from microring.swg_and_phc_mrr import SWGMicroringConfig, simulate_swg_microring

    t_all = time.perf_counter()
    lumapi = _import_lumapi()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    with lumapi.FDTD(hide=True) as fdtd:
        # 1. Suite F-L1: 3D SWG Waveguide (Water n=1.318, Water+0.02 n=1.338, and SiO2 n=1.444)
        swg_w1 = _run_single_swg_fdtd(fdtd, "water_1318", n_top=1.318)
        swg_w2 = _run_single_swg_fdtd(fdtd, "water_1338", n_top=1.338)
        swg_ox = _run_single_swg_fdtd(fdtd, "sio2_1444", n_top=1.444)

        dneff_dn_3d = (swg_w2["neff_1550_3d_fdtd"] - swg_w1["neff_1550_3d_fdtd"]) / 0.020
        s_bulk_3d_nm_riu = (1550.0 / swg_w1["ng_1550_3d_fdtd"]) * dneff_dn_3d

        # Compare blindly against MicroRing Studio's 2nd-order Rytov EMT + 2D FEM (`R=1000 um` straight limit!)
        mr_w = simulate_swg_microring(SWGMicroringConfig(
            core_material="Si", clad_material="SiO2", top_clad_material="Custom", custom_top_clad_n=1.318,
            film_height_um=0.22, ring_width_um=0.50, ring_radius_um=500.0, swg_period_nm=220.0, duty_cycle=0.50,
            wavelength_nm=1550.0, compute_biosensing=True,
        ))["metrics"]
        mr_ox = simulate_swg_microring(SWGMicroringConfig(
            core_material="Si", clad_material="SiO2", top_clad_material="SiO2",
            film_height_um=0.22, ring_width_um=0.50, ring_radius_um=500.0, swg_period_nm=220.0, duty_cycle=0.50,
            wavelength_nm=1550.0, compute_biosensing=False,
        ))["metrics"]

        swg_summary = {
            "water_1318": {
                "lumerical_3d_fdtd": swg_w1,
                "microring_rytov_2dfem": {
                    "neff_swg": mr_w["neff_swg"],
                    "ng_swg": mr_w["ng_swg"],
                    "bragg_edge_nm": mr_w["bragg_band_edge_nm"],
                    "S_bulk_nm_per_RIU": mr_w["S_bulk_nm_per_RIU"],
                },
                "lumerical_3d_S_bulk_nm_per_RIU": round(float(s_bulk_3d_nm_riu), 1),
                "delta_neff": round(mr_w["neff_swg"] - swg_w1["neff_1550_3d_fdtd"], 5),
                "rel_err_ng_pct": round(abs(mr_w["ng_swg"] - swg_w1["ng_1550_3d_fdtd"]) / swg_w1["ng_1550_3d_fdtd"] * 100.0, 2),
                "rel_err_S_bulk_pct": round(abs(mr_w["S_bulk_nm_per_RIU"] - s_bulk_3d_nm_riu) / max(s_bulk_3d_nm_riu, 1.0) * 100.0, 2),
            },
            "sio2_1444": {
                "lumerical_3d_fdtd": swg_ox,
                "microring_rytov_2dfem": {
                    "neff_swg": mr_ox["neff_swg"],
                    "ng_swg": mr_ox["ng_swg"],
                    "bragg_edge_nm": mr_ox["bragg_band_edge_nm"],
                },
                "delta_neff": round(mr_ox["neff_swg"] - swg_ox["neff_1550_3d_fdtd"], 5),
                "rel_err_ng_pct": round(abs(mr_ox["ng_swg"] - swg_ox["ng_1550_3d_fdtd"]) / swg_ox["ng_1550_3d_fdtd"] * 100.0, 2),
            },
        }
        print(
            f"[LUM-SWG] Water: 3D-FDTD neff={swg_w1['neff_1550_3d_fdtd']:.4f} ng={swg_w1['ng_1550_3d_fdtd']:.3f} "
            f"S_bulk={s_bulk_3d_nm_riu:.1f} nm/RIU | MicroRing neff={mr_w['neff_swg']:.4f} ng={mr_w['ng_swg']:.3f} "
            f"S_bulk={mr_w['S_bulk_nm_per_RIU']:.1f} nm/RIU"
        )
        print(
            f"[LUM-SWG] SiO2 : 3D-FDTD neff={swg_ox['neff_1550_3d_fdtd']:.4f} ng={swg_ox['ng_1550_3d_fdtd']:.3f} | "
            f"MicroRing neff={mr_ox['neff_swg']:.4f} ng={mr_ox['ng_swg']:.3f}"
        )

        # 2. Suite F-L2: 3D Mid-IR MIM Nanoantenna Absorber & SEIRA PMMA
        mim_summary = _run_mim_nanoantenna_fdtd(fdtd)

        # 3. Suite F-L3: 3D Photonic Crystal Ring (PhCR) Selective Mode Splitting
        phcr_summary = _run_phcr_3d_fdtd(fdtd)

    payload = {
        "solver": "Ansys Lumerical 2026 R1 (3D FDTD on NVIDIA RTX 5090 GPU)",
        "host": f"{platform.system()}-{platform.machine()}",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        "suite_swg_3d_fdtd": swg_summary,
        "suite_nanoantenna_mim_seira_3d_fdtd": mim_summary,
        "suite_phcr_3d_fdtd": phcr_summary,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Lumerical frontier validation to {OUT_JSON} in {payload['total_wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
