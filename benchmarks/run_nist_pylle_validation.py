"""Independent Head-to-Head LLE Validation: MicroRing Studio vs. Official NIST pyLLE (ComputeLLE.jl).

Runs the unmodified official NIST `usnistgov/pyLLE` Julia backend (`ComputeLLE.jl`,
G. Moille et al., J. Res. NIST 124, 124012, 2019) head-to-head against MicroRing Studio's
`nonlinear_lle.py` Strang SSFM solver on identical neutral parameters across three landmark
microcomb benchmarks:
  1. Brasch et al., Science 351, 357 (2016): Si3N4 800x1600 nm, R=23 um, 1550 nm, 80 mW
  2. Guidry et al., Optica 7, 1139 (2020): 4H-SiC 500x1850 nm, R=55 um, 1550 nm, 20 mW
  3. NIST Official Example (`RW1000_H430.csv`, Moille et al., J. Res. NIST 2019):
     Si3N4 1000x430 nm, R=23 um, 283 THz (1060 nm), 140 mW, octave-spanning dual-DW soliton

Usage:
  git clone --depth 1 https://github.com/usnistgov/pyLLE.git /tmp/pyLLE
  julia /tmp/pyLLE/InstallPkg.jl
  python benchmarks/run_nist_pylle_validation.py --pylle-dir /tmp/pyLLE
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

C0 = 299_792_458.0


def find_julia_exe(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    found = shutil.which("julia")
    if found:
        return found
    for base in (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
        Path(os.environ.get("USERPROFILE", "")) / ".juliaup" / "bin",
    ):
        if base.is_dir():
            for p in base.rglob("julia.exe"):
                return str(p)
    return "julia"


def run_nist_compute_lle_jl(
    *,
    julia_exe: str,
    compute_jl: Path,
    r_m: float,
    ng: float,
    qi: float,
    qc: float,
    gamma_per_w_m: float,
    f0_hz: float,
    d1_rad_s: float,
    mu_odd: np.ndarray,
    dint_odd_rad_s: np.ndarray,
    p_in_w: float,
    delta0_rad_s: float,
    a_init_ours_odd: np.ndarray,
    n_roundtrips: int = 12000,
    num_probe: int = 200,
) -> dict:
    """Invoke the unmodified NIST `pyLLE/ComputeLLE.jl` on an exact neutral parameter set."""
    import h5py

    n_modes = len(mu_odd)
    assert n_modes % 2 == 1, "NIST ComputeLLE.jl requires odd mode count [-M..+M]"
    m_half = n_modes // 2

    # Conjugate time-harmonic map keeping D_int(+mu) at +mu: u_NIST(theta) = -i * conj(a_init(-theta))
    a_rev = np.concatenate(([a_init_ours_odd[0]], a_init_ours_odd[1:][::-1]))
    dks_init_nist = -1j * np.conj(a_rev)
    dw_nist = -float(delta0_rad_s)

    with tempfile.TemporaryDirectory(prefix="nist_pylle_") as td:
        tdir = td + os.sep
        h5_in = Path(tdir) / "ParamLLEJulia.h5"
        with h5py.File(h5_in, "w") as f:
            gr = f.create_group("res")
            gr.create_dataset("Qc", data=np.array([float(qc)], dtype=np.float64))
            gr.create_dataset("Qi", data=np.array([float(qi)], dtype=np.float64))
            gr.create_dataset("R", data=np.array([float(r_m)], dtype=np.float64))
            gr.create_dataset("ng", data=np.array([float(ng)], dtype=np.float64))
            gr.create_dataset("gamma", data=np.array([float(gamma_per_w_m)], dtype=np.float64))
            gr.create_dataset("dispfile", data=np.bytes_("neutral_benchmark"))

            gs = f.create_group("sim")
            gs.create_dataset("Pin", data=np.array([float(p_in_w)], dtype=np.float64))
            gs.create_dataset("Tscan", data=np.array([float(n_roundtrips)], dtype=np.float64))
            gs.create_dataset("domega_init", data=np.array([dw_nist], dtype=np.float64))
            gs.create_dataset("domega_end", data=np.array([dw_nist], dtype=np.float64))
            gs.create_dataset("domega_stop", data=np.array([dw_nist], dtype=np.float64))
            gs.create_dataset("domega", data=np.array([dw_nist], dtype=np.float64))
            gs.create_dataset("f_pmp", data=np.array([float(f0_hz)], dtype=np.float64))
            gs.create_dataset("f_center", data=np.array([float(f0_hz)], dtype=np.float64))
            gs.create_dataset("phi_pmp", data=np.array([0.0], dtype=np.float64))
            gs.create_dataset("mu_sim_center", data=np.array([-m_half, m_half], dtype=np.int64))
            gs.create_dataset("ind_pmp", data=np.array([0], dtype=np.int64))
            gs.create_dataset("ind_pump_sweep", data=np.array([0], dtype=np.int64))
            gs.create_dataset("Dint", data=np.asarray(dint_odd_rad_s, dtype=np.float64))
            gs.create_dataset("D1", data=np.array([float(d1_rad_s)], dtype=np.float64))
            gs.create_dataset("DKSinit_real", data=np.real(dks_init_nist).astype(np.float64))
            gs.create_dataset("DKSinit_imag", data=np.imag(dks_init_nist).astype(np.float64))
            gs.create_dataset("num_probe", data=np.array([int(num_probe)], dtype=np.int64))

        t0 = time.perf_counter()
        proc = subprocess.run(
            [julia_exe, str(compute_jl), tdir, "1e-4", "10", "1.0"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        if proc.returncode != 0:
            raise RuntimeError(f"NIST ComputeLLE.jl failed (rc={proc.returncode}):\n{proc.stderr}")

        with h5py.File(Path(tdir) / "ResultsJulia.h5", "r") as f:
            u_probe = f["Results/u_probeReal"][:] + 1j * f["Results/u_probeImag"][:]
            u_final = u_probe[:, -1] if (u_probe.shape[0] == n_modes and u_probe.shape[1] == num_probe) else u_probe[-1, :]

    p_theta = np.abs(u_final) ** 2
    t_rt = (2.0 * math.pi) / d1_rad_s
    om0 = 2.0 * math.pi * f0_hz
    kappa_ext_rt = (om0 / qc) * t_rt

    a_mu_cav = np.fft.fftshift(np.fft.fft(1j * u_final)) / n_modes
    a_mu_out = -math.sqrt(kappa_ext_rt) * a_mu_cav
    idx0 = n_modes // 2
    a_mu_out[idx0] += math.sqrt(p_in_w)
    p_mu_out = np.abs(a_mu_out) ** 2
    p_mu_out_dbm = 10.0 * np.log10(np.maximum(p_mu_out * 1e3, 1e-15))

    p_peak = float(np.max(p_theta))
    p_bg = float(np.percentile(p_theta, 15))
    i_pk = int(np.argmax(p_theta))
    p_cent = np.roll(p_theta, idx0 - i_pk)
    half_lvl = p_bg + 0.5 * (p_peak - p_bg)
    above = np.flatnonzero(p_cent >= half_lvl)
    dth = 2.0 * math.pi / n_modes
    fwhm_fs = 0.0
    if len(above) >= 2:
        il, ir = int(above[0]), int(above[-1])
        xl = float(il - 1) + float((half_lvl - p_cent[il - 1]) / (p_cent[il] - p_cent[il - 1])) if (il > 0 and p_cent[il] > p_cent[il - 1]) else float(il)
        xr = float(ir) + float((p_cent[ir] - half_lvl) / (p_cent[ir] - p_cent[ir + 1])) if (ir + 1 < n_modes and p_cent[ir] > p_cent[ir + 1]) else float(ir)
        fwhm_fs = ((xr - xl) * dth / (2.0 * math.pi)) * t_rt * 1e15

    conv_eff_pct = ((float(np.sum(p_mu_out)) - float(p_mu_out[idx0])) / p_in_w) * 100.0
    return {
        "elapsed_ms": round(elapsed_ms, 1),
        "p_peak_W": p_peak,
        "p_avg_mW": float(np.mean(p_theta)) * 1e3,
        "fwhm_fs": fwhm_fs,
        "conv_eff_pct": conv_eff_pct,
        "comb_dbm": p_mu_out_dbm,
        "mu": mu_odd,
    }


def make_dks_seed(mu_arr: np.ndarray, d2_rad: float, g_nl: float, kappa_tot: float, kappa_ext: float, t_rt: float, p_in_w: float, delta0: float) -> np.ndarray:
    n_modes = len(mu_arr)
    theta = np.linspace(-math.pi, math.pi, n_modes, endpoint=False)
    s_in = math.sqrt(kappa_ext * p_in_w / t_rt)
    p_th = (kappa_tot ** 3) * t_rt / (8.0 * g_nl * kappa_ext)
    f_sq = p_in_w / p_th
    zeta = 2.0 * delta0 / kappa_tot
    zmax = (math.pi ** 2 / 8.0) * f_sq
    theta_s = math.sqrt(max(d2_rad / max(2.0 * delta0, 1e-12), 1e-9))
    a_s = math.sqrt(max(2.0 * delta0 / g_nl, 0.0))
    phi_s = -math.acos(min(max(math.sqrt(max(zeta / zmax, 0.0)), 0.0), 0.995))
    a_cw = s_in / complex(0.5 * kappa_tot, delta0)
    return a_cw + a_s * (1.0 / np.cosh(np.clip(theta / max(theta_s, 1e-4), -40.0, 40.0))) * complex(math.cos(phi_s), math.sin(phi_s))


def compare_comb_spectra(mu_mr: np.ndarray, dbm_mr: np.ndarray, mu_nist: np.ndarray, dbm_nist: np.ndarray, floor_dbm: float = -60.0) -> dict:
    common = np.intersect1d(mu_mr.astype(int), mu_nist.astype(int))
    i_mr = np.searchsorted(mu_mr.astype(int), common)
    i_ni = np.searchsorted(mu_nist.astype(int), common)
    y_mr, y_ni = dbm_mr[i_mr], dbm_nist[i_ni]
    mask = (y_ni >= floor_dbm) & (y_mr >= floor_dbm) & (common != 0)
    diff = y_mr[mask] - y_ni[mask]
    return {
        "n_lines": int(np.count_nonzero(mask)),
        "rmse_dB": round(float(np.sqrt(np.mean(diff ** 2))), 4),
        "median_abs_dB": round(float(np.median(np.abs(diff))), 4),
        "max_abs_dB": round(float(np.max(np.abs(diff))), 4),
        "corr": round(float(np.corrcoef(y_mr[mask], y_ni[mask])[0, 1]), 6),
    }


def main() -> None:
    from microring import LLEConfig, compute_resonator_dispersion_and_nl, simulate_lle_microcomb

    ap = argparse.ArgumentParser(description="Validate MicroRing Studio LLE against NIST usnistgov/pyLLE (ComputeLLE.jl)")
    ap.add_argument("--pylle-dir", required=True, help="Path to cloned usnistgov/pyLLE repository")
    ap.add_argument("--julia", default=None, help="Optional path to julia executable")
    ap.add_argument("--out-json", default=str(REPO_ROOT / "benchmarks" / "outputs" / "nist_pylle_validation.json"))
    args = ap.parse_args()

    pylle_dir = Path(args.pylle_dir)
    compute_jl = pylle_dir / "pyLLE" / "ComputeLLE.jl"
    nist_csv = pylle_dir / "example" / "RW1000_H430.csv"
    julia_exe = find_julia_exe(args.julia)
    results = {"nist_solver": "usnistgov/pyLLE/pyLLE/ComputeLLE.jl", "cases": []}

    for name, cfg in (
        (
            "Brasch_2016_Si3N4_800x1600",
            LLEConfig(
                core_material="Si3N4", film_height_um=0.80, ring_width_um=1.60, ring_radius_um=23.0,
                q_int_override=1.5e6, q_ext_override=1.5e6, pump_power_mw=80.0, detuning_end_kappa=17.0,
                mode="single_soliton", include_coupler_dispersion=False, num_comb_modes=256, num_steps=3000,
            ),
        ),
        (
            "Guidry_2020_4HSiC_500x1850",
            LLEConfig(
                core_material="4H-SiC", film_height_um=0.50, ring_width_um=1.85, ring_radius_um=55.0,
                q_int_override=1.1e6, q_ext_override=1.1e6, pump_power_mw=20.0, detuning_end_kappa=6.5,
                mode="single_soliton", include_coupler_dispersion=False, num_comb_modes=256, num_steps=3000,
            ),
        ),
    ):
        disp = compute_resonator_dispersion_and_nl(cfg)
        r_mr = simulate_lle_microcomb(cfg, disp_cache=disp)
        inn = disp["_internal"]
        mu_f = np.asarray(disp["arrays"]["mu"], dtype=int)
        dint_f = np.asarray(disp["arrays"]["D_int_rad_per_s"], dtype=float)
        mask_o = (mu_f >= -127) & (mu_f <= 127)
        mu_o, dint_o = mu_f[mask_o], dint_f[mask_o]
        d0 = 0.5 * inn["kappa_tot0"] * float(r_mr["metrics"]["final_detuning_kappa"])
        seed = make_dks_seed(mu_o, inn["d2_rad"], inn["g_nl_rad_per_s_w"], inn["kappa_tot0"], inn["kappa_ext0"], inn["t_rt_s"], cfg.pump_power_mw * 1e-3, d0)
        n_rt = int(round(30.0 / (inn["kappa_tot0"] * inn["t_rt_s"])))
        r_ni = run_nist_compute_lle_jl(
            julia_exe=julia_exe, compute_jl=compute_jl, r_m=cfg.ring_radius_um * 1e-6, ng=inn["ng0"],
            qi=float(cfg.q_int_override or 1e6), qc=float(cfg.q_ext_override or 1e6),
            gamma_per_w_m=disp["metrics"]["gamma_NL_per_W_m"], f0_hz=C0 / (cfg.wavelength_nm * 1e-9),
            d1_rad_s=inn["d1_rad"], mu_odd=mu_o, dint_odd_rad_s=dint_o,
            p_in_w=cfg.pump_power_mw * 1e-3, delta0_rad_s=d0, a_init_ours_odd=seed, n_roundtrips=n_rt,
        )
        sp = compare_comb_spectra(np.asarray(r_mr["spectrum"]["mu"]), np.asarray(r_mr["spectrum"]["comb_power_dBm"]), r_ni["mu"], r_ni["comb_dbm"])
        row = {
            "name": name,
            "mr_peak_W": r_mr["metrics"]["soliton_peak_power_W"],
            "nist_peak_W": round(r_ni["p_peak_W"], 4),
            "peak_diff_pct": round(abs(r_mr["metrics"]["soliton_peak_power_W"] - r_ni["p_peak_W"]) / r_ni["p_peak_W"] * 100.0, 3),
            "mr_fwhm_fs": r_mr["metrics"]["pulse_fwhm_fs"],
            "nist_fwhm_fs": round(r_ni["fwhm_fs"], 2),
            "fwhm_diff_pct": round(abs(r_mr["metrics"]["pulse_fwhm_fs"] - r_ni["fwhm_fs"]) / r_ni["fwhm_fs"] * 100.0, 3),
            "mr_pavg_mW": r_mr["metrics"]["intracavity_avg_power_mW"],
            "nist_pavg_mW": round(r_ni["p_avg_mW"], 3),
            "pavg_diff_pct": round(abs(r_mr["metrics"]["intracavity_avg_power_mW"] - r_ni["p_avg_mW"]) / r_ni["p_avg_mW"] * 100.0, 3),
            "spectrum_parity": sp,
            "mr_ms": r_mr["elapsed_ms"],
            "nist_julia_ms": r_ni["elapsed_ms"],
        }
        print(json.dumps(row, indent=2), flush=True)
        results["cases"].append(row)

    if nist_csv.is_file():
        data = np.loadtxt(nist_csv, delimiter=",")
        m_c, f_c = data[:, 0], data[:, 1]
        idx_p = int(np.argmin(np.abs(f_c - 283e12)))
        f0 = float(f_c[idx_p])
        dm5 = np.array([-2, -1, 0, 1, 2], dtype=float)
        pfit = np.polyfit(dm5, f_c[idx_p + dm5.astype(int)] - f0, 2)
        fsr_hz = float(pfit[1])
        d1_r = 2.0 * math.pi * fsr_hz
        d2_r = 2.0 * math.pi * float(2.0 * pfit[0])
        ng = C0 / (fsr_hz * 2.0 * math.pi * 23e-6)
        mu_rel = m_c - m_c[idx_p]
        dint_c = 2.0 * math.pi * ((f_c - f0) - fsr_hz * mu_rel)
        mu_o3 = np.arange(-190, 191, dtype=int)
        dint_o3 = np.interp(mu_o3, mu_rel, dint_c)
        n3 = 384
        mu_m3 = np.arange(-(n3 // 2), n3 // 2, dtype=float)
        dint_m3 = np.interp(mu_m3, mu_rel, dint_c)
        om0 = 2.0 * math.pi * f0
        t_rt = (2.0 * math.pi) / d1_r
        ki = ke = om0 / 1e6
        kt = ki + ke
        g_nl = 3.2 * (C0 / ng)
        p_th = (kt ** 3) * t_rt / (8.0 * g_nl * ke)
        d0_3 = 2.0 * math.pi * 4.2e9
        disp_cache = {
            "metrics": {"FSR_GHz": fsr_hz * 1e-9, "D2_2pi_MHz": d2_r / (2.0 * math.pi * 1e6), "D2_2pi_kHz": d2_r / (2.0 * math.pi * 1e3), "P_th_OPO_mW": round(p_th * 1e3, 4), "gamma_NL_per_W_m": 3.2},
            "arrays": {"mu": mu_m3, "D_int_rad_per_s": dint_m3, "D_int_2pi_GHz": dint_m3 / (2.0 * math.pi * 1e9), "kappa_tot_mu_rad_per_s": np.full(n3, kt), "kappa_ext_mu_rad_per_s": np.full(n3, ke), "wavelength_comb_nm": (C0 / np.maximum(f0 + mu_m3 * fsr_hz, 1e12)) * 1e9},
            "_internal": {"om0": om0, "d1_rad": d1_r, "d2_rad": d2_r, "t_rt_s": t_rt, "l_rt_m": 2.0 * math.pi * 23e-6, "ng0": ng, "kappa_int": ki, "kappa_ext0": ke, "kappa_tot0": kt, "g_nl_rad_per_s_w": g_nl, "p_th_w": p_th, "a_eff_nl_m2": 0.45e-12, "dn_dt_modal": 2.4e-5},
        }
        r3_mr = simulate_lle_microcomb(LLEConfig(core_material="Si3N4", top_clad_material="Air", film_height_um=0.43, ring_width_um=1.00, ring_radius_um=23.0, wavelength_nm=(C0 / f0) * 1e9, q_int_override=1e6, q_ext_override=1e6, pump_power_mw=140.0, detuning_end_kappa=2.0 * d0_3 / kt, mode="single_soliton", include_coupler_dispersion=False, num_comb_modes=n3, num_steps=4000), disp_cache=disp_cache)
        seed3 = make_dks_seed(mu_o3, d2_r, g_nl, kt, ke, t_rt, 0.140, d0_3)
        r3_ni = run_nist_compute_lle_jl(julia_exe=julia_exe, compute_jl=compute_jl, r_m=23e-6, ng=ng, qi=1e6, qc=1e6, gamma_per_w_m=3.2, f0_hz=f0, d1_rad_s=d1_r, mu_odd=mu_o3, dint_odd_rad_s=dint_o3, p_in_w=0.140, delta0_rad_s=d0_3, a_init_ours_odd=seed3, n_roundtrips=int(round(30.0 / (kt * t_rt))))
        sp3 = compare_comb_spectra(np.asarray(r3_mr["spectrum"]["mu"]), np.asarray(r3_mr["spectrum"]["comb_power_dBm"]), r3_ni["mu"], r3_ni["comb_dbm"])
        row3 = {
            "name": "NIST_Official_RW1000_H430_Octave_DualDW",
            "mr_peak_W": r3_mr["metrics"]["soliton_peak_power_W"],
            "nist_peak_W": round(r3_ni["p_peak_W"], 4),
            "peak_diff_pct": round(abs(r3_mr["metrics"]["soliton_peak_power_W"] - r3_ni["p_peak_W"]) / r3_ni["p_peak_W"] * 100.0, 3),
            "mr_fwhm_fs": r3_mr["metrics"]["pulse_fwhm_fs"],
            "nist_fwhm_fs": round(r3_ni["fwhm_fs"], 2),
            "fwhm_diff_pct": round(abs(r3_mr["metrics"]["pulse_fwhm_fs"] - r3_ni["fwhm_fs"]) / r3_ni["fwhm_fs"] * 100.0, 3),
            "mr_pavg_mW": r3_mr["metrics"]["intracavity_avg_power_mW"],
            "nist_pavg_mW": round(r3_ni["p_avg_mW"], 3),
            "pavg_diff_pct": round(abs(r3_mr["metrics"]["intracavity_avg_power_mW"] - r3_ni["p_avg_mW"]) / r3_ni["p_avg_mW"] * 100.0, 3),
            "spectrum_parity": sp3,
            "mr_ms": r3_mr["elapsed_ms"],
            "nist_julia_ms": r3_ni["elapsed_ms"],
        }
        print(json.dumps(row3, indent=2), flush=True)
        results["cases"].append(row3)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
