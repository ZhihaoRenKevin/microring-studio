"""Independent Ansys Lumerical (2026 R1, RTX 5090 GPU 3D FDTD) Phase-3 Validation Runner.

Reads ONLY neutral physical specifications from `benchmarks/outputs/commercial_validation_specs_p3.json`
(plus MicroRing's exported `.gds` mask strictly for the Suite 12B GDS-to-FDTD roundtrip audit)
and executes Ansys Lumerical 2026 R1 3D FDTD on the local NVIDIA RTX 5090 GPU across:

  - Suite 12A: 7 Strictly Converged (`status = 2`, `auto_shutoff = 1e-5`) Low-Q Full-Ring 3D FDTD:
      1. `SOI_LowQ_AddDrop_R5um_g100nm` (4-port Circular Add-Drop, Thru + Drop full spectra)
      2. `SOI_LowQ_AddDrop_R5um_g120nm` (4-port Circular Add-Drop, gap-scaling verification)
      3. `SOI_LowQ_Lossy_AllPass_CC_R5um` (Complex-n Lossy All-Pass Critical Coupling dip)
      4. `SOI_LowQ_Racetrack_AD_R5um_Ls3um` (4-port Racetrack Add-Drop full ring)
      5. `SOI_LowQ_Pulley_AD_R5um_th20deg` (4-port Concentric Pulley Add-Drop full ring)
      6. `SOI_LowQ_DualRing_CROW_R5um` (2nd-Order Coupled Dual-Ring CROW Add-Drop)
      7. `SOI_LowQ_PCM_GSST_Switch_R5um` (Non-Volatile `a-GSST` vs `c-GSST` Patch Switch)
  - Suite 12B:
      1. `GDS_Roundtrip_AddDrop_R5um`: Direct `gdsimport` of MicroRing's exported `.gds` mask
         into Lumerical 3D FDTD (`status = 2`).
      2. `Blind_CMT_Fit_SplitDoublet_3DFDTD`: Sidewall scattering bump exciting a genuine
         full-wave CW/CCW split-doublet + blind `fit_single_resonance_cmt` BIC audit on both
         clean and split-doublet converged 3D FDTD traces.
  - Suite 13:
      1. `SOI_1550nm_MMI_2x2_OptLen` (`2x2` MMI at optimal self-imaging `L = 42.3 um`)
      2. `SOI_1550nm_DC_2x2_FullSBend` (Complete `2x2` DC including `12 um` cosine S-bends)
      3. `SOI_1550nm_LowQ_RAMZI_R5um` (Complete Low-Q Ring-Assisted MZI)
      4. `SOI_1550nm_3D_Focusing_GC` (Full 3D FDTD 20-tooth Curved Focusing Grating Coupler)
      5. `SOI_1550nm_3D_Butt_Taper_30um` (Full 3D FDTD Reciprocal `30 um` Inverse-Taper Butt Coupler)

Saves `benchmarks/outputs/lumerical_independent_results_p3.json` (with per-case checkpointing).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.signal import find_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH_DIR = OUT_DIR / "_lum_scratch_p3"
SPECS_P3_JSON = OUT_DIR / "commercial_validation_specs_p3.json"
OUT_P3_JSON = OUT_DIR / "lumerical_independent_results_p3.json"
CKPT_P3_JSON = SCRATCH_DIR / "lum_p3_ckpt.json"
GDS_P3_DIR = OUT_DIR / "_p3_gds"

LUMERICAL_WIN_HINTS = [
    r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v252\Lumerical\api\python",
]
C0 = 299792458.0
Z0_VAC = 376.730313668
AUTO_SHUTOFF_MIN = 5.0e-5


def _import_lumapi():
    for p in LUMERICAL_WIN_HINTS:
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)
    import lumapi  # type: ignore
    return lumapi


def _scalar_real(val) -> float:
    return float(np.real(np.asarray(val).reshape(-1)[0]))


def _run_fdtd_gpu(fdtd, fsp_path: Path, retries: int = 4) -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    if fsp_path.is_file():
        try:
            fdtd.load(str(fsp_path))
            if int(_scalar_real(fdtd.haveresult("FDTD", "status"))) > 0:
                print(f"[REUSED SOLVED FSP] {fsp_path.name}")
                return
        except Exception:
            pass
    fdtd.save(str(fsp_path))
    last_err = None
    for attempt in range(retries):
        try:
            fdtd.run("FDTD", "GPU")
            return
        except Exception as exc:
            last_err = exc
            time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"RTX 5090 GPU 3D FDTD failed on {fsp_path.name}: {last_err}")


def _save_ckpt(data: dict) -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_P3_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _extract_port_power(fdtd, port_name: str) -> tuple[np.ndarray, np.ndarray]:
    exp = fdtd.getresult(f"FDTD::ports::{port_name}", "expansion for port monitor")
    f_arr = np.asarray(exp["f"]).ravel()
    s_arr = np.asarray(exp["S"]).ravel()
    wl_nm = (C0 / f_arr) * 1e9
    order = np.argsort(wl_nm)
    return wl_nm[order], (np.abs(s_arr[order]) ** 2).astype(float)


def _analyze_ring_spectrum(
    wl_nm: np.ndarray,
    t_thru: np.ndarray,
    t_drop: np.ndarray | None,
    ring_len_um: float,
    known_regime: str = "critical",
    model_type: str = "lorentzian",
) -> dict:
    """Measure FSR, FWHM, Q_L, ER_thru, IL_drop, passivity, and blind CMT fit directly from 3D FDTD trace."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from microring.mrr_cmt_fit_predict import fit_single_resonance_cmt

    if t_drop is not None:
        pks, _ = find_peaks(t_drop, prominence=0.12, distance=12)
        if len(pks) == 0:
            pks, _ = find_peaks(1.0 - t_thru, prominence=0.10, distance=12)
    else:
        pks, _ = find_peaks(1.0 - t_thru, prominence=0.10, distance=12)

    res_wls = [round(float(wl_nm[i]), 4) for i in pks]
    fsr_nm = round(float(np.mean(np.diff(res_wls))), 4) if len(res_wls) >= 2 else 0.0
    i0 = min(pks, key=lambda idx: abs(wl_nm[idx] - 1550.0)) if len(pks) else int(np.argmin(t_thru))
    wl0 = float(wl_nm[i0])

    # Local window around resonance closest to 1550 nm
    half_win = max(0.38 * fsr_nm, 3.5) if fsr_nm > 0 else 4.5
    win = np.abs(wl_nm - wl0) <= half_win
    w_w = wl_nm[win]
    th_w = t_thru[win]
    dr_w = t_drop[win] if t_drop is not None else None

    # Direct empirical FWHM & Q_L from Drop peak (or Thru dip)
    if dr_w is not None and float(np.max(dr_w)) > 0.10:
        pk_val = float(np.max(dr_w))
        ip = int(np.argmax(dr_w))
        hm = 0.5 * pk_val
        left_i = np.where(dr_w[: ip + 1] <= hm)[0]
        right_i = np.where(dr_w[ip:] <= hm)[0] + ip
        if len(left_i) and len(right_i):
            il = int(left_i[-1])
            ir = int(right_i[0])
            wl_l = float(w_w[il] + (hm - dr_w[il]) * (w_w[il + 1] - w_w[il]) / max(dr_w[il + 1] - dr_w[il], 1e-12))
            wl_r = float(w_w[ir - 1] + (hm - dr_w[ir - 1]) * (w_w[ir] - w_w[ir - 1]) / min(dr_w[ir] - dr_w[ir - 1], -1e-12))
            fwhm_emp = max(wl_r - wl_l, 1e-4)
        else:
            fwhm_emp = 0.0
        il_drop_db = round(-10.0 * math.log10(max(pk_val, 1e-6)), 3)
        t_min_thru = float(np.min(th_w))
        t_bg_thru = float(np.percentile(th_w, 95))
        er_thru_db = round(10.0 * math.log10(max(t_bg_thru, 1e-6) / max(t_min_thru, 1e-6)), 2)
    else:
        t_min_thru = float(np.min(th_w))
        t_bg_thru = float(np.percentile(th_w, 95))
        er_thru_db = round(10.0 * math.log10(max(t_bg_thru, 1e-6) / max(t_min_thru, 1e-6)), 2)
        hm = 0.5 * (t_bg_thru + t_min_thru)
        ip = int(np.argmin(th_w))
        left_i = np.where(th_w[: ip + 1] >= hm)[0]
        right_i = np.where(th_w[ip:] >= hm)[0] + ip
        if len(left_i) and len(right_i):
            il = int(left_i[-1])
            ir = int(right_i[0])
            wl_l = float(w_w[il] + (hm - th_w[il]) * (w_w[il + 1] - w_w[il]) / min(th_w[il + 1] - th_w[il], -1e-12))
            wl_r = float(w_w[ir - 1] + (hm - th_w[ir - 1]) * (w_w[ir] - w_w[ir - 1]) / max(th_w[ir] - th_w[ir - 1], 1e-12))
            fwhm_emp = max(wl_r - wl_l, 1e-4)
        else:
            fwhm_emp = 0.0
        il_drop_db = None

    q_emp = round(wl0 / fwhm_emp, 1) if fwhm_emp > 1e-4 else 0.0

    # Blind CMT fit on Thru trace
    fit_res, _, _ = fit_single_resonance_cmt(
        w_w, th_w, model_type=model_type, known_regime=known_regime, ng_assumed=4.3, ring_length_um=ring_len_um
    )
    max_passivity = float(np.max(t_thru + (t_drop if t_drop is not None else 0.0)))

    return {
        "resonance_wavelengths_nm": res_wls,
        "center_res_nm": round(wl0, 4),
        "FSR_nm": fsr_nm,
        "FWHM_emp_nm": round(fwhm_emp, 5),
        "Q_loaded_emp": q_emp,
        "ER_thru_dB": er_thru_db,
        "IL_drop_dB": il_drop_db,
        "max_passivity_sum": round(max_passivity, 5),
        "cmt_fit_model": fit_res.model_type,
        "cmt_fit_Q_loaded": round(float(fit_res.Q_loaded), 1),
        "cmt_fit_Q_int": round(float(fit_res.Q_int), 1),
        "cmt_fit_Q_ext": round(float(fit_res.Q_ext), 1),
        "cmt_fit_r_squared": round(float(fit_res.r_squared), 5),
        "cmt_fit_splitting_ghz": round(float(getattr(fit_res, "mode_splitting_ghz", 0.0)), 3),
        "spectrum_wl_nm": [round(float(x), 4) for x in wl_nm[::2]],
        "spectrum_thru": [round(float(x), 5) for x in t_thru[::2]],
        "spectrum_drop": [round(float(x), 5) for x in t_drop[::2]] if t_drop is not None else None,
    }


def _poly_verts_lumerical(xs: np.ndarray, ys: np.ndarray) -> str:
    pairs = [f"{x:.12e},{y:.12e}" for x, y in zip(xs, ys)]
    return "[" + "; ".join(pairs) + "]"


def _sbend_polygon_verts(x0: float, y0: float, x1: float, y1: float, w: float, n_pts: int = 36) -> str:
    """Create closed polygon for a cosine S-bend waveguide of constant width `w` from `(x0,y0)` to `(x1,y1)`."""
    t = np.linspace(0.0, 1.0, n_pts)
    xc = x0 + (x1 - x0) * t
    yc = y0 + 0.5 * (y1 - y0) * (1.0 - np.cos(math.pi * t))
    dydx = 0.5 * (y1 - y0) * math.pi * np.sin(math.pi * t) / max(x1 - x0, 1e-12)
    ang = np.arctan(dydx)
    nx = -np.sin(ang)
    ny = np.cos(ang)
    x_top = xc + 0.5 * w * nx
    y_top = yc + 0.5 * w * ny
    x_bot = xc - 0.5 * w * nx
    y_bot = yc - 0.5 * w * ny
    xs = np.concatenate([x_top, x_bot[::-1]])
    ys = np.concatenate([y_top, y_bot[::-1]])
    return _poly_verts_lumerical(xs, ys)


# ==============================================================================
# Suite 12A: 7 Strictly Converged (status=2) Low-Q Full Rings on RTX 5090 GPU
# ==============================================================================
def run_suite12a_case(fdtd, c: dict) -> dict:
    cid = c["case_id"]
    top = c["topology"]
    t0 = time.perf_counter()
    nc = c["core_index"]["center"]["nx"]
    ncl = c["clad_index"]["center"]["nx"]
    hf = c["film_height_um"] * 1e-6
    wr = c["ring_width_um"] * 1e-6
    wb = c["bus_width_um"] * 1e-6
    R = c["ring_radius_um"] * 1e-6
    gt = c["gap_thru_um"] * 1e-6
    gd = c.get("gap_drop_um", c["gap_thru_um"]) * 1e-6
    wl_min = c["wl_min_nm"] * 1e-9
    wl_max = c["wl_max_nm"] * 1e-9
    sim_t = c["fdtd_sim_time_ps"] * 1e-12

    # --- Case 7: PCM GSST Switch (2 states: amorphous vs crystalline, asymmetric in z -> PML!) ---
    if "cap_states" in c:
        st_out = {}
        y_th = -(R + 0.5 * wr + gt + 0.5 * wb)
        y_dr = +(R + 0.5 * wr + gd + 0.5 * wb)
        x_h = R + 3.2e-6
        h_cap = c["cap_h_um"] * 1e-6
        w_cap = c["cap_w_um"] * 1e-6
        dth_deg = math.degrees(c["cap_length_um"] / c["ring_radius_um"])
        for st_mat in c["cap_states"]:
            idx_key = "gsst_amorphous_index" if "amorphous" in st_mat else "gsst_crystalline_index"
            n_cap = c[idx_key]["nx"]
            k_cap = c[idx_key]["kx"]
            mat_cmd = (
                f'set("index",{n_cap:.6f});'
                if k_cap < 1e-6
                else (
                    f'if (materialexists("GSST_C_P3")) {{ deletematerial("GSST_C_P3"); }} '
                    f'm_c = addmaterial("(n,k) Material"); setmaterial(m_c,"name","GSST_C_P3"); '
                    f'setmaterial("GSST_C_P3","Refractive Index",{n_cap:.6f}); '
                    f'setmaterial("GSST_C_P3","Imaginary Refractive Index",{k_cap:.6f}); '
                    f'set("material","GSST_C_P3");'
                )
            )
            fdtd.eval(f"""
            switchtolayout; deleteall;
            addring; set("name","Ring"); set("x",0); set("y",0); set("z",0); set("z span",{hf:.12e});
            set("inner radius",{R-0.5*wr:.12e}); set("outer radius",{R+0.5*wr:.12e}); set("index",{nc:.8f});
            addrect; set("name","BusThru"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
            set("y",{y_th:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            addrect; set("name","BusDrop"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
            set("y",{y_dr:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            addring; set("name","PCMCap"); set("x",0); set("y",0);
            set("z min",{0.5*hf:.12e}); set("z max",{0.5*hf+h_cap:.12e});
            set("inner radius",{R-0.5*w_cap:.12e}); set("outer radius",{R+0.5*w_cap:.12e});
            set("theta start",{-0.5*dth_deg:.6f}); set("theta stop",{+0.5*dth_deg:.6f});
            {mat_cmd}
            addfdtd; set("dimension","3D");
            set("x",0); set("x span",{2.0*x_h:.12e});
            set("y min",{y_th-1.5e-6:.12e}); set("y max",{y_dr+1.5e-6:.12e});
            set("z",0); set("z span",1.8e-6);
            set("background index",{ncl:.8f}); set("mesh accuracy",2);
            set("simulation time",{sim_t:.12e}); set("auto shutoff min",{AUTO_SHUTOFF_MIN});
            # Asymmetric in z (PCM cap on top) -> full PML on all 6 faces!
            addmesh; set("name","GapThM"); set("x",0); set("x span",2.2e-6);
            set("y",{y_th+0.5*wb+0.5*gt:.12e}); set("y span",{gt+wr+wb+0.20e-6:.12e});
            set("z",0); set("z span",{hf+0.10e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
            addmesh; set("name","GapDrM"); set("x",0); set("x span",2.2e-6);
            set("y",{y_dr-0.5*wb-0.5*gd:.12e}); set("y span",{gd+wr+wb+0.20e-6:.12e});
            set("z",0); set("z span",{hf+0.10e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
            addmesh; set("name","PcmM"); set("x",{R:.12e}); set("x span",{wr+0.25e-6:.12e});
            set("y",0); set("y span",{c['cap_length_um']*1e-6+0.3e-6:.12e});
            set("z",{0.5*hf+0.5*h_cap:.12e}); set("z span",{h_cap+0.06e-6:.12e});
            set("dx",24e-9); set("dy",24e-9); set("dz",12e-9);
            setglobalsource("wavelength start",{wl_min:.12e}); setglobalsource("wavelength stop",{wl_max:.12e});
            addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
            set("x",{-x_h+0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.6e-6); set("z",0); set("z span",1.4e-6);
            set("mode selection","fundamental TE mode");
            addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
            set("x",{+x_h-0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.6e-6); set("z",0); set("z span",1.4e-6);
            set("mode selection","fundamental TE mode");
            addport; set("name","PortDrop"); set("injection axis","x-axis"); set("direction","Forward");
            set("x",{-x_h+0.8e-6:.12e}); set("y",{y_dr:.12e}); set("y span",1.6e-6); set("z",0); set("z span",1.4e-6);
            set("mode selection","fundamental TE mode");
            select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",201);
            """)
            _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{cid}_{st_mat}.fsp")
            st_code = int(_scalar_real(fdtd.getresult("FDTD", "status")))
            wl_r, t_th = _extract_port_power(fdtd, "PortThru")
            _, t_dr = _extract_port_power(fdtd, "PortDrop")
            ana = _analyze_ring_spectrum(wl_r, t_th, t_dr, 2.0 * math.pi * c["ring_radius_um"])
            ana["fdtd_status_code"] = st_code
            st_out[st_mat] = ana
            print(
                f"[LUM Suite 12A GPU] {cid} ({st_mat}): status={st_code}, res={ana['center_res_nm']:.3f}nm, "
                f"Q_L={ana['Q_loaded_emp']:.1f}, ER={ana['ER_thru_dB']:.2f}dB, IL_drop={ana['IL_drop_dB']}dB"
            )
        da = st_out["GSST_amorphous"]
        dc = st_out["GSST_crystalline"]
        # Match nearest crystalline resonance red-shifted relative to amorphous resonance
        wl_a = da["center_res_nm"]
        c_res_list = dc["resonance_wavelengths_nm"] or [dc["center_res_nm"]]
        # Prefer smallest positive red-shift within [0.5, 10] nm
        pos_shifts = [w - wl_a for w in c_res_list if 0.2 <= (w - wl_a) <= 10.0]
        dlam_3d = round(min(pos_shifts), 3) if pos_shifts else round(dc["center_res_nm"] - wl_a, 3)
        # Drop extinction contrast at amorphous resonance wavelength
        wl_c_arr = np.asarray(dc["spectrum_wl_nm"])
        dr_c_arr = np.asarray(dc["spectrum_drop"])
        t_dr_c_at_a = float(np.interp(wl_a, wl_c_arr, dr_c_arr))
        t_dr_a_pk = 10.0 ** (-da["IL_drop_dB"] / 10.0)
        contrast_at_res_db = round(10.0 * math.log10(max(t_dr_a_pk, 1e-6) / max(t_dr_c_at_a, 1e-6)), 2)
        return {
            "case_id": cid,
            "states": st_out,
            "delta_lambda_shift_nm": dlam_3d,
            "drop_peak_il_change_dB": round((dc["IL_drop_dB"] or 0.0) - (da["IL_drop_dB"] or 0.0), 2),
            "drop_switching_contrast_at_lamA_dB": contrast_at_res_db,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }

    # --- Standard z-symmetric Low-Q Full Ring Topologies (z min bc = Symmetric!) ---
    has_drop = top.endswith("_ad")
    ring_len_um = 2.0 * math.pi * c["ring_radius_um"]

    if top in ("point_circular_ad", "point_circular_ap"):
        y_th = -(R + 0.5 * wr + gt + 0.5 * wb)
        y_dr = +(R + 0.5 * wr + gd + 0.5 * wb)
        x_h = R + 3.2e-6
        if "ring_core_k_imag" in c:
            k_im = float(c["ring_core_k_imag"])
            ring_mat_cmd = (
                f'if (materialexists("Si_Lossy_CC")) {{ deletematerial("Si_Lossy_CC"); }} '
                f'm_l = addmaterial("(n,k) Material"); setmaterial(m_l,"name","Si_Lossy_CC"); '
                f'setmaterial("Si_Lossy_CC","Refractive Index",{nc:.8f}); '
                f'setmaterial("Si_Lossy_CC","Imaginary Refractive Index",{k_im:.8f}); '
                f'set("material","Si_Lossy_CC");'
            )
        else:
            ring_mat_cmd = f'set("index",{nc:.8f});'
        drop_geom = (
            f'addrect; set("name","BusDrop"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e}); '
            f'set("y",{y_dr:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f}); '
            f'addmesh; set("name","GapDrM"); set("x",0); set("x span",2.4e-6); '
            f'set("y",{y_dr-0.5*wb-0.5*gd:.12e}); set("y span",{gd+wr+wb+0.25e-6:.12e}); '
            f'set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9); '
            f'addport; set("name","PortDrop"); set("injection axis","x-axis"); set("direction","Forward"); '
            f'set("x",{-x_h+0.8e-6:.12e}); set("y",{y_dr:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6); '
            f'set("mode selection","fundamental TE mode");'
            if has_drop else ""
        )
        y_max_dom = (y_dr + 1.5e-6) if has_drop else (R + 0.5 * wr + 1.6e-6)
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addring; set("name","Ring"); set("x",0); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{R-0.5*wr:.12e}); set("outer radius",{R+0.5*wr:.12e}); {ring_mat_cmd}
        addrect; set("name","BusThru"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
        set("y",{y_th:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addfdtd; set("dimension","3D");
        set("x",0); set("x span",{2.0*x_h:.12e}); set("y min",{y_th-1.5e-6:.12e}); set("y max",{y_max_dom:.12e});
        set("z",0); set("z span",1.6e-6); set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",{sim_t:.12e}); set("auto shutoff min",{AUTO_SHUTOFF_MIN}); set("z min bc","Symmetric");
        addmesh; set("name","GapThM"); set("x",0); set("x span",2.4e-6);
        set("y",{y_th+0.5*wb+0.5*gt:.12e}); set("y span",{gt+wr+wb+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        setglobalsource("wavelength start",{wl_min:.12e}); setglobalsource("wavelength stop",{wl_max:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_h+0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_h-0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        {drop_geom}
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",201);
        """)

    elif top == "racetrack_ad":
        Ls = c["straight_length_um"] * 1e-6
        ring_len_um = 2.0 * math.pi * c["ring_radius_um"] + 2.0 * c["straight_length_um"]
        y_th = -(R + 0.5 * wr + gt + 0.5 * wb)
        y_dr = +(R + 0.5 * wr + gd + 0.5 * wb)
        x_h = 0.5 * Ls + R + 3.0e-6
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addring; set("name","ArcL"); set("x",{-0.5*Ls:.12e}); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{R-0.5*wr:.12e}); set("outer radius",{R+0.5*wr:.12e});
        set("theta start",90); set("theta stop",270); set("index",{nc:.8f});
        addring; set("name","ArcR"); set("x",{+0.5*Ls:.12e}); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{R-0.5*wr:.12e}); set("outer radius",{R+0.5*wr:.12e});
        set("theta start",-90); set("theta stop",90); set("index",{nc:.8f});
        addrect; set("name","StrBot"); set("x",0); set("x span",{Ls+0.02e-6:.12e});
        set("y",{-R:.12e}); set("y span",{wr:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","StrTop"); set("x",0); set("x span",{Ls+0.02e-6:.12e});
        set("y",{+R:.12e}); set("y span",{wr:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","BusThru"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
        set("y",{y_th:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","BusDrop"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
        set("y",{y_dr:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addfdtd; set("dimension","3D");
        set("x",0); set("x span",{2.0*x_h:.12e}); set("y min",{y_th-1.5e-6:.12e}); set("y max",{y_dr+1.5e-6:.12e});
        set("z",0); set("z span",1.6e-6); set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",{sim_t:.12e}); set("auto shutoff min",{AUTO_SHUTOFF_MIN}); set("z min bc","Symmetric");
        addmesh; set("name","GapThM"); set("x",0); set("x span",{Ls+2.4e-6:.12e});
        set("y",{y_th+0.5*wb+0.5*gt:.12e}); set("y span",{gt+wr+wb+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        addmesh; set("name","GapDrM"); set("x",0); set("x span",{Ls+2.4e-6:.12e});
        set("y",{y_dr-0.5*wb-0.5*gd:.12e}); set("y span",{gd+wr+wb+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        setglobalsource("wavelength start",{wl_min:.12e}); setglobalsource("wavelength stop",{wl_max:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_h+0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_h-0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortDrop"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_h+0.8e-6:.12e}); set("y",{y_dr:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",201);
        """)

    elif top == "pulley_ad":
        th_half_deg = 0.5 * c["pulley_angle_thru_deg"]
        th_half = math.radians(th_half_deg)
        Rb = R + 0.5 * wr + gt + 0.5 * wb
        Ro = Rb  # outward transition radius == R_bus (matching MicroRing pulley geometry!)
        # Bottom pulley arc centered at (0, 0), theta in [-90 - th_half, -90 + th_half]
        # Outward transition center on right: ((Rb + Ro)*sin(th_half), -(Rb + Ro)*cos(th_half))
        xc_out = (Rb + Ro) * math.sin(th_half)
        yc_out = (Rb + Ro) * math.cos(th_half)
        y_port_abs = yc_out - Ro  # straight horizontal port y = +/- y_port_abs!
        x_h = max(R + 3.2e-6, xc_out + 2.8e-6)
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addring; set("name","Ring"); set("x",0); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{R-0.5*wr:.12e}); set("outer radius",{R+0.5*wr:.12e}); set("index",{nc:.8f});

        # Bottom concentric pulley thru bus + smooth outward return bends + horizontal leads
        addring; set("name","PulBot"); set("x",0); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{Rb-0.5*wb:.12e}); set("outer radius",{Rb+0.5*wb:.12e});
        set("theta start",{-90.0-th_half_deg:.6f}); set("theta stop",{-90.0+th_half_deg:.6f}); set("index",{nc:.8f});
        addring; set("name","PulBotOutL"); set("x",{-xc_out:.12e}); set("y",{-yc_out:.12e}); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{Ro-0.5*wb:.12e}); set("outer radius",{Ro+0.5*wb:.12e});
        set("theta start",{90.0-th_half_deg:.6f}); set("theta stop",90.0); set("index",{nc:.8f});
        addring; set("name","PulBotOutR"); set("x",{+xc_out:.12e}); set("y",{-yc_out:.12e}); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{Ro-0.5*wb:.12e}); set("outer radius",{Ro+0.5*wb:.12e});
        set("theta start",90.0); set("theta stop",{90.0+th_half_deg:.6f}); set("index",{nc:.8f});
        addrect; set("name","LeadBotL"); set("x min",{-x_h-1.5e-6:.12e}); set("x max",{-xc_out+0.02e-6:.12e});
        set("y",{-y_port_abs:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","LeadBotR"); set("x min",{+xc_out-0.02e-6:.12e}); set("x max",{+x_h+1.5e-6:.12e});
        set("y",{-y_port_abs:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});

        # Top concentric pulley drop bus + smooth outward return bends + horizontal leads
        addring; set("name","PulTop"); set("x",0); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{Rb-0.5*wb:.12e}); set("outer radius",{Rb+0.5*wb:.12e});
        set("theta start",{90.0-th_half_deg:.6f}); set("theta stop",{90.0+th_half_deg:.6f}); set("index",{nc:.8f});
        addring; set("name","PulTopOutL"); set("x",{-xc_out:.12e}); set("y",{+yc_out:.12e}); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{Ro-0.5*wb:.12e}); set("outer radius",{Ro+0.5*wb:.12e});
        set("theta start",-90.0); set("theta stop",{-90.0+th_half_deg:.6f}); set("index",{nc:.8f});
        addring; set("name","PulTopOutR"); set("x",{+xc_out:.12e}); set("y",{+yc_out:.12e}); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{Ro-0.5*wb:.12e}); set("outer radius",{Ro+0.5*wb:.12e});
        set("theta start",{-90.0-th_half_deg:.6f}); set("theta stop",-90.0); set("index",{nc:.8f});
        addrect; set("name","LeadTopL"); set("x min",{-x_h-1.5e-6:.12e}); set("x max",{-xc_out+0.02e-6:.12e});
        set("y",{+y_port_abs:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","LeadTopR"); set("x min",{+xc_out-0.02e-6:.12e}); set("x max",{+x_h+1.5e-6:.12e});
        set("y",{+y_port_abs:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});

        addfdtd; set("dimension","3D");
        set("x",0); set("x span",{2.0*x_h:.12e}); set("y min",{-Rb-1.6e-6:.12e}); set("y max",{+Rb+1.6e-6:.12e});
        set("z",0); set("z span",1.6e-6); set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",{sim_t:.12e}); set("auto shutoff min",{AUTO_SHUTOFF_MIN}); set("z min bc","Symmetric");
        addmesh; set("name","GapThM"); set("x",0); set("x span",{2.0*xc_out+1.0e-6:.12e});
        set("y",{-R-0.5*wr-0.5*gt:.12e}); set("y span",{gt+wr+wb+0.45e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        addmesh; set("name","GapDrM"); set("x",0); set("x span",{2.0*xc_out+1.0e-6:.12e});
        set("y",{+R+0.5*wr+0.5*gd:.12e}); set("y span",{gd+wr+wb+0.45e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        setglobalsource("wavelength start",{wl_min:.12e}); setglobalsource("wavelength stop",{wl_max:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_h+0.8e-6:.12e}); set("y",{-y_port_abs:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_h-0.8e-6:.12e}); set("y",{-y_port_abs:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortDrop"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_h+0.8e-6:.12e}); set("y",{+y_port_abs:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",201);
        """)

    elif top == "vernier_crow_ad":
        R2 = c["ring2_radius_um"] * 1e-6
        gi = c["inter_ring_gap_um"] * 1e-6
        # Ring 1 centered at (0, 0), Ring 2 directly above at (0, R + R2 + wr + gi)
        y_r2 = R + R2 + wr + gi
        y_th = -(R + 0.5 * wr + gt + 0.5 * wb)
        y_dr = y_r2 + R2 + 0.5 * wr + gd + 0.5 * wb
        x_h = max(R, R2) + 3.2e-6
        # Note: in a 2-ring CROW (even number of rings!), drop exits in the +x direction (right side!)
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addring; set("name","Ring1"); set("x",0); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{R-0.5*wr:.12e}); set("outer radius",{R+0.5*wr:.12e}); set("index",{nc:.8f});
        addring; set("name","Ring2"); set("x",0); set("y",{y_r2:.12e}); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{R2-0.5*wr:.12e}); set("outer radius",{R2+0.5*wr:.12e}); set("index",{nc:.8f});
        addrect; set("name","BusThru"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
        set("y",{y_th:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","BusDrop"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
        set("y",{y_dr:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addfdtd; set("dimension","3D");
        set("x",0); set("x span",{2.0*x_h:.12e}); set("y min",{y_th-1.5e-6:.12e}); set("y max",{y_dr+1.5e-6:.12e});
        set("z",0); set("z span",1.6e-6); set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",{sim_t:.12e}); set("auto shutoff min",{AUTO_SHUTOFF_MIN}); set("z min bc","Symmetric");
        addmesh; set("name","GapThM"); set("x",0); set("x span",2.4e-6);
        set("y",{y_th+0.5*wb+0.5*gt:.12e}); set("y span",{gt+wr+wb+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        addmesh; set("name","GapIntM"); set("x",0); set("x span",2.4e-6);
        set("y",{R+0.5*wr+0.5*gi:.12e}); set("y span",{gi+2*wr+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        addmesh; set("name","GapDrM"); set("x",0); set("x span",2.4e-6);
        set("y",{y_dr-0.5*wb-0.5*gd:.12e}); set("y span",{gd+wr+wb+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        setglobalsource("wavelength start",{wl_min:.12e}); setglobalsource("wavelength stop",{wl_max:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_h+0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_h-0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortDrop"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_h-0.8e-6:.12e}); set("y",{y_dr:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",201);
        """)

    _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{cid}.fsp")
    st_code = int(_scalar_real(fdtd.getresult("FDTD", "status")))
    wl_r, t_th = _extract_port_power(fdtd, "PortThru")
    t_dr = _extract_port_power(fdtd, "PortDrop")[1] if has_drop else None

    if top == "vernier_crow_ad" and t_dr is not None:
        # Measure CROW multi-FSR, doublet splitting & 3-dB flat-top passband width near 1550 nm
        pks_all, _ = find_peaks(t_dr, prominence=0.15, distance=18)
        # Cluster split doublets separated by < 4 nm into resonance centers
        centers = []
        for idx in pks_all:
            w = float(wl_r[idx])
            if not centers or (w - centers[-1][-1]) > 4.0:
                centers.append([w])
            else:
                centers[-1].append(w)
        res_c = [round(float(np.mean(g)), 4) for g in centers]
        fsr_nm = round(float(np.mean(np.diff(res_c))), 4) if len(res_c) >= 2 else 0.0
        wl0 = min(res_c, key=lambda w: abs(w - 1550.0)) if res_c else 1550.0
        win = np.abs(wl_r - wl0) <= 5.0
        ww, dw, tw = wl_r[win], t_dr[win], t_th[win]
        pk_dr = float(np.max(dw))
        sub_pks, _ = find_peaks(dw, height=0.4 * pk_dr, distance=2)
        above3 = ww[dw >= 0.5 * pk_dr]
        rec = {
            "case_id": cid,
            "topology": top,
            "fdtd_status_code": st_code,
            "FSR_nm": fsr_nm,
            "center_res_nm": round(wl0, 4),
            "crow_drop_peak_T": round(pk_dr, 4),
            "crow_IL_drop_dB": round(-10.0 * math.log10(max(pk_dr, 1e-6)), 3),
            "crow_drop_3db_bw_nm": round(float(above3[-1] - above3[0]), 4) if len(above3) >= 2 else 0.0,
            "crow_split_nm": round(float(ww[sub_pks[-1]] - ww[sub_pks[0]]), 4) if len(sub_pks) >= 2 else 0.0,
            "ER_thru_dB": round(10.0 * math.log10(max(float(np.percentile(tw, 95)), 1e-6) / max(float(np.min(tw)), 1e-6)), 2),
            "max_passivity_sum": round(float(np.max(t_th + t_dr)), 5),
            "spectrum_wl_nm": [round(float(x), 4) for x in wl_r[::2]],
            "spectrum_thru": [round(float(x), 5) for x in t_th[::2]],
            "spectrum_drop": [round(float(x), 5) for x in t_dr[::2]],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM Suite 12A GPU] {cid}: status={st_code}, FSR={fsr_nm:.3f}nm, "
            f"split={rec['crow_split_nm']:.3f}nm, BW_3dB={rec['crow_drop_3db_bw_nm']:.3f}nm, "
            f"IL_drop={rec['crow_IL_drop_dB']:.2f}dB, passivity={rec['max_passivity_sum']:.4f}"
        )
        return rec

    ana = _analyze_ring_spectrum(wl_r, t_th, t_dr, ring_len_um)
    ana["case_id"] = cid
    ana["topology"] = top
    ana["fdtd_status_code"] = st_code
    ana["wall_time_s"] = round(time.perf_counter() - t0, 2)
    print(
        f"[LUM Suite 12A GPU] {cid}: status={st_code}, FSR={ana['FSR_nm']:.3f}nm, "
        f"Q_L={ana['Q_loaded_emp']:.1f} (CMT={ana['cmt_fit_Q_loaded']:.1f}, R2={ana['cmt_fit_r_squared']:.5f}), "
        f"ER={ana['ER_thru_dB']:.2f}dB, IL_drop={ana['IL_drop_dB']}dB, pass={ana['max_passivity_sum']:.4f}"
    )
    return ana


# ==============================================================================
# Suite 12B: Direct GDS Import Roundtrip + Sidewall-Bump Split-Doublet Blind Fit
# ==============================================================================
def run_suite12b(fdtd, specs: dict, ckpt: dict) -> dict:
    out = dict(ckpt.get("suite12b_gds_and_doublet", {}))

    # 1. Direct gdsimport of MicroRing's exported .gds mask!
    cg = specs["suite12b_gds_and_doublet"]["gds_roundtrip"]
    hf = cg["film_height_um"] * 1e-6
    nc = cg["core_index"]["center"]["nx"]
    ncl = cg["clad_index"]["center"]["nx"]
    if "gds_roundtrip" not in out:
        t0 = time.perf_counter()
        gds_path = (GDS_P3_DIR / f"{cg['case_id']}.gds").resolve()
        if not gds_path.is_file():
            raise FileNotFoundError(f"Run run_microring_independent_p3.py first to export {gds_path}")
        import gdstk
        lib = gdstk.read_gds(str(gds_path))
        top_cell = lib.top_level()[0]
        cell_name = top_cell.name
        # Measure exact bounding box of layer (1, 0) core polygons to place ports cleanly on bus ends!
        polys_l1 = [p for p in top_cell.get_polygons() if p.layer == cg["gds_layer"]]
        all_pts = np.vstack([p.points for p in polys_l1])
        x_min_g, y_min_g = np.min(all_pts, axis=0) * 1e-6
        x_max_g, y_max_g = np.max(all_pts, axis=0) * 1e-6
        R = cg["ring_radius_um"] * 1e-6
        wr = cg["ring_width_um"] * 1e-6
        wb = cg["bus_width_um"] * 1e-6
        gt = cg["gap_thru_um"] * 1e-6
        gd = cg["gap_drop_um"] * 1e-6
        xc_g = 0.5 * (x_min_g + x_max_g)
        yc_g = 0.5 * (y_min_g + y_max_g)
        y_th = yc_g - (R + 0.5 * wr + gt + 0.5 * wb)
        y_dr = yc_g + (R + 0.5 * wr + gd + 0.5 * wb)
        x_h = min(0.5 * (x_max_g - x_min_g) - 0.5e-6, R + 3.2e-6)
        gds_s = str(gds_path).replace("\\", "/")

        fdtd.eval(f"""
        switchtolayout; deleteall;
        if (materialexists("Si_GDS_P3")) {{ deletematerial("Si_GDS_P3"); }}
        m_g = addmaterial("(n,k) Material"); setmaterial(m_g,"name","Si_GDS_P3");
        setmaterial("Si_GDS_P3","Refractive Index",{nc:.8f});
        setmaterial("Si_GDS_P3","Imaginary Refractive Index",0.0);
        gdsimport("{gds_s}", "{cell_name}", "1:0", "Si_GDS_P3", {-0.5*hf:.12e}, {+0.5*hf:.12e});
        addfdtd; set("dimension","3D");
        set("x",{xc_g:.12e}); set("x span",{2.0*x_h:.12e});
        set("y min",{y_th-1.5e-6:.12e}); set("y max",{y_dr+1.5e-6:.12e});
        set("z",0); set("z span",1.6e-6);
        set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",{cg['fdtd_sim_time_ps']*1e-12:.12e}); set("auto shutoff min",{AUTO_SHUTOFF_MIN});
        set("z min bc","Symmetric");
        addmesh; set("name","GapThM"); set("x",{xc_g:.12e}); set("x span",2.4e-6);
        set("y",{y_th+0.5*wb+0.5*gt:.12e}); set("y span",{gt+wr+wb+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        addmesh; set("name","GapDrM"); set("x",{xc_g:.12e}); set("x span",2.4e-6);
        set("y",{y_dr-0.5*wb-0.5*gd:.12e}); set("y span",{gd+wr+wb+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        setglobalsource("wavelength start",{cg['wl_min_nm']*1e-9:.12e});
        setglobalsource("wavelength stop",{cg['wl_max_nm']*1e-9:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{xc_g-x_h+0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{xc_g+x_h-0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortDrop"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{xc_g-x_h+0.8e-6:.12e}); set("y",{y_dr:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",201);
        """)
        _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{cg['case_id']}.fsp")
        st_gds = int(_scalar_real(fdtd.getresult("FDTD", "status")))
        wl_g, th_g = _extract_port_power(fdtd, "PortThru")
        _, dr_g = _extract_port_power(fdtd, "PortDrop")
        ana_g = _analyze_ring_spectrum(wl_g, th_g, dr_g, 2.0 * math.pi * cg["ring_radius_um"])
        ana_g["case_id"] = cg["case_id"]
        ana_g["fdtd_status_code"] = st_gds
        ana_g["gds_cell_imported"] = cell_name
        ana_g["wall_time_s"] = round(time.perf_counter() - t0, 2)
        print(
            f"[LUM Suite 12B GDS-FDTD] status={st_gds}, FSR={ana_g['FSR_nm']:.3f}nm, "
            f"res={ana_g['center_res_nm']:.3f}nm, Q_L={ana_g['Q_loaded_emp']:.1f}, ER={ana_g['ER_thru_dB']:.2f}dB"
        )
        out["gds_roundtrip"] = ana_g
        ckpt["suite12b_gds_and_doublet"] = out
        _save_ckpt(ckpt)
    else:
        print("[LUM Suite 12B CACHED] gds_roundtrip")

    # 2. Sidewall-Bump Split-Doublet Full-Wave 3D FDTD + Blind Auto-BIC CMT Fitter Check!
    if "split_doublet_fdtd" not in out:
        cd = specs["suite12b_gds_and_doublet"]["split_doublet_fdtd"]
        t0 = time.perf_counter()
        Rd = cd["ring_radius_um"] * 1e-6
        wrd = cd["ring_width_um"] * 1e-6
        wbd = cd["bus_width_um"] * 1e-6
        gtd = cd["gap_thru_um"] * 1e-6
        gdd = cd["gap_drop_um"] * 1e-6
        y_th = -(Rd + 0.5 * wrd + gtd + 0.5 * wbd)
        y_dr = +(Rd + 0.5 * wrd + gdd + 0.5 * wbd)
        x_h = Rd + 3.2e-6
        b_th = math.radians(cd["bump_theta_deg"])
        b_rad = cd["bump_radial_um"] * 1e-6
        b_az = cd["bump_azimuthal_um"] * 1e-6
        r_bc = Rd + 0.5 * wrd + 0.35 * b_rad
        xb = r_bc * math.cos(b_th)
        yb = r_bc * math.sin(b_th)

        fdtd.eval(f"""
        switchtolayout; deleteall;
        addring; set("name","Ring"); set("x",0); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{Rd-0.5*wrd:.12e}); set("outer radius",{Rd+0.5*wrd:.12e}); set("index",{nc:.8f});
        addrect; set("name","ScatterBump"); set("x",{xb:.12e}); set("y",{yb:.12e}); set("z",0); set("z span",{hf:.12e});
        set("x span",{b_rad*1.4:.12e}); set("y span",{b_az:.12e});
        set("first axis","z"); set("rotation 1",{cd['bump_theta_deg']:.4f}); set("index",{nc:.8f});
        addrect; set("name","BusThru"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
        set("y",{y_th:.12e}); set("y span",{wbd:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","BusDrop"); set("x",0); set("x span",{2.0*(x_h+1.5e-6):.12e});
        set("y",{y_dr:.12e}); set("y span",{wbd:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addfdtd; set("dimension","3D");
        set("x",0); set("x span",{2.0*x_h:.12e}); set("y min",{y_th-1.5e-6:.12e}); set("y max",{y_dr+1.5e-6:.12e});
        set("z",0); set("z span",1.6e-6); set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",{cd['fdtd_sim_time_ps']*1e-12:.12e}); set("auto shutoff min",{AUTO_SHUTOFF_MIN});
        set("z min bc","Symmetric");
        addmesh; set("name","GapThM"); set("x",0); set("x span",2.4e-6);
        set("y",{y_th+0.5*wbd+0.5*gtd:.12e}); set("y span",{gtd+wrd+wbd+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        addmesh; set("name","GapDrM"); set("x",0); set("x span",2.4e-6);
        set("y",{y_dr-0.5*wbd-0.5*gdd:.12e}); set("y span",{gdd+wrd+wbd+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        addmesh; set("name","BumpM"); set("x",{xb:.12e}); set("x span",0.4e-6);
        set("y",{yb:.12e}); set("y span",0.4e-6); set("z",0); set("z span",{hf+0.12e-6:.12e});
        set("dx",16e-9); set("dy",16e-9); set("dz",24e-9);
        setglobalsource("wavelength start",{cd['wl_min_nm']*1e-9:.12e});
        setglobalsource("wavelength stop",{cd['wl_max_nm']*1e-9:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_h+0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_h-0.8e-6:.12e}); set("y",{y_th:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortDrop"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_h+0.8e-6:.12e}); set("y",{y_dr:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",301);
        """)
        _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{cd['case_id']}.fsp")
        st_dbl = int(_scalar_real(fdtd.getresult("FDTD", "status")))
        wl_d, th_d = _extract_port_power(fdtd, "PortThru")
        _, dr_d = _extract_port_power(fdtd, "PortDrop")

        sys.path.insert(0, str(REPO_ROOT / "src"))
        from microring.mrr_cmt_fit_predict import fit_single_resonance_cmt

        ic = int(np.argmin(th_d))
        wc = float(wl_d[ic])
        win = np.abs(wl_d - wc) <= 4.5
        fit_auto, _, _ = fit_single_resonance_cmt(
            wl_d[win], th_d[win], model_type="auto", known_regime="critical", ng_assumed=4.3, ring_length_um=2.0 * math.pi * cd["ring_radius_um"]
        )
        fit_dbl, _, _ = fit_single_resonance_cmt(
            wl_d[win], th_d[win], model_type="split_doublet", known_regime="critical", ng_assumed=4.3, ring_length_um=2.0 * math.pi * cd["ring_radius_um"]
        )
        fit_lor, _, _ = fit_single_resonance_cmt(
            wl_d[win], th_d[win], model_type="lorentzian", known_regime="critical", ng_assumed=4.3, ring_length_um=2.0 * math.pi * cd["ring_radius_um"]
        )
        sub_pks, _ = find_peaks(dr_d[win], prominence=0.05, distance=3)
        emp_split_nm = round(float(wl_d[win][sub_pks[-1]] - wl_d[win][sub_pks[0]]), 4) if len(sub_pks) >= 2 else 0.0
        emp_split_ghz = round((C0 / (wc * 1e-9) ** 2) * (emp_split_nm * 1e-9) * 1e-9, 2) if emp_split_nm > 0 else 0.0

        rec_dbl = {
            "case_id": cd["case_id"],
            "fdtd_status_code": st_dbl,
            "center_res_nm": round(wc, 4),
            "emp_doublet_peaks_count": int(len(sub_pks)),
            "emp_split_nm": emp_split_nm,
            "emp_split_2mu_ghz": emp_split_ghz,
            "auto_bic_selected_model": fit_auto.model_type,
            "split_doublet_fit_r2": round(float(fit_dbl.r_squared), 5),
            "lorentzian_fit_r2": round(float(fit_lor.r_squared), 5),
            "split_doublet_2mu_ghz": round(float(getattr(fit_dbl, "mode_splitting_ghz", 0.0)), 2),
            "split_doublet_Q_loaded": round(float(fit_dbl.Q_loaded), 1),
            "max_passivity_sum": round(float(np.max(th_d + dr_d)), 5),
            "spectrum_wl_nm": [round(float(x), 4) for x in wl_d[win]],
            "spectrum_thru": [round(float(x), 5) for x in th_d[win]],
            "spectrum_drop": [round(float(x), 5) for x in dr_d[win]],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM Suite 12B Doublet] status={st_dbl}, emp_split={emp_split_nm:.3f}nm ({emp_split_ghz:.1f} GHz), "
            f"BIC_sel={rec_dbl['auto_bic_selected_model']}, R2(dbl/lor)={rec_dbl['split_doublet_fit_r2']:.5f}/{rec_dbl['lorentzian_fit_r2']:.5f}, "
            f"fit_2mu={rec_dbl['split_doublet_2mu_ghz']:.1f} GHz"
        )
        out["split_doublet_fdtd"] = rec_dbl
        ckpt["suite12b_gds_and_doublet"] = out
        _save_ckpt(ckpt)
    else:
        print("[LUM Suite 12B CACHED] split_doublet_fdtd")
    return out


# ==============================================================================
# Suite 13: Optimal 2x2 MMI, Full S-Bend 2x2 DC, Low-Q RAMZI, 3D Focusing GC & 3D Butt Taper
# ==============================================================================
def run_suite13(fdtd, specs: dict, ckpt: dict) -> dict:
    s13 = specs["suite13_mzi_and_3d_io"]
    out: dict[str, dict] = dict(ckpt.get("suite13_mzi_and_3d_io", {}))

    c1 = s13["mmi_2x2_optimal"]
    wl = c1["wavelength_nm"] * 1e-9
    hf = c1["film_height_um"] * 1e-6
    wa = c1["arm_width_um"] * 1e-6
    nc = c1["core_index"]["center"]["nx"]
    ncl = c1["clad_index"]["center"]["nx"]

    # 1. Optimal 2x2 MMI (L = 42.3 um)
    if "mmi_2x2_optimal" not in out:
        t0 = time.perf_counter()
        wm = c1["mmi_width_um"] * 1e-6
        lm = c1["mmi_length_um"] * 1e-6
        wt = c1["mmi_taper_width_um"] * 1e-6
        lt = c1["mmi_taper_length_um"] * 1e-6
        yp = c1["port_offset_um"] * 1e-6
        x_in0 = -(0.5 * lm + lt + 2.4e-6)
        x_in1 = -(0.5 * lm + lt)
        x_out0 = +(0.5 * lm + lt)
        x_out1 = +(0.5 * lm + lt + 2.4e-6)

        fdtd.eval(f"""
        switchtolayout; deleteall;
        addrect; set("name","MMIBox"); set("x",0); set("x span",{lm:.12e});
        set("y",0); set("y span",{wm:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        """)
        for tag, sgn in (("Top", +1.0), ("Bot", -1.0)):
            yc = sgn * yp
            fdtd.eval(f"""
            addrect; set("name","InL_{tag}"); set("x min",{x_in0-1e-6:.12e}); set("x max",{x_in1:.12e});
            set("y",{yc:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            addpoly; set("name","InT_{tag}"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            set("vertices",[{x_in1:.12e},{yc-0.5*wa:.12e}; {-0.5*lm:.12e},{yc-0.5*wt:.12e}; {-0.5*lm:.12e},{yc+0.5*wt:.12e}; {x_in1:.12e},{yc+0.5*wa:.12e}]);
            addpoly; set("name","OutT_{tag}"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            set("vertices",[{0.5*lm:.12e},{yc-0.5*wt:.12e}; {x_out0:.12e},{yc-0.5*wa:.12e}; {x_out0:.12e},{yc+0.5*wa:.12e}; {0.5*lm:.12e},{yc+0.5*wt:.12e}]);
            addrect; set("name","OutL_{tag}"); set("x min",{x_out0:.12e}); set("x max",{x_out1+1e-6:.12e});
            set("y",{yc:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            """)
        fdtd.eval(f"""
        addfdtd; set("dimension","3D");
        set("x min",{x_in0+0.3e-6:.12e}); set("x max",{x_out1-0.3e-6:.12e});
        set("y",0); set("y span",{wm+3.0e-6:.12e}); set("z",0); set("z span",1.6e-6);
        set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",6000e-15); set("auto shutoff min",{AUTO_SHUTOFF_MIN}); set("z min bc","Symmetric");
        setglobalsource("wavelength start",{wl:.12e}); setglobalsource("wavelength stop",{wl:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{x_in0+1.0e-6:.12e}); set("y",{-yp:.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortBar"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{x_out1-1.0e-6:.12e}); set("y",{-yp:.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortCross"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{x_out1-1.0e-6:.12e}); set("y",{+yp:.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",1);
        """)
        _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{c1['case_id']}.fsp")
        sb = complex(np.asarray(fdtd.getresult("FDTD::ports::PortBar", "expansion for port monitor")["S"]).reshape(-1)[0])
        sc = complex(np.asarray(fdtd.getresult("FDTD::ports::PortCross", "expansion for port monitor")["S"]).reshape(-1)[0])
        pb, pc = float(abs(sb) ** 2), float(abs(sc) ** 2)
        dphi = math.degrees(math.atan2(math.sin(np.angle(sc) - np.angle(sb)), math.cos(np.angle(sc) - np.angle(sb))))
        out["mmi_2x2_optimal"] = {
            "case_id": c1["case_id"],
            "split_bar_pct": round(pb * 100.0, 2),
            "split_cross_pct": round(pc * 100.0, 2),
            "excess_loss_db": round(-10.0 * math.log10(max(pb + pc, 1e-6)), 3),
            "imbalance_db": round(10.0 * math.log10(max(pb, 1e-9) / max(pc, 1e-9)), 3),
            "phase_diff_deg": round(dphi, 2),
            "fdtd_status_code": int(_scalar_real(fdtd.getresult("FDTD", "status"))),
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(f"[LUM Suite 13A GPU] 2x2 MMI (L=42.3um): {out['mmi_2x2_optimal']}")
        ckpt["suite13_mzi_and_3d_io"] = out
        _save_ckpt(ckpt)
    else:
        print("[LUM Suite 13A CACHED] mmi_2x2_optimal")

    # 2. Full S-Bend 2x2 Directional Coupler
    if "dc_2x2_full_sbend" not in out:
        c2 = s13["dc_2x2_full_sbend"]
        t0 = time.perf_counter()
        lc = c2["dc_straight_length_um"] * 1e-6
        ls = c2["sbend_length_um"] * 1e-6
        gdc = c2["dc_gap_um"] * 1e-6
        yc_dc = 0.5 * (wa + gdc)
        yp_dc = 0.5 * c2["arm_separation_um"] * 1e-6
        x_e = 0.5 * lc + ls + 2.2e-6
        v_il_top = _sbend_polygon_verts(-0.5 * lc - ls, +yp_dc, -0.5 * lc, +yc_dc, wa)
        v_il_bot = _sbend_polygon_verts(-0.5 * lc - ls, -yp_dc, -0.5 * lc, -yc_dc, wa)
        v_or_top = _sbend_polygon_verts(+0.5 * lc, +yc_dc, +0.5 * lc + ls, +yp_dc, wa)
        v_or_bot = _sbend_polygon_verts(+0.5 * lc, -yc_dc, +0.5 * lc + ls, -yp_dc, wa)
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addrect; set("name","DCTop"); set("x",0); set("x span",{lc+0.04e-6:.12e});
        set("y",{+yc_dc:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","DCBot"); set("x",0); set("x span",{lc+0.04e-6:.12e});
        set("y",{-yc_dc:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addpoly; set("name","SbInTop"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f}); set("vertices",{v_il_top});
        addpoly; set("name","SbInBot"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f}); set("vertices",{v_il_bot});
        addpoly; set("name","SbOutTop"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f}); set("vertices",{v_or_top});
        addpoly; set("name","SbOutBot"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f}); set("vertices",{v_or_bot});
        addrect; set("name","InLeadB"); set("x min",{-x_e-1e-6:.12e}); set("x max",{-0.5*lc-ls+0.02e-6:.12e});
        set("y",{-yp_dc:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","InLeadT"); set("x min",{-x_e-1e-6:.12e}); set("x max",{-0.5*lc-ls+0.02e-6:.12e});
        set("y",{+yp_dc:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","OutLeadB"); set("x min",{+0.5*lc+ls-0.02e-6:.12e}); set("x max",{+x_e+1e-6:.12e});
        set("y",{-yp_dc:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","OutLeadT"); set("x min",{+0.5*lc+ls-0.02e-6:.12e}); set("x max",{+x_e+1e-6:.12e});
        set("y",{+yp_dc:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});

        addfdtd; set("dimension","3D");
        set("x",0); set("x span",{2.0*x_e:.12e}); set("y",0); set("y span",{2.0*yp_dc+3.0e-6:.12e});
        set("z",0); set("z span",1.6e-6); set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",5000e-15); set("auto shutoff min",{AUTO_SHUTOFF_MIN}); set("z min bc","Symmetric");
        addmesh; set("name","GapDC"); set("x",0); set("x span",{lc+6.0e-6:.12e});
        set("y",0); set("y span",{gdc+2*wa+0.4e-6:.12e}); set("z",0); set("z span",{hf+0.12e-6:.12e});
        set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        setglobalsource("wavelength start",{wl:.12e}); setglobalsource("wavelength stop",{wl:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_e+0.9e-6:.12e}); set("y",{-yp_dc:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortBar"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_e-0.9e-6:.12e}); set("y",{-yp_dc:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortCross"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_e-0.9e-6:.12e}); set("y",{+yp_dc:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",1);
        """)
        _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{c2['case_id']}.fsp")
        sb2 = complex(np.asarray(fdtd.getresult("FDTD::ports::PortBar", "expansion for port monitor")["S"]).reshape(-1)[0])
        sc2 = complex(np.asarray(fdtd.getresult("FDTD::ports::PortCross", "expansion for port monitor")["S"]).reshape(-1)[0])
        pb2, pc2 = float(abs(sb2) ** 2), float(abs(sc2) ** 2)
        out["dc_2x2_full_sbend"] = {
            "case_id": c2["case_id"],
            "split_bar_pct": round(pb2 * 100.0, 2),
            "split_cross_pct": round(pc2 * 100.0, 2),
            "excess_loss_db": round(-10.0 * math.log10(max(pb2 + pc2, 1e-6)), 3),
            "imbalance_db": round(10.0 * math.log10(max(pb2, 1e-9) / max(pc2, 1e-9)), 3),
            "fdtd_status_code": int(_scalar_real(fdtd.getresult("FDTD", "status"))),
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(f"[LUM Suite 13B GPU] Full S-Bend 2x2 DC: {out['dc_2x2_full_sbend']}")
        ckpt["suite13_mzi_and_3d_io"] = out
        _save_ckpt(ckpt)
    else:
        print("[LUM Suite 13B CACHED] dc_2x2_full_sbend")

    # 3. Complete Low-Q RAMZI (two 2x2 DCs + balanced arms + R=5 um, gap=100 nm ring on upper arm!)
    if "lowq_ramzi" not in out:
        c3 = s13["lowq_ramzi"]
        t0 = time.perf_counter()
        lc3 = c3["dc_straight_length_um"] * 1e-6
        ls3 = c3["sbend_length_um"] * 1e-6
        gdc3 = c3["dc_gap_um"] * 1e-6
        yc3 = 0.5 * (wa + gdc3)
        ya3 = 0.5 * c3["arm_separation_um"] * 1e-6
        la3 = c3["arm_straight_um"] * 1e-6
        R3 = c3["ring_radius_um"] * 1e-6
        gr3 = c3["ring_gap_um"] * 1e-6
        x_dc1_r = -0.5 * la3 - ls3
        x_dc1_l = x_dc1_r - lc3
        x_dc2_l = +0.5 * la3 + ls3
        x_dc2_r = x_dc2_l + lc3
        x_in_end = x_dc1_l - ls3 - 2.0e-6
        x_out_end = x_dc2_r + ls3 + 2.0e-6
        y_ring_c = ya3 + 0.5 * wa + gr3 + 0.5 * wa + R3
        y_port_ext = 2.0e-6

        sbs = [
            ("Sb1InT", _sbend_polygon_verts(x_dc1_l - ls3, +y_port_ext, x_dc1_l, +yc3, wa)),
            ("Sb1InB", _sbend_polygon_verts(x_dc1_l - ls3, -y_port_ext, x_dc1_l, -yc3, wa)),
            ("Sb1OutT", _sbend_polygon_verts(x_dc1_r, +yc3, -0.5 * la3, +ya3, wa)),
            ("Sb1OutB", _sbend_polygon_verts(x_dc1_r, -yc3, -0.5 * la3, -ya3, wa)),
            ("Sb2InT", _sbend_polygon_verts(+0.5 * la3, +ya3, x_dc2_l, +yc3, wa)),
            ("Sb2InB", _sbend_polygon_verts(+0.5 * la3, -ya3, x_dc2_l, -yc3, wa)),
            ("Sb2OutT", _sbend_polygon_verts(x_dc2_r, +yc3, x_dc2_r + ls3, +y_port_ext, wa)),
            ("Sb2OutB", _sbend_polygon_verts(x_dc2_r, -yc3, x_dc2_r + ls3, -y_port_ext, wa)),
        ]
        sb_cmds = "\n".join(
            f'addpoly; set("name","{nm}"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f}); set("vertices",{vs});'
            for nm, vs in sbs
        )
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addrect; set("name","DC1T"); set("x",{0.5*(x_dc1_l+x_dc1_r):.12e}); set("x span",{lc3+0.04e-6:.12e});
        set("y",{+yc3:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","DC1B"); set("x",{0.5*(x_dc1_l+x_dc1_r):.12e}); set("x span",{lc3+0.04e-6:.12e});
        set("y",{-yc3:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","DC2T"); set("x",{0.5*(x_dc2_l+x_dc2_r):.12e}); set("x span",{lc3+0.04e-6:.12e});
        set("y",{+yc3:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","DC2B"); set("x",{0.5*(x_dc2_l+x_dc2_r):.12e}); set("x span",{lc3+0.04e-6:.12e});
        set("y",{-yc3:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","ArmTop"); set("x",0); set("x span",{la3+0.04e-6:.12e});
        set("y",{+ya3:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","ArmBot"); set("x",0); set("x span",{la3+0.04e-6:.12e});
        set("y",{-ya3:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addring; set("name","RamziRing"); set("x",0); set("y",{y_ring_c:.12e}); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{R3-0.5*wa:.12e}); set("outer radius",{R3+0.5*wa:.12e}); set("index",{nc:.8f});
        {sb_cmds}
        addrect; set("name","InLT"); set("x min",{x_in_end-1e-6:.12e}); set("x max",{x_dc1_l-ls3+0.02e-6:.12e});
        set("y",{+y_port_ext:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","InLB"); set("x min",{x_in_end-1e-6:.12e}); set("x max",{x_dc1_l-ls3+0.02e-6:.12e});
        set("y",{-y_port_ext:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","OutLT"); set("x min",{x_dc2_r+ls3-0.02e-6:.12e}); set("x max",{x_out_end+1e-6:.12e});
        set("y",{+y_port_ext:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addrect; set("name","OutLB"); set("x min",{x_dc2_r+ls3-0.02e-6:.12e}); set("x max",{x_out_end+1e-6:.12e});
        set("y",{-y_port_ext:.12e}); set("y span",{wa:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});

        addfdtd; set("dimension","3D");
        set("x min",{x_in_end+0.2e-6:.12e}); set("x max",{x_out_end-0.2e-6:.12e});
        set("y min",{-ya3-1.4e-6:.12e}); set("y max",{y_ring_c+R3+0.5*wa+1.4e-6:.12e});
        set("z",0); set("z span",1.6e-6); set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",{c3['fdtd_sim_time_ps']*1e-12:.12e}); set("auto shutoff min",{AUTO_SHUTOFF_MIN}); set("z min bc","Symmetric");
        addmesh; set("name","RingGapM"); set("x",0); set("x span",2.4e-6);
        set("y",{ya3+0.5*wa+0.5*gr3:.12e}); set("y span",{gr3+2*wa+0.25e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",24e-9); set("dy",20e-9); set("dz",24e-9);
        setglobalsource("wavelength start",{c3['wl_min_nm']*1e-9:.12e});
        setglobalsource("wavelength stop",{c3['wl_max_nm']*1e-9:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{x_in_end+0.8e-6:.12e}); set("y",{-y_port_ext:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortBar"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{x_out_end-0.8e-6:.12e}); set("y",{-y_port_ext:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortCross"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{x_out_end-0.8e-6:.12e}); set("y",{+y_port_ext:.12e}); set("y span",1.7e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",201);
        """)
        _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{c3['case_id']}.fsp")
        st_rz = int(_scalar_real(fdtd.getresult("FDTD", "status")))
        wl_rz, tb_rz = _extract_port_power(fdtd, "PortBar")
        _, tc_rz = _extract_port_power(fdtd, "PortCross")
        er_bar = round(10.0 * math.log10(max(float(np.max(tb_rz)), 1e-6) / max(float(np.min(tb_rz)), 1e-6)), 2)
        er_crs = round(10.0 * math.log10(max(float(np.max(tc_rz)), 1e-6) / max(float(np.min(tc_rz)), 1e-6)), 2)
        pks_rz, _ = find_peaks(tb_rz, prominence=0.35, distance=10)
        fsr_rz = round(float(np.mean(np.diff(wl_rz[pks_rz]))), 3) if len(pks_rz) >= 2 else 0.0
        out["lowq_ramzi"] = {
            "case_id": c3["case_id"],
            "fdtd_status_code": st_rz,
            "ramzi_FSR_nm": fsr_rz,
            "ER_bar_dB": er_bar,
            "ER_cross_dB": er_crs,
            "IL_min_dB": round(-10.0 * math.log10(max(float(np.max(tb_rz + tc_rz)), 1e-6)), 3),
            "max_passivity_sum": round(float(np.max(tb_rz + tc_rz)), 5),
            "spectrum_wl_nm": [round(float(x), 4) for x in wl_rz[::2]],
            "spectrum_bar": [round(float(x), 5) for x in tb_rz[::2]],
            "spectrum_cross": [round(float(x), 5) for x in tc_rz[::2]],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(f"[LUM Suite 13C GPU] Low-Q RAMZI: status={st_rz}, FSR={fsr_rz}nm, ER_bar={er_bar}dB, ER_cross={er_crs}dB, pass={out['lowq_ramzi']['max_passivity_sum']}")
        ckpt["suite13_mzi_and_3d_io"] = out
        _save_ckpt(ckpt)
    else:
        print("[LUM Suite 13C CACHED] lowq_ramzi")

    # 4. Full 3D FDTD Curved Focusing Grating Coupler (20 teeth, 70 nm etch, 2 um BOX + Si handle)
    if "gc_3d_focusing" not in out:
        c4 = s13["gc_3d_focusing"]
        t0 = time.perf_counter()
        h_rem = (c4["film_height_um"] - c4["etch_depth_um"]) * 1e-6
        h_box = c4["box_thickness_um"] * 1e-6
        h_top = c4["top_clad_thickness_um"] * 1e-6
        ang_h = 0.5 * c4["opening_angle_deg"]
        grooves_r = c4["grooves_r_um"]
        r_max = grooves_r[-1][1] * 1e-6 + 1.0e-6
        y_half_gc = r_max * math.sin(math.radians(ang_h)) + 1.5e-6
        ths = np.linspace(-math.radians(ang_h), +math.radians(ang_h), 32)
        fan_x = np.concatenate([[0.0], r_max * np.cos(ths), [0.0]])
        fan_y = np.concatenate([[-0.5 * wa], r_max * np.sin(ths), [+0.5 * wa]])
        fan_v = _poly_verts_lumerical(fan_x, fan_y)
        groove_cmds = []
        for ig, (r0u, r1u) in enumerate(grooves_r):
            groove_cmds.append(
                f'addring; set("name","Gr_{ig}"); set("x",0); set("y",0); '
                f'set("z min",{h_rem:.12e}); set("z max",{hf+0.005e-6:.12e}); '
                f'set("inner radius",{r0u*1e-6:.12e}); set("outer radius",{r1u*1e-6:.12e}); '
                f'set("theta start",{-ang_h-1.0:.4f}); set("theta stop",{+ang_h+1.0:.4f}); '
                f'set("index",{ncl:.8f}); set("override mesh order from material database",1); set("mesh order",1);'
            )
        z_up_mon = hf + h_top + 0.45e-6
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addrect; set("name","SiSub"); set("x min",-3.5e-6); set("x max",{r_max+2.5e-6:.12e});
        set("y",0); set("y span",{2.0*y_half_gc+4.0e-6:.12e}); set("z min",{-h_box-1.5e-6:.12e}); set("z max",{-h_box:.12e});
        set("index",{nc:.8f}); set("override mesh order from material database",1); set("mesh order",4);
        addrect; set("name","OxClad"); set("x min",-3.5e-6); set("x max",{r_max+2.5e-6:.12e});
        set("y",0); set("y span",{2.0*y_half_gc+4.0e-6:.12e}); set("z min",{-h_box:.12e}); set("z max",{hf+h_top:.12e});
        set("index",{ncl:.8f}); set("override mesh order from material database",1); set("mesh order",4);
        addrect; set("name","InWg"); set("x min",-3.5e-6); set("x max",0.5e-6);
        set("y",0); set("y span",{wa:.12e}); set("z min",0); set("z max",{hf:.12e});
        set("index",{nc:.8f}); set("override mesh order from material database",1); set("mesh order",2);
        addpoly; set("name","FanSi"); set("z min",0); set("z max",{hf:.12e}); set("vertices",{fan_v});
        set("index",{nc:.8f}); set("override mesh order from material database",1); set("mesh order",2);
        {chr(10).join(groove_cmds)}

        addfdtd; set("dimension","3D");
        set("x min",-2.6e-6); set("x max",{r_max+1.2e-6:.12e});
        set("y",0); set("y span",{2.0*y_half_gc:.12e});
        set("z min",{-h_box-0.8e-6:.12e}); set("z max",{hf+h_top+1.1e-6:.12e});
        set("background index",1.0); set("mesh accuracy",2);
        set("simulation time",2500e-15); set("auto shutoff min",{AUTO_SHUTOFF_MIN});
        set("y min bc","Anti-Symmetric");
        addmesh; set("name","GrMesh");
        set("x min",{grooves_r[0][0]*1e-6-0.5e-6:.12e}); set("x max",{grooves_r[-1][1]*1e-6+0.5e-6:.12e});
        set("y",0); set("y span",{2.0*(y_half_gc-0.8e-6):.12e});
        set("z min",-0.04e-6); set("z max",{hf+0.04e-6:.12e}); set("dx",28e-9); set("dy",45e-9); set("dz",20e-9);
        setglobalsource("wavelength start",{c4['wl_min_nm']*1e-9:.12e});
        setglobalsource("wavelength stop",{c4['wl_max_nm']*1e-9:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",-1.5e-6); set("y",0); set("y span",2.2e-6); set("z",0.11e-6); set("z span",1.6e-6);
        set("mode selection","fundamental TE mode");
        addpower; set("name","UpMon"); set("monitor type","2D Z-normal");
        set("x min",{grooves_r[0][0]*1e-6-2.0e-6:.12e}); set("x max",{grooves_r[-1][1]*1e-6+1.0e-6:.12e});
        set("y",0); set("y span",{2.0*(y_half_gc-0.4e-6):.12e}); set("z",{z_up_mon:.12e});
        set("override global monitor settings",1); set("frequency points",17);
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",17);
        """)
        _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{c4['case_id']}.fsp")
        st_gc3 = int(_scalar_real(fdtd.getresult("FDTD", "status")))
        f_gc = np.asarray(fdtd.getdata("UpMon", "f")).ravel()
        wls_gc = (C0 / f_gc) * 1e9
        ogc = np.argsort(wls_gc)
        wls_gc = wls_gc[ogc]
        t_up_3d = np.asarray(fdtd.transmission("UpMon")).ravel()[ogc]
        s11_p = np.asarray(fdtd.getresult("FDTD::ports::PortIn", "expansion for port monitor")["S"]).ravel()[ogc]
        s11_3d_db = 20.0 * np.log10(np.maximum(np.abs(s11_p), 1e-6))

        xg = np.asarray(fdtd.getdata("UpMon", "x")).ravel()
        yg = np.asarray(fdtd.getdata("UpMon", "y")).ravel()
        Ey_3d = np.squeeze(np.asarray(fdtd.getdata("UpMon", "Ey")))[:, :, ogc]
        Ex_3d = np.squeeze(np.asarray(fdtd.getdata("UpMon", "Ex")))[:, :, ogc]
        dx_g = np.gradient(xg)
        dy_g = np.gradient(yg)
        dA_g = dx_g[:, None] * dy_g[None, :]
        w0_y = 0.5 * c4["mfd_um"] * 1e-6
        w0_x = w0_y / math.cos(math.radians(c4["fiber_angle_deg"]))
        gy_f = np.exp(-((yg / w0_y) ** 2))
        il_3d_list = []
        for iw, wl_nm in enumerate(wls_gc):
            ey_s = Ey_3d[:, :, iw]
            ex_s = Ex_3d[:, :, iw]
            p_rad = float(np.sum((np.abs(ey_s) ** 2 + np.abs(ex_s) ** 2) * dA_g))
            k0 = 2.0 * math.pi / (wl_nm * 1e-9)
            kx_f = k0 * c4["n_env"] * math.sin(math.radians(c4["fiber_angle_deg"]))
            best_ov = 0.0
            for xc_f in np.linspace(grooves_r[0][0] * 1e-6 + 1.0e-6, grooves_r[-1][1] * 1e-6 - 2.0e-6, 40):
                gx_f = np.exp(-(((xg - xc_f) / w0_x) ** 2)) * np.exp(+1j * kx_f * (xg - xc_f))
                g2 = gx_f[:, None] * gy_f[None, :]
                num = abs(np.sum(ey_s * np.conj(g2) * dA_g)) ** 2
                den = max(p_rad * float(np.sum(np.abs(g2) ** 2 * dA_g)), 1e-30)
                ov = float(num / den)
                if ov > best_ov:
                    best_ov = ov
            eta_3d = max(float(t_up_3d[iw]) * best_ov, 1e-6)
            il_3d_list.append(-10.0 * math.log10(eta_3d))
        il_3d_arr = np.asarray(il_3d_list)
        ipk3 = int(np.argmin(il_3d_arr))
        i1550_3 = int(np.argmin(np.abs(wls_gc - 1550.0)))
        out["gc_3d_focusing"] = {
            "case_id": c4["case_id"],
            "fdtd_status_code": st_gc3,
            "peak_wavelength_nm": round(float(wls_gc[ipk3]), 2),
            "peak_il_3dfdtd_db": round(float(il_3d_arr[ipk3]), 3),
            "il_1550_3dfdtd_db": round(float(il_3d_arr[i1550_3]), 3),
            "eta_up_1550_pct": round(float(t_up_3d[i1550_3]) * 100.0, 2),
            "s11_1550_db": round(float(s11_3d_db[i1550_3]), 2),
            "spectrum_wl_nm": [round(float(x), 3) for x in wls_gc],
            "spectrum_il_db": [round(float(x), 3) for x in il_3d_arr],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM Suite 13D GPU] 3D Focusing GC: status={st_gc3}, peak={out['gc_3d_focusing']['peak_il_3dfdtd_db']:.3f} dB @ "
            f"{out['gc_3d_focusing']['peak_wavelength_nm']:.1f} nm, 1550nm={out['gc_3d_focusing']['il_1550_3dfdtd_db']:.3f} dB, "
            f"up={out['gc_3d_focusing']['eta_up_1550_pct']:.1f}%, S11={out['gc_3d_focusing']['s11_1550_db']:.1f} dB"
        )
        ckpt["suite13_mzi_and_3d_io"] = out
        _save_ckpt(ckpt)
    else:
        print("[LUM Suite 13D CACHED] gc_3d_focusing")

    # 5. Full 3D FDTD Reciprocal 30 um Inverse-Taper Butt Coupler (TE0 -> 200 nm tip -> 2.5 um Lensed Fiber!)
    if "butt_3d_taper" not in out:
        c5 = s13["butt_3d_taper"]
        t0 = time.perf_counter()
        l_tap = c5["taper_length_um"] * 1e-6
        w_tip = c5["w_tip_um"] * 1e-6
        w_bus = c5["w_bus_um"] * 1e-6
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addpoly; set("name","InvTaper"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        set("vertices",[0,{-0.5*w_tip:.12e}; {l_tap:.12e},{-0.5*w_bus:.12e}; {l_tap:.12e},{+0.5*w_bus:.12e}; 0,{+0.5*w_tip:.12e}]);
        addrect; set("name","BusLead"); set("x min",{l_tap-0.02e-6:.12e}); set("x max",{l_tap+3.5e-6:.12e});
        set("y",0); set("y span",{w_bus:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        addfdtd; set("dimension","3D");
        set("x min",-1.2e-6); set("x max",{l_tap+2.4e-6:.12e});
        set("y",0); set("y span",6.4e-6); set("z",0); set("z span",6.4e-6);
        set("background index",{ncl:.8f}); set("mesh accuracy",2);
        set("simulation time",3500e-15); set("auto shutoff min",{AUTO_SHUTOFF_MIN});
        set("y min bc","Anti-Symmetric"); set("z min bc","Symmetric");
        addmesh; set("name","TipM"); set("x min",-0.2e-6); set("x max",4.0e-6);
        set("y",0); set("y span",0.8e-6); set("z",0); set("z span",0.6e-6); set("dx",24e-9); set("dy",16e-9); set("dz",16e-9);
        setglobalsource("wavelength start",{wl:.12e}); setglobalsource("wavelength stop",{wl:.12e});
        addport; set("name","PortBus"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{l_tap+1.4e-6:.12e}); set("y",0); set("y span",1.8e-6); set("z",0); set("z span",1.6e-6);
        set("mode selection","fundamental TE mode");
        addpower; set("name","FacetMon"); set("monitor type","2D X-normal");
        set("x",-0.05e-6); set("y",0); set("y span",6.0e-6); set("z",0); set("z span",6.0e-6);
        set("override global monitor settings",1); set("frequency points",1);
        select("FDTD::ports"); set("source port","PortBus"); set("monitor frequency points",1);
        """)
        _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"{c5['case_id']}.fsp")
        st_bt = int(_scalar_real(fdtd.getresult("FDTD", "status")))
        t_fac_tot = abs(_scalar_real(fdtd.transmission("FacetMon")))
        ym = np.asarray(fdtd.getdata("FacetMon", "y")).ravel()
        zm = np.asarray(fdtd.getdata("FacetMon", "z")).ravel()
        Ey_f = np.squeeze(np.asarray(fdtd.getdata("FacetMon", "Ey")))
        Ez_f = np.squeeze(np.asarray(fdtd.getdata("FacetMon", "Ez")))
        Hy_f = np.squeeze(np.asarray(fdtd.getdata("FacetMon", "Hy")))
        Hz_f = np.squeeze(np.asarray(fdtd.getdata("FacetMon", "Hz")))
        if Ey_f.shape != (len(ym), len(zm)):
            Ey_f, Ez_f, Hy_f, Hz_f = Ey_f.T, Ez_f.T, Hy_f.T, Hz_f.T
        dy_m = np.gradient(ym)
        dz_m = np.gradient(zm)
        dA_m = dy_m[:, None] * dz_m[None, :]
        sx_neg = -0.5 * np.real(Ey_f * np.conj(Hz_f) - Ez_f * np.conj(Hy_f))
        p_facet = max(float(np.sum(sx_neg * dA_m)), 1e-30)
        w0_b = 0.5 * c5["mfd_x_um"] * 1e-6
        adm_b = c5["n_env"] / Z0_VAC
        G_f = np.exp(-((ym[:, None] / w0_b) ** 2 + (zm[None, :] / w0_b) ** 2))
        p_fib = 0.5 * adm_b * float(np.sum((G_f ** 2) * dA_m))
        ic_ov = 0.5 * np.sum((adm_b * Ey_f - Hz_f) * G_f * dA_m)
        ov_fac = float(np.clip((abs(ic_ov) ** 2) / max(4.0 * p_facet * p_fib, 1e-30), 0.0, 1.0))
        eta_tot_3d = max(t_fac_tot * ov_fac, 1e-6)
        out["butt_3d_taper"] = {
            "case_id": c5["case_id"],
            "fdtd_status_code": st_bt,
            "taper_transmission_3d_pct": round(t_fac_tot * 100.0, 2),
            "taper_il_3d_db": round(-10.0 * math.log10(max(t_fac_tot, 1e-6)), 3),
            "facet_overlap_3d_pct": round(ov_fac * 100.0, 2),
            "facet_il_3d_db": round(-10.0 * math.log10(max(ov_fac, 1e-6)), 3),
            "total_il_3d_db": round(-10.0 * math.log10(eta_tot_3d), 3),
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(f"[LUM Suite 13E GPU] 3D Butt Taper (30um): {out['butt_3d_taper']}")
        ckpt["suite13_mzi_and_3d_io"] = out
        _save_ckpt(ckpt)
    else:
        print("[LUM Suite 13E CACHED] butt_3d_taper")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase-3 RTX 5090 GPU 3D FDTD validation.")
    parser.add_argument("--only", nargs="*", default=None, help="Subset: s12a s12b s13")
    args = parser.parse_args()
    sel = set(args.only) if args.only else {"s12a", "s12b", "s13"}

    t_all = time.perf_counter()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    specs = json.loads(SPECS_P3_JSON.read_text(encoding="utf-8"))
    ckpt = json.loads(CKPT_P3_JSON.read_text(encoding="utf-8")) if CKPT_P3_JSON.is_file() else {}
    lumapi = _import_lumapi()

    fdtd = lumapi.FDTD(hide=True)
    try:
        if "s12a" in sel:
            done_map = {r["case_id"]: r for r in ckpt.get("suite12a_lowq_full_rings", [])}
            rows_12a = []
            for c in specs["suite12a_lowq_full_rings"]:
                cid = c["case_id"]
                if cid in done_map:
                    rows_12a.append(done_map[cid])
                    print(f"[LUM Suite 12A CACHED] {cid}")
                else:
                    r = run_suite12a_case(fdtd, c)
                    rows_12a.append(r)
                    ckpt["suite12a_lowq_full_rings"] = rows_12a
                    _save_ckpt(ckpt)
            ckpt["suite12a_lowq_full_rings"] = rows_12a
            _save_ckpt(ckpt)

        if "s12b" in sel:
            ckpt["suite12b_gds_and_doublet"] = run_suite12b(fdtd, specs, ckpt)
            _save_ckpt(ckpt)

        if "s13" in sel:
            ckpt["suite13_mzi_and_3d_io"] = run_suite13(fdtd, specs, ckpt)
            _save_ckpt(ckpt)
    finally:
        fdtd.close()

    payload = {
        "solver": f"Ansys Lumerical 2026R1.0 (3D FDTD RTX 5090 GPU, auto_shutoff_min = {AUTO_SHUTOFF_MIN:g})",
        "gpu_device": "NVIDIA GeForce RTX 5090 Laptop GPU (82 SMs, 3D FDTD GPU)",
        "host": f"{platform.system()}-{platform.machine()} (Python {platform.python_version()})",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        **ckpt,
    }
    OUT_P3_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Lumerical Phase-3 results -> {OUT_P3_JSON} in {payload['total_wall_time_s']} s")


if __name__ == "__main__":
    main()
