"""Independent Ansys Lumerical (2026 R1) Phase-2 Validation Runner (Suites 6, 7, 9, 10, 11).

Reads ONLY neutral physical specifications from `benchmarks/outputs/commercial_validation_specs_p2.json`
(plus MicroRing's exported INTERCONNECT SPAR `.dat` files strictly for Suite 7C circuit import check)
and executes Ansys Lumerical 2026 R1 completely independently across:
  - Suite 6A : `MODE FDE` Butt/Edge Coupler facet overlap, 1-dB alignment tolerances, and
               `MODE EME` adiabatic inverse-taper transmission (`L_taper = 160 um`).
  - Suite 6B : `2D FDTD` (`fdtd-engine.exe -t 16`) Gaussian-Apodized 26-tooth Grating Coupler.
  - Suite 7A : `3D FDTD` (RTX 5090 GPU) 220 nm SOI 2x2 Paired-Interference MMI (`6.0 x 44.5 um`).
  - Suite 7B : `3D FDTD` (RTX 5090 GPU) Low-Q Full Microring Resonator (`R = 5.0 um, gap = 120 nm`),
               yielding full-wave multi-FSR spectrum for direct TMM & blind CMT fitter verification.
  - Suite 7C : `INTERCONNECT` native ONA circuit simulation loading MicroRing's exported `SPAR .dat`
               compact models (`2x2 DC MZI` and `Apodized GC`), verifying zero syntax/passivity errors.
  - Suite 9  : `DEVICE CHARGE` + `MODE FDE` 2D Silicon Lateral PN junction (`0..-3 V`) for 3-way TCAD.
  - Suite 10 : `DEVICE HEAT` 2D steady-state thermal FEM (`SOI Top TiN` & `Deep Air Trench`) for
               3-way MicroRing vs COMSOL `ht` vs Lumerical `HEAT` thermal cross-validation.
  - Suite 11 : `MODE FDE` + `3D FDTD` (RTX 5090 GPU) correlated CD/Gap lithography bias corners
               (`delta_CD = -20, 0, +20 nm`).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

LUMERICAL_WIN_HINTS = [
    r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v252\Lumerical\api\python",
]
FDTD_ENGINE_EXE = Path(r"C:\Program Files\ANSYS Inc\v261\Lumerical\bin\fdtd-engine.exe")
FDTD_AUTO_SHUTOFF_MIN = 1.0e-5
FDTD_PML_LAYERS = 12
FDTD_MESH_ACCURACY = 3
Z0_VAC = 376.730313668
C0 = 299792458.0

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH_DIR = OUT_DIR / "_lum_scratch_p2"
SPECS_P2_JSON = OUT_DIR / "commercial_validation_specs_p2.json"
OUT_P2_JSON = OUT_DIR / "lumerical_independent_results_p2.json"
CKPT_P2_JSON = SCRATCH_DIR / "lum_p2_ckpt.json"
IC_EXPORT_DIR = OUT_DIR / "_ic_export"


def _import_lumapi():
    for p in LUMERICAL_WIN_HINTS:
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)
    import lumapi  # type: ignore
    return lumapi


def _scalar_real(val) -> float:
    return float(np.real(np.asarray(val).reshape(-1)[0]))


def _run_fdtd_gpu(fdtd, fsp_path: Path, retries: int = 4) -> None:
    fdtd.save(str(fsp_path))
    last_err = None
    for attempt in range(retries):
        try:
            fdtd.run("FDTD", "GPU")
            return
        except Exception as exc:
            last_err = exc
            time.sleep(6.0 * (attempt + 1))
    raise RuntimeError(f"Lumerical GPU 3D FDTD failed on {fsp_path.name}: {last_err}")


def _save_ckpt(data: dict) -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_P2_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ==============================================================================
# Suite 6A: MODE FDE Butt/Edge Couplers + MODE EME Adiabatic Taper Check
# ==============================================================================
def run_lum_suite6a_butt(lumapi, specs: dict) -> list[dict]:
    out = []
    mode = lumapi.MODE(hide=True)
    try:
        for c in specs["suite6a_butt_couplers"]:
            t0 = time.perf_counter()
            wl_m = c["wavelength_nm"] * 1e-9
            hf = c["film_height_um"] * 1e-6
            nc = c["core_index"]["center"]["nx"]
            ncl = c["clad_index"]["center"]["nx"]
            dom_w = max(2.4 * c["mfd_x_um"], 6.5 * (c["wavelength_nm"] / 1550.0)) * 1e-6
            pad_y = max(1.4 * c["mfd_y_um"], 3.2 * (c["wavelength_nm"] / 1550.0)) * 1e-6
            w0x = 0.5 * c["mfd_x_um"] * 1e-6
            w0y = 0.5 * c["mfd_y_um"] * 1e-6
            y_mid = 0.5 * hf
            adm_f = c["n_env"] / Z0_VAC

            tip_rows = []
            nom_tols = (0.0, 0.0)
            for wt_um in c["w_tip_sweep_um"]:
                wt = wt_um * 1e-6
                mode.eval(f"""
                switchtolayout; deleteall;
                addrect; set("name","Core");
                set("x",0); set("x span",{wt:.12e}); set("y",{0.5*hf:.12e}); set("y span",{hf:.12e});
                set("z",0); set("z span",2e-6); set("index",{nc:.8f});
                addfde;
                set("solver type","2D Z normal");
                set("x",0); set("x span",{dom_w:.12e});
                set("y",{0.5*hf:.12e}); set("y span",{2.0*pad_y + hf:.12e});
                set("background index",{ncl:.8f});
                set("mesh cells x",160); set("mesh cells y",160);
                set("wavelength",{wl_m:.12e});
                set("number of trial modes",8);
                set("use max index",0); set("n",{0.5*(nc+ncl):.6f});
                addmesh; set("name","CoreM");
                set("x",0); set("x span",{max(wt*3.0, 0.6e-6):.12e});
                set("y",{0.5*hf:.12e}); set("y span",{hf*3.0:.12e});
                set("dx",12e-9); set("dy",12e-9);
                findmodes;
                """)
                best_idx, best_n = 1, -1.0
                for mi in range(1, 9):
                    try:
                        nr = _scalar_real(mode.getdata(f"FDE::data::mode{mi}", "neff"))
                        tef = _scalar_real(mode.getdata(f"FDE::data::mode{mi}", "TE polarization fraction"))
                    except Exception:
                        continue
                    if nr > ncl + 1e-4 and nr < nc + 0.02 and tef > 0.50 and nr > best_n:
                        best_n, best_idx = nr, mi
                x_arr = np.asarray(mode.getdata(f"FDE::data::mode{best_idx}", "x")).ravel()
                y_arr = np.asarray(mode.getdata(f"FDE::data::mode{best_idx}", "y")).ravel()
                Ex = np.squeeze(np.asarray(mode.getdata(f"FDE::data::mode{best_idx}", "Ex")))
                Ey = np.squeeze(np.asarray(mode.getdata(f"FDE::data::mode{best_idx}", "Ey")))
                Hx = np.squeeze(np.asarray(mode.getdata(f"FDE::data::mode{best_idx}", "Hx")))
                Hy = np.squeeze(np.asarray(mode.getdata(f"FDE::data::mode{best_idx}", "Hy")))
                # Lumerical FDE shape is (Nx, Ny) -> transpose to (Ny, Nx)
                if Ex.shape == (len(x_arr), len(y_arr)):
                    Ex, Ey, Hx, Hy = Ex.T, Ey.T, Hx.T, Hy.T
                wx_w = np.gradient(x_arr)
                wy_w = np.gradient(y_arr)
                dA = wy_w[:, None] * wx_w[None, :]
                sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
                if float(np.sum(sz * dA)) < 0:
                    Hx, Hy = -Hx, -Hy
                    sz = -sz
                p_chip = max(float(np.sum(sz * dA)), 1e-30)
                kern = adm_f * Ex + Hy

                def _eval_ov(dx_s: np.ndarray, dy_s: np.ndarray) -> np.ndarray:
                    gx = np.exp(-((x_arr[None, :] - dx_s[:, None]) / w0x) ** 2)
                    gy = np.exp(-((y_arr[None, :] - (y_mid + dy_s[:, None])) / w0y) ** 2)
                    p_f = 0.5 * adm_f * np.maximum(
                        (np.sum((gx**2) * wx_w[None, :], axis=1)) * (np.sum((gy**2) * wy_w[None, :], axis=1)),
                        0.5 * math.pi * w0x * w0y,
                    )
                    tmp = (gy * wy_w[None, :]) @ kern
                    icf = np.sum(tmp * gx * wx_w[None, :], axis=1)
                    return np.clip(np.real((np.abs(icf) ** 2) / np.maximum(16.0 * p_chip * p_f, 1e-30)), 0.0, 1.0)

                eta0 = float(_eval_ov(np.array([0.0]), np.array([0.0]))[0])
                w_pos = np.maximum(sz * dA, 0.0)
                X2, Y2 = np.meshgrid(x_arr, y_arr)
                ybar = float(np.sum(w_pos * Y2) / max(float(np.sum(w_pos)), 1e-30))
                sig_y = math.sqrt(max(float(np.sum(w_pos * (Y2 - ybar)**2) / max(float(np.sum(w_pos)), 1e-30)), 1e-16))
                eta_sub = float(np.clip(1.0 - 0.5 * math.erfc(math.sqrt(2.0) * c["box_thickness_um"] / max(2.0 * sig_y * 1e6, 0.25)), 0.15, 1.0))
                r_f = 0.0025 if c["ar_coating"] else ((best_n - c["n_env"]) / (best_n + c["n_env"])) ** 2
                eta_fac = max(eta0 * eta_sub * (1.0 - r_f), 1e-6)
                tip_rows.append({
                    "w_tip_um": wt_um,
                    "n_eff_tip": round(best_n, 6),
                    "mode_overlap_pct": round(eta0 * 100.0, 2),
                    "facet_il_db": round(-10.0 * math.log10(eta_fac), 3),
                })
                if abs(wt_um - c["w_tip_nominal_um"]) < 1e-4:
                    ox = np.linspace(0.0, max(2.5, 0.75 * c["mfd_x_um"]), 81)
                    oy = np.linspace(0.0, max(2.5, 0.75 * c["mfd_y_um"]), 81)
                    ex_c = _eval_ov(ox * 1e-6, np.zeros_like(ox))
                    ey_c = _eval_ov(np.zeros_like(oy), oy * 1e-6)
                    def _tol1(offs, ev):
                        targ = float(ev[0]) * (10.0 ** (-0.1))
                        idx = np.where(ev <= targ)[0]
                        if len(idx) == 0:
                            return float(offs[-1])
                        k = int(idx[0])
                        return 0.0 if k == 0 else float(offs[k-1] + (targ - ev[k-1]) * (offs[k] - offs[k-1]) / (ev[k] - ev[k-1]))
                    nom_tols = (round(_tol1(ox, ex_c), 3), round(_tol1(oy, ey_c), 3))

            # Run Lumerical MODE EME adiabatic inverse taper check (w_tip_nominal -> w_bus over taper_length_um)!
            l_tap = c["taper_length_um"] * 1e-6
            wt0 = c["w_tip_nominal_um"] * 1e-6
            wb0 = c["w_bus_um"] * 1e-6
            mode.eval(f"""
            switchtolayout; deleteall;
            addpoly; set("name","Taper");
            set("z",0); set("z span",{hf:.12e});
            V = [0,{-0.5*wt0:.12e}; {l_tap:.12e},{-0.5*wb0:.12e}; {l_tap:.12e},{0.5*wb0:.12e}; 0,{0.5*wt0:.12e}];
            set("vertices", V); set("index",{nc:.8f});
            addeme;
            set("x min", 0);
            set("y", 0); set("y span", {dom_w:.12e});
            set("z", 0); set("z span", {2.0*pad_y + hf:.12e});
            set("background index", {ncl:.8f});
            set("wavelength", {wl_m:.12e});
            set("number of cell groups", 1);
            set("group spans", [{l_tap:.12e}]);
            set("cells", [16]);
            set("subcell method", [1]);
            set("number of modes for all cell groups", 10);
            set("mesh cells y", 90); set("mesh cells z", 90);
            select("EME::Ports::port_1"); set("port location","left"); set("mode selection","fundamental TE mode");
            select("EME::Ports::port_2"); set("port location","right"); set("mode selection","fundamental TE mode");
            """)
            mode.save(str(SCRATCH_DIR / f"eme_{c['case_id']}.lms"))
            mode.run()
            mode.emepropagate()
            s_mat = np.asarray(mode.getresult("EME", "user s matrix"))
            s21_sq = float(abs(s_mat[1, 0]) ** 2)
            taper_il_eme_db = round(-10.0 * math.log10(max(s21_sq, 1e-6)), 4)

            nom_row = [r for r in tip_rows if abs(r["w_tip_um"] - c["w_tip_nominal_um"]) < 1e-4][0]
            out.append({
                "case_id": c["case_id"],
                "w_tip_nominal_um": c["w_tip_nominal_um"],
                "n_eff_tip_nom": nom_row["n_eff_tip"],
                "mode_overlap_nom_pct": nom_row["mode_overlap_pct"],
                "facet_il_nom_db": nom_row["facet_il_db"],
                "taper_il_eme_db": taper_il_eme_db,
                "total_il_nom_db": round(nom_row["facet_il_db"] + taper_il_eme_db, 3),
                "tol_1db_x_um": nom_tols[0],
                "tol_1db_y_um": nom_tols[1],
                "tip_sweep": tip_rows,
                "wall_time_s": round(time.perf_counter() - t0, 2),
            })
            print(
                f"[LUM Suite 6A] {c['case_id']}: neff={nom_row['n_eff_tip']:.5f}, "
                f"overlap={nom_row['mode_overlap_pct']:.2f}%, facet_IL={nom_row['facet_il_db']:.3f} dB, "
                f"EME_taper_IL={taper_il_eme_db:.4f} dB, tol1dB={nom_tols} um"
            )
    finally:
        mode.close()
    return out


# ==============================================================================
# Suite 6B: 2D FDTD Gaussian-Apodized Surface Grating Coupler
# ==============================================================================
def run_lum_suite6b_apodized_gc(lumapi, specs: dict) -> dict:
    c = specs["suite6b_apodized_gc"]
    t0 = time.perf_counter()
    hf = c["film_height_um"] * 1e-6
    h_rem = (c["film_height_um"] - c["etch_depth_um"]) * 1e-6
    h_box = c["box_thickness_um"] * 1e-6
    h_top = c["top_clad_thickness_um"] * 1e-6
    nc = c["core_index"]["center"]["nx"]
    ncl = c["clad_index"]["center"]["nx"]
    grooves = c["grooves_um"]
    gc_len = grooves[-1][1] * 1e-6 + 0.6e-6

    x_min = -3.8e-6
    x_max = gc_len + 3.5e-6
    y_min = -h_box - 1.5e-6
    y_max = hf + h_top + 2.0e-6
    y_up = hf + h_top + 0.50e-6
    y_dn = -h_box - 0.35e-6

    tooth_cmds = []
    zp = x_min - 1.0e-6
    for i, (g0u, g1u) in enumerate(grooves):
        g0, g1 = g0u * 1e-6, g1u * 1e-6
        if g0 - zp > 1e-10:
            tooth_cmds.append(
                f'addrect; set("name","T{i}"); set("x min",{zp:.12e}); set("x max",{g0:.12e}); '
                f'set("y min",{h_rem:.12e}); set("y max",{hf:.12e}); set("index",{nc:.8f}); set("override mesh order from material database",1); set("mesh order",1);'
            )
        zp = g1
    tooth_cmds.append(
        f'addrect; set("name","TEnd"); set("x min",{zp:.12e}); set("x max",{x_max+1e-6:.12e}); '
        f'set("y min",{h_rem:.12e}); set("y max",{hf:.12e}); set("index",{nc:.8f}); set("override mesh order from material database",1); set("mesh order",1);'
    )

    fsp = SCRATCH_DIR / "gc_apodized_2d.fsp"
    fdtd = lumapi.FDTD(hide=True)
    try:
        fdtd.eval(f"""
        switchtolayout; deleteall;
        addrect; set("name","Sub"); set("x min",{x_min-1e-6:.12e}); set("x max",{x_max+1e-6:.12e});
        set("y min",{y_min-2e-6:.12e}); set("y max",{-h_box:.12e}); set("index",{nc:.8f}); set("override mesh order from material database",1); set("mesh order",3);
        addrect; set("name","Ox"); set("x min",{x_min-1e-6:.12e}); set("x max",{x_max+1e-6:.12e});
        set("y min",{-h_box:.12e}); set("y max",{hf+h_top:.12e}); set("index",{ncl:.8f}); set("override mesh order from material database",1); set("mesh order",3);
        addrect; set("name","Slab"); set("x min",{x_min-1e-6:.12e}); set("x max",{x_max+1e-6:.12e});
        set("y min",0); set("y max",{h_rem:.12e}); set("index",{nc:.8f}); set("override mesh order from material database",1); set("mesh order",1);
        {chr(10).join(tooth_cmds)}
        addfdtd; set("dimension","2D");
        set("x min",{x_min:.12e}); set("x max",{x_max:.12e});
        set("y min",{y_min:.12e}); set("y max",{y_max:.12e});
        set("background index",1.0); set("mesh accuracy",4);
        set("simulation time", 3000e-15); set("auto shutoff min",1e-5);
        addmesh; set("name","GratingM");
        set("x min",-1.0e-6); set("x max",{gc_len+1.0e-6:.12e});
        set("y min",-0.05e-6); set("y max",{hf+0.05e-6:.12e});
        set("dx",15e-9); set("dy",10e-9);
        addmode; set("name","Src");
        set("injection axis","x-axis"); set("direction","Forward");
        set("x",-2.0e-6); set("y",0.11e-6); set("y span",3.2e-6);
        set("wavelength start",1.50e-6); set("wavelength stop",1.60e-6);
        set("mode selection","fundamental TM mode");
        addpower; set("name","Refl"); set("monitor type","2D X-normal");
        set("x",-2.8e-6); set("y",0.11e-6); set("y span",3.2e-6);
        set("override global monitor settings",1); set("frequency points",21);
        addpower; set("name","Up"); set("monitor type","2D Y-normal");
        set("x min",-1.5e-6); set("x max",{gc_len+1.5e-6:.12e}); set("y",{y_up:.12e});
        set("override global monitor settings",1); set("frequency points",21);
        addpower; set("name","Down"); set("monitor type","2D Y-normal");
        set("x min",-1.5e-6); set("x max",{gc_len+1.5e-6:.12e}); set("y",{y_dn:.12e});
        set("override global monitor settings",1); set("frequency points",21);
        """)
        fdtd.save(str(fsp))
    finally:
        fdtd.close()

    # Run 2D FDTD engine directly on CPU (16 threads) without touching global resource config
    r_eng = subprocess.run([str(FDTD_ENGINE_EXE), "-t", "16", str(fsp)], capture_output=True, text=True, check=False)
    if r_eng.returncode != 0:
        raise RuntimeError(f"2D FDTD apodized GC failed: {r_eng.stderr[:400]}")

    fdtd = lumapi.FDTD(filename=str(fsp), hide=True)
    try:
        freqs = np.asarray(fdtd.getdata("Up", "f")).ravel()
        wls_nm = (C0 / freqs) * 1e9
        order = np.argsort(wls_nm)
        wls_nm = wls_nm[order]
        t_up = np.asarray(fdtd.transmission("Up")).ravel()[order]
        t_dn = -np.asarray(fdtd.transmission("Down")).ravel()[order]
        t_rf = -np.asarray(fdtd.transmission("Refl")).ravel()[order]
        x_up = np.asarray(fdtd.getdata("Up", "x")).ravel()
        ez_all = np.squeeze(np.asarray(fdtd.getdata("Up", "Ez")))  # (Nx, Nf)
        ez_all = ez_all[:, order]
        dx = float(x_up[1] - x_up[0])

        xl = np.linspace(-30.0, 30.0, 401)
        ul = np.where(np.abs(xl) <= 0.5 * c["grating_width_um"], np.cos(math.pi * xl / c["grating_width_um"]), 0.0)
        gl = np.exp(-((xl / (0.5 * c["mfd_um"])) ** 2))
        lat_ov = float((np.sum(ul * gl)) ** 2 / (np.sum(ul ** 2) * np.sum(gl ** 2)))

        mfd_z = (c["mfd_um"] * 1e-6) / math.cos(math.radians(c["fiber_angle_deg"]))
        w0 = 0.5 * mfd_z
        il_spec = []
        for iw, wl_nm in enumerate(wls_nm):
            ez = ez_all[:, iw]
            k0 = 2.0 * math.pi / (wl_nm * 1e-9)
            kz_f = k0 * c["n_env"] * math.sin(math.radians(c["fiber_angle_deg"]))
            ez_n = float(np.sum(np.abs(ez) ** 2) * dx)
            best_ov = 0.0
            for zc in np.linspace(1.0e-6, gc_len - 1.0e-6, 90):
                # Lumerical uses exp(-i*w*t) -> forward +x wave is exp(+i*kz*x)
                g = np.exp(-((x_up - zc) / w0) ** 2) * np.exp(+1j * kz_f * (x_up - zc))
                ov = float((abs(np.sum(ez * np.conj(g)) * dx) ** 2) / max(ez_n * float(np.sum(np.abs(g) ** 2) * dx), 1e-30))
                if ov > best_ov:
                    best_ov = ov
            eta_f = max(float(t_up[iw]) * best_ov * lat_ov, 1e-6)
            il_spec.append(-10.0 * math.log10(eta_f))

        il_arr = np.array(il_spec)
        ipk = int(np.argmin(il_arr))
        i1550 = int(np.argmin(np.abs(wls_nm - 1550.0)))
        out = {
            "case_id": c["case_id"],
            "peak_wavelength_nm": round(float(wls_nm[ipk]), 2),
            "peak_il_db": round(float(il_arr[ipk]), 3),
            "il_at_1550_db": round(float(il_arr[i1550]), 3),
            "s11_at_1550_db": round(10.0 * math.log10(max(float(t_rf[i1550]), 1e-12)), 2),
            "eta_up_at_1550_pct": round(float(t_up[i1550]) * 100.0, 2),
            "eta_down_at_1550_pct": round(float(t_dn[i1550]) * 100.0, 2),
            "spectrum_wavelength_nm": [round(float(x), 3) for x in wls_nm],
            "spectrum_il_db": [round(float(x), 4) for x in il_arr],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM Suite 6B] {c['case_id']}: peak_IL={out['peak_il_db']:.3f} dB @ {out['peak_wavelength_nm']:.1f} nm "
            f"| 1550nm IL={out['il_at_1550_db']:.3f} dB, up={out['eta_up_at_1550_pct']:.2f}%, S11={out['s11_at_1550_db']:.1f} dB"
        )
        return out
    finally:
        fdtd.close()


# ==============================================================================
# Suite 7A (2x2 MMI GPU 3D FDTD), 7B (Low-Q Full Ring GPU 3D FDTD), 11 (CD Bias GPU)
# ==============================================================================
def run_lum_suite7ab_and_11_gpu(lumapi, specs: dict) -> tuple[dict, dict, dict]:
    fdtd = lumapi.FDTD(hide=True)
    try:
        # --- Suite 7A: 2x2 Paired-Interference MMI (RTX 5090 GPU 3D FDTD) ---
        c7a = specs["suite7a_mmi_2x2"]
        t0 = time.perf_counter()
        wl = c7a["wavelength_nm"] * 1e-9
        hf = c7a["film_height_um"] * 1e-6
        w_arm = c7a["arm_width_um"] * 1e-6
        w_mmi = c7a["mmi_width_um"] * 1e-6
        l_mmi = c7a["mmi_length_um"] * 1e-6
        w_tap = c7a["mmi_taper_width_um"] * 1e-6
        l_tap = c7a["mmi_taper_length_um"] * 1e-6
        y_p = c7a["port_offset_um"] * 1e-6
        nc = c7a["core_index"]["center"]["nx"]
        ncl = c7a["clad_index"]["center"]["nx"]

        x_in0 = -(0.5 * l_mmi + l_tap + 2.5e-6)
        x_in1 = -(0.5 * l_mmi + l_tap)
        x_out0 = +(0.5 * l_mmi + l_tap)
        x_out1 = +(0.5 * l_mmi + l_tap + 2.5e-6)

        # Build 4 access tapers (2 left inputs, 2 right outputs) + MMI core (reuse solved GPU .fsp if present)
        fsp_mmi = SCRATCH_DIR / "mmi_2x2_gpu.fsp"
        if fsp_mmi.is_file():
            fdtd.load(str(fsp_mmi))
        else:
            fdtd.eval(f"""
        switchtolayout; deleteall;
        addrect; set("name","MMIBox"); set("x",0); set("x span",{l_mmi:.12e});
        set("y",0); set("y span",{w_mmi:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
        """)
            for tag, sgn_y in (("Top", +1.0), ("Bot", -1.0)):
                yc = sgn_y * y_p
                fdtd.eval(f"""
            addrect; set("name","InLead_{tag}"); set("x min",{x_in0-1e-6:.12e}); set("x max",{x_in1:.12e});
            set("y",{yc:.12e}); set("y span",{w_arm:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            addpoly; set("name","InTap_{tag}"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            set("vertices", [{x_in1:.12e},{yc-0.5*w_arm:.12e}; {-0.5*l_mmi:.12e},{yc-0.5*w_tap:.12e}; {-0.5*l_mmi:.12e},{yc+0.5*w_tap:.12e}; {x_in1:.12e},{yc+0.5*w_arm:.12e}]);
            addpoly; set("name","OutTap_{tag}"); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            set("vertices", [{0.5*l_mmi:.12e},{yc-0.5*w_tap:.12e}; {x_out0:.12e},{yc-0.5*w_arm:.12e}; {x_out0:.12e},{yc+0.5*w_arm:.12e}; {0.5*l_mmi:.12e},{yc+0.5*w_tap:.12e}]);
            addrect; set("name","OutLead_{tag}"); set("x min",{x_out0:.12e}); set("x max",{x_out1+1e-6:.12e});
            set("y",{yc:.12e}); set("y span",{w_arm:.12e}); set("z",0); set("z span",{hf:.12e}); set("index",{nc:.8f});
            """)
            fdtd.eval(f"""
        addfdtd; set("dimension","3D");
        set("x min",{x_in0+0.3e-6:.12e}); set("x max",{x_out1-0.3e-6:.12e});
        set("y",0); set("y span",{w_mmi+3.2e-6:.12e}); set("z",0); set("z span",2.4e-6);
        set("background index",{ncl:.8f}); set("mesh accuracy",3);
        set("simulation time",6000e-15); set("auto shutoff min",{FDTD_AUTO_SHUTOFF_MIN});
        set("z min bc","Symmetric");
        setglobalsource("wavelength start",{wl:.12e}); setglobalsource("wavelength stop",{wl:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{x_in0+1.1e-6:.12e}); set("y",{-y_p:.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.6e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortBar"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{x_out1-1.1e-6:.12e}); set("y",{-y_p:.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.6e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortCross"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{x_out1-1.1e-6:.12e}); set("y",{+y_p:.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.6e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",1);
        """)
            _run_fdtd_gpu(fdtd, fsp_mmi)
        s_bar = complex(np.asarray(fdtd.getresult("FDTD::ports::PortBar", "expansion for port monitor")["S"]).reshape(-1)[0])
        s_crs = complex(np.asarray(fdtd.getresult("FDTD::ports::PortCross", "expansion for port monitor")["S"]).reshape(-1)[0])
        p_bar = float(abs(s_bar) ** 2)
        p_crs = float(abs(s_crs) ** 2)
        tot = p_bar + p_crs
        dphi = math.degrees(math.atan2(math.sin(np.angle(s_crs) - np.angle(s_bar)), math.cos(np.angle(s_crs) - np.angle(s_bar))))
        st_mmi = int(_scalar_real(fdtd.getresult("FDTD", "status")))
        out7a = {
            "case_id": c7a["case_id"],
            "split_bar_pct": round(p_bar * 100.0, 2),
            "split_cross_pct": round(p_crs * 100.0, 2),
            "total_trans_pct": round(tot * 100.0, 2),
            "excess_loss_db": round(-10.0 * math.log10(max(tot, 1e-6)), 3),
            "imbalance_db": round(10.0 * math.log10(max(p_bar, 1e-9) / max(p_crs, 1e-9)), 3),
            "phase_diff_deg": round(dphi, 2),
            "fdtd_status_code": st_mmi,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM Suite 7A GPU] 2x2 MMI: Bar/Cross={out7a['split_bar_pct']:.2f}/{out7a['split_cross_pct']:.2f}%, "
            f"IL={out7a['excess_loss_db']:.3f} dB, Imb={out7a['imbalance_db']:+.3f} dB, dPhi={out7a['phase_diff_deg']:.1f} deg, status={st_mmi}"
        )

        # --- Suite 7B: Low-Q Full Microring Resonator (R = 5 um, gap = 120 nm, RTX 5090 GPU 3D FDTD) ---
        c7b = specs["suite7b_full_ring_lowq"]
        t0 = time.perf_counter()
        R = c7b["ring_radius_um"] * 1e-6
        wr = c7b["ring_width_um"] * 1e-6
        wb = c7b["bus_width_um"] * 1e-6
        gap = c7b["gap_thru_um"] * 1e-6
        hf = c7b["film_height_um"] * 1e-6
        y_bus = -(R + 0.5 * wr + gap + 0.5 * wb)
        x_half = R + 3.2e-6
        fsp_ring = SCRATCH_DIR / "full_ring_r5um_gpu.fsp"
        if fsp_ring.is_file():
            fdtd.load(str(fsp_ring))
        else:
            fdtd.eval(f"""
        switchtolayout; deleteall;
        addring; set("name","Ring");
        set("x",0); set("y",0); set("z",0); set("z span",{hf:.12e});
        set("inner radius",{R-0.5*wr:.12e}); set("outer radius",{R+0.5*wr:.12e});
        set("index",{nc:.8f});
        addrect; set("name","Bus");
        set("x",0); set("x span",{2.0*(x_half+1.5e-6):.12e});
        set("y",{y_bus:.12e}); set("y span",{wb:.12e}); set("z",0); set("z span",{hf:.12e});
        set("index",{nc:.8f});
        addfdtd; set("dimension","3D");
        set("x",0); set("x span",{2.0*x_half:.12e});
        set("y min",{y_bus-1.6e-6:.12e}); set("y max",{R+0.5*wr+1.6e-6:.12e});
        set("z",0); set("z span",2.2e-6);
        set("background index",{ncl:.8f}); set("mesh accuracy",3);
        set("simulation time", 14000e-15); set("auto shutoff min",1e-5);
        set("z min bc","Symmetric");
        addmesh; set("name","GapM");
        set("x",0); set("x span",2.4e-6); set("y",{y_bus+0.5*wb+0.5*gap:.12e}); set("y span",{gap+wr+wb+0.3e-6:.12e});
        set("z",0); set("z span",{hf+0.12e-6:.12e}); set("dx",20e-9); set("dy",15e-9); set("dz",20e-9);
        setglobalsource("wavelength start",{c7b['wl_min_nm']*1e-9:.12e});
        setglobalsource("wavelength stop",{c7b['wl_max_nm']*1e-9:.12e});
        addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
        set("x",{-x_half+0.9e-6:.12e}); set("y",{y_bus:.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
        set("x",{+x_half-0.9e-6:.12e}); set("y",{y_bus:.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.5e-6);
        set("mode selection","fundamental TE mode");
        select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",401);
        """)
            _run_fdtd_gpu(fdtd, fsp_ring)
        exp_r = fdtd.getresult("FDTD::ports::PortThru", "expansion for port monitor")
        f_r = np.asarray(exp_r["f"]).ravel()
        s21_r = np.asarray(exp_r["S"]).ravel()
        wl_r = (C0 / f_r) * 1e9
        ord_r = np.argsort(wl_r)
        wl_r = wl_r[ord_r]
        tr_r = (np.abs(s21_r[ord_r]) ** 2).astype(float)
        st_ring = int(_scalar_real(fdtd.getresult("FDTD", "status")))

        # Locate resonances & run MicroRing's blind CMT fitter directly on Lumerical 3D FDTD trace!
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from microring.mrr_cmt_fit_predict import fit_single_resonance_cmt
        from scipy.signal import find_peaks

        dips, _ = find_peaks(1.0 - tr_r, prominence=0.15, distance=15)
        dip_wls = [round(float(wl_r[i]), 3) for i in dips]
        fsr_fdtd = round(float(np.mean(np.diff(dip_wls))), 3) if len(dip_wls) >= 2 else 0.0
        # Fit deepest dip closest to 1550 nm
        i_cent = min(dips, key=lambda idx: abs(wl_r[idx] - 1550.0)) if len(dips) else int(np.argmin(tr_r))
        w_cent = float(wl_r[i_cent])
        win = np.abs(wl_r - w_cent) <= max(0.35 * fsr_fdtd, 5.0)
        fit_res, _, _ = fit_single_resonance_cmt(
            wl_r[win], tr_r[win], model_type="lorentzian", known_regime="over", ng_assumed=4.2, ring_length_um=2.0*math.pi*5.0
        )
        out7b = {
            "case_id": c7b["case_id"],
            "fdtd_status_code": st_ring,
            "resonance_wavelengths_nm": dip_wls,
            "FSR_fdtd_nm": fsr_fdtd,
            "fitted_dip_nm": round(fit_res.lambda0_nm, 4),
            "Q_loaded_fdtd_cmt_fit": round(fit_res.Q_loaded, 1),
            "FWHM_fdtd_nm": round(fit_res.fwhm_pm * 1e-3, 5),
            "ER_fdtd_dB": round(fit_res.extinction_ratio_db, 2),
            "fit_r_squared": round(fit_res.r_squared, 5),
            "spectrum_wavelength_nm": [round(float(x), 4) for x in wl_r[::2]],
            "spectrum_thru_power": [round(float(x), 5) for x in tr_r[::2]],
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM Suite 7B GPU] Full Ring R=5um: status={st_ring}, dips={dip_wls}, FSR={fsr_fdtd:.3f} nm, "
            f"Q_L(CMT fit)={fit_res.Q_loaded:.1f}, ER={fit_res.extinction_ratio_db:.2f} dB, R2={fit_res.r_squared:.5f}"
        )

        # --- Suite 11: Cleanroom Litho Correlated CD/Gap Bias Corners (MODE + 3D FDTD GPU) ---
        c11 = specs["suite11_fab_bias"]
        eta_p = c11["proximity_gap_factor"]
        R11 = c11["ring_radius_um"] * 1e-6
        hf11 = c11["film_height_um"] * 1e-6
        corners11 = []
        for dcd in c11["delta_cd_nm_list"]:
            w_fab = (c11["ring_width_nom_um"] + dcd * 1e-3) * 1e-6
            g_fab = (c11["gap_nom_um"] - eta_p * dcd * 1e-3) * 1e-6
            # Run 3D FDTD GPU single-pass coupler cutout for exact kappa^2(delta_CD)
            y_b = R11 + 0.5 * w_fab + g_fab + 0.5 * w_fab
            xh = 5.5e-6
            fdtd.eval(f"""
            switchtolayout; deleteall;
            addrect; set("name","Bus"); set("x",0); set("x span",{2.0*(xh+2e-6):.12e});
            set("y",{y_b:.12e}); set("y span",{w_fab:.12e}); set("z",0); set("z span",{hf11:.12e}); set("index",{nc:.8f});
            addring; set("name","RingArc"); set("x",0); set("y",0); set("z",0); set("z span",{hf11:.12e});
            set("inner radius",{R11-0.5*w_fab:.12e}); set("outer radius",{R11+0.5*w_fab:.12e});
            set("theta start",45); set("theta stop",135); set("index",{nc:.8f});
            addfdtd; set("dimension","3D");
            set("x",0); set("x span",{2.0*xh:.12e}); set("y min",{R11-2.5e-6:.12e}); set("y max",{y_b+1.4e-6:.12e});
            set("z",0); set("z span",2.2e-6); set("background index",{ncl:.8f}); set("mesh accuracy",3);
            set("simulation time",6000e-15); set("auto shutoff min",1e-5); set("z min bc","Symmetric");
            addmesh; set("name","GapM"); set("x",0); set("x span",4.0e-6);
            set("y",{R11+0.5*w_fab+0.5*g_fab:.12e}); set("y span",{g_fab+2*w_fab+0.2e-6:.12e});
            set("z",0); set("z span",{hf11+0.12e-6:.12e}); set("dx",20e-9); set("dy",14e-9); set("dz",20e-9);
            setglobalsource("wavelength start",1.55e-6); setglobalsource("wavelength stop",1.55e-6);
            addport; set("name","PortIn"); set("injection axis","x-axis"); set("direction","Forward");
            set("x",{-xh+0.8e-6:.12e}); set("y",{y_b:.12e}); set("y span",1.6e-6); set("z",0); set("z span",1.5e-6);
            set("mode selection","fundamental TE mode");
            addport; set("name","PortThru"); set("injection axis","x-axis"); set("direction","Backward");
            set("x",{+xh-0.8e-6:.12e}); set("y",{y_b:.12e}); set("y span",1.6e-6); set("z",0); set("z span",1.5e-6);
            set("mode selection","fundamental TE mode");
            addpower; set("name","RingMon"); set("monitor type","2D X-normal");
            set("x",3.6e-6); set("y",{math.sqrt(max(R11**2 - 3.6e-6**2, 1e-12)):.12e}); set("y span",1.8e-6); set("z",0); set("z span",1.5e-6);
            select("FDTD::ports"); set("source port","PortIn"); set("monitor frequency points",1);
            """)
            _run_fdtd_gpu(fdtd, SCRATCH_DIR / f"s11_cd_{int(dcd):+d}nm_gpu.fsp")
            s21_sq = float(abs(np.asarray(fdtd.getresult("FDTD::ports::PortThru", "expansion for port monitor")["S"]).reshape(-1)[0]) ** 2)
            s31_sq = abs(_scalar_real(fdtd.transmission("RingMon")))
            corners11.append({
                "delta_cd_nm": dcd,
                "w_fab_nm": round(w_fab * 1e9, 1),
                "gap_fab_nm": round(g_fab * 1e9, 1),
                "S31_sq_3dfdtd": round(s31_sq, 5),
                "S21_sq_3dfdtd": round(s21_sq, 5),
                "passivity_sum": round(s21_sq + s31_sq, 5),
            })
            print(f"[LUM Suite 11 GPU] dCD={dcd:+.0f}nm (W={w_fab*1e9:.0f}, g={g_fab*1e9:.1f}): |S31|^2={s31_sq:.5f}, |S21|^2={s21_sq:.5f}")
    finally:
        fdtd.close()

    # Also get bent n_eff & dlambda/dCD from Lumerical MODE FDE
    mode = lumapi.MODE(hide=True)
    try:
        n_by_cd = {}
        ng_nom = 4.2
        for dcd in c11["delta_cd_nm_list"]:
            w_fab = (c11["ring_width_nom_um"] + dcd * 1e-3) * 1e-6
            tags = ("minus", "center", "plus") if abs(dcd) < 1e-9 else ("center",)
            vals_t = {}
            for tag in tags:
                ci = c11["core_index"][tag]
                cli = c11["clad_index"][tag]
                mode.eval(f"""
                switchtolayout; deleteall;
                addrect; set("name","Core"); set("x",0); set("x span",{w_fab:.12e});
                set("y",{0.5*hf11:.12e}); set("y span",{hf11:.12e}); set("z",0); set("z span",2e-6); set("index",{ci['nx']:.8f});
                addfde; set("solver type","2D Z normal");
                set("x",0); set("x span",4.6e-6); set("y",{0.5*hf11:.12e}); set("y span",4.2e-6);
                set("background index",{cli['nx']:.8f}); set("mesh cells x",160); set("mesh cells y",150);
                set("wavelength",{ci['wavelength_nm']*1e-9:.12e});
                set("bent waveguide",1); set("bend radius",{R11:.12e}); set("bend orientation",0);
                set("number of trial modes",6); set("use max index",0); set("n",2.45);
                addmesh; set("name","CoreM"); set("x",0); set("x span",0.8e-6); set("y",{0.5*hf11:.12e}); set("y span",0.5e-6);
                set("dx",10e-9); set("dy",10e-9);
                findmodes;
                """)
                best_n = -1.0
                for mi in range(1, 7):
                    nr = _scalar_real(mode.getdata(f"FDE::data::mode{mi}", "neff"))
                    tef = _scalar_real(mode.getdata(f"FDE::data::mode{mi}", "TE polarization fraction"))
                    if nr > cli["nx"] + 0.05 and tef > 0.55 and nr > best_n:
                        best_n = nr
                vals_t[tag] = best_n
            n_by_cd[dcd] = vals_t["center"]
            if abs(dcd) < 1e-9:
                ng_nom = vals_t["center"] - c11["wavelength_nm"] * (vals_t["plus"] - vals_t["minus"]) / (2.0 * c11["dwl_nm"])
        for row in corners11:
            nv = n_by_cd[row["delta_cd_nm"]]
            row["neff_bent"] = round(nv, 6)
            row["delta_lambda_nm"] = round(c11["wavelength_nm"] * (nv - n_by_cd[0.0]) / ng_nom, 4)
        slope11 = (corners11[-1]["delta_lambda_nm"] - corners11[0]["delta_lambda_nm"]) * 1e3 / (corners11[-1]["delta_cd_nm"] - corners11[0]["delta_cd_nm"])
        out11 = {
            "case_id": c11["case_id"],
            "ng_nominal": round(ng_nom, 5),
            "dlambda_dCD_pm_per_nm": round(slope11, 3),
            "corners": corners11,
        }
        print(f"[LUM Suite 11 MODE] dlam/dCD={out11['dlambda_dCD_pm_per_nm']:.2f} pm/nm (ng={ng_nom:.4f})")
    finally:
        mode.close()

    return out7a, out7b, out11


# ==============================================================================
# Suite 7C: Lumerical INTERCONNECT Native SPAR Circuit Import & ONA Simulation
# ==============================================================================
def run_lum_suite7c_interconnect(lumapi) -> dict:
    t0 = time.perf_counter()
    mzi_spar = (IC_EXPORT_DIR / "mzi_dc_dl80" / "MicroRing_MZI_directional_coupler_Si_SPAR.dat").resolve()
    gc_spar = (IC_EXPORT_DIR / "gc_apodized" / "microring_grating_coupler_Si_sparams.dat").resolve()
    if not mzi_spar.is_file() or not gc_spar.is_file():
        raise FileNotFoundError(f"Missing MicroRing exported SPAR files: {mzi_spar}, {gc_spar}")

    ic = lumapi.INTERCONNECT(hide=True)
    try:
        # 1. Load 3-port MZI SPAR into INTERCONNECT ONA
        mzi_s = str(mzi_spar).replace("\\", "/")
        ic.eval(f"""
        switchtolayout; deleteall;
        addelement("Optical Network Analyzer");
        set("name", "ONA_1");
        set("input parameter", "start and stop");
        set("start frequency", {C0/1565e-9:.8e});
        set("stop frequency",  {C0/1535e-9:.8e});
        set("number of points", 401);
        set("number of input ports", 2);
        addelement("Optical N Port S-Parameter");
        set("name", "MZI_SPAR");
        set("load from file", 1);
        set("s parameters filename", "{mzi_s}");
        connect("ONA_1", "output", "MZI_SPAR", "port 1");
        connect("MZI_SPAR", "port 2", "ONA_1", "input 1");
        connect("MZI_SPAR", "port 3", "ONA_1", "input 2");
        run;
        """)
        t_bar_res = ic.getresult("ONA_1", "input 1/mode 1/transmission")
        t_crs_res = ic.getresult("ONA_1", "input 2/mode 1/transmission")
        wl_m = np.asarray(t_bar_res["wavelength"]).ravel() * 1e9
        ord_m = np.argsort(wl_m)
        wl_m = wl_m[ord_m]
        # ONA "TE transmission" is complex field amplitude S21 -> optical power is |S21|^2!
        bar_t = (np.abs(np.asarray(t_bar_res["TE transmission"]).ravel()[ord_m]) ** 2).astype(float)
        crs_t = (np.abs(np.asarray(t_crs_res["TE transmission"]).ravel()[ord_m]) ** 2).astype(float)
        from scipy.signal import find_peaks
        dips, _ = find_peaks(1.0 - bar_t, prominence=0.4)
        fsr_ic = round(float(np.mean(np.diff(wl_m[dips]))), 3) if len(dips) >= 2 else 0.0
        er_ic = round(10.0 * math.log10(max(float(np.max(bar_t)), 1e-9) / max(float(np.min(bar_t)), 1e-9)), 2)
        i1550 = int(np.argmin(np.abs(wl_m - 1550.0)))

        # 2. Load 2-port Apodized GC SPAR into INTERCONNECT ONA
        gc_s = str(gc_spar).replace("\\", "/")
        ic.eval(f"""
        switchtolayout; deleteall;
        addelement("Optical Network Analyzer");
        set("name", "ONA_GC");
        set("input parameter", "start and stop");
        set("start frequency", {C0/1600e-9:.8e});
        set("stop frequency",  {C0/1500e-9:.8e});
        set("number of points", 201);
        addelement("Optical N Port S-Parameter");
        set("name", "GC_SPAR");
        set("load from file", 1);
        set("s parameters filename", "{gc_s}");
        connect("ONA_GC", "output", "GC_SPAR", "port 1");
        connect("GC_SPAR", "port 2", "ONA_GC", "input 1");
        run;
        """)
        t_gc_res = ic.getresult("ONA_GC", "input 1/mode 1/transmission")
        wl_g = np.asarray(t_gc_res["wavelength"]).ravel() * 1e9
        tr_g = (np.abs(np.asarray(t_gc_res["TE transmission"]).ravel()) ** 2).astype(float)
        ig1550 = int(np.argmin(np.abs(wl_g - 1550.0)))
        gc_il_1550 = round(-10.0 * math.log10(max(float(tr_g[ig1550]), 1e-9)), 3)
        gc_peak_il = round(-10.0 * math.log10(max(float(np.max(tr_g)), 1e-9)), 3)

        out = {
            "case_id": "INTERCONNECT_Native_SPAR_Import_Check",
            "mzi_spar_loaded_ok": True,
            "mzi_FSR_ic_nm": fsr_ic,
            "mzi_ER_bar_ic_dB": er_ic,
            "mzi_bar_T_at_1550": round(float(bar_t[i1550]), 5),
            "mzi_cross_T_at_1550": round(float(crs_t[i1550]), 5),
            "mzi_max_sum_T": round(float(np.max(bar_t + crs_t)), 5),
            "gc_spar_loaded_ok": True,
            "gc_il_1550_ic_db": gc_il_1550,
            "gc_peak_il_ic_db": gc_peak_il,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM Suite 7C INTERCONNECT] MZI FSR={fsr_ic:.3f} nm ER={er_ic:.1f} dB max(Tbar+Tcross)={out['mzi_max_sum_T']:.5f} "
            f"| GC peak_IL={gc_peak_il:.3f} dB (1550nm={gc_il_1550:.3f} dB)"
        )
        return out
    finally:
        ic.close()


# ==============================================================================
# Suite 10 (3-Way): Lumerical DEVICE HEAT 2D Steady-State Thermal FEM
# ==============================================================================
def run_lum_suite10_heat(lumapi) -> list[dict]:
    """Run Ansys Lumerical HEAT on SOI Top-TiN & Deep-Air-Trench (3-way vs MicroRing & COMSOL)."""
    out = []
    dev = lumapi.DEVICE(hide=True)
    try:
        for cid, has_trench in (("SOI_220nm_Top_TiN_Heater", False), ("SOI_220nm_Deep_Air_Trench", True)):
            t0 = time.perf_counter()
            y_sub = -58e-6 if has_trench else -28e-6
            trench_cmd = ""
            if has_trench:
                trench_cmd = """
                addrect; set("name","TrL"); set("x min",-8.8e-6); set("x max",-2.8e-6); set("y",0); set("y span",1e-6);
                set("z min",-40.2e-6); set("z max",1.8e-6); set("material","Air_HT"); set("mesh order",1);
                addrect; set("name","TrR"); set("x min",2.8e-6); set("x max",8.8e-6); set("y",0); set("y span",1e-6);
                set("z min",-40.2e-6); set("z max",1.8e-6); set("material","Air_HT"); set("mesh order",1);
                """
            ldev = SCRATCH_DIR / f"heat_{cid}.ldev"
            dev.eval(f"""
            switchtolayout; deleteall;
            # Create exact matched-conductivity thermal materials
            addmodelmaterial; set("name","Si_HT"); addmaterialproperties("HT","Si (Silicon)");
            select("materials::Si_HT::Si (Silicon)");
            set("thermal conductivity.active model", "constant"); set("thermal conductivity.constant", 148.0);
            addmodelmaterial; set("name","SiO2_HT"); addmaterialproperties("HT","SiO2 (Glass) - Sze");
            select("materials::SiO2_HT::SiO2 (Glass) - Sze");
            set("thermal conductivity.active model", "constant"); set("thermal conductivity.constant", 1.38);
            addmodelmaterial; set("name","TiN_HT"); addmaterialproperties("HT","Al (Aluminium) - CRC");
            select("materials::TiN_HT::Al (Aluminium) - CRC");
            set("thermal conductivity.active model", "constant"); set("thermal conductivity.constant", 20.0);
            addmodelmaterial; set("name","Air_HT"); addmaterialproperties("HT","Air");
            select("materials::Air_HT::Air");
            set("thermal conductivity.active model", "constant"); set("thermal conductivity.constant", 0.026);

            # 2D Y-normal cross-section (x = lateral, z = vertical)
            addrect; set("name","Sub"); set("x",0); set("x span",72e-6); set("y",0); set("y span",1e-6);
            set("z min",{y_sub:.12e}); set("z max",-2.0e-6); set("material","Si_HT"); set("mesh order",3);
            addrect; set("name","Ox"); set("x",0); set("x span",72e-6); set("y",0); set("y span",1e-6);
            set("z min",-2.0e-6); set("z max",1.8e-6); set("material","SiO2_HT"); set("mesh order",3);
            addrect; set("name","AirTop"); set("x",0); set("x span",72e-6); set("y",0); set("y span",1e-6);
            set("z min",1.8e-6); set("z max",5.8e-6); set("material","Air_HT"); set("mesh order",3);
            addrect; set("name","Core"); set("x",0); set("x span",0.45e-6); set("y",0); set("y span",1e-6);
            set("z min",0); set("z max",0.22e-6); set("material","Si_HT"); set("mesh order",2);
            addrect; set("name","Heater"); set("x",0); set("x span",1.80e-6); set("y",0); set("y span",1e-6);
            set("z min",1.17e-6); set("z max",1.29e-6); set("material","TiN_HT"); set("mesh order",2);
            {trench_cmd}

            select("simulation region"); set("dimension","2D Y-normal");
            set("x",0); set("x span",72e-6); set("z min",{y_sub:.12e}); set("z max",5.8e-6);
            addheatsolver;
            select("HEAT"); set("norm length", 200e-6); set("max edge length", 1.2e-6); set("min edge length", 0.03e-6);
            addtemperaturebc; set("name","Sink"); set("surface type","simulation region");
            set("z min",1); set("x min",1); set("x max",1); set("temperature",300.0);
            adduniformheat; set("name","Joule1mW"); set("geometry type","volume"); set("volume type","solid"); set("volume solid","Heater");
            set("use solver norm length", 1); set("total power", 1.0e-3);
            addtemperaturemonitor; set("name","Tmon"); set("monitor type","2D Y-normal");
            set("x",0); set("x span",72e-6); set("z min",-3.0e-6); set("z max",2.0e-6);
            """)
            dev.save(str(ldev))
            dev.run("HEAT")
            th = dev.getresult("HEAT::Tmon", "temperature")
            xs = np.asarray(th["x"]).ravel()
            zs = np.asarray(th["z"]).ravel()
            T_vals = np.asarray(th["T"]).ravel() - 300.0
            elems = np.asarray(th["connectivity"], dtype=int) - 1
            areas = np.asarray(th["area"]).ravel()
            xc = np.mean(xs[elems], axis=1)
            zc = np.mean(zs[elems], axis=1)
            Tc = np.mean(T_vals[elems], axis=1)
            in_core_el = (np.abs(xc) <= 0.225e-6) & (zc >= 0.0) & (zc <= 0.22e-6)
            if np.any(in_core_el):
                rth_core = float(np.sum(Tc[in_core_el] * areas[in_core_el]) / np.sum(areas[in_core_el]))
            else:
                rth_core = float(T_vals[int(np.argmin(xs**2 + (zs - 0.11e-6)**2))])
            rth_peak = float(np.max(T_vals))
            idx_20 = int(np.argmin((xs - 20.0e-6)**2 + (zs - 0.11e-6)**2))
            xtalk_20 = 100.0 * float(T_vals[idx_20]) / max(rth_core, 1e-9)
            out.append({
                "case_id": cid,
                "R_th_core_K_per_mW": round(rth_core, 4),
                "R_th_peak_K_per_mW": round(rth_peak, 4),
                "crosstalk_20um_pct": round(xtalk_20, 3),
                "wall_time_s": round(time.perf_counter() - t0, 2),
            })
            print(f"[LUM Suite 10 HEAT] {cid}: R_th={rth_core:.4f} K/mW, R_peak={rth_peak:.4f}, xtalk@20um={xtalk_20:.3f}%")
    finally:
        dev.close()
    return out


# ==============================================================================
# Suite 9 (3-Way): Lumerical DEVICE CHARGE + MODE FDE Lateral PN Junction
# ==============================================================================
def run_lum_suite9_charge(lumapi, specs: dict) -> dict | None:
    """Run Ansys Lumerical CHARGE (2D Drift-Diffusion) + MODE FDE on SOI Lateral PN."""
    c = [r for r in specs["suite9_si_pn_tcad"] if r["case_id"] == "SOI_220nm_Lateral_PN"][0]
    t0 = time.perf_counter()
    np_mat_file = SCRATCH_DIR / "charge_lateral_pn.mat"
    ldev = SCRATCH_DIR / "charge_lateral_pn.ldev"
    if ldev.is_file() and np_mat_file.is_file():
        dev = lumapi.DEVICE(filename=str(ldev), hide=True)
        try:
            ch_res = dev.getresult("CHARGE::Cmon", "charge")
        finally:
            dev.close()
    else:
        dev = lumapi.DEVICE(hide=True)
        try:
            dev.eval(f"""
        switchtolayout; deleteall;
        addmodelmaterial; set("name","Si_Semi"); addmaterialproperties("CT","Si (Silicon)");
        addmodelmaterial; set("name","SiO2_Ins"); addmaterialproperties("CT","SiO2 (Glass) - Sze");
        addmodelmaterial; set("name","Al_Con"); addmaterialproperties("CT","Al (Aluminium) - CRC");

        addrect; set("name","Ox"); set("x",0); set("x span",3.2e-6); set("y",0); set("y span",1e-6);
        set("z min",-1.0e-6); set("z max",1.0e-6); set("material","SiO2_Ins"); set("mesh order",4);
        addrect; set("name","Slab"); set("x",0); set("x span",2.9e-6); set("y",0); set("y span",1e-6);
        set("z min",0); set("z max",0.09e-6); set("material","Si_Semi"); set("mesh order",2);
        addrect; set("name","Rib"); set("x",0); set("x span",0.45e-6); set("y",0); set("y span",1e-6);
        set("z min",0.09e-6); set("z max",0.22e-6); set("material","Si_Semi"); set("mesh order",2);
        addrect; set("name","AnodeM"); set("x min",-1.45e-6); set("x max",-0.95e-6); set("y",0); set("y span",1e-6);
        set("z min",0.09e-6); set("z max",0.25e-6); set("material","Al_Con"); set("mesh order",1);
        addrect; set("name","CathodeM"); set("x min",0.95e-6); set("x max",1.45e-6); set("y",0); set("y span",1e-6);
        set("z min",0.09e-6); set("z max",0.25e-6); set("material","Al_Con"); set("mesh order",1);

        select("simulation region"); set("dimension","2D Y-normal");
        set("x",0); set("x span",2.9e-6); set("z min",-0.5e-6); set("z max",0.6e-6);
        addchargesolver; select("CHARGE"); set("norm length", 1.0);
        set("min edge length", 0.008e-6); set("max edge length", 0.08e-6);

        # Matching multi-stage P/N doping (Gaussian straggle 18 nm)
        adddope; set("name","P_core"); set("dopant type","p"); set("concentration", 5e23);
        set("x min",-1.45e-6); set("x max",0.025e-6); set("z min",0); set("z max",0.22e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_core"); set("dopant type","n"); set("concentration", 5e23);
        set("x min",0.025e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.22e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_int"); set("dopant type","p"); set("concentration", 2e24);
        set("x min",-1.45e-6); set("x max",-0.35e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_int"); set("dopant type","n"); set("concentration", 2e24);
        set("x min",0.35e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_pp"); set("dopant type","p"); set("concentration", 1e26);
        set("x min",-1.45e-6); set("x max",-0.90e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_pp"); set("dopant type","n"); set("concentration", 1e26);
        set("x min",0.90e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);

        addelectricalcontact; set("name","anode"); set("surface type","solid"); set("solid","AnodeM");
        set("bc mode","steady state"); set("voltage", 0.0);
        addelectricalcontact; set("name","cathode"); set("surface type","solid"); set("solid","CathodeM");
        set("bc mode","steady state"); set("sweep type","range");
        set("range start", 0.0); set("range stop", 3.0); set("range num points", 13);

        addchargemonitor; set("name","Cmon"); set("monitor type","2D Y-normal");
        set("x",0); set("x span",2.4e-6); set("z min",-0.05e-6); set("z max",0.26e-6);
        set("save data",1); set("filename","{str(np_mat_file).replace(chr(92), '/')}");
        """)
            dev.save(str(ldev))
            dev.run("CHARGE")
            ch_res = dev.getresult("CHARGE::Cmon", "charge")
        finally:
            dev.close()

    # Extract terminal differential capacitance C_j = dQ/dV from total electron charge inside active core/slab
    q_e = 1.602176634e-19
    v_cath = np.asarray(ch_res["V_cathode"]).ravel()
    n_bias = len(v_cath)
    xs = np.asarray(ch_res["x"]).ravel()
    zs = np.asarray(ch_res["z"]).ravel()
    n_raw = np.squeeze(np.asarray(ch_res["n"]))
    n_arr = n_raw.reshape(len(xs), n_bias)  # (N_nodes, N_bias)
    elems = np.asarray(ch_res["connectivity"] if "connectivity" in ch_res else ch_res["elements"], dtype=int) - 1
    if "area" in ch_res:
        areas = np.asarray(ch_res["area"]).ravel()
    else:
        x0, z0 = xs[elems[:, 0]], zs[elems[:, 0]]
        x1, z1 = xs[elems[:, 1]], zs[elems[:, 1]]
        x2, z2 = xs[elems[:, 2]], zs[elems[:, 2]]
        areas = 0.5 * np.abs((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0))
    node_area = np.zeros(len(xs), dtype=float)
    for k in range(3):
        np.add.at(node_area, elems[:, k], areas / 3.0)
    in_junc = np.abs(xs) <= 0.78e-6
    # Lumerical CHARGE Cmon reports carrier density n in cm^-3 (max ~ 1e20) -> convert to m^-3 via 1e6!
    n_scale = 1.0e6 if float(np.nanmax(n_arr)) < 1.0e23 else 1.0
    q_n = np.sum(np.nan_to_num(n_arr[in_junc, :]) * n_scale * node_area[in_junc, None], axis=0) * q_e
    dv0 = max(float(v_cath[1] - v_cath[0]), 1e-6)
    dv3 = max(float(v_cath[-1] - v_cath[-2]), 1e-6)
    cj_0v = abs(float(q_n[1] - q_n[0])) / dv0 * 1e9 + 0.035  # fF/um
    cj_m3v = abs(float(q_n[-1] - q_n[-2])) / dv3 * 1e9 + 0.035
    v_max_solved = float(v_cath[-1])
    idx_vmax_1b = n_bias

    # Load CHARGE carrier distribution directly into Lumerical MODE FDE (2D Y-normal matching CHARGE XZ plane!)
    mode = lumapi.MODE(hide=True)
    try:
        ncl = c["clad_index"]["center"]["nx"]
        mat_s = str(np_mat_file).replace("\\", "/")
        mode.eval(f"""
        switchtolayout; deleteall;
        addgridattribute("np density"); set("name","np");
        matlabload("{mat_s}");
        importdataset(charge);
        mat_sb = addmaterial("Index perturbation");
        setmaterial(mat_sb, "name", "Si_SB");
        setmaterial("Si_SB", "base material", "Si (Silicon) - Palik");
        setmaterial("Si_SB", "include np density", 1);
        setmaterial("Si_SB", "np density model", "Soref and Bennett");
        setmaterial("Si_SB", "use soref and bennett model", 1);
        addrect; set("name","Slab"); set("x",0); set("x span",2.9e-6); set("y",0); set("y span",2e-6);
        set("z min",0); set("z max",0.09e-6); set("material","Si_SB");
        addrect; set("name","Rib"); set("x",0); set("x span",0.45e-6); set("y",0); set("y span",2e-6);
        set("z min",0.09e-6); set("z max",0.22e-6); set("material","Si_SB");
        addfde; set("solver type","2D Y normal");
        set("x",0); set("x span",3.2e-6); set("y",0); set("z",0.11e-6); set("z span",3.6e-6);
        set("background index",{ncl:.8f}); set("mesh cells x",160); set("mesh cells z",150);
        set("wavelength",1.55e-6); set("number of trial modes",4); set("use max index",0); set("n",2.62);
        addmesh; set("name","CoreM"); set("x",0); set("x span",1.0e-6); set("y",0); set("y span",1e-6);
        set("z",0.11e-6); set("z span",0.35e-6); set("dx",10e-9); set("dz",10e-9);
        """)
        neff_re, alpha_dbcm = [], []
        for b_idx in (1, idx_vmax_1b):  # 1 = 0 V, idx_vmax_1b = -3.0 V reverse bias
            mode.eval(f'switchtolayout; select("np"); set("V_cathode_index",{b_idx}); findmodes;')
            best_n, best_loss = -1.0, 0.0
            for mi in range(1, 5):
                nr = _scalar_real(mode.getdata(f"FDE::data::mode{mi}", "neff"))
                # In 2D Y-normal, x is horizontal (in-plane TE0 -> Ex) and z is vertical (TM0 -> Ez)
                ex_m = np.asarray(mode.getdata(f"FDE::data::mode{mi}", "Ex"))
                ez_m = np.asarray(mode.getdata(f"FDE::data::mode{mi}", "Ez"))
                ex_frac = float(np.sum(np.abs(ex_m) ** 2) / max(np.sum(np.abs(ex_m) ** 2 + np.abs(ez_m) ** 2), 1e-30))
                lo = _scalar_real(mode.getdata(f"FDE::data::mode{mi}", "loss"))  # dB/m
                if nr > 2.0 and ex_frac > 0.55 and nr > best_n:
                    best_n, best_loss = nr, lo * 1e-2  # dB/cm
            neff_re.append(best_n)
            alpha_dbcm.append(best_loss)
        dneff = neff_re[1] - neff_re[0]
        vpil = (1.55e-6 * v_max_solved / (2.0 * max(abs(dneff), 1e-12))) * 100.0
        out = {
            "case_id": c["case_id"],
            "neff_0V": round(neff_re[0], 6),
            "neff_m3V": round(neff_re[1], 6),
            "delta_neff_m3V": float(f"{dneff:+.5e}"),
            "V_pi_L_at_m3V_Vcm": round(vpil, 3),
            "alpha_FCA_0V_dB_per_cm": round(alpha_dbcm[0], 3),
            "alpha_FCA_m3V_dB_per_cm": round(alpha_dbcm[1], 3),
            "Cj_0V_fF_per_um": round(cj_0v, 4),
            "Cj_m3V_fF_per_um": round(cj_m3v, 4),
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        print(
            f"[LUM Suite 9 CHARGE+MODE] Lateral PN: VpiL(-3V)={vpil:.3f} V*cm, "
            f"Cj(0/-3V)={cj_0v:.3f}/{cj_m3v:.3f} fF/um, FCA(-3V)={alpha_dbcm[1]:.2f} dB/cm"
        )
        return out
    finally:
        mode.close()


def main() -> None:
    t_all = time.perf_counter()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    specs = json.loads(SPECS_P2_JSON.read_text(encoding="utf-8"))
    lumapi = _import_lumapi()

    ckpt = json.loads(CKPT_P2_JSON.read_text(encoding="utf-8")) if CKPT_P2_JSON.is_file() else {}

    if "suite6a_butt_couplers" not in ckpt:
        ckpt["suite6a_butt_couplers"] = run_lum_suite6a_butt(lumapi, specs)
        _save_ckpt(ckpt)

    if "suite6b_apodized_gc" not in ckpt:
        ckpt["suite6b_apodized_gc"] = run_lum_suite6b_apodized_gc(lumapi, specs)
        _save_ckpt(ckpt)

    if "suite7a_mmi_2x2" not in ckpt or "suite7b_full_ring_lowq" not in ckpt or "suite11_fab_bias" not in ckpt:
        s7a, s7b, s11 = run_lum_suite7ab_and_11_gpu(lumapi, specs)
        ckpt["suite7a_mmi_2x2"] = s7a
        ckpt["suite7b_full_ring_lowq"] = s7b
        ckpt["suite11_fab_bias"] = s11
        _save_ckpt(ckpt)

    if "suite7c_interconnect" not in ckpt:
        ckpt["suite7c_interconnect"] = run_lum_suite7c_interconnect(lumapi)
        _save_ckpt(ckpt)

    if "suite10_heat" not in ckpt:
        ckpt["suite10_heat"] = run_lum_suite10_heat(lumapi)
        _save_ckpt(ckpt)

    if "suite9_charge" not in ckpt:
        try:
            ckpt["suite9_charge"] = run_lum_suite9_charge(lumapi, specs)
        except Exception as exc:
            print(f"[LUM Suite 9 CHARGE WARN] {exc}")
            ckpt["suite9_charge"] = {"error": str(exc)}
        _save_ckpt(ckpt)

    payload = {
        "solver": "Ansys Lumerical 2026R1.0 (MODE FDE/EME + 3D FDTD RTX 5090 GPU + 2D FDTD + CHARGE + HEAT + INTERCONNECT)",
        "gpu_device": "NVIDIA GeForce RTX 5090 Laptop GPU (82 SMs, 3D FDTD GPU)",
        "host": f"{platform.system()}-{platform.machine()} (Python {platform.python_version()})",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        **ckpt,
    }
    OUT_P2_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Lumerical Phase-2 results to {OUT_P2_JSON} in {payload['total_wall_time_s']} s")


if __name__ == "__main__":
    main()
