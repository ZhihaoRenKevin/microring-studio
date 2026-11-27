"""Independent COMSOL Multiphysics 6.4 Phase-2 Validation Runner (Suites 6, 8, 9, 10, 11).

Reads ONLY neutral physical specifications from `benchmarks/outputs/commercial_validation_specs_p2.json`
and executes COMSOL Multiphysics 6.4 completely independently across:
  - Suite 6A : `Wave Optics (ewfd)` 2D quadratic FEM Butt/Edge Coupler facet modes, complex
               vectorial Poynting overlap against Gaussian fiber/laser beams, and 1-dB tolerances.
  - Suite 6B : `Wave Optics (ewfd)` 2D full-wave Gaussian-Apodized 26-tooth Grating Coupler.
  - Suite 8  : `AC/DC (es)` + `RF Module (emw)` 2D lossy transmission-line modes (10/40/100 GHz)
               across HR-Si, Quartz, Undercut-Si, and Slow-Wave T-Rail (including 3D unit cell!).
  - Suite 9  : `General Form PDE (g)` 2D Nonlinear Poisson-Boltzmann Silicon PN TCAD (Lateral,
               L-Shaped, U-Shaped) coupled into `Wave Optics (ewfd)` Soref-Bennett V_pi*L & FCA.
  - Suite 10 : `Heat Transfer (ht)` + `ewfd` for XeF2 Undercut, Doped-Si Rib, Co-Integrated
               PN+TiN Contact Heat-Sinking (bare vs sinked), and a TRUE 3D Finite-Length (200 um)
               SOI Top-TiN thermal model quantifying 2D-vs-3D longitudinal heat spreading.
  - Suite 11 : `Wave Optics (ewfd)` bent SOI eigenmodes across correlated CD bias (-20, 0, +20 nm).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

COMSOL_BIN = Path(r"C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64")
COMSOL_COMPILE = COMSOL_BIN / "comsolcompile.exe"
COMSOL_BATCH = COMSOL_BIN / "comsolbatch.exe"

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH_DIR = OUT_DIR / "_comsol_scratch"
SPECS_P2_JSON = OUT_DIR / "commercial_validation_specs_p2.json"
OUT_P2_JSON = OUT_DIR / "comsol_independent_results_p2.json"

C0 = 299792458.0
EPS0 = 8.8541878128e-12
Z0_VAC = 376.730313668


def _run_comsol_java(class_name: str, java_src: str) -> list[str]:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    java_path = SCRATCH_DIR / f"{class_name}.java"
    class_path = SCRATCH_DIR / f"{class_name}.class"
    log_path = SCRATCH_DIR / f"{class_name}.log"
    java_path.write_text(java_src, encoding="ascii")

    if class_path.is_file():
        class_path.unlink()
    comp = subprocess.run(
        [str(COMSOL_COMPILE), str(java_path)],
        capture_output=True, encoding="utf-8", errors="replace", check=False,
    )
    if comp.returncode != 0 or not class_path.is_file():
        raise RuntimeError(f"comsolcompile failed for {class_name}:\n{comp.stdout}\n{comp.stderr}")

    batch = subprocess.run(
        [str(COMSOL_BATCH), "-inputfile", str(class_path), "-batchlog", str(log_path)],
        capture_output=True, encoding="utf-8", errors="replace", check=False,
    )
    log_txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.is_file() else ""
    if batch.returncode != 0 or "/*****Error********/" in log_txt:
        raise RuntimeError(f"comsolbatch failed for {class_name}:\n{log_txt[-2800:]}\n{batch.stderr}")
    return ((batch.stdout or "") + "\n" + log_txt).splitlines()


def _parse_kv_lines(lines: list[str], prefix: str) -> list[dict]:
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(prefix):
            out.append(json.loads(s[len(prefix):].strip()))
    return out


# ==============================================================================
# Block 1: Suite 6A (Butt/Edge Couplers) & Suite 11 (Litho CD Bias Corners)
# ==============================================================================
def run_comsol_suite6a_and_11(specs: dict) -> tuple[list[dict], dict]:
    calls = []
    for c in specs["suite6a_butt_couplers"]:
        wl_m = c["wavelength_nm"] * 1e-9
        hf = c["film_height_um"] * 1e-6
        nc = c["core_index"]["center"]["nx"]
        ncl = c["clad_index"]["center"]["nx"]
        fx = c["mfd_x_um"] * 1e-6
        fy = c["mfd_y_um"] * 1e-6
        dom_w = max(2.4 * c["mfd_x_um"], 6.5 * (c["wavelength_nm"] / 1550.0)) * 1e-6
        pad_y = max(1.4 * c["mfd_y_um"], 3.2 * (c["wavelength_nm"] / 1550.0)) * 1e-6
        for wt_um in c["w_tip_sweep_um"]:
            is_nom = 1 if abs(wt_um - c["w_tip_nominal_um"]) < 1e-6 else 0
            csv_name = f"butt_{c['case_id']}_{int(round(wt_um*1e3))}nm.txt"
            calls.append(
                f'solveButt("{c["case_id"]}", {wt_um*1e-6:.12e}, {hf:.12e}, {wl_m:.12e}, '
                f'{dom_w:.12e}, {pad_y:.12e}, {nc:.8f}, {ncl:.8f}, {is_nom}, "{csv_name}");'
            )

    # Suite 11: Bent SOI R=15um across delta_cd = -20, 0, +20 nm (+ dispersion at nominal)
    c11 = specs["suite11_fab_bias"]
    r_m = c11["ring_radius_um"] * 1e-6
    hf11 = c11["film_height_um"] * 1e-6
    for dcd in c11["delta_cd_nm_list"]:
        w_m = (c11["ring_width_nom_um"] + dcd * 1e-3) * 1e-6
        for tag in (("minus", "center", "plus") if abs(dcd) < 1e-9 else ("center",)):
            ci = c11["core_index"][tag]
            cli = c11["clad_index"][tag]
            calls.append(
                f'solveBentS11({dcd:.1f}, "{tag}", {ci["wavelength_nm"]*1e-9:.12e}, {w_m:.12e}, '
                f'{hf11:.12e}, {r_m:.12e}, {ci["nx"]:.8f}, {cli["nx"]:.8f});'
            )

    scratch_java = str(SCRATCH_DIR).replace("\\", "/")
    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;

public class ComsolSuite6A11 {{
  static void setIso(Model m, String fTag, double n, String rho) {{
    m.component("comp1").physics("ewfd").feature(fTag).set("DisplacementFieldModel", "RelativePermittivity");
    m.component("comp1").physics("ewfd").feature(fTag).set("epsilonr_mat", "userdef");
    String e = "(" + (n*n) + ")";
    if (rho == null) {{
      m.component("comp1").physics("ewfd").feature(fTag).set("epsilonr", new String[]{{e,"0","0","0",e,"0","0","0",e}});
      m.component("comp1").physics("ewfd").feature(fTag).set("mur_mat", "userdef");
      m.component("comp1").physics("ewfd").feature(fTag).set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    }} else {{
      m.component("comp1").physics("ewfd").feature(fTag).set("epsilonr", new String[]{{e+"*"+rho,"0","0","0",e+"*"+rho,"0","0","0",e+"/"+rho}});
      m.component("comp1").physics("ewfd").feature(fTag).set("mur_mat", "userdef");
      m.component("comp1").physics("ewfd").feature(fTag).set("mur", new String[]{{rho,"0","0","0",rho,"0","0","0","1/"+rho}});
    }}
    m.component("comp1").physics("ewfd").feature(fTag).set("sigma_mat", "userdef");
    m.component("comp1").physics("ewfd").feature(fTag).set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});
  }}

  static void solveButt(String cid, double wt, double hf, double wl, double domW, double padY,
      double nc, double ncl, int exportField, String outName) throws Exception {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);
    m.param().set("wl0", wl + "[m]");

    m.component("comp1").geom("geom1").create("rClad", "Rectangle");
    m.component("comp1").geom("geom1").feature("rClad").set("pos", new double[]{{-0.5*domW, -padY}});
    m.component("comp1").geom("geom1").feature("rClad").set("size", new double[]{{domW, 2.0*padY + hf}});
    m.component("comp1").geom("geom1").create("rCore", "Rectangle");
    m.component("comp1").geom("geom1").feature("rCore").set("pos", new double[]{{-0.5*wt, 0.0}});
    m.component("comp1").geom("geom1").feature("rCore").set("size", new double[]{{wt, hf}});
    m.component("comp1").geom("geom1").run();

    m.component("comp1").selection().create("selCore", "Box");
    m.component("comp1").selection("selCore").set("entitydim", 2);
    m.component("comp1").selection("selCore").set("condition", "inside");
    m.component("comp1").selection("selCore").set("xmin", -0.5*wt - 1e-9);
    m.component("comp1").selection("selCore").set("xmax",  0.5*wt + 1e-9);
    m.component("comp1").selection("selCore").set("ymin", -1e-9);
    m.component("comp1").selection("selCore").set("ymax", hf + 1e-9);

    m.component("comp1").physics().create("ewfd", "ElectromagneticWaves", "geom1");
    setIso(m, "wee1", ncl, null);
    m.component("comp1").physics("ewfd").create("weeCore", "WaveEquationElectric", 2);
    m.component("comp1").physics("ewfd").feature("weeCore").selection().named("selCore");
    setIso(m, "weeCore", nc, null);

    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("size").set("hmax", Math.min(domW/18.0, 260e-9));
    m.component("comp1").mesh("mesh1").feature("size").set("hmin", 10e-9);
    m.component("comp1").mesh("mesh1").create("szC", "Size");
    m.component("comp1").mesh("mesh1").feature("szC").selection().named("selCore");
    m.component("comp1").mesh("mesh1").feature("szC").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szC").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szC").set("hmax", Math.min(wt/5.0, 28e-9));
    m.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    m.component("comp1").mesh("mesh1").run();

    m.study().create("std1");
    m.study("std1").create("mode", "ModeAnalysis");
    m.study("std1").feature("mode").set("modeFreq", "c_const/wl0");
    m.study("std1").feature("mode").set("neigs", 6);
    m.study("std1").feature("mode").set("shift", String.valueOf(0.45*nc + 0.55*ncl));
    m.study("std1").run();

    m.result().numerical().create("gev1", "EvalGlobal");
    m.result().numerical("gev1").set("expr", new String[]{{"real(emw.neff)"}});
    double[][] neffs = m.result().numerical("gev1").getReal();
    m.result().numerical().create("intEx", "IntSurface");
    m.result().numerical("intEx").selection().all();
    m.result().numerical("intEx").set("expr", new String[]{{"abs(emw.Ex)^2", "abs(emw.Ey)^2"}});
    double[][] pols = m.result().numerical("intEx").getReal();

    int best = -1; double bestN = -1.0;
    for (int i = 0; i < neffs[0].length; i++) {{
      double nr = neffs[0][i];
      double tef = pols[0][i] / Math.max(pols[0][i] + pols[1][i], 1e-30);
      if (nr > ncl + 1e-4 && nr < nc + 0.02 && tef > 0.50 && nr > bestN) {{ bestN = nr; best = i; }}
    }}
    if (best < 0) {{ best = 0; bestN = neffs[0][0]; }}

    // Export regular grid (Ex, Ey, Hx, Hy) for exact vectorial Poynting overlap in Python
    m.result().export().create("dat1", "Data");
    m.result().export("dat1").set("data", "dset1");
    m.result().export("dat1").set("solnum", new int[]{{best + 1}});
    m.result().export("dat1").set("expr", new String[]{{"real(emw.Ex)","imag(emw.Ex)","real(emw.Ey)","imag(emw.Ey)","real(emw.Hx)","imag(emw.Hx)","real(emw.Hy)","imag(emw.Hy)"}});
    m.result().export("dat1").set("location", "grid");
    m.result().export("dat1").set("gridstruct", "grid");
    m.result().export("dat1").set("gridx2", "range(" + (-0.48*domW) + "," + (0.96*domW/96.0) + "," + (0.48*domW) + ")");
    m.result().export("dat1").set("gridy2", "range(" + (-0.95*padY) + "," + ((1.90*padY+hf)/96.0) + "," + (0.95*padY+hf) + ")");
    m.result().export("dat1").set("filename", "{scratch_java}/" + outName);
    m.result().export("dat1").run();

    System.out.println("S6A_JSON:" + "{{\\"case_id\\":\\"" + cid + "\\",\\"w_tip_um\\":" + (wt*1e6) + ",\\"neff\\":" + bestN + ",\\"grid_file\\":\\"" + outName + "\\"}}");
  }}

  static void solveBentS11(double dcd, String tag, double wl, double w, double hf, double rBend, double nc, double ncl) {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);
    m.param().set("wl0", wl + "[m]");
    m.param().set("R_bend", rBend + "[m]");
    String rho = "(1 + x/R_bend)";

    double domW = 4.6e-6, pad = 2.0e-6;
    m.component("comp1").geom("geom1").create("rClad", "Rectangle");
    m.component("comp1").geom("geom1").feature("rClad").set("pos", new double[]{{-0.5*domW, -pad}});
    m.component("comp1").geom("geom1").feature("rClad").set("size", new double[]{{domW, 2.0*pad + hf}});
    m.component("comp1").geom("geom1").create("rCore", "Rectangle");
    m.component("comp1").geom("geom1").feature("rCore").set("pos", new double[]{{-0.5*w, 0.0}});
    m.component("comp1").geom("geom1").feature("rCore").set("size", new double[]{{w, hf}});
    m.component("comp1").geom("geom1").run();

    m.component("comp1").selection().create("selCore", "Box");
    m.component("comp1").selection("selCore").set("entitydim", 2);
    m.component("comp1").selection("selCore").set("condition", "inside");
    m.component("comp1").selection("selCore").set("xmin", -0.5*w - 1e-9);
    m.component("comp1").selection("selCore").set("xmax",  0.5*w + 1e-9);
    m.component("comp1").selection("selCore").set("ymin", -1e-9);
    m.component("comp1").selection("selCore").set("ymax", hf + 1e-9);

    m.component("comp1").physics().create("ewfd", "ElectromagneticWaves", "geom1");
    setIso(m, "wee1", ncl, rho);
    m.component("comp1").physics("ewfd").create("weeCore", "WaveEquationElectric", 2);
    m.component("comp1").physics("ewfd").feature("weeCore").selection().named("selCore");
    setIso(m, "weeCore", nc, rho);

    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("size").set("hmax", 220e-9);
    m.component("comp1").mesh("mesh1").feature("size").set("hmin", 8e-9);
    m.component("comp1").mesh("mesh1").create("szC", "Size");
    m.component("comp1").mesh("mesh1").feature("szC").selection().named("selCore");
    m.component("comp1").mesh("mesh1").feature("szC").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szC").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szC").set("hmax", 22e-9);
    m.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    m.component("comp1").mesh("mesh1").run();

    m.study().create("std1");
    m.study("std1").create("mode", "ModeAnalysis");
    m.study("std1").feature("mode").set("modeFreq", "c_const/wl0");
    m.study("std1").feature("mode").set("neigs", 6);
    m.study("std1").feature("mode").set("shift", "2.45");
    m.study("std1").run();

    m.result().numerical().create("gev1", "EvalGlobal");
    m.result().numerical("gev1").set("expr", new String[]{{"real(emw.neff)"}});
    double[][] neffs = m.result().numerical("gev1").getReal();
    m.result().numerical().create("intEx", "IntSurface");
    m.result().numerical("intEx").selection().all();
    m.result().numerical("intEx").set("expr", new String[]{{"abs(emw.Ex)^2", "abs(emw.Ey)^2"}});
    double[][] pols = m.result().numerical("intEx").getReal();
    int best = 0; double bestN = -1.0;
    for (int i = 0; i < neffs[0].length; i++) {{
      double nr = neffs[0][i];
      double tef = pols[0][i] / Math.max(pols[0][i] + pols[1][i], 1e-30);
      if (nr > ncl + 0.05 && nr < nc + 0.02 && tef > 0.55 && nr > bestN) {{ bestN = nr; best = i; }}
    }}
    System.out.println("S11_JSON:" + "{{\\"delta_cd_nm\\":" + dcd + ",\\"tag\\":\\"" + tag + "\\",\\"neff\\":" + bestN + "}}");
  }}

  public static void run() throws Exception {{
    {chr(10).join("    " + ln for ln in calls)}
  }}
  public static void main(String[] args) throws Exception {{ run(); }}
}}
'''
    lines = _run_comsol_java("ComsolSuite6A11", java_src)
    s6a_raw = _parse_kv_lines(lines, "S6A_JSON:")
    s11_raw = _parse_kv_lines(lines, "S11_JSON:")

    # Post-process exported COMSOL (Ex, Ey, Hx, Hy) grids for exact vectorial Poynting overlap & 1-dB tol
    def _eval_butt_grid(grid_path: Path, mfd_x_um: float, mfd_y_um: float, y_mid_m: float, n_env: float, box_um: float, ar_coating: bool, neff_r: float):
        # Parse COMSOL grid export format (% Grid / x / y / % Data / blocks)
        raw_lines = grid_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        x_vals, y_vals = None, None
        data_blocks = []
        cur = []
        mode_sec = ""
        for ln in raw_lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("%"):
                if "Grid" in s:
                    mode_sec = "grid"
                elif "Data" in s:
                    if cur:
                        data_blocks.append(cur)
                        cur = []
                    mode_sec = "data"
                continue
            vals = [float(t) for t in s.split()]
            if mode_sec == "grid":
                if x_vals is None:
                    x_vals = np.array(vals, dtype=float)
                elif y_vals is None:
                    y_vals = np.array(vals, dtype=float)
            elif mode_sec == "data":
                cur.append(vals)
        if cur:
            data_blocks.append(cur)
        nx, ny = len(x_vals), len(y_vals)
        arrs = [np.array(b, dtype=float).reshape((ny, nx)) for b in data_blocks]
        Ex = arrs[0] + 1j * arrs[1]
        Ey = arrs[2] + 1j * arrs[3]
        Hx = arrs[4] + 1j * arrs[5]
        Hy = arrs[6] + 1j * arrs[7]
        dx = float(x_vals[1] - x_vals[0])
        dy = float(y_vals[1] - y_vals[0])
        dA = dx * dy
        sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        if float(np.sum(sz)) < 0:
            Hx, Hy = -Hx, -Hy
            sz = -sz
        p_chip = max(float(np.sum(sz) * dA), 1e-30)
        w0x = 0.5 * mfd_x_um * 1e-6
        w0y = 0.5 * mfd_y_um * 1e-6
        adm_f = n_env / Z0_VAC
        kern = adm_f * Ex + Hy

        def _ov(dx_s: np.ndarray, dy_s: np.ndarray) -> np.ndarray:
            gx = np.exp(-((x_vals[None, :] - dx_s[:, None]) / w0x) ** 2)
            gy = np.exp(-((y_vals[None, :] - (y_mid_m + dy_s[:, None])) / w0y) ** 2)
            p_f = 0.5 * adm_f * np.maximum(np.sum(gx**2, axis=1) * np.sum(gy**2, axis=1) * dA, 0.5 * math.pi * w0x * w0y)
            icf = np.sum((gy @ kern) * gx, axis=1) * dA
            return np.clip(np.real((np.abs(icf) ** 2) / np.maximum(16.0 * p_chip * p_f, 1e-30)), 0.0, 1.0)

        eta0 = float(_ov(np.array([0.0]), np.array([0.0]))[0])
        offs_x = np.linspace(0.0, max(2.5, 0.75 * mfd_x_um), 81)
        offs_y = np.linspace(0.0, max(2.5, 0.75 * mfd_y_um), 81)
        ex_c = _ov(offs_x * 1e-6, np.zeros_like(offs_x))
        ey_c = _ov(np.zeros_like(offs_y), offs_y * 1e-6)

        def _tol(offs, cur_e, db):
            targ = float(cur_e[0]) * (10.0 ** (-db / 10.0))
            idx = np.where(cur_e <= targ)[0]
            if len(idx) == 0:
                return float(offs[-1])
            k = int(idx[0])
            if k == 0:
                return 0.0
            return float(offs[k-1] + (targ - cur_e[k-1]) * (offs[k] - offs[k-1]) / (cur_e[k] - cur_e[k-1]))

        # Same substrate keep & AR reflection accounting for apples-to-apples facet IL
        w_pos = np.maximum(sz, 0.0)
        X2, Y2 = np.meshgrid(x_vals, y_vals)
        ybar = float(np.sum(w_pos * Y2) / max(float(np.sum(w_pos)), 1e-30))
        sig_y = math.sqrt(max(float(np.sum(w_pos * (Y2 - ybar)**2) / max(float(np.sum(w_pos)), 1e-30)), 1e-16))
        w0y_chip_um = max(2.0 * sig_y * 1e6, 0.25)
        eta_sub = float(np.clip(1.0 - 0.5 * math.erfc(math.sqrt(2.0) * box_um / w0y_chip_um), 0.15, 1.0))
        r_f = 0.0025 if ar_coating else ((neff_r - n_env) / (neff_r + n_env)) ** 2
        eta_fac = max(eta0 * eta_sub * (1.0 - r_f), 1e-6)
        return {
            "mode_overlap_pct": round(eta0 * 100.0, 2),
            "facet_il_db": round(-10.0 * math.log10(eta_fac), 3),
            "tol_1db_x_um": round(_tol(offs_x, ex_c, 1.0), 3),
            "tol_1db_y_um": round(_tol(offs_y, ey_c, 1.0), 3),
        }

    s6a_out = []
    for c in specs["suite6a_butt_couplers"]:
        cid = c["case_id"]
        sub_rows = [r for r in s6a_raw if r["case_id"] == cid]
        tip_out = []
        nom_ov = None
        for r in sub_rows:
            ov = _eval_butt_grid(
                SCRATCH_DIR / r["grid_file"],
                c["mfd_x_um"], c["mfd_y_um"], 0.5 * c["film_height_um"] * 1e-6,
                c["n_env"], c["box_thickness_um"], c["ar_coating"], r["neff"],
            )
            row = {
                "w_tip_um": round(r["w_tip_um"], 4),
                "n_eff_tip": round(r["neff"], 6),
                "mode_overlap_pct": ov["mode_overlap_pct"],
                "facet_il_db": ov["facet_il_db"],
            }
            tip_out.append(row)
            if abs(r["w_tip_um"] - c["w_tip_nominal_um"]) < 1e-4:
                nom_ov = (row, ov)
        s6a_out.append({
            "case_id": cid,
            "w_tip_nominal_um": c["w_tip_nominal_um"],
            "n_eff_tip_nom": nom_ov[0]["n_eff_tip"],
            "mode_overlap_nom_pct": nom_ov[0]["mode_overlap_pct"],
            "facet_il_nom_db": nom_ov[0]["facet_il_db"],
            "tol_1db_x_um": nom_ov[1]["tol_1db_x_um"],
            "tol_1db_y_um": nom_ov[1]["tol_1db_y_um"],
            "tip_sweep": tip_out,
        })
        print(
            f"[COMSOL Suite 6A] {cid}: neff={nom_ov[0]['n_eff_tip']:.5f}, "
            f"overlap={nom_ov[0]['mode_overlap_pct']:.2f}%, facet_IL={nom_ov[0]['facet_il_db']:.3f} dB, "
            f"tol1dB=({nom_ov[1]['tol_1db_x_um']:.2f}, {nom_ov[1]['tol_1db_y_um']:.2f}) um"
        )

    # Suite 11 summary
    by_s11 = {(round(r["delta_cd_nm"], 1), r["tag"]): r["neff"] for r in s11_raw}
    c11 = specs["suite11_fab_bias"]
    n0 = by_s11[(0.0, "center")]
    ng0 = n0 - c11["wavelength_nm"] * (by_s11[(0.0, "plus")] - by_s11[(0.0, "minus")]) / (2.0 * c11["dwl_nm"])
    corners11 = []
    for dcd in c11["delta_cd_nm_list"]:
        nv = by_s11[(round(dcd, 1), "center")]
        dlam = c11["wavelength_nm"] * (nv - n0) / ng0
        corners11.append({"delta_cd_nm": dcd, "neff_bent": round(nv, 6), "delta_lambda_nm": round(dlam, 4)})
    slope11 = (corners11[-1]["delta_lambda_nm"] - corners11[0]["delta_lambda_nm"]) * 1e3 / (corners11[-1]["delta_cd_nm"] - corners11[0]["delta_cd_nm"])
    s11_out = {
        "case_id": c11["case_id"],
        "ng_nominal": round(ng0, 5),
        "dlambda_dCD_pm_per_nm": round(slope11, 3),
        "corners": corners11,
    }
    print(f"[COMSOL Suite 11] dlam/dCD={s11_out['dlambda_dCD_pm_per_nm']:.2f} pm/nm (ng={ng0:.4f})")
    return s6a_out, s11_out


# ==============================================================================
# Block 2: Suite 6B (COMSOL 2D Full-Wave Apodized Grating Coupler)
# ==============================================================================
def run_comsol_suite6b_apodized_gc(specs: dict) -> dict:
    c = specs["suite6b_apodized_gc"]
    wl_m = c["wavelength_nm"] * 1e-9
    hf = c["film_height_um"] * 1e-6
    h_rem = (c["film_height_um"] - c["etch_depth_um"]) * 1e-6
    h_box = c["box_thickness_um"] * 1e-6
    h_top = c["top_clad_thickness_um"] * 1e-6
    nc = c["core_index"]["center"]["nx"]
    ncl = c["clad_index"]["center"]["nx"]
    grooves = c["grooves_um"]
    gc_len = grooves[-1][1] * 1e-6 + 0.6e-6

    z_in = 3.0e-6
    z_out = gc_len + 3.0e-6
    pml = 1.0e-6
    y_sub = -h_box - 1.2e-6
    y_air = hf + h_top + 1.8e-6
    y_up_mon = hf + h_top + 0.50e-6
    y_dn_mon = -h_box - 0.35e-6
    x_port_in = -z_in

    tooth_lines = []
    z_prev = x_port_in
    idx_t = 0
    for g0_u, g1_u in grooves:
        g0 = g0_u * 1e-6
        g1 = g1_u * 1e-6
        if g0 - z_prev > 1e-10:
            tooth_lines.append(f'addRect(m, "t{idx_t}", {z_prev:.12e}, {h_rem:.12e}, {g0 - z_prev:.12e}, {hf - h_rem:.12e});')
            idx_t += 1
        z_prev = g1
    tooth_lines.append(f'addRect(m, "tEnd", {z_prev:.12e}, {h_rem:.12e}, {(z_out + pml) - z_prev:.12e}, {hf - h_rem:.12e});')

    scratch_java = str(SCRATCH_DIR).replace("\\", "/")
    up_file = f"{scratch_java}/comsol_gc_apo_up.txt"

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolSuite6B {{
  static void addRect(Model m, String tag, double x, double y, double w, double h) {{
    m.component("comp1").geom("geom1").create(tag, "Rectangle");
    m.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{{x, y}});
    m.component("comp1").geom("geom1").feature(tag).set("size", new double[]{{w, h}});
  }}
  static void setIso(Model m, String fTag, double n) {{
    m.component("comp1").physics("ewfd").feature(fTag).set("DisplacementFieldModel", "RelativePermittivity");
    m.component("comp1").physics("ewfd").feature(fTag).set("epsilonr_mat", "userdef");
    String e = String.valueOf(n*n);
    m.component("comp1").physics("ewfd").feature(fTag).set("epsilonr", new String[]{{e,"0","0","0",e,"0","0","0",e}});
    m.component("comp1").physics("ewfd").feature(fTag).set("mur_mat", "userdef");
    m.component("comp1").physics("ewfd").feature(fTag).set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    m.component("comp1").physics("ewfd").feature(fTag).set("sigma_mat", "userdef");
    m.component("comp1").physics("ewfd").feature(fTag).set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});
  }}
  public static void run() {{
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);
    m.param().set("wl0", "{wl_m:.12e}[m]");

    double x0 = {x_port_in:.12e}, x1 = {z_out + pml:.12e}, y0 = {y_sub - pml:.12e}, y1 = {y_air + pml:.12e};
    addRect(m, "rSub", x0, y0, x1 - x0, {-h_box - (y_sub - pml):.12e});
    addRect(m, "rOx",  x0, {-h_box:.12e}, x1 - x0, {hf + h_top + h_box:.12e});
    addRect(m, "rAir", x0, {hf + h_top:.12e}, x1 - x0, {(y_air + pml) - (hf + h_top):.12e});
    addRect(m, "rSlab", x0, 0.0, x1 - x0, {h_rem:.12e});
    {chr(10).join("    " + ln for ln in tooth_lines)}
    // PML partitions (right, top, bottom)
    addRect(m, "pR", {z_out:.12e}, y0, {pml:.12e}, y1 - y0);
    addRect(m, "pT", x0, {y_air:.12e}, x1 - x0, {pml:.12e});
    addRect(m, "pB", x0, y0, x1 - x0, {pml:.12e});
    // Monitor cutlines
    m.component("comp1").geom("geom1").create("lsUp", "LineSegment");
    m.component("comp1").geom("geom1").feature("lsUp").set("specify1", "coord");
    m.component("comp1").geom("geom1").feature("lsUp").set("specify2", "coord");
    m.component("comp1").geom("geom1").feature("lsUp").set("coord1", new double[]{{-1.5e-6, {y_up_mon:.12e}}});
    m.component("comp1").geom("geom1").feature("lsUp").set("coord2", new double[]{{{gc_len + 1.5e-6:.12e}, {y_up_mon:.12e}}});
    m.component("comp1").geom("geom1").create("lsDn", "LineSegment");
    m.component("comp1").geom("geom1").feature("lsDn").set("specify1", "coord");
    m.component("comp1").geom("geom1").feature("lsDn").set("specify2", "coord");
    m.component("comp1").geom("geom1").feature("lsDn").set("coord1", new double[]{{-1.5e-6, {y_dn_mon:.12e}}});
    m.component("comp1").geom("geom1").feature("lsDn").set("coord2", new double[]{{{gc_len + 1.5e-6:.12e}, {y_dn_mon:.12e}}});
    m.component("comp1").geom("geom1").run();

    // Selections
    m.component("comp1").selection().create("selSub", "Box");
    m.component("comp1").selection("selSub").set("entitydim", 2);
    m.component("comp1").selection("selSub").set("condition", "inside");
    m.component("comp1").selection("selSub").set("ymax", {-h_box + 1e-9:.12e});

    m.component("comp1").selection().create("selAir", "Box");
    m.component("comp1").selection("selAir").set("entitydim", 2);
    m.component("comp1").selection("selAir").set("condition", "inside");
    m.component("comp1").selection("selAir").set("ymin", {hf + h_top - 1e-9:.12e});

    m.component("comp1").selection().create("selSiCore", "Box");
    m.component("comp1").selection("selSiCore").set("entitydim", 2);
    m.component("comp1").selection("selSiCore").set("condition", "inside");
    m.component("comp1").selection("selSiCore").set("ymin", -1e-9);
    m.component("comp1").selection("selSiCore").set("ymax", {hf + 1e-9:.12e});
    // Actually grooves inside [0, hf] have SiO2 between teeth; let's select only silicon tooth + slab domains via coordinates!
    // Even cleaner: build Si selection by explicit Union of Box selections around slab & each tooth!
    m.component("comp1").selection().create("selSlab", "Box");
    m.component("comp1").selection("selSlab").set("entitydim", 2);
    m.component("comp1").selection("selSlab").set("condition", "inside");
    m.component("comp1").selection("selSlab").set("ymin", -1e-9);
    m.component("comp1").selection("selSlab").set("ymax", {h_rem + 1e-9:.12e});

    java.util.List<String> siSels = new java.util.ArrayList<String>();
    siSels.add("selSlab");
    double[][] grv = new double[][]{{{", ".join("{" + f"{a*1e-6:.12e},{b*1e-6:.12e}" + "}" for a, b in grooves)}}};
    double zp = x0;
    for (int i = 0; i <= grv.length; i++) {{
      double a = zp;
      double b = (i < grv.length) ? grv[i][0] : x1;
      if (b - a > 1e-10) {{
        String st = "sT" + i;
        m.component("comp1").selection().create(st, "Box");
        m.component("comp1").selection(st).set("entitydim", 2);
        m.component("comp1").selection(st).set("condition", "inside");
        m.component("comp1").selection(st).set("xmin", a - 1e-9);
        m.component("comp1").selection(st).set("xmax", b + 1e-9);
        m.component("comp1").selection(st).set("ymin", {h_rem - 1e-9:.12e});
        m.component("comp1").selection(st).set("ymax", {hf + 1e-9:.12e});
        siSels.add(st);
      }}
      if (i < grv.length) zp = grv[i][1];
    }}
    m.component("comp1").selection().create("selSiAll", "Union");
    m.component("comp1").selection("selSiAll").set("entitydim", 2);
    m.component("comp1").selection("selSiAll").set("input", siSels.toArray(new String[0]));

    m.component("comp1").selection().create("selPmlR", "Box");
    m.component("comp1").selection("selPmlR").set("entitydim", 2);
    m.component("comp1").selection("selPmlR").set("condition", "inside");
    m.component("comp1").selection("selPmlR").set("xmin", {z_out - 1e-9:.12e});
    m.component("comp1").selection().create("selPmlT", "Box");
    m.component("comp1").selection("selPmlT").set("entitydim", 2);
    m.component("comp1").selection("selPmlT").set("condition", "inside");
    m.component("comp1").selection("selPmlT").set("ymin", {y_air - 1e-9:.12e});
    m.component("comp1").selection().create("selPmlB", "Box");
    m.component("comp1").selection("selPmlB").set("entitydim", 2);
    m.component("comp1").selection("selPmlB").set("condition", "inside");
    m.component("comp1").selection("selPmlB").set("ymax", {y_sub + 1e-9:.12e});
    m.component("comp1").selection().create("selPml", "Union");
    m.component("comp1").selection("selPml").set("entitydim", 2);
    m.component("comp1").selection("selPml").set("input", new String[]{{"selPmlR","selPmlT","selPmlB"}});
    m.component("comp1").coordSystem().create("pml1", "PML");
    m.component("comp1").coordSystem("pml1").selection().named("selPml");

    // Left port boundary (only oxide+core window, excluding bottom Si handle)
    m.component("comp1").selection().create("selPort", "Box");
    m.component("comp1").selection("selPort").set("entitydim", 1);
    m.component("comp1").selection("selPort").set("condition", "inside");
    m.component("comp1").selection("selPort").set("xmin", x0 - 1e-9);
    m.component("comp1").selection("selPort").set("xmax", x0 + 1e-9);
    m.component("comp1").selection("selPort").set("ymin", {-h_box - 1e-9:.12e});
    m.component("comp1").selection("selPort").set("ymax", {hf + h_top + 1e-9:.12e});

    m.component("comp1").selection().create("selUp", "Box");
    m.component("comp1").selection("selUp").set("entitydim", 1);
    m.component("comp1").selection("selUp").set("condition", "inside");
    m.component("comp1").selection("selUp").set("xmin", -1.5e-6 - 1e-9);
    m.component("comp1").selection("selUp").set("xmax", {gc_len + 1.5e-6 + 1e-9:.12e});
    m.component("comp1").selection("selUp").set("ymin", {y_up_mon - 1e-9:.12e});
    m.component("comp1").selection("selUp").set("ymax", {y_up_mon + 1e-9:.12e});

    m.component("comp1").selection().create("selDn", "Box");
    m.component("comp1").selection("selDn").set("entitydim", 1);
    m.component("comp1").selection("selDn").set("condition", "inside");
    m.component("comp1").selection("selDn").set("xmin", -1.5e-6 - 1e-9);
    m.component("comp1").selection("selDn").set("xmax", {gc_len + 1.5e-6 + 1e-9:.12e});
    m.component("comp1").selection("selDn").set("ymin", {y_dn_mon - 1e-9:.12e});
    m.component("comp1").selection("selDn").set("ymax", {y_dn_mon + 1e-9:.12e});

    m.component("comp1").physics().create("ewfd", "ElectromagneticWaves", "geom1");
    m.component("comp1").physics("ewfd").prop("components").set("components", "outofplane");
    setIso(m, "wee1", {ncl:.8f});
    m.component("comp1").physics("ewfd").create("weeSub", "WaveEquationElectric", 2);
    m.component("comp1").physics("ewfd").feature("weeSub").selection().named("selSub");
    setIso(m, "weeSub", {nc:.8f});
    m.component("comp1").physics("ewfd").create("weeAir", "WaveEquationElectric", 2);
    m.component("comp1").physics("ewfd").feature("weeAir").selection().named("selAir");
    setIso(m, "weeAir", 1.0);
    m.component("comp1").physics("ewfd").create("weeSi", "WaveEquationElectric", 2);
    m.component("comp1").physics("ewfd").feature("weeSi").selection().named("selSiAll");
    setIso(m, "weeSi", {nc:.8f});

    m.component("comp1").physics("ewfd").create("port1", "Port", 1);
    m.component("comp1").physics("ewfd").feature("port1").selection().named("selPort");
    m.component("comp1").physics("ewfd").feature("port1").set("PortType", "Numeric");
    m.component("comp1").physics("ewfd").feature("port1").set("PortExcitation", "on");
    m.component("comp1").physics("ewfd").feature("port1").set("Pin", "1[W]");

    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("size").set("hmax", 75e-9);
    m.component("comp1").mesh("mesh1").feature("size").set("hmin", 10e-9);
    m.component("comp1").mesh("mesh1").create("szSi", "Size");
    m.component("comp1").mesh("mesh1").feature("szSi").selection().named("selSiAll");
    m.component("comp1").mesh("mesh1").feature("szSi").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szSi").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szSi").set("hmax", 28e-9);
    m.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    m.component("comp1").mesh("mesh1").run();

    m.study().create("std1");
    m.study("std1").create("bma1", "BoundaryModeAnalysis");
    m.study("std1").feature("bma1").set("PortName", "1");
    m.study("std1").feature("bma1").set("modeFreq", "c_const/wl0");
    m.study("std1").feature("bma1").set("neigs", 1);
    m.study("std1").feature("bma1").set("shift", "2.84");
    m.study("std1").feature("bma1").set("shiftactive", true);
    m.study("std1").create("freq", "Frequency");
    m.study("std1").feature("freq").set("plist", "c_const/wl0");
    m.study("std1").run();

    m.result().numerical().create("gev1", "EvalGlobal");
    m.result().numerical("gev1").set("expr", new String[]{{"abs(emw.S11)^2"}});
    double s11Sq = m.result().numerical("gev1").getReal()[0][0];

    m.result().numerical().create("intUp", "IntLine");
    m.result().numerical("intUp").selection().named("selUp");
    m.result().numerical("intUp").set("expr", new String[]{{"emw.Poavy"}});
    double pUp = m.result().numerical("intUp").getReal()[0][0];

    m.result().numerical().create("intDn", "IntLine");
    m.result().numerical("intDn").selection().named("selDn");
    m.result().numerical("intDn").set("expr", new String[]{{"-emw.Poavy"}});
    double pDn = m.result().numerical("intDn").getReal()[0][0];

    m.result().export().create("datUp", "Data");
    m.result().export("datUp").set("data", "dset1");
    m.result().export("datUp").set("expr", new String[]{{"real(emw.Ez)","imag(emw.Ez)","emw.Poavy"}});
    m.result().export("datUp").set("location", "grid");
    m.result().export("datUp").set("gridstruct", "grid");
    m.result().export("datUp").set("gridx2", "range(-1.5e-6, 40e-9, " + ({gc_len + 1.5e-6:.12e}) + ")");
    m.result().export("datUp").set("gridy2", "{y_up_mon:.12e}");
    m.result().export("datUp").set("filename", "{up_file}");
    m.result().export("datUp").run();

    System.out.println("S6B_JSON:" + "{{\\"s11_sq\\":" + s11Sq + ",\\"p_up\\":" + pUp + ",\\"p_dn\\":" + pDn + "}}");
  }}
  public static void main(String[] args) {{ run(); }}
}}
'''
    t0 = time.perf_counter()
    lines = _run_comsol_java("ComsolSuite6B", java_src)
    r6b = _parse_kv_lines(lines, "S6B_JSON:")[0]

    # Compute exact SMF-28 10 deg Gaussian projection (COMSOL e^{+jwt} -> exp(-j*kz*z))
    raw = Path(up_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    xs, blocks, cur, sec = None, [], [], ""
    for ln in raw:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("%"):
            if "Grid" in s:
                sec = "grid"
            elif "Data" in s:
                if cur:
                    blocks.append(cur)
                    cur = []
                sec = "data"
            continue
        vs = [float(t) for t in s.split()]
        if sec == "grid" and xs is None:
            xs = np.array(vs, dtype=float)
        elif sec == "data":
            cur.extend(vs)
    if cur:
        blocks.append(cur)
    ez = np.array(blocks[0]) + 1j * np.array(blocks[1])
    dz = float(xs[1] - xs[0])
    mfd_z = (c["mfd_um"] * 1e-6) / math.cos(math.radians(c["fiber_angle_deg"]))
    w0 = 0.5 * mfd_z
    k0 = 2.0 * math.pi / wl_m
    kz_fib = k0 * c["n_env"] * math.sin(math.radians(c["fiber_angle_deg"]))
    best_ov = 0.0
    ez_norm = float(np.sum(np.abs(ez) ** 2) * dz)
    for zc in np.linspace(1.0e-6, gc_len - 1.0e-6, 120):
        g = np.exp(-((xs - zc) / w0) ** 2) * np.exp(-1j * kz_fib * (xs - zc))
        ov = float((abs(np.sum(ez * np.conj(g)) * dz) ** 2) / max(ez_norm * float(np.sum(np.abs(g) ** 2) * dz), 1e-30))
        if ov > best_ov:
            best_ov = ov

    # Same lateral width overlap factor as 2D FDFD & 2D FDTD
    xl = np.linspace(-30.0, 30.0, 401)
    ul = np.where(np.abs(xl) <= 0.5 * c["grating_width_um"], np.cos(math.pi * xl / c["grating_width_um"]), 0.0)
    gl = np.exp(-((xl / (0.5 * c["mfd_um"])) ** 2))
    lat_ov = float((np.sum(ul * gl)) ** 2 / (np.sum(ul ** 2) * np.sum(gl ** 2)))
    eta_fib = max(r6b["p_up"] * best_ov * lat_ov, 1e-6)
    out = {
        "case_id": c["case_id"],
        "eta_up_at_1550_pct": round(r6b["p_up"] * 100.0, 2),
        "eta_down_at_1550_pct": round(r6b["p_dn"] * 100.0, 2),
        "mode_match_2d_pct": round(best_ov * 100.0, 2),
        "il_at_1550_db": round(-10.0 * math.log10(eta_fib), 3),
        "s11_at_1550_db": round(10.0 * math.log10(max(r6b["s11_sq"], 1e-12)), 2),
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(
        f"[COMSOL Suite 6B] {c['case_id']}: up={out['eta_up_at_1550_pct']:.2f}%, dn={out['eta_down_at_1550_pct']:.2f}%, "
        f"match={out['mode_match_2d_pct']:.1f}%, IL={out['il_at_1550_db']:.3f} dB, S11={out['s11_at_1550_db']:.2f} dB"
    )
    return out


# ==============================================================================
# Block 3: Suite 8 (COMSOL AC/DC es + Full-Wave RF emw 2D & 3D Slow-Wave T-Rail)
# ==============================================================================
def run_comsol_suite8_rf(specs: dict) -> list[dict]:
    calls = []
    for c in specs["suite8_rf_electrodes"]:
        is_trail = 1 if c["electrode_topology"] == "t_rail_cpw" else 0
        is_uc = 1 if c["substrate_type"] == "undercut_si" else 0
        slab_corr = float(c["slab_corridor_half_um"] * 1e-6) if c["slab_corridor_half_um"] is not None else -1.0
        calls.append(
            f'solveRFCase("{c["case_id"]}", {is_trail}, {is_uc}, '
            f'{c["signal_width_um"]*1e-6:.12e}, {c["electrode_gap_um"]*1e-6:.12e}, {c["ground_width_um"]*1e-6:.12e}, '
            f'{c["metal_thickness_um"]*1e-6:.12e}, {c["film_height_um"]*1e-6:.12e}, {c["slab_height_um"]*1e-6:.12e}, '
            f'{c["waveguide_width_um"]*1e-6:.12e}, {c["box_thickness_um"]*1e-6:.12e}, {c["top_clad_thickness_um"]*1e-6:.12e}, '
            f'{c["t_rail_length_um"]*1e-6:.12e}, {c["t_rail_cap_width_um"]*1e-6:.12e}, {c["t_rail_duty"]:.6f}, {slab_corr:.12e}, '
            f'{c["eps_core_rf"][0]:.4f}, {c["eps_core_rf"][1]:.4f}, {c["eps_box_rf"][0]:.4f}, {c["eps_sub_rf"]:.4f}, '
            f'{c["tan_d_core"]:.6f}, {c["tan_d_box"]:.6f}, {c["tan_d_sub"]:.6f}, {c["sigma_sub_s_per_m"]:.6f}, {c["rho_metal_ohm_m"]:.6e});'
        )
    # Also run 3D Periodic Unit Cell Electrostatics for Slow-Wave T-Rail on Quartz!
    tr = [c for c in specs["suite8_rf_electrodes"] if c["case_id"] == "TFLN_SlowWave_TRail_Quartz"][0]
    calls.append(
        f'solveTRail3D("{tr["case_id"]}", {tr["signal_width_um"]*1e-6:.12e}, {tr["electrode_gap_um"]*1e-6:.12e}, '
        f'{tr["ground_width_um"]*1e-6:.12e}, {tr["metal_thickness_um"]*1e-6:.12e}, {tr["film_height_um"]*1e-6:.12e}, '
        f'{tr["slab_height_um"]*1e-6:.12e}, {tr["box_thickness_um"]*1e-6:.12e}, {tr["top_clad_thickness_um"]*1e-6:.12e}, '
        f'{tr["t_rail_length_um"]*1e-6:.12e}, {tr["t_rail_cap_width_um"]*1e-6:.12e}, {tr["t_rail_stem_width_um"]*1e-6:.12e}, '
        f'{tr["t_rail_period_um"]*1e-6:.12e}, {tr["t_rail_duty"]:.6f}, {tr["slab_corridor_half_um"]*1e-6:.12e}, '
        f'{tr["eps_core_rf"][0]:.4f}, {tr["eps_core_rf"][1]:.4f}, {tr["eps_box_rf"][0]:.4f}, {tr["eps_sub_rf"]:.4f});'
    )

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolSuite8RF {{
  static void addR(Model m, String tag, double x0, double y0, double x1, double y1) {{
    m.component("comp1").geom("geom1").create(tag, "Rectangle");
    m.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{{x0, y0}});
    m.component("comp1").geom("geom1").feature(tag).set("size", new double[]{{x1 - x0, y1 - y0}});
  }}
  static void addB3(Model m, String tag, double x0, double y0, double z0, double x1, double y1, double z1) {{
    m.component("comp1").geom("geom1").create(tag, "Block");
    m.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{{x0, y0, z0}});
    m.component("comp1").geom("geom1").feature(tag).set("size", new double[]{{x1 - x0, y1 - y0, z1 - z0}});
  }}

  // Returns [C_diel, C_vac, alpha40G_dB_cm, nRF40G, alpha10G_dB_cm, alpha100G_dB_cm] for a single 2D CPW cross-section
  static double[] solveSingle2D(double ws, double gTrunk, double wg, double tm, double hf, double hs, double wwg,
      double hbox, double htop, double xWg, int isUc, double lrBar, double wcBar, double slabCorr,
      double exCore, double eyCore, double eBox, double eSub,
      double tdCore, double tdBox, double tdSub, double sigSub, double rhoAu, int doRFModes) {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);

    double yEl0 = hs, yEl1 = hs + tm;
    double yOxTop = Math.max(hf + 0.25e-6, hs + htop);
    double span = ws + 2.0 * gTrunk;
    double hAir = Math.max(45e-6, 2.2 * span);
    double hSub = Math.max(45e-6, 2.2 * span);
    double xDom = 0.5 * ws + gTrunk + wg + 16e-6;

    // Right-half symmetry (x >= 0) -> 2x faster & identical capacitance! (Multiply half-domain energy by 2)
    //Wait: full domain is equally fast (<2s) and avoids any symmetry confusion! Use full domain.
    addR(m, "rSub", -xDom, -hbox - hSub, xDom, -hbox);
    addR(m, "rBox", -xDom, -hbox, xDom, 0.0);
    addR(m, "rTop", -xDom, 0.0, xDom, yOxTop);
    addR(m, "rAir", -xDom, yOxTop, xDom, yEl1 + hAir);

    if (isUc == 1) {{
      double rUc = Math.min(0.5 * gTrunk + 16e-6, 26e-6);
      addR(m, "rUcL", -xWg - rUc, -hbox - 26e-6, -xWg + rUc, -hbox);
      addR(m, "rUcR",  xWg - rUc, -hbox - 26e-6,  xWg + rUc, -hbox);
    }}
    if (hs > 1e-9) {{
      if (slabCorr > 0.0) {{
        addR(m, "rSlL", -xWg - slabCorr, 0.0, -xWg + slabCorr, hs);
        addR(m, "rSlR",  xWg - slabCorr, 0.0,  xWg + slabCorr, hs);
      }} else {{
        addR(m, "rSlAll", -xDom, 0.0, xDom, hs);
      }}
    }}
    addR(m, "rWgL", -xWg - 0.5*wwg, hs, -xWg + 0.5*wwg, hf);
    addR(m, "rWgR",  xWg - 0.5*wwg, hs,  xWg + 0.5*wwg, hf);

    // Metal conductors (Signal + Left/Right Grounds + optional T-cap bars)
    addR(m, "mSig", -0.5*ws, yEl0, 0.5*ws, yEl1);
    addR(m, "mGndL", -(0.5*ws + gTrunk + wg), yEl0, -(0.5*ws + gTrunk), yEl1);
    addR(m, "mGndR",   0.5*ws + gTrunk,       yEl0,   0.5*ws + gTrunk + wg, yEl1);
    if (lrBar > 1e-9) {{
      addR(m, "mScL", -(0.5*ws + lrBar), yEl0, -(0.5*ws + lrBar - wcBar), yEl1);
      addR(m, "mScR",  (0.5*ws + lrBar - wcBar), yEl0, (0.5*ws + lrBar), yEl1);
      addR(m, "mGcL", -(0.5*ws + gTrunk - lrBar + wcBar), yEl0, -(0.5*ws + gTrunk - lrBar), yEl1);
      addR(m, "mGcR",  (0.5*ws + gTrunk - lrBar), yEl0, (0.5*ws + gTrunk - lrBar + wcBar), yEl1);
    }}
    // Subtract metal interiors from dielectric geometry so boundaries are clean PEC/IBC surfaces!
    m.component("comp1").geom("geom1").create("dif1", "Difference");
    java.util.List<String> bg = new java.util.ArrayList<String>();
    bg.add("rSub"); bg.add("rBox"); bg.add("rTop"); bg.add("rAir"); bg.add("rWgL"); bg.add("rWgR");
    if (isUc == 1) {{ bg.add("rUcL"); bg.add("rUcR"); }}
    if (hs > 1e-9) {{ if (slabCorr > 0.0) {{ bg.add("rSlL"); bg.add("rSlR"); }} else {{ bg.add("rSlAll"); }} }}
    java.util.List<String> mets = new java.util.ArrayList<String>();
    mets.add("mSig"); mets.add("mGndL"); mets.add("mGndR");
    if (lrBar > 1e-9) {{ mets.add("mScL"); mets.add("mScR"); mets.add("mGcL"); mets.add("mGcR"); }}
    m.component("comp1").geom("geom1").feature("dif1").selection("input").set(bg.toArray(new String[0]));
    m.component("comp1").geom("geom1").feature("dif1").selection("input2").set(mets.toArray(new String[0]));
    m.component("comp1").geom("geom1").run();

    // Domain selections
    m.component("comp1").selection().create("selSub", "Box");
    m.component("comp1").selection("selSub").set("entitydim", 2);
    m.component("comp1").selection("selSub").set("condition", "inside");
    m.component("comp1").selection("selSub").set("ymax", -hbox + 1e-9);

    m.component("comp1").selection().create("selBox", "Box");
    m.component("comp1").selection("selBox").set("entitydim", 2);
    m.component("comp1").selection("selBox").set("condition", "inside");
    m.component("comp1").selection("selBox").set("ymin", -hbox - 1e-9);
    m.component("comp1").selection("selBox").set("ymax", 1e-9);

    m.component("comp1").selection().create("selTop", "Box");
    m.component("comp1").selection("selTop").set("entitydim", 2);
    m.component("comp1").selection("selTop").set("condition", "inside");
    m.component("comp1").selection("selTop").set("ymin", -1e-9);
    m.component("comp1").selection("selTop").set("ymax", yOxTop + 1e-9);

    java.util.List<String> lnList = new java.util.ArrayList<String>();
    m.component("comp1").selection().create("sWgL", "Box");
    m.component("comp1").selection("sWgL").set("entitydim", 2);
    m.component("comp1").selection("sWgL").set("condition", "inside");
    m.component("comp1").selection("sWgL").set("xmin", -xWg - 0.5*wwg - 1e-9);
    m.component("comp1").selection("sWgL").set("xmax", -xWg + 0.5*wwg + 1e-9);
    m.component("comp1").selection("sWgL").set("ymin", hs - 1e-9);
    m.component("comp1").selection("sWgL").set("ymax", hf + 1e-9);
    lnList.add("sWgL");
    m.component("comp1").selection().create("sWgR", "Box");
    m.component("comp1").selection("sWgR").set("entitydim", 2);
    m.component("comp1").selection("sWgR").set("condition", "inside");
    m.component("comp1").selection("sWgR").set("xmin", xWg - 0.5*wwg - 1e-9);
    m.component("comp1").selection("sWgR").set("xmax", xWg + 0.5*wwg + 1e-9);
    m.component("comp1").selection("sWgR").set("ymin", hs - 1e-9);
    m.component("comp1").selection("sWgR").set("ymax", hf + 1e-9);
    lnList.add("sWgR");
    if (hs > 1e-9) {{
      if (slabCorr > 0.0) {{
        m.component("comp1").selection().create("sSlL", "Box");
        m.component("comp1").selection("sSlL").set("entitydim", 2);
        m.component("comp1").selection("sSlL").set("condition", "inside");
        m.component("comp1").selection("sSlL").set("xmin", -xWg - slabCorr - 1e-9);
        m.component("comp1").selection("sSlL").set("xmax", -xWg + slabCorr + 1e-9);
        m.component("comp1").selection("sSlL").set("ymin", -1e-9);
        m.component("comp1").selection("sSlL").set("ymax", hs + 1e-9);
        lnList.add("sSlL");
        m.component("comp1").selection().create("sSlR", "Box");
        m.component("comp1").selection("sSlR").set("entitydim", 2);
        m.component("comp1").selection("sSlR").set("condition", "inside");
        m.component("comp1").selection("sSlR").set("xmin", xWg - slabCorr - 1e-9);
        m.component("comp1").selection("sSlR").set("xmax", xWg + slabCorr + 1e-9);
        m.component("comp1").selection("sSlR").set("ymin", -1e-9);
        m.component("comp1").selection("sSlR").set("ymax", hs + 1e-9);
        lnList.add("sSlR");
      }} else {{
        m.component("comp1").selection().create("sSlA", "Box");
        m.component("comp1").selection("sSlA").set("entitydim", 2);
        m.component("comp1").selection("sSlA").set("condition", "inside");
        m.component("comp1").selection("sSlA").set("ymin", -1e-9);
        m.component("comp1").selection("sSlA").set("ymax", hs + 1e-9);
        lnList.add("sSlA");
      }}
    }}
    m.component("comp1").selection().create("selLN", "Union");
    m.component("comp1").selection("selLN").set("entitydim", 2);
    m.component("comp1").selection("selLN").set("input", lnList.toArray(new String[0]));

    if (isUc == 1) {{
      double rUc = Math.min(0.5 * gTrunk + 16e-6, 26e-6);
      m.component("comp1").selection().create("sUcL", "Box");
      m.component("comp1").selection("sUcL").set("entitydim", 2);
      m.component("comp1").selection("sUcL").set("condition", "inside");
      m.component("comp1").selection("sUcL").set("xmin", -xWg - rUc - 1e-9);
      m.component("comp1").selection("sUcL").set("xmax", -xWg + rUc + 1e-9);
      m.component("comp1").selection("sUcL").set("ymin", -hbox - 26e-6 - 1e-9);
      m.component("comp1").selection("sUcL").set("ymax", -hbox + 1e-9);
      m.component("comp1").selection().create("sUcR", "Box");
      m.component("comp1").selection("sUcR").set("entitydim", 2);
      m.component("comp1").selection("sUcR").set("condition", "inside");
      m.component("comp1").selection("sUcR").set("xmin", xWg - rUc - 1e-9);
      m.component("comp1").selection("sUcR").set("xmax", xWg + rUc + 1e-9);
      m.component("comp1").selection("sUcR").set("ymin", -hbox - 26e-6 - 1e-9);
      m.component("comp1").selection("sUcR").set("ymax", -hbox + 1e-9);
      m.component("comp1").selection().create("selUc", "Union");
      m.component("comp1").selection("selUc").set("entitydim", 2);
      m.component("comp1").selection("selUc").set("input", new String[]{{"sUcL","sUcR"}});
    }}

    // Boundary selections for Signal and Ground metal perimeters
    java.util.List<String> bSig = new java.util.ArrayList<String>();
    java.util.List<String> bGnd = new java.util.ArrayList<String>();
    double epsB = 5e-9;
    m.component("comp1").selection().create("bs0", "Box");
    m.component("comp1").selection("bs0").set("entitydim", 1); m.component("comp1").selection("bs0").set("condition", "inside");
    m.component("comp1").selection("bs0").set("xmin", -0.5*ws - epsB); m.component("comp1").selection("bs0").set("xmax", 0.5*ws + epsB);
    m.component("comp1").selection("bs0").set("ymin", yEl0 - epsB); m.component("comp1").selection("bs0").set("ymax", yEl1 + epsB);
    bSig.add("bs0");

    m.component("comp1").selection().create("bgL", "Box");
    m.component("comp1").selection("bgL").set("entitydim", 1); m.component("comp1").selection("bgL").set("condition", "inside");
    m.component("comp1").selection("bgL").set("xmin", -(0.5*ws+gTrunk+wg) - epsB); m.component("comp1").selection("bgL").set("xmax", -(0.5*ws+gTrunk) + epsB);
    m.component("comp1").selection("bgL").set("ymin", yEl0 - epsB); m.component("comp1").selection("bgL").set("ymax", yEl1 + epsB);
    bGnd.add("bgL");

    m.component("comp1").selection().create("bgR", "Box");
    m.component("comp1").selection("bgR").set("entitydim", 1); m.component("comp1").selection("bgR").set("condition", "inside");
    m.component("comp1").selection("bgR").set("xmin", (0.5*ws+gTrunk) - epsB); m.component("comp1").selection("bgR").set("xmax", (0.5*ws+gTrunk+wg) + epsB);
    m.component("comp1").selection("bgR").set("ymin", yEl0 - epsB); m.component("comp1").selection("bgR").set("ymax", yEl1 + epsB);
    bGnd.add("bgR");

    if (lrBar > 1e-9) {{
      double[][] sC = new double[][]{{{{-(0.5*ws+lrBar), -(0.5*ws+lrBar-wcBar)}}, {{(0.5*ws+lrBar-wcBar), (0.5*ws+lrBar)}}}};
      for (int i = 0; i < 2; i++) {{
        String nm = "bsC" + i;
        m.component("comp1").selection().create(nm, "Box");
        m.component("comp1").selection(nm).set("entitydim", 1); m.component("comp1").selection(nm).set("condition", "inside");
        m.component("comp1").selection(nm).set("xmin", sC[i][0] - epsB); m.component("comp1").selection(nm).set("xmax", sC[i][1] + epsB);
        m.component("comp1").selection(nm).set("ymin", yEl0 - epsB); m.component("comp1").selection(nm).set("ymax", yEl1 + epsB);
        bSig.add(nm);
      }}
      double[][] gC = new double[][]{{{{-(0.5*ws+gTrunk-lrBar+wcBar), -(0.5*ws+gTrunk-lrBar)}}, {{(0.5*ws+gTrunk-lrBar), (0.5*ws+gTrunk-lrBar+wcBar)}}}};
      for (int i = 0; i < 2; i++) {{
        String nm = "bgC" + i;
        m.component("comp1").selection().create(nm, "Box");
        m.component("comp1").selection(nm).set("entitydim", 1); m.component("comp1").selection(nm).set("condition", "inside");
        m.component("comp1").selection(nm).set("xmin", gC[i][0] - epsB); m.component("comp1").selection(nm).set("xmax", gC[i][1] + epsB);
        m.component("comp1").selection(nm).set("ymin", yEl0 - epsB); m.component("comp1").selection(nm).set("ymax", yEl1 + epsB);
        bGnd.add(nm);
      }}
    }}
    m.component("comp1").selection().create("selSigB", "Union");
    m.component("comp1").selection("selSigB").set("entitydim", 1);
    m.component("comp1").selection("selSigB").set("input", bSig.toArray(new String[0]));
    m.component("comp1").selection().create("selGndB", "Union");
    m.component("comp1").selection("selGndB").set("entitydim", 1);
    m.component("comp1").selection("selGndB").set("input", bGnd.toArray(new String[0]));
    m.component("comp1").selection().create("selAllMetB", "Union");
    m.component("comp1").selection("selAllMetB").set("entitydim", 1);
    m.component("comp1").selection("selAllMetB").set("input", new String[]{{"selSigB","selGndB"}});

    // Mesh
    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("size").set("hmax", 6.0e-6);
    m.component("comp1").mesh("mesh1").feature("size").set("hmin", 0.04e-6);
    m.component("comp1").mesh("mesh1").create("szLN", "Size");
    m.component("comp1").mesh("mesh1").feature("szLN").selection().named("selLN");
    m.component("comp1").mesh("mesh1").feature("szLN").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szLN").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szLN").set("hmax", 0.14e-6);
    m.component("comp1").mesh("mesh1").create("szM", "Size");
    m.component("comp1").mesh("mesh1").feature("szM").selection().geom("geom1", 1);
    m.component("comp1").mesh("mesh1").feature("szM").selection().named("selAllMetB");
    m.component("comp1").mesh("mesh1").feature("szM").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szM").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szM").set("hmax", 0.22e-6);
    m.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    m.component("comp1").mesh("mesh1").run();

    // Physics 1: AC/DC Electrostatics (Dielectric & Vacuum)
    m.component("comp1").physics().create("es", "Electrostatics", "geom1");
    m.component("comp1").physics("es").create("ccSub", "ChargeConservation", 2);
    m.component("comp1").physics("es").feature("ccSub").selection().named("selSub");
    m.component("comp1").physics("es").feature("ccSub").set("epsilonr_mat", "userdef");
    m.component("comp1").physics("es").feature("ccSub").set("epsilonr", new String[]{{String.valueOf(eSub),"0","0","0",String.valueOf(eSub),"0","0","0",String.valueOf(eSub)}});

    m.component("comp1").physics("es").create("ccBox", "ChargeConservation", 2);
    m.component("comp1").physics("es").feature("ccBox").selection().named("selBox");
    m.component("comp1").physics("es").feature("ccBox").set("epsilonr_mat", "userdef");
    m.component("comp1").physics("es").feature("ccBox").set("epsilonr", new String[]{{String.valueOf(eBox),"0","0","0",String.valueOf(eBox),"0","0","0",String.valueOf(eBox)}});

    m.component("comp1").physics("es").create("ccTop", "ChargeConservation", 2);
    m.component("comp1").physics("es").feature("ccTop").selection().named("selTop");
    m.component("comp1").physics("es").feature("ccTop").set("epsilonr_mat", "userdef");
    m.component("comp1").physics("es").feature("ccTop").set("epsilonr", new String[]{{String.valueOf(eBox),"0","0","0",String.valueOf(eBox),"0","0","0",String.valueOf(eBox)}});

    m.component("comp1").physics("es").create("ccLN", "ChargeConservation", 2);
    m.component("comp1").physics("es").feature("ccLN").selection().named("selLN");
    m.component("comp1").physics("es").feature("ccLN").set("epsilonr_mat", "userdef");
    m.component("comp1").physics("es").feature("ccLN").set("epsilonr", new String[]{{String.valueOf(exCore),"0","0","0",String.valueOf(eyCore),"0","0","0",String.valueOf(exCore)}});

    if (isUc == 1) {{
      m.component("comp1").physics("es").create("ccUc", "ChargeConservation", 2);
      m.component("comp1").physics("es").feature("ccUc").selection().named("selUc");
      m.component("comp1").physics("es").feature("ccUc").set("epsilonr_mat", "userdef");
      m.component("comp1").physics("es").feature("ccUc").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    }}

    m.component("comp1").physics("es").create("pot1", "ElectricPotential", 1);
    m.component("comp1").physics("es").feature("pot1").selection().named("selSigB");
    m.component("comp1").physics("es").feature("pot1").set("V0", "1[V]");
    m.component("comp1").physics("es").create("gnd1", "Ground", 1);
    m.component("comp1").physics("es").feature("gnd1").selection().named("selGndB");

    m.study().create("stdDiel");
    m.study("stdDiel").create("stat", "Stationary");
    m.study("stdDiel").run();
    m.result().numerical().create("intW", "IntSurface");
    m.result().numerical("intW").selection().all();
    m.result().numerical("intW").set("expr", new String[]{{"2*es.We"}});
    double cDiel = m.result().numerical("intW").getReal()[0][0];

    // Vacuum capacitance (override all eps_r = 1)
    m.component("comp1").physics("es").feature("ccSub").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    m.component("comp1").physics("es").feature("ccBox").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    m.component("comp1").physics("es").feature("ccTop").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    m.component("comp1").physics("es").feature("ccLN").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    m.study("stdDiel").run();
    double cVac = m.result().numerical("intW").getReal()[0][0];

    double a10 = 0.0, a40 = 0.0, a100 = 0.0, n40 = Math.sqrt(cDiel / cVac);
    if (doRFModes == 1) {{
      // Physics 2: COMSOL RF Module (emw) 2D Full-Wave Lossy Mode Analysis (Au IBC + complex eps_r)
      m.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
      // Default Air
      m.component("comp1").physics("emw").feature("wee1").set("DisplacementFieldModel", "RelativePermittivity");
      m.component("comp1").physics("emw").feature("wee1").set("epsilonr_mat", "userdef");
      m.component("comp1").physics("emw").feature("wee1").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
      m.component("comp1").physics("emw").feature("wee1").set("mur_mat", "userdef");
      m.component("comp1").physics("emw").feature("wee1").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
      m.component("comp1").physics("emw").feature("wee1").set("sigma_mat", "userdef");
      m.component("comp1").physics("emw").feature("wee1").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

      // Substrate (lossy + conductivity)
      m.component("comp1").physics("emw").create("wSub", "WaveEquationElectric", 2);
      m.component("comp1").physics("emw").feature("wSub").selection().named("selSub");
      m.component("comp1").physics("emw").feature("wSub").set("DisplacementFieldModel", "RelativePermittivity");
      m.component("comp1").physics("emw").feature("wSub").set("epsilonr_mat", "userdef");
      String esSub = "(" + eSub + ")*(1-j*(" + tdSub + "))";
      m.component("comp1").physics("emw").feature("wSub").set("epsilonr", new String[]{{esSub,"0","0","0",esSub,"0","0","0",esSub}});
      m.component("comp1").physics("emw").feature("wSub").set("mur_mat", "userdef");
      m.component("comp1").physics("emw").feature("wSub").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
      m.component("comp1").physics("emw").feature("wSub").set("sigma_mat", "userdef");
      m.component("comp1").physics("emw").feature("wSub").set("sigma", new String[]{{String.valueOf(sigSub),"0","0","0",String.valueOf(sigSub),"0","0","0",String.valueOf(sigSub)}});

      // BOX & Top oxide
      for (String[] pair : new String[][]{{{{"wBox","selBox"}},{{"wTop","selTop"}}}}) {{
        m.component("comp1").physics("emw").create(pair[0], "WaveEquationElectric", 2);
        m.component("comp1").physics("emw").feature(pair[0]).selection().named(pair[1]);
        m.component("comp1").physics("emw").feature(pair[0]).set("DisplacementFieldModel", "RelativePermittivity");
        m.component("comp1").physics("emw").feature(pair[0]).set("epsilonr_mat", "userdef");
        String eOx = "(" + eBox + ")*(1-j*(" + tdBox + "))";
        m.component("comp1").physics("emw").feature(pair[0]).set("epsilonr", new String[]{{eOx,"0","0","0",eOx,"0","0","0",eOx}});
        m.component("comp1").physics("emw").feature(pair[0]).set("mur_mat", "userdef");
        m.component("comp1").physics("emw").feature(pair[0]).set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
        m.component("comp1").physics("emw").feature(pair[0]).set("sigma_mat", "userdef");
        m.component("comp1").physics("emw").feature(pair[0]).set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});
      }}
      // Anisotropic TFLN
      m.component("comp1").physics("emw").create("wLN", "WaveEquationElectric", 2);
      m.component("comp1").physics("emw").feature("wLN").selection().named("selLN");
      m.component("comp1").physics("emw").feature("wLN").set("DisplacementFieldModel", "RelativePermittivity");
      m.component("comp1").physics("emw").feature("wLN").set("epsilonr_mat", "userdef");
      String eLx = "(" + exCore + ")*(1-j*(" + tdCore + "))";
      String eLy = "(" + eyCore + ")*(1-j*(" + tdCore + "))";
      m.component("comp1").physics("emw").feature("wLN").set("epsilonr", new String[]{{eLx,"0","0","0",eLy,"0","0","0",eLx}});
      m.component("comp1").physics("emw").feature("wLN").set("mur_mat", "userdef");
      m.component("comp1").physics("emw").feature("wLN").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
      m.component("comp1").physics("emw").feature("wLN").set("sigma_mat", "userdef");
      m.component("comp1").physics("emw").feature("wLN").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});
      if (isUc == 1) {{
        m.component("comp1").physics("emw").create("wUc", "WaveEquationElectric", 2);
        m.component("comp1").physics("emw").feature("wUc").selection().named("selUc");
        m.component("comp1").physics("emw").feature("wUc").set("DisplacementFieldModel", "RelativePermittivity");
        m.component("comp1").physics("emw").feature("wUc").set("epsilonr_mat", "userdef");
        m.component("comp1").physics("emw").feature("wUc").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
        m.component("comp1").physics("emw").feature("wUc").set("mur_mat", "userdef");
        m.component("comp1").physics("emw").feature("wUc").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
        m.component("comp1").physics("emw").feature("wUc").set("sigma_mat", "userdef");
        m.component("comp1").physics("emw").feature("wUc").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});
      }}

      // Finite-conductivity Au Impedance Boundary Condition on all CPW conductor boundaries!
      double sigAu = 1.0 / rhoAu;
      m.component("comp1").physics("emw").create("ibc1", "Impedance", 1);
      m.component("comp1").physics("emw").feature("ibc1").selection().named("selAllMetB");
      m.component("comp1").physics("emw").feature("ibc1").set("epsilonr_mat", "userdef");
      m.component("comp1").physics("emw").feature("ibc1").set("epsilonr", "1");
      m.component("comp1").physics("emw").feature("ibc1").set("murbnd_mat", "userdef");
      m.component("comp1").physics("emw").feature("ibc1").set("murbnd", "1");
      m.component("comp1").physics("emw").feature("ibc1").set("sigmabnd_mat", "userdef");
      m.component("comp1").physics("emw").feature("ibc1").set("sigmabnd", String.valueOf(sigAu));

      double nGuess = Math.sqrt(cDiel / cVac);
      double[] freqsGHz = new double[]{{10.0, 40.0, 100.0}};
      double[] alphas = new double[3];
      for (int fi = 0; fi < 3; fi++) {{
        String stTag = "stdRF" + fi;
        m.study().create(stTag);
        m.study(stTag).create("mode", "ModeAnalysis");
        m.study(stTag).feature("mode").activate("es", false);
        m.study(stTag).feature("mode").set("modeFreq", freqsGHz[fi] + "[GHz]");
        m.study(stTag).feature("mode").set("neigs", 4);
        m.study(stTag).feature("mode").set("shift", String.valueOf(nGuess * 1.05));
        m.study(stTag).feature("mode").set("shiftactive", true);
        m.study(stTag).run();

        String gTag = "gevRF" + fi;
        m.result().numerical().create(gTag, "EvalGlobal");
        m.result().numerical(gTag).set("data", "dset" + (fi + 2));
        m.result().numerical(gTag).set("expr", new String[]{{"real(emw.neff)", "abs(imag(emw.neff))*2*pi*emw.freq/c_const"}});
        double[][] res = m.result().numerical(gTag).getReal();
        // Pick quasi-TEM even mode closest to nGuess (note: CPW has 2 quasi-TEM modes: even & odd; pick closest to nGuess with lowest loss or check symmetry!)
        int pick = 0; double bestErr = 1e9;
        for (int k = 0; k < res[0].length; k++) {{
          double nr = res[0][k];
          double err = Math.abs(nr - nGuess);
          if (nr > 1.0 && err < bestErr) {{ bestErr = err; pick = k; }}
        }}
        if (fi == 1) n40 = res[0][pick];
        alphas[fi] = res[1][pick] * 8.6858896 * 1e-2; // Np/m -> dB/cm
      }}
      a10 = alphas[0]; a40 = alphas[1]; a100 = alphas[2];
    }}
    return new double[]{{cDiel, cVac, a40, n40, a10, a100}};
  }}

  static void solveRFCase(String cid, int isTrail, int isUc, double ws, double gIn, double wg, double tm,
      double hf, double hs, double wwg, double hbox, double htop, double lr, double wc, double duty, double slabCorr,
      double exCore, double eyCore, double eBox, double eSub,
      double tdCore, double tdBox, double tdSub, double sigSub, double rhoAu) {{
    double c0 = 299792458.0;
    if (isTrail == 1) {{
      double gTrunk = gIn + 2.0 * lr;
      double xWg = 0.5 * ws + lr + 0.5 * gIn;
      double[] tr = solveSingle2D(ws, gTrunk, wg, tm, hf, hs, wwg, hbox, htop, xWg, isUc, 0.0, 0.0, slabCorr,
                                  exCore, eyCore, eBox, eSub, tdCore, tdBox, tdSub, sigSub, rhoAu, 1);
      double[] cp = solveSingle2D(ws, gTrunk, wg, tm, hf, hs, wwg, hbox, htop, xWg, isUc, lr, wc, slabCorr,
                                  exCore, eyCore, eBox, eSub, tdCore, tdBox, tdSub, sigSub, rhoAu, 0);
      double cTot = duty * cp[0] + (1.0 - duty) * tr[0];
      double lExt = 1.0 / (c0 * c0 * tr[1]);
      double z0 = Math.sqrt(lExt / cTot);
      double nrf = c0 * Math.sqrt(lExt * cTot);
      System.out.println("S8_JSON:" + "{{\\"case_id\\":\\"" + cid + "\\",\\"Z0_2d_es\\":" + z0 + ",\\"n_RF_2d_es\\":" + nrf +
          ",\\"C_diel_pF_m\\":" + (cTot*1e12) + ",\\"L_ext_nH_m\\":" + (lExt*1e9) +
          ",\\"alpha_10G_emw\\":" + tr[4] + ",\\"alpha_40G_emw\\":" + tr[2] + ",\\"alpha_100G_emw\\":" + tr[5] +
          ",\\"n_RF_40G_emw\\":" + tr[3] + "}}");
    }} else {{
      double xWg = 0.5 * ws + 0.5 * gIn;
      double[] r = solveSingle2D(ws, gIn, wg, tm, hf, hs, wwg, hbox, htop, xWg, isUc, 0.0, 0.0, slabCorr,
                                 exCore, eyCore, eBox, eSub, tdCore, tdBox, tdSub, sigSub, rhoAu, 1);
      double lExt = 1.0 / (c0 * c0 * r[1]);
      double z0 = Math.sqrt(lExt / r[0]);
      double nrf = c0 * Math.sqrt(lExt * r[0]);
      System.out.println("S8_JSON:" + "{{\\"case_id\\":\\"" + cid + "\\",\\"Z0_2d_es\\":" + z0 + ",\\"n_RF_2d_es\\":" + nrf +
          ",\\"C_diel_pF_m\\":" + (r[0]*1e12) + ",\\"L_ext_nH_m\\":" + (lExt*1e9) +
          ",\\"alpha_10G_emw\\":" + r[4] + ",\\"alpha_40G_emw\\":" + r[2] + ",\\"alpha_100G_emw\\":" + r[5] +
          ",\\"n_RF_40G_emw\\":" + r[3] + "}}");
    }}
  }}

  // True 3D Periodic Unit Cell (Lambda = 25 um) Electrostatic Slow-Wave T-Rail Validation!
  static void solveTRail3D(String cid, double ws, double gIn, double wg, double tm, double hf, double hs,
      double hbox, double htop, double lr, double wc, double wStem, double period, double duty, double slabCorr,
      double exCore, double eyCore, double eBox, double eSub) {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 3);

    // Exploit x >= 0 symmetry AND z in [0, period/2] symmetry (quarter-cell -> 4x faster, exact Neumann symmetry!)
    double gTrunk = gIn + 2.0 * lr;
    double xWg = 0.5 * ws + lr + 0.5 * gIn;
    double xDom = 0.5 * ws + gTrunk + wg + 16e-6;
    double yEl0 = hs, yEl1 = hs + tm;
    double yOxTop = Math.max(hf + 0.25e-6, hs + htop);
    double hAir = 110e-6, hSub = 110e-6;
    double zHalf = 0.5 * period;
    double zCapHalf = 0.5 * duty * period;
    double zStemHalf = 0.5 * wStem;

    addB3(m, "bSub", 0.0, -hbox - hSub, 0.0, xDom, -hbox, zHalf);
    addB3(m, "bBox", 0.0, -hbox, 0.0, xDom, 0.0, zHalf);
    addB3(m, "bTop", 0.0, 0.0, 0.0, xDom, yOxTop, zHalf);
    addB3(m, "bAir", 0.0, yOxTop, 0.0, xDom, yEl1 + hAir, zHalf);
    addB3(m, "bSlab", xWg - slabCorr, 0.0, 0.0, xWg + slabCorr, hs, zHalf);
    // Right-half conductors: Signal trunk + Signal T-stem + Signal T-cap; Ground T-cap + Ground T-stem + Ground plane
    addB3(m, "mSig", 0.0, yEl0, 0.0, 0.5*ws, yEl1, zHalf);
    addB3(m, "mSStem", 0.5*ws, yEl0, 0.0, 0.5*ws + lr - wc, yEl1, zStemHalf);
    addB3(m, "mSCap", 0.5*ws + lr - wc, yEl0, 0.0, 0.5*ws + lr, yEl1, zCapHalf);
    addB3(m, "mGCap", 0.5*ws + gTrunk - lr, yEl0, 0.0, 0.5*ws + gTrunk - lr + wc, yEl1, zCapHalf);
    addB3(m, "mGStem", 0.5*ws + gTrunk - lr + wc, yEl0, 0.0, 0.5*ws + gTrunk, yEl1, zStemHalf);
    addB3(m, "mGnd", 0.5*ws + gTrunk, yEl0, 0.0, 0.5*ws + gTrunk + wg, yEl1, zHalf);

    m.component("comp1").geom("geom1").create("dif1", "Difference");
    m.component("comp1").geom("geom1").feature("dif1").selection("input").set(new String[]{{"bSub","bBox","bTop","bAir","bSlab"}});
    m.component("comp1").geom("geom1").feature("dif1").selection("input2").set(new String[]{{"mSig","mSStem","mSCap","mGCap","mGStem","mGnd"}});
    m.component("comp1").geom("geom1").run();

    m.component("comp1").selection().create("selSub", "Box"); m.component("comp1").selection("selSub").set("entitydim", 3);
    m.component("comp1").selection("selSub").set("condition", "inside"); m.component("comp1").selection("selSub").set("ymax", -hbox + 1e-9);

    m.component("comp1").selection().create("selOx", "Box"); m.component("comp1").selection("selOx").set("entitydim", 3);
    m.component("comp1").selection("selOx").set("condition", "inside"); m.component("comp1").selection("selOx").set("ymin", -hbox - 1e-9);
    m.component("comp1").selection("selOx").set("ymax", yOxTop + 1e-9);

    m.component("comp1").selection().create("selLN", "Box"); m.component("comp1").selection("selLN").set("entitydim", 3);
    m.component("comp1").selection("selLN").set("condition", "inside");
    m.component("comp1").selection("selLN").set("xmin", xWg - slabCorr - 1e-9); m.component("comp1").selection("selLN").set("xmax", xWg + slabCorr + 1e-9);
    m.component("comp1").selection("selLN").set("ymin", -1e-9); m.component("comp1").selection("selLN").set("ymax", hs + 1e-9);

    double ep = 5e-9;
    m.component("comp1").selection().create("selSigB", "Box"); m.component("comp1").selection("selSigB").set("entitydim", 2);
    m.component("comp1").selection("selSigB").set("condition", "inside");
    m.component("comp1").selection("selSigB").set("xmin", -ep); m.component("comp1").selection("selSigB").set("xmax", 0.5*ws + lr + ep);
    m.component("comp1").selection("selSigB").set("ymin", yEl0 - ep); m.component("comp1").selection("selSigB").set("ymax", yEl1 + ep);

    m.component("comp1").selection().create("selGndB", "Box"); m.component("comp1").selection("selGndB").set("entitydim", 2);
    m.component("comp1").selection("selGndB").set("condition", "inside");
    m.component("comp1").selection("selGndB").set("xmin", 0.5*ws + gTrunk - lr - ep); m.component("comp1").selection("selGndB").set("xmax", 0.5*ws + gTrunk + wg + ep);
    m.component("comp1").selection("selGndB").set("ymin", yEl0 - ep); m.component("comp1").selection("selGndB").set("ymax", yEl1 + ep);

    m.component("comp1").physics().create("es", "Electrostatics", "geom1");
    m.component("comp1").physics("es").create("cSub", "ChargeConservation", 3);
    m.component("comp1").physics("es").feature("cSub").selection().named("selSub");
    m.component("comp1").physics("es").feature("cSub").set("epsilonr_mat", "userdef");
    m.component("comp1").physics("es").feature("cSub").set("epsilonr", new String[]{{String.valueOf(eSub),"0","0","0",String.valueOf(eSub),"0","0","0",String.valueOf(eSub)}});

    m.component("comp1").physics("es").create("cOx", "ChargeConservation", 3);
    m.component("comp1").physics("es").feature("cOx").selection().named("selOx");
    m.component("comp1").physics("es").feature("cOx").set("epsilonr_mat", "userdef");
    m.component("comp1").physics("es").feature("cOx").set("epsilonr", new String[]{{String.valueOf(eBox),"0","0","0",String.valueOf(eBox),"0","0","0",String.valueOf(eBox)}});

    m.component("comp1").physics("es").create("cLN", "ChargeConservation", 3);
    m.component("comp1").physics("es").feature("cLN").selection().named("selLN");
    m.component("comp1").physics("es").feature("cLN").set("epsilonr_mat", "userdef");
    m.component("comp1").physics("es").feature("cLN").set("epsilonr", new String[]{{String.valueOf(exCore),"0","0","0",String.valueOf(eyCore),"0","0","0",String.valueOf(exCore)}});

    m.component("comp1").physics("es").create("pot1", "ElectricPotential", 2);
    m.component("comp1").physics("es").feature("pot1").selection().named("selSigB");
    m.component("comp1").physics("es").feature("pot1").set("V0", "1[V]");
    m.component("comp1").physics("es").create("gnd1", "Ground", 2);
    m.component("comp1").physics("es").feature("gnd1").selection().named("selGndB");

    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("hauto", 4);
    m.component("comp1").mesh("mesh1").create("ftet1", "FreeTet");
    m.component("comp1").mesh("mesh1").run();

    m.study().create("std3d");
    m.study("std3d").create("stat", "Stationary");
    m.study("std3d").run();
    m.result().numerical().create("intW3", "IntVolume");
    m.result().numerical("intW3").selection().all();
    m.result().numerical("intW3").set("expr", new String[]{{"2*es.We"}});
    // Quarter-cell: multiply by 4 / period to get full capacitance per unit length [F/m]
    double c3dDiel = 4.0 * m.result().numerical("intW3").getReal()[0][0] / period;

    System.out.println("S8_3D_JSON:" + "{{\\"case_id\\":\\"" + cid + "\\",\\"C_3d_diel_pF_m\\":" + (c3dDiel*1e12) + "}}");
  }}

  public static void run() {{
    {chr(10).join("    " + ln for ln in calls)}
  }}
  public static void main(String[] args) {{ run(); }}
}}
'''
    lines = _run_comsol_java("ComsolSuite8RF", java_src)
    rows = _parse_kv_lines(lines, "S8_JSON:")
    r3d = _parse_kv_lines(lines, "S8_3D_JSON:")
    by3d = {r["case_id"]: r for r in r3d}
    for r in rows:
        if r["case_id"] in by3d:
            c3d = by3d[r["case_id"]]["C_3d_diel_pF_m"] * 1e-12
            l_ext = r["L_ext_nH_m"] * 1e-9
            r["C_3d_diel_pF_m"] = round(c3d * 1e12, 3)
            r["Z0_3d_unitcell_ohm"] = round(math.sqrt(l_ext / c3d), 3)
            r["n_RF_3d_unitcell"] = round(C0 * math.sqrt(l_ext * c3d), 4)
        for k in list(r.keys()):
            if isinstance(r[k], float):
                r[k] = round(r[k], 4)
        print(
            f"[COMSOL Suite 8] {r['case_id']}: Z0={r['Z0_2d_es']:.2f} Ohm, n_RF={r['n_RF_2d_es']:.4f} "
            f"(emw 40G n_RF={r['n_RF_40G_emw']:.4f}, a_40G={r['alpha_40G_emw']:.2f} dB/cm)"
            + (f" | 3D unit-cell Z0={r['Z0_3d_unitcell_ohm']:.2f}, n_RF={r['n_RF_3d_unitcell']:.4f}" if "Z0_3d_unitcell_ohm" in r else "")
        )
    return rows


# ==============================================================================
# Block 4: Suite 9 (COMSOL 2D Nonlinear Poisson-Boltzmann PDE + Coupled ewfd)
# ==============================================================================
def run_comsol_suite9_pn_tcad(specs: dict) -> list[dict]:
    calls = []
    for c in specs["suite9_si_pn_tcad"]:
        jcode = {"lateral_pn": 0, "l_shaped_pn": 1, "u_shaped_pn": 2}[c["junction_type"]]
        nc = c["core_index"]["center"]["nx"]
        ncl = c["clad_index"]["center"]["nx"]
        calls.append(
            f'solvePN("{c["case_id"]}", {jcode}, {c["wavelength_nm"]*1e-9:.12e}, '
            f'{c["waveguide_width_um"]*1e-6:.12e}, {c["film_height_um"]*1e-6:.12e}, {c["slab_height_um"]*1e-6:.12e}, '
            f'{c["pn_offset_nm"]*1e-9:.12e}, {c["l_vertical_y_nm"]*1e-9:.12e}, {c["implant_straggle_nm"]*1e-9:.12e}, '
            f'{c["inter_implant_offset_um"]*1e-6:.12e}, {c["heavy_contact_offset_um"]*1e-6:.12e}, '
            f'{c["na_core_cm3"]*1e6:.6e}, {c["nd_core_cm3"]*1e6:.6e}, {c["na_inter_cm3"]*1e6:.6e}, {c["nd_inter_cm3"]*1e6:.6e}, '
            f'{c["n_heavy_contact_cm3"]*1e6:.6e}, {nc:.8f}, {ncl:.8f});'
        )

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolSuite9PN {{
  static void solvePN(String cid, int jcode, double wl, double w, double hf, double hs,
      double xj, double yj, double sig, double xInter, double dCont,
      double naCore, double ndCore, double naInt, double ndInt, double nHeavy, double nSi, double nOx) {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);

    double xDom = dCont + 0.55e-6;
    double padY = 1.60e-6;
    m.param().set("wl0", wl + "[m]");
    m.param().set("Vt", "0.02585199[V]");
    m.param().set("ni", "1.0e16[1/m^3]");
    m.param().set("q_e", "1.602176634e-19[C]");
    m.param().set("Vbias", "0[V]");
    m.param().set("s2", (Math.sqrt(2.0) * sig) + "[m]");

    // Geometry: Full cladding box + Si Rib + Si Slab
    m.component("comp1").geom("geom1").create("rOx", "Rectangle");
    m.component("comp1").geom("geom1").feature("rOx").set("pos", new double[]{{-xDom - 0.8e-6, -padY}});
    m.component("comp1").geom("geom1").feature("rOx").set("size", new double[]{{2.0*(xDom + 0.8e-6), 2.0*padY + hf}});
    m.component("comp1").geom("geom1").create("rSlab", "Rectangle");
    m.component("comp1").geom("geom1").feature("rSlab").set("pos", new double[]{{-xDom, 0.0}});
    m.component("comp1").geom("geom1").feature("rSlab").set("size", new double[]{{2.0*xDom, hs}});
    m.component("comp1").geom("geom1").create("rRib", "Rectangle");
    m.component("comp1").geom("geom1").feature("rRib").set("pos", new double[]{{-0.5*w, hs}});
    m.component("comp1").geom("geom1").feature("rRib").set("size", new double[]{{w, hf - hs}});
    // Contact segment splits at +/- (dCont + 0.05 um)
    m.component("comp1").geom("geom1").create("lsPL", "LineSegment");
    m.component("comp1").geom("geom1").feature("lsPL").set("specify1", "coord");
    m.component("comp1").geom("geom1").feature("lsPL").set("specify2", "coord");
    m.component("comp1").geom("geom1").feature("lsPL").set("coord1", new double[]{{-dCont - 0.05e-6, 0.0}});
    m.component("comp1").geom("geom1").feature("lsPL").set("coord2", new double[]{{-dCont - 0.05e-6, hs}});
    m.component("comp1").geom("geom1").create("lsNR", "LineSegment");
    m.component("comp1").geom("geom1").feature("lsNR").set("specify1", "coord");
    m.component("comp1").geom("geom1").feature("lsNR").set("specify2", "coord");
    m.component("comp1").geom("geom1").feature("lsNR").set("coord1", new double[]{{dCont + 0.05e-6, 0.0}});
    m.component("comp1").geom("geom1").feature("lsNR").set("coord2", new double[]{{dCont + 0.05e-6, hs}});
    // Active junction integration mask cuts at +/- (dCont - 0.12 um)
    m.component("comp1").geom("geom1").create("lsJL", "LineSegment");
    m.component("comp1").geom("geom1").feature("lsJL").set("specify1", "coord");
    m.component("comp1").geom("geom1").feature("lsJL").set("specify2", "coord");
    m.component("comp1").geom("geom1").feature("lsJL").set("coord1", new double[]{{-(dCont - 0.12e-6), 0.0}});
    m.component("comp1").geom("geom1").feature("lsJL").set("coord2", new double[]{{-(dCont - 0.12e-6), hs}});
    m.component("comp1").geom("geom1").create("lsJR", "LineSegment");
    m.component("comp1").geom("geom1").feature("lsJR").set("specify1", "coord");
    m.component("comp1").geom("geom1").feature("lsJR").set("specify2", "coord");
    m.component("comp1").geom("geom1").feature("lsJR").set("coord1", new double[]{{+(dCont - 0.12e-6), 0.0}});
    m.component("comp1").geom("geom1").feature("lsJR").set("coord2", new double[]{{+(dCont - 0.12e-6), hs}});
    m.component("comp1").geom("geom1").run();

    // Selections
    m.component("comp1").selection().create("selSlab", "Box"); m.component("comp1").selection("selSlab").set("entitydim", 2);
    m.component("comp1").selection("selSlab").set("condition", "inside");
    m.component("comp1").selection("selSlab").set("xmin", -xDom - 1e-9); m.component("comp1").selection("selSlab").set("xmax", xDom + 1e-9);
    m.component("comp1").selection("selSlab").set("ymin", -1e-9); m.component("comp1").selection("selSlab").set("ymax", hs + 1e-9);

    m.component("comp1").selection().create("selRib", "Box"); m.component("comp1").selection("selRib").set("entitydim", 2);
    m.component("comp1").selection("selRib").set("condition", "inside");
    m.component("comp1").selection("selRib").set("xmin", -0.5*w - 1e-9); m.component("comp1").selection("selRib").set("xmax", 0.5*w + 1e-9);
    m.component("comp1").selection("selRib").set("ymin", hs - 1e-9); m.component("comp1").selection("selRib").set("ymax", hf + 1e-9);

    m.component("comp1").selection().create("selSi", "Union"); m.component("comp1").selection("selSi").set("entitydim", 2);
    m.component("comp1").selection("selSi").set("input", new String[]{{"selSlab","selRib"}});

    // Active junction Silicon region |x| <= dCont - 0.12 um
    m.component("comp1").selection().create("selJunc", "Box"); m.component("comp1").selection("selJunc").set("entitydim", 2);
    m.component("comp1").selection("selJunc").set("condition", "inside");
    m.component("comp1").selection("selJunc").set("xmin", -(dCont - 0.12e-6) - 1e-9);
    m.component("comp1").selection("selJunc").set("xmax",  (dCont - 0.12e-6) + 1e-9);
    m.component("comp1").selection("selJunc").set("ymin", -1e-9); m.component("comp1").selection("selJunc").set("ymax", hf + 1e-9);

    // Ohmic contact boundaries on left P++ and right N++ slab regions
    m.component("comp1").selection().create("selPCont", "Box"); m.component("comp1").selection("selPCont").set("entitydim", 1);
    m.component("comp1").selection("selPCont").set("condition", "inside");
    m.component("comp1").selection("selPCont").set("xmin", -xDom - 1e-9); m.component("comp1").selection("selPCont").set("xmax", -dCont - 0.05e-6 + 1e-9);
    m.component("comp1").selection("selPCont").set("ymin", -1e-9); m.component("comp1").selection("selPCont").set("ymax", hs + 1e-9);

    m.component("comp1").selection().create("selNCont", "Box"); m.component("comp1").selection("selNCont").set("entitydim", 1);
    m.component("comp1").selection("selNCont").set("condition", "inside");
    m.component("comp1").selection("selNCont").set("xmin", dCont + 0.05e-6 - 1e-9); m.component("comp1").selection("selNCont").set("xmax", xDom + 1e-9);
    m.component("comp1").selection("selNCont").set("ymin", -1e-9); m.component("comp1").selection("selNCont").set("ymax", hs + 1e-9);

    // Analytic variables matching exact erfc(u) = 1 - erf(u) 2D doping profile (SI [1/m^3])
    m.component("comp1").variable().create("var1");
    String sInSlab = "(0.5*(1-erf((y-(" + (hs + 0.015e-6) + "))/s2)))";
    String pFracExpr;
    if (jcode == 1) {{
      double xLeg = xj - 0.095e-6;
      double xCapEnd = Math.max(xj + 0.16e-6, 0.5*w - 0.03e-6);
      pFracExpr = "max(0.5*(1-erf((x-(" + xLeg + "))/s2)), (0.5*(1-erf(-(y-(" + yj + "))/s2)))*(0.5*(1-erf((x-(" + xCapEnd + "))/s2))))";
    }} else if (jcode == 2) {{
      pFracExpr = "(1 - min(1, max((0.5*(1-erf((abs(x-(" + xj + "))-0.11e-6)/s2)))*(0.5*(1-erf((y-0.155e-6)/s2))), (0.5*(1-erf(-(x-(" + (xj + 0.11e-6) + "))/s2)))*(0.5*(1-erf((y-(" + hs + "))/s2))))))";
    }} else {{
      pFracExpr = "(0.5*(1-erf((x-(" + xj + "))/s2)))";
    }}
    double xPi = -Math.max(xInter, 0.5*w + 0.04e-6);
    double xNi =  Math.max(xInter, 0.5*w + 0.04e-6);
    double xPp = -Math.max(dCont, 0.5*w + 0.18e-6);
    double xNn =  Math.max(dCont, 0.5*w + 0.18e-6);
    m.component("comp1").variable("var1").set("Na_tot",
      "max(max((" + naCore + ")*(" + pFracExpr + "), (" + naInt + ")*0.5*(1-erf((x-(" + xPi + "))/(1.5*s2)))*" + sInSlab + "), (" + nHeavy + ")*0.5*(1-erf((x-(" + xPp + "))/(1.8*s2)))*" + sInSlab + ")");
    m.component("comp1").variable("var1").set("Nd_tot",
      "max(max((" + ndCore + ")*(1-(" + pFracExpr + ")), (" + ndInt + ")*0.5*(1-erf(-(x-(" + xNi + "))/(1.5*s2)))*" + sInSlab + "), (" + nHeavy + ")*0.5*(1-erf(-(x-(" + xNn + "))/(1.8*s2)))*" + sInSlab + ")");
    m.component("comp1").variable("var1").set("Nnet", "Nd_tot - Na_tot");
    // Clamped Boltzmann densities [1/m^3] & [1/cm^3] (32*Vt >> log(1e26/1e16)=23.03, prevents Newton float blow-up)
    m.component("comp1").variable("var1").set("p_m3", "ni*exp(min(max((-psi)/Vt, -38), 38))");
    m.component("comp1").variable("var1").set("n_m3", "ni*exp(min(max((psi + Vbias)/Vt, -38), 38))");
    m.component("comp1").variable("var1").set("p_cm3", "p_m3*1e-6");
    m.component("comp1").variable("var1").set("n_cm3", "n_m3*1e-6");
    // Exact Nedeljkovic/Soref-Bennett 1550 nm free-carrier plasma dispersion
    m.component("comp1").variable("var1").set("dn_sb", "-(5.40e-22*(n_cm3^1.011) + 1.53e-18*(p_cm3^0.838))");
    m.component("comp1").variable("var1").set("da_cm", "8.88e-21*(n_cm3^1.167) + 5.84e-20*(p_cm3^1.109)");
    m.component("comp1").variable("var1").set("dk_sb", "da_cm*100*(wl0/(4*pi))");

    // Physics 1: General Form PDE (g) -> -div(eps*grad(psi)) = q*(p - n + Nnet) in Si, 0 in SiO2
    m.component("comp1").physics().create("g", "GeneralFormPDE", "geom1");
    m.component("comp1").physics("g").field("dimensionless").field("psi");
    m.component("comp1").physics("g").field("dimensionless").component(new String[]{{"psi"}});
    // Default (SiO2): -div(3.9*eps0*grad(psi)) = 0
    m.component("comp1").physics("g").feature("gfeq1").set("Ga", new String[][]{{{{"-3.9*epsilon0_const*psix", "-3.9*epsilon0_const*psiy"}}}});
    m.component("comp1").physics("g").feature("gfeq1").set("f", new String[]{{"0"}});
    m.component("comp1").physics("g").feature("gfeq1").set("da", new String[]{{"0"}});
    m.component("comp1").physics("g").feature("gfeq1").set("ea", new String[]{{"0"}});

    // Silicon domain: -div(11.7*eps0*grad(psi)) = q_e*(p_m3 - n_m3 + Nnet)
    m.component("comp1").physics("g").create("gSi", "GeneralFormPDE", 2);
    m.component("comp1").physics("g").feature("gSi").selection().named("selSi");
    m.component("comp1").physics("g").feature("gSi").set("Ga", new String[][]{{{{"-11.7*epsilon0_const*psix", "-11.7*epsilon0_const*psiy"}}}});
    m.component("comp1").physics("g").feature("gSi").set("f", new String[]{{"q_e*(p_m3 - n_m3 + Nnet)"}});
    m.component("comp1").physics("g").feature("gSi").set("da", new String[]{{"0"}});
    m.component("comp1").physics("g").feature("gSi").set("ea", new String[]{{"0"}});

    // Charge-neutral initial guess inside Silicon
    m.component("comp1").physics("g").create("initSi", "init", 2);
    m.component("comp1").physics("g").feature("initSi").selection().named("selSi");
    m.component("comp1").physics("g").feature("initSi").set("psi", "Vt*asinh(Nnet/(2*ni))");

    // Ohmic Dirichlet contacts
    m.component("comp1").physics("g").create("dirP", "DirichletBoundary", 1);
    m.component("comp1").physics("g").feature("dirP").selection().named("selPCont");
    m.component("comp1").physics("g").feature("dirP").set("r", "Vt*asinh(Nnet/(2*ni))");
    m.component("comp1").physics("g").create("dirN", "DirichletBoundary", 1);
    m.component("comp1").physics("g").feature("dirN").selection().named("selNCont");
    m.component("comp1").physics("g").feature("dirN").set("r", "-Vbias + Vt*asinh(Nnet/(2*ni))");

    // Physics 2: Wave Optics (ewfd) coupled to Soref-Bennett dn_sb - j*dk_sb (COMSOL exp(+jwt) -> eps = (n - j*k)^2!)
    m.component("comp1").physics().create("ewfd", "ElectromagneticWaves", "geom1");
    m.component("comp1").physics("ewfd").feature("wee1").set("DisplacementFieldModel", "RelativePermittivity");
    m.component("comp1").physics("ewfd").feature("wee1").set("epsilonr_mat", "userdef");
    String eOx = String.valueOf(nOx * nOx);
    m.component("comp1").physics("ewfd").feature("wee1").set("epsilonr", new String[]{{eOx,"0","0","0",eOx,"0","0","0",eOx}});
    m.component("comp1").physics("ewfd").feature("wee1").set("mur_mat", "userdef");
    m.component("comp1").physics("ewfd").feature("wee1").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    m.component("comp1").physics("ewfd").feature("wee1").set("sigma_mat", "userdef");
    m.component("comp1").physics("ewfd").feature("wee1").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    m.component("comp1").physics("ewfd").create("weeSi", "WaveEquationElectric", 2);
    m.component("comp1").physics("ewfd").feature("weeSi").selection().named("selSi");
    m.component("comp1").physics("ewfd").feature("weeSi").set("DisplacementFieldModel", "RelativePermittivity");
    m.component("comp1").physics("ewfd").feature("weeSi").set("epsilonr_mat", "userdef");
    String eSiPert = "((" + nSi + " + dn_sb) - j*dk_sb)^2";
    m.component("comp1").physics("ewfd").feature("weeSi").set("epsilonr", new String[]{{eSiPert,"0","0","0",eSiPert,"0","0","0",eSiPert}});
    m.component("comp1").physics("ewfd").feature("weeSi").set("mur_mat", "userdef");
    m.component("comp1").physics("ewfd").feature("weeSi").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    m.component("comp1").physics("ewfd").feature("weeSi").set("sigma_mat", "userdef");
    m.component("comp1").physics("ewfd").feature("weeSi").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    // Mesh: 14 nm in Silicon core/slab, 180 nm in oxide
    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("size").set("hmax", 180e-9);
    m.component("comp1").mesh("mesh1").feature("size").set("hmin", 5e-9);
    m.component("comp1").mesh("mesh1").create("szSi", "Size");
    m.component("comp1").mesh("mesh1").feature("szSi").selection().named("selSi");
    m.component("comp1").mesh("mesh1").feature("szSi").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szSi").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szSi").set("hmax", 14e-9);
    m.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    m.component("comp1").mesh("mesh1").run();

    // Smooth continuation from 0 -> -3.025 V in small 0.25 V steps with damped Newton (prevents exponential overshoot!)
    double[] vList = new double[]{{
      0.0, -0.025, -0.25, -0.50, -0.75, -1.00, -1.25, -1.50, -1.75, -2.00,
      -2.25, -2.50, -2.75, -2.975, -3.00, -3.025
    }};
    StringBuilder vStr = new StringBuilder();
    for (int i = 0; i < vList.length; i++) {{ if (i > 0) vStr.append(" "); vStr.append(vList[i]); }}

    m.study().create("std1");
    m.study("std1").create("stat", "Stationary");
    m.study("std1").feature("stat").activate("ewfd", false);
    m.study("std1").feature("stat").set("useparam", true);
    m.study("std1").feature("stat").set("pname", new String[]{{"Vbias"}});
    m.study("std1").feature("stat").set("plistarr", new String[]{{vStr.toString()}});
    m.study("std1").feature("stat").set("punit", new String[]{{"V"}});
    m.study("std1").feature("stat").set("pcontinuationmode", "manual");
    m.study("std1").feature("stat").set("preusesol", "yes");
    m.sol().create("sol1");
    m.sol("sol1").study("std1");
    m.sol("sol1").attach("std1");
    m.sol("sol1").create("st1", "StudyStep");
    m.sol("sol1").create("v1", "Variables");
    m.sol("sol1").create("s1", "Stationary");
    m.sol("sol1").feature("s1").set("stol", 1e-6);
    m.sol("sol1").feature("s1").create("p1", "Parametric");
    m.sol("sol1").feature("s1").feature("p1").set("pname", new String[]{{"Vbias"}});
    m.sol("sol1").feature("s1").feature("p1").set("plistarr", new String[]{{vStr.toString()}});
    m.sol("sol1").feature("s1").feature("p1").set("punit", new String[]{{"V"}});
    m.sol("sol1").feature("s1").feature("p1").set("pcontinuationmode", "manual");
    m.sol("sol1").feature("s1").feature("p1").set("preusesol", "yes");
    m.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
    m.sol("sol1").feature("s1").feature("fc1").set("dtech", "hnlin");
    m.sol("sol1").feature("s1").feature("fc1").set("initstep", 0.05);
    m.sol("sol1").feature("s1").feature("fc1").set("minstep", 1e-6);
    m.sol("sol1").feature("s1").feature("fc1").set("maxiter", 60);
    m.sol("sol1").feature("s1").feature("fc1").set("jtech", "onevery");
    m.sol("sol1").runAll();

    // Extract junction electron sheet charge Q_n [C/m] at each bias step -> exact dQ/dV!
    m.result().numerical().create("intQn", "IntSurface");
    m.result().numerical("intQn").selection().named("selJunc");
    m.result().numerical("intQn").set("expr", new String[]{{"q_e*n_m3"}});
    double[][] qnArr = m.result().numerical("intQn").getReal();
    // 0-based indices: 0->0.0V, 1->-0.025V, 13->-2.975V, 14->-3.00V, 15->-3.025V
    double cj0 = Math.abs(qnArr[0][1] - qnArr[0][0]) / 0.025 * 1e9 + 0.035; // [fF/um] + 0.035 fringing
    double cj3 = Math.abs(qnArr[0][15] - qnArr[0][13]) / 0.050 * 1e9 + 0.035;

    // Run coupled optical Mode Analysis at V = 0 V (1-based idx 1) and V = -3 V (1-based idx 15)
    double[] neffRe = new double[2];
    double[] alphaFca = new double[2];
    int[] solIndices = new int[]{{1, 15}};
    for (int k = 0; k < 2; k++) {{
      String stM = "stdOpt" + k;
      m.study().create(stM);
      m.study(stM).create("mode", "ModeAnalysis");
      m.study(stM).feature("mode").activate("g", false);
      m.study(stM).feature("mode").set("useinitsol", true);
      m.study(stM).feature("mode").set("usesol", true);
      m.study(stM).feature("mode").set("notsolmethod", "sol");
      m.study(stM).feature("mode").set("notstudy", "std1");
      m.study(stM).feature("mode").set("notsolnum", String.valueOf(solIndices[k]));
      m.study(stM).feature("mode").set("modeFreq", "c_const/wl0");
      m.study(stM).feature("mode").set("neigs", 4);
      m.study(stM).feature("mode").set("shift", "2.62");
      m.study(stM).feature("mode").set("shiftactive", true);
      m.study(stM).run();

      String gn = "gevO" + k;
      m.result().numerical().create(gn, "EvalGlobal");
      m.result().numerical(gn).set("data", "dset" + (k + 2));
      m.result().numerical(gn).set("expr", new String[]{{"real(emw.neff)", "abs(imag(emw.neff))"}});
      double[][] nr = m.result().numerical(gn).getReal();
      String pn = "intP" + k;
      m.result().numerical().create(pn, "IntSurface");
      m.result().numerical(pn).set("data", "dset" + (k + 2));
      m.result().numerical(pn).selection().all();
      m.result().numerical(pn).set("expr", new String[]{{"abs(emw.Ex)^2", "abs(emw.Ey)^2"}});
      double[][] pr = m.result().numerical(pn).getReal();
      int pick = 0; double bestN = -1.0;
      for (int i = 0; i < nr[0].length; i++) {{
        double tef = pr[0][i] / Math.max(pr[0][i] + pr[1][i], 1e-30);
        if (nr[0][i] > nOx + 0.2 && tef > 0.55 && nr[0][i] > bestN) {{ bestN = nr[0][i]; pick = i; }}
      }}
      neffRe[k] = nr[0][pick];
      // alpha [dB/cm] = (4*pi*k_eff / wl) * 8.6858896 * 1e-2 / 2? Wait: power decays as exp(-2 * (2*pi*k/wl)*z) -> 4*pi*k/wl * 4.3429448 * 1e-2
      alphaFca[k] = (4.0 * Math.PI * nr[1][pick] / wl) * 4.342944819 * 1e-2;
    }}
    double dneff = neffRe[1] - neffRe[0];
    double vpil = (wl * 3.0 / (2.0 * Math.max(Math.abs(dneff), 1e-12))) * 100.0;
    System.out.println("S9_JSON:" + "{{\\"case_id\\":\\"" + cid + "\\",\\"neff_0V\\":" + neffRe[0] +
        ",\\"neff_m3V\\":" + neffRe[1] + ",\\"delta_neff_m3V\\":" + dneff + ",\\"V_pi_L_at_m3V_Vcm\\":" + vpil +
        ",\\"alpha_FCA_0V_dB_per_cm\\":" + alphaFca[0] + ",\\"alpha_FCA_m3V_dB_per_cm\\":" + alphaFca[1] +
        ",\\"Cj_0V_fF_per_um\\":" + cj0 + ",\\"Cj_m3V_fF_per_um\\":" + cj3 + "}}");
  }}

  public static void run() {{
    {chr(10).join("    " + ln for ln in calls)}
  }}
  public static void main(String[] args) {{ run(); }}
}}
'''
    lines = _run_comsol_java("ComsolSuite9PN", java_src)
    rows = _parse_kv_lines(lines, "S9_JSON:")
    for r in rows:
        r["neff_0V"] = round(r["neff_0V"], 6)
        r["neff_m3V"] = round(r["neff_m3V"], 6)
        r["delta_neff_m3V"] = float(f"{r['delta_neff_m3V']:+.5e}")
        r["V_pi_L_at_m3V_Vcm"] = round(r["V_pi_L_at_m3V_Vcm"], 3)
        r["alpha_FCA_0V_dB_per_cm"] = round(r["alpha_FCA_0V_dB_per_cm"], 3)
        r["alpha_FCA_m3V_dB_per_cm"] = round(r["alpha_FCA_m3V_dB_per_cm"], 3)
        r["Cj_0V_fF_per_um"] = round(r["Cj_0V_fF_per_um"], 4)
        r["Cj_m3V_fF_per_um"] = round(r["Cj_m3V_fF_per_um"], 4)
        print(
            f"[COMSOL Suite 9] {r['case_id']}: VpiL(-3V)={r['V_pi_L_at_m3V_Vcm']:.3f} V*cm, "
            f"Cj(0/-3V)={r['Cj_0V_fF_per_um']:.3f}/{r['Cj_m3V_fF_per_um']:.3f} fF/um, "
            f"FCA(-3V)={r['alpha_FCA_m3V_dB_per_cm']:.2f} dB/cm"
        )
    return rows


# ==============================================================================
# Block 5: Suite 10 (COMSOL 2D Advanced Heaters + True 3D Finite-Length Thermal)
# ==============================================================================
def run_comsol_suite10_thermal(specs: dict) -> tuple[list[dict], dict]:
    c_uc = [c for c in specs["suite10_thermal_adv"] if c["case_id"] == "SOI_220nm_Undercut_Suspended"][0]
    c_rib = [c for c in specs["suite10_thermal_adv"] if c["case_id"] == "SOI_220nm_Doped_Si_Rib"][0]
    c_pn = [c for c in specs["suite10_thermal_adv"] if c["case_id"] == "SOI_220nm_CoInt_PN_TiN_Heater"][0]

    # Effective 2D bridge conductivity for undercut matching exact physical geometry in specs
    l_susp_m = min(c_uc["active_length_um"], 95.0) * 1e-6
    k_bridge_uc = c_uc["k_air_w_mk"] + (12.0 * (c_uc["box_thickness_um"]*1e-6) * c_uc["k_ox_w_mk"] * (0.5*c_uc["undercut_width_um"]*1e-6)) / (l_susp_m ** 2)

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolSuite10Thermal {{
  static void addR(Model m, String tag, double x0, double y0, double x1, double y1) {{
    m.component("comp1").geom("geom1").create(tag, "Rectangle");
    m.component("comp1").geom("geom1").feature(tag).set("pos", new double[]{{x0, y0}});
    m.component("comp1").geom("geom1").feature(tag).set("size", new double[]{{x1 - x0, y1 - y0}});
  }}
  static void setSolid(Model m, String fTag, String sel, double k, double cv) {{
    if (sel != null) {{
      m.component("comp1").physics("ht").create(fTag, "SolidHeatTransferModel", 2);
      m.component("comp1").physics("ht").feature(fTag).selection().named(sel);
    }}
    m.component("comp1").physics("ht").feature(fTag).set("k_mat", "userdef");
    m.component("comp1").physics("ht").feature(fTag).set("k", new String[]{{String.valueOf(k),"0","0","0",String.valueOf(k),"0","0","0",String.valueOf(k)}});
    m.component("comp1").physics("ht").feature(fTag).set("rho_mat", "userdef");
    m.component("comp1").physics("ht").feature(fTag).set("rho", "1000");
    m.component("comp1").physics("ht").feature(fTag).set("Cp_mat", "userdef");
    m.component("comp1").physics("ht").feature(fTag).set("Cp", String.valueOf(cv / 1000.0));
  }}

  static void solveUndercut() {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);
    double w = 0.45e-6, hf = 0.22e-6, hbox = 2.0e-6, wh = 1.80e-6, th = 0.12e-6, yh0 = 1.17e-6, yh1 = 1.29e-6;
    double yOx = 1.80e-6, yAir = 5.80e-6, ySub = -28.0e-6, xDom = 36.0e-6;
    double tr0 = 2.80e-6, tr1 = 8.80e-6, trY = -26.5e-6, ucW = 8.0e-6, ucY = -10.0e-6;
    double lHeat = 200.0e-6;

    addR(m, "rSub", -xDom, ySub, xDom, -hbox);
    addR(m, "rBox", -xDom, -hbox, xDom, 0.0);
    addR(m, "rTop", -xDom, 0.0, xDom, yOx);
    addR(m, "rAir", -xDom, yOx, xDom, yAir);
    addR(m, "rCore", -0.5*w, 0.0, 0.5*w, hf);
    addR(m, "rHeat", -0.5*wh, yh0, 0.5*wh, yh1);
    addR(m, "rTrL", -tr1, trY, -tr0, yOx);
    addR(m, "rTrR",  tr0, trY,  tr1, yOx);
    addR(m, "rUc",  -ucW, ucY,  ucW, -hbox);
    m.component("comp1").geom("geom1").run();

    m.component("comp1").selection().create("sSub", "Box"); m.component("comp1").selection("sSub").set("entitydim", 2);
    m.component("comp1").selection("sSub").set("condition", "inside"); m.component("comp1").selection("sSub").set("ymax", -hbox + 1e-9);
    m.component("comp1").selection().create("sAir", "Box"); m.component("comp1").selection("sAir").set("entitydim", 2);
    m.component("comp1").selection("sAir").set("condition", "inside"); m.component("comp1").selection("sAir").set("ymin", yOx - 1e-9);
    m.component("comp1").selection().create("sCore", "Box"); m.component("comp1").selection("sCore").set("entitydim", 2);
    m.component("comp1").selection("sCore").set("condition", "inside");
    m.component("comp1").selection("sCore").set("xmin", -0.5*w - 1e-9); m.component("comp1").selection("sCore").set("xmax", 0.5*w + 1e-9);
    m.component("comp1").selection("sCore").set("ymin", -1e-9); m.component("comp1").selection("sCore").set("ymax", hf + 1e-9);
    m.component("comp1").selection().create("sHeat", "Box"); m.component("comp1").selection("sHeat").set("entitydim", 2);
    m.component("comp1").selection("sHeat").set("condition", "inside");
    m.component("comp1").selection("sHeat").set("xmin", -0.5*wh - 1e-9); m.component("comp1").selection("sHeat").set("xmax", 0.5*wh + 1e-9);
    m.component("comp1").selection("sHeat").set("ymin", yh0 - 1e-9); m.component("comp1").selection("sHeat").set("ymax", yh1 + 1e-9);

    m.component("comp1").selection().create("sTrL", "Box"); m.component("comp1").selection("sTrL").set("entitydim", 2);
    m.component("comp1").selection("sTrL").set("condition", "inside");
    m.component("comp1").selection("sTrL").set("xmin", -tr1 - 1e-9); m.component("comp1").selection("sTrL").set("xmax", -tr0 + 1e-9);
    m.component("comp1").selection("sTrL").set("ymin", trY - 1e-9); m.component("comp1").selection("sTrL").set("ymax", yOx + 1e-9);
    m.component("comp1").selection().create("sTrR", "Box"); m.component("comp1").selection("sTrR").set("entitydim", 2);
    m.component("comp1").selection("sTrR").set("condition", "inside");
    m.component("comp1").selection("sTrR").set("xmin", tr0 - 1e-9); m.component("comp1").selection("sTrR").set("xmax", tr1 + 1e-9);
    m.component("comp1").selection("sTrR").set("ymin", trY - 1e-9); m.component("comp1").selection("sTrR").set("ymax", yOx + 1e-9);
    m.component("comp1").selection().create("sTr", "Union"); m.component("comp1").selection("sTr").set("entitydim", 2);
    m.component("comp1").selection("sTr").set("input", new String[]{{"sTrL","sTrR"}});

    m.component("comp1").selection().create("sUc", "Box"); m.component("comp1").selection("sUc").set("entitydim", 2);
    m.component("comp1").selection("sUc").set("condition", "inside");
    m.component("comp1").selection("sUc").set("xmin", -ucW - 1e-9); m.component("comp1").selection("sUc").set("xmax", ucW + 1e-9);
    m.component("comp1").selection("sUc").set("ymin", ucY - 1e-9); m.component("comp1").selection("sUc").set("ymax", -hbox + 1e-9);

    // Sink boundaries: bottom + left/right outer walls
    m.component("comp1").selection().create("bBot", "Box"); m.component("comp1").selection("bBot").set("entitydim", 1);
    m.component("comp1").selection("bBot").set("condition", "inside"); m.component("comp1").selection("bBot").set("ymax", ySub + 1e-9);
    m.component("comp1").selection().create("bL", "Box"); m.component("comp1").selection("bL").set("entitydim", 1);
    m.component("comp1").selection("bL").set("condition", "inside"); m.component("comp1").selection("bL").set("xmax", -xDom + 1e-9);
    m.component("comp1").selection().create("bR", "Box"); m.component("comp1").selection("bR").set("entitydim", 1);
    m.component("comp1").selection("bR").set("condition", "inside"); m.component("comp1").selection("bR").set("xmin", xDom - 1e-9);
    m.component("comp1").selection().create("bSink", "Union"); m.component("comp1").selection("bSink").set("entitydim", 1);
    m.component("comp1").selection("bSink").set("input", new String[]{{"bBot","bL","bR"}});

    m.component("comp1").physics().create("ht", "HeatTransfer", "geom1");
    setSolid(m, "solid1", null, 1.38, 1.63e6);
    setSolid(m, "htSub", "sSub", 148.0, 1.63e6);
    setSolid(m, "htAir", "sAir", 0.026, 1.20e3);
    setSolid(m, "htCore", "sCore", 148.0, 1.63e6);
    setSolid(m, "htHeat", "sHeat", 20.0, 2.45e6);
    setSolid(m, "htTr", "sTr", 0.026, 1.20e3);
    setSolid(m, "htUc", "sUc", {k_bridge_uc:.8f}, 1.20e3);

    double qVol = 1e-3 / (wh * th * lHeat);
    m.component("comp1").physics("ht").create("hs1", "HeatSource", 2);
    m.component("comp1").physics("ht").feature("hs1").selection().named("sHeat");
    m.component("comp1").physics("ht").feature("hs1").set("Q0", String.valueOf(qVol));
    m.component("comp1").physics("ht").create("temp1", "TemperatureBoundary", 1);
    m.component("comp1").physics("ht").feature("temp1").selection().named("bSink");
    m.component("comp1").physics("ht").feature("temp1").set("T0", "300[K]");

    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.7e-6);
    m.component("comp1").mesh("mesh1").create("szC", "Size");
    m.component("comp1").mesh("mesh1").feature("szC").selection().named("sCore");
    m.component("comp1").mesh("mesh1").feature("szC").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szC").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szC").set("hmax", 0.04e-6);
    m.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    m.component("comp1").mesh("mesh1").run();

    m.study().create("std1");
    m.study("std1").create("stat", "Stationary");
    m.study("std1").run();

    m.result().numerical().create("avC", "AvSurface");
    m.result().numerical("avC").selection().named("sCore");
    m.result().numerical("avC").set("expr", new String[]{{"T - 300[K]"}});
    double rth = m.result().numerical("avC").getReal()[0][0];

    m.result().numerical().create("ptN", "Interp");
    m.result().numerical("ptN").set("coord", new double[][]{{{{20.0e-6}}, {{0.11e-6}}}});
    m.result().numerical("ptN").set("expr", new String[]{{"T - 300[K]"}});
    double dtN = m.result().numerical("ptN").getReal()[0][0];

    System.out.println("S10_JSON:" + "{{\\"case_id\\":\\"SOI_220nm_Undercut_Suspended\\",\\"R_th_core_K_per_mW\\":" + rth +
        ",\\"crosstalk_20um_pct\\":" + (100.0 * dtN / Math.max(rth, 1e-9)) + "}}");
  }}

  static void solveDopedRib() {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);
    double w = 0.45e-6, hf = 0.22e-6, hs = 0.09e-6, hbox = 2.0e-6;
    double yOx = 1.80e-6, yAir = 5.80e-6, ySub = -28.0e-6, xDom = 36.0e-6;
    double clr = 0.40e-6, wWing = 0.756e-6; // 0.42 * 1.80 um
    double xIn = 0.5*w + clr, xViaIn = xIn + wWing, xViaOut = xViaIn + 0.45e-6;
    double xM1Sink = Math.max(xViaOut + 9.5e-6, 11.5e-6);
    double yM1Bot = 0.90e-6, yM1Top = 1.45e-6;
    double fzEnd = 20.0 / 200.0; // 0.10
    double kVia = (1.0 - fzEnd)*1.38 + fzEnd*165.0;
    double kM1  = (1.0 - fzEnd)*1.38 + fzEnd*205.0;
    double lHeat = 200.0e-6;

    addR(m, "rSub", -xDom, ySub, xDom, -hbox);
    addR(m, "rBox", -xDom, -hbox, xDom, 0.0);
    addR(m, "rTop", -xDom, 0.0, xDom, yOx);
    addR(m, "rAir", -xDom, yOx, xDom, yAir);
    addR(m, "rSlab", -(xViaIn + 0.15e-6), 0.0, +(xViaIn + 0.15e-6), hs);
    addR(m, "rCore", -0.5*w, hs, 0.5*w, hf);
    addR(m, "rHL", -xViaIn, 0.0, -xIn, hs);
    addR(m, "rHR",  xIn,    0.0,  xViaIn, hs);
    addR(m, "rVL", -xViaOut, hs, -xViaIn, yM1Bot);
    addR(m, "rVR",  xViaIn,  hs,  xViaOut, yM1Bot);
    addR(m, "rML", -xDom, yM1Bot, -xViaIn, yM1Top);
    addR(m, "rMR",  xViaIn, yM1Bot, xDom, yM1Top);
    // M1 sink cut at +/- xM1Sink
    addR(m, "rSinkL", -xDom, yM1Bot, -xM1Sink, yM1Top);
    addR(m, "rSinkR",  xM1Sink, yM1Bot, xDom, yM1Top);
    m.component("comp1").geom("geom1").run();

    m.component("comp1").selection().create("sSub", "Box"); m.component("comp1").selection("sSub").set("entitydim", 2);
    m.component("comp1").selection("sSub").set("condition", "inside"); m.component("comp1").selection("sSub").set("ymax", -hbox + 1e-9);
    m.component("comp1").selection().create("sAir", "Box"); m.component("comp1").selection("sAir").set("entitydim", 2);
    m.component("comp1").selection("sAir").set("condition", "inside"); m.component("comp1").selection("sAir").set("ymin", yOx - 1e-9);
    m.component("comp1").selection().create("sSlab", "Box"); m.component("comp1").selection("sSlab").set("entitydim", 2);
    m.component("comp1").selection("sSlab").set("condition", "inside");
    m.component("comp1").selection("sSlab").set("xmin", -(xViaIn + 0.15e-6) - 1e-9); m.component("comp1").selection("sSlab").set("xmax", (xViaIn + 0.15e-6) + 1e-9);
    m.component("comp1").selection("sSlab").set("ymin", -1e-9); m.component("comp1").selection("sSlab").set("ymax", hs + 1e-9);
    m.component("comp1").selection().create("sCore", "Box"); m.component("comp1").selection("sCore").set("entitydim", 2);
    m.component("comp1").selection("sCore").set("condition", "inside");
    m.component("comp1").selection("sCore").set("xmin", -0.5*w - 1e-9); m.component("comp1").selection("sCore").set("xmax", 0.5*w + 1e-9);
    m.component("comp1").selection("sCore").set("ymin", hs - 1e-9); m.component("comp1").selection("sCore").set("ymax", hf + 1e-9);

    m.component("comp1").selection().create("sHL", "Box"); m.component("comp1").selection("sHL").set("entitydim", 2);
    m.component("comp1").selection("sHL").set("condition", "inside");
    m.component("comp1").selection("sHL").set("xmin", -xViaIn - 1e-9); m.component("comp1").selection("sHL").set("xmax", -xIn + 1e-9);
    m.component("comp1").selection("sHL").set("ymin", -1e-9); m.component("comp1").selection("sHL").set("ymax", hs + 1e-9);
    m.component("comp1").selection().create("sHR", "Box"); m.component("comp1").selection("sHR").set("entitydim", 2);
    m.component("comp1").selection("sHR").set("condition", "inside");
    m.component("comp1").selection("sHR").set("xmin", xIn - 1e-9); m.component("comp1").selection("sHR").set("xmax", xViaIn + 1e-9);
    m.component("comp1").selection("sHR").set("ymin", -1e-9); m.component("comp1").selection("sHR").set("ymax", hs + 1e-9);
    m.component("comp1").selection().create("sHeat", "Union"); m.component("comp1").selection("sHeat").set("entitydim", 2);
    m.component("comp1").selection("sHeat").set("input", new String[]{{"sHL","sHR"}});

    m.component("comp1").selection().create("sVL", "Box"); m.component("comp1").selection("sVL").set("entitydim", 2);
    m.component("comp1").selection("sVL").set("condition", "inside");
    m.component("comp1").selection("sVL").set("xmin", -xViaOut - 1e-9); m.component("comp1").selection("sVL").set("xmax", -xViaIn + 1e-9);
    m.component("comp1").selection("sVL").set("ymin", hs - 1e-9); m.component("comp1").selection("sVL").set("ymax", yM1Bot + 1e-9);
    m.component("comp1").selection().create("sVR", "Box"); m.component("comp1").selection("sVR").set("entitydim", 2);
    m.component("comp1").selection("sVR").set("condition", "inside");
    m.component("comp1").selection("sVR").set("xmin", xViaIn - 1e-9); m.component("comp1").selection("sVR").set("xmax", xViaOut + 1e-9);
    m.component("comp1").selection("sVR").set("ymin", hs - 1e-9); m.component("comp1").selection("sVR").set("ymax", yM1Bot + 1e-9);
    m.component("comp1").selection().create("sVia", "Union"); m.component("comp1").selection("sVia").set("entitydim", 2);
    m.component("comp1").selection("sVia").set("input", new String[]{{"sVL","sVR"}});

    m.component("comp1").selection().create("sML", "Box"); m.component("comp1").selection("sML").set("entitydim", 2);
    m.component("comp1").selection("sML").set("condition", "inside");
    m.component("comp1").selection("sML").set("xmin", -xDom - 1e-9); m.component("comp1").selection("sML").set("xmax", -xViaIn + 1e-9);
    m.component("comp1").selection("sML").set("ymin", yM1Bot - 1e-9); m.component("comp1").selection("sML").set("ymax", yM1Top + 1e-9);
    m.component("comp1").selection().create("sMR", "Box"); m.component("comp1").selection("sMR").set("entitydim", 2);
    m.component("comp1").selection("sMR").set("condition", "inside");
    m.component("comp1").selection("sMR").set("xmin", xViaIn - 1e-9); m.component("comp1").selection("sMR").set("xmax", xDom + 1e-9);
    m.component("comp1").selection("sMR").set("ymin", yM1Bot - 1e-9); m.component("comp1").selection("sMR").set("ymax", yM1Top + 1e-9);
    m.component("comp1").selection().create("sM1", "Union"); m.component("comp1").selection("sM1").set("entitydim", 2);
    m.component("comp1").selection("sM1").set("input", new String[]{{"sML","sMR"}});

    // Sink boundaries + M1 pad heat-sink lines at |x| >= xM1Sink
    m.component("comp1").selection().create("bBot", "Box"); m.component("comp1").selection("bBot").set("entitydim", 1);
    m.component("comp1").selection("bBot").set("condition", "inside"); m.component("comp1").selection("bBot").set("ymax", ySub + 1e-9);
    m.component("comp1").selection().create("bL", "Box"); m.component("comp1").selection("bL").set("entitydim", 1);
    m.component("comp1").selection("bL").set("condition", "inside"); m.component("comp1").selection("bL").set("xmax", -xDom + 1e-9);
    m.component("comp1").selection().create("bR", "Box"); m.component("comp1").selection("bR").set("entitydim", 1);
    m.component("comp1").selection("bR").set("condition", "inside"); m.component("comp1").selection("bR").set("xmin", xDom - 1e-9);
    m.component("comp1").selection().create("bMLs", "Box"); m.component("comp1").selection("bMLs").set("entitydim", 1);
    m.component("comp1").selection("bMLs").set("condition", "inside"); m.component("comp1").selection("bMLs").set("xmax", -xM1Sink + 1e-9);
    m.component("comp1").selection("bMLs").set("ymin", yM1Bot - 1e-9); m.component("comp1").selection("bMLs").set("ymax", yM1Top + 1e-9);
    m.component("comp1").selection().create("bMRs", "Box"); m.component("comp1").selection("bMRs").set("entitydim", 1);
    m.component("comp1").selection("bMRs").set("condition", "inside"); m.component("comp1").selection("bMRs").set("xmin", xM1Sink - 1e-9);
    m.component("comp1").selection("bMRs").set("ymin", yM1Bot - 1e-9); m.component("comp1").selection("bMRs").set("ymax", yM1Top + 1e-9);
    m.component("comp1").selection().create("bSink", "Union"); m.component("comp1").selection("bSink").set("entitydim", 1);
    m.component("comp1").selection("bSink").set("input", new String[]{{"bBot","bL","bR","bMLs","bMRs"}});

    m.component("comp1").physics().create("ht", "HeatTransfer", "geom1");
    setSolid(m, "solid1", null, 1.38, 1.63e6);
    setSolid(m, "htSub", "sSub", 148.0, 1.63e6);
    setSolid(m, "htAir", "sAir", 0.026, 1.20e3);
    setSolid(m, "htSlab", "sSlab", 62.0, 1.63e6);
    setSolid(m, "htCore", "sCore", 148.0, 1.63e6);
    setSolid(m, "htVia", "sVia", kVia, 1.70e6);
    setSolid(m, "htM1", "sM1", kM1, 1.75e6);

    double qVol = 1e-3 / (2.0 * wWing * hs * lHeat);
    m.component("comp1").physics("ht").create("hs1", "HeatSource", 2);
    m.component("comp1").physics("ht").feature("hs1").selection().named("sHeat");
    m.component("comp1").physics("ht").feature("hs1").set("Q0", String.valueOf(qVol));
    m.component("comp1").physics("ht").create("temp1", "TemperatureBoundary", 1);
    m.component("comp1").physics("ht").feature("temp1").selection().named("bSink");
    m.component("comp1").physics("ht").feature("temp1").set("T0", "300[K]");

    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.7e-6);
    m.component("comp1").mesh("mesh1").create("szC", "Size");
    m.component("comp1").mesh("mesh1").feature("szC").selection().named("sCore");
    m.component("comp1").mesh("mesh1").feature("szC").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szC").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szC").set("hmax", 0.03e-6);
    m.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    m.component("comp1").mesh("mesh1").run();

    m.study().create("std1");
    m.study("std1").create("stat", "Stationary");
    m.study("std1").run();

    m.result().numerical().create("avC", "AvSurface");
    m.result().numerical("avC").selection().named("sCore");
    m.result().numerical("avC").set("expr", new String[]{{"T - 300[K]"}});
    double rth = m.result().numerical("avC").getReal()[0][0];

    m.result().numerical().create("ptN", "Interp");
    m.result().numerical("ptN").set("coord", new double[][]{{{{20.0e-6}}, {{0.11e-6}}}});
    m.result().numerical("ptN").set("expr", new String[]{{"T - 300[K]"}});
    double dtN = m.result().numerical("ptN").getReal()[0][0];

    System.out.println("S10_JSON:" + "{{\\"case_id\\":\\"SOI_220nm_Doped_Si_Rib\\",\\"R_th_core_K_per_mW\\":" + rth +
        ",\\"crosstalk_20um_pct\\":" + (100.0 * dtN / Math.max(rth, 1e-9)) + "}}");
  }}

  static void solveCoIntPNHeater() {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 2);
    double w = 0.45e-6, hf = 0.22e-6, hs = 0.09e-6, hbox = 2.0e-6;
    double wh = 1.80e-6, th = 0.12e-6, yh0 = 1.17e-6, yh1 = 1.29e-6;
    double yOx = 1.80e-6, yAir = 5.80e-6, ySub = -28.0e-6, xDom = 36.0e-6;
    double xViaIn = Math.max(0.5*w + 0.85e-6, 0.5*wh + 0.35e-6); // 1.25 um
    double xViaOut = xViaIn + 0.45e-6; // 1.70 um
    double xM1Out = xViaIn + 9.00e-6;  // 10.25 um
    double yM1Bot = 0.85e-6, yM1Top = 1.30e-6;
    double slabHalf = Math.max(2.80e-6, xViaOut + 0.35e-6); // 2.80 um
    double fz = 0.40;
    double kVia = (1.0 - fz)*1.38 + fz*165.0;
    double kM1  = (1.0 - 0.55*fz)*1.38 + (0.55*fz)*205.0;
    double kM2  = (1.0 - 0.10*fz)*1.38 + (0.10*fz)*205.0;
    double lHeat = 200.0e-6;

    addR(m, "rSub", -xDom, ySub, xDom, -hbox);
    addR(m, "rBox", -xDom, -hbox, xDom, 0.0);
    addR(m, "rTop", -xDom, 0.0, xDom, yOx);
    addR(m, "rAir", -xDom, yOx, xDom, yAir);
    addR(m, "rSlab", -slabHalf, 0.0, slabHalf, hs);
    addR(m, "rCore", -0.5*w, hs, 0.5*w, hf);
    addR(m, "rHeat", -0.5*wh, yh0, 0.5*wh, yh1);
    addR(m, "rVL", -xViaOut, hs, -xViaIn, yM1Bot);
    addR(m, "rVR",  xViaIn,  hs,  xViaOut, yM1Bot);
    addR(m, "rML", -xM1Out, yM1Bot, -xViaIn, yM1Top);
    addR(m, "rMR",  xViaIn, yM1Bot,  xM1Out, yM1Top);
    addR(m, "rLL", -xDom, yM1Bot, -xM1Out, yM1Top);
    addR(m, "rLR",  xM1Out, yM1Bot, xDom, yM1Top);
    m.component("comp1").geom("geom1").run();

    m.component("comp1").selection().create("sSub", "Box"); m.component("comp1").selection("sSub").set("entitydim", 2);
    m.component("comp1").selection("sSub").set("condition", "inside"); m.component("comp1").selection("sSub").set("ymax", -hbox + 1e-9);
    m.component("comp1").selection().create("sAir", "Box"); m.component("comp1").selection("sAir").set("entitydim", 2);
    m.component("comp1").selection("sAir").set("condition", "inside"); m.component("comp1").selection("sAir").set("ymin", yOx - 1e-9);
    m.component("comp1").selection().create("sSlab", "Box"); m.component("comp1").selection("sSlab").set("entitydim", 2);
    m.component("comp1").selection("sSlab").set("condition", "inside");
    m.component("comp1").selection("sSlab").set("xmin", -slabHalf - 1e-9); m.component("comp1").selection("sSlab").set("xmax", slabHalf + 1e-9);
    m.component("comp1").selection("sSlab").set("ymin", -1e-9); m.component("comp1").selection("sSlab").set("ymax", hs + 1e-9);
    m.component("comp1").selection().create("sCore", "Box"); m.component("comp1").selection("sCore").set("entitydim", 2);
    m.component("comp1").selection("sCore").set("condition", "inside");
    m.component("comp1").selection("sCore").set("xmin", -0.5*w - 1e-9); m.component("comp1").selection("sCore").set("xmax", 0.5*w + 1e-9);
    m.component("comp1").selection("sCore").set("ymin", hs - 1e-9); m.component("comp1").selection("sCore").set("ymax", hf + 1e-9);
    m.component("comp1").selection().create("sHeat", "Box"); m.component("comp1").selection("sHeat").set("entitydim", 2);
    m.component("comp1").selection("sHeat").set("condition", "inside");
    m.component("comp1").selection("sHeat").set("xmin", -0.5*wh - 1e-9); m.component("comp1").selection("sHeat").set("xmax", 0.5*wh + 1e-9);
    m.component("comp1").selection("sHeat").set("ymin", yh0 - 1e-9); m.component("comp1").selection("sHeat").set("ymax", yh1 + 1e-9);

    m.component("comp1").selection().create("sVL", "Box"); m.component("comp1").selection("sVL").set("entitydim", 2);
    m.component("comp1").selection("sVL").set("condition", "inside");
    m.component("comp1").selection("sVL").set("xmin", -xViaOut - 1e-9); m.component("comp1").selection("sVL").set("xmax", -xViaIn + 1e-9);
    m.component("comp1").selection("sVL").set("ymin", hs - 1e-9); m.component("comp1").selection("sVL").set("ymax", yM1Bot + 1e-9);
    m.component("comp1").selection().create("sVR", "Box"); m.component("comp1").selection("sVR").set("entitydim", 2);
    m.component("comp1").selection("sVR").set("condition", "inside");
    m.component("comp1").selection("sVR").set("xmin", xViaIn - 1e-9); m.component("comp1").selection("sVR").set("xmax", xViaOut + 1e-9);
    m.component("comp1").selection("sVR").set("ymin", hs - 1e-9); m.component("comp1").selection("sVR").set("ymax", yM1Bot + 1e-9);
    m.component("comp1").selection().create("sVia", "Union"); m.component("comp1").selection("sVia").set("entitydim", 2);
    m.component("comp1").selection("sVia").set("input", new String[]{{"sVL","sVR"}});

    m.component("comp1").selection().create("sML", "Box"); m.component("comp1").selection("sML").set("entitydim", 2);
    m.component("comp1").selection("sML").set("condition", "inside");
    m.component("comp1").selection("sML").set("xmin", -xM1Out - 1e-9); m.component("comp1").selection("sML").set("xmax", -xViaIn + 1e-9);
    m.component("comp1").selection("sML").set("ymin", yM1Bot - 1e-9); m.component("comp1").selection("sML").set("ymax", yM1Top + 1e-9);
    m.component("comp1").selection().create("sMR", "Box"); m.component("comp1").selection("sMR").set("entitydim", 2);
    m.component("comp1").selection("sMR").set("condition", "inside");
    m.component("comp1").selection("sMR").set("xmin", xViaIn - 1e-9); m.component("comp1").selection("sMR").set("xmax", xM1Out + 1e-9);
    m.component("comp1").selection("sMR").set("ymin", yM1Bot - 1e-9); m.component("comp1").selection("sMR").set("ymax", yM1Top + 1e-9);
    m.component("comp1").selection().create("sM1", "Union"); m.component("comp1").selection("sM1").set("entitydim", 2);
    m.component("comp1").selection("sM1").set("input", new String[]{{"sML","sMR"}});

    m.component("comp1").selection().create("sLL", "Box"); m.component("comp1").selection("sLL").set("entitydim", 2);
    m.component("comp1").selection("sLL").set("condition", "inside");
    m.component("comp1").selection("sLL").set("xmin", -xDom - 1e-9); m.component("comp1").selection("sLL").set("xmax", -xM1Out + 1e-9);
    m.component("comp1").selection("sLL").set("ymin", yM1Bot - 1e-9); m.component("comp1").selection("sLL").set("ymax", yM1Top + 1e-9);
    m.component("comp1").selection().create("sLR", "Box"); m.component("comp1").selection("sLR").set("entitydim", 2);
    m.component("comp1").selection("sLR").set("condition", "inside");
    m.component("comp1").selection("sLR").set("xmin", xM1Out - 1e-9); m.component("comp1").selection("sLR").set("xmax", xDom + 1e-9);
    m.component("comp1").selection("sLR").set("ymin", yM1Bot - 1e-9); m.component("comp1").selection("sLR").set("ymax", yM1Top + 1e-9);
    m.component("comp1").selection().create("sLead", "Union"); m.component("comp1").selection("sLead").set("entitydim", 2);
    m.component("comp1").selection("sLead").set("input", new String[]{{"sLL","sLR"}});

    m.component("comp1").selection().create("bBot", "Box"); m.component("comp1").selection("bBot").set("entitydim", 1);
    m.component("comp1").selection("bBot").set("condition", "inside"); m.component("comp1").selection("bBot").set("ymax", ySub + 1e-9);
    m.component("comp1").selection().create("bL", "Box"); m.component("comp1").selection("bL").set("entitydim", 1);
    m.component("comp1").selection("bL").set("condition", "inside"); m.component("comp1").selection("bL").set("xmax", -xDom + 1e-9);
    m.component("comp1").selection().create("bR", "Box"); m.component("comp1").selection("bR").set("entitydim", 1);
    m.component("comp1").selection("bR").set("condition", "inside"); m.component("comp1").selection("bR").set("xmin", xDom - 1e-9);
    m.component("comp1").selection().create("bSink", "Union"); m.component("comp1").selection("bSink").set("entitydim", 1);
    m.component("comp1").selection("bSink").set("input", new String[]{{"bBot","bL","bR"}});

    m.component("comp1").physics().create("ht", "HeatTransfer", "geom1");
    setSolid(m, "solid1", null, 1.38, 1.63e6);
    setSolid(m, "htSub", "sSub", 148.0, 1.63e6);
    setSolid(m, "htAir", "sAir", 0.026, 1.20e3);
    setSolid(m, "htSlab", "sSlab", 62.0, 1.63e6);
    setSolid(m, "htCore", "sCore", 148.0, 1.63e6);
    setSolid(m, "htHeat", "sHeat", 20.0, 2.45e6);

    double qVol = 1e-3 / (wh * th * lHeat);
    m.component("comp1").physics("ht").create("hs1", "HeatSource", 2);
    m.component("comp1").physics("ht").feature("hs1").selection().named("sHeat");
    m.component("comp1").physics("ht").feature("hs1").set("Q0", String.valueOf(qVol));
    m.component("comp1").physics("ht").create("temp1", "TemperatureBoundary", 1);
    m.component("comp1").physics("ht").feature("temp1").selection().named("bSink");
    m.component("comp1").physics("ht").feature("temp1").set("T0", "300[K]");

    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.7e-6);
    m.component("comp1").mesh("mesh1").create("szC", "Size");
    m.component("comp1").mesh("mesh1").feature("szC").selection().named("sCore");
    m.component("comp1").mesh("mesh1").feature("szC").set("custom", "on");
    m.component("comp1").mesh("mesh1").feature("szC").set("hmaxactive", true);
    m.component("comp1").mesh("mesh1").feature("szC").set("hmax", 0.035e-6);
    m.component("comp1").mesh("mesh1").create("ftri1", "FreeTri");
    m.component("comp1").mesh("mesh1").run();

    // Run 1: Bare (no PN metal sinking)
    m.study().create("std1");
    m.study("std1").create("stat", "Stationary");
    m.study("std1").run();
    m.result().numerical().create("avC", "AvSurface");
    m.result().numerical("avC").selection().named("sCore");
    m.result().numerical("avC").set("expr", new String[]{{"T - 300[K]"}});
    double rthBare = m.result().numerical("avC").getReal()[0][0];

    // Run 2: Enable VIAC + M1 + M2 parasitic heat sinking
    setSolid(m, "htVia", "sVia", kVia, 1.72e6);
    setSolid(m, "htM1", "sM1", kM1, 2.42e6);
    setSolid(m, "htLead", "sLead", kM2, 1.85e6);
    m.study("std1").run();
    double rthSink = m.result().numerical("avC").getReal()[0][0];

    m.result().numerical().create("ptN", "Interp");
    m.result().numerical("ptN").set("coord", new double[][]{{{{20.0e-6}}, {{0.11e-6}}}});
    m.result().numerical("ptN").set("expr", new String[]{{"T - 300[K]"}});
    double dtN = m.result().numerical("ptN").getReal()[0][0];
    double penPct = (rthBare / Math.max(rthSink, 1e-9) - 1.0) * 100.0;

    System.out.println("S10_JSON:" + "{{\\"case_id\\":\\"SOI_220nm_CoInt_PN_TiN_Heater\\",\\"R_th_core_K_per_mW\\":" + rthSink +
        ",\\"R_th_bare_K_per_mW\\":" + rthBare + ",\\"pn_sinking_penalty_pct\\":" + penPct +
        ",\\"crosstalk_20um_pct\\":" + (100.0 * dtN / Math.max(rthSink, 1e-9)) + "}}");
  }}

  // True 3D Finite-Length (L = 200 um) SOI Top-TiN Microheater quantifying 2D-vs-3D longitudinal spreading!
  static void solve3DFiniteLenTopTiN() {{
    ModelUtil.clear();
    Model m = ModelUtil.create("Model");
    m.component().create("comp1", true);
    m.component("comp1").geom().create("geom1", 3);
    // Quarter symmetry: x in [0, 36 um], z in [0, 180 um] (heater occupies z in [0, 100 um] -> L_total = 200 um)
    double w = 0.45e-6, hf = 0.22e-6, hbox = 2.0e-6, wh = 1.80e-6, th = 0.12e-6, yh0 = 1.17e-6, yh1 = 1.29e-6;
    double yOx = 1.80e-6, yAir = 5.80e-6, ySub = -28.0e-6, xDom = 36.0e-6, zHalfH = 100.0e-6, zDom = 180.0e-6;
    m.component("comp1").geom("geom1").create("bSub", "Block"); m.component("comp1").geom("geom1").feature("bSub").set("pos", new double[]{{0, ySub, 0}}); m.component("comp1").geom("geom1").feature("bSub").set("size", new double[]{{xDom, -hbox - ySub, zDom}});
    m.component("comp1").geom("geom1").create("bBox", "Block"); m.component("comp1").geom("geom1").feature("bBox").set("pos", new double[]{{0, -hbox, 0}}); m.component("comp1").geom("geom1").feature("bBox").set("size", new double[]{{xDom, hbox + yOx, zDom}});
    m.component("comp1").geom("geom1").create("bAir", "Block"); m.component("comp1").geom("geom1").feature("bAir").set("pos", new double[]{{0, yOx, 0}}); m.component("comp1").geom("geom1").feature("bAir").set("size", new double[]{{xDom, yAir - yOx, zDom}});
    m.component("comp1").geom("geom1").create("bCore", "Block"); m.component("comp1").geom("geom1").feature("bCore").set("pos", new double[]{{0, 0, 0}}); m.component("comp1").geom("geom1").feature("bCore").set("size", new double[]{{0.5*w, hf, zDom}});
    m.component("comp1").geom("geom1").create("bHeat", "Block"); m.component("comp1").geom("geom1").feature("bHeat").set("pos", new double[]{{0, yh0, 0}}); m.component("comp1").geom("geom1").feature("bHeat").set("size", new double[]{{0.5*wh, th, zHalfH}});
    m.component("comp1").geom("geom1").run();

    m.component("comp1").selection().create("sSub", "Box"); m.component("comp1").selection("sSub").set("entitydim", 3);
    m.component("comp1").selection("sSub").set("condition", "inside"); m.component("comp1").selection("sSub").set("ymax", -hbox + 1e-9);
    m.component("comp1").selection().create("sAir", "Box"); m.component("comp1").selection("sAir").set("entitydim", 3);
    m.component("comp1").selection("sAir").set("condition", "inside"); m.component("comp1").selection("sAir").set("ymin", yOx - 1e-9);
    m.component("comp1").selection().create("sCore", "Box"); m.component("comp1").selection("sCore").set("entitydim", 3);
    m.component("comp1").selection("sCore").set("condition", "inside");
    m.component("comp1").selection("sCore").set("xmax", 0.5*w + 1e-9); m.component("comp1").selection("sCore").set("ymin", -1e-9); m.component("comp1").selection("sCore").set("ymax", hf + 1e-9);
    m.component("comp1").selection().create("sHeat", "Box"); m.component("comp1").selection("sHeat").set("entitydim", 3);
    m.component("comp1").selection("sHeat").set("condition", "inside");
    m.component("comp1").selection("sHeat").set("xmax", 0.5*wh + 1e-9); m.component("comp1").selection("sHeat").set("ymin", yh0 - 1e-9); m.component("comp1").selection("sHeat").set("ymax", yh1 + 1e-9);
    m.component("comp1").selection("sHeat").set("zmax", zHalfH + 1e-9);

    m.component("comp1").selection().create("bBot", "Box"); m.component("comp1").selection("bBot").set("entitydim", 2);
    m.component("comp1").selection("bBot").set("condition", "inside"); m.component("comp1").selection("bBot").set("ymax", ySub + 1e-9);
    m.component("comp1").selection().create("bX", "Box"); m.component("comp1").selection("bX").set("entitydim", 2);
    m.component("comp1").selection("bX").set("condition", "inside"); m.component("comp1").selection("bX").set("xmin", xDom - 1e-9);
    m.component("comp1").selection().create("bZ", "Box"); m.component("comp1").selection("bZ").set("entitydim", 2);
    m.component("comp1").selection("bZ").set("condition", "inside"); m.component("comp1").selection("bZ").set("zmin", zDom - 1e-9);
    m.component("comp1").selection().create("bSink", "Union"); m.component("comp1").selection("bSink").set("entitydim", 2);
    m.component("comp1").selection("bSink").set("input", new String[]{{"bBot","bX","bZ"}});

    m.component("comp1").physics().create("ht", "HeatTransfer", "geom1");
    m.component("comp1").physics("ht").feature("solid1").set("k_mat", "userdef"); m.component("comp1").physics("ht").feature("solid1").set("k", new String[]{{"1.38","0","0","0","1.38","0","0","0","1.38"}});
    for (String[] sp : new String[][]{{{{"htSub","sSub","148.0"}},{{"htAir","sAir","0.026"}},{{"htCore","sCore","148.0"}},{{"htHeat","sHeat","20.0"}}}}) {{
      m.component("comp1").physics("ht").create(sp[0], "SolidHeatTransferModel", 3);
      m.component("comp1").physics("ht").feature(sp[0]).selection().named(sp[1]);
      m.component("comp1").physics("ht").feature(sp[0]).set("k_mat", "userdef");
      m.component("comp1").physics("ht").feature(sp[0]).set("k", new String[]{{sp[2],"0","0","0",sp[2],"0","0","0",sp[2]}});
    }}
    double qVol = 1e-3 / (wh * th * 200.0e-6);
    m.component("comp1").physics("ht").create("hs1", "HeatSource", 3);
    m.component("comp1").physics("ht").feature("hs1").selection().named("sHeat");
    m.component("comp1").physics("ht").feature("hs1").set("Q0", String.valueOf(qVol));
    m.component("comp1").physics("ht").create("temp1", "TemperatureBoundary", 2);
    m.component("comp1").physics("ht").feature("temp1").selection().named("bSink");
    m.component("comp1").physics("ht").feature("temp1").set("T0", "300[K]");

    m.component("comp1").mesh().create("mesh1");
    m.component("comp1").mesh("mesh1").feature("size").set("hauto", 4);
    m.component("comp1").mesh("mesh1").create("ftet1", "FreeTet");
    m.component("comp1").mesh("mesh1").run();

    m.study().create("std1");
    m.study("std1").create("stat", "Stationary");
    m.study("std1").run();

    // Integrated optical phase shift is proportional to integral(Delta_T dz) along full waveguide / L_heater!
    m.result().numerical().create("intC", "IntVolume");
    m.result().numerical("intC").selection().named("sCore");
    m.result().numerical("intC").set("expr", new String[]{{"T - 300[K]"}});
    double volInt = m.result().numerical("intC").getReal()[0][0];
    // Core cross-section in quarter-domain is (0.5*w)*hf; effective mean Delta_T over L_heater = 200 um (half = 100 um) is volInt / ((0.5*w)*hf*100e-6)
    double rthEff3D = volInt / ((0.5 * w) * hf * zHalfH);

    m.result().numerical().create("ptMid", "Interp");
    m.result().numerical("ptMid").set("coord", new double[][]{{{{0.0}}, {{0.11e-6}}, {{0.0}}}});
    m.result().numerical("ptMid").set("expr", new String[]{{"T - 300[K]"}});
    double rthCenter3D = m.result().numerical("ptMid").getReal()[0][0];

    System.out.println("S10_3D_JSON:" + "{{\\"case_id\\":\\"SOI_220nm_Top_TiN_3D_L200um\\",\\"R_th_eff_integrated_K_per_mW\\":" + rthEff3D +
        ",\\"R_th_center_z0_K_per_mW\\":" + rthCenter3D + "}}");
  }}

  public static void run() {{
    solveUndercut();
    solveDopedRib();
    solveCoIntPNHeater();
    solve3DFiniteLenTopTiN();
  }}
  public static void main(String[] args) {{ run(); }}
}}
'''
    lines = _run_comsol_java("ComsolSuite10Thermal", java_src)
    rows = _parse_kv_lines(lines, "S10_JSON:")
    r3d = _parse_kv_lines(lines, "S10_3D_JSON:")[0]
    for r in rows:
        for k in list(r.keys()):
            if isinstance(r[k], float):
                r[k] = round(r[k], 4)
        print(f"[COMSOL Suite 10] {r['case_id']}: R_th={r['R_th_core_K_per_mW']:.4f} K/mW, xtalk={r['crosstalk_20um_pct']:.3f}%")
    r3d["R_th_eff_integrated_K_per_mW"] = round(r3d["R_th_eff_integrated_K_per_mW"], 4)
    r3d["R_th_center_z0_K_per_mW"] = round(r3d["R_th_center_z0_K_per_mW"], 4)
    print(f"[COMSOL Suite 10 3D] Top TiN L=200um: R_th_eff(int)={r3d['R_th_eff_integrated_K_per_mW']:.4f}, R_th(z=0)={r3d['R_th_center_z0_K_per_mW']:.4f} K/mW")
    return rows, r3d


def main() -> None:
    t_all = time.perf_counter()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_p = SCRATCH_DIR / "comsol_p2_ckpt.json"
    ckpt = json.loads(ckpt_p.read_text(encoding="utf-8")) if ckpt_p.is_file() else {}
    specs = json.loads(SPECS_P2_JSON.read_text(encoding="utf-8"))

    if "suite6a_butt_couplers" not in ckpt or "suite11_fab_bias" not in ckpt:
        s6a, s11 = run_comsol_suite6a_and_11(specs)
        ckpt["suite6a_butt_couplers"] = s6a
        ckpt["suite11_fab_bias"] = s11
        ckpt_p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

    if "suite6b_apodized_gc" not in ckpt:
        ckpt["suite6b_apodized_gc"] = run_comsol_suite6b_apodized_gc(specs)
        ckpt_p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

    if "suite8_rf_electrodes" not in ckpt:
        ckpt["suite8_rf_electrodes"] = run_comsol_suite8_rf(specs)
        ckpt_p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

    if "suite9_si_pn_tcad" not in ckpt:
        ckpt["suite9_si_pn_tcad"] = run_comsol_suite9_pn_tcad(specs)
        ckpt_p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

    if "suite10_thermal_adv" not in ckpt or "suite10_thermal_3d_check" not in ckpt:
        s10, s10_3d = run_comsol_suite10_thermal(specs)
        ckpt["suite10_thermal_adv"] = s10
        ckpt["suite10_thermal_3d_check"] = s10_3d
        ckpt_p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

    payload = {
        "solver": "COMSOL Multiphysics 6.4.0.293 (Wave Optics ewfd + RF emw + AC/DC es + GeneralFormPDE g + 2D/3D Heat ht)",
        "host": f"{platform.system()}-{platform.machine()}",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        **ckpt,
    }
    OUT_P2_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved COMSOL Phase-2 results to {OUT_P2_JSON} in {payload['total_wall_time_s']} s")


if __name__ == "__main__":
    main()
