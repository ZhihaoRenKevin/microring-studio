"""Independent Commercial Validation for MicroRing Studio v3.12.0 Live First-Principles Engines:
COMSOL Multiphysics 6.4 (`ewfd` + Gyrotropic Complex Tensor + `solid` Eigenfrequency) &
Ansys Lumerical 2026 R1 (`3D FDTD` RTX 5090 GPU Blind `.GDS` Import).

Validates ALL 6 new v3.12.0 physical engines blindly against commercial solvers:
  1. COMSOL 6.4 `ewfd` — OPA Dense Emitter Array Twin-Waveguide Mutual Coupling (220x450 nm SOI, gap=0.40 um, 1550 nm)
  2. COMSOL 6.4 `ewfd` — GeSi Franz-Keldysh EAM Complex Modal Absorption (400x600 nm Ge_{0.992}Si_{0.008}, 1610 nm, 0 V vs -3 V)
  3. COMSOL 6.4 `ewfd` — X-Cut TFLN Single-Pass PPLN QPM Poling Period Lambda_QPM (600/300 nm ridge, 1550 nm -> 775 nm)
  4. COMSOL 6.4 `ewfd` (Full 3x3 Gyrotropic Tensor eps_yz = -eps_zy = +-i*g) — Ce:YIG-on-Si Magneto-Optic NRPS Isolator (TM0, 1550 nm)
  5. COMSOL 6.4 `solid` (3D/2D Plane-Stress Eigenfrequency) — Silicon Photonic MEMS Cantilever & Bridge Mechanical Resonance f_0
  6. Ansys Lumerical 2026 R1 `3D FDTD` (RTX 5090 GPU) — Blind `.GDS` Import of Synthesized Self-Imaging MMI Waveguide Crossing!
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

COMSOL_BIN = Path(r"C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64")
COMSOL_COMPILE = COMSOL_BIN / "comsolcompile.exe"
COMSOL_BATCH = COMSOL_BIN / "comsolbatch.exe"
LUMERICAL_PY = r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python"

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH = OUT_DIR / "_v312_val_scratch"
OUT_JSON = OUT_DIR / "v312_commercial_validation.json"
C0 = 299792458.0


def _run_comsol_java(class_name: str, java_src: str) -> list[str]:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    jpath = SCRATCH / f"{class_name}.java"
    cpath = SCRATCH / f"{class_name}.class"
    lpath = SCRATCH / f"{class_name}.log"
    jpath.write_text(java_src, encoding="utf-8")
    rc = subprocess.run([str(COMSOL_COMPILE), str(jpath)], cwd=str(SCRATCH), capture_output=True, text=True, timeout=120)
    if rc.returncode != 0:
        raise RuntimeError(f"comsolcompile failed:\n{rc.stdout}\n{rc.stderr}")
    rb = subprocess.run(
        [str(COMSOL_BATCH), "-np", "4", "-inputfile", str(cpath), "-batchlog", str(lpath)],
        cwd=str(SCRATCH), capture_output=True, text=True, timeout=480,
    )
    txt = lpath.read_text(encoding="utf-8", errors="ignore") if lpath.exists() else ""
    combined = (rb.stdout or "") + "\n" + (rb.stderr or "") + "\n" + txt
    if rb.returncode != 0:
        raise RuntimeError(f"comsolbatch failed:\n{combined[-1500:]}")
    return combined.splitlines()


def run_comsol_v312_suites() -> dict:
    from microring import (
        EAMConfig,
        OPALiDARConfig,
        OpticalIsolatorConfig,
        PPLNQPMConfig,
        PhotonicMEMSConfig,
        franz_keldysh_gesi_alpha_dn,
        simulate_eam_modulator,
        simulate_optical_isolator,
        simulate_optical_phased_array,
        simulate_photonic_mems_switch,
        simulate_ppln_qpm,
    )
    from microring.materials import T_REF_K, get_material_fn

    # 1. MicroRing references
    mr_opa = simulate_optical_phased_array(OPALiDARConfig(pitch_layout="uniform", array_pitch_um=0.85))["metrics"]
    mr_eam = simulate_eam_modulator(EAMConfig(eam_type="gesi_fke", wavelength_nm=1610.0))["metrics"]
    mr_ppln = simulate_ppln_qpm(PPLNQPMConfig(sidewall_deg=90.0))["metrics"]
    mr_iso = simulate_optical_isolator(OpticalIsolatorConfig(iso_type="magneto_optic_ceyig"))["metrics"]
    mr_mems_c = simulate_photonic_mems_switch(PhotonicMEMSConfig(beam_type="cantilever"))["metrics"]
    mr_mems_b = simulate_photonic_mems_switch(PhotonicMEMSConfig(beam_type="doubly_clamped"))["metrics"]

    # Material constants at exact wavelengths
    n_si_1550 = float(get_material_fn("Si")(1.55e-6, T_REF_K).nx)
    n_sio2_1550 = float(get_material_fn("SiO2")(1.55e-6, T_REF_K).nx)
    n_sio2_1610 = float(get_material_fn("SiO2")(1.61e-6, T_REF_K).nx)
    n_sio2_775 = float(get_material_fn("SiO2")(0.775e-6, T_REF_K).nx)
    # X-cut TFLN: TE0 sees extraordinary n_x = n_e, TM sees n_y = n_o
    ln_1550 = get_material_fn("LN_Xcut")(1.55e-6, T_REF_K)
    ln_775 = get_material_fn("LN_Xcut")(0.775e-6, T_REF_K)

    # EAM GeSi complex indices at V_on=0V (E=14 kV/cm) and V_off=3V (E=74 kV/cm)
    _, k_eam_on, dn_eam_on = franz_keldysh_gesi_alpha_dn(1610.0, (0.0 + 0.70) / 0.50e-4, 0.008, 0.20)
    _, k_eam_off, dn_eam_off = franz_keldysh_gesi_alpha_dn(1610.0, (3.0 + 0.70) / 0.50e-4, 0.008, 0.20)
    n_gesi_base = 4.22 - 0.72 * 0.008
    n_eam_on, n_eam_off = n_gesi_base + dn_eam_on, n_gesi_base + dn_eam_off

    # Ce:YIG gyrotropic tensor g = 2 * n_yig * |theta_F| / k0
    k0_1550 = 2.0 * math.pi / 1.55e-6
    n_yig = 2.22
    k_yig = (40.0 * 1e2 * (math.log(10.0) / 10.0)) / (2.0 * k0_1550)
    eps_yig_c = complex(n_yig, k_yig) ** 2
    g_gyro = (2.0 * n_yig * math.radians(4500.0) * 1e2) / k0_1550

    java_src = f"""
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolV312Validation {{

  static double[] solveOPATwin() throws Exception {{
    double wl = 1.55e-6, w = 0.45e-6, h = 0.22e-6, g = 0.40e-6, eps = 1e-10;
    double xDom = w + 0.5*g + 1.8e-6;
    Model m = ModelUtil.create("opa");
    m.component().create("c1", true);
    m.component("c1").geom().create("g1", 2);
    m.component("c1").geom("g1").create("rC", "Rectangle");
    m.component("c1").geom("g1").feature("rC").set("pos", new double[]{{-xDom, -1.6e-6}}).set("size", new double[]{{2*xDom, 3.2e-6 + h}});
    m.component("c1").geom("g1").create("rL", "Rectangle");
    m.component("c1").geom("g1").feature("rL").set("pos", new double[]{{-0.5*g - w, 0.0}}).set("size", new double[]{{w, h}});
    m.component("c1").geom("g1").create("rR", "Rectangle");
    m.component("c1").geom("g1").feature("rR").set("pos", new double[]{{+0.5*g, 0.0}}).set("size", new double[]{{w, h}});
    m.component("c1").geom("g1").run();
    m.component("c1").selection().create("selC", "Box");
    m.component("c1").selection("selC").set("entitydim", 2).set("condition", "inside").set("xmin", -0.5*g-w-eps).set("xmax", 0.5*g+w+eps).set("ymin", -eps).set("ymax", h+eps);
    m.component("c1").physics().create("ewfd", "ElectromagneticWavesFrequencyDomain", "g1");
    m.component("c1").physics("ewfd").feature("wee1").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", "{n_sio2_1550:.8f}").set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").physics("ewfd").create("wC", "WaveEquationElectric", 2);
    m.component("c1").physics("ewfd").feature("wC").selection().named("selC");
    m.component("c1").physics("ewfd").feature("wC").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", "{n_si_1550:.8f}").set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").mesh().create("m1", "g1");
    m.component("c1").mesh("m1").feature().create("sz", "Size");
    m.component("c1").mesh("m1").feature("sz").selection().named("selC");
    m.component("c1").mesh("m1").feature("sz").set("custom", true).set("hmaxactive", true).set("hmax", 0.024e-6);
    m.component("c1").mesh("m1").feature().create("ftri1", "FreeTri");
    m.component("c1").mesh("m1").run();
    m.study().create("s1");
    m.study("s1").create("mode", "ModeAnalysis");
    m.study("s1").feature("mode").set("modeFreq", "299792458/" + wl).set("neigs", 6).set("neigsactive", true).set("shift", "2.40").set("shiftactive", true);
    m.study("s1").run();
    m.result().numerical().create("g1", "EvalGlobal");
    m.result().numerical("g1").set("expr", new String[]{{"real(ewfd.neff)"}}).set("data", "dset1");
    double[][] v = m.result().numerical("g1").getReal();
    m.result().numerical().create("i1", "IntSurface");
    m.result().numerical("i1").selection().named("selC");
    m.result().numerical("i1").set("expr", new String[]{{"abs(ewfd.Ex)^2", "abs(ewfd.Ey)^2"}}).set("data", "dset1");
    double[][] iv = m.result().numerical("i1").getReal();
    double n1=-1, n2=-1;
    for (int j=0; j<v[0].length; j++) {{
      double te = iv[0][j] / Math.max(iv[0][j]+iv[1][j], 1e-30);
      if (te > 0.7 && v[0][j] > 2.0) {{
        if (v[0][j] > n1) {{ n2=n1; n1=v[0][j]; }}
        else if (v[0][j] > n2) {{ n2=v[0][j]; }}
      }}
    }}
    double kappaCm = (Math.PI * (n1 - n2) / wl) * 1e-2;
    ModelUtil.remove("opa");
    return new double[]{{n1 - n2, kappaCm}};
  }}

  static double[] solveEAM(String tag, double nCore, double kCore) throws Exception {{
    double wl = 1.61e-6, w = 0.60e-6, h = 0.40e-6, eps = 1e-10;
    Model m = ModelUtil.create(tag);
    m.component().create("c1", true);
    m.component("c1").geom().create("g1", 2);
    m.component("c1").geom("g1").create("rCl", "Rectangle");
    m.component("c1").geom("g1").feature("rCl").set("pos", new double[]{{-1.8e-6, -1.4e-6}}).set("size", new double[]{{3.6e-6, 2.8e-6 + h}});
    m.component("c1").geom("g1").create("rCo", "Rectangle");
    m.component("c1").geom("g1").feature("rCo").set("pos", new double[]{{-0.5*w, 0.0}}).set("size", new double[]{{w, h}});
    m.component("c1").geom("g1").run();
    m.component("c1").selection().create("selC", "Box");
    m.component("c1").selection("selC").set("entitydim", 2).set("condition", "inside").set("xmin", -0.5*w-eps).set("xmax", 0.5*w+eps).set("ymin", -eps).set("ymax", h+eps);
    m.component("c1").physics().create("ewfd", "ElectromagneticWavesFrequencyDomain", "g1");
    m.component("c1").physics("ewfd").feature("wee1").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", "{n_sio2_1610:.8f}").set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").physics("ewfd").create("wC", "WaveEquationElectric", 2);
    m.component("c1").physics("ewfd").feature("wC").selection().named("selC");
    m.component("c1").physics("ewfd").feature("wC").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", ""+nCore).set("ki_mat", "userdef").set("ki", ""+kCore).set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").mesh().create("m1", "g1");
    m.component("c1").mesh("m1").feature().create("sz", "Size");
    m.component("c1").mesh("m1").feature("sz").selection().named("selC");
    m.component("c1").mesh("m1").feature("sz").set("custom", true).set("hmaxactive", true).set("hmax", 0.025e-6);
    m.component("c1").mesh("m1").feature().create("ftri1", "FreeTri");
    m.component("c1").mesh("m1").run();
    m.study().create("s1");
    m.study("s1").create("mode", "ModeAnalysis");
    m.study("s1").feature("mode").set("modeFreq", "299792458/" + wl).set("neigs", 4).set("neigsactive", true).set("shift", "3.90").set("shiftactive", true);
    m.study("s1").run();
    m.result().numerical().create("g1", "EvalGlobal");
    m.result().numerical("g1").set("expr", new String[]{{"real(ewfd.neff)", "abs(imag(ewfd.neff))"}}).set("data", "dset1");
    double[][] v = m.result().numerical("g1").getReal();
    m.result().numerical().create("i1", "IntSurface");
    m.result().numerical("i1").selection().named("selC");
    m.result().numerical("i1").set("expr", new String[]{{"abs(ewfd.Ex)^2", "abs(ewfd.Ey)^2"}}).set("data", "dset1");
    double[][] iv = m.result().numerical("i1").getReal();
    int best = 0; double bs = -1;
    for (int j=0; j<v[0].length; j++) {{
      double te = iv[0][j] / Math.max(iv[0][j]+iv[1][j], 1e-30);
      if (te > 0.55 && v[0][j] > bs) {{ bs = v[0][j]; best = j; }}
    }}
    double alphaDbUm = 2.0 * (2.0 * Math.PI / wl) * v[1][best] * (10.0 / Math.log(10.0)) * 1e-6;
    ModelUtil.remove(tag);
    return new double[]{{v[0][best], alphaDbUm}};
  }}

  static double solveTFLN(String tag, double wl, double nx, double ny, double nz, double nBox, double shift) throws Exception {{
    double w = 1.40e-6, h = 0.60e-6, hs = 0.30e-6, eps = 1e-10;
    Model m = ModelUtil.create(tag);
    m.component().create("c1", true);
    m.component("c1").geom().create("g1", 2);
    m.component("c1").geom("g1").create("rB", "Rectangle");
    m.component("c1").geom("g1").feature("rB").set("pos", new double[]{{-2.4e-6, -1.8e-6}}).set("size", new double[]{{4.8e-6, 1.8e-6}});
    m.component("c1").geom("g1").create("rT", "Rectangle");
    m.component("c1").geom("g1").feature("rT").set("pos", new double[]{{-2.4e-6, 0.0}}).set("size", new double[]{{4.8e-6, 1.8e-6 + h}});
    m.component("c1").geom("g1").create("rS", "Rectangle");
    m.component("c1").geom("g1").feature("rS").set("pos", new double[]{{-2.4e-6, 0.0}}).set("size", new double[]{{4.8e-6, hs}});
    m.component("c1").geom("g1").create("rR", "Rectangle");
    m.component("c1").geom("g1").feature("rR").set("pos", new double[]{{-0.5*w, hs}}).set("size", new double[]{{w, h - hs}});
    m.component("c1").geom("g1").run();
    m.component("c1").selection().create("selB", "Box");
    m.component("c1").selection("selB").set("entitydim", 2).set("condition", "inside").set("xmin", -2.5e-6).set("xmax", 2.5e-6).set("ymin", -1.9e-6).set("ymax", eps);
    m.component("c1").selection().create("selS", "Box");
    m.component("c1").selection("selS").set("entitydim", 2).set("condition", "inside").set("xmin", -2.5e-6).set("xmax", 2.5e-6).set("ymin", -eps).set("ymax", hs+eps);
    m.component("c1").selection().create("selR", "Box");
    m.component("c1").selection("selR").set("entitydim", 2).set("condition", "inside").set("xmin", -0.5*w-eps).set("xmax", 0.5*w+eps).set("ymin", hs-eps).set("ymax", h+eps);

    m.component("c1").physics().create("ewfd", "ElectromagneticWavesFrequencyDomain", "g1");
    m.component("c1").physics("ewfd").feature("wee1").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", "1.0").set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").physics("ewfd").create("wB", "WaveEquationElectric", 2);
    m.component("c1").physics("ewfd").feature("wB").selection().named("selB");
    m.component("c1").physics("ewfd").feature("wB").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", ""+nBox).set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    String[][] pairs = new String[][]{{ {{"wS", "selS"}}, {{"wR", "selR"}} }};
    for (String[] sp : pairs) {{
      m.component("c1").physics("ewfd").create(sp[0], "WaveEquationElectric", 2);
      m.component("c1").physics("ewfd").feature(sp[0]).selection().named(sp[1]);
      m.component("c1").physics("ewfd").feature(sp[0]).set("DisplacementFieldModel", "RelativePermittivity")
        .set("epsilonr_mat", "userdef")
        .set("epsilonr", new String[]{{""+(nx*nx), "0", "0", "0", ""+(ny*ny), "0", "0", "0", ""+(nz*nz)}})
        .set("mur_mat", "userdef").set("mur", "1").set("sigma_mat", "userdef").set("sigma", "0");
    }}
    m.component("c1").mesh().create("m1", "g1");
    m.component("c1").mesh("m1").feature().create("sz", "Size");
    m.component("c1").mesh("m1").feature("sz").selection().named("selR");
    m.component("c1").mesh("m1").feature("sz").set("custom", true).set("hmaxactive", true).set("hmax", 0.035e-6);
    m.component("c1").mesh("m1").feature().create("ftri1", "FreeTri");
    m.component("c1").mesh("m1").run();
    m.study().create("s1");
    m.study("s1").create("mode", "ModeAnalysis");
    m.study("s1").feature("mode").set("modeFreq", "299792458/" + wl).set("neigs", 4).set("neigsactive", true).set("shift", ""+shift).set("shiftactive", true);
    m.study("s1").run();
    m.result().numerical().create("g1", "EvalGlobal");
    m.result().numerical("g1").set("expr", new String[]{{"real(ewfd.neff)"}}).set("data", "dset1");
    double[][] v = m.result().numerical("g1").getReal();
    m.result().numerical().create("i1", "IntSurface");
    m.result().numerical("i1").selection().named("selR");
    m.result().numerical("i1").set("expr", new String[]{{"abs(ewfd.Ex)^2", "abs(ewfd.Ey)^2"}}).set("data", "dset1");
    double[][] iv = m.result().numerical("i1").getReal();
    double best = -1;
    for (int j=0; j<v[0].length; j++) {{
      double te = iv[0][j] / Math.max(iv[0][j]+iv[1][j], 1e-30);
      if (te > 0.7 && v[0][j] > best) best = v[0][j];
    }}
    ModelUtil.remove(tag);
    return best;
  }}

  static double solveCeYIG(String tag, double gSign) throws Exception {{
    double wl = 1.55e-6, w = 0.50e-6, h = 0.22e-6, hYig = 0.30e-6, eps = 1e-10;
    double gVal = gSign * {g_gyro:.9e};
    String eDiag = "({eps_yig_c.real:.8f})+({-eps_yig_c.imag:.8f})*i";
    String eyz = "(" + gVal + ")*i";
    String ezy = "(" + (-gVal) + ")*i";
    Model m = ModelUtil.create(tag);
    m.component().create("c1", true);
    m.component("c1").geom().create("g1", 2);
    m.component("c1").geom("g1").create("rB", "Rectangle");
    m.component("c1").geom("g1").feature("rB").set("pos", new double[]{{-2.0e-6, -1.6e-6}}).set("size", new double[]{{4.0e-6, 1.6e-6}});
    m.component("c1").geom("g1").create("rY", "Rectangle");
    m.component("c1").geom("g1").feature("rY").set("pos", new double[]{{-2.0e-6, 0.0}}).set("size", new double[]{{4.0e-6, h + hYig}});
    m.component("c1").geom("g1").create("rTop", "Rectangle");
    m.component("c1").geom("g1").feature("rTop").set("pos", new double[]{{-2.0e-6, h + hYig}}).set("size", new double[]{{4.0e-6, 1.3e-6}});
    m.component("c1").geom("g1").create("rC", "Rectangle");
    m.component("c1").geom("g1").feature("rC").set("pos", new double[]{{-0.5*w, 0.0}}).set("size", new double[]{{w, h}});
    m.component("c1").geom("g1").run();
    m.component("c1").selection().create("selB", "Box");
    m.component("c1").selection("selB").set("entitydim", 2).set("condition", "inside").set("xmin", -2.1e-6).set("xmax", 2.1e-6).set("ymin", -1.7e-6).set("ymax", eps);
    m.component("c1").selection().create("selC", "Box");
    m.component("c1").selection("selC").set("entitydim", 2).set("condition", "inside").set("xmin", -0.5*w-eps).set("xmax", 0.5*w+eps).set("ymin", -eps).set("ymax", h+eps);
    m.component("c1").selection().create("selTop", "Box");
    m.component("c1").selection("selTop").set("entitydim", 2).set("condition", "inside").set("xmin", -2.1e-6).set("xmax", 2.1e-6).set("ymin", h+hYig-eps).set("ymax", h+hYig+1.4e-6);

    m.component("c1").physics().create("ewfd", "ElectromagneticWavesFrequencyDomain", "g1");
    // Default domain = Ce:YIG (gyrotropic full 3x3 tensor!)
    m.component("c1").physics("ewfd").feature("wee1").set("DisplacementFieldModel", "RelativePermittivity")
      .set("epsilonr_mat", "userdef")
      .set("epsilonr", new String[]{{eDiag, "0", "0", "0", eDiag, eyz, "0", ezy, eDiag}})
      .set("mur_mat", "userdef").set("mur", "1").set("sigma_mat", "userdef").set("sigma", "0");
    m.component("c1").physics("ewfd").create("wB", "WaveEquationElectric", 2);
    m.component("c1").physics("ewfd").feature("wB").selection().named("selB");
    m.component("c1").physics("ewfd").feature("wB").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", "{n_sio2_1550:.8f}").set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").physics("ewfd").create("wT", "WaveEquationElectric", 2);
    m.component("c1").physics("ewfd").feature("wT").selection().named("selTop");
    m.component("c1").physics("ewfd").feature("wT").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", "{n_yig:.8f}").set("ki_mat", "userdef").set("ki", "{k_yig:.8e}").set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").physics("ewfd").create("wC", "WaveEquationElectric", 2);
    m.component("c1").physics("ewfd").feature("wC").selection().named("selC");
    m.component("c1").physics("ewfd").feature("wC").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", "{n_si_1550:.8f}").set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");

    m.component("c1").mesh().create("m1", "g1");
    m.component("c1").mesh("m1").feature().create("sz", "Size");
    m.component("c1").mesh("m1").feature("sz").selection().named("selC");
    m.component("c1").mesh("m1").feature("sz").set("custom", true).set("hmaxactive", true).set("hmax", 0.022e-6);
    m.component("c1").mesh("m1").feature().create("ftri1", "FreeTri");
    m.component("c1").mesh("m1").run();
    m.study().create("s1");
    m.study("s1").create("mode", "ModeAnalysis");
    m.study("s1").feature("mode").set("modeFreq", "299792458/" + wl).set("neigs", 6).set("neigsactive", true).set("shift", "2.35").set("shiftactive", true);
    m.study("s1").run();
    m.result().numerical().create("g1", "EvalGlobal");
    m.result().numerical("g1").set("expr", new String[]{{"real(ewfd.neff)"}}).set("data", "dset1");
    double[][] v = m.result().numerical("g1").getReal();
    m.result().numerical().create("i1", "IntSurface");
    m.result().numerical("i1").selection().named("selC");
    m.result().numerical("i1").set("expr", new String[]{{"abs(ewfd.Ex)^2", "abs(ewfd.Ey)^2"}}).set("data", "dset1");
    double[][] iv = m.result().numerical("i1").getReal();
    double best = -1;
    for (int j=0; j<v[0].length; j++) {{
      double tm = iv[1][j] / Math.max(iv[0][j]+iv[1][j], 1e-30);
      if (tm > 0.55 && v[0][j] > best) best = v[0][j];
    }}
    ModelUtil.remove(tag);
    return best;
  }}

  static double solveMEMSF0(String tag, boolean doublyClamped) throws Exception {{
    double L = 28.0e-6, t = 0.22e-6, eps = 1e-10;
    Model m = ModelUtil.create(tag);
    m.component().create("c1", true);
    m.component("c1").geom().create("g1", 2);
    m.component("c1").geom("g1").create("r1", "Rectangle");
    m.component("c1").geom("g1").feature("r1").set("pos", new double[]{{0.0, 0.0}}).set("size", new double[]{{L, t}});
    m.component("c1").geom("g1").run();
    m.component("c1").selection().create("selFixL", "Box");
    m.component("c1").selection("selFixL").set("entitydim", 1).set("condition", "inside").set("xmin", -eps).set("xmax", eps).set("ymin", -eps).set("ymax", t+eps);
    if (doublyClamped) {{
      m.component("c1").selection().create("selFixR", "Box");
      m.component("c1").selection("selFixR").set("entitydim", 1).set("condition", "inside").set("xmin", L-eps).set("xmax", L+eps).set("ymin", -eps).set("ymax", t+eps);
    }}
    m.component("c1").physics().create("solid", "SolidMechanics", "g1");
    m.component("c1").physics("solid").feature("lemm1").set("E_mat", "userdef").set("E", "169e9").set("nu_mat", "userdef").set("nu", "0.22").set("rho_mat", "userdef").set("rho", "2329");
    m.component("c1").physics("solid").create("fix1", "Fixed", 1);
    m.component("c1").physics("solid").feature("fix1").selection().named("selFixL");
    if (doublyClamped) {{
      m.component("c1").physics("solid").create("fix2", "Fixed", 1);
      m.component("c1").physics("solid").feature("fix2").selection().named("selFixR");
    }}
    m.component("c1").mesh().create("m1", "g1");
    m.component("c1").mesh("m1").feature().create("sz", "Size");
    m.component("c1").mesh("m1").feature("sz").set("custom", true).set("hmaxactive", true).set("hmax", 0.11e-6);
    m.component("c1").mesh("m1").feature().create("ftri1", "FreeTri");
    m.component("c1").mesh("m1").run();
    m.study().create("s1");
    m.study("s1").create("eig", "Eigenfrequency");
    m.study("s1").feature("eig").set("neigs", 4).set("neigsactive", true).set("shift", "1e5").set("shiftactive", true);
    m.study("s1").run();
    m.result().numerical().create("g1", "EvalGlobal");
    m.result().numerical("g1").set("expr", new String[]{{"real(solid.freq)"}}).set("data", "dset1");
    double[][] v = m.result().numerical("g1").getReal();
    double fMin = 1e30;
    for (int j=0; j<v[0].length; j++) if (v[0][j] > 1e3 && v[0][j] < fMin) fMin = v[0][j];
    ModelUtil.remove(tag);
    return fMin * 1e-6;
  }}

  public static void main(String[] args) throws Exception {{
    double[] opa = solveOPATwin();
    double[] eamOn = solveEAM("eamOn", {n_eam_on:.8f}, {k_eam_on:.9e});
    double[] eamOff = solveEAM("eamOff", {n_eam_off:.8f}, {k_eam_off:.9e});
    double n1550 = solveTFLN("ln1", 1.55e-6, {ln_1550.nx:.8f}, {ln_1550.ny:.8f}, {ln_1550.nz:.8f}, {n_sio2_1550:.8f}, 1.95);
    double n775  = solveTFLN("ln2", 0.775e-6, {ln_775.nx:.8f}, {ln_775.ny:.8f}, {ln_775.nz:.8f}, {n_sio2_775:.8f}, 2.15);
    double lamQpmUm = 0.775 / Math.abs(n775 - n1550);
    double nYigP = solveCeYIG("yigP", +1.0);
    double nYigM = solveCeYIG("yigM", -1.0);
    double nrpsCm = (2.0 * Math.PI / 1.55e-6) * Math.abs(nYigP - nYigM) * 1e-2;
    double fCant = solveMEMSF0("mcant", false);
    double fBrid = solveMEMSF0("mbrid", true);
    System.out.println("JSON_V312:" +
      "{{\\"opa_dn\\":" + opa[0] + ",\\"opa_kappa_cm\\":" + opa[1] +
      ",\\"eam_alpha_on\\":" + eamOn[1] + ",\\"eam_alpha_off\\":" + eamOff[1] +
      ",\\"ppln_n1550\\":" + n1550 + ",\\"ppln_n775\\":" + n775 + ",\\"ppln_lambda_qpm_um\\":" + lamQpmUm +
      ",\\"ceyig_nrps_cm\\":" + nrpsCm +
      ",\\"mems_cant_MHz\\":" + fCant + ",\\"mems_bridge_MHz\\":" + fBrid + "}}");
  }}
}}
"""
    print("Running COMSOL 6.4 validation across OPA, GeSi EAM, PPLN QPM, Ce:YIG Gyrotropic NRPS & MEMS...")
    lines = _run_comsol_java("ComsolV312Validation", java_src)
    cm = {}
    for ln in lines:
        if "JSON_V312:" in ln:
            cm = json.loads(ln.split("JSON_V312:", 1)[1].strip())
    return {
        "opa_mutual_coupling": {
            "comsol_kappa_per_cm": round(cm["opa_kappa_cm"], 3),
            "microring_kappa_per_cm": mr_opa["mutual_coupling_kappa_per_cm"],
            "rel_err_pct": round(abs(mr_opa["mutual_coupling_kappa_per_cm"] - cm["opa_kappa_cm"]) / max(cm["opa_kappa_cm"], 1e-6) * 100.0, 2),
        },
        "gesi_fke_eam": {
            "comsol_alpha_ON_dB_um": round(cm["eam_alpha_on"], 5),
            "microring_alpha_ON_dB_um": mr_eam["alpha_mode_ON_dB_per_um"],
            "comsol_alpha_OFF_dB_um": round(cm["eam_alpha_off"], 5),
            "microring_alpha_OFF_dB_um": mr_eam["alpha_mode_OFF_dB_per_um"],
            "rel_err_OFF_pct": round(abs(mr_eam["alpha_mode_OFF_dB_per_um"] - cm["eam_alpha_off"]) / max(cm["eam_alpha_off"], 1e-6) * 100.0, 2),
        },
        "tfln_ppln_qpm": {
            "comsol_n1550": round(cm["ppln_n1550"], 5),
            "microring_n1550": mr_ppln["n_eff_fundamental"],
            "comsol_n775": round(cm["ppln_n775"], 5),
            "microring_n775": mr_ppln["n_eff_harmonic"],
            "comsol_Lambda_QPM_um": round(cm["ppln_lambda_qpm_um"], 4),
            "microring_Lambda_QPM_um": mr_ppln["QPM_poling_period_um"],
            "rel_err_Lambda_pct": round(abs(mr_ppln["QPM_poling_period_um"] - cm["ppln_lambda_qpm_um"]) / max(cm["ppln_lambda_qpm_um"], 1e-6) * 100.0, 2),
        },
        "ceyig_gyrotropic_isolator": {
            "comsol_NRPS_rad_per_cm": round(cm["ceyig_nrps_cm"], 3),
            "microring_NRPS_rad_per_cm": mr_iso["NRPS_rad_per_cm"],
            "rel_err_pct": round(abs(mr_iso["NRPS_rad_per_cm"] - cm["ceyig_nrps_cm"]) / max(cm["ceyig_nrps_cm"], 1e-6) * 100.0, 2),
        },
        "photonic_mems_eigenfreq": {
            "comsol_cantilever_MHz": round(cm["mems_cant_MHz"], 4),
            "microring_cantilever_MHz": mr_mems_c["mechanical_resonance_MHz"],
            "cantilever_err_pct": round(abs(mr_mems_c["mechanical_resonance_MHz"] - cm["mems_cant_MHz"]) / max(cm["mems_cant_MHz"], 1e-6) * 100.0, 2),
            "comsol_bridge_MHz": round(cm["mems_bridge_MHz"], 4),
            "microring_bridge_MHz": mr_mems_b["mechanical_resonance_MHz"],
            "bridge_err_pct": round(abs(mr_mems_b["mechanical_resonance_MHz"] - cm["mems_bridge_MHz"]) / max(cm["mems_bridge_MHz"], 1e-6) * 100.0, 2),
        },
    }


def run_lumerical_3d_fdtd_crossing_blind_gds() -> dict:
    """Blind .GDS import of MicroRing Studio's synthesized MMI Waveguide Crossing into Lumerical 2026 R1 3D FDTD (RTX 5090 GPU)."""
    from microring import WaveguideCrossingConfig, simulate_waveguide_crossing

    SCRATCH.mkdir(parents=True, exist_ok=True)
    crs_gds = SCRATCH / "mmi_crossing_v312.gds"
    mr_crs = simulate_waveguide_crossing(WaveguideCrossingConfig(export_gds_path=str(crs_gds)))["metrics"]

    sys.path.insert(0, LUMERICAL_PY)
    import lumapi  # type: ignore

    print("\nRunning Ansys Lumerical 2026 R1 3D FDTD (RTX 5090 GPU) blind .GDS validation of MMI Crossing...")
    fdtd = lumapi.FDTD(hide=True)
    try:
        fdtd.newproject()
        fdtd.gdsimport(str(crs_gds), "MMI_CROSSING", "1:0", "Si (Silicon) - Palik", 0.0, 0.22e-6)
        hl = 0.5 * mr_crs["total_footprint_um"] * 1e-6
        span = 2.0 * (hl + 1.8e-6)
        fdtd.addfdtd()
        fdtd.set("dimension", "3D")
        fdtd.set("x", 0.0)
        fdtd.set("x span", span)
        fdtd.set("y", 0.0)
        fdtd.set("y span", span)
        fdtd.set("z", 0.11e-6)
        fdtd.set("z span", 1.8e-6)
        fdtd.set("background index", 1.4440)
        fdtd.set("mesh accuracy", 3)
        fdtd.set("simulation time", 450e-15)

        fdtd.addmode()
        fdtd.set("injection axis", "x-axis")
        fdtd.set("direction", "Forward")
        fdtd.set("x", -(hl + 1.1e-6))
        fdtd.set("y", 0.0)
        fdtd.set("y span", 2.0e-6)
        fdtd.set("z", 0.11e-6)
        fdtd.set("z span", 1.4e-6)
        fdtd.set("center wavelength", 1.55e-6)
        fdtd.set("wavelength span", 0.0)
        fdtd.set("mode selection", "fundamental TE mode")

        fdtd.addpower()
        fdtd.set("name", "thru")
        fdtd.set("monitor type", "2D X-normal")
        fdtd.set("x", +(hl + 1.1e-6))
        fdtd.set("y", 0.0)
        fdtd.set("y span", 2.0e-6)
        fdtd.set("z", 0.11e-6)
        fdtd.set("z span", 1.4e-6)

        fdtd.addpower()
        fdtd.set("name", "xt_top")
        fdtd.set("monitor type", "2D Y-normal")
        fdtd.set("x", 0.0)
        fdtd.set("x span", 2.0e-6)
        fdtd.set("y", +(hl + 1.1e-6))
        fdtd.set("z", 0.11e-6)
        fdtd.set("z span", 1.4e-6)

        fsp = SCRATCH / "mmi_crossing_3d.fsp"
        fdtd.save(str(fsp))
        fdtd.run("FDTD", "GPU")
        t_3d = float(np.real(fdtd.transmission("thru")))
        xt_3d = abs(float(np.real(fdtd.transmission("xt_top"))))
        il_3d_db = -10.0 * math.log10(max(t_3d, 1e-6))
        xt_3d_db = 10.0 * math.log10(max(xt_3d, 1e-7))
    finally:
        fdtd.close()

    return {
        "synthesized_MMI_length_um": mr_crs["synthesized_MMI_length_um"],
        "microring_2D_FDFD_IL_dB": mr_crs["FDFD_insertion_loss_dB"],
        "microring_2D_FDFD_XT_dB": mr_crs["FDFD_adjacent_crosstalk_dB"],
        "lumerical_3D_FDTD_T_pct": round(t_3d * 100.0, 2),
        "lumerical_3D_FDTD_IL_dB": round(il_3d_db, 4),
        "lumerical_3D_FDTD_XT_dB": round(xt_3d_db, 2),
    }


def main() -> None:
    t0 = time.perf_counter()
    comsol_res = run_comsol_v312_suites()
    lum_res = run_lumerical_3d_fdtd_crossing_blind_gds()
    elapsed_s = time.perf_counter() - t0
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.node(),
        "elapsed_s": round(elapsed_s, 1),
        "comsol_6_4": comsol_res,
        "lumerical_3d_fdtd_gpu": lum_res,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print("\n=== v3.12.0 INDEPENDENT COMMERCIAL VALIDATION SUMMARY ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
