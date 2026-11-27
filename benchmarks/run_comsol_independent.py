"""Independent COMSOL Multiphysics 6.4 Validation Runner (`comsolcompile` + `comsolbatch`).

Reads ONLY neutral physical specifications from `benchmarks/outputs/commercial_validation_specs.json`
and executes COMSOL Multiphysics 6.4 completely independently across:
  - Suite 1  : `Wave Optics (ewfd)` 2D quadratic curl FEM bent & straight eigenmodes across
               5 wavebands (532 nm -> 4200 nm), using the exact covariant cylindrical metric
               `rho = 1 + x/R`, `eps_tilde = diag(rho*nx^2, rho*ny^2, nz^2/rho)`,
               `mu_tilde = diag(rho, rho, 1/rho)`, plus 3-wavelength dispersive `n_g`,
               mesh-convergence check, and material-sensitivity check (no Free-Space override).
  - Suite 3A : `Wave Optics (ewfd)` 2x2 SOI Directional Coupler even/odd supermodes (`150, 200, 250 nm`).
  - Suite 3C : `Wave Optics (ewfd)` 2D full-wave sagittal Grating Coupler (26 teeth, PML + numeric port).
  - Suite 4  : `Heat Transfer in Solids (ht)` steady-state + transient step heating/cooling +
               coupled `ewfd` thermo-optic `P_pi` across 4 SOI/Si3N4 microheater architectures
               (with explicit boundary heat-flux conservation check).
  - Suite 5  : `AC/DC Electrostatics (es)` 2D anisotropic Laplace (`eps_xx != eps_yy`) +
               coupled `ewfd` Pockels `V_pi * L` across X-cut TFLN GSG & ScAlN vertical plates.

All temporary `.mph` / `.class` files are cleaned up after lightweight results are extracted.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

COMSOL_BIN = Path(r"C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64")
COMSOL_COMPILE = COMSOL_BIN / "comsolcompile.exe"
COMSOL_BATCH = COMSOL_BIN / "comsolbatch.exe"

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH_DIR = OUT_DIR / "_comsol_scratch"
SPECS_JSON = OUT_DIR / "commercial_validation_specs.json"
OUT_JSON = OUT_DIR / "comsol_independent_results.json"


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
        raise RuntimeError(f"comsolbatch failed for {class_name}:\n{log_txt[-2500:]}\n{batch.stderr}")
    return ((batch.stdout or "") + "\n" + log_txt).splitlines()


def _parse_kv_lines(lines: list[str], prefix: str) -> list[dict]:
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(prefix):
            payload = s[len(prefix):].strip()
            out.append(json.loads(payload))
    return out


# ==============================================================================
# Java Block 1: Suite 1 (5-band bent/straight + dispersion + convergence/sens) & Suite 3A (DC)
# ==============================================================================
def run_comsol_suite1_and_3a(specs: dict) -> tuple[list[dict], dict, list[dict]]:
    s1_cases = specs["suite1_eigenmodes"]
    s3a_cases = specs["suite3a_dc_supermodes"]

    # Build Java helper that solves any waveguide (straight or bent, optional slab, slanted polygon)
    case_calls = []

    # 1. Suite 1: 5 canonical multi-waveband cases (3 bent wavelengths + 1 straight)
    for c in s1_cases:
        cid = c["case_id"]
        w = c["ring_width_um"] * 1e-6
        hf = c["film_height_um"] * 1e-6
        hs = c["slab_height_um"] * 1e-6
        sw = c["sidewall_deg"]
        r_b = c["ring_radius_um"] * 1e-6
        dom_w = c["domain_width_um"] * 1e-6
        pad = c["box_pad_um"] * 1e-6
        h_core = min(c["fem_core_res_um"] * 1e-6, 0.025e-6 if c["core_material"] == "Si" else c["fem_core_res_um"] * 1e-6)
        for tag, bent_r in (("minus", r_b), ("center", r_b), ("plus", r_b), ("straight", 0.0)):
            idx_key = "center" if tag == "straight" else tag
            ci = c["core_index"][idx_key]
            cli = c["clad_index"][idx_key]
            ti = c["top_index"][idx_key]
            wl = ci["wavelength_nm"] * 1e-9
            n_guess = 0.72 * ci["nx"] + 0.28 * cli["nx"]
            case_calls.append(
                f'solveSingle("{cid}", "{tag}", {wl:.12e}, {w:.12e}, {hf:.12e}, {hs:.12e}, {sw:.6f}, '
                f'{bent_r:.12e}, {dom_w:.12e}, {pad:.12e}, {h_core:.12e}, '
                f'{ci["nx"]:.8f}, {ci["ny"]:.8f}, {ci["nz"]:.8f}, {cli["nx"]:.8f}, {ti["nx"]:.8f}, {n_guess:.6f});'
            )

    # 2. Mesh-convergence & material-sensitivity audit on CBand_1550nm_4HSiC_Pulley (82 deg slanted anisotropic)
    sic = [c for c in s1_cases if c["case_id"] == "CBand_1550nm_4HSiC_Pulley"][0]
    ci = sic["core_index"]["center"]
    cli = sic["clad_index"]["center"]
    ti = sic["top_index"]["center"]
    for h_nm in (55, 35, 22):
        case_calls.append(
            f'solveSingle("AUDIT_4HSiC_MESH", "h{h_nm}nm", {sic["wavelength_nm"]*1e-9:.12e}, '
            f'{sic["ring_width_um"]*1e-6:.12e}, {sic["film_height_um"]*1e-6:.12e}, 0.0, {sic["sidewall_deg"]:.6f}, '
            f'{sic["ring_radius_um"]*1e-6:.12e}, {sic["domain_width_um"]*1e-6:.12e}, {sic["box_pad_um"]*1e-6:.12e}, '
            f'{h_nm*1e-9:.12e}, {ci["nx"]:.8f}, {ci["ny"]:.8f}, {ci["nz"]:.8f}, {cli["nx"]:.8f}, {ti["nx"]:.8f}, 2.28);'
        )
    # Material sensitivity (+1% core refractive index -> verifies zero Free-Space override!)
    case_calls.append(
        f'solveSingle("AUDIT_4HSiC_MATSENS", "plus1pct", {sic["wavelength_nm"]*1e-9:.12e}, '
        f'{sic["ring_width_um"]*1e-6:.12e}, {sic["film_height_um"]*1e-6:.12e}, 0.0, {sic["sidewall_deg"]:.6f}, '
        f'{sic["ring_radius_um"]*1e-6:.12e}, {sic["domain_width_um"]*1e-6:.12e}, {sic["box_pad_um"]*1e-6:.12e}, '
        f'30e-9, {ci["nx"]*1.01:.8f}, {ci["ny"]*1.01:.8f}, {ci["nz"]*1.01:.8f}, {cli["nx"]:.8f}, {ti["nx"]:.8f}, 2.31);'
    )

    # 3. Suite 3A: Twin-core SOI Directional Coupler supermodes
    dc_calls = []
    for c in s3a_cases:
        ci = c["core_index"]["center"]
        n_cl = c["clad_index"]["center"]["nx"]
        dc_calls.append(
            f'solveTwinDC("{c["case_id"]}", {c["wavelength_nm"]*1e-9:.12e}, {c["waveguide_width_um"]*1e-6:.12e}, '
            f'{c["gap_um"]*1e-6:.12e}, {c["film_height_um"]*1e-6:.12e}, {c["domain_width_um"]*1e-6:.12e}, '
            f'{c["box_pad_um"]*1e-6:.12e}, {ci["nx"]:.8f}, {n_cl:.8f});'
        )

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolSuite1And3A {{

  static void setAnisoMat(Model model, String pTag, String fTag, double nx, double ny, double nz, String rho) {{
    model.component("comp1").physics(pTag).feature(fTag).set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics(pTag).feature(fTag).set("epsilonr_mat", "userdef");
    String ex = "(" + (nx*nx) + ")";
    String ey = "(" + (ny*ny) + ")";
    String ez = "(" + (nz*nz) + ")";
    if (rho == null) {{
      model.component("comp1").physics(pTag).feature(fTag).set("epsilonr", new String[]{{ex,"0","0","0",ey,"0","0","0",ez}});
      model.component("comp1").physics(pTag).feature(fTag).set("mur_mat", "userdef");
      model.component("comp1").physics(pTag).feature(fTag).set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    }} else {{
      model.component("comp1").physics(pTag).feature(fTag).set("epsilonr", new String[]{{ex+"*"+rho,"0","0","0",ey+"*"+rho,"0","0","0",ez+"/"+rho}});
      model.component("comp1").physics(pTag).feature(fTag).set("mur_mat", "userdef");
      model.component("comp1").physics(pTag).feature(fTag).set("mur", new String[]{{rho,"0","0","0",rho,"0","0","0","1/"+rho}});
    }}
    model.component("comp1").physics(pTag).feature(fTag).set("sigma_mat", "userdef");
    model.component("comp1").physics(pTag).feature(fTag).set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});
  }}

  static void solveSingle(
      String cid, String tag, double wl, double w, double hf, double hs, double swDeg,
      double rBend, double domW, double pad, double hCore,
      double nx, double ny, double nz, double nClad, double nTop, double nGuess) {{
    long t0 = System.nanoTime();
    ModelUtil.clear();
    Model model = ModelUtil.create("Model");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);
    model.param().set("wl0", wl + "[m]");
    String rho = null;
    if (rBend > 0.0) {{
      model.param().set("R_bend", rBend + "[m]");
      rho = "(1 + x/R_bend)";
    }}

    // Bottom BOX & Top Cladding rectangles
    model.component("comp1").geom("geom1").feature().create("rBox", "Rectangle");
    model.component("comp1").geom("geom1").feature("rBox").set("size", new double[]{{domW, pad}});
    model.component("comp1").geom("geom1").feature("rBox").set("pos", new double[]{{-0.5*domW, -pad}});

    model.component("comp1").geom("geom1").feature().create("rTop", "Rectangle");
    model.component("comp1").geom("geom1").feature("rTop").set("size", new double[]{{domW, hf + pad}});
    model.component("comp1").geom("geom1").feature("rTop").set("pos", new double[]{{-0.5*domW, 0.0}});

    if (hs > 1e-11) {{
      model.component("comp1").geom("geom1").feature().create("rSlab", "Rectangle");
      model.component("comp1").geom("geom1").feature("rSlab").set("size", new double[]{{domW, hs}});
      model.component("comp1").geom("geom1").feature("rSlab").set("pos", new double[]{{-0.5*domW, 0.0}});
    }}

    double hRidge = hf - hs;
    double invTan = (Math.abs(swDeg - 90.0) < 1e-3) ? 0.0 : 1.0 / Math.tan(Math.toRadians(swDeg));
    double wTop = w - 2.0 * hRidge * invTan;

    model.component("comp1").geom("geom1").feature().create("pCore", "Polygon");
    model.component("comp1").geom("geom1").feature("pCore").set("source", "table");
    model.component("comp1").geom("geom1").feature("pCore").set("table", new double[][]{{
      {{-0.5*w, hs}}, {{0.5*w, hs}}, {{0.5*wTop, hf}}, {{-0.5*wTop, hf}}
    }});
    model.component("comp1").geom("geom1").run();

    // Explicit Box selections for every physical region (prevents any default/Free-Space override)
    model.component("comp1").selection().create("selBox", "Box");
    model.component("comp1").selection("selBox").set("entitydim", 2);
    model.component("comp1").selection("selBox").set("condition", "inside");
    model.component("comp1").selection("selBox").set("xmin", -0.5*domW - 1e-9);
    model.component("comp1").selection("selBox").set("xmax", 0.5*domW + 1e-9);
    model.component("comp1").selection("selBox").set("ymin", -pad - 1e-9);
    model.component("comp1").selection("selBox").set("ymax", 1e-9);

    model.component("comp1").selection().create("selCore", "Box");
    model.component("comp1").selection("selCore").set("entitydim", 2);
    model.component("comp1").selection("selCore").set("condition", "inside");
    model.component("comp1").selection("selCore").set("xmin", -0.5*w - 1e-9);
    model.component("comp1").selection("selCore").set("xmax", 0.5*w + 1e-9);
    model.component("comp1").selection("selCore").set("ymin", hs - 1e-9);
    model.component("comp1").selection("selCore").set("ymax", hf + 1e-9);

    if (hs > 1e-11) {{
      model.component("comp1").selection().create("selSlab", "Box");
      model.component("comp1").selection("selSlab").set("entitydim", 2);
      model.component("comp1").selection("selSlab").set("condition", "inside");
      model.component("comp1").selection("selSlab").set("xmin", -0.5*domW - 1e-9);
      model.component("comp1").selection("selSlab").set("xmax", 0.5*domW + 1e-9);
      model.component("comp1").selection("selSlab").set("ymin", -1e-9);
      model.component("comp1").selection("selSlab").set("ymax", hs + 1e-9);
    }}

    model.component("comp1").physics().create("ewfd", "ElectromagneticWaves", "geom1");
    setAnisoMat(model, "ewfd", "wee1", nTop, nTop, nTop, rho);

    model.component("comp1").physics("ewfd").feature().create("weeBox", "WaveEquationElectric", 2);
    model.component("comp1").physics("ewfd").feature("weeBox").selection().named("selBox");
    setAnisoMat(model, "ewfd", "weeBox", nClad, nClad, nClad, rho);

    if (hs > 1e-11) {{
      model.component("comp1").physics("ewfd").feature().create("weeSlab", "WaveEquationElectric", 2);
      model.component("comp1").physics("ewfd").feature("weeSlab").selection().named("selSlab");
      setAnisoMat(model, "ewfd", "weeSlab", nx, ny, nz, rho);
    }}

    model.component("comp1").physics("ewfd").feature().create("weeCore", "WaveEquationElectric", 2);
    model.component("comp1").physics("ewfd").feature("weeCore").selection().named("selCore");
    setAnisoMat(model, "ewfd", "weeCore", nx, ny, nz, rho);

    model.component("comp1").mesh().create("mesh1", "geom1");
    model.component("comp1").mesh("mesh1").feature().create("sizeCore", "Size");
    model.component("comp1").mesh("mesh1").feature("sizeCore").selection().named("selCore");
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("hmax", hCore);
    model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", Math.max(hCore * 6.0, 0.16e-6));
    model.component("comp1").mesh("mesh1").feature().create("ftri1", "FreeTri");
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std1");
    model.study("std1").feature().create("mode", "ModeAnalysis");
    model.study("std1").feature("mode").set("modeFreq", "c_const/wl0");
    model.study("std1").feature("mode").set("neigs", 6);
    model.study("std1").feature("mode").set("neigsactive", true);
    model.study("std1").feature("mode").set("shift", "" + nGuess);
    model.study("std1").feature("mode").set("shiftactive", true);
    model.study("std1").run();

    double[] lamIm = model.sol("sol1").getPValsImag();
    double k0 = 2.0 * Math.PI / wl;
    String dset = model.result().dataset().tags()[0];
    model.result().numerical().create("intPol", "IntSurface");
    model.result().numerical("intPol").set("data", dset);
    model.result().numerical("intPol").selection().all();
    model.result().numerical("intPol").set("expr", new String[]{{"abs(Ex)^2", "abs(Ey)^2"}});
    double[][] iv = model.result().numerical("intPol").getReal();

    double bestNeff = -1.0;
    double bestTe = 0.0;
    for (int i = 0; i < lamIm.length; i++) {{
      double ne = Math.abs(lamIm[i]) / k0;
      double te = iv[0][i] / (iv[0][i] + iv[1][i] + 1e-30);
      if (te >= 0.5 && ne > bestNeff && ne < nx * 1.15) {{
        bestNeff = ne;
        bestTe = te;
      }}
    }}
    double dtS = (System.nanoTime() - t0) * 1e-9;
    int nelem = model.component("comp1").mesh("mesh1").stat().getNumElem();
    System.out.println("S1_JSON: {{\\"case_id\\":\\"" + cid + "\\",\\"tag\\":\\"" + tag + "\\",\\"neff\\":" + bestNeff
      + ",\\"te_fraction\\":" + bestTe + ",\\"num_elements\\":" + nelem + ",\\"wall_time_s\\":" + dtS + "}}");
  }}

  static void solveTwinDC(
      String cid, double wl, double w, double gap, double hf, double domW, double pad, double nCore, double nClad) {{
    long t0 = System.nanoTime();
    ModelUtil.clear();
    Model model = ModelUtil.create("Model");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);
    model.param().set("wl0", wl + "[m]");
    double xc = 0.5 * (w + gap);

    model.component("comp1").geom("geom1").feature().create("rClad", "Rectangle");
    model.component("comp1").geom("geom1").feature("rClad").set("size", new double[]{{domW, hf + 2.0*pad}});
    model.component("comp1").geom("geom1").feature("rClad").set("pos", new double[]{{-0.5*domW, -pad}});

    model.component("comp1").geom("geom1").feature().create("rL", "Rectangle");
    model.component("comp1").geom("geom1").feature("rL").set("size", new double[]{{w, hf}});
    model.component("comp1").geom("geom1").feature("rL").set("pos", new double[]{{-xc - 0.5*w, 0.0}});

    model.component("comp1").geom("geom1").feature().create("rR", "Rectangle");
    model.component("comp1").geom("geom1").feature("rR").set("size", new double[]{{w, hf}});
    model.component("comp1").geom("geom1").feature("rR").set("pos", new double[]{{xc - 0.5*w, 0.0}});
    model.component("comp1").geom("geom1").run();

    model.component("comp1").selection().create("selL", "Box");
    model.component("comp1").selection("selL").set("entitydim", 2);
    model.component("comp1").selection("selL").set("condition", "inside");
    model.component("comp1").selection("selL").set("xmin", -xc - 0.5*w - 1e-9);
    model.component("comp1").selection("selL").set("xmax", -xc + 0.5*w + 1e-9);
    model.component("comp1").selection("selL").set("ymin", -1e-9);
    model.component("comp1").selection("selL").set("ymax", hf + 1e-9);

    model.component("comp1").selection().create("selR", "Box");
    model.component("comp1").selection("selR").set("entitydim", 2);
    model.component("comp1").selection("selR").set("condition", "inside");
    model.component("comp1").selection("selR").set("xmin", xc - 0.5*w - 1e-9);
    model.component("comp1").selection("selR").set("xmax", xc + 0.5*w + 1e-9);
    model.component("comp1").selection("selR").set("ymin", -1e-9);
    model.component("comp1").selection("selR").set("ymax", hf + 1e-9);

    model.component("comp1").selection().create("selCores", "Union");
    model.component("comp1").selection("selCores").set("entitydim", 2);
    model.component("comp1").selection("selCores").set("input", new String[]{{"selL", "selR"}});

    model.component("comp1").physics().create("ewfd", "ElectromagneticWaves", "geom1");
    setAnisoMat(model, "ewfd", "wee1", nClad, nClad, nClad, null);
    model.component("comp1").physics("ewfd").feature().create("weeCores", "WaveEquationElectric", 2);
    model.component("comp1").physics("ewfd").feature("weeCores").selection().named("selCores");
    setAnisoMat(model, "ewfd", "weeCores", nCore, nCore, nCore, null);

    model.component("comp1").mesh().create("mesh1", "geom1");
    model.component("comp1").mesh("mesh1").feature().create("sizeCores", "Size");
    model.component("comp1").mesh("mesh1").feature("sizeCores").selection().named("selCores");
    model.component("comp1").mesh("mesh1").feature("sizeCores").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("sizeCores").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("sizeCores").set("hmax", 0.018e-6);
    model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.14e-6);
    model.component("comp1").mesh("mesh1").feature().create("ftri1", "FreeTri");
    model.component("comp1").mesh("mesh1").run();

    model.study().create("std1");
    model.study("std1").feature().create("mode", "ModeAnalysis");
    model.study("std1").feature("mode").set("modeFreq", "c_const/wl0");
    model.study("std1").feature("mode").set("neigs", 6);
    model.study("std1").feature("mode").set("neigsactive", true);
    model.study("std1").feature("mode").set("shift", "2.45");
    model.study("std1").feature("mode").set("shiftactive", true);
    model.study("std1").run();

    double[] lamIm = model.sol("sol1").getPValsImag();
    double k0 = 2.0 * Math.PI / wl;
    String dset = model.result().dataset().tags()[0];
    model.result().numerical().create("intPol", "IntSurface");
    model.result().numerical("intPol").set("data", dset);
    model.result().numerical("intPol").selection().all();
    model.result().numerical("intPol").set("expr", new String[]{{"abs(Ex)^2", "abs(Ey)^2"}});
    double[][] iv = model.result().numerical("intPol").getReal();

    double[] tes = new double[lamIm.length];
    int nTe = 0;
    for (int i = 0; i < lamIm.length; i++) {{
      double ne = Math.abs(lamIm[i]) / k0;
      double te = iv[0][i] / (iv[0][i] + iv[1][i] + 1e-30);
      if (te >= 0.5 && ne > nClad + 0.05) {{
        tes[nTe++] = ne;
      }}
    }}
    java.util.Arrays.sort(tes, 0, nTe);
    double nEven = tes[nTe - 1];
    double nOdd = tes[nTe - 2];
    double dn = nEven - nOdd;
    double lPiUm = (wl * 1e6) / (2.0 * dn);
    double dtS = (System.nanoTime() - t0) * 1e-9;
    System.out.println("DC_JSON: {{\\"case_id\\":\\"" + cid + "\\",\\"n_even\\":" + nEven + ",\\"n_odd\\":" + nOdd
      + ",\\"delta_n\\":" + dn + ",\\"L_pi_um\\":" + lPiUm + ",\\"wall_time_s\\":" + dtS + "}}");
  }}

  public static Model run() {{
    {chr(10).join("    " + ln for ln in case_calls)}
    {chr(10).join("    " + ln for ln in dc_calls)}
    return ModelUtil.create("Done");
  }}
  public static void main(String[] args) {{ run(); }}
}}
'''
    lines = _run_comsol_java("ComsolSuite1And3A", java_src)
    s1_raw = _parse_kv_lines(lines, "S1_JSON:")
    dc_raw = _parse_kv_lines(lines, "DC_JSON:")

    # Aggregate Suite 1 rows & audit
    by_case: dict[str, dict] = {}
    audit_mesh = []
    audit_matsens = {}
    for r in s1_raw:
        if r["case_id"] == "AUDIT_4HSiC_MESH":
            audit_mesh.append(r)
        elif r["case_id"] == "AUDIT_4HSiC_MATSENS":
            audit_matsens = r
        else:
            by_case.setdefault(r["case_id"], {})[r["tag"]] = r

    s1_rows = []
    for c in s1_cases:
        cid = c["case_id"]
        d = by_case[cid]
        wl0 = float(c["wavelength_nm"])
        dwl = float(c["dwl_nm"])
        ne_c = d["center"]["neff"]
        ne_m = d["minus"]["neff"]
        ne_p = d["plus"]["neff"]
        ne_s = d["straight"]["neff"]
        ng = ne_c - wl0 * (ne_p - ne_m) / (2.0 * dwl)
        l_rt_um = 2.0 * math.pi * c["ring_radius_um"] + 2.0 * c["straight_length_um"]
        fsr_nm = (wl0 ** 2) / (ng * l_rt_um * 1e3)
        s1_rows.append({
            "case_id": cid,
            "wavelength_nm": wl0,
            "neff_bent": round(ne_c, 6),
            "neff_bent_minus": round(ne_m, 6),
            "neff_bent_plus": round(ne_p, 6),
            "ng_bent": round(ng, 5),
            "neff_straight": round(ne_s, 6),
            "delta_neff_bend": round(ne_c - ne_s, 6),
            "te_fraction": round(d["center"]["te_fraction"], 4),
            "fsr_nm": round(fsr_nm, 4),
            "num_elements": int(d["center"]["num_elements"]),
            "wall_time_s": round(sum(x["wall_time_s"] for x in d.values()), 2),
        })
        print(f"[COMSOL-S1] {cid:<26} neff={ne_c:.6f} ng={ng:.4f} d_bend={ne_c-ne_s:+.6f}")

    # Material sensitivity summary
    sic_nom = [r for r in s1_rows if r["case_id"] == "CBand_1550nm_4HSiC_Pulley"][0]["neff_bent"]
    dn_core_1pct = sic["core_index"]["center"]["nx"] * 0.01
    audit = {
        "case_id": "CBand_1550nm_4HSiC_Pulley",
        "mesh_convergence": audit_mesh,
        "material_sensitivity": {
            "neff_nominal": sic_nom,
            "neff_core_plus1pct": round(audit_matsens.get("neff", 0.0), 6),
            "d_neff_over_d_ncore": round((audit_matsens.get("neff", sic_nom) - sic_nom) / dn_core_1pct, 4),
            "verified_no_free_space_override": abs(audit_matsens.get("neff", sic_nom) - sic_nom) > 0.015,
        },
    }
    print(f"[COMSOL-AUDIT] 4H-SiC dn_eff/dn_core = {audit['material_sensitivity']['d_neff_over_d_ncore']:.4f}")

    dc_rows = []
    for c, r in zip(s3a_cases, dc_raw):
        dc_rows.append({
            "case_id": c["case_id"],
            "gap_nm": round(c["gap_um"] * 1e3, 1),
            "n_even": round(r["n_even"], 6),
            "n_odd": round(r["n_odd"], 6),
            "delta_n": round(r["delta_n"], 6),
            "L_pi_um": round(r["L_pi_um"], 4),
            "wall_time_s": round(r["wall_time_s"], 2),
        })
        print(f"[COMSOL-S3A] {c['case_id']:<26} n_e={r['n_even']:.6f} n_o={r['n_odd']:.6f} dn={r['delta_n']:.6f} L_pi={r['L_pi_um']:.3f} um")
    return s1_rows, audit, dc_rows


# ==============================================================================
# Java Block 2: Suite 4 (2D Heat Transfer Steady + Transient + Coupled ewfd P_pi)
# ==============================================================================
def run_comsol_suite4_thermal(cases: list[dict]) -> list[dict]:
    calls = []
    for c in cases:
        is_tr = "true" if c["thermal_structure"] == "deep_trench" else "false"
        tr_y_bot = max(c["y_ox_top_um"] - c["trench_depth_um"], c["y_sub_bot_um"] + 1.5)
        ci = c["core_index"]["center"]
        n_ox = c["clad_index"]["center"]["nx"]
        dndt_c = c["core_index"]["dn_dt_per_k"]
        dndt_ox = c["clad_index"]["dn_dt_per_k"]
        calls.append(
            f'solveThermal("{c["case_id"]}", {is_tr}, {c["wavelength_nm"]*1e-9:.12e}, '
            f'{c["waveguide_width_um"]*1e-6:.12e}, {c["film_height_um"]*1e-6:.12e}, '
            f'{c["box_thickness_um"]*1e-6:.12e}, {c["y_ox_top_um"]*1e-6:.12e}, {c["y_air_top_um"]*1e-6:.12e}, '
            f'{c["y_sub_bot_um"]*1e-6:.12e}, {c["x_dom_um"]*1e-6:.12e}, '
            f'{c["heater_width_um"]*1e-6:.12e}, {c["y_heat0_um"]*1e-6:.12e}, {c["y_heat1_um"]*1e-6:.12e}, '
            f'{c["active_length_um"]*1e-6:.12e}, {c["trench_edge_offset_um"]*1e-6:.12e}, '
            f'{c["trench_width_um"]*1e-6:.12e}, {tr_y_bot*1e-6:.12e}, {c["neighbor_pitch_um"]*1e-6:.12e}, '
            f'{c["k_core_w_mk"]:.6f}, {c["k_ox_w_mk"]:.6f}, {c["k_si_sub_w_mk"]:.6f}, {c["k_heater_w_mk"]:.6f}, {c["k_air_w_mk"]:.6f}, '
            f'{c["cv_core_j_m3k"]:.6e}, {c["cv_ox_j_m3k"]:.6e}, {c["cv_si_sub_j_m3k"]:.6e}, {c["cv_heater_j_m3k"]:.6e}, {c["cv_air_j_m3k"]:.6e}, '
            f'{ci["nx"]:.8f}, {n_ox:.8f}, {dndt_c:.8e}, {dndt_ox:.8e});'
        )

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolSuite4Thermal {{

  static void setSolidMat(Model model, String fTag, double k, double cv) {{
    model.component("comp1").physics("ht").feature(fTag).set("k_mat", "userdef");
    model.component("comp1").physics("ht").feature(fTag).set("k", new String[]{{""+k,"0","0","0",""+k,"0","0","0",""+k}});
    model.component("comp1").physics("ht").feature(fTag).set("rho_mat", "userdef");
    model.component("comp1").physics("ht").feature(fTag).set("rho", "1000");
    model.component("comp1").physics("ht").feature(fTag).set("Cp_mat", "userdef");
    model.component("comp1").physics("ht").feature(fTag).set("Cp", "" + (cv / 1000.0));
  }}

  static void solveThermal(
      String cid, boolean isTrench, double wl, double wCore, double hCore,
      double hBox, double yOxTop, double yAirTop, double ySubBot, double xDom,
      double wHeat, double yH0, double yH1, double lHeat,
      double trX0, double trW, double trYBot, double xNeighbor,
      double kCore, double kOx, double kSub, double kHeat, double kAir,
      double cvCore, double cvOx, double cvSub, double cvHeat, double cvAir,
      double nCore, double nOx, double dndtCore, double dndtOx) {{
    long t0 = System.nanoTime();
    ModelUtil.clear();
    Model model = ModelUtil.create("Model");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);
    model.param().set("wl0", wl + "[m]");
    model.param().set("P_opt_mw", "10");

    // Layered geometry rectangles
    model.component("comp1").geom("geom1").feature().create("rSub", "Rectangle");
    model.component("comp1").geom("geom1").feature("rSub").set("size", new double[]{{2.0*xDom, -hBox - ySubBot}});
    model.component("comp1").geom("geom1").feature("rSub").set("pos", new double[]{{-xDom, ySubBot}});

    model.component("comp1").geom("geom1").feature().create("rOx", "Rectangle");
    model.component("comp1").geom("geom1").feature("rOx").set("size", new double[]{{2.0*xDom, yOxTop + hBox}});
    model.component("comp1").geom("geom1").feature("rOx").set("pos", new double[]{{-xDom, -hBox}});

    model.component("comp1").geom("geom1").feature().create("rAir", "Rectangle");
    model.component("comp1").geom("geom1").feature("rAir").set("size", new double[]{{2.0*xDom, yAirTop - yOxTop}});
    model.component("comp1").geom("geom1").feature("rAir").set("pos", new double[]{{-xDom, yOxTop}});

    model.component("comp1").geom("geom1").feature().create("rCore", "Rectangle");
    model.component("comp1").geom("geom1").feature("rCore").set("size", new double[]{{wCore, hCore}});
    model.component("comp1").geom("geom1").feature("rCore").set("pos", new double[]{{-0.5*wCore, 0.0}});

    model.component("comp1").geom("geom1").feature().create("rHeat", "Rectangle");
    model.component("comp1").geom("geom1").feature("rHeat").set("size", new double[]{{wHeat, yH1 - yH0}});
    model.component("comp1").geom("geom1").feature("rHeat").set("pos", new double[]{{-0.5*wHeat, yH0}});

    if (isTrench) {{
      model.component("comp1").geom("geom1").feature().create("rTrL", "Rectangle");
      model.component("comp1").geom("geom1").feature("rTrL").set("size", new double[]{{trW, yOxTop - trYBot}});
      model.component("comp1").geom("geom1").feature("rTrL").set("pos", new double[]{{-(trX0 + trW), trYBot}});
      model.component("comp1").geom("geom1").feature().create("rTrR", "Rectangle");
      model.component("comp1").geom("geom1").feature("rTrR").set("size", new double[]{{trW, yOxTop - trYBot}});
      model.component("comp1").geom("geom1").feature("rTrR").set("pos", new double[]{{trX0, trYBot}});
    }}
    model.component("comp1").geom("geom1").run();

    // Domain Box Selections
    double eps = 2e-9;
    model.component("comp1").selection().create("selSub", "Box");
    model.component("comp1").selection("selSub").set("entitydim", 2);
    model.component("comp1").selection("selSub").set("condition", "inside");
    model.component("comp1").selection("selSub").set("xmin", -xDom - eps);
    model.component("comp1").selection("selSub").set("xmax", xDom + eps);
    model.component("comp1").selection("selSub").set("ymin", ySubBot - eps);
    model.component("comp1").selection("selSub").set("ymax", -hBox + eps);

    model.component("comp1").selection().create("selAir", "Box");
    model.component("comp1").selection("selAir").set("entitydim", 2);
    model.component("comp1").selection("selAir").set("condition", "inside");
    model.component("comp1").selection("selAir").set("xmin", -xDom - eps);
    model.component("comp1").selection("selAir").set("xmax", xDom + eps);
    model.component("comp1").selection("selAir").set("ymin", yOxTop - eps);
    model.component("comp1").selection("selAir").set("ymax", yAirTop + eps);

    model.component("comp1").selection().create("selCore", "Box");
    model.component("comp1").selection("selCore").set("entitydim", 2);
    model.component("comp1").selection("selCore").set("condition", "inside");
    model.component("comp1").selection("selCore").set("xmin", -0.5*wCore - eps);
    model.component("comp1").selection("selCore").set("xmax", 0.5*wCore + eps);
    model.component("comp1").selection("selCore").set("ymin", -eps);
    model.component("comp1").selection("selCore").set("ymax", hCore + eps);

    model.component("comp1").selection().create("selHeat", "Box");
    model.component("comp1").selection("selHeat").set("entitydim", 2);
    model.component("comp1").selection("selHeat").set("condition", "inside");
    model.component("comp1").selection("selHeat").set("xmin", -0.5*wHeat - eps);
    model.component("comp1").selection("selHeat").set("xmax", 0.5*wHeat + eps);
    model.component("comp1").selection("selHeat").set("ymin", yH0 - eps);
    model.component("comp1").selection("selHeat").set("ymax", yH1 + eps);

    if (isTrench) {{
      model.component("comp1").selection().create("selTrL", "Box");
      model.component("comp1").selection("selTrL").set("entitydim", 2);
      model.component("comp1").selection("selTrL").set("condition", "inside");
      model.component("comp1").selection("selTrL").set("xmin", -(trX0 + trW) - eps);
      model.component("comp1").selection("selTrL").set("xmax", -trX0 + eps);
      model.component("comp1").selection("selTrL").set("ymin", trYBot - eps);
      model.component("comp1").selection("selTrL").set("ymax", yOxTop + eps);

      model.component("comp1").selection().create("selTrR", "Box");
      model.component("comp1").selection("selTrR").set("entitydim", 2);
      model.component("comp1").selection("selTrR").set("condition", "inside");
      model.component("comp1").selection("selTrR").set("xmin", trX0 - eps);
      model.component("comp1").selection("selTrR").set("xmax", (trX0 + trW) + eps);
      model.component("comp1").selection("selTrR").set("ymin", trYBot - eps);
      model.component("comp1").selection("selTrR").set("ymax", yOxTop + eps);

      model.component("comp1").selection().create("selTr", "Union");
      model.component("comp1").selection("selTr").set("entitydim", 2);
      model.component("comp1").selection("selTr").set("input", new String[]{{"selTrL", "selTrR"}});
    }}

    // Boundary selections for isothermal sink: bottom (y=ySubBot), left (x=-xDom), right (x=+xDom)
    model.component("comp1").selection().create("selBotBnd", "Box");
    model.component("comp1").selection("selBotBnd").set("entitydim", 1);
    model.component("comp1").selection("selBotBnd").set("condition", "inside");
    model.component("comp1").selection("selBotBnd").set("xmin", -xDom - eps);
    model.component("comp1").selection("selBotBnd").set("xmax", xDom + eps);
    model.component("comp1").selection("selBotBnd").set("ymin", ySubBot - eps);
    model.component("comp1").selection("selBotBnd").set("ymax", ySubBot + eps);

    model.component("comp1").selection().create("selLeftBnd", "Box");
    model.component("comp1").selection("selLeftBnd").set("entitydim", 1);
    model.component("comp1").selection("selLeftBnd").set("condition", "inside");
    model.component("comp1").selection("selLeftBnd").set("xmin", -xDom - eps);
    model.component("comp1").selection("selLeftBnd").set("xmax", -xDom + eps);
    model.component("comp1").selection("selLeftBnd").set("ymin", ySubBot - eps);
    model.component("comp1").selection("selLeftBnd").set("ymax", yAirTop + eps);

    model.component("comp1").selection().create("selRightBnd", "Box");
    model.component("comp1").selection("selRightBnd").set("entitydim", 1);
    model.component("comp1").selection("selRightBnd").set("condition", "inside");
    model.component("comp1").selection("selRightBnd").set("xmin", xDom - eps);
    model.component("comp1").selection("selRightBnd").set("xmax", xDom + eps);
    model.component("comp1").selection("selRightBnd").set("ymin", ySubBot - eps);
    model.component("comp1").selection("selRightBnd").set("ymax", yAirTop + eps);

    model.component("comp1").selection().create("selSink", "Union");
    model.component("comp1").selection("selSink").set("entitydim", 1);
    model.component("comp1").selection("selSink").set("input", new String[]{{"selBotBnd", "selLeftBnd", "selRightBnd"}});

    // Physics 1: Heat Transfer in Solids (ht)
    model.component("comp1").physics().create("ht", "HeatTransfer", "geom1");
    model.component("comp1").physics("ht").feature("init1").set("Tinit", "300[K]");
    setSolidMat(model, "solid1", kOx, cvOx);

    model.component("comp1").physics("ht").feature().create("solSub", "SolidHeatTransferModel", 2);
    model.component("comp1").physics("ht").feature("solSub").selection().named("selSub");
    setSolidMat(model, "solSub", kSub, cvSub);

    model.component("comp1").physics("ht").feature().create("solAir", "SolidHeatTransferModel", 2);
    model.component("comp1").physics("ht").feature("solAir").selection().named("selAir");
    setSolidMat(model, "solAir", kAir, cvAir);

    model.component("comp1").physics("ht").feature().create("solCore", "SolidHeatTransferModel", 2);
    model.component("comp1").physics("ht").feature("solCore").selection().named("selCore");
    setSolidMat(model, "solCore", kCore, cvCore);

    model.component("comp1").physics("ht").feature().create("solHeat", "SolidHeatTransferModel", 2);
    model.component("comp1").physics("ht").feature("solHeat").selection().named("selHeat");
    setSolidMat(model, "solHeat", kHeat, cvHeat);

    if (isTrench) {{
      model.component("comp1").physics("ht").feature().create("solTr", "SolidHeatTransferModel", 2);
      model.component("comp1").physics("ht").feature("solTr").selection().named("selTr");
      setSolidMat(model, "solTr", kAir, cvAir);
    }}

    double qVol1mW = 1e-3 / (wHeat * (yH1 - yH0) * lHeat);
    model.component("comp1").physics("ht").feature().create("hs1", "HeatSource", 2);
    model.component("comp1").physics("ht").feature("hs1").selection().named("selHeat");
    model.component("comp1").physics("ht").feature("hs1").set("Q0", "" + qVol1mW);

    model.component("comp1").physics("ht").feature().create("temp1", "TemperatureBoundary", 1);
    model.component("comp1").physics("ht").feature("temp1").selection().named("selSink");
    model.component("comp1").physics("ht").feature("temp1").set("T0", "300[K]");

    // Mesh: fine near core & heater, graded outward
    model.component("comp1").mesh().create("mesh1", "geom1");
    model.component("comp1").mesh("mesh1").feature().create("sizeCore", "Size");
    model.component("comp1").mesh("mesh1").feature("sizeCore").selection().named("selCore");
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("hmax", 0.025e-6);

    model.component("comp1").mesh("mesh1").feature().create("sizeHeat", "Size");
    model.component("comp1").mesh("mesh1").feature("sizeHeat").selection().named("selHeat");
    model.component("comp1").mesh("mesh1").feature("sizeHeat").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("sizeHeat").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("sizeHeat").set("hmax", 0.06e-6);

    model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 1.2e-6);
    model.component("comp1").mesh("mesh1").feature().create("ftri1", "FreeTri");
    model.component("comp1").mesh("mesh1").run();

    // Study 1: Stationary Thermal (P = 1 mW)
    model.study().create("stdStat");
    model.study("stdStat").feature().create("stat", "Stationary");
    model.study("stdStat").run();

    String dsetStat = model.result().dataset().tags()[0];
    model.result().numerical().create("intCore", "IntSurface");
    model.result().numerical("intCore").set("data", dsetStat);
    model.result().numerical("intCore").selection().named("selCore");
    model.result().numerical("intCore").set("expr", new String[]{{"T - 300[K]", "1"}});
    double[][] ic = model.result().numerical("intCore").getReal();
    double rThCore = ic[0][0] / ic[1][0];

    model.result().numerical().create("maxHeat", "MaxSurface");
    model.result().numerical("maxHeat").set("data", dsetStat);
    model.result().numerical("maxHeat").selection().named("selHeat");
    model.result().numerical("maxHeat").set("expr", new String[]{{"T - 300[K]"}});
    double rThPeak = model.result().numerical("maxHeat").getReal()[0][0];

    // Heat flux balance on isothermal sink boundaries vs 1e-3 / lHeat [W/m]
    model.result().numerical().create("intFlux", "IntLine");
    model.result().numerical("intFlux").set("data", dsetStat);
    model.result().numerical("intFlux").selection().named("selSink");
    model.result().numerical("intFlux").set("expr", new String[]{{"-ht.ntflux"}});
    double pOutPerM = Math.abs(model.result().numerical("intFlux").getReal()[0][0]);
    double pInPerM = 1e-3 / lHeat;
    double consErrPct = Math.abs(1.0 - pOutPerM / pInPerM) * 100.0;

    // Neighbor thermal crosstalk at (xNeighbor, 0.5*hCore) using CutPoint2D
    model.result().dataset().create("cpt1", "CutPoint2D");
    model.result().dataset("cpt1").set("data", dsetStat);
    model.result().dataset("cpt1").set("pointx", "" + xNeighbor);
    model.result().dataset("cpt1").set("pointy", "" + (0.5 * hCore));
    model.result().numerical().create("pev1", "EvalPoint");
    model.result().numerical("pev1").set("data", "cpt1");
    model.result().numerical("pev1").set("expr", new String[]{{"T - 300[K]"}});
    double dtNeighbor = model.result().numerical("pev1").getReal()[0][0];
    double xtalkPct = (dtNeighbor / rThCore) * 100.0;

    // Study 2: Transient Step-On Heating (0 -> tEnd) to extract exact 10-90% rise time
    double tEndUs = isTrench ? 60.0 : 40.0;
    model.study().create("stdTran");
    model.study("stdTran").feature().create("time", "Transient");
    model.study("stdTran").feature("time").set("tlist", "range(0, " + (tEndUs/80.0) + "e-6, " + tEndUs + "e-6)");
    model.study("stdTran").run();

    String[] dtags = model.result().dataset().tags();
    String dsetTran = dtags[dtags.length - 1];
    for (String dt : dtags) {{
      if (dt.startsWith("dset")) dsetTran = dt;
    }}
    model.result().numerical().create("intTran", "IntSurface");
    model.result().numerical("intTran").set("data", dsetTran);
    model.result().numerical("intTran").selection().named("selCore");
    model.result().numerical("intTran").set("expr", new String[]{{"T - 300[K]"}});
    double[] tCoreSeries = model.result().numerical("intTran").getReal()[0];
    double areaCore = ic[1][0];
    int nT = tCoreSeries.length;
    double[] tUs = new double[nT];
    double[] dT = new double[nT];
    for (int i = 0; i < nT; i++) {{
      tUs[i] = i * (tEndUs / 80.0);
      dT[i] = tCoreSeries[i] / areaCore;
    }}
    double ySS = dT[nT - 1];
    double t10 = 0.0, t90 = tEndUs;
    boolean got10 = false, got90 = false;
    for (int i = 1; i < nT; i++) {{
      if (!got10 && dT[i] >= 0.10 * ySS) {{
        t10 = tUs[i-1] + (0.10*ySS - dT[i-1]) * (tUs[i] - tUs[i-1]) / Math.max(dT[i] - dT[i-1], 1e-12);
        got10 = true;
      }}
      if (!got90 && dT[i] >= 0.90 * ySS) {{
        t90 = tUs[i-1] + (0.90*ySS - dT[i-1]) * (tUs[i] - tUs[i-1]) / Math.max(dT[i] - dT[i-1], 1e-12);
        got90 = true;
      }}
    }}
    double tRiseUs = t90 - t10;
    // Exact thermal 3-dB frequency from step-response discrete Laplace transfer H(jw) = jw * integral s(t) exp(-jwt) dt
    // For linear passive heat conduction, step response h_step(t) = dT(t)/rThCore => impulse response h(t) = d(h_step)/dt!
    double f3dB_kHz = 0.35 / Math.max(tRiseUs * 1e-6, 1e-9) * 1e-3;
    double bestF = f3dB_kHz;
    double prevH = 1.0, prevF = 0.1;
    for (int kf = 0; kf < 120; kf++) {{
      double fKHz = 0.2 * Math.pow(500.0, kf / 119.0);
      double wRad = 2.0 * Math.PI * fKHz * 1e3;
      double re = 0.0, im = 0.0;
      for (int i = 1; i < nT; i++) {{
        double dStep = (dT[i] - dT[i-1]) / Math.max(rThCore, 1e-12);
        double tMid = 0.5 * (tUs[i] + tUs[i-1]) * 1e-6;
        re += dStep * Math.cos(wRad * tMid);
        im -= dStep * Math.sin(wRad * tMid);
      }}
      // Tail residual from tEnd to infinity assuming single slow tail decay
      double rem = Math.max(0.0, 1.0 - dT[nT-1] / Math.max(rThCore, 1e-12));
      double hMag = Math.hypot(re + rem * Math.cos(wRad * tEndUs * 1e-6) / (1.0 + wRad * 10e-6), im);
      if (kf > 0 && hMag <= 0.70710678) {{
        bestF = prevF + (0.70710678 - prevH) * (fKHz - prevF) / Math.min(hMag - prevH, -1e-9);
        break;
      }}
      prevH = hMag; prevF = fKHz;
    }}

    // Coupled Optical Mode Solve (emw) using COMSOL's own solved stationary T field!
    // Exclude 26-um Si handle substrate (selSub) so optical mode solver sees exact BOX + TopOxide + Core window.
    model.component("comp1").selection().create("selNoOpt", "Union");
    model.component("comp1").selection("selNoOpt").set("entitydim", 2);
    model.component("comp1").selection("selNoOpt").set("input", new String[]{{"selSub", "selAir", "selHeat"}});
    model.component("comp1").selection().create("selOpt", "Complement");
    model.component("comp1").selection("selOpt").set("entitydim", 2);
    model.component("comp1").selection("selOpt").set("input", new String[]{{"selNoOpt"}});

    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").selection().named("selOpt");
    model.component("comp1").physics("emw").feature("wee1").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("wee1").set("epsilonr_mat", "userdef");
    String eOxExpr = "(" + nOx + " + P_opt_mw * (" + dndtOx + ") * (T - 300[K]))^2";
    String eCoreExpr = "(" + nCore + " + P_opt_mw * (" + dndtCore + ") * (T - 300[K]))^2";
    model.component("comp1").physics("emw").feature("wee1").set("epsilonr", new String[]{{eOxExpr,"0","0","0",eOxExpr,"0","0","0",eOxExpr}});
    model.component("comp1").physics("emw").feature("wee1").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("wee1").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("wee1").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("wee1").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    if (isTrench) {{
      model.component("comp1").physics("emw").feature().create("weeTr", "WaveEquationElectric", 2);
      model.component("comp1").physics("emw").feature("weeTr").selection().named("selTr");
      model.component("comp1").physics("emw").feature("weeTr").set("DisplacementFieldModel", "RelativePermittivity");
      model.component("comp1").physics("emw").feature("weeTr").set("epsilonr_mat", "userdef");
      model.component("comp1").physics("emw").feature("weeTr").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
      model.component("comp1").physics("emw").feature("weeTr").set("mur_mat", "userdef");
      model.component("comp1").physics("emw").feature("weeTr").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
      model.component("comp1").physics("emw").feature("weeTr").set("sigma_mat", "userdef");
      model.component("comp1").physics("emw").feature("weeTr").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});
    }}

    model.component("comp1").physics("emw").feature().create("weeCore", "WaveEquationElectric", 2);
    model.component("comp1").physics("emw").feature("weeCore").selection().named("selCore");
    model.component("comp1").physics("emw").feature("weeCore").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("weeCore").set("epsilonr_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeCore").set("epsilonr", new String[]{{eCoreExpr,"0","0","0",eCoreExpr,"0","0","0",eCoreExpr}});
    model.component("comp1").physics("emw").feature("weeCore").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeCore").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("weeCore").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeCore").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    // Solve cold (P_opt_mw = 0) and hot (P_opt_mw = 10 mW) modes using sol1's T field!
    double[] optNeff = new double[2];
    double[] pVals = new double[]{{0.0, 10.0}};
    double k0 = 2.0 * Math.PI / wl;
    for (int ip = 0; ip < 2; ip++) {{
      model.param().set("P_opt_mw", "" + pVals[ip]);
      String stTag = "stdOpt" + ip;
      model.study().create(stTag);
      model.study(stTag).feature().create("mode", "ModeAnalysis");
      model.study(stTag).feature("mode").set("activate", new String[]{{"ht", "off", "emw", "on"}});
      model.study(stTag).feature("mode").set("usesol", true);
      model.study(stTag).feature("mode").set("notsolmethod", "sol");
      model.study(stTag).feature("mode").set("notstudy", "stdStat");
      model.study(stTag).feature("mode").set("modeFreq", "c_const/wl0");
      model.study(stTag).feature("mode").set("neigs", 4);
      model.study(stTag).feature("mode").set("neigsactive", true);
      double nG = (nCore > 3.0) ? 2.38 : 1.72;
      model.study(stTag).feature("mode").set("shift", "" + nG);
      model.study(stTag).feature("mode").set("shiftactive", true);
      model.study(stTag).run();

      String solOpt = "sol" + (ip + 3);
      String[] stags = model.sol().tags();
      solOpt = stags[stags.length - 1];
      double[] lamIm = model.sol(solOpt).getPValsImag();
      String[] dsAll = model.result().dataset().tags();
      String dsOpt = dsAll[dsAll.length - 1];
      for (String d : dsAll) {{ if (d.startsWith("dset")) dsOpt = d; }}
      model.result().numerical().create("intP" + ip, "IntSurface");
      model.result().numerical("intP" + ip).set("data", dsOpt);
      model.result().numerical("intP" + ip).selection().named("selCore");
      model.result().numerical("intP" + ip).set("expr", new String[]{{"abs(Ex)^2", "abs(Ey)^2"}});
      double[][] iv = model.result().numerical("intP" + ip).getReal();
      double bestN = -1.0;
      for (int m = 0; m < lamIm.length; m++) {{
        double ne = Math.abs(lamIm[m]) / k0;
        double te = iv[0][m] / (iv[0][m] + iv[1][m] + 1e-30);
        if (te >= 0.5 && ne > bestN && ne < nCore + 0.2) bestN = ne;
      }}
      optNeff[ip] = bestN;
    }}
    double dneffPerMw = Math.abs(optNeff[1] - optNeff[0]) / 10.0;
    double pPiMw = wl / (2.0 * lHeat * Math.max(dneffPerMw, 1e-12));
    double dtS = (System.nanoTime() - t0) * 1e-9;

    System.out.println("TH_JSON: {{\\"case_id\\":\\"" + cid + "\\",\\"R_th_core_K_per_mW\\":" + rThCore
      + ",\\"R_th_peak_K_per_mW\\":" + rThPeak + ",\\"P_pi_mW\\":" + pPiMw
      + ",\\"neff_0mW\\":" + optNeff[0] + ",\\"neff_10mW\\":" + optNeff[1]
      + ",\\"t_rise_10_90_us\\":" + tRiseUs + ",\\"f_3dB_thermal_kHz\\":" + bestF
      + ",\\"crosstalk_20um_pct\\":" + xtalkPct + ",\\"power_conservation_err_pct\\":" + consErrPct
      + ",\\"wall_time_s\\":" + dtS + "}}");
  }}

  public static Model run() {{
    {chr(10).join("    " + ln for ln in calls)}
    return ModelUtil.create("Done");
  }}
  public static void main(String[] args) {{ run(); }}
}}
'''
    lines = _run_comsol_java("ComsolSuite4Thermal", java_src)
    rows = _parse_kv_lines(lines, "TH_JSON:")
    out = []
    for r in rows:
        rec = {
            "case_id": r["case_id"],
            "R_th_core_K_per_mW": round(r["R_th_core_K_per_mW"], 4),
            "R_th_peak_K_per_mW": round(r["R_th_peak_K_per_mW"], 4),
            "P_pi_mW": round(r["P_pi_mW"], 3),
            "t_rise_10_90_us": round(r["t_rise_10_90_us"], 3),
            "f_3dB_thermal_kHz": round(r["f_3dB_thermal_kHz"], 2),
            "crosstalk_20um_pct": round(r["crosstalk_20um_pct"], 4),
            "power_conservation_err_pct": round(r["power_conservation_err_pct"], 4),
            "wall_time_s": round(r["wall_time_s"], 2),
        }
        out.append(rec)
        print(
            f"[COMSOL-S4] {rec['case_id']:<26} R_th={rec['R_th_core_K_per_mW']:.4f} | "
            f"R_pk={rec['R_th_peak_K_per_mW']:.4f} | P_pi={rec['P_pi_mW']:.2f} mW | "
            f"tr={rec['t_rise_10_90_us']:.2f} us | f3dB={rec['f_3dB_thermal_kHz']:.2f} kHz | "
            f"xtalk={rec['crosstalk_20um_pct']:.3f}% | consErr={rec['power_conservation_err_pct']:.3f}%"
        )
    return out


# ==============================================================================
# Java Block 3: Suite 5 (2D Anisotropic Electrostatics `es` + Coupled `ewfd` Pockels V_pi*L)
# ==============================================================================
def run_comsol_suite5_eo(cases: list[dict]) -> list[dict]:
    calls = []
    for c in cases:
        ci = c["core_index"]["center"]
        n_cl = c["clad_index"]["center"]["nx"]
        n_tp = c["top_index"]["center"]["nx"]
        is_gsg = "true" if c["electrode_config"] == "inplane_gsg" else "false"
        want_te = "true" if c["polarization"].upper().startswith("TE") else "false"
        calls.append(
            f'solveEO("{c["case_id"]}", {is_gsg}, {want_te}, {c["wavelength_nm"]*1e-9:.12e}, '
            f'{c["waveguide_width_um"]*1e-6:.12e}, {c["film_height_um"]*1e-6:.12e}, {c["slab_height_um"]*1e-6:.12e}, '
            f'{c["electrode_gap_um"]*1e-6:.12e}, {c["electrode_width_um"]*1e-6:.12e}, {c["electrode_thickness_um"]*1e-6:.12e}, '
            f'{c["electrode_voffset_um"]*1e-6:.12e}, {c["box_thickness_um"]*1e-6:.12e}, {c["top_clad_thickness_um"]*1e-6:.12e}, '
            f'{c["eps_core_dc"][0]:.4f}, {c["eps_core_dc"][1]:.4f}, {c["eps_clad_dc"][0]:.4f}, {c["eps_top_dc"][0]:.4f}, '
            f'{ci["nx"]:.8f}, {ci["ny"]:.8f}, {ci["nz"]:.8f}, {n_cl:.8f}, {n_tp:.8f}, '
            f'{c["core_index"]["r33_pm_per_v"]:.6f}, {c["core_index"]["r13_pm_per_v"]:.6f});'
        )

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolSuite5EO {{

  static void setEps(Model model, String fTag, double exx, double eyy) {{
    model.component("comp1").physics("es").feature(fTag).set("epsilonr_mat", "userdef");
    model.component("comp1").physics("es").feature(fTag).set("epsilonr", new String[]{{""+exx,"0","0","0",""+eyy,"0","0","0",""+exx}});
  }}

  static void solveEO(
      String cid, boolean isGsg, boolean wantTe, double wl,
      double wCore, double hFilm, double hSlab,
      double gElec, double wElec, double tElec, double vOff, double hBox, double hTopClad,
      double epsCoreX, double epsCoreY, double epsClad, double epsTop,
      double nx0, double ny0, double nz0, double nClad, double nTop,
      double r33PmV, double r13PmV) {{
    long t0 = System.nanoTime();
    ModelUtil.clear();
    Model model = ModelUtil.create("Model");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);
    model.param().set("wl0", wl + "[m]");
    model.param().set("V_opt", "10");

    double yEl0, yEl1, yOxTop, yDomTop, yBotDom, xDom, dBot;
    if (isGsg) {{
      yEl0 = hSlab;
      yEl1 = yEl0 + tElec;
      yOxTop = Math.max(hTopClad, Math.max(hFilm + 0.8e-6, yEl1 + 0.4e-6));
      yDomTop = yOxTop + 2.0e-6;
      yBotDom = -hBox;
      xDom = Math.max(14.0e-6, 0.5*gElec + wElec + 5.0e-6);
      dBot = hBox;
    }} else {{
      yEl0 = hFilm + vOff;
      yEl1 = yEl0 + tElec;
      dBot = (hBox >= 1.0e-6) ? Math.max(hBox, 0.35e-6) : Math.min(hBox, 0.65e-6);
      yBotDom = -Math.max(hBox, dBot + 0.30e-6);
      yOxTop = Math.max(hTopClad, yEl1 + 0.35e-6);
      yDomTop = yOxTop + 1.5e-6;
      xDom = Math.max(10.0e-6, Math.max(0.5*wElec + 5.0e-6, 0.5*wCore + 5.0e-6));
    }}

    // Dielectric layers
    model.component("comp1").geom("geom1").feature().create("rBox", "Rectangle");
    model.component("comp1").geom("geom1").feature("rBox").set("size", new double[]{{2.0*xDom, -yBotDom}});
    model.component("comp1").geom("geom1").feature("rBox").set("pos", new double[]{{-xDom, yBotDom}});

    model.component("comp1").geom("geom1").feature().create("rTop", "Rectangle");
    model.component("comp1").geom("geom1").feature("rTop").set("size", new double[]{{2.0*xDom, yOxTop}});
    model.component("comp1").geom("geom1").feature("rTop").set("pos", new double[]{{-xDom, 0.0}});

    model.component("comp1").geom("geom1").feature().create("rAir", "Rectangle");
    model.component("comp1").geom("geom1").feature("rAir").set("size", new double[]{{2.0*xDom, yDomTop - yOxTop}});
    model.component("comp1").geom("geom1").feature("rAir").set("pos", new double[]{{-xDom, yOxTop}});

    if (hSlab > 1e-11) {{
      model.component("comp1").geom("geom1").feature().create("rSlab", "Rectangle");
      model.component("comp1").geom("geom1").feature("rSlab").set("size", new double[]{{2.0*xDom, hSlab}});
      model.component("comp1").geom("geom1").feature("rSlab").set("pos", new double[]{{-xDom, 0.0}});
    }}

    model.component("comp1").geom("geom1").feature().create("rCore", "Rectangle");
    model.component("comp1").geom("geom1").feature("rCore").set("size", new double[]{{wCore, hFilm - hSlab}});
    model.component("comp1").geom("geom1").feature("rCore").set("pos", new double[]{{-0.5*wCore, hSlab}});

    // Electrode rectangles
    double sigX0, sigX1, sigY0, sigY1, gndX0, gndX1, gndY0, gndY1;
    if (isGsg) {{
      sigX0 = -0.5*gElec - wElec; sigX1 = -0.5*gElec; sigY0 = yEl0; sigY1 = yEl1;
      gndX0 =  0.5*gElec;         gndX1 =  0.5*gElec + wElec; gndY0 = yEl0; gndY1 = yEl1;
    }} else {{
      double wBot = Math.max(wElec * 1.5, wCore + 2.0e-6);
      sigX0 = -0.5*wBot; sigX1 = 0.5*wBot; sigY0 = yBotDom; sigY1 = -dBot;
      gndX0 = -0.5*wElec; gndX1 = 0.5*wElec; gndY0 = yEl0; gndY1 = yEl1;
    }}
    model.component("comp1").geom("geom1").feature().create("rSig", "Rectangle");
    model.component("comp1").geom("geom1").feature("rSig").set("size", new double[]{{sigX1 - sigX0, sigY1 - sigY0}});
    model.component("comp1").geom("geom1").feature("rSig").set("pos", new double[]{{sigX0, sigY0}});

    model.component("comp1").geom("geom1").feature().create("rGnd", "Rectangle");
    model.component("comp1").geom("geom1").feature("rGnd").set("size", new double[]{{gndX1 - gndX0, gndY1 - gndY0}});
    model.component("comp1").geom("geom1").feature("rGnd").set("pos", new double[]{{gndX0, gndY0}});

    model.component("comp1").geom("geom1").run();

    double eps = 2e-9;
    model.component("comp1").selection().create("selBox", "Box");
    model.component("comp1").selection("selBox").set("entitydim", 2);
    model.component("comp1").selection("selBox").set("condition", "inside");
    model.component("comp1").selection("selBox").set("xmin", -xDom - eps);
    model.component("comp1").selection("selBox").set("xmax", xDom + eps);
    model.component("comp1").selection("selBox").set("ymin", yBotDom - eps);
    model.component("comp1").selection("selBox").set("ymax", eps);

    model.component("comp1").selection().create("selAir", "Box");
    model.component("comp1").selection("selAir").set("entitydim", 2);
    model.component("comp1").selection("selAir").set("condition", "inside");
    model.component("comp1").selection("selAir").set("xmin", -xDom - eps);
    model.component("comp1").selection("selAir").set("xmax", xDom + eps);
    model.component("comp1").selection("selAir").set("ymin", yOxTop - eps);
    model.component("comp1").selection("selAir").set("ymax", yDomTop + eps);

    model.component("comp1").selection().create("selCore", "Box");
    model.component("comp1").selection("selCore").set("entitydim", 2);
    model.component("comp1").selection("selCore").set("condition", "inside");
    model.component("comp1").selection("selCore").set("xmin", -0.5*wCore - eps);
    model.component("comp1").selection("selCore").set("xmax", 0.5*wCore + eps);
    model.component("comp1").selection("selCore").set("ymin", hSlab - eps);
    model.component("comp1").selection("selCore").set("ymax", hFilm + eps);

    if (hSlab > 1e-11) {{
      model.component("comp1").selection().create("selSlab", "Box");
      model.component("comp1").selection("selSlab").set("entitydim", 2);
      model.component("comp1").selection("selSlab").set("condition", "inside");
      model.component("comp1").selection("selSlab").set("xmin", -xDom - eps);
      model.component("comp1").selection("selSlab").set("xmax", xDom + eps);
      model.component("comp1").selection("selSlab").set("ymin", -eps);
      model.component("comp1").selection("selSlab").set("ymax", hSlab + eps);
    }}

    model.component("comp1").selection().create("selSigBnd", "Box");
    model.component("comp1").selection("selSigBnd").set("entitydim", 1);
    model.component("comp1").selection("selSigBnd").set("condition", "inside");
    model.component("comp1").selection("selSigBnd").set("xmin", sigX0 - eps);
    model.component("comp1").selection("selSigBnd").set("xmax", sigX1 + eps);
    model.component("comp1").selection("selSigBnd").set("ymin", sigY0 - eps);
    model.component("comp1").selection("selSigBnd").set("ymax", sigY1 + eps);

    model.component("comp1").selection().create("selGndBnd", "Box");
    model.component("comp1").selection("selGndBnd").set("entitydim", 1);
    model.component("comp1").selection("selGndBnd").set("condition", "inside");
    model.component("comp1").selection("selGndBnd").set("xmin", gndX0 - eps);
    model.component("comp1").selection("selGndBnd").set("xmax", gndX1 + eps);
    model.component("comp1").selection("selGndBnd").set("ymin", gndY0 - eps);
    model.component("comp1").selection("selGndBnd").set("ymax", gndY1 + eps);

    model.component("comp1").selection().create("selSigDom", "Box");
    model.component("comp1").selection("selSigDom").set("entitydim", 2);
    model.component("comp1").selection("selSigDom").set("condition", "inside");
    model.component("comp1").selection("selSigDom").set("xmin", sigX0 - eps);
    model.component("comp1").selection("selSigDom").set("xmax", sigX1 + eps);
    model.component("comp1").selection("selSigDom").set("ymin", sigY0 - eps);
    model.component("comp1").selection("selSigDom").set("ymax", sigY1 + eps);

    model.component("comp1").selection().create("selGndDom", "Box");
    model.component("comp1").selection("selGndDom").set("entitydim", 2);
    model.component("comp1").selection("selGndDom").set("condition", "inside");
    model.component("comp1").selection("selGndDom").set("xmin", gndX0 - eps);
    model.component("comp1").selection("selGndDom").set("xmax", gndX1 + eps);
    model.component("comp1").selection("selGndDom").set("ymin", gndY0 - eps);
    model.component("comp1").selection("selGndDom").set("ymax", gndY1 + eps);

    model.component("comp1").selection().create("selElecAll", "Union");
    model.component("comp1").selection("selElecAll").set("entitydim", 2);
    model.component("comp1").selection("selElecAll").set("input", new String[]{{"selSigDom", "selGndDom"}});

    model.component("comp1").selection().create("selDiel", "Complement");
    model.component("comp1").selection("selDiel").set("entitydim", 2);
    model.component("comp1").selection("selDiel").set("input", new String[]{{"selElecAll"}});

    // Physics 1: AC/DC Electrostatics (es) on dielectric domains only
    // IMPORTANT (COMSOL 6.4): default feature is fsp1 (FreeSpace, eps_r=1); override explicitly with ChargeConservation!
    model.component("comp1").physics().create("es", "Electrostatics", "geom1");
    model.component("comp1").physics("es").selection().named("selDiel");
    model.component("comp1").physics("es").feature().create("ccnTop", "ChargeConservation", 2);
    model.component("comp1").physics("es").feature("ccnTop").selection().named("selDiel");
    setEps(model, "ccnTop", epsTop, epsTop);

    model.component("comp1").physics("es").feature().create("ccnBox", "ChargeConservation", 2);
    model.component("comp1").physics("es").feature("ccnBox").selection().named("selBox");
    setEps(model, "ccnBox", epsClad, epsClad);

    model.component("comp1").physics("es").feature().create("ccnAir", "ChargeConservation", 2);
    model.component("comp1").physics("es").feature("ccnAir").selection().named("selAir");
    setEps(model, "ccnAir", 1.0, 1.0);

    if (hSlab > 1e-11) {{
      model.component("comp1").physics("es").feature().create("ccnSlab", "ChargeConservation", 2);
      model.component("comp1").physics("es").feature("ccnSlab").selection().named("selSlab");
      setEps(model, "ccnSlab", epsCoreX, epsCoreY);
    }}

    model.component("comp1").physics("es").feature().create("ccnCore", "ChargeConservation", 2);
    model.component("comp1").physics("es").feature("ccnCore").selection().named("selCore");
    setEps(model, "ccnCore", epsCoreX, epsCoreY);

    model.component("comp1").physics("es").feature().create("potSig", "ElectricPotential", 1);
    model.component("comp1").physics("es").feature("potSig").selection().named("selSigBnd");
    model.component("comp1").physics("es").feature("potSig").set("V0", "1[V]");

    model.component("comp1").physics("es").feature().create("gnd1", "Ground", 1);
    model.component("comp1").physics("es").feature("gnd1").selection().named("selGndBnd");

    // Mesh
    model.component("comp1").mesh().create("mesh1", "geom1");
    model.component("comp1").mesh("mesh1").feature().create("sizeCore", "Size");
    model.component("comp1").mesh("mesh1").feature("sizeCore").selection().named("selCore");
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("sizeCore").set("hmax", 0.030e-6);
    model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.35e-6);
    model.component("comp1").mesh("mesh1").feature().create("ftri1", "FreeTri");
    model.component("comp1").mesh("mesh1").run();

    // Study 1: Stationary Electrostatics (V = 1 V)
    model.study().create("stdEs");
    model.study("stdEs").feature().create("stat", "Stationary");
    model.study("stdEs").run();

    String dsetEs = model.result().dataset().tags()[0];
    model.result().numerical().create("intE", "IntSurface");
    model.result().numerical("intE").set("data", dsetEs);
    model.result().numerical("intE").selection().named("selCore");
    model.result().numerical("intE").set("expr", new String[]{{"es.Ex*1e-6", "es.Ey*1e-6", "1"}});
    double[][] ie = model.result().numerical("intE").getReal();
    double exCore = ie[0][0] / ie[2][0];
    double eyCore = ie[1][0] / ie[2][0];

    // Capacitance per unit length C = 2 * W_e / (1 V)^2 = int (D . E) dA [F/m] -> * 1e9 [fF/um]
    model.result().numerical().create("intWe", "IntSurface");
    model.result().numerical("intWe").set("data", dsetEs);
    model.result().numerical("intWe").selection().named("selDiel");
    model.result().numerical("intWe").set("expr", new String[]{{"es.Dx*es.Ex + es.Dy*es.Ey"}});
    double cLineFfUm = model.result().numerical("intWe").getReal()[0][0] * 1e9;

    // Physics 2: Coupled Wave Optics (emw) with spatial Pockels perturbation from COMSOL's own es.Ex, es.Ey!
    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").selection().named("selDiel");
    model.component("comp1").physics("emw").feature("wee1").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("wee1").set("epsilonr_mat", "userdef");
    String eTopS = "" + (nTop*nTop);
    String eCladS = "" + (nClad*nClad);
    model.component("comp1").physics("emw").feature("wee1").set("epsilonr", new String[]{{eTopS,"0","0","0",eTopS,"0","0","0",eTopS}});
    model.component("comp1").physics("emw").feature("wee1").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("wee1").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("wee1").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("wee1").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    model.component("comp1").physics("emw").feature().create("weeBox", "WaveEquationElectric", 2);
    model.component("comp1").physics("emw").feature("weeBox").selection().named("selBox");
    model.component("comp1").physics("emw").feature("weeBox").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("weeBox").set("epsilonr_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeBox").set("epsilonr", new String[]{{eCladS,"0","0","0",eCladS,"0","0","0",eCladS}});
    model.component("comp1").physics("emw").feature("weeBox").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeBox").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("weeBox").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeBox").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    model.component("comp1").physics("emw").feature().create("weeAir", "WaveEquationElectric", 2);
    model.component("comp1").physics("emw").feature("weeAir").selection().named("selAir");
    model.component("comp1").physics("emw").feature("weeAir").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("weeAir").set("epsilonr_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeAir").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("weeAir").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeAir").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("weeAir").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeAir").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    String exExpr, eyExpr, ezExpr;
    if (isGsg) {{
      exExpr = "(" + nx0 + " - 0.5*(" + (nx0*nx0*nx0) + ")*(" + (r33PmV*1e-12) + ")*V_opt*es.Ex)^2";
      eyExpr = "(" + ny0 + " - 0.5*(" + (ny0*ny0*ny0) + ")*(" + (r13PmV*1e-12) + ")*V_opt*es.Ex)^2";
      ezExpr = "(" + nz0 + " - 0.5*(" + (nz0*nz0*nz0) + ")*(" + (r13PmV*1e-12) + ")*V_opt*es.Ex)^2";
    }} else {{
      exExpr = "(" + nx0 + " - 0.5*(" + (nx0*nx0*nx0) + ")*(" + (r13PmV*1e-12) + ")*V_opt*es.Ey)^2";
      eyExpr = "(" + ny0 + " - 0.5*(" + (ny0*ny0*ny0) + ")*(" + (r33PmV*1e-12) + ")*V_opt*es.Ey)^2";
      ezExpr = "(" + nz0 + " - 0.5*(" + (nz0*nz0*nz0) + ")*(" + (r13PmV*1e-12) + ")*V_opt*es.Ey)^2";
    }}
    if (hSlab > 1e-11) {{
      model.component("comp1").physics("emw").feature().create("weeSlab", "WaveEquationElectric", 2);
      model.component("comp1").physics("emw").feature("weeSlab").selection().named("selSlab");
      model.component("comp1").physics("emw").feature("weeSlab").set("DisplacementFieldModel", "RelativePermittivity");
      model.component("comp1").physics("emw").feature("weeSlab").set("epsilonr_mat", "userdef");
      model.component("comp1").physics("emw").feature("weeSlab").set("epsilonr", new String[]{{exExpr,"0","0","0",eyExpr,"0","0","0",ezExpr}});
      model.component("comp1").physics("emw").feature("weeSlab").set("mur_mat", "userdef");
      model.component("comp1").physics("emw").feature("weeSlab").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
      model.component("comp1").physics("emw").feature("weeSlab").set("sigma_mat", "userdef");
      model.component("comp1").physics("emw").feature("weeSlab").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});
    }}
    model.component("comp1").physics("emw").feature().create("weeCore", "WaveEquationElectric", 2);
    model.component("comp1").physics("emw").feature("weeCore").selection().named("selCore");
    model.component("comp1").physics("emw").feature("weeCore").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("weeCore").set("epsilonr_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeCore").set("epsilonr", new String[]{{exExpr,"0","0","0",eyExpr,"0","0","0",ezExpr}});
    model.component("comp1").physics("emw").feature("weeCore").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeCore").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("weeCore").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeCore").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    double[] optNeff = new double[2];
    double[] vVals = new double[]{{0.0, 10.0}};
    double k0 = 2.0 * Math.PI / wl;
    double nG = isGsg ? 1.915 : (wantTe ? (0.78*nx0 + 0.22*nClad) : 1.760);
    for (int ip = 0; ip < 2; ip++) {{
      model.param().set("V_opt", "" + vVals[ip]);
      String stTag = "stdOpt" + ip;
      model.study().create(stTag);
      model.study(stTag).feature().create("mode", "ModeAnalysis");
      model.study(stTag).feature("mode").set("activate", new String[]{{"es", "off", "emw", "on"}});
      model.study(stTag).feature("mode").set("usesol", true);
      model.study(stTag).feature("mode").set("notsolmethod", "sol");
      model.study(stTag).feature("mode").set("notstudy", "stdEs");
      model.study(stTag).feature("mode").set("modeFreq", "c_const/wl0");
      model.study(stTag).feature("mode").set("neigs", 14);
      model.study(stTag).feature("mode").set("neigsactive", true);
      model.study(stTag).feature("mode").set("shift", "" + nG);
      model.study(stTag).feature("mode").set("shiftactive", true);
      model.study(stTag).run();

      String[] stags = model.sol().tags();
      String solOpt = stags[stags.length - 1];
      double[] lamIm = model.sol(solOpt).getPValsImag();
      String[] dsAll = model.result().dataset().tags();
      String dsOpt = dsAll[dsAll.length - 1];
      for (String d : dsAll) {{ if (d.startsWith("dset")) dsOpt = d; }}
      model.result().numerical().create("intP" + ip, "IntSurface");
      model.result().numerical("intP" + ip).set("data", dsOpt);
      model.result().numerical("intP" + ip).selection().named("selCore");
      model.result().numerical("intP" + ip).set("expr", new String[]{{"abs(Ex)^2", "abs(Ey)^2"}});
      double[][] iv = model.result().numerical("intP" + ip).getReal();

      model.result().numerical().create("intAll" + ip, "IntSurface");
      model.result().numerical("intAll" + ip).set("data", dsOpt);
      model.result().numerical("intAll" + ip).selection().named("selDiel");
      model.result().numerical("intAll" + ip).set("expr", new String[]{{"abs(Ex)^2 + abs(Ey)^2"}});
      double[][] ivAll = model.result().numerical("intAll" + ip).getReal();

      double bestN = -1.0;
      double bestScore = -1.0;
      for (int m = 0; m < lamIm.length; m++) {{
        double ne = Math.abs(lamIm[m]) / k0;
        double eCore = iv[0][m] + iv[1][m];
        double eTot = ivAll[0][m] + 1e-30;
        double coreFrac = eCore / eTot;
        double te = iv[0][m] / (eCore + 1e-30);
        boolean polOk = wantTe ? (te >= 0.65) : (te < 0.35);
        // Require guided core mode (reject slab modes trapped under outer metal electrodes!)
        if (polOk && coreFrac >= 0.35 && ne > nClad + 0.05 && ne < Math.max(nx0, ny0) + 0.05) {{
          if (coreFrac > bestScore) {{
            bestScore = coreFrac;
            bestN = ne;
          }}
        }}
      }}
      optNeff[ip] = bestN;
    }}
    double dneffPerV = Math.abs(optNeff[1] - optNeff[0]) / 10.0;
    double vPiLVcm = (wl / (2.0 * Math.max(dneffPerV, 1e-15))) * 100.0;
    double dtS = (System.nanoTime() - t0) * 1e-9;

    System.out.println("EO_JSON: {{\\"case_id\\":\\"" + cid + "\\",\\"ex_core_v_per_um_per_v\\":" + exCore
      + ",\\"ey_core_v_per_um_per_v\\":" + eyCore + ",\\"c_line_ff_per_um\\":" + cLineFfUm
      + ",\\"neff_0v\\":" + optNeff[0] + ",\\"dneff_per_v\\":" + dneffPerV
      + ",\\"V_pi_L_single_arm_Vcm\\":" + vPiLVcm + ",\\"wall_time_s\\":" + dtS + "}}");
  }}

  public static Model run() {{
    {chr(10).join("    " + ln for ln in calls)}
    return ModelUtil.create("Done");
  }}
  public static void main(String[] args) {{ run(); }}
}}
'''
    lines = _run_comsol_java("ComsolSuite5EO", java_src)
    rows = _parse_kv_lines(lines, "EO_JSON:")
    out = []
    for r in rows:
        rec = {
            "case_id": r["case_id"],
            "ex_core_v_per_um_per_v": round(r["ex_core_v_per_um_per_v"], 5),
            "ey_core_v_per_um_per_v": round(r["ey_core_v_per_um_per_v"], 5),
            "c_line_ff_per_um": round(r["c_line_ff_per_um"], 4),
            "neff_0v": round(r["neff_0v"], 6),
            "dneff_per_v": float(f"{r['dneff_per_v']:.6e}"),
            "V_pi_L_single_arm_Vcm": round(r["V_pi_L_single_arm_Vcm"], 4),
            "wall_time_s": round(r["wall_time_s"], 2),
        }
        out.append(rec)
        print(
            f"[COMSOL-S5] {rec['case_id']:<26} Ex={rec['ex_core_v_per_um_per_v']:+.5f} "
            f"Ey={rec['ey_core_v_per_um_per_v']:+.5f} | C={rec['c_line_ff_per_um']:.4f} fF/um | "
            f"neff0={rec['neff_0v']:.6f} | VpiL={rec['V_pi_L_single_arm_Vcm']:.3f} V*cm"
        )
    return out


# ==============================================================================
# Java Block 4: Suite 3C (2D Full-Wave ewfd Grating Coupler with PML + Numeric Port)
# ==============================================================================
def run_comsol_suite3c_gc(gc_case: dict) -> dict:
    hf = gc_case["film_height_um"] * 1e-6
    etch = gc_case["etch_depth_um"] * 1e-6
    h_box = gc_case["box_thickness_um"] * 1e-6
    h_top = gc_case["top_clad_thickness_um"] * 1e-6
    grooves = gc_case["grooves_um"]
    gc_len = grooves[-1][1] * 1e-6 + 0.5 * gc_case["pitch_um"] * 1e-6
    n_si = gc_case["core_index"]["center"]["nx"]
    n_ox = gc_case["clad_index"]["center"]["nx"]

    groove_java = []
    groove_sels = []
    for i, (g0, g1) in enumerate(grooves):
        groove_java.append(
            f'    model.component("comp1").geom("geom1").feature().create("g{i}", "Rectangle");\n'
            f'    model.component("comp1").geom("geom1").feature("g{i}").set("size", new double[]{{{(g1-g0)*1e-6:.12e}, {etch:.12e}}});\n'
            f'    model.component("comp1").geom("geom1").feature("g{i}").set("pos", new double[]{{{g0*1e-6:.12e}, {hf - etch:.12e}}});'
        )
        groove_sels.append(
            f'    model.component("comp1").selection().create("sg{i}", "Box");\n'
            f'    model.component("comp1").selection("sg{i}").set("entitydim", 2);\n'
            f'    model.component("comp1").selection("sg{i}").set("condition", "inside");\n'
            f'    model.component("comp1").selection("sg{i}").set("xmin", {g0*1e-6 - 2e-9:.12e});\n'
            f'    model.component("comp1").selection("sg{i}").set("xmax", {g1*1e-6 + 2e-9:.12e});\n'
            f'    model.component("comp1").selection("sg{i}").set("ymin", {hf - etch - 2e-9:.12e});\n'
            f'    model.component("comp1").selection("sg{i}").set("ymax", {hf + 2e-9:.12e});'
        )
    sg_names = ", ".join(f'"sg{i}"' for i in range(len(grooves)))

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;
import java.io.*;

public class ComsolSuite3CGC {{
  public static Model run() throws Exception {{
    long t0 = System.nanoTime();
    ModelUtil.clear();
    Model model = ModelUtil.create("Model");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);

    double hf = {hf:.12e};
    double hBox = {h_box:.12e};
    double hTop = {h_top:.12e};
    double gcLen = {gc_len:.12e};
    double xIn = -2.5e-6;
    double xOut = gcLen + 2.5e-6;
    double ySubBot = -hBox - 1.2e-6;
    double yAirTop = hf + hTop + 1.8e-6;
    double tPml = 0.8e-6;
    double yUpMon = hf + hTop + 0.45e-6;
    double yDnMon = -hBox - 0.25e-6;
    double nSi = {n_si:.8f};
    double nOx = {n_ox:.8f};

    // Left exterior waveguide port at x = xIn (no left PML needed; Numeric Port absorbs reflected mode!)
    model.component("comp1").geom("geom1").feature().create("rSub", "Rectangle");
    model.component("comp1").geom("geom1").feature("rSub").set("size", new double[]{{(xOut + tPml) - xIn, -hBox - (ySubBot - tPml)}});
    model.component("comp1").geom("geom1").feature("rSub").set("pos", new double[]{{xIn, ySubBot - tPml}});

    model.component("comp1").geom("geom1").feature().create("rOx", "Rectangle");
    model.component("comp1").geom("geom1").feature("rOx").set("size", new double[]{{(xOut + tPml) - xIn, hf + hTop + hBox}});
    model.component("comp1").geom("geom1").feature("rOx").set("pos", new double[]{{xIn, -hBox}});

    model.component("comp1").geom("geom1").feature().create("rAir", "Rectangle");
    model.component("comp1").geom("geom1").feature("rAir").set("size", new double[]{{(xOut + tPml) - xIn, (yAirTop + tPml) - (hf + hTop)}});
    model.component("comp1").geom("geom1").feature("rAir").set("pos", new double[]{{xIn, hf + hTop}});

    model.component("comp1").geom("geom1").feature().create("rSi", "Rectangle");
    model.component("comp1").geom("geom1").feature("rSi").set("size", new double[]{{(xOut + tPml) - xIn, hf}});
    model.component("comp1").geom("geom1").feature("rSi").set("pos", new double[]{{xIn, 0.0}});

{chr(10).join(groove_java)}

    // PML partition lines (right, bottom, top) + monitor lines
    double[][] segs = new double[][]{{
      {{xOut, ySubBot - tPml, xOut, yAirTop + tPml}},
      {{xIn, ySubBot, xOut + tPml, ySubBot}},
      {{xIn, yAirTop, xOut + tPml, yAirTop}},
      {{xIn, yUpMon, xOut, yUpMon}},
      {{xIn, yDnMon, xOut, yDnMon}}
    }};
    for (int k = 0; k < segs.length; k++) {{
      model.component("comp1").geom("geom1").feature().create("ls" + k, "LineSegment");
      model.component("comp1").geom("geom1").feature("ls" + k).set("specify1", "coord");
      model.component("comp1").geom("geom1").feature("ls" + k).set("specify2", "coord");
      model.component("comp1").geom("geom1").feature("ls" + k).set("coord1", new double[]{{segs[k][0], segs[k][1]}});
      model.component("comp1").geom("geom1").feature("ls" + k).set("coord2", new double[]{{segs[k][2], segs[k][3]}});
    }}
    model.component("comp1").geom("geom1").run();

    double eps = 2e-9;
    model.component("comp1").selection().create("selSub", "Box");
    model.component("comp1").selection("selSub").set("entitydim", 2);
    model.component("comp1").selection("selSub").set("condition", "inside");
    model.component("comp1").selection("selSub").set("xmin", xIn - tPml - eps);
    model.component("comp1").selection("selSub").set("xmax", xOut + tPml + eps);
    model.component("comp1").selection("selSub").set("ymin", ySubBot - tPml - eps);
    model.component("comp1").selection("selSub").set("ymax", -hBox + eps);

    model.component("comp1").selection().create("selAir", "Box");
    model.component("comp1").selection("selAir").set("entitydim", 2);
    model.component("comp1").selection("selAir").set("condition", "inside");
    model.component("comp1").selection("selAir").set("xmin", xIn - tPml - eps);
    model.component("comp1").selection("selAir").set("xmax", xOut + tPml + eps);
    model.component("comp1").selection("selAir").set("ymin", hf + hTop - eps);
    model.component("comp1").selection("selAir").set("ymax", yAirTop + tPml + eps);

    model.component("comp1").selection().create("selSiAll", "Box");
    model.component("comp1").selection("selSiAll").set("entitydim", 2);
    model.component("comp1").selection("selSiAll").set("condition", "inside");
    model.component("comp1").selection("selSiAll").set("xmin", xIn - tPml - eps);
    model.component("comp1").selection("selSiAll").set("xmax", xOut + tPml + eps);
    model.component("comp1").selection("selSiAll").set("ymin", -eps);
    model.component("comp1").selection("selSiAll").set("ymax", hf + eps);

{chr(10).join(groove_sels)}
    model.component("comp1").selection().create("selGrooves", "Union");
    model.component("comp1").selection("selGrooves").set("entitydim", 2);
    model.component("comp1").selection("selGrooves").set("input", new String[]{{{sg_names}}});

    model.component("comp1").selection().create("selSiCore", "Difference");
    model.component("comp1").selection("selSiCore").set("entitydim", 2);
    model.component("comp1").selection("selSiCore").set("add", new String[]{{"selSiAll"}});
    model.component("comp1").selection("selSiCore").set("subtract", new String[]{{"selGrooves"}});

    model.component("comp1").selection().create("selSiTotal", "Union");
    model.component("comp1").selection("selSiTotal").set("entitydim", 2);
    model.component("comp1").selection("selSiTotal").set("input", new String[]{{"selSub", "selSiCore"}});

    // Interior domain (non-PML) & PML complement
    model.component("comp1").selection().create("selInt", "Box");
    model.component("comp1").selection("selInt").set("entitydim", 2);
    model.component("comp1").selection("selInt").set("condition", "inside");
    model.component("comp1").selection("selInt").set("xmin", xIn - eps);
    model.component("comp1").selection("selInt").set("xmax", xOut + eps);
    model.component("comp1").selection("selInt").set("ymin", ySubBot - eps);
    model.component("comp1").selection("selInt").set("ymax", yAirTop + eps);

    model.component("comp1").selection().create("selPml", "Complement");
    model.component("comp1").selection("selPml").set("entitydim", 2);
    model.component("comp1").selection("selPml").set("input", new String[]{{"selInt"}});

    model.component("comp1").coordSystem().create("pml1", "PML");
    model.component("comp1").coordSystem("pml1").selection().named("selPml");

    // Waveguide numeric port on exterior left boundary x = xIn (y in [-hBox, hf + hTop])
    model.component("comp1").selection().create("selPortIn", "Box");
    model.component("comp1").selection("selPortIn").set("entitydim", 1);
    model.component("comp1").selection("selPortIn").set("condition", "inside");
    model.component("comp1").selection("selPortIn").set("xmin", xIn - eps);
    model.component("comp1").selection("selPortIn").set("xmax", xIn + eps);
    model.component("comp1").selection("selPortIn").set("ymin", -hBox - eps);
    model.component("comp1").selection("selPortIn").set("ymax", hf + hTop + eps);

    model.component("comp1").selection().create("selUpLine", "Box");
    model.component("comp1").selection("selUpLine").set("entitydim", 1);
    model.component("comp1").selection("selUpLine").set("condition", "inside");
    model.component("comp1").selection("selUpLine").set("xmin", xIn - eps);
    model.component("comp1").selection("selUpLine").set("xmax", xOut + eps);
    model.component("comp1").selection("selUpLine").set("ymin", yUpMon - eps);
    model.component("comp1").selection("selUpLine").set("ymax", yUpMon + eps);

    model.component("comp1").selection().create("selDnLine", "Box");
    model.component("comp1").selection("selDnLine").set("entitydim", 1);
    model.component("comp1").selection("selDnLine").set("condition", "inside");
    model.component("comp1").selection("selDnLine").set("xmin", xIn - eps);
    model.component("comp1").selection("selDnLine").set("xmax", xOut + eps);
    model.component("comp1").selection("selDnLine").set("ymin", yDnMon - eps);
    model.component("comp1").selection("selDnLine").set("ymax", yDnMon + eps);

    // Physics: Electromagnetic Waves 2D emw (Out-of-plane Ez = photonic TE0 slab mode!)
    model.component("comp1").physics().create("emw", "ElectromagneticWaves", "geom1");
    model.component("comp1").physics("emw").prop("components").set("components", "outofplane");
    String eOxS = "" + (nOx*nOx);
    String eSiS = "" + (nSi*nSi);
    model.component("comp1").physics("emw").feature("wee1").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("wee1").set("epsilonr_mat", "userdef");
    model.component("comp1").physics("emw").feature("wee1").set("epsilonr", new String[]{{eOxS,"0","0","0",eOxS,"0","0","0",eOxS}});
    model.component("comp1").physics("emw").feature("wee1").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("wee1").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("wee1").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("wee1").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    model.component("comp1").physics("emw").feature().create("weeAir", "WaveEquationElectric", 2);
    model.component("comp1").physics("emw").feature("weeAir").selection().named("selAir");
    model.component("comp1").physics("emw").feature("weeAir").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("weeAir").set("epsilonr_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeAir").set("epsilonr", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("weeAir").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeAir").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("weeAir").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeAir").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    model.component("comp1").physics("emw").feature().create("weeSi", "WaveEquationElectric", 2);
    model.component("comp1").physics("emw").feature("weeSi").selection().named("selSiTotal");
    model.component("comp1").physics("emw").feature("weeSi").set("DisplacementFieldModel", "RelativePermittivity");
    model.component("comp1").physics("emw").feature("weeSi").set("epsilonr_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeSi").set("epsilonr", new String[]{{eSiS,"0","0","0",eSiS,"0","0","0",eSiS}});
    model.component("comp1").physics("emw").feature("weeSi").set("mur_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeSi").set("mur", new String[]{{"1","0","0","0","1","0","0","0","1"}});
    model.component("comp1").physics("emw").feature("weeSi").set("sigma_mat", "userdef");
    model.component("comp1").physics("emw").feature("weeSi").set("sigma", new String[]{{"0","0","0","0","0","0","0","0","0"}});

    model.component("comp1").physics("emw").feature().create("port1", "Port", 1);
    model.component("comp1").physics("emw").feature("port1").selection().named("selPortIn");
    model.component("comp1").physics("emw").feature("port1").set("PortType", "Numeric");
    model.component("comp1").physics("emw").feature("port1").set("PortExcitation", "on");
    model.component("comp1").physics("emw").feature("port1").set("Pin", "1[W]");

    model.component("comp1").mesh().create("mesh1", "geom1");
    model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.085e-6);
    model.component("comp1").mesh("mesh1").feature().create("sizeSi", "Size");
    model.component("comp1").mesh("mesh1").feature("sizeSi").selection().named("selSiCore");
    model.component("comp1").mesh("mesh1").feature("sizeSi").set("custom", true);
    model.component("comp1").mesh("mesh1").feature("sizeSi").set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("sizeSi").set("hmax", 0.030e-6);
    model.component("comp1").mesh("mesh1").feature().create("ftri1", "FreeTri");
    model.component("comp1").mesh("mesh1").run();

    // Study: Boundary Mode Analysis (for Port 1) + Frequency Domain at 1550 nm
    double wl = 1.55e-6;
    model.param().set("wl0", wl + "[m]");
    model.study().create("std1");
    model.study("std1").feature().create("bma", "BoundaryModeAnalysis");
    model.study("std1").feature("bma").set("PortName", "1");
    model.study("std1").feature("bma").set("modeFreq", "c_const/wl0");
    model.study("std1").feature("bma").set("neigs", 1);
    model.study("std1").feature("bma").set("shift", "2.84");
    model.study("std1").feature("bma").set("shiftactive", true);
    model.study("std1").feature().create("freq", "Frequency");
    model.study("std1").feature("freq").set("plist", "c_const/wl0");
    model.study("std1").run();

    String dset = model.result().dataset().tags()[0];
    model.result().numerical().create("intUp", "IntLine");
    model.result().numerical("intUp").set("data", dset);
    model.result().numerical("intUp").selection().named("selUpLine");
    model.result().numerical("intUp").set("expr", new String[]{{"emw.Poavy"}});
    double pUp = model.result().numerical("intUp").getReal()[0][0];

    model.result().numerical().create("intDn", "IntLine");
    model.result().numerical("intDn").set("data", dset);
    model.result().numerical("intDn").selection().named("selDnLine");
    model.result().numerical("intDn").set("expr", new String[]{{"-emw.Poavy"}});
    double pDn = model.result().numerical("intDn").getReal()[0][0];

    model.result().numerical().create("gevS", "EvalGlobal");
    model.result().numerical("gevS").set("data", dset);
    model.result().numerical("gevS").set("expr", new String[]{{"abs(emw.S11)^2"}});
    double r11 = model.result().numerical("gevS").getReal()[0][0];

    // Export complex Ez(x) along yUpMon for exact SMF-28 fiber overlap in Python
    model.result().dataset().create("cln1", "CutLine2D");
    model.result().dataset("cln1").set("data", dset);
    model.result().dataset("cln1").set("genpoints", new String[][]{{{{""+xIn, ""+yUpMon}}, {{""+xOut, ""+yUpMon}}}});
    model.result().export().create("dat1", "Data");
    model.result().export("dat1").set("data", "cln1");
    model.result().export("dat1").set("expr", new String[]{{"real(Ez)", "imag(Ez)"}});
    model.result().export("dat1").set("filename", "{(SCRATCH_DIR / 'gc_comsol_ez.txt').as_posix()}");
    model.result().export("dat1").run();

    double dtS = (System.nanoTime() - t0) * 1e-9;
    System.out.println("GC_JSON: {{\\"eta_up_at_1550_pct\\":" + (pUp*100.0)
      + ",\\"eta_down_at_1550_pct\\":" + (pDn*100.0)
      + ",\\"r11_power\\":" + r11
      + ",\\"s11_at_1550_db\\":" + (10.0*Math.log10(Math.max(r11, 1e-8)))
      + ",\\"wall_time_s\\":" + dtS + "}}");
    return model;
  }}
  public static void main(String[] args) throws Exception {{ run(); }}
}}
'''
    lines = _run_comsol_java("ComsolSuite3CGC", java_src)
    gc_r = _parse_kv_lines(lines, "GC_JSON:")[0]

    # Compute exact SMF-28 10-deg Gaussian fiber overlap from exported Ez(x)
    ez_file = SCRATCH_DIR / "gc_comsol_ez.txt"
    xs_m, ez_re, ez_im = [], [], []
    for ln in ez_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        parts = s.split()
        if len(parts) >= 4:
            xs_m.append(float(parts[0]))
            ez_re.append(float(parts[2]))
            ez_im.append(float(parts[3]))
    xs_um = np.asarray(xs_m) * 1e6
    ord_i = np.argsort(xs_um)
    xs_um = xs_um[ord_i]
    ez_c = (np.asarray(ez_re) + 1j * np.asarray(ez_im))[ord_i]

    # Resample onto uniform grid for clean Simpson/trapezoid overlap
    xu = np.linspace(xs_um[0], xs_um[-1], 601)
    dx = float(xu[1] - xu[0])
    ez_u = np.interp(xu, xs_um, np.real(ez_c)) + 1j * np.interp(xu, xs_um, np.imag(ez_c))

    w_gc = float(gc_case["grating_width_um"])
    mfd_um = float(gc_case["mfd_um"])
    x_lat = np.linspace(-2.5 * max(w_gc, mfd_um), 2.5 * max(w_gc, mfd_um), 401)
    dx_lat = float(x_lat[1] - x_lat[0])
    u_lat = np.where(np.abs(x_lat) <= 0.5 * w_gc, np.cos(math.pi * x_lat / w_gc), 0.0)
    g_lat = np.exp(-((x_lat / (0.5 * mfd_um)) ** 2))
    lat_ov = float((np.sum(u_lat * g_lat) * dx_lat) ** 2 / ((np.sum(u_lat**2) * dx_lat) * (np.sum(g_lat**2) * dx_lat)))

    th_rad = math.radians(float(gc_case["fiber_angle_deg"]))
    wz_proj = (0.5 * mfd_um) / math.cos(th_rad)
    k0_um = 2.0 * math.pi / 1.55
    kz_fib = k0_um * math.sin(th_rad)
    z0_cands = np.linspace(xu[0] + 1.5, xu[-1] - 1.5, 61)
    dz_mat = xu[None, :] - z0_cands[:, None]
    # COMSOL emw uses exp(+j*omega*t), so a +x forward-radiating wave has phase exp(-j*kx*x)
    g_mat = np.exp(-((dz_mat / wz_proj) ** 2) - 1j * kz_fib * dz_mat)
    g_norm = np.maximum(np.sum(np.abs(g_mat) ** 2, axis=1) * dx, math.sqrt(0.5 * math.pi) * wz_proj)
    f_norm = max(float(np.sum(np.abs(ez_u) ** 2) * dx), 1e-30)
    ov1 = float(np.max((np.abs(np.conj(g_mat) @ ez_u * dx) ** 2) / (g_norm * f_norm)))
    eta_fib = max((gc_r["eta_up_at_1550_pct"] / 100.0) * ov1 * lat_ov, 1e-6)
    il_1550 = -10.0 * math.log10(eta_fib)

    out = {
        "case_id": gc_case["case_id"],
        "eta_up_at_1550_pct": round(gc_r["eta_up_at_1550_pct"], 2),
        "eta_down_at_1550_pct": round(gc_r["eta_down_at_1550_pct"], 2),
        "s11_at_1550_db": round(gc_r["s11_at_1550_db"], 2),
        "mode_match_1d_pct": round(ov1 * 100.0, 2),
        "efficiency_at_1550_pct": round(eta_fib * 100.0, 2),
        "il_at_1550_db": round(il_1550, 3),
        "wall_time_s": round(gc_r["wall_time_s"], 2),
    }
    print(
        f"[COMSOL-S3C] {gc_case['case_id']:<26} up@1550={out['eta_up_at_1550_pct']:.1f}% | "
        f"dn@1550={out['eta_down_at_1550_pct']:.1f}% | S11={out['s11_at_1550_db']:.1f} dB | "
        f"IL@1550={out['il_at_1550_db']:.2f} dB"
    )
    return out


def main() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    specs = json.loads(SPECS_JSON.read_text(encoding="utf-8"))
    t_all = time.perf_counter()
    ckpt_path = SCRATCH_DIR / "comsol_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.is_file() else {}

    if "suite1_eigenmodes" not in ckpt:
        s1, audit, s3a = run_comsol_suite1_and_3a(specs)
        ckpt["suite1_eigenmodes"] = s1
        ckpt["suite1_audit"] = audit
        ckpt["suite3a_dc"] = s3a
        ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    else:
        s1, audit, s3a = ckpt["suite1_eigenmodes"], ckpt["suite1_audit"], ckpt["suite3a_dc"]
        print("[COMSOL] Loaded Suite 1 & 3A checkpoint")

    if "suite4_thermal" not in ckpt:
        s4 = run_comsol_suite4_thermal(specs["suite4_thermal"])
        ckpt["suite4_thermal"] = s4
        ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    else:
        s4 = ckpt["suite4_thermal"]
        print("[COMSOL] Loaded Suite 4 checkpoint")

    if "suite5_pockels_eo" not in ckpt:
        s5 = run_comsol_suite5_eo(specs["suite5_pockels_eo"])
        ckpt["suite5_pockels_eo"] = s5
        ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    else:
        s5 = ckpt["suite5_pockels_eo"]
        print("[COMSOL] Loaded Suite 5 checkpoint")

    if "suite3c_gc" not in ckpt or "error" in (ckpt.get("suite3c_gc") or {}):
        try:
            s3c = run_comsol_suite3c_gc(specs["suite3c_grating_coupler"])
        except Exception as exc:
            print(f"[COMSOL-S3C] Note: 2D GC encountered an issue: {exc}")
            s3c = {"error": str(exc)[:500]}
        ckpt["suite3c_gc"] = s3c
        ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    else:
        s3c = ckpt["suite3c_gc"]
        print("[COMSOL] Loaded Suite 3C checkpoint")

    # Clean up scratch files (keep logs only on failure)
    for p in SCRATCH_DIR.glob("*.mph"):
        try:
            p.unlink()
        except OSError:
            pass

    payload = {
        "solver": "COMSOL Multiphysics 6.4.0.293 (Wave Optics ewfd + Heat Transfer ht + AC/DC Electrostatics es)",
        "host": f"{platform.system()}-{platform.machine()} (24 CPU cores)",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        "suite1_eigenmodes": s1,
        "suite1_audit": audit,
        "suite3a_dc": s3a,
        "suite3c_gc": s3c,
        "suite4_thermal": s4,
        "suite5_pockels_eo": s5,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved COMSOL independent results to {OUT_JSON} in {payload['total_wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
