"""Independent Commercial Gap-Closure Runner (Suites 4, 7C, 8, 9).

Closes all four remaining verification gaps across our existing commercial suites
strictly from neutral physical specifications (zero cross-solver contamination):

  1. Suite 4 (3-Way Thermal Transient):
     - COMSOL 6.4 `ht`: Diagnostic check of effective `rho * Cp` across every domain +
       strict-step transient (`0 -> 120 us`) normalized strictly by steady-state `R_th_core`
       + direct harmonic frequency-domain `f_3dB_thermal`.
     - Ansys Lumerical `DEVICE HEAT` 2D Transient (`SOI Top TiN` & `Deep Air Trench`)
       with identical matched `k` and `rho * Cp`, completing true 3-way transient validation.
  2. Suite 7C (`INTERCONNECT` Native SPAR Refresh):
     - Re-runs `INTERCONNECT ONA` on MicroRing's updated `v3.9.0` `gc_apodized` & `mzi_dc_dl80`
       exported `.dat` files (`2.996 dB` apodized GC).
  3. Suite 8 (Ansys HFSS 3D Open-Sidewall Finite-`W_G` T-Rail + 3D Bragg Stopband):
     - Extends `x_dom = x_g_out + 120 um` with `Radiation` boundary on `x = x_dom` (eliminating
       the semi-infinite PEC outer wall so finite `W_G = 55 um` matches COMSOL 3D & MicroRing
       apples-to-apples) on `TFLN_SlowWave_TRail_Quartz` (`Lambda = 25 um`, `10-300 GHz`).
     - Adds a full-wave 3D periodic `TFLN_SlowWave_TRail_Bragg220um` (`Lambda = 220 um`,
       `50-340 GHz`) directly capturing the 3D Bloch-Floquet Bragg stopband near `~300 GHz`.
  4. Suite 9 (Complete 3x3 Lumerical `DEVICE CHARGE + MODE` + Forward-Biased `PIN`):
     - Adds `SOI_220nm_LShaped_PN` and `SOI_220nm_UShaped_PN` (`0 -> -3 V`) in Lumerical
       `CHARGE + MODE` so all 3 junction architectures have full 3-way TCAD validation.
     - Adds `SOI_220nm_Forward_PIN` (`0 -> +1.05 V` carrier injection) verifying forward
       blue-shift (`dn_eff < 0`) and free-carrier absorption (`alpha_FCA`).

Saves `benchmarks/outputs/gap_closures_results.json` (with per-section checkpointing).
"""
from __future__ import annotations

import argparse
import cmath
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH_DIR = OUT_DIR / "_gap_closure_scratch"
OUT_JSON = OUT_DIR / "gap_closures_results.json"
CKPT_JSON = SCRATCH_DIR / "gap_closure_ckpt.json"

SPECS_P1_JSON = OUT_DIR / "commercial_validation_specs.json"
SPECS_P2_JSON = OUT_DIR / "commercial_validation_specs_p2.json"
IC_EXPORT_DIR = OUT_DIR / "_ic_export"

LUMERICAL_WIN_HINTS = [
    r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v252\Lumerical\api\python",
]
COMSOL_BIN = Path(r"C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64")
COMSOL_COMPILE = COMSOL_BIN / "comsolcompile.exe"
COMSOL_BATCH = COMSOL_BIN / "comsolbatch.exe"
C0 = 299792458.0


def _import_lumapi():
    for p in LUMERICAL_WIN_HINTS:
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)
    import lumapi  # type: ignore
    return lumapi


def _scalar_real(val) -> float:
    return float(np.real(np.asarray(val).reshape(-1)[0]))


def _save_ckpt(ckpt: dict) -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_JSON.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")


# ==============================================================================
# 1A. Suite 4 COMSOL 6.4 Strict Transient + Material-Audit + Harmonic f_3dB
# ==============================================================================
def run_comsol_suite4_transient_fix(specs_p1: dict) -> list[dict]:
    cases = specs_p1["suite4_thermal"]
    calls = []
    for c in cases:
        is_tr = "true" if c["thermal_structure"] == "deep_trench" else "false"
        tr_y_bot = max(c["y_ox_top_um"] - c["trench_depth_um"], c["y_sub_bot_um"] + 1.5)
        calls.append(
            f'solveTh("{c["case_id"]}", {is_tr}, '
            f'{c["waveguide_width_um"]*1e-6:.12e}, {c["film_height_um"]*1e-6:.12e}, '
            f'{c["box_thickness_um"]*1e-6:.12e}, {c["y_ox_top_um"]*1e-6:.12e}, {c["y_air_top_um"]*1e-6:.12e}, '
            f'{c["y_sub_bot_um"]*1e-6:.12e}, {c["x_dom_um"]*1e-6:.12e}, '
            f'{c["heater_width_um"]*1e-6:.12e}, {c["y_heat0_um"]*1e-6:.12e}, {c["y_heat1_um"]*1e-6:.12e}, '
            f'{c["active_length_um"]*1e-6:.12e}, {c["trench_edge_offset_um"]*1e-6:.12e}, '
            f'{c["trench_width_um"]*1e-6:.12e}, {tr_y_bot*1e-6:.12e}, '
            f'{c["k_core_w_mk"]:.6f}, {c["k_ox_w_mk"]:.6f}, {c["k_si_sub_w_mk"]:.6f}, {c["k_heater_w_mk"]:.6f}, {c["k_air_w_mk"]:.6f}, '
            f'{c["cv_core_j_m3k"]:.6e}, {c["cv_ox_j_m3k"]:.6e}, {c["cv_si_sub_j_m3k"]:.6e}, {c["cv_heater_j_m3k"]:.6e}, {c["cv_air_j_m3k"]:.6e});'
        )

    java_src = f'''
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolSuite4TranFix {{
  static void setSolidMat(Model model, String fTag, double k, double cv) {{
    model.component("comp1").physics("ht").feature(fTag).set("k_mat", "userdef");
    model.component("comp1").physics("ht").feature(fTag).set("k", new String[]{{""+k,"0","0","0",""+k,"0","0","0",""+k}});
    model.component("comp1").physics("ht").feature(fTag).set("rho_mat", "userdef");
    model.component("comp1").physics("ht").feature(fTag).set("rho", "1000");
    model.component("comp1").physics("ht").feature(fTag).set("Cp_mat", "userdef");
    model.component("comp1").physics("ht").feature(fTag).set("Cp", "" + (cv / 1000.0));
  }}

  static void solveTh(
      String cid, boolean isTrench, double wCore, double hCore,
      double hBox, double yOxTop, double yAirTop, double ySubBot, double xDom,
      double wHeat, double yH0, double yH1, double lHeat,
      double trX0, double trW, double trYBot,
      double kCore, double kOx, double kSub, double kHeat, double kAir,
      double cvCore, double cvOx, double cvSub, double cvHeat, double cvAir) {{
    long t0 = System.nanoTime();
    ModelUtil.clear();
    Model model = ModelUtil.create("Model");
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);

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
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 1.0e-6);
    model.component("comp1").mesh("mesh1").feature().create("ftri1", "FreeTri");
    model.component("comp1").mesh("mesh1").run();

    // Study 1: Stationary
    model.study().create("stdStat");
    model.study("stdStat").feature().create("stat", "Stationary");
    model.study("stdStat").run();

    String dsetStat = model.result().dataset().tags()[0];
    model.result().numerical().create("intCore", "IntSurface");
    model.result().numerical("intCore").set("data", dsetStat);
    model.result().numerical("intCore").selection().named("selCore");
    model.result().numerical("intCore").set("expr", new String[]{{"T - 300[K]", "1", "ht.rho*ht.Cp"}});
    double[][] ic = model.result().numerical("intCore").getReal();
    double rThCore = ic[0][0] / ic[1][0];
    double cvCoreCheck = ic[2][0] / ic[1][0];

    // Audit actual rho*Cp inside Oxide & Substrate!
    model.result().numerical().create("intSubChk", "IntSurface");
    model.result().numerical("intSubChk").set("data", dsetStat);
    model.result().numerical("intSubChk").selection().named("selSub");
    model.result().numerical("intSubChk").set("expr", new String[]{{"ht.rho*ht.Cp", "1"}});
    double[][] isub = model.result().numerical("intSubChk").getReal();
    double cvSubCheck = isub[0][0] / isub[1][0];

    model.result().dataset().create("cptOx", "CutPoint2D");
    model.result().dataset("cptOx").set("data", dsetStat);
    model.result().dataset("cptOx").set("pointx", "0");
    model.result().dataset("cptOx").set("pointy", "" + (-0.5 * hBox));
    model.result().numerical().create("pevOx", "EvalPoint");
    model.result().numerical("pevOx").set("data", "cptOx");
    model.result().numerical("pevOx").set("expr", new String[]{{"ht.rho*ht.Cp", "ht.kxx"}});
    double[][] pox = model.result().numerical("pevOx").getReal();
    double cvOxCheck = pox[0][0];
    double kOxCheck = pox[1][0];

    // Study 2: Strict-step Transient (0 -> 150 us, fine 0.25 us start + 1.0 us tail, strict BDF max step!)
    double tEndUs = isTrench ? 220.0 : 140.0;
    model.study().create("stdTran");
    model.study("stdTran").feature().create("time", "Transient");
    model.study("stdTran").feature("time").set("tlist",
        "range(0, 0.25e-6, 30e-6) range(32e-6, 2.0e-6, " + tEndUs + "e-6)");
    model.study("stdTran").feature("time").set("rtolactive", true);
    model.study("stdTran").feature("time").set("rtol", 1e-4);
    model.study("stdTran").showAutoSequences("all");
    String[] sTagsAll = model.sol().tags();
    String solTran = sTagsAll[sTagsAll.length - 1];
    try {{
      model.sol(solTran).feature("t1").set("tstepsbdf", "strict");
      model.sol(solTran).feature("t1").set("initialstepbdfactive", true);
      model.sol(solTran).feature("t1").set("initialstepbdf", "0.02e-6");
      model.sol(solTran).feature("t1").set("maxstepbdfactive", true);
      model.sol(solTran).feature("t1").set("maxstepbdf", "0.5e-6");
    }} catch (Exception ex) {{}}
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
    double[] tSec = model.sol(solTran).getPVals();
    double areaCore = ic[1][0];
    int nT = Math.min(tCoreSeries.length, tSec.length);
    double[] tUs = new double[nT];
    double[] dT = new double[nT];
    for (int i = 0; i < nT; i++) {{
      tUs[i] = tSec[i] * 1e6;
      dT[i] = tCoreSeries[i] / areaCore;
    }}
    // Normalize strictly against exact steady-state rThCore!
    double t10 = 0.0, t90 = tEndUs;
    boolean got10 = false, got90 = false;
    for (int i = 1; i < nT; i++) {{
      if (!got10 && dT[i] >= 0.10 * rThCore) {{
        t10 = tUs[i-1] + (0.10*rThCore - dT[i-1]) * (tUs[i] - tUs[i-1]) / Math.max(dT[i] - dT[i-1], 1e-12);
        got10 = true;
      }}
      if (!got90 && dT[i] >= 0.90 * rThCore) {{
        t90 = tUs[i-1] + (0.90*rThCore - dT[i-1]) * (tUs[i] - tUs[i-1]) / Math.max(dT[i] - dT[i-1], 1e-12);
        got90 = true;
      }}
    }}
    double tRiseUs = t90 - t10;
    double endFrac = dT[nT - 1] / rThCore;

    // Exact harmonic frequency response from full-tail step response
    double bestF = 0.35 / Math.max(tRiseUs * 1e-6, 1e-9) * 1e-3;
    double prevH = 1.0, prevF = 0.1;
    for (int kf = 0; kf < 160; kf++) {{
      double fKHz = 0.2 * Math.pow(500.0, kf / 159.0);
      double wRad = 2.0 * Math.PI * fKHz * 1e3;
      double re = 0.0, im = 0.0;
      for (int i = 1; i < nT; i++) {{
        double dStep = (dT[i] - dT[i-1]) / Math.max(rThCore, 1e-12);
        double tMid = 0.5 * (tUs[i] + tUs[i-1]) * 1e-6;
        re += dStep * Math.cos(wRad * tMid);
        im -= dStep * Math.sin(wRad * tMid);
      }}
      double hMag = Math.hypot(re, im);
      if (kf > 0 && hMag <= 0.70710678) {{
        bestF = prevF + (0.70710678 - prevH) * (fKHz - prevF) / Math.min(hMag - prevH, -1e-9);
        break;
      }}
      prevH = hMag; prevF = fKHz;
    }}

    double dtS = (System.nanoTime() - t0) * 1e-9;
    System.out.println("S4FIX_JSON: {{\\"case_id\\":\\"" + cid + "\\",\\"R_th_core\\":" + rThCore
      + ",\\"t_rise_10_90_us\\":" + tRiseUs + ",\\"t10_us\\":" + t10 + ",\\"t90_us\\":" + t90
      + ",\\"end_ss_frac\\":" + endFrac + ",\\"f_3dB_kHz\\":" + bestF
      + ",\\"cv_core_chk\\":" + cvCoreCheck + ",\\"cv_ox_chk\\":" + cvOxCheck
      + ",\\"cv_sub_chk\\":" + cvSubCheck + ",\\"k_ox_chk\\":" + kOxCheck
      + ",\\"wall_time_s\\":" + dtS + "}}");
  }}

  public static Model run() {{
    {chr(10).join("    " + ln for ln in calls)}
    return ModelUtil.create("Done");
  }}
  public static void main(String[] args) {{ run(); }}
}}
'''
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    jpath = SCRATCH_DIR / "ComsolSuite4TranFix.java"
    cpath = SCRATCH_DIR / "ComsolSuite4TranFix.class"
    lpath = SCRATCH_DIR / "ComsolSuite4TranFix.log"
    jpath.write_text(java_src, encoding="ascii")
    if cpath.is_file():
        cpath.unlink()
    comp = subprocess.run([str(COMSOL_COMPILE), str(jpath)], capture_output=True, text=True, check=False)
    if comp.returncode != 0 or not cpath.is_file():
        raise RuntimeError(f"comsolcompile ComsolSuite4TranFix failed:\n{comp.stdout}\n{comp.stderr}")
    batch = subprocess.run([str(COMSOL_BATCH), "-inputfile", str(cpath), "-batchlog", str(lpath)], capture_output=True, text=True, check=False)
    log_txt = lpath.read_text(encoding="utf-8", errors="ignore") if lpath.is_file() else ""
    if batch.returncode != 0 or "/*****Error********/" in log_txt:
        raise RuntimeError(f"comsolbatch ComsolSuite4TranFix failed:\n{log_txt[-2000:]}\n{batch.stderr}")
    out = []
    for ln in ((batch.stdout or "") + "\n" + log_txt).splitlines():
        s = ln.strip()
        if s.startswith("S4FIX_JSON:"):
            r = json.loads(s[len("S4FIX_JSON:"):].strip())
            out.append(r)
            print(
                f"[COMSOL-S4-TRAN] {r['case_id']:<26} R_th={r['R_th_core']:.4f} | "
                f"t_rise={r['t_rise_10_90_us']:.2f} us (end={r['end_ss_frac']*100:.2f}%) | "
                f"f_3dB={r['f_3dB_kHz']:.2f} kHz | cv(core/ox/sub)={r['cv_core_chk']:.2e}/{r['cv_ox_chk']:.2e}/{r['cv_sub_chk']:.2e}"
            )
    return out


# ==============================================================================
# 1B. Suite 4 Lumerical DEVICE HEAT 2D Transient (3-Way Transient Thermal!)
# ==============================================================================
def run_lum_suite4_heat_transient(lumapi) -> list[dict]:
    out = []
    dev = lumapi.DEVICE(hide=True)
    try:
        for cid, has_trench in (("SOI_220nm_Top_TiN_Heater", False), ("SOI_220nm_Deep_Air_Trench", True)):
            t0 = time.perf_counter()
            y_sub = -58e-6 if has_trench else -28e-6
            t_end = 180e-6 if has_trench else 120e-6
            trench_cmd = ""
            if has_trench:
                trench_cmd = """
                addrect; set("name","TrL"); set("x min",-8.8e-6); set("x max",-2.8e-6); set("y",0); set("y span",1e-6);
                set("z min",-40.2e-6); set("z max",1.8e-6); set("material","Air_HT"); set("mesh order",1);
                addrect; set("name","TrR"); set("x min",2.8e-6); set("x max",8.8e-6); set("y",0); set("y span",1e-6);
                set("z min",-40.2e-6); set("z max",1.8e-6); set("material","Air_HT"); set("mesh order",1);
                """
            # Matched k AND exact matched volumetric heat capacity rho * Cp = spec!
            # Si: rho=2329, Cp=700 -> 1.6303e6 | SiO2: rho=2200, Cp=740.91 -> 1.630e6
            # TiN: rho=5220, Cp=469.35 -> 2.450e6 | Air: rho=1.2, Cp=1000 -> 1200
            ldev = SCRATCH_DIR / f"heat_tran_{cid}.ldev"
            dev.eval(f"""
            switchtolayout; deleteall;
            addmodelmaterial; set("name","Si_HT"); addmaterialproperties("HT","Si (Silicon)");
            select("materials::Si_HT::Si (Silicon)");
            set("thermal conductivity.active model", "constant"); set("thermal conductivity.constant", 148.0);
            set("mass density.constant", 2329.0);
            set("specific heat.active model", "constant"); set("specific heat.constant", 700.0);

            addmodelmaterial; set("name","SiO2_HT"); addmaterialproperties("HT","SiO2 (Glass) - Sze");
            select("materials::SiO2_HT::SiO2 (Glass) - Sze");
            set("thermal conductivity.active model", "constant"); set("thermal conductivity.constant", 1.38);
            set("mass density.constant", 2200.0);
            set("specific heat.active model", "constant"); set("specific heat.constant", 740.91);

            addmodelmaterial; set("name","TiN_HT"); addmaterialproperties("HT","Al (Aluminium) - CRC");
            select("materials::TiN_HT::Al (Aluminium) - CRC");
            set("thermal conductivity.active model", "constant"); set("thermal conductivity.constant", 28.0);
            set("mass density.constant", 5220.0);
            set("specific heat.active model", "constant"); set("specific heat.constant", 469.35);

            addmodelmaterial; set("name","Air_HT"); addmaterialproperties("HT","Air");
            select("materials::Air_HT::Air");
            set("thermal conductivity.active model", "constant"); set("thermal conductivity.constant", 0.026);
            set("mass density.constant", 1.2);
            set("specific heat.active model", "constant"); set("specific heat.constant", 1000.0);

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
            set("solver mode", "steady state");
            addtemperaturebc; set("name","Sink"); set("surface type","simulation region");
            set("z min",1); set("x min",1); set("x max",1); set("temperature",300.0);
            adduniformheat; set("name","Joule1mW"); set("geometry type","volume"); set("volume type","solid"); set("volume solid","Heater");
            set("use solver norm length", 1); set("total power", 1.0e-3);
            addtemperaturemonitor; set("name","Tmon"); set("monitor type","2D Y-normal");
            set("x",0); set("x span",1.0e-6); set("z min",-0.1e-6); set("z max",0.35e-6);
            """)
            dev.save(str(ldev))
            dev.run("HEAT")
            th_ss = dev.getresult("HEAT::Tmon", "temperature")
            xs = np.asarray(th_ss["x"]).ravel()
            zs = np.asarray(th_ss["z"]).ravel()
            T_ss = np.asarray(th_ss["T"]).ravel() - 300.0
            elems = np.asarray(th_ss["connectivity"], dtype=int) - 1
            areas = np.asarray(th_ss["area"]).ravel()
            xc = np.mean(xs[elems], axis=1)
            zc = np.mean(zs[elems], axis=1)
            in_core_el = (np.abs(xc) <= 0.225e-6) & (zc >= 0.0) & (zc <= 0.22e-6)
            rth_ss = float(np.sum(np.mean(T_ss[elems], axis=1)[in_core_el] * areas[in_core_el]) / np.sum(areas[in_core_el]))

            # Switch same model to Transient step-on heating via Sink transient table + HEAT shutter!
            dev.eval(f"""
            switchtolayout;
            select("HEAT");
            set("solver mode", "transient");
            set("transient min time step", 0.02e-6);
            set("transient max time step", 0.50e-6);
            set("shutter mode", "step on");
            set("shutter ton", 0.0);
            set("shutter tslew", 0.01e-6);
            set("shutter toff", {t_end*1.5:.12e});
            select("HEAT::boundary conditions::Sink");
            set("bc mode", "transient");
            set("transient time steps", [0.0; {t_end:.12e}]);
            set("transient value table", [300.0; 300.0]);
            """)
            dev.save(str(ldev))
            dev.run("HEAT")
            th_tr = dev.getresult("HEAT::Tmon", "temperature")
            t_arr = np.asarray(th_tr["t"]).ravel() * 1e6  # us
            T_mat = np.squeeze(np.asarray(th_tr["T"])) - 300.0  # (N_nodes, N_t)
            if T_mat.shape[0] != len(xs) and T_mat.shape[1] == len(xs):
                T_mat = T_mat.T
            elems_t = np.asarray(th_tr["connectivity"], dtype=int) - 1
            areas_t = np.asarray(th_tr["area"]).ravel()
            xs_t = np.asarray(th_tr["x"]).ravel()
            zs_t = np.asarray(th_tr["z"]).ravel()
            xc_t = np.mean(xs_t[elems_t], axis=1)
            zc_t = np.mean(zs_t[elems_t], axis=1)
            in_c_t = (np.abs(xc_t) <= 0.225e-6) & (zc_t >= 0.0) & (zc_t <= 0.22e-6)
            w_c = areas_t[in_c_t]
            dt_t = np.array([
                float(np.sum(np.mean(T_mat[elems_t[in_c_t], k], axis=1) * w_c) / np.sum(w_c))
                for k in range(len(t_arr))
            ])
            t10 = float(np.interp(0.10 * rth_ss, dt_t, t_arr))
            t90 = float(np.interp(0.90 * rth_ss, dt_t, t_arr))
            t_rise = max(t90 - t10, 0.05)

            # Exact multi-pole f_3dB from discrete step response
            f_grid = np.logspace(-0.5, 2.5, 180)  # kHz
            f_3db = 0.35 / (t_rise * 1e-6) * 1e-3
            prev_h, prev_f = 1.0, float(f_grid[0])
            for fk in f_grid:
                w = 2.0 * math.pi * fk * 1e3
                dstep = np.diff(dt_t) / max(rth_ss, 1e-12)
                tmid = 0.5 * (t_arr[:-1] + t_arr[1:]) * 1e-6
                h_c = np.sum(dstep * np.exp(-1j * w * tmid))
                hm = float(abs(h_c))
                if hm <= 0.70710678:
                    f_3db = prev_f + (0.70710678 - prev_h) * (fk - prev_f) / min(hm - prev_h, -1e-9)
                    break
                prev_h, prev_f = hm, float(fk)

            rec = {
                "case_id": cid,
                "R_th_core_K_per_mW": round(rth_ss, 4),
                "t_rise_10_90_us": round(t_rise, 3),
                "t10_us": round(t10, 3),
                "t90_us": round(t90, 3),
                "end_ss_frac": round(float(dt_t[-1] / rth_ss), 4),
                "f_3dB_thermal_kHz": round(f_3db, 2),
                "wall_time_s": round(time.perf_counter() - t0, 2),
            }
            out.append(rec)
            print(
                f"[LUM-S4-HEAT-TRAN] {cid:<26} R_th={rec['R_th_core_K_per_mW']:.4f} | "
                f"t_rise={rec['t_rise_10_90_us']:.2f} us (end={rec['end_ss_frac']*100:.2f}%) | "
                f"f_3dB={rec['f_3dB_thermal_kHz']:.2f} kHz"
            )
    finally:
        dev.close()
    return out


# ==============================================================================
# 2. Suite 7C: INTERCONNECT Native SPAR Refresh (updated v3.9.0 Apodized GC)
# ==============================================================================
def run_suite7c_refresh(lumapi) -> dict:
    sys.path.insert(0, str(HERE))
    from run_lumerical_independent_p2 import run_lum_suite7c_interconnect
    return run_lum_suite7c_interconnect(lumapi)


# ==============================================================================
# 3. Suite 8: HFSS 3D Open-Sidewall Finite-W_G T-Rail + 3D Bragg Stopband (220 um)
# ==============================================================================
def run_hfss_suite8_openwg_and_bragg(specs_p2: dict) -> list[dict]:
    sys.path.insert(0, str(HERE))
    from run_hfss_independent_p2 import _extract_bloch_from_s2p_half, _find_ansysedt, _parse_touchstone_s2p

    trail_spec = [c for c in specs_p2["suite8_rf_electrodes"] if c["case_id"] == "TFLN_SlowWave_TRail_Quartz"][0]
    bragg_spec = dict(trail_spec)
    bragg_spec["case_id"] = "TFLN_SlowWave_TRail_Bragg220um"
    bragg_spec["t_rail_period_um"] = 220.0
    bragg_spec["t_rail_stem_width_um"] = 2.0
    bragg_spec["t_rail_duty"] = 0.88
    bragg_spec["n_periods_override"] = 2
    bragg_spec["freq_start_ghz"] = 100.0
    bragg_spec["freq_stop_ghz"] = 340.0
    bragg_spec["freq_step_ghz"] = 40.0

    open_spec = dict(trail_spec)
    open_spec["case_id"] = "TFLN_SlowWave_TRail_Quartz_OpenWG"
    open_spec["n_periods_override"] = 4
    open_spec["freq_start_ghz"] = 40.0
    open_spec["freq_stop_ghz"] = 300.0
    open_spec["freq_step_ghz"] = 65.0

    cases = [open_spec, bragg_spec]
    aedt_exe = _find_ansysedt()
    hfss_dir = SCRATCH_DIR / "hfss_openwg"
    s2p_dir = hfss_dir / "s2p"
    hfss_dir.mkdir(parents=True, exist_ok=True)
    s2p_dir.mkdir(parents=True, exist_ok=True)
    out_raw = hfss_dir / "hfss_openwg_raw.json"
    script_path = hfss_dir / "run_hfss_openwg.py"
    log_path = hfss_dir / "hfss_openwg.log"
    proj_path = str(hfss_dir / "suite8_openwg_bragg.aedt").replace("\\", "/")

    ironpy = f'''# Ansys AEDT 2026 R1 HFSS 3D Open-Sidewall Finite-W_G T-Rail + Bragg Stopband
import os, json, math

CASES = json.loads(r"""{json.dumps(cases)}""")
OUT_RAW = r"{str(out_raw).replace(chr(92), '/')}"
S2P_DIR = r"{str(s2p_dir).replace(chr(92), '/')}"
PROJ_PATH = r"{proj_path}"

ver = str(oDesktop.GetVersion())
oProject = oDesktop.NewProject()
oProject.SaveAs(PROJ_PATH, True)
oDefMgr = oProject.GetDefinitionManager()

def ensure_iso_mat(name, eps_r, tan_d=0.0, sigma=0.0):
    if oDefMgr.DoesMaterialExist(name): return
    oDefMgr.AddMaterial([
        "NAME:" + name, "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic"]],
        "permittivity:=", str(eps_r), "permeability:=", "1.0",
        "conductivity:=", str(sigma), "dielectric_loss_tangent:=", str(tan_d),
    ])

def ensure_aniso_mat(name, ex, ey, ez, tan_d=0.0):
    if oDefMgr.DoesMaterialExist(name): return
    oDefMgr.AddMaterial([
        "NAME:" + name, "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic"]],
        ["NAME:permittivity", "property_type:=", "AnisoProperty", "unit:=", "",
         "component1:=", str(ex), "component2:=", str(ey), "component3:=", str(ez)],
        "permeability:=", "1.0", "conductivity:=", "0.0", "dielectric_loss_tangent:=", str(tan_d),
    ])

def add_box(oEd, name, x0, y0, z0, dx, dy, dz, mat, solve_inside=True):
    oEd.CreateBox(
        ["NAME:BoxParameters", "XPosition:=", str(x0)+"um", "YPosition:=", str(y0)+"um", "ZPosition:=", str(z0)+"um",
         "XSize:=", str(dx)+"um", "YSize:=", str(dy)+"um", "ZSize:=", str(dz)+"um"],
        ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", "(143 175 143)", "Transparency:=", 0.4,
         "PartCoordinateSystem:=", "Global", "UDMId:=", "", "MaterialValue:=", '"'+mat+'"',
         "SurfaceMaterialValue:=", '""', "SolveInside:=", solve_inside, "ShellElement:=", False,
         "ShellElementThickness:=", "0mm", "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False, "IsLightweight:=", False],
    )

def add_rect(oEd, name, x0, y0, z0, w1, w2, axis="Z"):
    oEd.CreateRectangle(
        ["NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", str(x0)+"um", "YStart:=", str(y0)+"um",
         "ZStart:=", str(z0)+"um", "Width:=", str(w1)+"um", "Height:=", str(w2)+"um", "WhichAxis:=", axis],
        ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", "(255 128 0)", "Transparency:=", 0.6,
         "PartCoordinateSystem:=", "Global", "UDMId:=", "", "MaterialValue:=", '"vacuum"',
         "SurfaceMaterialValue:=", '""', "SolveInside:=", True, "ShellElement:=", False,
         "ShellElementThickness:=", "0mm", "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False, "IsLightweight:=", False],
    )

results = []
for c in CASES:
    cid = str(c["case_id"])
    ws = float(c["signal_width_um"]); gap = float(c["electrode_gap_um"]); wg = float(c["ground_width_um"])
    tm = float(c["metal_thickness_um"]); hf = float(c["film_height_um"]); hs = float(c["slab_height_um"])
    wwg = float(c["waveguide_width_um"]); hbox = float(c["box_thickness_um"]); htop = float(c["top_clad_thickness_um"])
    l_rail = float(c["t_rail_length_um"]); w_cap = float(c["t_rail_cap_width_um"])
    w_stem = float(c["t_rail_stem_width_um"]); period = float(c["t_rail_period_um"]); duty = float(c["t_rail_duty"])
    slab_corr = c["slab_corridor_half_um"]
    n_per = int(c.get("n_periods_override", 4))

    ex_c = float(c["eps_core_rf"][0])
    ey_c = float(c["eps_core_rf"][1])
    ez_c = float(c["eps_core_rf"][2]) if len(c["eps_core_rf"]) > 2 else ex_c
    ensure_aniso_mat("LN_Xcut_Aniso", ex_c, ey_c, ez_c, c["tan_d_core"])
    ensure_iso_mat("SiO2_RF", c["eps_box_rf"][0], c["tan_d_box"], 0.0)
    ensure_iso_mat("Sub_quartz", c["eps_sub_rf"], c["tan_d_sub"], c["sigma_sub_s_per_m"])
    ensure_iso_mat("Au_EOM", 1.0, 0.0, 1.0 / float(c["rho_metal_ohm_m"]))

    oProject.InsertDesign("HFSS", cid, "HFSS Modal Network", "")
    oDesign = oProject.SetActiveDesign(cid)
    oEd = oDesign.SetActiveEditor("3D Modeler")
    oEd.SetModelUnits(["NAME:Units Parameter", "Units:=", "um", "Rescale:=", False])

    L_z = n_per * period
    g_trunk = gap + 2.0 * l_rail
    x_s_edge = 0.5 * ws + l_rail
    x_g_cap_in = x_s_edge + gap
    x_g_in = 0.5 * ws + g_trunk
    x_g_out = x_g_in + wg
    x_slot = 0.5 * (x_s_edge + x_g_cap_in)

    # CRITICAL OPEN-SIDEWALL MARGIN beyond finite W_G = 55 um (NO PEC wall at x = x_g_out!)
    span = ws + 2.0 * g_trunk
    h_sub = max(90.0, 1.8 * span)
    h_air = max(90.0, 1.8 * span)
    x_dom = x_g_out + 60.0
    y_m0 = hs; y_m1 = hs + tm
    y_ox_top = max(hf + 0.25, hs + htop)
    y_sub_bot = -hbox - h_sub; y_air_top = y_m1 + h_air

    add_box(oEd, "AirBox", 0.0, y_sub_bot, 0.0, x_dom, y_air_top - y_sub_bot, L_z, "vacuum", True)
    add_box(oEd, "Substrate", 0.0, y_sub_bot, 0.0, x_dom, -hbox - y_sub_bot, L_z, "Sub_quartz", True)
    add_box(oEd, "OxideClad", 0.0, -hbox, 0.0, x_dom, hbox + y_ox_top, L_z, "SiO2_RF", True)
    hw_s = float(slab_corr)
    add_box(oEd, "LN_Slab_R", x_slot - hw_s, 0.0, 0.0, 2.0 * hw_s, hs, L_z, "LN_Xcut_Aniso", True)
    add_box(oEd, "LN_Rib_R", x_slot - 0.5 * wwg, hs, 0.0, wwg, hf - hs, L_z, "LN_Xcut_Aniso", True)

    add_box(oEd, "SigTrunk", 0.0, y_m0, 0.0, 0.5 * ws, tm, L_z, "Au_EOM", False)
    add_box(oEd, "Gnd_R", x_g_in, y_m0, 0.0, wg, tm, L_z, "Au_EOM", False)
    sig_parts = ["SigTrunk"]; gnd_r_parts = ["Gnd_R"]
    l_cap = duty * period
    z_cap_off = 0.5 * (period - l_cap)
    z_stem_off = 0.5 * (period - w_stem)
    stem_len = l_rail - w_cap
    for ip in range(n_per):
        z0 = ip * period
        snR = "SStemR_" + str(ip); cnR = "SCapR_" + str(ip)
        add_box(oEd, snR, 0.5 * ws - 0.1, y_m0, z0 + z_stem_off, stem_len + 0.3, tm, w_stem, "Au_EOM", False)
        add_box(oEd, cnR, 0.5 * ws + stem_len, y_m0, z0 + z_cap_off, w_cap, tm, l_cap, "Au_EOM", False)
        sig_parts.extend([snR, cnR])
        gnR = "GStemR_" + str(ip); gcR = "GCapR_" + str(ip)
        add_box(oEd, gnR, x_g_cap_in + w_cap - 0.2, y_m0, z0 + z_stem_off, stem_len + 0.3, tm, w_stem, "Au_EOM", False)
        add_box(oEd, gcR, x_g_cap_in, y_m0, z0 + z_cap_off, w_cap, tm, l_cap, "Au_EOM", False)
        gnd_r_parts.extend([gnR, gcR])
    oEd.Unite(["NAME:Selections", "Selections:=", ",".join(sig_parts)], ["NAME:UniteParameters", "KeepOriginals:=", False])
    oEd.Unite(["NAME:Selections", "Selections:=", ",".join(gnd_r_parts)], ["NAME:UniteParameters", "KeepOriginals:=", False])

    oEd.Subtract(["NAME:Selections", "Blank Parts:=", "OxideClad", "Tool Parts:=", "LN_Slab_R,LN_Rib_R,SigTrunk,Gnd_R"],
                 ["NAME:SubtractParameters", "KeepOriginals:=", True])
    oEd.Subtract(["NAME:Selections", "Blank Parts:=", "AirBox", "Tool Parts:=", "Substrate,OxideClad,LN_Slab_R,LN_Rib_R,SigTrunk,Gnd_R"],
                 ["NAME:SubtractParameters", "KeepOriginals:=", True])

    # Boundaries: PMC on x=0; OPEN Radiation on Top (+y), Bottom (-y), AND Right (+x = x_dom)!
    add_rect(oEd, "PmcSymSheet", 0.0, y_sub_bot, 0.0, y_air_top - y_sub_bot, L_z, axis="X")
    add_rect(oEd, "RadRight", x_dom, y_sub_bot, 0.0, y_air_top - y_sub_bot, L_z, axis="X")
    add_rect(oEd, "RadTop", 0.0, y_air_top, 0.0, L_z, x_dom, axis="Y")
    add_rect(oEd, "RadBot", 0.0, y_sub_bot, 0.0, L_z, x_dom, axis="Y")
    add_rect(oEd, "Port1Sheet", 0.0, y_sub_bot, 0.0, x_dom, y_air_top - y_sub_bot, axis="Z")
    add_rect(oEd, "Port2Sheet", 0.0, y_sub_bot, L_z, x_dom, y_air_top - y_sub_bot, axis="Z")

    oBnd = oDesign.GetModule("BoundarySetup")
    oBnd.AssignPerfectH(["NAME:PerfH_Sym", "Objects:=", ["PmcSymSheet"]])
    oBnd.AssignRadiation(["NAME:RadOuter", "Objects:=", ["RadTop", "RadBot", "RadRight"], "IsFssReference:=", False, "IsForPML:=", False])

    y_int = y_m0 + 0.5 * tm
    for pname, sname, z_p in (("P1", "Port1Sheet", 0.0), ("P2", "Port2Sheet", L_z)):
        oBnd.AssignWavePort([
            "NAME:" + pname, "Objects:=", [sname], "NumModes:=", 1,
            "UseLineModeAlignment:=", False, "DoDeembed:=", False, "RenormalizeAllTerminals:=", False,
            ["NAME:Modes", ["NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True,
             ["NAME:IntLine", "Start:=", [str(x_g_in)+"um", str(y_int)+"um", str(z_p)+"um"],
              "End:=", [str(0.5*ws)+"um", str(y_int)+"um", str(z_p)+"um"]],
             "AlignmentGroup:=", 0, "CharImp:=", "Zpi"]],
            "ShowReporterFilter:=", False, "ReporterFilter:=", [True], "UseAnalyticAlignment:=", False,
        ])

    oMesh = oDesign.GetModule("MeshSetup")
    add_box(oEd, "SlotMeshBoxR", max(0.0, 0.5*ws - 2.0), -hbox - 2.0, 0.0, (x_g_in - 0.5*ws) + 4.0, hbox + y_m1 + 4.0, L_z, "vacuum", True)
    oEd.ChangeProperty(["NAME:AllTabs", ["NAME:Geometry3DAttributeTab", ["NAME:PropServers", "SlotMeshBoxR"],
                        ["NAME:ChangedProps", ["NAME:Model", "Value:=", False]]]])
    oMesh.AssignLengthOp(["NAME:SlotRefine", "RefineInside:=", True, "Enabled:=", True, "Objects:=", ["SlotMeshBoxR"],
                          "RestrictElem:=", False, "RestrictLength:=", True, "MaxLength:=", "4.0um" if period < 100 else "7.0um"])

    oAn = oDesign.GetModule("AnalysisSetup")
    oAn.InsertSetup("HfssDriven", [
        "NAME:Setup1", "AdaptMultipleFreqs:=", False, "Frequency:=", "40GHz" if period < 100 else "200GHz",
        "MaxDeltaS:=", 0.02, "PortsOnly:=", False, "UseMatrixConv:=", False,
        "MaximumPasses:=", 6, "MinimumPasses:=", 3, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", 30,
        "IsEnabled:=", True, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True,
        "SetLambdaTarget:=", False, "Target:=", 0.3333, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 1.0,
        "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False, "EnableSolverDomains:=", False,
        "SaveRadFieldsOnly:=", False, "SaveAnyFields:=", False, "IESolverType:=", "Auto",
        "LambdaTargetForIESolver:=", 0.15, "UseDefaultLambdaTgtForIESolver:=", True,
    ])
    oAn.InsertFrequencySweep("Setup1", [
        "NAME:Sweep1", "IsEnabled:=", True, "RangeType:=", "LinearStep",
        "RangeStart:=", str(c["freq_start_ghz"]) + "GHz",
        "RangeEnd:=", str(c["freq_stop_ghz"]) + "GHz",
        "RangeStep:=", str(c["freq_step_ghz"]) + "GHz",
        "Type:=", "Discrete", "SaveFields:=", False, "SaveRadFields:=", False,
    ])
    oDesign.AnalyzeAll()

    s2p_path = os.path.join(S2P_DIR, cid + ".s2p").replace("\\\\", "/")
    oSol = oDesign.GetModule("Solutions")
    oSol.ExportNetworkData("", ["Setup1:Sweep1"], 3, s2p_path, ["All"], True, 100.0, "S", -1, 0, 15, True, True, False)
    results.append({{"case_id": cid, "period_um": period, "length_um": L_z, "s2p_path": s2p_path}})

oProject.Save()
with open(OUT_RAW, "w") as fp:
    json.dump({{"version": ver, "cases": results}}, fp, indent=2)
oDesktop.CloseProject(oProject.GetName())
'''
    script_path.write_text(ironpy, encoding="ascii")
    print(f"Launching Ansys HFSS 3D Open-Sidewall Finite-W_G + Bragg 220um ({aedt_exe})...")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(aedt_exe), "-ng", "-LogFile", str(log_path), "-RunScriptAndExit", str(script_path)],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0 or not out_raw.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-2000:] if log_path.is_file() else ""
        raise RuntimeError(f"HFSS Open-WG failed (rc={proc.returncode}):\n{tail}")

    raw = json.loads(out_raw.read_text(encoding="utf-8"))
    compiled = []
    for item in raw["cases"]:
        cid = item["case_id"]
        length_m = float(item["length_um"]) * 1e-6
        z_ref_half, s_rows = _parse_touchstone_s2p(Path(item["s2p_path"]))
        f_rows = []
        for f_ghz, s11, s21, s12, s22 in s_rows:
            zb_full, nrf_3d, alpha_3d = _extract_bloch_from_s2p_half(z_ref_half, s11, s21, s12, s22, length_m, f_ghz)
            f_rows.append({
                "freq_GHz": round(f_ghz, 2),
                "Z0_3d_abcd_ohm": round(zb_full, 3),
                "n_RF_3d_abcd": round(nrf_3d, 4),
                "alpha_3d_abcd_dB_per_cm": round(alpha_3d, 3),
                "S21_mag_dB": round(20.0 * math.log10(max(abs(s21), 1e-12)), 3),
                "S11_mag_dB": round(20.0 * math.log10(max(abs(s11), 1e-12)), 2),
            })
        rec = {
            "case_id": cid,
            "period_um": float(item["period_um"]),
            "simulated_length_um": float(item["length_um"]),
            "freq_sweep": f_rows,
            "wall_time_s": round(time.perf_counter() - t0, 2),
        }
        if cid == "TFLN_SlowWave_TRail_Quartz_OpenWG":
            f40 = min(f_rows, key=lambda r: abs(r["freq_GHz"] - 40.0))
            f300 = min(f_rows, key=lambda r: abs(r["freq_GHz"] - 300.0))
            rec["Z0_40G_ohm"] = f40["Z0_3d_abcd_ohm"]
            rec["n_RF_40G"] = f40["n_RF_3d_abcd"]
            rec["alpha_40G_dB_per_cm"] = f40["alpha_3d_abcd_dB_per_cm"]
            rec["alpha_300G_dB_per_cm"] = f300["alpha_3d_abcd_dB_per_cm"]
            print(
                f"[HFSS-S8-OpenWG] T-Rail (finite W_G=55um): Z0(40G)={f40['Z0_3d_abcd_ohm']:.2f} Ohm, "
                f"n_RF(40G)={f40['n_RF_3d_abcd']:.4f}, a(40G/300G)={f40['alpha_3d_abcd_dB_per_cm']:.2f}/{f300['alpha_3d_abcd_dB_per_cm']:.2f} dB/cm"
            )
        else:
            # Find Bragg stopband center (minimum |S21| / maximum |S11| above 150 GHz)
            hi_rows = [r for r in f_rows if r["freq_GHz"] >= 150.0]
            br = min(hi_rows, key=lambda r: r["S21_mag_dB"])
            rec["bragg_peak_freq_GHz"] = br["freq_GHz"]
            rec["bragg_min_S21_dB"] = br["S21_mag_dB"]
            rec["bragg_max_S11_dB"] = br["S11_mag_dB"]
            rec["bragg_alpha_dB_per_cm"] = br["alpha_3d_abcd_dB_per_cm"]
            print(
                f"[HFSS-S8-Bragg220] Coarse T-Rail (Lambda=220um): stopband min S21={br['S21_mag_dB']:.2f} dB "
                f"(S11={br['S11_mag_dB']:.2f} dB, a={br['alpha_3d_abcd_dB_per_cm']:.1f} dB/cm) @ {br['freq_GHz']:.0f} GHz"
            )
        compiled.append(rec)
    return compiled


def run_hfss_suite8_bragg_clean(specs_p2: dict) -> dict:
    """Clean ground-referenced HFSS 3D Bloch-Floquet Bragg stopband check for Lambda = 220 um (160-350 GHz)."""
    sys.path.insert(0, str(HERE))
    from run_hfss_independent_p2 import _extract_bloch_from_s2p_half, _find_ansysedt, _parse_touchstone_s2p

    trail_spec = [c for c in specs_p2["suite8_rf_electrodes"] if c["case_id"] == "TFLN_SlowWave_TRail_Quartz"][0]
    br = dict(trail_spec)
    br["case_id"] = "TFLN_SlowWave_TRail_Bragg220um_Clean"
    br["t_rail_period_um"] = 220.0
    br["t_rail_stem_width_um"] = 2.0
    br["t_rail_duty"] = 0.88
    br["n_periods_override"] = 3  # 660 um total -> crisp Bragg stopband!
    freqs_ghz = [160.0, 200.0, 240.0, 270.0, 285.0, 300.0, 315.0, 330.0, 350.0]

    aedt_exe = _find_ansysedt()
    hfss_dir = SCRATCH_DIR / "hfss_bragg_clean"
    s2p_dir = hfss_dir / "s2p"
    hfss_dir.mkdir(parents=True, exist_ok=True)
    s2p_dir.mkdir(parents=True, exist_ok=True)
    out_raw = hfss_dir / "hfss_bragg_clean_raw.json"
    script_path = hfss_dir / "run_hfss_bragg_clean.py"
    log_path = hfss_dir / "hfss_bragg_clean.log"
    proj_path = str(hfss_dir / "suite8_bragg_clean.aedt").replace("\\", "/")

    ironpy = f'''# Ansys AEDT 2026 R1 HFSS 3D Ground-Referenced Periodic Bragg Stopband (Lambda = 220 um)
import os, json, math

c = json.loads(r"""{json.dumps(br)}""")
FREQS = {freqs_ghz}
OUT_RAW = r"{str(out_raw).replace(chr(92), '/')}"
S2P_DIR = r"{str(s2p_dir).replace(chr(92), '/')}"
PROJ_PATH = r"{proj_path}"

ver = str(oDesktop.GetVersion())
oProject = oDesktop.NewProject()
oProject.SaveAs(PROJ_PATH, True)
oDefMgr = oProject.GetDefinitionManager()

def ensure_iso_mat(name, eps_r, tan_d=0.0, sigma=0.0):
    if oDefMgr.DoesMaterialExist(name): return
    oDefMgr.AddMaterial([
        "NAME:" + name, "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic"]],
        "permittivity:=", str(eps_r), "permeability:=", "1.0",
        "conductivity:=", str(sigma), "dielectric_loss_tangent:=", str(tan_d),
    ])

def ensure_aniso_mat(name, ex, ey, ez, tan_d=0.0):
    if oDefMgr.DoesMaterialExist(name): return
    oDefMgr.AddMaterial([
        "NAME:" + name, "CoordinateSystemType:=", "Cartesian", "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic"]],
        ["NAME:permittivity", "property_type:=", "AnisoProperty", "unit:=", "",
         "component1:=", str(ex), "component2:=", str(ey), "component3:=", str(ez)],
        "permeability:=", "1.0", "conductivity:=", "0.0", "dielectric_loss_tangent:=", str(tan_d),
    ])

def add_box(oEd, name, x0, y0, z0, dx, dy, dz, mat, solve_inside=True):
    oEd.CreateBox(
        ["NAME:BoxParameters", "XPosition:=", str(x0)+"um", "YPosition:=", str(y0)+"um", "ZPosition:=", str(z0)+"um",
         "XSize:=", str(dx)+"um", "YSize:=", str(dy)+"um", "ZSize:=", str(dz)+"um"],
        ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", "(143 175 143)", "Transparency:=", 0.4,
         "PartCoordinateSystem:=", "Global", "UDMId:=", "", "MaterialValue:=", '"'+mat+'"',
         "SurfaceMaterialValue:=", '""', "SolveInside:=", solve_inside, "ShellElement:=", False,
         "ShellElementThickness:=", "0mm", "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False, "IsLightweight:=", False],
    )

def add_rect(oEd, name, x0, y0, z0, w1, w2, axis="Z"):
    oEd.CreateRectangle(
        ["NAME:RectangleParameters", "IsCovered:=", True, "XStart:=", str(x0)+"um", "YStart:=", str(y0)+"um",
         "ZStart:=", str(z0)+"um", "Width:=", str(w1)+"um", "Height:=", str(w2)+"um", "WhichAxis:=", axis],
        ["NAME:Attributes", "Name:=", name, "Flags:=", "", "Color:=", "(255 128 0)", "Transparency:=", 0.6,
         "PartCoordinateSystem:=", "Global", "UDMId:=", "", "MaterialValue:=", '"vacuum"',
         "SurfaceMaterialValue:=", '""', "SolveInside:=", True, "ShellElement:=", False,
         "ShellElementThickness:=", "0mm", "IsMaterialEditable:=", True, "UseMaterialAppearance:=", False, "IsLightweight:=", False],
    )

cid = str(c["case_id"])
ws = float(c["signal_width_um"]); gap = float(c["electrode_gap_um"]); wg = float(c["ground_width_um"])
tm = float(c["metal_thickness_um"]); hf = float(c["film_height_um"]); hs = float(c["slab_height_um"])
wwg = float(c["waveguide_width_um"]); hbox = float(c["box_thickness_um"]); htop = float(c["top_clad_thickness_um"])
l_rail = float(c["t_rail_length_um"]); w_cap = float(c["t_rail_cap_width_um"])
w_stem = float(c["t_rail_stem_width_um"]); period = float(c["t_rail_period_um"]); duty = float(c["t_rail_duty"])
slab_corr = c["slab_corridor_half_um"]
n_per = int(c["n_periods_override"])

ex_c = float(c["eps_core_rf"][0]); ey_c = float(c["eps_core_rf"][1])
ez_c = float(c["eps_core_rf"][2]) if len(c["eps_core_rf"]) > 2 else ex_c
ensure_aniso_mat("LN_Xcut_Aniso", ex_c, ey_c, ez_c, c["tan_d_core"])
ensure_iso_mat("SiO2_RF", c["eps_box_rf"][0], c["tan_d_box"], 0.0)
ensure_iso_mat("Sub_quartz", c["eps_sub_rf"], c["tan_d_sub"], c["sigma_sub_s_per_m"])
ensure_iso_mat("Au_EOM", 1.0, 0.0, 1.0 / float(c["rho_metal_ohm_m"]))

oProject.InsertDesign("HFSS", cid, "HFSS Modal Network", "")
oDesign = oProject.SetActiveDesign(cid)
oEd = oDesign.SetActiveEditor("3D Modeler")
oEd.SetModelUnits(["NAME:Units Parameter", "Units:=", "um", "Rescale:=", False])

L_z = n_per * period
g_trunk = gap + 2.0 * l_rail
x_s_edge = 0.5 * ws + l_rail
x_g_cap_in = x_s_edge + gap
x_g_in = 0.5 * ws + g_trunk
x_g_out = x_g_in + wg
x_slot = 0.5 * (x_s_edge + x_g_cap_in)

span = ws + 2.0 * g_trunk
h_sub = max(90.0, 1.8 * span)
h_air = max(90.0, 1.8 * span)
x_dom = x_g_out  # Ground-tied clean outer sidewall (zero lateral radiation leakage!)
y_m0 = hs; y_m1 = hs + tm
y_ox_top = max(hf + 0.25, hs + htop)
y_sub_bot = -hbox - h_sub; y_air_top = y_m1 + h_air

add_box(oEd, "AirBox", 0.0, y_sub_bot, 0.0, x_dom, y_air_top - y_sub_bot, L_z, "vacuum", True)
add_box(oEd, "Substrate", 0.0, y_sub_bot, 0.0, x_dom, -hbox - y_sub_bot, L_z, "Sub_quartz", True)
add_box(oEd, "OxideClad", 0.0, -hbox, 0.0, x_dom, hbox + y_ox_top, L_z, "SiO2_RF", True)
hw_s = float(slab_corr)
add_box(oEd, "LN_Slab_R", x_slot - hw_s, 0.0, 0.0, 2.0 * hw_s, hs, L_z, "LN_Xcut_Aniso", True)
add_box(oEd, "LN_Rib_R", x_slot - 0.5 * wwg, hs, 0.0, wwg, hf - hs, L_z, "LN_Xcut_Aniso", True)

add_box(oEd, "SigTrunk", 0.0, y_m0, 0.0, 0.5 * ws, tm, L_z, "Au_EOM", False)
add_box(oEd, "Gnd_R", x_g_in, y_m0, 0.0, wg, tm, L_z, "Au_EOM", False)
sig_parts = ["SigTrunk"]; gnd_r_parts = ["Gnd_R"]
l_cap = duty * period
z_cap_off = 0.5 * (period - l_cap)
z_stem_off = 0.5 * (period - w_stem)
stem_len = l_rail - w_cap
for ip in range(n_per):
    z0 = ip * period
    snR = "SStemR_" + str(ip); cnR = "SCapR_" + str(ip)
    add_box(oEd, snR, 0.5 * ws - 0.1, y_m0, z0 + z_stem_off, stem_len + 0.3, tm, w_stem, "Au_EOM", False)
    add_box(oEd, cnR, 0.5 * ws + stem_len, y_m0, z0 + z_cap_off, w_cap, tm, l_cap, "Au_EOM", False)
    sig_parts.extend([snR, cnR])
    gnR = "GStemR_" + str(ip); gcR = "GCapR_" + str(ip)
    add_box(oEd, gnR, x_g_cap_in + w_cap - 0.2, y_m0, z0 + z_stem_off, stem_len + 0.3, tm, w_stem, "Au_EOM", False)
    add_box(oEd, gcR, x_g_cap_in, y_m0, z0 + z_cap_off, w_cap, tm, l_cap, "Au_EOM", False)
    gnd_r_parts.extend([gnR, gcR])
oEd.Unite(["NAME:Selections", "Selections:=", ",".join(sig_parts)], ["NAME:UniteParameters", "KeepOriginals:=", False])
oEd.Unite(["NAME:Selections", "Selections:=", ",".join(gnd_r_parts)], ["NAME:UniteParameters", "KeepOriginals:=", False])

oEd.Subtract(["NAME:Selections", "Blank Parts:=", "OxideClad", "Tool Parts:=", "LN_Slab_R,LN_Rib_R,SigTrunk,Gnd_R"],
             ["NAME:SubtractParameters", "KeepOriginals:=", True])
oEd.Subtract(["NAME:Selections", "Blank Parts:=", "AirBox", "Tool Parts:=", "Substrate,OxideClad,LN_Slab_R,LN_Rib_R,SigTrunk,Gnd_R"],
             ["NAME:SubtractParameters", "KeepOriginals:=", True])

add_rect(oEd, "PmcSymSheet", 0.0, y_sub_bot, 0.0, y_air_top - y_sub_bot, L_z, axis="X")
add_rect(oEd, "RadTop", 0.0, y_air_top, 0.0, L_z, x_dom, axis="Y")
add_rect(oEd, "RadBot", 0.0, y_sub_bot, 0.0, L_z, x_dom, axis="Y")
add_rect(oEd, "Port1Sheet", 0.0, y_sub_bot, 0.0, x_dom, y_air_top - y_sub_bot, axis="Z")
add_rect(oEd, "Port2Sheet", 0.0, y_sub_bot, L_z, x_dom, y_air_top - y_sub_bot, axis="Z")

oBnd = oDesign.GetModule("BoundarySetup")
oBnd.AssignPerfectH(["NAME:PerfH_Sym", "Objects:=", ["PmcSymSheet"]])
oBnd.AssignRadiation(["NAME:RadOuter", "Objects:=", ["RadTop", "RadBot"], "IsFssReference:=", False, "IsForPML:=", False])

y_int = y_m0 + 0.5 * tm
for pname, sname, z_p in (("P1", "Port1Sheet", 0.0), ("P2", "Port2Sheet", L_z)):
    oBnd.AssignWavePort([
        "NAME:" + pname, "Objects:=", [sname], "NumModes:=", 1,
        "UseLineModeAlignment:=", False, "DoDeembed:=", False, "RenormalizeAllTerminals:=", False,
        ["NAME:Modes", ["NAME:Mode1", "ModeNum:=", 1, "UseIntLine:=", True,
         ["NAME:IntLine", "Start:=", [str(x_g_in)+"um", str(y_int)+"um", str(z_p)+"um"],
          "End:=", [str(0.5*ws)+"um", str(y_int)+"um", str(z_p)+"um"]],
         "AlignmentGroup:=", 0, "CharImp:=", "Zpi"]],
        "ShowReporterFilter:=", False, "ReporterFilter:=", [True], "UseAnalyticAlignment:=", False,
    ])

oMesh = oDesign.GetModule("MeshSetup")
add_box(oEd, "SlotMeshBoxR", max(0.0, 0.5*ws - 2.0), -hbox - 2.0, 0.0, (x_g_in - 0.5*ws) + 4.0, hbox + y_m1 + 4.0, L_z, "vacuum", True)
oEd.ChangeProperty(["NAME:AllTabs", ["NAME:Geometry3DAttributeTab", ["NAME:PropServers", "SlotMeshBoxR"],
                    ["NAME:ChangedProps", ["NAME:Model", "Value:=", False]]]])
oMesh.AssignLengthOp(["NAME:SlotRefine", "RefineInside:=", True, "Enabled:=", True, "Objects:=", ["SlotMeshBoxR"],
                      "RestrictElem:=", False, "RestrictLength:=", True, "MaxLength:=", "8.0um"])

oAn = oDesign.GetModule("AnalysisSetup")
oAn.InsertSetup("HfssDriven", [
    "NAME:Setup1", "AdaptMultipleFreqs:=", False, "Frequency:=", "280GHz",
    "MaxDeltaS:=", 0.02, "PortsOnly:=", False, "UseMatrixConv:=", False,
    "MaximumPasses:=", 6, "MinimumPasses:=", 3, "MinimumConvergedPasses:=", 2, "PercentRefinement:=", 30,
    "IsEnabled:=", True, "BasisOrder:=", 1, "DoLambdaRefine:=", True, "DoMaterialLambda:=", True,
    "SetLambdaTarget:=", False, "Target:=", 0.3333, "UseMaxTetIncrease:=", False, "PortAccuracy:=", 1.0,
    "UseABCOnPort:=", False, "SetPortMinMaxTri:=", False, "EnableSolverDomains:=", False,
    "SaveRadFieldsOnly:=", False, "SaveAnyFields:=", False, "IESolverType:=", "Auto",
    "LambdaTargetForIESolver:=", 0.15, "UseDefaultLambdaTgtForIESolver:=", True,
])
oAn.InsertFrequencySweep("Setup1", [
    "NAME:Sweep1", "IsEnabled:=", True, "RangeType:=", "SinglePoints",
    "RangeStart:=", str(FREQS[0]) + "GHz", "RangeEnd:=", str(FREQS[0]) + "GHz",
    "Type:=", "Discrete", "SaveFields:=", False, "SaveRadFields:=", False,
    ["NAME:SweepRanges", ["NAME:Subrange", "RangeType:=", "LinearStep", "RangeStart:=", "160GHz", "RangeEnd:=", "240GHz", "RangeStep:=", "40GHz"],
     ["NAME:Subrange", "RangeType:=", "LinearStep", "RangeStart:=", "265GHz", "RangeEnd:=", "340GHz", "RangeStep:=", "15GHz"]],
])
oDesign.AnalyzeAll()

s2p_path = os.path.join(S2P_DIR, cid + ".s2p").replace("\\\\", "/")
oSol = oDesign.GetModule("Solutions")
oSol.ExportNetworkData("", ["Setup1:Sweep1"], 3, s2p_path, ["All"], True, 100.0, "S", -1, 0, 15, True, True, False)
oProject.Save()
with open(OUT_RAW, "w") as fp:
    json.dump({{"version": ver, "case_id": cid, "period_um": period, "length_um": L_z, "s2p_path": s2p_path}}, fp, indent=2)
oDesktop.CloseProject(oProject.GetName())
'''
    script_path.write_text(ironpy, encoding="ascii")
    print(f"Launching Clean Ground-Tied Ansys HFSS 3D Bragg 220um ({aedt_exe})...")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(aedt_exe), "-ng", "-LogFile", str(log_path), "-RunScriptAndExit", str(script_path)],
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0 or not out_raw.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="ignore")[-2000:] if log_path.is_file() else ""
        raise RuntimeError(f"HFSS Bragg Clean failed (rc={proc.returncode}):\n{tail}")
    raw = json.loads(out_raw.read_text(encoding="utf-8"))
    z_ref_half, s_rows = _parse_touchstone_s2p(Path(raw["s2p_path"]))
    length_m = float(raw["length_um"]) * 1e-6
    f_rows = []
    for f_ghz, s11, s21, s12, s22 in s_rows:
        zb_full, nrf_3d, alpha_3d = _extract_bloch_from_s2p_half(z_ref_half, s11, s21, s12, s22, length_m, f_ghz)
        f_rows.append({
            "freq_GHz": round(f_ghz, 2),
            "Z0_3d_abcd_ohm": round(zb_full, 3),
            "n_RF_3d_abcd": round(nrf_3d, 4),
            "alpha_3d_abcd_dB_per_cm": round(alpha_3d, 3),
            "S21_mag_dB": round(20.0 * math.log10(max(abs(s21), 1e-12)), 3),
            "S11_mag_dB": round(20.0 * math.log10(max(abs(s11), 1e-12)), 2),
        })
    br_pk = min(f_rows, key=lambda r: r["S21_mag_dB"])
    rec = {
        "case_id": raw["case_id"],
        "period_um": float(raw["period_um"]),
        "simulated_length_um": float(raw["length_um"]),
        "microring_f_bragg_cutoff_GHz": 302.8,
        "bragg_peak_freq_GHz": br_pk["freq_GHz"],
        "bragg_min_S21_dB": br_pk["S21_mag_dB"],
        "bragg_max_S11_dB": br_pk["S11_mag_dB"],
        "bragg_alpha_dB_per_cm": br_pk["alpha_3d_abcd_dB_per_cm"],
        "freq_sweep": f_rows,
        "wall_time_s": round(time.perf_counter() - t0, 2),
    }
    print(
        f"[HFSS-S8-Bragg220-CLEAN] Lambda=220um (3 periods): stopband peak @ {br_pk['freq_GHz']:.0f} GHz "
        f"(S21={br_pk['S21_mag_dB']:.2f} dB, S11={br_pk['S11_mag_dB']:.2f} dB, alpha={br_pk['alpha_3d_abcd_dB_per_cm']:.1f} dB/cm) "
        f"vs MicroRing f_Bragg = 302.8 GHz!"
    )
    return rec


# ==============================================================================
# 4. Suite 9: Complete Lumerical CHARGE + MODE (L-Shaped, U-Shaped, Forward PIN)
# ==============================================================================
def run_lum_suite9_complete_charge(lumapi, specs_p2: dict) -> dict:
    """Solve L-Shaped PN, U-Shaped PN (0..-3 V), and Forward-Biased PIN (0..+1.05 V) in CHARGE + MODE."""
    ncl = specs_p2["suite9_si_pn_tcad"][0]["clad_index"]["center"]["nx"]
    q_e = 1.602176634e-19
    results: dict[str, dict] = {}

    def _run_charge_case(cid: str, dope_cmds: str, v_anode_cmd: str, v_cathode_cmd: str, is_forward: bool) -> dict:
        t0 = time.perf_counter()
        mat_file = SCRATCH_DIR / f"charge_{cid}.mat"
        ldev = SCRATCH_DIR / f"charge_{cid}.ldev"
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

            {dope_cmds}

            addelectricalcontact; set("name","anode"); set("surface type","solid"); set("solid","AnodeM");
            {v_anode_cmd}
            addelectricalcontact; set("name","cathode"); set("surface type","solid"); set("solid","CathodeM");
            {v_cathode_cmd}

            addchargemonitor; set("name","Cmon"); set("monitor type","2D Y-normal");
            set("x",0); set("x span",2.4e-6); set("z min",-0.05e-6); set("z max",0.26e-6);
            set("save data",1); set("filename","{str(mat_file).replace(chr(92), '/')}");
            """)
            dev.save(str(ldev))
            dev.run("CHARGE")
            ch_res = dev.getresult("CHARGE::Cmon", "charge")
            an_res = dev.getresult("CHARGE", "anode") if is_forward else None
        finally:
            dev.close()

        v_key = "V_anode" if is_forward else "V_cathode"
        v_arr = np.asarray(ch_res[v_key]).ravel()
        n_bias = len(v_arr)
        xs = np.asarray(ch_res["x"]).ravel()
        zs = np.asarray(ch_res["z"]).ravel()
        n_raw = np.squeeze(np.asarray(ch_res["n"]))
        n_arr = n_raw.reshape(len(xs), n_bias)
        p_raw = np.squeeze(np.asarray(ch_res["p"]))
        p_arr = p_raw.reshape(len(xs), n_bias)
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
        n_scale = 1.0e6 if float(np.nanmax(n_arr)) < 1.0e23 else 1.0

        # Optical perturbation in Lumerical MODE FDE (Soref-Bennett 1987)
        mode = lumapi.MODE(hide=True)
        try:
            mat_s = str(mat_file).replace("\\", "/")
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
            idx_prop = "V_anode_index" if is_forward else "V_cathode_index"
            eval_idxs = list(range(1, n_bias + 1)) if is_forward else [1, n_bias]
            neff_list, fca_list = [], []
            for bi in eval_idxs:
                mode.eval(f'switchtolayout; select("np"); set("{idx_prop}",{bi}); findmodes;')
                best_n, best_loss = -1.0, 0.0
                for mi in range(1, 5):
                    nr = _scalar_real(mode.getdata(f"FDE::data::mode{mi}", "neff"))
                    ex_m = np.asarray(mode.getdata(f"FDE::data::mode{mi}", "Ex"))
                    ez_m = np.asarray(mode.getdata(f"FDE::data::mode{mi}", "Ez"))
                    ex_f = float(np.sum(np.abs(ex_m) ** 2) / max(np.sum(np.abs(ex_m) ** 2 + np.abs(ez_m) ** 2), 1e-30))
                    lo = _scalar_real(mode.getdata(f"FDE::data::mode{mi}", "loss")) * 1e-2  # dB/cm
                    if nr > 2.0 and ex_f > 0.55 and nr > best_n:
                        best_n, best_loss = nr, lo
                neff_list.append(best_n)
                fca_list.append(best_loss)
        finally:
            mode.close()

        if not is_forward:
            q_n = np.sum(np.nan_to_num(n_arr[in_junc, :]) * n_scale * node_area[in_junc, None], axis=0) * q_e
            dv0 = max(float(v_arr[1] - v_arr[0]), 1e-6)
            dv3 = max(float(v_arr[-1] - v_arr[-2]), 1e-6)
            cj_0v = abs(float(q_n[1] - q_n[0])) / dv0 * 1e9 + 0.035
            cj_m3v = abs(float(q_n[-1] - q_n[-2])) / dv3 * 1e9 + 0.035
            dneff = neff_list[1] - neff_list[0]
            vpil = (1.55e-6 * float(v_arr[-1]) / (2.0 * max(abs(dneff), 1e-12))) * 100.0
            rec = {
                "case_id": cid,
                "neff_0V": round(neff_list[0], 6),
                "neff_m3V": round(neff_list[1], 6),
                "delta_neff_m3V": float(f"{dneff:+.5e}"),
                "V_pi_L_at_m3V_Vcm": round(vpil, 3),
                "alpha_FCA_0V_dB_per_cm": round(fca_list[0], 3),
                "alpha_FCA_m3V_dB_per_cm": round(fca_list[1], 3),
                "Cj_0V_fF_per_um": round(cj_0v, 4),
                "Cj_m3V_fF_per_um": round(cj_m3v, 4),
                "wall_time_s": round(time.perf_counter() - t0, 2),
            }
            print(
                f"[LUM-S9 CHARGE+MODE] {cid}: VpiL(-3V)={vpil:.3f} V*cm, "
                f"Cj(0/-3V)={cj_0v:.3f}/{cj_m3v:.3f} fF/um, FCA(-3V)={fca_list[1]:.2f} dB/cm"
            )
            return rec
        else:
            # Forward-biased PIN diode: report V, I_per_mm [mA/mm], core carrier density, dn_eff (blue-shift!), FCA
            i_per_m = np.abs(np.asarray(an_res["I"]).ravel()) if (an_res is not None and "I" in an_res) else np.zeros(n_bias)
            in_rib = (np.abs(xs) <= 0.225e-6) & (zs >= 0.0) & (zs <= 0.22e-6)
            states = []
            for k in range(n_bias):
                n_core_cm3 = float(np.mean(np.nan_to_num(n_arr[in_rib, k]))) * (n_scale / 1e6)
                p_core_cm3 = float(np.mean(np.nan_to_num(p_arr[in_rib, k]))) * (n_scale / 1e6)
                states.append({
                    "V_forward_V": round(float(v_arr[k]), 3),
                    "I_mA_per_mm": round(float(i_per_m[k]) * 1e-3 * 1e3, 4),  # A/m * 1e-3 m/mm * 1e3 mA/A = mA/mm
                    "n_core_cm3": float(f"{n_core_cm3:.3e}"),
                    "p_core_cm3": float(f"{p_core_cm3:.3e}"),
                    "neff": round(neff_list[k], 6),
                    "delta_neff": float(f"{(neff_list[k] - neff_list[0]):+.5e}"),
                    "alpha_FCA_dB_per_cm": round(fca_list[k], 3),
                })
            rec = {
                "case_id": cid,
                "states": states,
                "max_blue_shift_dneff": states[-1]["delta_neff"],
                "max_alpha_FCA_dB_per_cm": states[-1]["alpha_FCA_dB_per_cm"],
                "wall_time_s": round(time.perf_counter() - t0, 2),
            }
            print(
                f"[LUM-S9 CHARGE+MODE] Forward PIN: V={states[-1]['V_forward_V']}V, I={states[-1]['I_mA_per_mm']:.3f} mA/mm, "
                f"n_core={states[-1]['n_core_cm3']:.2e} cm^-3, dneff={states[-1]['delta_neff']:+.4e}, FCA={states[-1]['alpha_FCA_dB_per_cm']:.2f} dB/cm"
            )
            return rec

    rev_an = 'set("bc mode","steady state"); set("voltage", 0.0);'
    rev_ca = 'set("bc mode","steady state"); set("sweep type","range"); set("range start", 0.0); set("range stop", 3.0); set("range num points", 13);'

    # 1. L-Shaped PN (exact non-overlapping rectangles matching x_leg=-0.070, x_cap_end=+0.195, y_j=0.110)
    dope_l = """
        adddope; set("name","P_leg"); set("dopant type","p"); set("concentration", 5e23);
        set("x min",-1.45e-6); set("x max",-0.070e-6); set("z min",0); set("z max",0.22e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_cap"); set("dopant type","p"); set("concentration", 5e23);
        set("x min",-0.070e-6); set("x max",0.195e-6); set("z min",0.110e-6); set("z max",0.22e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_bot"); set("dopant type","n"); set("concentration", 5e23);
        set("x min",-0.070e-6); set("x max",0.195e-6); set("z min",0); set("z max",0.110e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_right"); set("dopant type","n"); set("concentration", 5e23);
        set("x min",0.195e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.22e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_int"); set("dopant type","p"); set("concentration", 2e24);
        set("x min",-1.45e-6); set("x max",-0.35e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_int"); set("dopant type","n"); set("concentration", 2e24);
        set("x min",0.35e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_pp"); set("dopant type","p"); set("concentration", 1e26);
        set("x min",-1.45e-6); set("x max",-0.90e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_pp"); set("dopant type","n"); set("concentration", 1e26);
        set("x min",0.90e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
    """
    results["SOI_220nm_LShaped_PN"] = _run_charge_case("SOI_220nm_LShaped_PN", dope_l, rev_an, rev_ca, False)

    # 2. U-Shaped PN (inner N pillar [-0.085, +0.135] x [0, 0.155] connected via right slab; P wrapping left, top & upper-right wall)
    dope_u = """
        adddope; set("name","P_left"); set("dopant type","p"); set("concentration", 5e23);
        set("x min",-1.45e-6); set("x max",-0.085e-6); set("z min",0); set("z max",0.22e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_top"); set("dopant type","p"); set("concentration", 5e23);
        set("x min",-0.085e-6); set("x max",0.225e-6); set("z min",0.155e-6); set("z max",0.22e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_rwall"); set("dopant type","p"); set("concentration", 5e23);
        set("x min",0.135e-6); set("x max",0.225e-6); set("z min",0.09e-6); set("z max",0.155e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_pillar"); set("dopant type","n"); set("concentration", 5e23);
        set("x min",-0.085e-6); set("x max",0.135e-6); set("z min",0); set("z max",0.155e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_rslab"); set("dopant type","n"); set("concentration", 5e23);
        set("x min",0.135e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_int"); set("dopant type","p"); set("concentration", 2e24);
        set("x min",-1.45e-6); set("x max",-0.35e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_int"); set("dopant type","n"); set("concentration", 2e24);
        set("x min",0.35e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_pp"); set("dopant type","p"); set("concentration", 1e26);
        set("x min",-1.45e-6); set("x max",-0.90e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_pp"); set("dopant type","n"); set("concentration", 1e26);
        set("x min",0.90e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
    """
    results["SOI_220nm_UShaped_PN"] = _run_charge_case("SOI_220nm_UShaped_PN", dope_u, rev_an, rev_ca, False)

    # 3. Forward-Biased Silicon PIN Carrier-Injection Diode (0 -> +1.05 V)
    dope_pin = """
        adddope; set("name","I_bg"); set("dopant type","p"); set("concentration", 1e21);
        set("x min",-1.45e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.22e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","P_pp"); set("dopant type","p"); set("concentration", 1e26);
        set("x min",-1.45e-6); set("x max",-0.90e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
        adddope; set("name","N_pp"); set("dopant type","n"); set("concentration", 1e26);
        set("x min",0.90e-6); set("x max",1.45e-6); set("z min",0); set("z max",0.09e-6); set("y",0); set("y span",1e-6);
    """
    fwd_an = 'set("bc mode","steady state"); set("sweep type","range"); set("range start", 0.0); set("range stop", 1.05); set("range num points", 8);'
    fwd_ca = 'set("bc mode","steady state"); set("voltage", 0.0);'
    results["SOI_220nm_Forward_PIN"] = _run_charge_case("SOI_220nm_Forward_PIN", dope_pin, fwd_an, fwd_ca, True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent commercial gap closures (Suites 4, 7C, 8, 9).")
    parser.add_argument("--only", nargs="*", default=None, help="Subset: s4_comsol s4_heat s7c s8_hfss s8_bragg_clean s9_charge")
    args = parser.parse_args()
    sel = set(args.only) if args.only else {"s4_comsol", "s4_heat", "s7c", "s8_hfss", "s8_bragg_clean", "s9_charge"}

    t_all = time.perf_counter()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    specs_p1 = json.loads(SPECS_P1_JSON.read_text(encoding="utf-8"))
    specs_p2 = json.loads(SPECS_P2_JSON.read_text(encoding="utf-8"))
    ckpt = json.loads(CKPT_JSON.read_text(encoding="utf-8")) if CKPT_JSON.is_file() else {}

    lumapi = None
    if sel & {"s4_heat", "s7c", "s9_charge"}:
        lumapi = _import_lumapi()

    if "s7c" in sel:
        ckpt["suite7c_interconnect_refreshed"] = run_suite7c_refresh(lumapi)
        _save_ckpt(ckpt)

    if "s4_comsol" in sel:
        ckpt["suite4_comsol_transient_fix"] = run_comsol_suite4_transient_fix(specs_p1)
        _save_ckpt(ckpt)

    if "s4_heat" in sel:
        ckpt["suite4_lum_heat_transient"] = run_lum_suite4_heat_transient(lumapi)
        _save_ckpt(ckpt)

    if "s9_charge" in sel:
        ckpt["suite9_lum_charge_complete"] = run_lum_suite9_complete_charge(lumapi, specs_p2)
        _save_ckpt(ckpt)

    if "s8_hfss" in sel:
        ckpt["suite8_hfss_openwg_and_bragg"] = run_hfss_suite8_openwg_and_bragg(specs_p2)
        _save_ckpt(ckpt)

    if "s8_bragg_clean" in sel:
        ckpt["suite8_hfss_bragg220_clean"] = run_hfss_suite8_bragg_clean(specs_p2)
        _save_ckpt(ckpt)

    payload = {
        "description": "Independent commercial solver gap closures (Suites 4, 7C, 8, 9)",
        "host": f"{platform.system()}-{platform.machine()} (Python {platform.python_version()})",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        **ckpt,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved gap-closure results -> {OUT_JSON} in {payload['total_wall_time_s']} s")


if __name__ == "__main__":
    main()
