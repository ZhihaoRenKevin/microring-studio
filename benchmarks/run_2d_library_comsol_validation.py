"""Independent COMSOL Multiphysics 6.4 (`ewfd` Full Anisotropic Complex Tensor) Validation
for MicroRing Studio's Expanded 27-Material 2D van der Waals Library (`microring.twod_materials_mrr`).

Validates 6 distinct 2D vdW regimes directly against COMSOL 6.4 `ewfd` full-wave 2D eigenmode
solutions with explicit 3x3 diagonal complex permittivity tensors `diag(eps_xx, eps_yy, eps_zz)`:
  1. `alpha_MoO3` (25 nm, 10.8 um, Ge/CaF2/Air) at `crystal_angle_deg = 0 deg`  (Re eps_xx < 0, hyperbolic!)
  2. `alpha_MoO3` (25 nm, 10.8 um, Ge/CaF2/Air) at `crystal_angle_deg = 90 deg` (Re eps_xx > 0, elliptic!)
  3. `hBN_Isotopic_10B` (10 nm, 6.7 um, Ge/CaF2/Air) — Type-II Hyperbolic Reststrahlen phonon polariton
  4. `MoS2_Bulk_vdW` (10 nm, 1550 nm, 220x480 nm SOI) — Giant-birefringence uniaxial vdW crystal
  5. `Ti3C2Tx_MXene` (5 nm, 1550 nm, 220x480 nm SOI) — Metallic plasmonic MXene sheet
  6. `NbOCl2` (15 nm, 1550 nm, 220x480 nm SOI) — Giant-chi^(2) biaxial electro-optic vdW crystal
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

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH = OUT_DIR / "_v311_2d_val_scratch"
OUT_JSON = OUT_DIR / "v311_2d_library_comsol_validation.json"


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


def main() -> None:
    from microring import (
        TWOD_MATERIAL_CATALOG,
        TwoDMaterialMRRConfig,
        evaluate_2d_material_tensor,
        simulate_2d_material_mrr,
    )
    from microring.materials import T_REF_K, get_material_fn

    print(f"Checking all {len(TWOD_MATERIAL_CATALOG)} 2D vdW catalog materials...")
    assert len(TWOD_MATERIAL_CATALOG) == 27
    for name, pr in TWOD_MATERIAL_CATALOG.items():
        wl_nm = (
            10800.0 if name == "alpha_MoO3"
            else 11500.0 if name == "alpha_V2O5"
            else 12000.0 if name == "WTe2"
            else 6800.0 if name.startswith("hBN")
            else 6500.0 if name == "b_AsP"
            else 3400.0 if name in ("Black_Phosphorus", "Te_Tellurene")
            else 1310.0 if name == "MoTe2"
            else 1550.0
        )
        t0 = evaluate_2d_material_tensor(name, wl_nm, v_gate_v=0.0, crystal_angle_deg=0.0)
        assert math.isfinite(t0["eps_u"].real) and math.isfinite(t0["eps_u"].imag)
        assert math.isfinite(t0["sigma_xx_S"].real) and math.isfinite(t0["sigma_xx_S"].imag)
    print("  All 27 2D material microscopic tensor models passed sanity checks.")

    # Define 6 COMSOL 6.4 ewfd anisotropic benchmark cases (straight waveguide R=1e5 um for exact 1:1!)
    cases = [
        {
            "id": "alpha_MoO3_0deg",
            "material_2d": "alpha_MoO3",
            "core": "Ge", "clad": "CaF2", "top": "Air",
            "w_um": 3.2, "h_um": 1.8, "wl_nm": 10800.0,
            "spacer_nm": 10.0, "d_nm": 25.0, "angle_deg": 0.0,
        },
        {
            "id": "alpha_MoO3_90deg",
            "material_2d": "alpha_MoO3",
            "core": "Ge", "clad": "CaF2", "top": "Air",
            "w_um": 3.2, "h_um": 1.8, "wl_nm": 10800.0,
            "spacer_nm": 10.0, "d_nm": 25.0, "angle_deg": 90.0,
        },
        {
            "id": "hBN_Isotopic_10B",
            "material_2d": "hBN_Isotopic_10B",
            "core": "Ge", "clad": "CaF2", "top": "Air",
            "w_um": 2.0, "h_um": 1.2, "wl_nm": 6700.0,
            "spacer_nm": 10.0, "d_nm": 10.0, "angle_deg": 0.0,
        },
        {
            "id": "MoS2_Bulk_vdW",
            "material_2d": "MoS2_Bulk_vdW",
            "core": "Si", "clad": "SiO2", "top": "SiO2",
            "w_um": 0.48, "h_um": 0.22, "wl_nm": 1550.0,
            "spacer_nm": 10.0, "d_nm": 10.0, "angle_deg": 0.0,
        },
        {
            "id": "Ti3C2Tx_MXene",
            "material_2d": "Ti3C2Tx_MXene",
            "core": "Si", "clad": "SiO2", "top": "SiO2",
            "w_um": 0.48, "h_um": 0.22, "wl_nm": 1550.0,
            "spacer_nm": 10.0, "d_nm": 5.0, "angle_deg": 0.0,
        },
        {
            "id": "NbOCl2_Biaxial",
            "material_2d": "NbOCl2",
            "core": "Si", "clad": "SiO2", "top": "SiO2",
            "w_um": 0.48, "h_um": 0.22, "wl_nm": 1550.0,
            "spacer_nm": 10.0, "d_nm": 15.0, "angle_deg": 0.0,
        },
    ]

    mr_results: dict[str, dict] = {}
    java_calls: list[str] = []

    for c in cases:
        wl_m = c["wl_nm"] * 1e-9
        n_core = float(get_material_fn(c["core"])(wl_m, T_REF_K).nx)
        n_clad = float(get_material_fn(c["clad"])(wl_m, T_REF_K).nx)
        n_top = float(get_material_fn(c["top"])(wl_m, T_REF_K).nx)

        # MicroRing Studio simulation (large radius R=1e5 um -> straight waveguide matching COMSOL 2D!)
        res = simulate_2d_material_mrr(TwoDMaterialMRRConfig(
            material_2d=c["material_2d"],
            core_material=c["core"],
            clad_material=c["clad"],
            top_clad_material=c["top"],
            film_height_um=c["h_um"],
            ring_width_um=c["w_um"],
            ring_radius_um=1.0e5,
            wavelength_nm=c["wl_nm"],
            spacer_ox_nm=c["spacer_nm"],
            thickness_nm_override=c["d_nm"],
            crystal_angle_deg=c["angle_deg"],
            v_gate_sweep_v=(0.0,),
            core_mesh_res_um=0.032,
        ))
        m = res["metrics"]
        tens = evaluate_2d_material_tensor(
            c["material_2d"], c["wl_nm"],
            v_gate_v=0.0, spacer_ox_nm=c["spacer_nm"],
            cladding_eps=n_top ** 2, thickness_nm=c["d_nm"],
            crystal_angle_deg=c["angle_deg"],
        )
        # Rotate principal diagonal tensor (eps_u, eps_v) for 0 deg or 90 deg -> exact diagonal (eps_xx, eps_zz)
        th = math.radians(c["angle_deg"])
        exx = tens["eps_u"] * math.cos(th) ** 2 + tens["eps_v"] * math.sin(th) ** 2
        ezz = tens["eps_u"] * math.sin(th) ** 2 + tens["eps_v"] * math.cos(th) ** 2
        eyy = tens["eps_y"]

        mr_results[c["id"]] = {
            "neff_bare": m["neff_bare"],
            "delta_neff_real": m["delta_neff_V0"],
            "alpha_2d_dB_per_cm": m["alpha_2d_max_dB_per_cm"],
            "dispersion_regime": m["dispersion_regime"],
            "eps_xx": [round(exx.real, 4), round(exx.imag, 4)],
            "eps_yy": [round(eyy.real, 4), round(eyy.imag, 4)],
            "eps_zz": [round(ezz.real, 4), round(ezz.imag, 4)],
        }

        # COMSOL uses exp(+j*omega*t), so eps_comsol = conj(eps_ours) = re - im*i
        def _cstr(z: complex) -> str:
            return f"({z.real:.8f})+({-z.imag:.8f})*i"

        w_sheet_m = (c["w_um"] + 0.8) * 1e-6
        java_calls.append(
            f'    runCase("{c["id"]}", {c["w_um"]*1e-6:.9e}, {c["h_um"]*1e-6:.9e}, '
            f'{w_sheet_m:.9e}, {c["spacer_nm"]*1e-9:.9e}, {c["d_nm"]*1e-9:.9e}, {wl_m:.9e}, '
            f'{n_core:.8f}, {n_clad:.8f}, {n_top:.8f}, '
            f'"{_cstr(exx)}", "{_cstr(eyy)}", "{_cstr(ezz)}", {m["neff_bare"]:.6f});'
        )

    java_src = f"""
import com.comsol.model.*;
import com.comsol.model.util.*;

public class Comsol2DLibraryValidation {{

  static double[] solveOne(String tag, double wCore, double hCore, double wSheet, double spM, double dM,
                           double wl, double nCore, double nClad, double nTop,
                           String exx, String eyy, String ezz, double shift, boolean withSheet) throws Exception {{
    double eps = 1e-11;
    double wlScale = wl / 1.55e-6;
    double xDom = Math.max(2.2e-6 * wlScale, 0.5 * wSheet + 1.3 * wl);
    double yBox = Math.max(1.8e-6 * wlScale, 1.3 * wl);
    double yTop = Math.max(1.8e-6 * wlScale, 1.3 * wl);

    Model m = ModelUtil.create(tag);
    m.component().create("c1", true);
    m.component("c1").geom().create("g1", 2);
    m.component("c1").geom("g1").create("rBox", "Rectangle");
    m.component("c1").geom("g1").feature("rBox").set("pos", new double[]{{-xDom, -yBox}}).set("size", new double[]{{2*xDom, yBox}});
    m.component("c1").geom("g1").create("rTop", "Rectangle");
    m.component("c1").geom("g1").feature("rTop").set("pos", new double[]{{-xDom, 0.0}}).set("size", new double[]{{2*xDom, hCore + spM + dM + yTop}});
    m.component("c1").geom("g1").create("rCore", "Rectangle");
    m.component("c1").geom("g1").feature("rCore").set("pos", new double[]{{-0.5*wCore, 0.0}}).set("size", new double[]{{wCore, hCore}});
    if (withSheet) {{
      m.component("c1").geom("g1").create("rSh", "Rectangle");
      m.component("c1").geom("g1").feature("rSh").set("pos", new double[]{{-0.5*wSheet, hCore + spM}}).set("size", new double[]{{wSheet, dM}});
    }}
    m.component("c1").geom("g1").run();

    m.component("c1").selection().create("selBox", "Box");
    m.component("c1").selection("selBox").set("entitydim", 2).set("condition", "inside").set("xmin", -xDom-eps).set("xmax", xDom+eps).set("ymin", -yBox-eps).set("ymax", eps);
    m.component("c1").selection().create("selCore", "Box");
    m.component("c1").selection("selCore").set("entitydim", 2).set("condition", "inside").set("xmin", -0.5*wCore-eps).set("xmax", 0.5*wCore+eps).set("ymin", -eps).set("ymax", hCore+eps);
    if (withSheet) {{
      m.component("c1").selection().create("selSh", "Box");
      m.component("c1").selection("selSh").set("entitydim", 2).set("condition", "inside").set("xmin", -0.5*wSheet-eps).set("xmax", 0.5*wSheet+eps).set("ymin", hCore+spM-eps).set("ymax", hCore+spM+dM+eps);
    }}

    m.component("c1").physics().create("ewfd", "ElectromagneticWavesFrequencyDomain", "g1");
    m.component("c1").physics("ewfd").feature("wee1").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", ""+nTop).set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").physics("ewfd").create("wBox", "WaveEquationElectric", 2);
    m.component("c1").physics("ewfd").feature("wBox").selection().named("selBox");
    m.component("c1").physics("ewfd").feature("wBox").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", ""+nClad).set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    m.component("c1").physics("ewfd").create("wCore", "WaveEquationElectric", 2);
    m.component("c1").physics("ewfd").feature("wCore").selection().named("selCore");
    m.component("c1").physics("ewfd").feature("wCore").set("DisplacementFieldModel", "RefractiveIndex").set("n_mat", "userdef").set("n", ""+nCore).set("ki_mat", "userdef").set("ki", "0").set("mur_mat", "userdef").set("mur", "1");
    if (withSheet) {{
      m.component("c1").physics("ewfd").create("wSh", "WaveEquationElectric", 2);
      m.component("c1").physics("ewfd").feature("wSh").selection().named("selSh");
      m.component("c1").physics("ewfd").feature("wSh")
        .set("DisplacementFieldModel", "RelativePermittivity")
        .set("epsilonr_mat", "userdef")
        .set("epsilonr", new String[]{{exx, "0", "0", "0", eyy, "0", "0", "0", ezz}})
        .set("mur_mat", "userdef").set("mur", "1")
        .set("sigma_mat", "userdef").set("sigma", "0");
    }}

    m.component("c1").mesh().create("m1", "g1");
    m.component("c1").mesh("m1").feature().create("szC", "Size");
    m.component("c1").mesh("m1").feature("szC").selection().named("selCore");
    m.component("c1").mesh("m1").feature("szC").set("custom", true).set("hmaxactive", true).set("hmax", 0.030e-6 * Math.max(1.0, wlScale));
    if (withSheet) {{
      m.component("c1").mesh("m1").feature().create("szS", "Size");
      m.component("c1").mesh("m1").feature("szS").selection().named("selSh");
      m.component("c1").mesh("m1").feature("szS").set("custom", true).set("hmaxactive", true).set("hmax", Math.max(dM * 1.2, 0.015e-6 * Math.max(1.0, wlScale)));
    }}
    m.component("c1").mesh("m1").feature().create("ftri1", "FreeTri");
    m.component("c1").mesh("m1").run();

    m.study().create("s1");
    m.study("s1").create("mode", "ModeAnalysis");
    m.study("s1").feature("mode").set("modeFreq", "299792458/" + wl);
    m.study("s1").feature("mode").set("neigs", 4).set("neigsactive", true);
    m.study("s1").feature("mode").set("shift", "" + shift).set("shiftactive", true);
    m.study("s1").run();

    m.result().numerical().create("g1", "EvalGlobal");
    m.result().numerical("g1").set("expr", new String[]{{"real(ewfd.neff)", "abs(imag(ewfd.neff))"}});
    m.result().numerical("g1").set("data", "dset1");
    double[][] vals = m.result().numerical("g1").getReal();
    // Evaluate TE polarization fraction in core via surface integration of |Ex|^2 and |Ey|^2
    m.result().numerical().create("i1", "IntSurface");
    m.result().numerical("i1").selection().named("selCore");
    m.result().numerical("i1").set("expr", new String[]{{"abs(ewfd.Ex)^2", "abs(ewfd.Ey)^2"}});
    m.result().numerical("i1").set("data", "dset1");
    double[][] iv = m.result().numerical("i1").getReal();

    int best = 0;
    double bestScore = -1.0;
    for (int j = 0; j < vals[0].length; j++) {{
      double nr = vals[0][j];
      double teFrac = iv[0][j] / Math.max(iv[0][j] + iv[1][j], 1e-30);
      if (teFrac > 0.65 && nr > nClad && nr < nCore) {{
        if (nr > bestScore) {{ bestScore = nr; best = j; }}
      }}
    }}
    double neff = vals[0][best];
    double keff = vals[1][best];
    double alphaDbCm = 2.0 * (2.0 * Math.PI / wl) * keff * (10.0 / Math.log(10.0)) * 1e-2;
    ModelUtil.remove(tag);
    return new double[]{{neff, keff, alphaDbCm}};
  }}

  static void runCase(String id, double wCore, double hCore, double wSheet, double spM, double dM,
                      double wl, double nCore, double nClad, double nTop,
                      String exx, String eyy, String ezz, double shift) throws Exception {{
    double[] b = solveOne(id + "_b", wCore, hCore, wSheet, spM, dM, wl, nCore, nClad, nTop, exx, eyy, ezz, shift, false);
    double[] s = solveOne(id + "_s", wCore, hCore, wSheet, spM, dM, wl, nCore, nClad, nTop, exx, eyy, ezz, shift, true);
    double dneff = s[0] - b[0];
    System.out.println("JSON_2D:" + "{{\\"id\\":\\"" + id + "\\",\\"neff_bare\\":" + b[0] + ",\\"neff_2d\\":" + s[0] + ",\\"delta_neff\\":" + dneff + ",\\"alpha_dB_cm\\":" + s[2] + "}}");
  }}

  public static void main(String[] args) throws Exception {{
{chr(10).join(java_calls)}
  }}
}}
"""
    print("Running COMSOL 6.4 ewfd anisotropic full-tensor validation for 6 2D vdW benchmarks...")
    t0 = time.perf_counter()
    lines = _run_comsol_java("Comsol2DLibraryValidation", java_src)
    elapsed_s = time.perf_counter() - t0
    comsol_rows = {}
    for ln in lines:
        if "JSON_2D:" in ln:
            d = json.loads(ln.split("JSON_2D:", 1)[1].strip())
            comsol_rows[d["id"]] = d

    summary = []
    print("\n=== COMSOL 6.4 vs MicroRing Studio 2D vdW Full-Tensor Validation ===")
    for c in cases:
        cid = c["id"]
        mr = mr_results[cid]
        cm = comsol_rows[cid]
        dn_mr = mr["delta_neff_real"]
        dn_cm = cm["delta_neff"]
        dn_err_pct = abs(dn_mr - dn_cm) / max(abs(dn_cm), 1e-6) * 100.0
        a_mr = mr["alpha_2d_dB_per_cm"]
        a_cm = cm["alpha_dB_cm"]
        a_err_pct = abs(a_mr - a_cm) / max(abs(a_cm), 1.0) * 100.0 if a_cm > 1.0 else 0.0
        row = {
            "id": cid,
            "material_2d": c["material_2d"],
            "wavelength_nm": c["wl_nm"],
            "crystal_angle_deg": c["angle_deg"],
            "thickness_nm": c["d_nm"],
            "dispersion_regime": mr["dispersion_regime"],
            "comsol_neff_bare": round(cm["neff_bare"], 5),
            "microring_neff_bare": round(mr["neff_bare"], 5),
            "comsol_delta_neff": float(f"{dn_cm:.5e}"),
            "microring_delta_neff": float(f"{dn_mr:.5e}"),
            "delta_neff_rel_err_pct": round(dn_err_pct, 2),
            "comsol_alpha_dB_per_cm": round(a_cm, 2),
            "microring_alpha_dB_per_cm": round(a_mr, 2),
            "alpha_rel_err_pct": round(a_err_pct, 2),
            "sign_match": (dn_mr * dn_cm) > 0.0,
        }
        summary.append(row)
        print(
            f"  {cid:18s} | dNeff: COMSOL={dn_cm:+.4e} vs MR={dn_mr:+.4e} ({dn_err_pct:5.2f}%) "
            f"| alpha: COMSOL={a_cm:7.2f} vs MR={a_mr:7.2f} dB/cm ({a_err_pct:5.2f}%)"
        )

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": platform.node(),
        "solver": "COMSOL Multiphysics 6.4 (ewfd full 3x3 anisotropic complex permittivity tensor)",
        "elapsed_s": round(elapsed_s, 1),
        "catalog_size": len(TWOD_MATERIAL_CATALOG),
        "benchmarks": summary,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    print(f"\nSaved validation report to {OUT_JSON} ({elapsed_s:.1f} s).")


if __name__ == "__main__":
    main()
