"""Independent COMSOL Multiphysics 6.4 Frontier Validation Runner (`comsolcompile` + `comsolbatch`).

Independently validates MicroRing Studio's new first-principles solid-mechanics, piezoelectric,
and photonic-crystal engines head-to-head against COMSOL Multiphysics 6.4:
  1. Suite F-C1 (2D Piezoelectric SAW & Lamb Wave FEM: `solid` + `es` + `pze` Floquet PBC):
     - Case 1a: 500 nm c-axis AlN / 1.0 um SiO2 / Si Rayleigh SAW (Lambda = 2.0 um)
     - Case 1b: 500 nm c-axis Sc0.20Al0.80N / 1.0 um SiO2 / Si Rayleigh SAW (Lambda = 2.0 um)
     - Case 1c: 400 nm Suspended X-cut LiNbO3 Lamb S0 membrane (Lambda = 2.0 um)
     Validates open-circuit v_o, short-circuit v_s, f_SAW [GHz], and exact K_eff^2 = 2*(v_o - v_s)/v_o!
  2. Suite F-C2 (2D/3D Solid Mechanics Waveguide SBS Acoustic Modes: `solid`):
     - Case 2a: Suspended 220 x 450 nm [100] Si Forward-SBS (q_z = 0, 2D plane-strain breathing mode)
     - Case 2b: Suspended 220 x 450 nm [100] Si Backward-SBS (q_z = 2*beta_opt, 3D Floquet slice)
     Validates exact acoustic breathing eigenfrequencies Omega_B / (2*pi) [GHz]!
  3. Suite F-C3 (2D Wave Optics `ewfd` Graphene Surface Current Density `J_s = sigma * E_t`):
     - Single-layer Graphene on 220 x 480 nm SOI (10 nm oxide spacer, 1550 nm) at bare, mu_c=0.0 eV, and mu_c=0.55 eV
     Validates exact modal attenuation alpha_2d [dB/cm] and Pauli-blocking Delta_neff against `simulate_2d_material_mrr`!
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

COMSOL_BIN = Path(r"C:\Program Files\COMSOL\COMSOL64\Multiphysics\bin\win64")
COMSOL_COMPILE = COMSOL_BIN / "comsolcompile.exe"
COMSOL_BATCH = COMSOL_BIN / "comsolbatch.exe"

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH_DIR = OUT_DIR / "_comsol_frontier_scratch"
OUT_JSON = OUT_DIR / "frontier_comsol_validation.json"


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
        raise RuntimeError(f"comsolbatch failed for {class_name}:\n{log_txt[-3000:]}\n{batch.stderr}")
    return ((batch.stdout or "") + "\n" + log_txt).splitlines()


def _parse_kv_lines(lines: list[str], prefix: str) -> list[dict]:
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(prefix):
            out.append(json.loads(s[len(prefix):].strip()))
    return out


def _voigt21_java(pr: dict[str, float]) -> str:
    """COMSOL 21-component symmetric Voigt lower-triangle vector:
    [C11, C12, C22, C13, C23, C33, 0,0,0, C44, 0,0,0,0, C55, 0,0,0,0,0, C66]
    """
    vals = [
        pr["C11"],
        pr["C12"], pr["C22"],
        pr["C13"], pr["C23"], pr["C33"],
        0.0, 0.0, 0.0, pr["C44"],
        0.0, 0.0, 0.0, 0.0, pr["C55"],
        0.0, 0.0, 0.0, 0.0, 0.0, pr["C66"],
    ]
    return "new String[]{" + ", ".join(f'"{v:.6e}"' for v in vals) + "}"


def _voigt21_piezo_java(pr: dict[str, float]) -> str:
    """COMSOL 2D PiezoelectricMaterialModel maps 2D geometry (x_geom, y_geom, z_out) -> crystal (1=X, 3=Z_vert, 2=Y_out)!
    Therefore: C11_c = C11, C33_c = C22(vert), C22_c = C33(out), C13_c = C12, C12_c = C13, C23_c = C23,
    C55_c (1-3 shear = geom xy) = C66, C44_c (2-3) = C44, C66_c (1-2) = C55!
    Lower-triangle Voigt21: [C11, C12, C22, C13, C23, C33, 0,0,0, C44, 0,0,0,0, C55, 0,0,0,0,0, C66].
    """
    vals = [
        pr["C11"],
        pr["C13"], pr["C33"],
        pr["C12"], pr["C23"], pr["C22"],
        0.0, 0.0, 0.0, pr["C44"],
        0.0, 0.0, 0.0, 0.0, pr["C66"],
        0.0, 0.0, 0.0, 0.0, 0.0, pr["C55"],
    ]
    return "new String[]{" + ", ".join(f'"{v:.6e}"' for v in vals) + "}"


def _piezo_e_java(pr: dict[str, float]) -> str:
    """COMSOL 2D PiezoelectricMaterialModel (x_geom=1, z_out=2, y_geom=3) 1D column-major String[18]:
    Active rows: row 0 = E_x (1), row 2 = E_y (3).
    Active cols: col 0 = S_xx (1), col 2 = S_yy (3), col 4 = 2*S_xy (5).
    """
    ex1 = pr.get("e_x1", 0.0)
    ex2 = pr.get("e_x2", 0.0)
    ex6 = pr.get("e_x6", 0.0)
    ey1 = pr.get("e_y1", 0.0)
    ey2 = pr.get("e_y2", 0.0)
    ey6 = pr.get("e_y6", 0.0)
    cols = [
        ex1, 0.0, ey1,   # col 0: S_xx (1) -> idx 0 (Ex), idx 2 (Ey)
        ex2, 0.0, ey1,   # col 1: S_zz_out (2)
        ex2, 0.0, ey2,   # col 2: S_yy (3) -> idx 6 (Ex), idx 8 (Ey)
        0.0, 0.0, 0.0,   # col 3: 2*S_yz_out (4)
        ex6, 0.0, ey6,   # col 4: 2*S_xy (5) -> idx 12 (Ex), idx 14 (Ey)
        0.0, 0.0, 0.0,   # col 5: 2*S_xz_out (6)
    ]
    return "new String[]{" + ", ".join(f'"{v:.6f}"' for v in cols) + "}"


# ==============================================================================
# Suite F-C1: COMSOL 6.4 2D Piezoelectric SAW & Lamb S0 (solid + es + pze)
# ==============================================================================

def run_comsol_saw_suite() -> list[dict]:
    from microring.acousto_optic_saw import (
        ELASTIC_MATERIAL_PROPS,
        SAWAcoustoOpticConfig,
        simulate_saw_acousto_optic_mrr,
    )

    cases = [
        {
            "case_id": "AlN_500nm_SiO2_1um_Si_SAW",
            "cfg": SAWAcoustoOpticConfig(
                waveguide_material="AlN", box_material="SiO2", substrate_material="Si",
                film_height_um=0.50, box_thickness_um=1.00, idt_period_um=2.00,
                acoustic_mode="rayleigh_or_s0",
            ),
            "is_suspended": False,
            "want_s0": False,
            "f_shift_ghz": 1.6,
        },
        {
            "case_id": "ScAlN20_500nm_SiO2_1um_Si_SAW",
            "cfg": SAWAcoustoOpticConfig(
                waveguide_material="ScAlN_20", box_material="SiO2", substrate_material="Si",
                film_height_um=0.50, box_thickness_um=1.00, idt_period_um=2.00,
                acoustic_mode="rayleigh_or_s0",
            ),
            "is_suspended": False,
            "want_s0": False,
            "f_shift_ghz": 1.5,
        },
        {
            "case_id": "LN_Xcut_400nm_Suspended_Lamb_S0",
            "cfg": SAWAcoustoOpticConfig(
                waveguide_material="LN_Xcut", box_material="Air", substrate_material="Air",
                film_height_um=0.40, box_thickness_um=0.0, idt_period_um=2.00,
                acoustic_mode="lamb_s0",
            ),
            "is_suspended": True,
            "want_s0": True,
            "f_shift_ghz": 3.0,
        },
    ]

    java_calls = []
    for c in cases:
        cfg = c["cfg"]
        pf = ELASTIC_MATERIAL_PROPS[cfg.waveguide_material]
        pb = ELASTIC_MATERIAL_PROPS["SiO2"]
        ps = ELASTIC_MATERIAL_PROPS["Si"]
        lam_m = cfg.idt_period_um * 1e-6
        h_film = cfg.film_height_um * 1e-6
        h_box = cfg.box_thickness_um * 1e-6
        java_calls.append(
            f'solveSawCase("{c["case_id"]}", {str(c["is_suspended"]).lower()}, {str(c["want_s0"]).lower()}, '
            f'{lam_m:.9e}, {h_film:.9e}, {h_box:.9e}, {c["f_shift_ghz"]:.4f}e9, '
            f'{_voigt21_piezo_java(pf)}, {_piezo_e_java(pf)}, {pf["eps_r"]:.4f}, {pf["rho"]:.2f}, '
            f'{_voigt21_java(pb)}, {pb["eps_r"]:.4f}, {pb["rho"]:.2f}, '
            f'{_voigt21_java(ps)}, {ps["eps_r"]:.4f}, {ps["rho"]:.2f});'
        )

    java_src = f"""
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolFrontierSAW {{

  static double solveOneBC(
      String tag, boolean isSuspended, boolean wantS0, boolean shortTop,
      double lamM, double hFilm, double hBox, double fShift,
      String[] cFilm, String[] eFilm, double epsFilm, double rhoFilm,
      String[] cBox, double epsBox, double rhoBox,
      String[] cSub, double epsSub, double rhoSub
  ) throws Exception {{
    double wCell = 0.25 * lamM; // Lambda/4 -> kF*W = pi/2, zero aliasing with folded bulk modes!
    double kx = 2.0 * Math.PI / lamM;
    double hSub = isSuspended ? 0.0 : (3.5 * lamM);
    double hAirTop = 1.5 * lamM;
    double hAirBot = isSuspended ? (1.5 * lamM) : 0.0;

    ModelUtil.remove(tag);
    Model model = ModelUtil.create(tag);
    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);

    // Vertical stack coordinates (y = 0 at bottom of solid)
    double y0 = 0.0;
    if (isSuspended) {{
      model.component("comp1").geom("geom1").create("rAirBot", "Rectangle");
      model.component("comp1").geom("geom1").feature("rAirBot").set("pos", new double[]{{0.0, -hAirBot}});
      model.component("comp1").geom("geom1").feature("rAirBot").set("size", new double[]{{wCell, hAirBot}});

      model.component("comp1").geom("geom1").create("rFilm", "Rectangle");
      model.component("comp1").geom("geom1").feature("rFilm").set("pos", new double[]{{0.0, 0.0}});
      model.component("comp1").geom("geom1").feature("rFilm").set("size", new double[]{{wCell, hFilm}});
      y0 = hFilm;
    }} else {{
      model.component("comp1").geom("geom1").create("rSub", "Rectangle");
      model.component("comp1").geom("geom1").feature("rSub").set("pos", new double[]{{0.0, 0.0}});
      model.component("comp1").geom("geom1").feature("rSub").set("size", new double[]{{wCell, hSub}});
      y0 = hSub;
      if (hBox > 1e-10) {{
        model.component("comp1").geom("geom1").create("rBox", "Rectangle");
        model.component("comp1").geom("geom1").feature("rBox").set("pos", new double[]{{0.0, y0}});
        model.component("comp1").geom("geom1").feature("rBox").set("size", new double[]{{wCell, hBox}});
        y0 += hBox;
      }}
      model.component("comp1").geom("geom1").create("rFilm", "Rectangle");
      model.component("comp1").geom("geom1").feature("rFilm").set("pos", new double[]{{0.0, y0}});
      model.component("comp1").geom("geom1").feature("rFilm").set("size", new double[]{{wCell, hFilm}});
      y0 += hFilm;
    }}
    double yFilmTop = y0;
    double yFilmBot = yFilmTop - hFilm;
    model.component("comp1").geom("geom1").create("rAirTop", "Rectangle");
    model.component("comp1").geom("geom1").feature("rAirTop").set("pos", new double[]{{0.0, yFilmTop}});
    model.component("comp1").geom("geom1").feature("rAirTop").set("size", new double[]{{wCell, hAirTop}});
    model.component("comp1").geom("geom1").run();

    // Box selections for each layer & boundary
    double eps = 1e-9;
    model.component("comp1").selection().create("selFilm", "Box");
    model.component("comp1").selection("selFilm").set("entitydim", 2);
    model.component("comp1").selection("selFilm").set("condition", "inside");
    model.component("comp1").selection("selFilm").set("xmin", -eps).set("xmax", wCell + eps)
      .set("ymin", yFilmBot - eps).set("ymax", yFilmTop + eps);

    model.component("comp1").selection().create("selSolid", "Box");
    model.component("comp1").selection("selSolid").set("entitydim", 2);
    model.component("comp1").selection("selSolid").set("condition", "inside");
    model.component("comp1").selection("selSolid").set("xmin", -eps).set("xmax", wCell + eps)
      .set("ymin", -eps).set("ymax", yFilmTop + eps);

    if (!isSuspended) {{
      model.component("comp1").selection().create("selSub", "Box");
      model.component("comp1").selection("selSub").set("entitydim", 2);
      model.component("comp1").selection("selSub").set("condition", "inside");
      model.component("comp1").selection("selSub").set("xmin", -eps).set("xmax", wCell + eps)
        .set("ymin", -eps).set("ymax", hSub + eps);

      model.component("comp1").selection().create("selBox", "Box");
      model.component("comp1").selection("selBox").set("entitydim", 2);
      model.component("comp1").selection("selBox").set("condition", "inside");
      model.component("comp1").selection("selBox").set("xmin", -eps).set("xmax", wCell + eps)
        .set("ymin", hSub - eps).set("ymax", hSub + hBox + eps);
    }}

    // Left + Right periodic boundaries (solid only & all es)
    model.component("comp1").selection().create("selLeftSolid", "Box");
    model.component("comp1").selection("selLeftSolid").set("entitydim", 1).set("condition", "inside")
      .set("xmin", -eps).set("xmax", eps).set("ymin", -eps).set("ymax", yFilmTop + eps);
    model.component("comp1").selection().create("selRightSolid", "Box");
    model.component("comp1").selection("selRightSolid").set("entitydim", 1).set("condition", "inside")
      .set("xmin", wCell - eps).set("xmax", wCell + eps).set("ymin", -eps).set("ymax", yFilmTop + eps);
    model.component("comp1").selection().create("selPerSolid", "Union");
    model.component("comp1").selection("selPerSolid").set("entitydim", 1)
      .set("input", new String[]{{"selLeftSolid", "selRightSolid"}});

    double yMinAll = isSuspended ? -hAirBot : 0.0;
    double yMaxAll = yFilmTop + hAirTop;
    model.component("comp1").selection().create("selLeftAll", "Box");
    model.component("comp1").selection("selLeftAll").set("entitydim", 1).set("condition", "inside")
      .set("xmin", -eps).set("xmax", eps).set("ymin", yMinAll - eps).set("ymax", yMaxAll + eps);
    model.component("comp1").selection().create("selRightAll", "Box");
    model.component("comp1").selection("selRightAll").set("entitydim", 1).set("condition", "inside")
      .set("xmin", wCell - eps).set("xmax", wCell + eps).set("ymin", yMinAll - eps).set("ymax", yMaxAll + eps);
    model.component("comp1").selection().create("selPerAll", "Union");
    model.component("comp1").selection("selPerAll").set("entitydim", 1)
      .set("input", new String[]{{"selLeftAll", "selRightAll"}});

    model.component("comp1").selection().create("selBotSolid", "Box");
    model.component("comp1").selection("selBotSolid").set("entitydim", 1).set("condition", "inside")
      .set("xmin", -eps).set("xmax", wCell + eps).set("ymin", -eps).set("ymax", eps);

    model.component("comp1").selection().create("selFilmTop", "Box");
    model.component("comp1").selection("selFilmTop").set("entitydim", 1).set("condition", "inside")
      .set("xmin", -eps).set("xmax", wCell + eps).set("ymin", yFilmTop - eps).set("ymax", yFilmTop + eps);

    // Physics 1: Solid Mechanics (only on solid domains!)
    model.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    model.component("comp1").physics("solid").selection().named("selSolid");

    // Piezoelectric film domain
    model.component("comp1").physics("solid").create("pzm1", "PiezoelectricMaterialModel", 2);
    model.component("comp1").physics("solid").feature("pzm1").selection().named("selFilm");
    model.component("comp1").physics("solid").feature("pzm1").set("ConstitutiveRelation", "StressCharge");
    model.component("comp1").physics("solid").feature("pzm1").set("cE_mat", "userdef");
    model.component("comp1").physics("solid").feature("pzm1").set("cE", cFilm);
    model.component("comp1").physics("solid").feature("pzm1").set("eES_mat", "userdef");
    model.component("comp1").physics("solid").feature("pzm1").set("eES", eFilm);
    model.component("comp1").physics("solid").feature("pzm1").set("epsilonrS_mat", "userdef");
    model.component("comp1").physics("solid").feature("pzm1").set("epsilonrS", "" + epsFilm);
    model.component("comp1").physics("solid").feature("pzm1").set("rho_mat", "userdef");
    model.component("comp1").physics("solid").feature("pzm1").set("rho", "" + rhoFilm);

    if (!isSuspended) {{
      // Default lemm1 automatically applies to remaining solid domain (selSub) after pzm1 & lemmBox override!
      model.component("comp1").physics("solid").feature("lemm1").set("SolidModel", "Anisotropic");
      model.component("comp1").physics("solid").feature("lemm1").set("AnisotropicOption", "AnisotropicVo");
      model.component("comp1").physics("solid").feature("lemm1").set("DVo_mat", "userdef");
      model.component("comp1").physics("solid").feature("lemm1").set("DVo", cSub);
      model.component("comp1").physics("solid").feature("lemm1").set("rho_mat", "userdef");
      model.component("comp1").physics("solid").feature("lemm1").set("rho", "" + rhoSub);

      // SiO2 BOX layer
      model.component("comp1").physics("solid").create("lemmBox", "LinearElasticModel", 2);
      model.component("comp1").physics("solid").feature("lemmBox").selection().named("selBox");
      model.component("comp1").physics("solid").feature("lemmBox").set("SolidModel", "Anisotropic");
      model.component("comp1").physics("solid").feature("lemmBox").set("AnisotropicOption", "AnisotropicVo");
      model.component("comp1").physics("solid").feature("lemmBox").set("DVo_mat", "userdef");
      model.component("comp1").physics("solid").feature("lemmBox").set("DVo", cBox);
      model.component("comp1").physics("solid").feature("lemmBox").set("rho_mat", "userdef");
      model.component("comp1").physics("solid").feature("lemmBox").set("rho", "" + rhoBox);

      // Fixed bottom substrate
      model.component("comp1").physics("solid").create("fix1", "Fixed", 1);
      model.component("comp1").physics("solid").feature("fix1").selection().named("selBotSolid");
    }}

    // Solid Floquet PBC along x
    model.component("comp1").physics("solid").create("pc1", "PeriodicCondition", 1);
    model.component("comp1").physics("solid").feature("pc1").selection().named("selPerSolid");
    model.component("comp1").physics("solid").feature("pc1").set("PeriodicType", "Floquet");
    model.component("comp1").physics("solid").feature("pc1").set("kFloquet", new String[]{{"" + kx, "0", "0"}});

    // Physics 2: Electrostatics on ALL domains (COMSOL 6.4 default fsp1 = FreeSpace eps_r = 1 for Air!)
    model.component("comp1").physics().create("es", "Electrostatics", "geom1");

    // ChargeConservationPiezo on piezo film
    model.component("comp1").physics("es").create("ccnp1", "ChargeConservationPiezo", 2);
    model.component("comp1").physics("es").feature("ccnp1").selection().named("selFilm");
    model.component("comp1").physics("es").feature("ccnp1").set("epsilonrS_mat", "userdef");
    model.component("comp1").physics("es").feature("ccnp1").set("epsilonrS", "" + epsFilm);

    if (!isSuspended) {{
      model.component("comp1").physics("es").create("ccnSub", "ChargeConservation", 2);
      model.component("comp1").physics("es").feature("ccnSub").selection().named("selSub");
      model.component("comp1").physics("es").feature("ccnSub").set("epsilonr_mat", "userdef");
      model.component("comp1").physics("es").feature("ccnSub").set("epsilonr", "" + epsSub);

      model.component("comp1").physics("es").create("ccnBox", "ChargeConservation", 2);
      model.component("comp1").physics("es").feature("ccnBox").selection().named("selBox");
      model.component("comp1").physics("es").feature("ccnBox").set("epsilonr_mat", "userdef");
      model.component("comp1").physics("es").feature("ccnBox").set("epsilonr", "" + epsBox);

      model.component("comp1").physics("es").create("gndBot", "Ground", 1);
      model.component("comp1").physics("es").feature("gndBot").selection().named("selBotSolid");
    }}
    if (shortTop) {{
      model.component("comp1").physics("es").create("gndTop", "Ground", 1);
      model.component("comp1").physics("es").feature("gndTop").selection().named("selFilmTop");
    }}

    // ES Floquet PBC
    model.component("comp1").physics("es").create("pcEs", "PeriodicCondition", 1);
    model.component("comp1").physics("es").feature("pcEs").selection().named("selPerAll");
    model.component("comp1").physics("es").feature("pcEs").set("PeriodicType", "Floquet");
    model.component("comp1").physics("es").feature("pcEs").set("kFloquet", new String[]{{"" + kx, "0", "0"}});

    // Multiphysics Piezoelectric Coupling (auto-selects domains with PiezoelectricMaterialModel + ChargeConservationPiezo)
    model.component("comp1").multiphysics().create("pze1", "PiezoelectricEffect", "geom1", 2);

    // Mapped/FreeTri quadratic mesh (identical left/right edge nodes via CopyEdge or fine uniform size)
    model.component("comp1").mesh().create("mesh1", "geom1");
    model.component("comp1").mesh("mesh1").feature().create("map1", "Map");
    model.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    model.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.08e-6);
    model.component("comp1").mesh("mesh1").run();

    // Eigenfrequency study
    model.study().create("std1");
    model.study("std1").feature().create("eig", "Eigenfrequency");
    model.study("std1").feature("eig").set("neigs", 6);
    model.study("std1").feature("eig").set("shift", "" + fShift);
    model.study("std1").feature("eig").set("shiftactive", true);
    model.study("std1").run();

    String dset = model.result().dataset().tags()[0];
    model.result().numerical().create("intMode", "IntSurface");
    model.result().numerical("intMode").set("data", dset);
    model.result().numerical("intMode").selection().named("selFilm");
    model.result().numerical("intMode").set("expr", new String[]{{"real(freq)", "abs(u)", "abs(v)", "1"}});
    double[][] vals = model.result().numerical("intMode").getReal();
    // vals[exprIdx][solIdx]
    int nSol = vals[0].length;
    double chosenFreq = -1.0;
    for (int j = 0; j < nSol; j++) {{
      double fj = vals[0][j] / Math.max(vals[3][j], 1e-30);
      double meanUx = vals[1][j] / Math.max(vals[3][j], 1e-30);
      double meanUy = vals[2][j] / Math.max(vals[3][j], 1e-30);
      if (fj < 1e7) continue;
      if (wantS0) {{
        // Extensional Lamb S0 has |u_x| >> |u_y|!
        if (meanUx > 1.5 * meanUy && (chosenFreq < 0 || fj < chosenFreq)) chosenFreq = fj;
      }} else {{
        // Fundamental Rayleigh SAW is lowest positive sagittal eigenmode!
        if (chosenFreq < 0 || fj < chosenFreq) chosenFreq = fj;
      }}
    }}
    ModelUtil.remove(tag);
    return chosenFreq;
  }}

  static void solveSawCase(
      String cid, boolean isSuspended, boolean wantS0,
      double lamM, double hFilm, double hBox, double fShift,
      String[] cFilm, String[] eFilm, double epsFilm, double rhoFilm,
      String[] cBox, double epsBox, double rhoBox,
      String[] cSub, double epsSub, double rhoSub
  ) throws Exception {{
    long t0 = System.nanoTime();
    double fOpen = solveOneBC(cid + "_open", isSuspended, wantS0, false, lamM, hFilm, hBox, fShift,
      cFilm, eFilm, epsFilm, rhoFilm, cBox, epsBox, rhoBox, cSub, epsSub, rhoSub);
    double fShort = solveOneBC(cid + "_short", isSuspended, wantS0, true, lamM, hFilm, hBox, fShift,
      cFilm, eFilm, epsFilm, rhoFilm, cBox, epsBox, rhoBox, cSub, epsSub, rhoSub);
    double vOpen = fOpen * lamM;
    double vShort = fShort * lamM;
    double keff2Pct = 2.0 * (vOpen - vShort) / Math.max(vOpen, 1.0) * 100.0;
    double dt = (System.nanoTime() - t0) * 1e-9;
    System.out.println("SAW_JSON: {{\\"case_id\\":\\"" + cid + "\\",\\"f_open_GHz\\":" + (fOpen * 1e-9)
      + ",\\"f_short_GHz\\":" + (fShort * 1e-9) + ",\\"v_open_m_per_s\\":" + vOpen
      + ",\\"v_short_m_per_s\\":" + vShort + ",\\"K_eff_sq_pct\\":" + keff2Pct
      + ",\\"wall_time_s\\":" + dt + "}}");
  }}

  public static void main(String[] args) throws Exception {{
    {chr(10).join("    " + ln for ln in java_calls)}
  }}
}}
"""
    lines = _run_comsol_java("ComsolFrontierSAW", java_src)
    comsol_rows = {r["case_id"]: r for r in _parse_kv_lines(lines, "SAW_JSON:")}

    out = []
    for c in cases:
        cid = c["case_id"]
        cr = comsol_rows[cid]
        mr = simulate_saw_acousto_optic_mrr(c["cfg"])["metrics"]
        err_v_pct = abs(mr["v_open_m_per_s"] - cr["v_open_m_per_s"]) / cr["v_open_m_per_s"] * 100.0
        err_k_abs = abs(mr["K_eff_sq_pct"] - cr["K_eff_sq_pct"])
        rec = {
            "case_id": cid,
            "comsol_64": {
                "v_open_m_per_s": round(cr["v_open_m_per_s"], 2),
                "v_short_m_per_s": round(cr["v_short_m_per_s"], 2),
                "f_SAW_GHz": round(cr["f_open_GHz"], 4),
                "K_eff_sq_pct": round(cr["K_eff_sq_pct"], 3),
                "wall_time_s": round(cr["wall_time_s"], 2),
            },
            "microring_fem": {
                "v_open_m_per_s": mr["v_open_m_per_s"],
                "v_short_m_per_s": mr["v_short_m_per_s"],
                "f_SAW_GHz": mr["f_SAW_GHz"],
                "K_eff_sq_pct": mr["K_eff_sq_pct"],
            },
            "rel_err_v_open_pct": round(err_v_pct, 4),
            "abs_err_Keff2_pct_pts": round(err_k_abs, 4),
        }
        out.append(rec)
        print(
            f"[COMSOL-SAW] {cid:<32} COMSOL v_o={cr['v_open_m_per_s']:.2f} m/s K2={cr['K_eff_sq_pct']:.3f}% | "
            f"MicroRing v_o={mr['v_open_m_per_s']:.2f} m/s K2={mr['K_eff_sq_pct']:.3f}% (err_v={err_v_pct:.3f}%)"
        )
    return out


# ==============================================================================
# Suite F-C2: COMSOL 6.4 2D & 3D Solid Mechanics Si SBS Breathing Modes
# ==============================================================================

def run_comsol_sbs_suite() -> list[dict]:
    from microring.acousto_optic_saw import (
        ELASTIC_MATERIAL_PROPS,
        SBSBrillouinConfig,
        simulate_sbs_brillouin_mrr,
    )

    pr_si = ELASTIC_MATERIAL_PROPS["Si"]
    # Run MicroRing first ONLY to read exact optical propagation wavevector q_z = 2*beta_opt so both solve same q_z!
    mr_fwd = simulate_sbs_brillouin_mrr(SBSBrillouinConfig(
        core_material="Si", clad_material="Air", film_height_um=0.22, ring_width_um=0.45,
        sbs_direction="forward", mesh_nx=14, mesh_ny=10,
    ))["metrics"]
    mr_bwd = simulate_sbs_brillouin_mrr(SBSBrillouinConfig(
        core_material="Si", clad_material="Air", film_height_um=0.22, ring_width_um=0.45,
        sbs_direction="backward", mesh_nx=14, mesh_ny=10,
    ))["metrics"]
    qz_bwd = mr_bwd["acoustic_qz_per_um"] * 1e6  # rad/m

    java_src = f"""
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolFrontierSBS {{

  public static void main(String[] args) throws Exception {{
    double w = 0.45e-6;
    double h = 0.22e-6;
    double rhoSi = {pr_si["rho"]:.2f};
    String[] cSi = {_voigt21_java(pr_si)};
    double qzBwd = {qz_bwd:.8e};

    // 1. Forward SBS (q_z = 0): 2D Plane Strain Solid Mechanics on 450 x 220 nm Si
    long t0 = System.nanoTime();
    Model m2 = ModelUtil.create("SBS_Fwd");
    m2.component().create("comp1", true);
    m2.component("comp1").geom().create("geom1", 2);
    m2.component("comp1").geom("geom1").create("r1", "Rectangle");
    m2.component("comp1").geom("geom1").feature("r1").set("pos", new double[]{{-0.5*w, 0.0}});
    m2.component("comp1").geom("geom1").feature("r1").set("size", new double[]{{w, h}});
    m2.component("comp1").geom("geom1").run();

    m2.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    m2.component("comp1").physics("solid").feature("lemm1").set("SolidModel", "Anisotropic");
    m2.component("comp1").physics("solid").feature("lemm1").set("AnisotropicOption", "AnisotropicVo");
    m2.component("comp1").physics("solid").feature("lemm1").set("DVo_mat", "userdef");
    m2.component("comp1").physics("solid").feature("lemm1").set("DVo", cSi);
    m2.component("comp1").physics("solid").feature("lemm1").set("rho_mat", "userdef");
    m2.component("comp1").physics("solid").feature("lemm1").set("rho", "" + rhoSi);

    m2.component("comp1").mesh().create("mesh1", "geom1");
    m2.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m2.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.020e-6);
    m2.component("comp1").mesh("mesh1").feature().create("ftri1", "FreeTri");
    m2.component("comp1").mesh("mesh1").run();

    m2.study().create("std1");
    m2.study("std1").feature().create("eig", "Eigenfrequency");
    m2.study("std1").feature("eig").set("neigs", 8);
    m2.study("std1").feature("eig").set("shift", "7.5e9");
    m2.study("std1").feature("eig").set("shiftactive", true);
    m2.study("std1").run();

    String ds2 = m2.result().dataset().tags()[0];
    m2.result().numerical().create("int2", "IntSurface");
    m2.result().numerical("int2").set("data", ds2);
    m2.result().numerical("int2").selection().all();
    // Symmetric breathing mode has largest |int (du/dx) dA| normalized by sqrt(int (|u|^2+|v|^2) dA)!
    m2.result().numerical("int2").set("expr", new String[]{{"real(freq)", "abs(solid.eXX)", "solid.eXX", "abs(u)^2+abs(v)^2", "1"}});
    double[][] v2 = m2.result().numerical("int2").getReal();
    double fFwdGHz = 0.0, bestScore2 = -1.0;
    for (int j = 0; j < v2[0].length; j++) {{
      double fj = v2[0][j] / Math.max(v2[4][j], 1e-30);
      double breath = Math.abs(v2[2][j]) / Math.max(v2[1][j], 1e-30);
      if (fj > 5e9 && fj < 14e9 && breath > bestScore2) {{
        bestScore2 = breath;
        fFwdGHz = fj * 1e-9;
      }}
    }}
    double dt2 = (System.nanoTime() - t0) * 1e-9;
    System.out.println("SBS_JSON: {{\\"case_id\\":\\"Si_220x450_Forward_SBS_qz0\\",\\"Omega_B_GHz\\":" + fFwdGHz
      + ",\\"breathing_symmetry\\":" + bestScore2 + ",\\"wall_time_s\\":" + dt2 + "}}");

    // 2. Backward SBS (q_z = 2*beta_opt): 3D thin slice Lz = 35 nm with axial Floquet PBC!
    long t1 = System.nanoTime();
    double lz = 0.035e-6;
    Model m3 = ModelUtil.create("SBS_Bwd");
    m3.component().create("comp1", true);
    m3.component("comp1").geom().create("geom1", 3);
    m3.component("comp1").geom("geom1").create("blk1", "Block");
    m3.component("comp1").geom("geom1").feature("blk1").set("pos", new double[]{{-0.5*w, 0.0, 0.0}});
    m3.component("comp1").geom("geom1").feature("blk1").set("size", new double[]{{w, h, lz}});
    m3.component("comp1").geom("geom1").run();

    double eps = 1e-9;
    m3.component("comp1").selection().create("selZ0", "Box");
    m3.component("comp1").selection("selZ0").set("entitydim", 2).set("condition", "inside")
      .set("xmin", -w).set("xmax", w).set("ymin", -h).set("ymax", 2*h).set("zmin", -eps).set("zmax", eps);
    m3.component("comp1").selection().create("selZ1", "Box");
    m3.component("comp1").selection("selZ1").set("entitydim", 2).set("condition", "inside")
      .set("xmin", -w).set("xmax", w).set("ymin", -h).set("ymax", 2*h).set("zmin", lz-eps).set("zmax", lz+eps);
    m3.component("comp1").selection().create("selPerZ", "Union");
    m3.component("comp1").selection("selPerZ").set("entitydim", 2).set("input", new String[]{{"selZ0", "selZ1"}});

    m3.component("comp1").physics().create("solid", "SolidMechanics", "geom1");
    m3.component("comp1").physics("solid").feature("lemm1").set("SolidModel", "Anisotropic");
    m3.component("comp1").physics("solid").feature("lemm1").set("AnisotropicOption", "AnisotropicVo");
    m3.component("comp1").physics("solid").feature("lemm1").set("DVo_mat", "userdef");
    m3.component("comp1").physics("solid").feature("lemm1").set("DVo", cSi);
    m3.component("comp1").physics("solid").feature("lemm1").set("rho_mat", "userdef");
    m3.component("comp1").physics("solid").feature("lemm1").set("rho", "" + rhoSi);

    m3.component("comp1").physics("solid").create("pcZ", "PeriodicCondition", 2);
    m3.component("comp1").physics("solid").feature("pcZ").selection().named("selPerZ");
    m3.component("comp1").physics("solid").feature("pcZ").set("PeriodicType", "Floquet");
    m3.component("comp1").physics("solid").feature("pcZ").set("kFloquet", new String[]{{"0", "0", "" + qzBwd}});

    // Swept hexahedral mesh (identical z=0 and z=lz face triangulation!)
    m3.component("comp1").mesh().create("mesh1", "geom1");
    m3.component("comp1").mesh("mesh1").feature().create("map1", "Map");
    m3.component("comp1").mesh("mesh1").feature("map1").selection().named("selZ0");
    m3.component("comp1").mesh("mesh1").feature("size").set("custom", "on");
    m3.component("comp1").mesh("mesh1").feature("size").set("hmax", 0.032e-6);
    m3.component("comp1").mesh("mesh1").feature().create("swe1", "Sweep");
    m3.component("comp1").mesh("mesh1").feature("swe1").create("dis1", "Distribution");
    m3.component("comp1").mesh("mesh1").feature("swe1").feature("dis1").set("numelem", 2);
    m3.component("comp1").mesh("mesh1").run();

    m3.study().create("std1");
    m3.study("std1").feature().create("eig", "Eigenfrequency");
    m3.study("std1").feature("eig").set("neigs", 14);
    m3.study("std1").feature("eig").set("shift", "15.5e9");
    m3.study("std1").feature("eig").set("shiftactive", true);
    m3.study("std1").run();

    String ds3 = m3.result().dataset().tags()[0];
    m3.result().numerical().create("int3", "IntVolume");
    m3.result().numerical("int3").set("data", ds3);
    m3.result().numerical("int3").selection().all();
    // Backward SBS breathing/longitudinal mode: largest coherent |int real(eXX + eZZ) dV| / int |eXX|+|eZZ| dV!
    m3.result().numerical("int3").set("expr", new String[]{{"real(freq)", "abs(solid.eXX)+abs(solid.eZZ)", "real(solid.eXX+solid.eZZ)", "imag(solid.eXX+solid.eZZ)", "1"}});
    double[][] v3 = m3.result().numerical("int3").getReal();
    double fBwdGHz = 0.0, bestScore3 = -1.0;
    for (int j = 0; j < v3[0].length; j++) {{
      double fj = v3[0][j] / Math.max(v3[4][j], 1e-30);
      double coh = Math.hypot(v3[2][j], v3[3][j]) / Math.max(v3[1][j], 1e-30);
      // Among strongly breathing-symmetric modes in 14-20 GHz, pick strongest symmetric mode near 16-18 GHz
      if (fj > 14.0e9 && fj < 20.0e9 && coh > bestScore3) {{
        bestScore3 = coh;
        fBwdGHz = fj * 1e-9;
      }}
    }}
    double dt3 = (System.nanoTime() - t1) * 1e-9;
    System.out.println("SBS_JSON: {{\\"case_id\\":\\"Si_220x450_Backward_SBS_qz2beta\\",\\"Omega_B_GHz\\":" + fBwdGHz
      + ",\\"breathing_symmetry\\":" + bestScore3 + ",\\"wall_time_s\\":" + dt3 + "}}");
  }}
}}
"""
    lines = _run_comsol_java("ComsolFrontierSBS", java_src)
    comsol_rows = {r["case_id"]: r for r in _parse_kv_lines(lines, "SBS_JSON:")}

    out = []
    for cid, mr in (
        ("Si_220x450_Forward_SBS_qz0", mr_fwd),
        ("Si_220x450_Backward_SBS_qz2beta", mr_bwd),
    ):
        cr = comsol_rows[cid]
        err_pct = abs(mr["Omega_B_2pi_GHz"] - cr["Omega_B_GHz"]) / max(cr["Omega_B_GHz"], 1e-6) * 100.0
        rec = {
            "case_id": cid,
            "comsol_64_Omega_B_GHz": round(cr["Omega_B_GHz"], 4),
            "microring_2dfem_Omega_B_GHz": mr["Omega_B_2pi_GHz"],
            "rel_err_pct": round(err_pct, 4),
            "microring_G_B_per_W_m": mr["G_B_per_W_m"],
            "wall_time_s": round(cr["wall_time_s"], 2),
        }
        out.append(rec)
        print(
            f"[COMSOL-SBS] {cid:<34} COMSOL={cr['Omega_B_GHz']:.4f} GHz | "
            f"MicroRing={mr['Omega_B_2pi_GHz']:.4f} GHz (err={err_pct:.3f}%) G_B={mr['G_B_per_W_m']:.1f}"
        )
    return out


# ==============================================================================
# Suite F-C3: COMSOL 6.4 2D Wave Optics (`ewfd`) Graphene SurfaceCurrent
# ==============================================================================

def run_comsol_graphene_suite() -> dict:
    from microring.twod_materials_mrr import (
        TwoDMaterialMRRConfig,
        graphene_kubo_conductivity_s,
        simulate_2d_material_mrr,
    )
    from microring.materials import get_material_fn, T_REF_K

    wl0_m = 1.55e-6
    n_si = float(get_material_fn("Si")(wl0_m, T_REF_K).nx)
    n_sio2 = float(get_material_fn("SiO2")(wl0_m, T_REF_K).nx)
    sig_0 = graphene_kubo_conductivity_s(wl0_m, mu_c_ev=0.0, tau_ps=0.10)
    sig_55 = graphene_kubo_conductivity_s(wl0_m, mu_c_ev=0.55, tau_ps=0.10)

    # Note: COMSOL ewfd uses exp(+i*om*t), so sigma_COMSOL = conj(sigma_exp_minus_iwt)
    java_src = f"""
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ComsolFrontierGraphene {{
  public static void main(String[] args) throws Exception {{
    double[][] sigs = {{
      {{0.0, 0.0}},
      {{{sig_0.real:.8e}, {-sig_0.imag:.8e}}},
      {{{sig_55.real:.8e}, {-sig_55.imag:.8e}}}
    }};
    String[] tags = {{"bare", "mu0p0eV", "mu0p55eV"}};
    double w = 0.48e-6, h = 0.22e-6, hSp = 0.010e-6, wl = 1.55e-6;
    double xDom = 2.0e-6, yBot = -1.6e-6, yTop = 1.8e-6, eps = 1e-10;
    for (int k = 0; k < 3; k++) {{
      Model m = ModelUtil.create("Gr_" + k);
      m.component().create("c1", true);
      m.component("c1").geom().create("g1", 2);
      m.component("c1").geom("g1").create("rClad","Rectangle");
      m.component("c1").geom("g1").feature("rClad").set("pos",new double[]{{-xDom, yBot}}).set("size",new double[]{{2*xDom, yTop-yBot}});
      m.component("c1").geom("g1").create("rCore","Rectangle");
      m.component("c1").geom("g1").feature("rCore").set("pos",new double[]{{-0.5*w, 0.0}}).set("size",new double[]{{w, h}});
      m.component("c1").geom("g1").create("lsGr","LineSegment");
      m.component("c1").geom("g1").feature("lsGr").set("specify1","coord").set("coord1",new double[]{{-0.5*w-0.4e-6, h+hSp}});
      m.component("c1").geom("g1").feature("lsGr").set("specify2","coord").set("coord2",new double[]{{+0.5*w+0.4e-6, h+hSp}});
      m.component("c1").geom("g1").run();

      m.component("c1").selection().create("selCore","Box");
      m.component("c1").selection("selCore").set("entitydim",2).set("condition","inside")
        .set("xmin",-0.5*w-eps).set("xmax",0.5*w+eps).set("ymin",-eps).set("ymax",h+eps);
      m.component("c1").selection().create("selGr","Box");
      m.component("c1").selection("selGr").set("entitydim",1).set("condition","inside")
        .set("xmin",-0.5*w-0.4e-6-eps).set("xmax",0.5*w+0.4e-6+eps).set("ymin",h+hSp-eps).set("ymax",h+hSp+eps);

      m.component("c1").physics().create("ewfd","ElectromagneticWavesFrequencyDomain","g1");
      m.component("c1").physics("ewfd").feature("wee1").set("DisplacementFieldModel","RefractiveIndex")
        .set("n_mat","userdef").set("n","{n_sio2:.5f}").set("ki_mat","userdef").set("ki","0").set("mur_mat","userdef").set("mur","1");
      m.component("c1").physics("ewfd").create("weeCore","WaveEquationElectric",2);
      m.component("c1").physics("ewfd").feature("weeCore").selection().named("selCore");
      m.component("c1").physics("ewfd").feature("weeCore").set("DisplacementFieldModel","RefractiveIndex")
        .set("n_mat","userdef").set("n","{n_si:.5f}").set("ki_mat","userdef").set("ki","0").set("mur_mat","userdef").set("mur","1");

      if (k > 0) {{
        String sExpr = "(" + sigs[k][0] + "+(" + sigs[k][1] + ")*i)";
        m.component("c1").physics("ewfd").create("scd1","SurfaceCurrent",1);
        m.component("c1").physics("ewfd").feature("scd1").selection().named("selGr");
        m.component("c1").physics("ewfd").feature("scd1").set("Js0", new String[]{{sExpr + "*ewfd.Ex", "0", sExpr + "*ewfd.Ez"}});
      }}

      m.component("c1").mesh().create("m1","g1");
      m.component("c1").mesh("m1").feature().create("szC","Size");
      m.component("c1").mesh("m1").feature("szC").selection().named("selCore");
      m.component("c1").mesh("m1").feature("szC").set("custom",true).set("hmaxactive",true).set("hmax",0.020e-6);
      m.component("c1").mesh("m1").feature().create("szG","Size");
      m.component("c1").mesh("m1").feature("szG").selection().named("selGr");
      m.component("c1").mesh("m1").feature("szG").set("custom",true).set("hmaxactive",true).set("hmax",0.015e-6);
      m.component("c1").mesh("m1").feature("size").set("custom","on").set("hmax",0.20e-6);
      m.component("c1").mesh("m1").feature().create("ft1","FreeTri");
      m.component("c1").mesh("m1").run();

      m.study().create("std1");
      m.study("std1").feature().create("mode","ModeAnalysis");
      m.study("std1").feature("mode").set("modeFreq","c_const/" + wl);
      m.study("std1").feature("mode").set("neigs",1);
      m.study("std1").feature("mode").set("shift","2.45");
      m.study("std1").feature("mode").set("shiftactive",true);
      m.study("std1").run();

      String ds = m.result().dataset().tags()[0];
      m.result().numerical().create("gv","EvalGlobal");
      m.result().numerical("gv").set("data",ds);
      m.result().numerical("gv").set("expr",new String[]{{"real(ewfd.neff)","imag(ewfd.neff)"}});
      double[][] r = m.result().numerical("gv").getReal();
      double nr = r[0][0], ni = Math.abs(r[1][0]);
      double alphaDbCm = 2.0 * (2.0 * Math.PI / wl) * ni * (10.0 / Math.log(10.0)) * 1e-2;
      System.out.println("GR_JSON: {{\\"tag\\":\\"" + tags[k] + "\\",\\"neff\\":" + nr + ",\\"alpha_dB_cm\\":" + alphaDbCm + "}}");
    }}
  }}
}}
"""
    lines = _run_comsol_java("ComsolFrontierGraphene", java_src)
    rows = {r["tag"]: r for r in _parse_kv_lines(lines, "GR_JSON:")}

    mr_gr = simulate_2d_material_mrr(TwoDMaterialMRRConfig(
        material_2d="Graphene", capacitor_type="SLG", core_material="Si", clad_material="SiO2",
        top_clad_material="SiO2", film_height_um=0.22, ring_width_um=0.48, ring_radius_um=5000.0,
        spacer_nm=10.0, graphene_tau_ps=0.10, graphene_sheet_width_um=1.28,
        v_gate_sweep_v=(0.0, 9.0),
    ))["metrics"]
    err_mu0 = abs(mr_gr["alpha_2d_max_dB_per_cm"] - rows["mu0p0eV"]["alpha_dB_cm"]) / rows["mu0p0eV"]["alpha_dB_cm"] * 100.0
    return {
        "comsol_64": {
            "bare_neff": round(rows["bare"]["neff"], 5),
            "mu0p0eV_alpha_dB_cm": round(rows["mu0p0eV"]["alpha_dB_cm"], 2),
            "mu0p55eV_alpha_dB_cm": round(rows["mu0p55eV"]["alpha_dB_cm"], 2),
            "mu0p55eV_delta_neff": round(rows["mu0p55eV"]["neff"] - rows["bare"]["neff"], 7),
        },
        "microring_kubo_2dfem": {
            "bare_neff": mr_gr["bare_neff"],
            "mu0p0eV_alpha_dB_cm": mr_gr["alpha_2d_max_dB_per_cm"],
            "mu0p55eV_alpha_dB_cm": mr_gr["alpha_2d_min_dB_per_cm"],
            "rel_err_alpha_mu0_pct": round(err_mu0, 3),
        },
    }


def main() -> None:
    t_all = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saw_res = run_comsol_saw_suite()
    sbs_res = run_comsol_sbs_suite()
    gr_res = run_comsol_graphene_suite()

    payload = {
        "solver": "COMSOL Multiphysics 6.4.0.293 (Solid Mechanics solid + Electrostatics es + PiezoelectricEffect pze + Wave Optics ewfd SurfaceCurrent)",
        "host": f"{platform.system()}-{platform.machine()}",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        "suite_saw_piezo_2d": saw_res,
        "suite_sbs_solid_2d3d": sbs_res,
        "suite_graphene_ewfd_surface_current": gr_res,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved COMSOL frontier validation to {OUT_JSON} in {payload['total_wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
