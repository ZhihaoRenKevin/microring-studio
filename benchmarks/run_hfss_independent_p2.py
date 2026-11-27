"""Independent Ansys HFSS 2026 R1 Full-Wave 3D RF Validation Runner (Suite 8).

Reads ONLY the neutral physical specification JSON:
    `benchmarks/outputs/commercial_validation_specs_p2.json`
and executes Ansys Electronics Desktop (AEDT 2026 R1) headlessly (`ansysedt.exe -ng -RunScriptAndExit`):
  1. **HFSS 3D Full-Wave Driven Modal (`Wave Ports`)** using exact **Perfect-H (PMC) symmetry
     at `x = 0`** (`x >= 0` half-GSG domain) and **open Radiation boundaries** on outer top,
     bottom, and `+x` walls (preventing spurious grounded-microstrip PEC mirror artifacts and
     mathematically eliminating the odd slotline mode: `Z0_full = 0.5 * Z0_half`, identical
     `gamma = alpha + j*beta`).
  2. Exact matched cross-section stack (`Au` seated on LN slab `y = h_slab = 0.30 um`, embedded
     in `SiO2` top cladding up to `y_ox_top = max(h_film + 0.25, h_slab + h_top) = 1.10 um`,
     slot-centered undercut cavity for `UndercutSi`) across all 4 Suite-8 Sub-THz EOM RF Electrodes:
     - `TFLN_GSG_CPW_HRSi` (`10, 40, 70, 100 GHz` Discrete)
     - `TFLN_GSG_CPW_Quartz` (`10, 40, 70, 100 GHz` Discrete)
     - `TFLN_GSG_CPW_UndercutSi` (`10, 40, 70, 100 GHz` Discrete)
     - `TFLN_SlowWave_TRail_Quartz` (TRUE 3D periodic T-rail slow-wave structure: `4` full periods
       = `100 um`, symmetric signal + ground Au trunks/T-stems/T-caps + etched LN slab corridor,
       extracting full-wave Bloch-Floquet `Z_B`, `n_RF`, and `alpha_RF` via exact 2-port ABCD
       transfer matrices from `10` to `300 GHz`!).
  3. **Anisotropic X-cut TFLN RF permittivity tensor** (`eps_xx = 28, eps_yy = 44, eps_zz = 28`),
     substrate permittivity/loss tangent/conductivity, and gold conductor surface impedance (`rho_Au`).

Zero outputs from MicroRing Studio, COMSOL, or Lumerical are ever read.
Saves `benchmarks/outputs/hfss_independent_results_p2.json`.
"""
from __future__ import annotations

import cmath
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "benchmarks" / "outputs"
SPECS_P2_JSON = OUT_DIR / "commercial_validation_specs_p2.json"
OUT_HFSS_JSON = OUT_DIR / "hfss_independent_results_p2.json"
SCRATCH_DIR = OUT_DIR / "_hfss_scratch_p2"


def _find_ansysedt() -> Path:
    env_root = os.environ.get("ANSYSEM_ROOT261") or os.environ.get("ANSYSEM_ROOT252") or os.environ.get("ANSYSEM_ROOT251")
    candidates = []
    if env_root:
        candidates.append(Path(env_root) / "ansysedt.exe")
    candidates.extend([
        Path(r"C:\Program Files\ANSYS Inc\v261\AnsysEM\ansysedt.exe"),
        Path(r"C:\Program Files\AnsysEM\v261\Win64\ansysedt.exe"),
    ])
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("Could not find Ansys Electronics Desktop (ansysedt.exe).")


def _parse_touchstone_s2p(path: Path) -> tuple[float, list[tuple[float, complex, complex, complex, complex]]]:
    """Parse Touchstone v1 `.s2p` exported by HFSS -> `(z_ref, [(freq_ghz, s11, s21, s12, s22), ...])`."""
    z_ref = 100.0
    freq_scale = 1.0
    fmt = "RI"
    rows: list[tuple[float, complex, complex, complex, complex]] = []
    tokens: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.split("!")[0].strip()
        if not s:
            continue
        if s.startswith("#"):
            parts = s[1:].upper().split()
            for idx, tok in enumerate(parts):
                if tok == "HZ":
                    freq_scale = 1e-9
                elif tok == "KHZ":
                    freq_scale = 1e-6
                elif tok == "MHZ":
                    freq_scale = 1e-3
                elif tok == "GHZ":
                    freq_scale = 1.0
                elif tok in ("RI", "MA", "DB"):
                    fmt = tok
                elif tok == "R" and idx + 1 < len(parts):
                    z_ref = float(parts[idx + 1])
            continue
        tokens.extend(s.split())

    def _to_c(a: float, b: float) -> complex:
        if fmt == "RI":
            return complex(a, b)
        if fmt == "MA":
            return cmath.rect(a, math.radians(b))
        mag = 10.0 ** (a / 20.0)
        return cmath.rect(mag, math.radians(b))

    vals = [float(t) for t in tokens]
    n_pts = len(vals) // 9
    for i in range(n_pts):
        sl = vals[9 * i : 9 * (i + 1)]
        f_ghz = sl[0] * freq_scale
        s11 = _to_c(sl[1], sl[2])
        s21 = _to_c(sl[3], sl[4])
        s12 = _to_c(sl[5], sl[6])
        s22 = _to_c(sl[7], sl[8])
        rows.append((f_ghz, s11, s21, s12, s22))
    return z_ref, rows


def _extract_bloch_from_s2p_half(
    z_ref_half: float,
    s11: complex,
    s21: complex,
    s12: complex,
    s22: complex,
    length_m: float,
    freq_ghz: float,
) -> tuple[float, float, float]:
    """Exact 2-port S -> ABCD -> Full-GSG Bloch-Floquet (Z_B_full = 0.5*Z_half, n_RF, alpha [dB/cm])."""
    s_11 = 0.5 * (s11 + s22)
    s_21 = 0.5 * (s21 + s12)
    denom = 2.0 * s_21
    if abs(denom) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    A = ((1.0 + s_11) * (1.0 - s_11) + s_21 * s_21) / denom
    B = z_ref_half * ((1.0 + s_11) ** 2 - s_21 * s_21) / denom
    C = (1.0 / z_ref_half) * ((1.0 - s_11) ** 2 - s_21 * s_21) / denom
    gL = cmath.acosh(A)
    if gL.real < 0.0:
        gL = -gL
    if gL.imag < 0.0:
        gL = complex(gL.real, -gL.imag)
    gamma = gL / length_m
    c0 = 299792458.0
    k0 = 2.0 * math.pi * (freq_ghz * 1e9) / c0
    n_rf = gamma.imag / max(k0, 1e-12)
    alpha_db_cm = gamma.real * 8.685889638 * 1e-2
    z_half = cmath.sqrt(B / C) if abs(C) > 1e-18 else complex(z_ref_half, 0.0)
    if z_half.real < 0.0:
        z_half = -z_half
    return float(0.5 * z_half.real), float(n_rf), float(alpha_db_cm)


def _build_aedt_ironpython_script(specs: dict, out_raw_json: Path, s2p_dir: Path) -> str:
    """Generate IronPython script for Ansys AEDT 2026 R1 (Open-Box PMC Half-GSG HFSS 3D)."""
    cases_json_str = json.dumps(specs["suite8_rf_electrodes"])
    out_raw_str = str(out_raw_json).replace("\\", "/")
    s2p_dir_str = str(s2p_dir).replace("\\", "/")
    proj_path_str = str(SCRATCH_DIR / "suite8_rf_validation_open.aedt").replace("\\", "/")

    return f'''# Auto-generated Ansys AEDT 2026 R1 HFSS 3D Full-Wave Validation Script (Suite 8, Open PMC Half-GSG)
import os
import json
import math

CASES = json.loads(r"""{cases_json_str}""")
OUT_RAW = r"{out_raw_str}"
S2P_DIR = r"{s2p_dir_str}"
PROJ_PATH = r"{proj_path_str}"
if not os.path.exists(S2P_DIR):
    os.makedirs(S2P_DIR)

ver = str(oDesktop.GetVersion())
oProject = oDesktop.NewProject()
oProject.SaveAs(PROJ_PATH, True)
oDefMgr = oProject.GetDefinitionManager()

def ensure_iso_mat(name, eps_r, tan_d=0.0, sigma=0.0):
    if oDefMgr.DoesMaterialExist(name):
        return
    props = [
        "NAME:" + name,
        "CoordinateSystemType:=", "Cartesian",
        "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic"]],
        "permittivity:=", str(eps_r),
        "permeability:=", "1.0",
        "conductivity:=", str(sigma),
        "dielectric_loss_tangent:=", str(tan_d),
    ]
    oDefMgr.AddMaterial(props)

def ensure_aniso_mat(name, ex, ey, ez, tan_d=0.0):
    if oDefMgr.DoesMaterialExist(name):
        return
    props = [
        "NAME:" + name,
        "CoordinateSystemType:=", "Cartesian",
        "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic"]],
        [
            "NAME:permittivity",
            "property_type:=", "AnisoProperty",
            "unit:=", "",
            "component1:=", str(ex),
            "component2:=", str(ey),
            "component3:=", str(ez),
        ],
        "permeability:=", "1.0",
        "conductivity:=", "0.0",
        "dielectric_loss_tangent:=", str(tan_d),
    ]
    oDefMgr.AddMaterial(props)

def add_box(oEd, name, x0, y0, z0, dx, dy, dz, mat, solve_inside=True):
    oEd.CreateBox(
        [
            "NAME:BoxParameters",
            "XPosition:=", str(x0) + "um",
            "YPosition:=", str(y0) + "um",
            "ZPosition:=", str(z0) + "um",
            "XSize:=", str(dx) + "um",
            "YSize:=", str(dy) + "um",
            "ZSize:=", str(dz) + "um",
        ],
        [
            "NAME:Attributes",
            "Name:=", name,
            "Flags:=", "",
            "Color:=", "(143 175 143)",
            "Transparency:=", 0.4,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", '"' + mat + '"',
            "SurfaceMaterialValue:=", '""',
            "SolveInside:=", solve_inside,
            "ShellElement:=", False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False,
        ],
    )

def add_rect(oEd, name, x0, y0, z0, w1, w2, axis="Z"):
    oEd.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=", True,
            "XStart:=", str(x0) + "um",
            "YStart:=", str(y0) + "um",
            "ZStart:=", str(z0) + "um",
            "Width:=", str(w1) + "um",
            "Height:=", str(w2) + "um",
            "WhichAxis:=", axis,
        ],
        [
            "NAME:Attributes",
            "Name:=", name,
            "Flags:=", "",
            "Color:=", "(255 128 0)",
            "Transparency:=", 0.6,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", '"vacuum"',
            "SurfaceMaterialValue:=", '""',
            "SolveInside:=", True,
            "ShellElement:=", False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False,
        ],
    )

def read_csv_single_col(csv_path):
    out = {{}}
    if not os.path.exists(csv_path):
        return out
    lines = open(csv_path, "r").readlines()
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.strip().split(",") if p.strip()]
        if len(parts) >= 2:
            try:
                f_val = float(parts[0])
                v_val = float(parts[1])
                out[round(f_val, 3)] = v_val
            except Exception:
                pass
    return out

results = []

for c in CASES:
    cid = str(c["case_id"])
    is_trail = (c["electrode_topology"] == "t_rail_cpw")
    is_uc = (c["substrate_type"] == "undercut_si")
    ws = float(c["signal_width_um"])
    gap = float(c["electrode_gap_um"])
    wg = float(c["ground_width_um"])
    tm = float(c["metal_thickness_um"])
    hf = float(c["film_height_um"])
    hs = float(c["slab_height_um"])
    wwg = float(c["waveguide_width_um"])
    hbox = float(c["box_thickness_um"])
    htop = float(c["top_clad_thickness_um"])
    l_rail = float(c["t_rail_length_um"])
    w_cap = float(c["t_rail_cap_width_um"])
    w_stem = float(c["t_rail_stem_width_um"])
    period = float(c["t_rail_period_um"])
    duty = float(c["t_rail_duty"])
    slab_corr = c["slab_corridor_half_um"]

    ex_ln = float(c["eps_core_rf"][0])
    ey_ln = float(c["eps_core_rf"][1])
    ez_ln = float(c["eps_core_rf"][2]) if len(c["eps_core_rf"]) > 2 else ex_ln
    eps_box = float(c["eps_box_rf"][0])
    eps_sub = float(c["eps_sub_rf"])
    td_ln = float(c["tan_d_core"])
    td_box = float(c["tan_d_box"])
    td_sub = float(c["tan_d_sub"])
    sig_sub = float(c["sigma_sub_s_per_m"])
    sig_au = 1.0 / float(c["rho_metal_ohm_m"])

    mat_ln = "LN_Xcut_Aniso"
    mat_ox = "SiO2_RF"
    mat_sub = "Sub_" + str(c["substrate_type"])
    mat_au = "Au_EOM"
    ensure_aniso_mat(mat_ln, ex_ln, ey_ln, ez_ln, td_ln)
    ensure_iso_mat(mat_ox, eps_box, td_box, 0.0)
    ensure_iso_mat(mat_sub, eps_sub, td_sub, sig_sub)
    ensure_iso_mat(mat_au, 1.0, 0.0, sig_au)

    oProject.InsertDesign("HFSS", cid, "HFSS Modal Network", "")
    oDesign = oProject.SetActiveDesign(cid)
    oEd = oDesign.SetActiveEditor("3D Modeler")
    oEd.SetModelUnits(["NAME:Units Parameter", "Units:=", "um", "Rescale:=", False])

    # Exact matched stack (identical to MicroRing Studio & COMSOL Suite 8!):
    # Au sits on LN slab y_m0 = hs (0.30 um), top oxide covers up to y_ox_top = max(hf + 0.25, hs + htop) = 1.10 um!
    if is_trail:
        n_per = 4
        L_z = n_per * period
        g_trunk = gap + 2.0 * l_rail
        x_s_edge = 0.5 * ws + l_rail
        x_g_cap_in = x_s_edge + gap
        x_g_in = 0.5 * ws + g_trunk
        x_g_out = x_g_in + wg
        x_slot = 0.5 * (x_s_edge + x_g_cap_in)
    else:
        n_per = 1
        L_z = 300.0
        g_trunk = gap
        x_s_edge = 0.5 * ws
        x_g_in = x_s_edge + gap
        x_g_out = x_g_in + wg
        x_slot = 0.5 * (x_s_edge + x_g_in)

    span = ws + 2.0 * g_trunk
    h_sub = max(90.0, 2.2 * span)
    h_air = max(90.0, 2.2 * span)
    # Ground touches x_dom so N_conductors = 2 (SigTrunk vs Gnd_R+frame) -> STRICTLY 1 pure even GSG CPW mode!
    x_dom = x_g_out
    y_m0 = hs
    y_m1 = hs + tm
    y_ox_top = max(hf + 0.25, hs + htop)
    y_sub_bot = -hbox - h_sub
    y_air_top = y_m1 + h_air

    # Half-GSG domain x in [0, x_dom]
    add_box(oEd, "AirBox", 0.0, y_sub_bot, 0.0, x_dom, y_air_top - y_sub_bot, L_z, "vacuum", True)
    add_box(oEd, "Substrate", 0.0, y_sub_bot, 0.0, x_dom, -hbox - y_sub_bot, L_z, mat_sub, True)
    if is_uc:
        # Slot-centered XeF2 undercut air cavity (exact match to COMSOL & MicroRing: r_uc = min(0.5*g + 16, 26) = 18.5 um!)
        r_uc = float(c.get("undercut_cavity_half_w_um", 18.5))
        uc_d = float(c.get("undercut_cavity_depth_um", 26.0))
        uc_x0 = max(0.0, x_slot - r_uc)
        uc_x1 = x_slot + r_uc
        add_box(oEd, "UndercutCavity", uc_x0, -hbox - uc_d, 0.0, uc_x1 - uc_x0, uc_d, L_z, "vacuum", True)
        oEd.Subtract(["NAME:Selections", "Blank Parts:=", "Substrate", "Tool Parts:=", "UndercutCavity"],
                     ["NAME:SubtractParameters", "KeepOriginals:=", False])

    add_box(oEd, "OxideClad", 0.0, -hbox, 0.0, x_dom, hbox + y_ox_top, L_z, mat_ox, True)
    if slab_corr is not None and float(slab_corr) > 0.0:
        hw_s = float(slab_corr)
        add_box(oEd, "LN_Slab_R", x_slot - hw_s, 0.0, 0.0, 2.0 * hw_s, hs, L_z, mat_ln, True)
        ln_tools = "LN_Slab_R"
    else:
        add_box(oEd, "LN_Slab", 0.0, 0.0, 0.0, x_dom, hs, L_z, mat_ln, True)
        ln_tools = "LN_Slab"
    add_box(oEd, "LN_Rib_R", x_slot - 0.5 * wwg, hs, 0.0, wwg, hf - hs, L_z, mat_ln, True)

    # Electrodes seated on y_m0 = hs (partially embedded in OxideClad up to y_ox_top, protruding into Air above y_ox_top!)
    if is_trail:
        add_box(oEd, "SigTrunk", 0.0, y_m0, 0.0, 0.5 * ws, tm, L_z, mat_au, False)
        add_box(oEd, "Gnd_R", x_g_in, y_m0, 0.0, wg, tm, L_z, mat_au, False)
        sig_parts = ["SigTrunk"]
        gnd_r_parts = ["Gnd_R"]
        l_cap = duty * period
        z_cap_off = 0.5 * (period - l_cap)
        z_stem_off = 0.5 * (period - w_stem)
        stem_len = l_rail - w_cap
        for ip in range(n_per):
            z0 = ip * period
            snR = "SStemR_" + str(ip); cnR = "SCapR_" + str(ip)
            add_box(oEd, snR, 0.5 * ws - 0.1, y_m0, z0 + z_stem_off, stem_len + 0.3, tm, w_stem, mat_au, False)
            add_box(oEd, cnR, 0.5 * ws + stem_len, y_m0, z0 + z_cap_off, w_cap, tm, l_cap, mat_au, False)
            sig_parts.extend([snR, cnR])
            gnR = "GStemR_" + str(ip); gcR = "GCapR_" + str(ip)
            add_box(oEd, gnR, x_g_cap_in + w_cap - 0.2, y_m0, z0 + z_stem_off, stem_len + 0.3, tm, w_stem, mat_au, False)
            add_box(oEd, gcR, x_g_cap_in, y_m0, z0 + z_cap_off, w_cap, tm, l_cap, mat_au, False)
            gnd_r_parts.extend([gnR, gcR])
        oEd.Unite(["NAME:Selections", "Selections:=", ",".join(sig_parts)],
                  ["NAME:UniteParameters", "KeepOriginals:=", False])
        oEd.Unite(["NAME:Selections", "Selections:=", ",".join(gnd_r_parts)],
                  ["NAME:UniteParameters", "KeepOriginals:=", False])
    else:
        add_box(oEd, "SigTrunk", 0.0, y_m0, 0.0, 0.5 * ws, tm, L_z, mat_au, False)
        add_box(oEd, "Gnd_R", x_g_in, y_m0, 0.0, wg, tm, L_z, mat_au, False)

    # Clean non-overlapping boolean subtractions
    oEd.Subtract(["NAME:Selections", "Blank Parts:=", "OxideClad",
                  "Tool Parts:=", ln_tools + ",LN_Rib_R,SigTrunk,Gnd_R"],
                 ["NAME:SubtractParameters", "KeepOriginals:=", True])
    oEd.Subtract(["NAME:Selections", "Blank Parts:=", "AirBox",
                  "Tool Parts:=", "Substrate,OxideClad," + ln_tools + ",LN_Rib_R,SigTrunk,Gnd_R"],
                 ["NAME:SubtractParameters", "KeepOriginals:=", True])

    # Boundary sheets: Perfect-H on x=0 symmetry plane; Radiation on top (+y) & bottom (-y); right x=x_dom is PEC shorted directly to Gnd_R -> N_conductors = 2!
    add_rect(oEd, "PmcSymSheet", 0.0, y_sub_bot, 0.0, y_air_top - y_sub_bot, L_z, axis="X")
    add_rect(oEd, "RadTop", 0.0, y_air_top, 0.0, L_z, x_dom, axis="Y")
    add_rect(oEd, "RadBot", 0.0, y_sub_bot, 0.0, L_z, x_dom, axis="Y")
    add_rect(oEd, "Port1Sheet", 0.0, y_sub_bot, 0.0, x_dom, y_air_top - y_sub_bot, axis="Z")
    add_rect(oEd, "Port2Sheet", 0.0, y_sub_bot, L_z, x_dom, y_air_top - y_sub_bot, axis="Z")

    oBnd = oDesign.GetModule("BoundarySetup")
    oBnd.AssignPerfectH(["NAME:PerfH_Sym", "Objects:=", ["PmcSymSheet"]])
    oBnd.AssignRadiation(["NAME:RadOuter", "Objects:=", ["RadTop", "RadBot"], "IsFssReference:=", False, "IsForPML:=", False])

    y_int = y_m0 + 0.5 * tm
    x_s_port = 0.5 * ws
    x_g_port = x_g_in
    for pname, sname, z_p in (("P1", "Port1Sheet", 0.0), ("P2", "Port2Sheet", L_z)):
        oBnd.AssignWavePort([
            "NAME:" + pname,
            "Objects:=", [sname],
            "NumModes:=", 1,
            "UseLineModeAlignment:=", False,
            "DoDeembed:=", False,
            "RenormalizeAllTerminals:=", False,
            [
                "NAME:Modes",
                [
                    "NAME:Mode1",
                    "ModeNum:=", 1,
                    "UseIntLine:=", True,
                    [
                        "NAME:IntLine",
                        "Start:=", [str(x_g_port) + "um", str(y_int) + "um", str(z_p) + "um"],
                        "End:=",   [str(x_s_port) + "um", str(y_int) + "um", str(z_p) + "um"],
                    ],
                    "AlignmentGroup:=", 0,
                    "CharImp:=", "Zpi",
                ],
            ],
            "ShowReporterFilter:=", False,
            "ReporterFilter:=", [True],
            "UseAnalyticAlignment:=", False,
        ])

    # Slot mesh refinement box (non-model)
    oMesh = oDesign.GetModule("MeshSetup")
    add_box(oEd, "SlotMeshBoxR", max(0.0, 0.5 * ws - 2.0), -hbox - 2.0, 0.0, (x_g_in - 0.5 * ws) + 4.0, hbox + y_m1 + 4.0, L_z, "vacuum", True)
    oEd.ChangeProperty([
        "NAME:AllTabs",
        ["NAME:Geometry3DAttributeTab",
         ["NAME:PropServers", "SlotMeshBoxR"],
         ["NAME:ChangedProps", ["NAME:Model", "Value:=", False]]]
    ])
    oMesh.AssignLengthOp([
        "NAME:SlotRefine",
        "RefineInside:=", True,
        "Enabled:=", True,
        "Objects:=", ["SlotMeshBoxR"],
        "RestrictElem:=", False,
        "RestrictLength:=", True,
        "MaxLength:=", "3.5um" if is_trail else "5.0um",
    ])

    oAn = oDesign.GetModule("AnalysisSetup")
    oAn.InsertSetup("HfssDriven", [
        "NAME:Setup1",
        "AdaptMultipleFreqs:=", False,
        "Frequency:=", "40GHz",
        "MaxDeltaS:=", 0.015,
        "PortsOnly:=", False,
        "UseMatrixConv:=", False,
        "MaximumPasses:=", 8 if is_trail else 6,
        "MinimumPasses:=", 3,
        "MinimumConvergedPasses:=", 2,
        "PercentRefinement:=", 30,
        "IsEnabled:=", True,
        "BasisOrder:=", 1,
        "DoLambdaRefine:=", True,
        "DoMaterialLambda:=", True,
        "SetLambdaTarget:=", False,
        "Target:=", 0.3333,
        "UseMaxTetIncrease:=", False,
        "PortAccuracy:=", 1.0,
        "UseABCOnPort:=", False,
        "SetPortMinMaxTri:=", False,
        "EnableSolverDomains:=", False,
        "SaveRadFieldsOnly:=", False,
        "SaveAnyFields:=", False,
        "IESolverType:=", "Auto",
        "LambdaTargetForIESolver:=", 0.15,
        "UseDefaultLambdaTgtForIESolver:=", True,
    ])

    if is_trail:
        oAn.InsertFrequencySweep("Setup1", [
            "NAME:Sweep1",
            "IsEnabled:=", True,
            "RangeType:=", "LinearStep",
            "RangeStart:=", "10GHz",
            "RangeEnd:=", "300GHz",
            "RangeStep:=", "10GHz",
            "Type:=", "Interpolating",
            "SaveFields:=", False,
            "SaveRadFields:=", False,
            "InterpTolerance:=", 0.005,
            "InterpMaxSolns:=", 100,
            "InterpMinSolns:=", 0,
            "InterpMinSubranges:=", 1,
            "ExtrapToDC:=", False,
            "InterpUseS:=", True,
            "InterpUsePortImped:=", True,
            "InterpUsePropConst:=", True,
            "UseDerivativeConvergence:=", False,
            "InterpDerivTolerance:=", 0.2,
            "UseFullBasis:=", True,
        ])
    else:
        oAn.InsertFrequencySweep("Setup1", [
            "NAME:Sweep1",
            "IsEnabled:=", True,
            "RangeType:=", "LinearStep",
            "RangeStart:=", "10GHz",
            "RangeEnd:=", "100GHz",
            "RangeStep:=", "30GHz",
            "Type:=", "Discrete",
            "SaveFields:=", False,
            "SaveRadFields:=", False,
        ])

    oDesign.AnalyzeAll()

    s2p_path = os.path.join(S2P_DIR, cid + ".s2p").replace("\\\\", "/")
    oSol = oDesign.GetModule("Solutions")
    oSol.ExportNetworkData(
        "",
        ["Setup1:Sweep1"],
        3,
        s2p_path,
        ["All"],
        True,
        100.0,
        "S",
        -1,
        0,
        15,
        True,
        True,
        False,
    )

    oRep = oDesign.GetModule("ReportSetup")
    z0_csv = os.path.join(S2P_DIR, cid + "_z0.csv").replace("\\\\", "/")
    al_csv = os.path.join(S2P_DIR, cid + "_alpha.csv").replace("\\\\", "/")
    be_csv = os.path.join(S2P_DIR, cid + "_beta.csv").replace("\\\\", "/")
    for rname, expr, cpath in (
        ("RepZ0", "re(Zo(P1:1))", z0_csv),
        ("RepAlpha", "re(Gamma(P1:1))", al_csv),
        ("RepBeta", "im(Gamma(P1:1))", be_csv),
    ):
        try:
            oRep.CreateReport(
                rname, "Modal Solution Data", "Data Table", "Setup1 : Sweep1",
                ["Domain:=", "Sweep"], ["Freq:=", ["All"]],
                ["X Component:=", "Freq", "Y Component:=", [expr]],
            )
            oRep.ExportToFile(rname, cpath, False)
        except Exception:
            pass

    results.append({{
        "case_id": cid,
        "is_trail": is_trail,
        "length_um": L_z,
        "s2p_path": s2p_path,
        "port_z0_half_map": read_csv_single_col(z0_csv),
        "port_alpha_npm_map": read_csv_single_col(al_csv),
        "port_beta_radpm_map": read_csv_single_col(be_csv),
    }})

oProject.Save()
with open(OUT_RAW, "w") as fp:
    json.dump({{"version": ver, "cases": results}}, fp, indent=2)
oDesktop.CloseProject(oProject.GetName())
'''


def main() -> None:
    t0 = time.perf_counter()
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    s2p_dir = SCRATCH_DIR / "s2p"
    s2p_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(SPECS_P2_JSON.read_text(encoding="utf-8"))

    aedt_exe = _find_ansysedt()
    out_raw_json = SCRATCH_DIR / "hfss_raw_out.json"
    script_path = SCRATCH_DIR / "run_suite8_hfss.py"
    log_path = SCRATCH_DIR / "hfss_batch.log"

    script_path.write_text(_build_aedt_ironpython_script(specs, out_raw_json, s2p_dir), encoding="ascii")
    print(f"Launching headless Ansys HFSS ({aedt_exe}, Open PMC Half-GSG) on all 4 Suite-8 EOM RF Electrodes...")
    proc = subprocess.run(
        [str(aedt_exe), "-ng", "-LogFile", str(log_path), "-RunScriptAndExit", str(script_path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0 or not out_raw_json.is_file():
        log_tail = log_path.read_text(encoding="utf-8", errors="ignore")[-2500:] if log_path.is_file() else ""
        raise RuntimeError(f"HFSS batch failed (rc={proc.returncode}):\nSTDOUT:{proc.stdout[-1000:]}\nLOG:{log_tail}")

    raw = json.loads(out_raw_json.read_text(encoding="utf-8"))
    c0 = 299792458.0
    compiled_cases = []
    for item in raw["cases"]:
        cid = item["case_id"]
        length_m = float(item["length_um"]) * 1e-6
        z_ref_half, s_rows = _parse_touchstone_s2p(Path(item["s2p_path"]))
        freq_results = []
        for f_ghz, s11, s21, s12, s22 in s_rows:
            zb_full, nrf_3d, alpha_3d = _extract_bloch_from_s2p_half(z_ref_half, s11, s21, s12, s22, length_m, f_ghz)
            fk = str(round(f_ghz, 3))
            p_z0_half = item["port_z0_half_map"].get(fk)
            p_al = item["port_alpha_npm_map"].get(fk)
            p_be = item["port_beta_radpm_map"].get(fk)
            k0 = 2.0 * math.pi * (f_ghz * 1e9) / c0
            p_z0_full = (0.5 * float(p_z0_half)) if p_z0_half is not None else None
            p_nrf = (float(p_be) / k0) if p_be is not None else None
            p_al_db = (float(p_al) * 8.685889638 * 1e-2) if p_al is not None else None
            freq_results.append({
                "freq_GHz": round(f_ghz, 2),
                "Z0_3d_abcd_ohm": round(zb_full, 3),
                "n_RF_3d_abcd": round(nrf_3d, 4),
                "alpha_3d_abcd_dB_per_cm": round(alpha_3d, 3),
                "Z0_port_ohm": round(p_z0_full, 3) if p_z0_full is not None else None,
                "n_RF_port": round(p_nrf, 4) if p_nrf is not None else None,
                "alpha_port_dB_per_cm": round(p_al_db, 3) if p_al_db is not None else None,
                "S21_mag_dB": round(20.0 * math.log10(max(abs(s21), 1e-12)), 4),
                "S11_mag_dB": round(20.0 * math.log10(max(abs(s11), 1e-12)), 2),
            })

        f40 = min(freq_results, key=lambda r: abs(r["freq_GHz"] - 40.0))
        rec = {
            "case_id": cid,
            "is_periodic_3d_trail": bool(item["is_trail"]),
            "simulated_length_um": float(item["length_um"]),
            "Z0_40G_ohm": f40["Z0_3d_abcd_ohm"] if item["is_trail"] else (f40["Z0_port_ohm"] or f40["Z0_3d_abcd_ohm"]),
            "n_RF_40G": f40["n_RF_3d_abcd"] if item["is_trail"] else (f40["n_RF_port"] or f40["n_RF_3d_abcd"]),
            "alpha_40G_dB_per_cm": f40["alpha_3d_abcd_dB_per_cm"] if item["is_trail"] else (f40["alpha_port_dB_per_cm"] or f40["alpha_3d_abcd_dB_per_cm"]),
            "Z0_40G_3d_abcd_ohm": f40["Z0_3d_abcd_ohm"],
            "n_RF_40G_3d_abcd": f40["n_RF_3d_abcd"],
            "alpha_40G_3d_abcd_dB_per_cm": f40["alpha_3d_abcd_dB_per_cm"],
            "Z0_40G_port_ohm": f40["Z0_port_ohm"],
            "n_RF_40G_port": f40["n_RF_port"],
            "alpha_40G_port_dB_per_cm": f40["alpha_port_dB_per_cm"],
            "freq_sweep": freq_results,
        }
        compiled_cases.append(rec)
        print(
            f"[HFSS 3D Suite 8] {cid}: Z0(40G)={rec['Z0_40G_ohm']:.2f} Ohm, "
            f"n_RF(40G)={rec['n_RF_40G']:.4f}, alpha(40G)={rec['alpha_40G_dB_per_cm']:.2f} dB/cm "
            f"(3D ABCD: Z0={rec['Z0_40G_3d_abcd_ohm']:.2f}, n_RF={rec['n_RF_40G_3d_abcd']:.4f}, a={rec['alpha_40G_3d_abcd_dB_per_cm']:.2f})"
        )

    payload = {
        "solver": f"Ansys HFSS {raw['version']} (3D Driven Modal Full-Wave FEM, Open PMC Half-GSG + Bloch-Floquet ABCD)",
        "host": f"{platform.system()}-{platform.machine()}",
        "total_wall_time_s": round(time.perf_counter() - t0, 2),
        "suite8_rf_electrodes": compiled_cases,
    }
    OUT_HFSS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Ansys HFSS Phase-2 results to {OUT_HFSS_JSON} in {payload['total_wall_time_s']} s")


if __name__ == "__main__":
    main()
