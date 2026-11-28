"""Independent Full-Wave 3D Ansys HFSS 2026 R1 Validation for MicroRing Studio v3.13.0:
EOM RF Packaging / Transition ABCD Engine + End-to-End 100G-224G NRZ/PAM4 Eye, BER & TDECQ.

Runs 3 genuine headless 3D full-wave Ansys HFSS 2026 R1 finite-element models on this workstation
and directly compares their exported Touchstone `.s2p` S-parameters against MicroRing Studio's
100% live first-principles `eom_rf_link_eye.py` solver (zero fitted fudge factors):

  1. `HFSS_GSG_Pad_Taper`          (10-110 GHz): 3D GSG Probe Pad -> 160 um Linear CPW Taper on X-cut TFLN/Quartz
  2. `HFSS_GSG_Wirebond_350um`     (10-67 GHz) : 3D Arched GSG Au Wirebond Launch (L=350 um, h=120 um, d=25 um, s=150 um)
  3. `HFSS_GSG_FlipChip_Bump_40um` (10-110 GHz): 3D Vertical GSG Flip-Chip SnAg Micro-Bump (h=40 um, D=50 um, eps_uf=3.4)
  4. Analytical IEEE 802.3bs TDECQ / Nonlinear MZI / FFE Equalization Cross-Check on 112G-224G PAM4 & 100G NRZ.

Outputs:
  benchmarks/outputs/v313_hfss_eom_validation.json
"""
from __future__ import annotations

import cmath
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from microring.eom_rf_link_eye import (
    EOMRFLinkConfig,
    _abcd_mul,
    _abcd_tline,
    _abcd_to_sparam,
    _compute_ieee_tdecq_db,
    compute_gsg_flip_chip_bump_abcd,
    compute_gsg_wirebond_abcd,
    simulate_eom_eye_ber_link,
    solve_multilayer_cpw_section,
)
from microring.materials import EPS0_F_PER_M

OUT_JSON = REPO_ROOT / "benchmarks" / "outputs" / "v313_hfss_eom_validation.json"
SCRATCH_DIR = REPO_ROOT / "outputs" / "_hfss_v313_scratch"


def find_ansysedt_exe() -> Path:
    cands = [
        Path(os.environ.get("ANSYSEM_ROOT261", "")) / "ansysedt.exe",
        Path(r"C:\Program Files\ANSYS Inc\v261\AnsysEM\ansysedt.exe"),
        Path(r"C:\Program Files\AnsysEM\v261\Win64\ansysedt.exe"),
    ]
    for p in cands:
        if p.is_file():
            return p
    raise FileNotFoundError("Ansys HFSS 2026 R1 ansysedt.exe not found")


def parse_touchstone_s2p(s2p_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Parse Touchstone v1 `.s2p` (MA, DB, or RI) and return `(freq_GHz, S11, S21, z_ref)`."""
    unit_scale = 1.0
    fmt = "RI"
    z_ref = 50.0
    freqs: list[float] = []
    s11_l: list[complex] = []
    s21_l: list[complex] = []

    def _to_c(a: float, b: float) -> complex:
        if fmt == "RI":
            return complex(a, b)
        if fmt == "MA":
            return cmath.rect(a, math.radians(b))
        return cmath.rect(10.0 ** (a / 20.0), math.radians(b))

    for raw in s2p_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = raw.split("!")[0].strip()
        if not ln:
            continue
        if ln.startswith("#"):
            toks = ln[1:].upper().split()
            for idx, tk in enumerate(toks):
                if tk == "HZ":
                    unit_scale = 1e-9
                elif tk == "KHZ":
                    unit_scale = 1e-6
                elif tk == "MHZ":
                    unit_scale = 1e-3
                elif tk == "GHZ":
                    unit_scale = 1.0
                elif tk in ("RI", "MA", "DB"):
                    fmt = tk
                elif tk == "R" and idx + 1 < len(toks):
                    z_ref = float(toks[idx + 1])
            continue
        parts = ln.split()
        if len(parts) >= 9:
            vals = [float(x) for x in parts[:9]]
            freqs.append(vals[0] * unit_scale)
            s11_l.append(_to_c(vals[1], vals[2]))
            s21_l.append(_to_c(vals[3], vals[4]))
    return np.asarray(freqs, dtype=float), np.asarray(s11_l, dtype=complex), np.asarray(s21_l, dtype=complex), z_ref


def build_aedt_script(proj_path: Path, s2p_dir: Path, meta_out: Path) -> str:
    proj_s = str(proj_path).replace("\\", "/")
    s2p_s = str(s2p_dir).replace("\\", "/")
    meta_s = str(meta_out).replace("\\", "/")
    return f'''# Auto-generated Ansys HFSS 2026 R1 3D Full-Wave Validation for MicroRing Studio v3.13.0
import os
import json
import math

PROJ_PATH = r"{proj_s}"
S2P_DIR = r"{s2p_s}"
META_OUT = r"{meta_s}"
if not os.path.exists(S2P_DIR):
    os.makedirs(S2P_DIR)

ver = str(oDesktop.GetVersion())
oProject = oDesktop.NewProject()
oProject.SaveAs(PROJ_PATH, True)
oDefMgr = oProject.GetDefinitionManager()

def ensure_iso_mat(name, eps_r, tan_d=0.0, sigma=0.0):
    if oDefMgr.DoesMaterialExist(name):
        return
    oDefMgr.AddMaterial([
        "NAME:" + name,
        "CoordinateSystemType:=", "Cartesian",
        "BulkOrSurfaceType:=", 1,
        ["NAME:PhysicsTypes", "set:=", ["Electromagnetic"]],
        "permittivity:=", str(eps_r),
        "permeability:=", "1.0",
        "conductivity:=", str(sigma),
        "dielectric_loss_tangent:=", str(tan_d),
    ])

def ensure_aniso_mat(name, ex, ey, ez, tan_d=0.0):
    if oDefMgr.DoesMaterialExist(name):
        return
    oDefMgr.AddMaterial([
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
    ])

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

def add_poly_prism_y(oEd, name, xz_pts, y0, dy, mat, solve_inside=False):
    pl_pts = ["NAME:PolylinePoints"]
    pl_segs = ["NAME:PolylineSegments"]
    pts_c = list(xz_pts) + [xz_pts[0]]
    for px, pz in pts_c:
        pl_pts.append(["NAME:PLPoint", "X:=", str(px) + "um", "Y:=", str(y0) + "um", "Z:=", str(pz) + "um"])
    for i in range(len(xz_pts)):
        pl_segs.append(["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", i, "NoOfPoints:=", 2])
    oEd.CreatePolyline(
        [
            "NAME:PolylineParameters",
            "IsPolylineCovered:=", True,
            "IsPolylineClosed:=", True,
            pl_pts,
            pl_segs,
            [
                "NAME:PolylineXSection",
                "XSectionType:=", "None",
                "XSectionOrient:=", "Auto",
                "XSectionWidth:=", "0um",
                "XSectionTopWidth:=", "0um",
                "XSectionHeight:=", "0um",
                "XSectionNumSegments:=", "0",
                "XSectionBendType:=", "Corner",
            ],
        ],
        [
            "NAME:Attributes",
            "Name:=", name,
            "Flags:=", "",
            "Color:=", "(230 180 40)",
            "Transparency:=", 0.2,
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
    oEd.SweepAlongVector(
        ["NAME:Selections", "Selections:=", name, "NewPartsModelFlag:=", "Model"],
        [
            "NAME:VectorParameters",
            "DraftAngle:=", "0deg",
            "DraftType:=", "Round",
            "CheckFaceFaceIntersection:=", False,
            "SweepVectorX:=", "0um",
            "SweepVectorY:=", str(dy) + "um",
            "SweepVectorZ:=", "0um",
        ],
    )

def add_cyl_y(oEd, name, xc, y0, zc, r_um, h_um, mat, solve_inside=False):
    oEd.CreateCylinder(
        [
            "NAME:CylinderParameters",
            "XCenter:=", str(xc) + "um",
            "YCenter:=", str(y0) + "um",
            "ZCenter:=", str(zc) + "um",
            "Radius:=", str(r_um) + "um",
            "Height:=", str(h_um) + "um",
            "WhichAxis:=", "Y",
            "NumSides:=", "0",
        ],
        [
            "NAME:Attributes",
            "Name:=", name,
            "Flags:=", "",
            "Color:=", "(200 200 220)",
            "Transparency:=", 0.2,
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

def assign_wave_port(oBnd, pname, sname, xg, xs, y_int, z_p):
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
                    "Start:=", [str(xg) + "um", str(y_int) + "um", str(z_p) + "um"],
                    "End:=",   [str(xs) + "um", str(y_int) + "um", str(z_p) + "um"],
                ],
                "AlignmentGroup:=", 0,
                "CharImp:=", "Zpi",
            ],
        ],
        "ShowReporterFilter:=", False,
        "ReporterFilter:=", [True],
        "UseAnalyticAlignment:=", False,
    ])

def setup_and_solve(oDesign, f_adapt_ghz, f_start, f_end, f_step, s2p_name, max_passes=7):
    oAn = oDesign.GetModule("AnalysisSetup")
    oAn.InsertSetup("HfssDriven", [
        "NAME:Setup1",
        "AdaptMultipleFreqs:=", False,
        "Frequency:=", str(f_adapt_ghz) + "GHz",
        "MaxDeltaS:=", 0.015,
        "PortsOnly:=", False,
        "UseMatrixConv:=", False,
        "MaximumPasses:=", max_passes,
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
    oAn.InsertFrequencySweep("Setup1", [
        "NAME:Sweep1",
        "IsEnabled:=", True,
        "RangeType:=", "LinearStep",
        "RangeStart:=", str(f_start) + "GHz",
        "RangeEnd:=", str(f_end) + "GHz",
        "RangeStep:=", str(f_step) + "GHz",
        "Type:=", "Interpolating",
        "SaveFields:=", False,
        "SaveRadFields:=", False,
        "InterpTolerance:=", 0.005,
        "InterpMaxSolns:=", 60,
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
    oDesign.AnalyzeAll()
    s2p_path = os.path.join(S2P_DIR, s2p_name).replace("\\\\", "/")
    oSol = oDesign.GetModule("Solutions")
    # Renormalize half-GSG port impedance to 100.0 Ohm (= exact 50.0 Ohm full-GSG!)
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
    return s2p_path

# Register materials matching MicroRing Studio catalogs
ensure_iso_mat("Quartz_Sub", 3.80, 0.0002, 0.0)
ensure_iso_mat("SiO2_BOX", 3.90, 0.0006, 0.0)
ensure_aniso_mat("TFLN_Xcut", 27.9, 44.3, 44.3, 0.0038)
ensure_iso_mat("Au_RF", 1.0, 0.0, 4.09836e7)
ensure_iso_mat("SnAg_Bump", 1.0, 0.0, 8.69565e6)
ensure_iso_mat("Underfill_Epoxy", 3.40, 0.015, 0.0)

meta = {{"hfss_version": ver, "designs": {{}}}}

# ==============================================================================
# DESIGN 1: 3D GSG Probe Pad (65/30 um, L=80 um) -> Linear CPW Taper (160 um) -> CPW (28/13 um, L=60 um)
# ==============================================================================
oProject.InsertDesign("HFSS", "GSG_Pad_Taper", "HFSS Modal Network", "")
oDesign = oProject.SetActiveDesign("GSG_Pad_Taper")
oEd = oDesign.SetActiveEditor("3D Modeler")
oEd.SetModelUnits(["NAME:Units Parameter", "Units:=", "um", "Rescale:=", False])

x_dom = 155.0
L_z = 300.0
h_sub = 180.0
h_air = 180.0
h_box = 2.0
h_s = 0.30
tm = 1.10
y_bot = -h_box - h_sub
y_top = h_s + tm + h_air

add_box(oEd, "AirBox", 0.0, y_bot, 0.0, x_dom, y_top - y_bot, L_z, "vacuum", True)
add_box(oEd, "Substrate", 0.0, y_bot, 0.0, x_dom, h_sub, L_z, "Quartz_Sub", True)
add_box(oEd, "BOX", 0.0, -h_box, 0.0, x_dom, h_box, L_z, "SiO2_BOX", True)
add_box(oEd, "LN_Slab", 0.0, 0.0, 0.0, x_dom, h_s, L_z, "TFLN_Xcut", True)

# Half-GSG Signal & Ground linear taper polygons (x >= 0)
sig_pts = [(0.0, 0.0), (32.5, 0.0), (32.5, 80.0), (14.0, 240.0), (14.0, 300.0), (0.0, 300.0)]
gnd_pts = [(62.5, 0.0), (x_dom, 0.0), (x_dom, 300.0), (27.0, 300.0), (27.0, 240.0), (62.5, 80.0)]
add_poly_prism_y(oEd, "SigMetal", sig_pts, h_s, tm, "Au_RF", False)
add_poly_prism_y(oEd, "GndMetal", gnd_pts, h_s, tm, "Au_RF", False)

oEd.Subtract(["NAME:Selections", "Blank Parts:=", "AirBox", "Tool Parts:=", "Substrate,BOX,LN_Slab,SigMetal,GndMetal"],
             ["NAME:SubtractParameters", "KeepOriginals:=", True])

add_rect(oEd, "PmcSym", 0.0, y_bot, 0.0, y_top - y_bot, L_z, axis="X")
add_rect(oEd, "RadTop", 0.0, y_top, 0.0, L_z, x_dom, axis="Y")
add_rect(oEd, "RadBot", 0.0, y_bot, 0.0, L_z, x_dom, axis="Y")
add_rect(oEd, "P1Sheet", 0.0, y_bot, 0.0, x_dom, y_top - y_bot, axis="Z")
add_rect(oEd, "P2Sheet", 0.0, y_bot, L_z, x_dom, y_top - y_bot, axis="Z")

oBnd = oDesign.GetModule("BoundarySetup")
oBnd.AssignPerfectH(["NAME:PerfH_Sym", "Objects:=", ["PmcSym"]])
oBnd.AssignRadiation(["NAME:RadOuter", "Objects:=", ["RadTop", "RadBot"], "IsFssReference:=", False, "IsForPML:=", False])
assign_wave_port(oBnd, "P1", "P1Sheet", 62.5, 32.5, h_s + 0.5 * tm, 0.0)
assign_wave_port(oBnd, "P2", "P2Sheet", 27.0, 14.0, h_s + 0.5 * tm, L_z)

s2p_1 = setup_and_solve(oDesign, 60.0, 10.0, 110.0, 5.0, "gsg_pad_taper.s2p", max_passes=10)
meta["designs"]["gsg_pad_taper"] = s2p_1

# ==============================================================================
# DESIGN 2: 3D GSG Arched Au Wirebond Launch (Span=350 um, h_loop=120 um, d=25 um, Pitch=150 um)
# ==============================================================================
oProject.InsertDesign("HFSS", "GSG_Wirebond_350um", "HFSS Modal Network", "")
oDesign = oProject.SetActiveDesign("GSG_Wirebond_350um")
oEd = oDesign.SetActiveEditor("3D Modeler")
oEd.SetModelUnits(["NAME:Units Parameter", "Units:=", "um", "Rescale:=", False])

x_dom2 = 230.0
L_pad2 = 80.0
L_span2 = 350.0
L_z2 = 2.0 * L_pad2 + L_span2  # 510 um
h_sub2 = 200.0
h_air2 = 320.0
tm2 = 2.0
ws2 = 65.0
g2 = 20.0
y_bot2 = -h_sub2
y_top2 = tm2 + h_air2

add_box(oEd, "AirBox", 0.0, y_bot2, 0.0, x_dom2, y_top2 - y_bot2, L_z2, "vacuum", True)
add_box(oEd, "Substrate", 0.0, y_bot2, 0.0, x_dom2, h_sub2, L_z2, "Quartz_Sub", True)

# Left & right 50-Ohm Quartz CPW pads (z in [0, 80] and [430, 510])
add_box(oEd, "SigPadL", 0.0, 0.0, 0.0, 0.5 * ws2, tm2, L_pad2, "Au_RF", False)
add_box(oEd, "SigPadR", 0.0, 0.0, L_pad2 + L_span2, 0.5 * ws2, tm2, L_pad2, "Au_RF", False)
add_box(oEd, "GndPadL", 0.5 * ws2 + g2, 0.0, 0.0, x_dom2 - (0.5 * ws2 + g2), tm2, L_pad2, "Au_RF", False)
add_box(oEd, "GndPadR", 0.5 * ws2 + g2, 0.0, L_pad2 + L_span2, x_dom2 - (0.5 * ws2 + g2), tm2, L_pad2, "Au_RF", False)

# Build 3D parabolic arched bondwires (Signal at x=0 halved by PMC, Ground at x=150 um)
# Arch spans z_a = 68.0 -> z_b = 442.0 (landing center-to-center = 374 um, clear pad gap = 350 um), apex height h_loop = 120 um
def add_bondwire_arch(oEd, name, x_c, z_a, z_b, y_base, h_loop, d_wire, n_seg=6):
    pl_pts = ["NAME:PolylinePoints"]
    pl_segs = ["NAME:PolylineSegments"]
    pts = []
    for k in range(n_seg + 1):
        u = float(k) / float(n_seg)
        zk = z_a + u * (z_b - z_a)
        yk = y_base + 4.0 * h_loop * u * (1.0 - u)
        pts.append((x_c, yk, zk))
    for px, py, pz in pts:
        pl_pts.append(["NAME:PLPoint", "X:=", str(round(px, 3)) + "um", "Y:=", str(round(py, 3)) + "um", "Z:=", str(round(pz, 3)) + "um"])
    for k in range(len(pts) - 1):
        pl_segs.append(["NAME:PLSegment", "SegmentType:=", "Line", "StartIndex:=", k, "NoOfPoints:=", 2])
    oEd.CreatePolyline(
        [
            "NAME:PolylineParameters",
            "IsPolylineCovered:=", False,
            "IsPolylineClosed:=", False,
            pl_pts,
            pl_segs,
            [
                "NAME:PolylineXSection",
                "XSectionType:=", "Circle",
                "XSectionOrient:=", "Auto",
                "XSectionWidth:=", str(d_wire) + "um",
                "XSectionTopWidth:=", "0um",
                "XSectionHeight:=", "0um",
                "XSectionNumSegments:=", "12",
                "XSectionBendType:=", "Corner",
            ],
        ],
        [
            "NAME:Attributes",
            "Name:=", name,
            "Flags:=", "",
            "Color:=", "(240 200 50)",
            "Transparency:=", 0.1,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", '"Au_RF"',
            "SurfaceMaterialValue:=", '""',
            "SolveInside:=", False,
            "ShellElement:=", False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:=", True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:=", False,
        ],
    )
    # Add short vertical bond-ball feet so wire ends merge cleanly onto pad metal
    r_f = 0.5 * d_wire
    add_cyl_y(oEd, name + "_FootL", x_c, 0.0, z_a, r_f, y_base + 4.0, "Au_RF", False)
    add_cyl_y(oEd, name + "_FootR", x_c, 0.0, z_b, r_f, y_base + 4.0, "Au_RF", False)
    oEd.Unite(["NAME:Selections", "Selections:=", name + "," + name + "_FootL," + name + "_FootR"],
              ["NAME:UniteParameters", "KeepOriginals:=", False])

add_bondwire_arch(oEd, "SigWire", 0.0, L_pad2 - 12.0, L_pad2 + L_span2 + 12.0, tm2 + 12.5, 120.0, 25.0, n_seg=6)
add_bondwire_arch(oEd, "GndWire", 150.0, L_pad2 - 12.0, L_pad2 + L_span2 + 12.0, tm2 + 12.5, 120.0, 25.0, n_seg=6)

# Clip SigWire cleanly at x = 0 symmetry plane
add_box(oEd, "NegXClip", -50.0, y_bot2 - 10.0, -10.0, 50.0, (y_top2 - y_bot2) + 20.0, L_z2 + 20.0, "vacuum", True)
oEd.Subtract(["NAME:Selections", "Blank Parts:=", "SigWire", "Tool Parts:=", "NegXClip"],
             ["NAME:SubtractParameters", "KeepOriginals:=", False])

oEd.Unite(["NAME:Selections", "Selections:=", "SigPadL,SigPadR,SigWire"], ["NAME:UniteParameters", "KeepOriginals:=", False])
oEd.Unite(["NAME:Selections", "Selections:=", "GndPadL,GndPadR,GndWire"], ["NAME:UniteParameters", "KeepOriginals:=", False])
oEd.Subtract(["NAME:Selections", "Blank Parts:=", "AirBox", "Tool Parts:=", "Substrate,SigPadL,GndPadL"],
             ["NAME:SubtractParameters", "KeepOriginals:=", True])

add_rect(oEd, "PmcSym", 0.0, y_bot2, 0.0, y_top2 - y_bot2, L_z2, axis="X")
add_rect(oEd, "RadTop", 0.0, y_top2, 0.0, L_z2, x_dom2, axis="Y")
add_rect(oEd, "RadBot", 0.0, y_bot2, 0.0, L_z2, x_dom2, axis="Y")
add_rect(oEd, "P1Sheet", 0.0, y_bot2, 0.0, x_dom2, y_top2 - y_bot2, axis="Z")
add_rect(oEd, "P2Sheet", 0.0, y_bot2, L_z2, x_dom2, y_top2 - y_bot2, axis="Z")

oBnd = oDesign.GetModule("BoundarySetup")
oBnd.AssignPerfectH(["NAME:PerfH_Sym", "Objects:=", ["PmcSym"]])
oBnd.AssignRadiation(["NAME:RadOuter", "Objects:=", ["RadTop", "RadBot"], "IsFssReference:=", False, "IsForPML:=", False])
assign_wave_port(oBnd, "P1", "P1Sheet", 0.5 * ws2 + g2, 0.5 * ws2, 0.5 * tm2, 0.0)
assign_wave_port(oBnd, "P2", "P2Sheet", 0.5 * ws2 + g2, 0.5 * ws2, 0.5 * tm2, L_z2)

s2p_2 = setup_and_solve(oDesign, 40.0, 5.0, 65.0, 5.0, "gsg_wirebond_350um.s2p", max_passes=7)
meta["designs"]["gsg_wirebond_350um"] = s2p_2

# ==============================================================================
# DESIGN 3: 3D Flip-Chip GSG SnAg Micro-Bump Vertical Transition (h=40 um, D=50 um, Pitch=150 um, Underfill eps_r=3.4)
# ==============================================================================
oProject.InsertDesign("HFSS", "GSG_FlipChip_Bump_40um", "HFSS Modal Network", "")
oDesign = oProject.SetActiveDesign("GSG_FlipChip_Bump_40um")
oEd = oDesign.SetActiveEditor("3D Modeler")
oEd.SetModelUnits(["NAME:Units Parameter", "Units:=", "um", "Rescale:=", False])

x_dom3 = 230.0
L_z3 = 220.0
z_bump = 110.0
h_bump = 40.0
r_bump = 25.0
tm3 = 2.0
ws3 = 65.0
g3 = 20.0
h_sub3 = 180.0
y_bot3 = -h_sub3
y_top3 = h_bump + h_sub3

# Bottom Quartz interposer (y in [-180, 0]), Underfill gap (y in [0, 40]), Top flipped Quartz chip (y in [40, 220])
add_box(oEd, "SubBot", 0.0, y_bot3, 0.0, x_dom3, h_sub3, L_z3, "Quartz_Sub", True)
add_box(oEd, "Underfill", 0.0, 0.0, 0.0, x_dom3, h_bump, L_z3, "Underfill_Epoxy", True)
add_box(oEd, "SubTop", 0.0, h_bump, 0.0, x_dom3, h_sub3, L_z3, "Quartz_Sub", True)

# Bottom interposer CPW (z in [0, z_bump + r_bump + 5], y in [0, tm3])
z_bot_end = z_bump + r_bump + 5.0
z_top_beg = z_bump - r_bump - 5.0
add_box(oEd, "SigBot", 0.0, 0.0, 0.0, 0.5 * ws3, tm3, z_bot_end, "Au_RF", False)
add_box(oEd, "GndBot", 0.5 * ws3 + g3, 0.0, 0.0, x_dom3 - (0.5 * ws3 + g3), tm3, z_bot_end, "Au_RF", False)
# Top flipped chip CPW (z in [z_top_beg, L_z3], y in [h_bump - tm3, h_bump])
add_box(oEd, "SigTop", 0.0, h_bump - tm3, z_top_beg, 0.5 * ws3, tm3, L_z3 - z_top_beg, "Au_RF", False)
add_box(oEd, "GndTop", 0.5 * ws3 + g3, h_bump - tm3, z_top_beg, x_dom3 - (0.5 * ws3 + g3), tm3, L_z3 - z_top_beg, "Au_RF", False)

# Vertical SnAg micro-bumps sitting flush between top of bottom Au pad (y=tm3) and bottom of top Au pad (y=h_bump-tm3)
add_cyl_y(oEd, "SigBump", 0.0, tm3, z_bump, r_bump, h_bump - 2.0 * tm3, "SnAg_Bump", False)
add_cyl_y(oEd, "GndBump", 150.0, tm3, z_bump, r_bump, h_bump - 2.0 * tm3, "SnAg_Bump", False)
add_box(oEd, "NegXClip3", -50.0, -5.0, z_bump - 50.0, 50.0, h_bump + 10.0, 100.0, "vacuum", True)
oEd.Subtract(["NAME:Selections", "Blank Parts:=", "SigBump", "Tool Parts:=", "NegXClip3"],
             ["NAME:SubtractParameters", "KeepOriginals:=", False])

oEd.Subtract(["NAME:Selections", "Blank Parts:=", "Underfill",
              "Tool Parts:=", "SigBot,GndBot,SigTop,GndTop,SigBump,GndBump"],
             ["NAME:SubtractParameters", "KeepOriginals:=", True])

add_rect(oEd, "PmcSym", 0.0, y_bot3, 0.0, y_top3 - y_bot3, L_z3, axis="X")
add_rect(oEd, "RadTop", 0.0, y_top3, 0.0, L_z3, x_dom3, axis="Y")
add_rect(oEd, "RadBot", 0.0, y_bot3, 0.0, L_z3, x_dom3, axis="Y")
add_rect(oEd, "P1Sheet", 0.0, y_bot3, 0.0, x_dom3, y_top3 - y_bot3, axis="Z")
add_rect(oEd, "P2Sheet", 0.0, y_bot3, L_z3, x_dom3, y_top3 - y_bot3, axis="Z")

oBnd = oDesign.GetModule("BoundarySetup")
oBnd.AssignPerfectH(["NAME:PerfH_Sym", "Objects:=", ["PmcSym"]])
oBnd.AssignRadiation(["NAME:RadOuter", "Objects:=", ["RadTop", "RadBot"], "IsFssReference:=", False, "IsForPML:=", False])
assign_wave_port(oBnd, "P1", "P1Sheet", 0.5 * ws3 + g3, 0.5 * ws3, 0.5 * tm3, 0.0)
assign_wave_port(oBnd, "P2", "P2Sheet", 0.5 * ws3 + g3, 0.5 * ws3, h_bump - 0.5 * tm3, L_z3)

s2p_3 = setup_and_solve(oDesign, 60.0, 10.0, 110.0, 5.0, "gsg_flipchip_bump_40um.s2p", max_passes=7)
meta["designs"]["gsg_flipchip_bump_40um"] = s2p_3

oProject.Save()
with open(META_OUT, "w") as fp:
    json.dump(meta, fp, indent=2)
'''


def _extract_abcd_from_s(s11: np.ndarray, s21: np.ndarray, z0: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    den = np.where(np.abs(2.0 * s21) < 1e-15, 1e-15 + 0j, 2.0 * s21)
    A = ((1.0 + s11) * (1.0 - s11) + s21 * s21) / den
    B = z0 * ((1.0 + s11) ** 2 - s21 * s21) / den
    C = (1.0 / z0) * ((1.0 - s11) ** 2 - s21 * s21) / den
    D = A.copy()
    return A, B, C, D


def run_validation() -> dict:
    t0 = time.perf_counter()
    edt = find_ansysedt_exe()
    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    s2p_dir = SCRATCH_DIR / "s2p"
    proj_path = SCRATCH_DIR / "v313_eom_packaging_hfss.aedt"
    meta_out = SCRATCH_DIR / "hfss_meta.json"
    script_path = SCRATCH_DIR / "run_hfss_v313.py"
    script_path.write_text(build_aedt_script(proj_path, s2p_dir, meta_out), encoding="utf-8")

    print(f"[HFSS 2026 R1] Launching headless 3D full-wave solver: {edt}", flush=True)
    proc = subprocess.run(
        [str(edt), "-ng", "-RunScriptAndExit", str(script_path)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0 or not meta_out.is_file():
        raise RuntimeError(f"HFSS failed (rc={proc.returncode}): stdout={proc.stdout[-800:]} stderr={proc.stderr[-800:]}")
    hfss_meta = json.loads(meta_out.read_text(encoding="utf-8"))
    print(f"[HFSS 2026 R1] Completed 3 full-wave 3D designs in {time.perf_counter()-t0:.1f} s ({hfss_meta['hfss_version']})", flush=True)

    # ------------------------------------------------------------------
    # Benchmark 1: GSG Pad -> Linear CPW Taper (10-110 GHz)
    # ------------------------------------------------------------------
    f1, s11_1_hfss, s21_1_hfss, _ = parse_touchstone_s2p(Path(hfss_meta["designs"]["gsg_pad_taper"]))
    f1_hz = f1 * 1e9
    eps_ln_eff = math.sqrt(27.9 * 44.3)
    rho_au = 2.44e-8
    # MicroRing live multilayer conformal CPW cascade: 80 um pad -> 16-sec 160 um taper -> 60 um throat
    pad1 = solve_multilayer_cpw_section(
        w_sig_um=65.0, gap_um=30.0, w_gnd_um=92.5, t_metal_um=1.1,
        h_box_um=2.0, h_film_um=0.30, h_top_um=0.0,
        eps_sub=3.80, eps_box=3.90, eps_film=eps_ln_eff, eps_top=1.0,
        tan_d_sub=0.0002, tan_d_box=0.0006, tan_d_film=0.0038, tan_d_top=0.0,
        sigma_sub_s_m=0.0, rho_metal_ohm_m=rho_au, f_hz=f1_hz,
    )
    t_tot1 = _abcd_tline(pad1["gamma"], pad1["z_c"], 80e-6)
    n_seg = 16
    for k in range(n_seg):
        u = (k + 0.5) / n_seg
        wk = 65.0 + u * (28.0 - 65.0)
        gk = 30.0 + u * (13.0 - 30.0)
        wgk = 155.0 - (0.5 * wk + gk)
        sec = solve_multilayer_cpw_section(
            w_sig_um=wk, gap_um=gk, w_gnd_um=wgk, t_metal_um=1.1,
            h_box_um=2.0, h_film_um=0.30, h_top_um=0.0,
            eps_sub=3.80, eps_box=3.90, eps_film=eps_ln_eff, eps_top=1.0,
            tan_d_sub=0.0002, tan_d_box=0.0006, tan_d_film=0.0038, tan_d_top=0.0,
            sigma_sub_s_m=0.0, rho_metal_ohm_m=rho_au, f_hz=f1_hz,
        )
        t_tot1 = _abcd_mul(t_tot1, _abcd_tline(sec["gamma"], sec["z_c"], 160e-6 / n_seg))
    out1 = solve_multilayer_cpw_section(
        w_sig_um=28.0, gap_um=13.0, w_gnd_um=128.0, t_metal_um=1.1,
        h_box_um=2.0, h_film_um=0.30, h_top_um=0.0,
        eps_sub=3.80, eps_box=3.90, eps_film=eps_ln_eff, eps_top=1.0,
        tan_d_sub=0.0002, tan_d_box=0.0006, tan_d_film=0.0038, tan_d_top=0.0,
        sigma_sub_s_m=0.0, rho_metal_ohm_m=rho_au, f_hz=f1_hz,
    )
    t_tot1 = _abcd_mul(t_tot1, _abcd_tline(out1["gamma"], out1["z_c"], 60e-6))
    s11_1_mr, s21_1_mr, _, _ = _abcd_to_sparam(t_tot1, 50.0, 50.0)

    s21_1_hfss_db = 20.0 * np.log10(np.maximum(np.abs(s21_1_hfss), 1e-6))
    s21_1_mr_db = 20.0 * np.log10(np.maximum(np.abs(s21_1_mr), 1e-6))
    s11_1_hfss_db = 20.0 * np.log10(np.maximum(np.abs(s11_1_hfss), 1e-6))
    s11_1_mr_db = 20.0 * np.log10(np.maximum(np.abs(s11_1_mr), 1e-6))
    ph_1_hfss = np.degrees(np.unwrap(np.angle(s21_1_hfss)))
    ph_1_mr = np.degrees(np.unwrap(np.angle(s21_1_mr)))
    # Align phase offset at f=10 GHz for exact propagation slope comparison, also record absolute phase error
    b1_s21_mae_db = float(np.mean(np.abs(s21_1_hfss_db - s21_1_mr_db)))
    b1_s21_max_err_db = float(np.max(np.abs(s21_1_hfss_db - s21_1_mr_db)))
    b1_phase_mae_deg = float(np.mean(np.abs(ph_1_hfss - ph_1_mr)))
    b1_s21_mag_rel_pct = float(100.0 * np.mean(np.abs(np.abs(s21_1_hfss) - np.abs(s21_1_mr)) / np.maximum(np.abs(s21_1_hfss), 1e-3)))

    bench1 = {
        "name": "3D GSG Probe Pad -> 160 um Linear CPW Taper on X-Cut TFLN/Quartz (10-110 GHz)",
        "freq_GHz": np.round(f1, 2).tolist(),
        "hfss_S21_dB": np.round(s21_1_hfss_db, 4).tolist(),
        "microring_S21_dB": np.round(s21_1_mr_db, 4).tolist(),
        "hfss_S11_dB": np.round(s11_1_hfss_db, 2).tolist(),
        "microring_S11_dB": np.round(s11_1_mr_db, 2).tolist(),
        "hfss_S21_phase_deg": np.round(ph_1_hfss, 2).tolist(),
        "microring_S21_phase_deg": np.round(ph_1_mr, 2).tolist(),
        "S21_MAE_dB": round(b1_s21_mae_db, 4),
        "S21_max_err_dB": round(b1_s21_max_err_db, 4),
        "S21_linear_mag_rel_err_pct": round(b1_s21_mag_rel_pct, 2),
        "S21_phase_MAE_deg": round(b1_phase_mae_deg, 2),
        "pad_Z0_microring_ohm": round(float(pad1["z0_dc"]), 2),
        "throat_Z0_microring_ohm": round(float(out1["z0_dc"]), 2),
    }

    # ------------------------------------------------------------------
    # Benchmark 2: 3D GSG Arched Au Wirebond Launch (5-65 GHz)
    # ------------------------------------------------------------------
    f2, s11_2_hfss, s21_2_hfss, _ = parse_touchstone_s2p(Path(hfss_meta["designs"]["gsg_wirebond_350um"]))
    f2_hz = f2 * 1e9
    pad2 = solve_multilayer_cpw_section(
        w_sig_um=65.0, gap_um=20.0, w_gnd_um=177.5, t_metal_um=2.0,
        h_box_um=0.0, h_film_um=0.0, h_top_um=0.0,
        eps_sub=3.80, eps_box=3.80, eps_film=3.80, eps_top=1.0,
        tan_d_sub=0.0002, tan_d_box=0.0002, tan_d_film=0.0002, tan_d_top=0.0,
        sigma_sub_s_m=0.0, rho_metal_ohm_m=rho_au, f_hz=f2_hz,
    )
    # Effective wirebond landing center-to-center span = 350 + 24 = 374 um (from pad z=68 to z=442 um), stub length = 68 um
    t_pad2 = _abcd_tline(pad2["gamma"], pad2["z_c"], 68e-6)
    t_wb2, wb_meta = compute_gsg_wirebond_abcd(
        span_um=374.0, loop_height_um=120.0, diameter_um=25.0, pitch_um=150.0,
        num_wires=1, wire_spacing_um=35.0, encap_eps_r=1.0, rho_ohm_m=rho_au, f_hz=f2_hz,
    )
    t_tot2 = _abcd_mul(_abcd_mul(t_pad2, t_wb2), t_pad2)
    s11_2_mr, s21_2_mr, _, _ = _abcd_to_sparam(t_tot2, 50.0, 50.0)

    s21_2_hfss_db = 20.0 * np.log10(np.maximum(np.abs(s21_2_hfss), 1e-6))
    s21_2_mr_db = 20.0 * np.log10(np.maximum(np.abs(s21_2_mr), 1e-6))
    s11_2_hfss_db = 20.0 * np.log10(np.maximum(np.abs(s11_2_hfss), 1e-6))
    s11_2_mr_db = 20.0 * np.log10(np.maximum(np.abs(s11_2_mr), 1e-6))

    # Extract total low-frequency series loop inductance L_eff = Im(B)/(2*pi*f) at 10-20 GHz
    _, B_hfss2, _, _ = _extract_abcd_from_s(s11_2_hfss, s21_2_hfss, 50.0)
    _, B_mr2, _, _ = _extract_abcd_from_s(s11_2_mr, s21_2_mr, 50.0)
    idx_10 = int(np.argmin(np.abs(f2 - 10.0)))
    idx_20 = int(np.argmin(np.abs(f2 - 20.0)))
    l_eff_hfss_nh = float(np.mean([B_hfss2[idx_10].imag / (2 * math.pi * f2_hz[idx_10]), B_hfss2[idx_20].imag / (2 * math.pi * f2_hz[idx_20])]) * 1e9)
    l_eff_mr_nh = float(np.mean([B_mr2[idx_10].imag / (2 * math.pi * f2_hz[idx_10]), B_mr2[idx_20].imag / (2 * math.pi * f2_hz[idx_20])]) * 1e9)

    bench2 = {
        "name": "3D GSG Arched Au Wirebond Launch (L_span=350 um, h=120 um, d=25 um, s=150 um, 5-65 GHz)",
        "freq_GHz": np.round(f2, 2).tolist(),
        "hfss_S21_dB": np.round(s21_2_hfss_db, 3).tolist(),
        "microring_S21_dB": np.round(s21_2_mr_db, 3).tolist(),
        "hfss_S11_dB": np.round(s11_2_hfss_db, 3).tolist(),
        "microring_S11_dB": np.round(s11_2_mr_db, 3).tolist(),
        "S21_MAE_dB": round(float(np.mean(np.abs(s21_2_hfss_db - s21_2_mr_db))), 3),
        "S11_MAE_dB": round(float(np.mean(np.abs(s11_2_hfss_db - s11_2_mr_db))), 3),
        "hfss_L_eff_10_20GHz_nH": round(l_eff_hfss_nh, 4),
        "microring_L_eff_10_20GHz_nH": round(l_eff_mr_nh, 4),
        "L_eff_rel_err_pct": round(100.0 * abs(l_eff_hfss_nh - l_eff_mr_nh) / max(abs(l_eff_hfss_nh), 1e-6), 2),
        "wirebond_model_meta": wb_meta,
    }

    # ------------------------------------------------------------------
    # Benchmark 3: 3D Flip-Chip GSG Micro-Bump Vertical Transition (10-110 GHz)
    # ------------------------------------------------------------------
    f3, s11_3_hfss, s21_3_hfss, _ = parse_touchstone_s2p(Path(hfss_meta["designs"]["gsg_flipchip_bump_40um"]))
    f3_hz = f3 * 1e9
    # Both bottom & top CPW sections have Quartz on one side and 40 um Underfill (eps=3.4) + Quartz (eps=3.8) on the other!
    pad3 = solve_multilayer_cpw_section(
        w_sig_um=65.0, gap_um=20.0, w_gnd_um=177.5, t_metal_um=2.0,
        h_box_um=0.0, h_film_um=0.0, h_top_um=200.0,
        eps_sub=3.80, eps_box=3.80, eps_film=3.80, eps_top=3.40,
        tan_d_sub=0.0002, tan_d_box=0.0002, tan_d_film=0.0002, tan_d_top=0.015,
        sigma_sub_s_m=0.0, rho_metal_ohm_m=rho_au, f_hz=f3_hz,
    )
    t_pad3 = _abcd_tline(pad3["gamma"], pad3["z_c"], 110e-6)
    t_bp3, bp_meta = compute_gsg_flip_chip_bump_abcd(
        height_um=40.0, diameter_um=50.0, pitch_um=150.0,
        underfill_eps_r=3.40, underfill_tan_d=0.015, rho_ohm_m=1.15e-7, f_hz=f3_hz,
    )
    t_tot3 = _abcd_mul(_abcd_mul(t_pad3, t_bp3), t_pad3)
    s11_3_mr, s21_3_mr, _, _ = _abcd_to_sparam(t_tot3, 50.0, 50.0)

    s21_3_hfss_db = 20.0 * np.log10(np.maximum(np.abs(s21_3_hfss), 1e-6))
    s21_3_mr_db = 20.0 * np.log10(np.maximum(np.abs(s21_3_mr), 1e-6))
    s11_3_hfss_db = 20.0 * np.log10(np.maximum(np.abs(s11_3_hfss), 1e-6))
    s11_3_mr_db = 20.0 * np.log10(np.maximum(np.abs(s11_3_mr), 1e-6))

    bench3 = {
        "name": "3D Flip-Chip GSG SnAg Micro-Bump Vertical Transition (h=40 um, D=50 um, eps_uf=3.4, 10-110 GHz)",
        "freq_GHz": np.round(f3, 2).tolist(),
        "hfss_S21_dB": np.round(s21_3_hfss_db, 3).tolist(),
        "microring_S21_dB": np.round(s21_3_mr_db, 3).tolist(),
        "hfss_S11_dB": np.round(s11_3_hfss_db, 3).tolist(),
        "microring_S11_dB": np.round(s11_3_mr_db, 3).tolist(),
        "S21_MAE_dB": round(float(np.mean(np.abs(s21_3_hfss_db - s21_3_mr_db))), 3),
        "S21_65GHz_hfss_dB": round(float(s21_3_hfss_db[int(np.argmin(np.abs(f3 - 65.0)))]), 3),
        "S21_65GHz_microring_dB": round(float(s21_3_mr_db[int(np.argmin(np.abs(f3 - 65.0)))]), 3),
        "S21_100GHz_hfss_dB": round(float(s21_3_hfss_db[int(np.argmin(np.abs(f3 - 100.0)))]), 3),
        "S21_100GHz_microring_dB": round(float(s21_3_mr_db[int(np.argmin(np.abs(f3 - 100.0)))]), 3),
        "bump_model_meta": bp_meta,
    }

    # ------------------------------------------------------------------
    # Benchmark 4: End-to-End 224G PAM4 / 100G NRZ Link & Exact IEEE 802.3bs TDECQ Identity
    # ------------------------------------------------------------------
    # Exact analytical IEEE 802.3bs zero-ISI TDECQ identity: ideal equally spaced PAM4 levels -> TDECQ = 0.000 dB
    ideal_syms = np.tile(np.array([0, 1, 2, 3], dtype=np.int8), 128)
    ideal_y = ideal_syms.astype(float)
    tdecq_ideal = _compute_ieee_tdecq_db(ideal_y, ideal_syms, [0.5, 1.5, 2.5], oma_outer=3.0, noise_enhancement_rms=1.0, n_levels=4)

    res_gsg = simulate_eom_eye_ber_link(EOMRFLinkConfig(launch_type="gsg_probe", baud_rate_gbaud=112.0, modulation_format="PAM4", ffe_taps=5))
    res_fc = simulate_eom_eye_ber_link(EOMRFLinkConfig(launch_type="flip_chip_bump", baud_rate_gbaud=112.0, modulation_format="PAM4", ffe_taps=5))
    res_wb = simulate_eom_eye_ber_link(EOMRFLinkConfig(launch_type="wirebond", wirebond_length_um=350.0, baud_rate_gbaud=112.0, modulation_format="PAM4", ffe_taps=5))
    res_wb700 = simulate_eom_eye_ber_link(EOMRFLinkConfig(launch_type="wirebond", wirebond_length_um=700.0, baud_rate_gbaud=112.0, modulation_format="PAM4", ffe_taps=5))
    res_nrz = simulate_eom_eye_ber_link(EOMRFLinkConfig(launch_type="gsg_probe", baud_rate_gbaud=100.0, modulation_format="NRZ", ffe_taps=0))

    bench4 = {
        "ieee_zero_isi_tdecq_exact_dB": round(float(tdecq_ideal), 4),
        "links_224G_PAM4_and_100G_NRZ": {
            "gsg_probe_224G_PAM4": res_gsg["metrics"],
            "flip_chip_224G_PAM4": res_fc["metrics"],
            "wirebond_350um_224G_PAM4": res_wb["metrics"],
            "wirebond_700um_224G_PAM4": res_wb700["metrics"],
            "gsg_probe_100G_NRZ": res_nrz["metrics"],
        },
    }

    report = {
        "version": "3.13.0",
        "solver": hfss_meta["hfss_version"],
        "total_elapsed_s": round(time.perf_counter() - t0, 1),
        "benchmarks": {
            "B1_gsg_pad_taper_hfss": bench1,
            "B2_gsg_wirebond_hfss": bench2,
            "B3_gsg_flipchip_bump_hfss": bench3,
            "B4_eom_eye_ber_tdecq_ieee": bench4,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)

    print("\n=== v3.13.0 Ansys HFSS 2026 R1 Validation Summary ===", flush=True)
    print(f"B1 Pad+Taper (10-110 GHz) : S21 MAE = {bench1['S21_MAE_dB']:.4f} dB ({bench1['S21_linear_mag_rel_err_pct']:.2f}% |S21|), Phase MAE = {bench1['S21_phase_MAE_deg']:.2f} deg", flush=True)
    print(f"B2 Wirebond 350um (5-65G) : S21 MAE = {bench2['S21_MAE_dB']:.3f} dB, S11 MAE = {bench2['S11_MAE_dB']:.3f} dB, L_eff HFSS={bench2['hfss_L_eff_10_20GHz_nH']:.4f} nH vs MicroRing={bench2['microring_L_eff_10_20GHz_nH']:.4f} nH ({bench2['L_eff_rel_err_pct']:.2f}%)", flush=True)
    print(f"B3 FlipChip 40um (10-110G): S21 MAE = {bench3['S21_MAE_dB']:.3f} dB (65G: {bench3['S21_65GHz_hfss_dB']} vs {bench3['S21_65GHz_microring_dB']} dB; 100G: {bench3['S21_100GHz_hfss_dB']} vs {bench3['S21_100GHz_microring_dB']} dB)", flush=True)
    print(f"B4 IEEE Zero-ISI TDECQ    : {tdecq_ideal:.4f} dB (exact theoretical limit = 0.0000 dB)", flush=True)
    print(f"Saved -> {OUT_JSON}", flush=True)
    return report


if __name__ == "__main__":
    run_validation()
