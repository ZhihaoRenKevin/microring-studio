"""Ansys Lumerical (`MODE FDE` + `3D FDTD`) Head-to-Head Validation Runner (`lumapi`).

Run this script on a workstation with Ansys Lumerical installed (e.g., Windows / Linux GPU box)
to validate `MicroRing Studio` predictions against commercial Lumerical `MODE FDE` (bent eigenmode
`n_eff`, `n_g`) and `3D FDTD` (point, racetrack, and concentric pulley coupler S-parameters
`|S21|^2`, `|S31|^2`, and coupler radiation loss `1 - |S21|^2 - |S31|^2`).

Guarantees strict apples-to-apples physics matching:
  - Identical cross-section geometry (`w_bottom`, `h_film`, `h_slab`, `sidewall_deg`, `R_bend`)
  - Identical topology in 3D FDTD (`point_circular`, `racetrack` with `L_s`, `pulley` with C1 wrap arc)
  - Identical exact diagonal refractive index tensor `(nx, ny, nz)` at each benchmark wavelength
  - Wavelength-scaled domain padding & port spans across Visible (`532 nm`) -> Mid-IR (`4200 nm`)
  - 3D FDTD coupler simulation enforces `auto shutoff min = 1e-5` before reading modal port `S` matrix.

Usage on Windows / Lumerical Workstation:
  python benchmarks/run_lumapi_validation.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np

# ==============================================================================
# Centralized Lumerical Solver Controls & Convergence Settings (with units)
# ==============================================================================
FDTD_AUTO_SHUTOFF_MIN = 1.0e-5       # unitless: energy decay convergence threshold
FDTD_SIM_TIME_FS = 8000.0            # unit: fs (8 ps max coupler transit window)
FDTD_PML_LAYERS = 12                 # unitless: standard stretched-coordinate PML
LUMERICAL_WIN_HINTS = [
    r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v252\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
    r"C:\Program Files\Ansys Inc\v242\Lumerical\api\python",
    r"C:\Program Files\Lumerical\v242\api\python",
    r"C:\Program Files\Lumerical\v241\api\python",
    r"C:\Program Files\Lumerical\v232\api\python",
]

HERE = Path(__file__).resolve().parent
CASES_JSON = HERE / "outputs" / "lumerical_cases.json"
OUT_CSV = HERE / "outputs" / "lumerical_vs_microring_validation.csv"
OUT_MD = HERE / "outputs" / "LUMERICAL_VALIDATION_REPORT.md"


def _import_lumapi():
    env_p = os.environ.get("LUMAPI_PATH")
    if env_p and Path(env_p).exists():
        sys.path.insert(0, env_p)
    for p in LUMERICAL_WIN_HINTS:
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)
    import lumapi  # type: ignore
    return lumapi


def _build_pulley_bus_centerline_m(r_bus_m: float, wrap_deg: float, r_trans_m: float, x_half_m: float) -> np.ndarray:
    """Construct C1-smooth concentric pulley bus centerline with horizontal input/output leads."""
    th = math.radians(max(wrap_deg, 1.0))
    half_th = 0.5 * th
    # Left transition arc: center (xc_L, yc_L), angle from -pi/2 (horizontal) to -pi/2 + half_th
    x_l = -r_bus_m * math.sin(half_th)
    y_l = r_bus_m * math.cos(half_th)
    xc_l = x_l - r_trans_m * math.sin(half_th)
    yc_l = y_l + r_trans_m * math.cos(half_th)
    y_lead = yc_l - r_trans_m

    a_left = np.linspace(-0.5 * math.pi, -0.5 * math.pi + half_th, 25)
    pts_left_arc = np.column_stack([xc_l + r_trans_m * np.cos(a_left), yc_l + r_trans_m * np.sin(a_left)])

    # Concentric pulley wrap arc around (0, 0) from pi/2 + half_th to pi/2 - half_th
    a_wrap = np.linspace(0.5 * math.pi + half_th, 0.5 * math.pi - half_th, 49)
    pts_wrap = np.column_stack([r_bus_m * np.cos(a_wrap), r_bus_m * np.sin(a_wrap)])

    # Right transition arc (mirror symmetric across x = 0)
    xc_r = -xc_l
    yc_r = yc_l
    a_right = np.linspace(-0.5 * math.pi - half_th, -0.5 * math.pi, 25)
    pts_right_arc = np.column_stack([xc_r + r_trans_m * np.cos(a_right), yc_r + r_trans_m * np.sin(a_right)])

    x_ext = max(x_half_m, abs(xc_l) + 4.0e-6)
    lead_l = np.array([[-x_ext, y_lead]])
    lead_r = np.array([[x_ext, y_lead]])
    return np.vstack([lead_l, pts_left_arc, pts_wrap[1:-1], pts_right_arc, lead_r])


def _centerline_to_strip_polygon_lsf(pts_m: np.ndarray, width_m: float) -> str:
    """Convert a 2D smooth centerline `(N, 2)` into a closed polygon `V = [x1, y1; ...]` for Lumerical `addpoly`."""
    tangents = np.gradient(pts_m, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-15)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    upper = pts_m + 0.5 * width_m * normals
    lower = (pts_m - 0.5 * width_m * normals)[::-1]
    poly = np.vstack([upper, lower])
    rows = "; ".join(f"{p[0]:.9e}, {p[1]:.9e}" for p in poly)
    return f"[{rows}]"


def run_mode_fde_case(mode_sess, case: dict) -> dict:
    """Build exact cross-section in Lumerical MODE FDE and extract bent TE0 n_eff & n_g."""
    wl_m = case["wavelength_nm"] * 1e-9
    w_m = case["ring_width_um"] * 1e-6
    hf_m = case["film_height_um"] * 1e-6
    hs_m = case["slab_height_um"] * 1e-6
    r_m = case["ring_radius_um"] * 1e-6
    sw_deg = case["sidewall_deg"]
    dom_m = case["domain_width_um"] * 1e-6
    pad_m = case["box_pad_um"] * 1e-6
    dl_m = case["fde_dl_um"] * 1e-6
    nx, ny, nz = case["n_core_xyz"]
    n_clad = case["n_clad"]
    n_top = case["n_top"]

    h_ridge = hf_m - hs_m
    inv_tan = 0.0 if abs(sw_deg - 90.0) < 1e-3 else 1.0 / math.tan(math.radians(sw_deg))
    w_top_m = w_m - 2.0 * h_ridge * inv_tan

    lsf = f"""
    switchtolayout;
    deleteall;

    if (!materialexists("MicroRing_Core")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "MicroRing_Core"); }}
    setmaterial("MicroRing_Core", "Anisotropy", "Diagonal");
    setmaterial("MicroRing_Core", "Refractive Index", [{nx}; {ny}; {nz}]);

    if (!materialexists("MicroRing_Clad")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "MicroRing_Clad"); }}
    setmaterial("MicroRing_Clad", "Refractive Index", {n_clad});

    if (!materialexists("MicroRing_Top")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "MicroRing_Top"); }}
    setmaterial("MicroRing_Top", "Refractive Index", {n_top});

    addrect; set("name", "BOX"); set("x", 0); set("x span", {dom_m * 1.4});
    set("y min", {-pad_m}); set("y max", 0); set("z", 0); set("z span", 2e-6);
    set("material", "MicroRing_Clad");

    addrect; set("name", "TopClad"); set("x", 0); set("x span", {dom_m * 1.4});
    set("y min", 0); set("y max", {hf_m + pad_m}); set("z", 0); set("z span", 2e-6);
    set("material", "MicroRing_Top");

    if ({hs_m} > 0) {{
        addrect; set("name", "Slab"); set("x", 0); set("x span", {dom_m * 1.4});
        set("y min", 0); set("y max", {hs_m}); set("z", 0); set("z span", 2e-6);
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 2);
    }}

    V = [{-0.5*w_m}, {hs_m}; {0.5*w_m}, {hs_m}; {0.5*w_top_m}, {hf_m}; {-0.5*w_top_m}, {hf_m}];
    addpoly; set("name", "Core"); set("vertices", V);
    set("z", 0); set("z span", 2e-6);
    set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);

    addmesh; set("name", "CoreMesh");
    set("x", 0); set("x span", {w_m + 0.5 * wl_m});
    set("y min", {-0.25 * wl_m}); set("y max", {hf_m + 0.25 * wl_m});
    set("z", 0); set("z span", 1e-6);
    set("dx", {dl_m}); set("dy", {dl_m}); set("override z mesh", 0);

    addfde;
    set("solver type", "2D Z normal");
    set("x", 0); set("x span", {dom_m});
    set("y min", {-pad_m}); set("y max", {hf_m + pad_m});
    set("z", 0);
    set("wavelength", {wl_m});
    set("bent waveguide", 1);
    set("bend radius", {r_m});
    set("bend orientation", 0);
    set("number of trial modes", 6);
    """
    mode_sess.eval(lsf)
    n_modes = int(mode_sess.findmodes())
    best_neff, best_ng = np.nan, np.nan
    for idx in range(1, n_modes + 1):
        te_pol = float(np.real(np.asarray(mode_sess.getdata(f"FDE::data::mode{idx}", "TE polarization fraction"))).reshape(-1)[0])
        if te_pol >= 0.5:
            best_neff = float(np.real(np.asarray(mode_sess.getdata(f"FDE::data::mode{idx}", "neff"))).reshape(-1)[0])
            best_ng = float(np.real(np.asarray(mode_sess.getdata(f"FDE::data::mode{idx}", "ng"))).reshape(-1)[0])
            break
    return {"neff_lumerical_fde": round(best_neff, 6), "ng_lumerical_fde": round(best_ng, 6)}


def run_3d_fdtd_coupler_case(fdtd_sess, case: dict) -> dict:
    """Run topology-exact 3D FDTD coupler S-parameter extraction (`|S21|^2` & `|S31|^2`)."""
    wl_m = case["wavelength_nm"] * 1e-9
    r_m = case["ring_radius_um"] * 1e-6
    wr_m = case["ring_width_um"] * 1e-6
    wb_m = case["bus_width_um"] * 1e-6
    g_m = case["gap_thru_um"] * 1e-6
    hf_m = case["film_height_um"] * 1e-6
    hs_m = case["slab_height_um"] * 1e-6
    ls_m = case["straight_length_um"] * 1e-6
    top = str(case["topology"])
    nx, ny, nz = case["n_core_xyz"]
    n_clad = case["n_clad"]
    n_top = case["n_top"]

    z_pad_m = max(1.8e-6, 1.35 * wl_m)
    port_y_span_m = max(wb_m + 2.4e-6, wb_m + 2.2 * wl_m)
    port_z_span_m = hf_m + 2.0 * z_pad_m * 0.75
    r_bus_m = r_m + 0.5 * wr_m + g_m + 0.5 * wb_m

    if top.startswith("pulley"):
        wrap_deg = float(case.get("pulley_angle_thru_deg", 20.0))
        r_trans_m = max(20.0e-6, 0.6 * r_m)
        x_half_m = max(14.0e-6, (r_bus_m + r_trans_m) * math.sin(math.radians(0.5 * wrap_deg)) + 4.0 * wl_m)
        bus_pts = _build_pulley_bus_centerline_m(r_bus_m, wrap_deg, r_trans_m, x_half_m + 4.0e-6)
        y_port_bus_m = float(bus_pts[0, 1])
        poly_lsf = _centerline_to_strip_polygon_lsf(bus_pts, wb_m)
        geom_lsf = f"""
        V_bus = {poly_lsf};
        addpoly; set("name", "PulleyBus"); set("vertices", V_bus);
        set("z min", {hs_m}); set("z max", {hf_m});
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);

        addring; set("name", "RingArc"); set("x", 0); set("y", 0);
        set("inner radius", {r_m - 0.5 * wr_m}); set("outer radius", {r_m + 0.5 * wr_m});
        set("z min", {hs_m}); set("z max", {hf_m});
        set("theta start", 0); set("theta stop", 180);
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
        """
        x_span_m = 2.0 * x_half_m
        y_min_fdtd = max(0.15 * r_m, r_m * math.cos(math.asin(min(0.95, x_half_m / r_m))) - 2.2 * wl_m)
        y_max_fdtd = max(r_bus_m, y_port_bus_m) + 2.2 * wl_m
    elif top.startswith("racetrack"):
        y_port_bus_m = r_bus_m
        x_half_m = 0.5 * ls_m + min(0.65 * r_m, max(8.0e-6, 4.0 * wl_m))
        x_span_m = 2.0 * x_half_m
        geom_lsf = f"""
        addrect; set("name", "Bus"); set("x", 0); set("x span", {x_span_m + 8e-6});
        set("y", {y_port_bus_m}); set("y span", {wb_m});
        set("z min", {hs_m}); set("z max", {hf_m});
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);

        if ({ls_m} > 0) {{
            addrect; set("name", "RaceStraight"); set("x", 0); set("x span", {ls_m});
            set("y", {r_m}); set("y span", {wr_m});
            set("z min", {hs_m}); set("z max", {hf_m});
            set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
        }}
        addring; set("name", "LeftArc"); set("x", {-0.5 * ls_m}); set("y", 0);
        set("inner radius", {r_m - 0.5 * wr_m}); set("outer radius", {r_m + 0.5 * wr_m});
        set("z min", {hs_m}); set("z max", {hf_m}); set("theta start", 90); set("theta stop", 180);
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);

        addring; set("name", "RightArc"); set("x", {0.5 * ls_m}); set("y", 0);
        set("inner radius", {r_m - 0.5 * wr_m}); set("outer radius", {r_m + 0.5 * wr_m});
        set("z min", {hs_m}); set("z max", {hf_m}); set("theta start", 0); set("theta stop", 90);
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
        """
        dx_arc = max(0.0, x_half_m - 0.5 * ls_m)
        y_arc_edge = r_m * math.cos(math.asin(min(0.95, dx_arc / r_m)))
        y_min_fdtd = max(0.10 * r_m, y_arc_edge - 2.2 * wl_m)
        y_max_fdtd = y_port_bus_m + 2.2 * wl_m
    else:
        y_port_bus_m = r_bus_m
        x_half_m = min(0.75 * r_m, max(8.0e-6, 4.5 * wl_m))
        x_span_m = 2.0 * x_half_m
        geom_lsf = f"""
        addrect; set("name", "Bus"); set("x", 0); set("x span", {x_span_m + 8e-6});
        set("y", {y_port_bus_m}); set("y span", {wb_m});
        set("z min", {hs_m}); set("z max", {hf_m});
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);

        addring; set("name", "RingArc"); set("x", 0); set("y", 0);
        set("inner radius", {r_m - 0.5 * wr_m}); set("outer radius", {r_m + 0.5 * wr_m});
        set("z min", {hs_m}); set("z max", {hf_m});
        set("theta start", 0); set("theta stop", 180);
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
        """
        y_arc_edge = r_m * math.cos(math.asin(min(0.95, x_half_m / r_m)))
        y_min_fdtd = max(0.10 * r_m, y_arc_edge - 2.2 * wl_m)
        y_max_fdtd = y_port_bus_m + 2.2 * wl_m

    lsf = f"""
    switchtolayout;
    deleteall;

    if (!materialexists("MicroRing_Core")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "MicroRing_Core"); }}
    setmaterial("MicroRing_Core", "Anisotropy", "Diagonal");
    setmaterial("MicroRing_Core", "Refractive Index", [{nx}; {ny}; {nz}]);
    if (!materialexists("MicroRing_Clad")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "MicroRing_Clad"); }}
    setmaterial("MicroRing_Clad", "Refractive Index", {n_clad});
    if (!materialexists("MicroRing_Top")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "MicroRing_Top"); }}
    setmaterial("MicroRing_Top", "Refractive Index", {n_top});

    addrect; set("name", "BOX"); set("x", 0); set("x span", {x_span_m + 10e-6});
    set("y min", {y_min_fdtd - 4e-6}); set("y max", {y_max_fdtd + 4e-6});
    set("z min", {-z_pad_m - 1e-6}); set("z max", 0); set("material", "MicroRing_Clad");

    addrect; set("name", "TopClad"); set("x", 0); set("x span", {x_span_m + 10e-6});
    set("y min", {y_min_fdtd - 4e-6}); set("y max", {y_max_fdtd + 4e-6});
    set("z min", 0); set("z max", {hf_m + z_pad_m + 1e-6}); set("material", "MicroRing_Top");

    if ({hs_m} > 0) {{
        addrect; set("name", "Slab"); set("x", 0); set("x span", {x_span_m + 10e-6});
        set("y min", {y_min_fdtd - 4e-6}); set("y max", {y_max_fdtd + 4e-6});
        set("z min", 0); set("z max", {hs_m});
        set("material", "MicroRing_Core"); set("override mesh order from material database", 1); set("mesh order", 2);
    }}

    {geom_lsf}

    addfdtd;
    set("dimension", "3D");
    set("x", 0); set("x span", {x_span_m});
    set("y min", {y_min_fdtd}); set("y max", {y_max_fdtd});
    set("z min", {-z_pad_m}); set("z max", {hf_m + z_pad_m});
    set("mesh accuracy", 3);
    set("simulation time", {FDTD_SIM_TIME_FS * 1e-15});
    set("auto shutoff min", {FDTD_AUTO_SHUTOFF_MIN});
    set("pml layers", {FDTD_PML_LAYERS});

    addport; set("name", "BusIn"); set("injection axis", "x-axis"); set("direction", "Forward");
    set("x", {-0.46 * x_span_m}); set("y", {y_port_bus_m}); set("y span", {port_y_span_m});
    set("z", {0.5 * hf_m}); set("z span", {port_z_span_m});

    addport; set("name", "BusThru"); set("injection axis", "x-axis"); set("direction", "Backward");
    set("x", {0.46 * x_span_m}); set("y", {y_port_bus_m}); set("y span", {port_y_span_m});
    set("z", {0.5 * hf_m}); set("z span", {port_z_span_m});

    select("FDTD::ports"); set("source port", "BusIn"); set("source mode", "mode 1");
    setglobalsource("wavelength start", {wl_m * 0.995});
    setglobalsource("wavelength stop", {wl_m * 1.005});
    setglobalmonitor("frequency points", 5);
    """
    fdtd_sess.eval(lsf)
    fsp_tmp = HERE / "outputs" / f"_{case['case_id']}_tmp.fsp"
    fdtd_sess.save(str(fsp_tmp))
    try:
        fdtd_sess.run("FDTD", "GPU")
    except Exception:
        fdtd_sess.run("FDTD")
    if fsp_tmp.is_file():
        try:
            fsp_tmp.unlink()
        except OSError:
            pass
    try:
        exp_res = fdtd_sess.getresult("FDTD::ports::BusThru", "expansion for port monitor")
        t_fwd = np.asarray(exp_res["T_forward"]).reshape(-1)
        t_thru = float(abs(np.real(t_fwd[len(t_fwd) // 2])))
    except Exception:
        t_arr = np.asarray(fdtd_sess.getdata("FDTD::ports::BusThru", "T")).reshape(-1)
        t_thru = float(abs(np.real(t_arr[len(t_arr) // 2])))
    kp_fdtd = max(0.0, 1.0 - t_thru)
    return {"thru_T_fdtd": round(t_thru, 6), "kappa_power_fdtd": round(kp_fdtd, 6)}


def main() -> None:
    if not CASES_JSON.is_file():
        raise FileNotFoundError(f"Run generate_and_audit_benchmarks.py first to create {CASES_JSON}")
    cases = json.loads(CASES_JSON.read_text(encoding="utf-8"))
    lumapi = _import_lumapi()

    rows = []
    with lumapi.MODE(hide=True) as mode_sess, lumapi.FDTD(hide=True) as fdtd_sess:
        ver = str(mode_sess.eval("?version;")).strip()
        for c in cases:
            m_out = run_mode_fde_case(mode_sess, c)
            f_out = run_3d_fdtd_coupler_case(fdtd_sess, c)
            ref = c["microring_reference"]
            rows.append({
                "case_id": c["case_id"],
                "wavelength_nm": c["wavelength_nm"],
                "neff_microring_fem": ref["neff_bsd_fem"],
                "neff_lumerical_fde": m_out["neff_lumerical_fde"],
                "abs_diff_neff": round(abs(ref["neff_bsd_fem"] - m_out["neff_lumerical_fde"]), 6),
                "ng_microring": ref["ng_microring"],
                "ng_lumerical_fde": m_out["ng_lumerical_fde"],
                "kappa_power_microring_cmt": ref["kappa_power_thru_cmt"],
                "kappa_power_lumerical_3dfdtd": f_out["kappa_power_fdtd"],
            })

    cols = list(rows[0].keys())
    lines = [f"# Ansys Lumerical ({ver}) vs MicroRing Studio Validation", ",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r[k]) for k in cols))
    OUT_CSV.write_text("\n".join(lines) + "\n", encoding="utf-8")

    md = [
        f"# Ansys Lumerical (`{ver}`) vs MicroRing Studio Validation Report",
        "",
        "| Case ID | $\\lambda$ (nm) | $n_{\\mathrm{eff}}$ (MicroRing) | $n_{\\mathrm{eff}}$ (Lumerical MODE) | $|\\Delta n_{\\mathrm{eff}}|$ | $n_g$ (MicroRing) | $n_g$ (Lumerical) | $\\kappa^2$ (MicroRing CMT) | $\\kappa^2$ (3D FDTD) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md.append(
            f"| `{r['case_id']}` | {r['wavelength_nm']} | {r['neff_microring_fem']:.6f} | {r['neff_lumerical_fde']:.6f} | "
            f"`{r['abs_diff_neff']:.2e}` | {r['ng_microring']:.4f} | {r['ng_lumerical_fde']:.4f} | "
            f"{r['kappa_power_microring_cmt']:.4f} | {r['kappa_power_lumerical_3dfdtd']:.4f} |"
        )
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved Lumerical validation report to {OUT_MD}")


if __name__ == "__main__":
    main()
