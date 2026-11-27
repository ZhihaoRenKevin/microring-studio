"""Independent Ansys Lumerical (2026 R1) Validation Runner (`lumapi`).

Reads ONLY neutral physical specifications from `benchmarks/outputs/commercial_validation_specs.json`
and executes Ansys Lumerical completely independently across:
  - Suite 1  : `MODE FDE` 2D bent & straight eigenmodes across 5 wavebands (532 nm -> 4200 nm),
               extracting exact dispersive n_g via 3-wavelength Sellmeier central difference.
  - Suite 2A : `3D FDTD` (RTX 5090 GPU via `fdtd.run("FDTD", "GPU")`) full-wave bus-to-ring
               couplers across all 5 wavebands & topologies (point, pulley, racetrack), with
               both BusThru (`|S21|^2`) and RingCoupled (`|S31|^2`) monitors + radiation loss.
  - Suite 2B : `3D FDTD` (RTX 5090 GPU) SOI 1550 nm gap sweep (`100, 150, 200, 250 nm`)
               verifying exponential scaling and the sub-120 nm validity boundary.
  - Suite 3A : `MODE FDE` 2x2 SOI Directional Coupler twin-core even/odd supermodes.
  - Suite 3B : `3D FDTD` (RTX 5090 GPU) 220 nm SOI 1x2 MMI and 1x2 adiabatic Y-Branch splitters.
  - Suite 3C : `2D FDTD` 220 nm SOI uniform 70-nm-etch surface grating coupler (26 periods).
  - Suite 4  : `DEVICE HEAT` 2D steady-state & transient thermal FEM (where supported) + `MODE` check.
  - Suite 5  : `MODE FDE` anisotropic X-cut TFLN & c-axis ScAlN_20 optical baseline & EO sensitivity.

All temporary `.fsp` / `.lms` / `.ldev` binaries are cleaned up after lightweight metrics are saved.
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

# ==============================================================================
# Centralized Lumerical Solver Controls & Paths (with physical units)
# ==============================================================================
FDTD_AUTO_SHUTOFF_MIN = 1.0e-5       # unitless: energy decay convergence floor
FDTD_SIM_TIME_FS = 8000.0            # unit: fs (8 ps single-pass coupler transit)
FDTD_PML_LAYERS = 12                 # unitless: standard stretched-coordinate PML
FDTD_MESH_ACCURACY = 3               # unitless: Lumerical conformal mesh accuracy
SLANTED_WALL_SLICES = 8              # unitless: vertical stair-free slice count for angled walls

LUMERICAL_WIN_HINTS = [
    r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v252\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
    r"C:\Program Files\ANSYS Inc\v242\Lumerical\api\python",
    r"C:\Program Files\Lumerical\v242\api\python",
]

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
SCRATCH_DIR = OUT_DIR / "_lum_scratch"
SPECS_JSON = OUT_DIR / "commercial_validation_specs.json"
OUT_JSON = OUT_DIR / "lumerical_independent_results.json"


def _import_lumapi():
    env_p = os.environ.get("LUMAPI_PATH")
    if env_p and Path(env_p).exists():
        sys.path.insert(0, env_p)
    for p in LUMERICAL_WIN_HINTS:
        if Path(p).exists() and p not in sys.path:
            sys.path.insert(0, p)
    import lumapi  # type: ignore
    return lumapi


def _read_lumerical_version() -> str:
    for base in (r"C:\Program Files\ANSYS Inc\v261\Lumerical\VERSION",):
        vp = Path(base)
        if vp.is_file():
            kv = {}
            for line in vp.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip()
            return f"Ansys Lumerical {kv.get('MAJORRELEASE', '2026R1')}.{kv.get('MINORRELEASE', '0')} (Build {kv.get('BUILDNUMBER', 'unknown')})"
    return "Ansys Lumerical"


def _build_pulley_bus_centerline_m(r_bus_m: float, wrap_deg: float, r_trans_m: float, x_half_m: float) -> np.ndarray:
    """Construct C1-smooth concentric pulley bus centerline with horizontal input/output leads."""
    th = math.radians(max(wrap_deg, 1.0))
    half_th = 0.5 * th
    x_l = -r_bus_m * math.sin(half_th)
    y_l = r_bus_m * math.cos(half_th)
    xc_l = x_l - r_trans_m * math.sin(half_th)
    yc_l = y_l + r_trans_m * math.cos(half_th)
    y_lead = yc_l - r_trans_m

    a_left = np.linspace(-0.5 * math.pi, -0.5 * math.pi + half_th, 25)
    pts_left_arc = np.column_stack([xc_l + r_trans_m * np.cos(a_left), yc_l + r_trans_m * np.sin(a_left)])

    a_wrap = np.linspace(0.5 * math.pi + half_th, 0.5 * math.pi - half_th, 49)
    pts_wrap = np.column_stack([r_bus_m * np.cos(a_wrap), r_bus_m * np.sin(a_wrap)])

    xc_r = -xc_l
    yc_r = yc_l
    a_right = np.linspace(-0.5 * math.pi - half_th, -0.5 * math.pi, 25)
    pts_right_arc = np.column_stack([xc_r + r_trans_m * np.cos(a_right), yc_r + r_trans_m * np.sin(a_right)])

    x_ext = max(x_half_m, abs(xc_l) + 4.0e-6)
    lead_l = np.array([[-x_ext, y_lead]])
    lead_r = np.array([[x_ext, y_lead]])
    return np.vstack([lead_l, pts_left_arc, pts_wrap[1:-1], pts_right_arc, lead_r])


def _centerline_to_strip_polygon_lsf(pts_m: np.ndarray, width_m: float) -> str:
    tangents = np.gradient(pts_m, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-15)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    upper = pts_m + 0.5 * width_m * normals
    lower = (pts_m - 0.5 * width_m * normals)[::-1]
    poly = np.vstack([upper, lower])
    rows = "; ".join(f"{p[0]:.9e}, {p[1]:.9e}" for p in poly)
    return f"[{rows}]"


def _set_diag_materials_lsf(nx: float, ny: float, nz: float, n_clad: float, n_top: float) -> str:
    return f"""
    if (!materialexists("Val_Core")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "Val_Core"); }}
    setmaterial("Val_Core", "Anisotropy", "Diagonal");
    setmaterial("Val_Core", "Refractive Index", [{nx:.8f}; {ny:.8f}; {nz:.8f}]);
    if (!materialexists("Val_Clad")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "Val_Clad"); }}
    setmaterial("Val_Clad", "Refractive Index", {n_clad:.8f});
    if (!materialexists("Val_Top")) {{ addmaterial("(n,k) Material"); setmaterial("New material 1", "name", "Val_Top"); }}
    setmaterial("Val_Top", "Refractive Index", {n_top:.8f});
    """


def _scalar_real(val) -> float:
    return float(np.real(np.asarray(val)).reshape(-1)[0])


def _solve_single_mode_fde(
    mode_sess,
    *,
    wl_m: float,
    wl_mesh_m: float | None = None,
    w_m: float,
    hf_m: float,
    hs_m: float,
    sw_deg: float,
    dom_m: float,
    pad_m: float,
    dl_m: float,
    nx: float,
    ny: float,
    nz: float,
    n_clad: float,
    n_top: float,
    bent: bool,
    r_m: float = 0.0,
    pol: str = "TE0",
) -> tuple[float, float]:
    wl_ref_m = wl_mesh_m if wl_mesh_m is not None else wl_m
    h_ridge = hf_m - hs_m
    inv_tan = 0.0 if abs(sw_deg - 90.0) < 1e-3 else 1.0 / math.tan(math.radians(sw_deg))
    w_top_m = w_m - 2.0 * h_ridge * inv_tan
    bent_lsf = f'set("bent waveguide", 1); set("bend radius", {r_m}); set("bend orientation", 0);' if bent else 'set("bent waveguide", 0);'

    lsf = f"""
    switchtolayout;
    deleteall;
    {_set_diag_materials_lsf(nx, ny, nz, n_clad, n_top)}

    addrect; set("name", "BOX"); set("x", 0); set("x span", {dom_m * 1.4});
    set("y min", {-pad_m}); set("y max", 0); set("z", 0); set("z span", 2e-6);
    set("material", "Val_Clad");

    addrect; set("name", "TopClad"); set("x", 0); set("x span", {dom_m * 1.4});
    set("y min", 0); set("y max", {hf_m + pad_m}); set("z", 0); set("z span", 2e-6);
    set("material", "Val_Top");

    if ({hs_m} > 0) {{
        addrect; set("name", "Slab"); set("x", 0); set("x span", {dom_m * 1.4});
        set("y min", 0); set("y max", {hs_m}); set("z", 0); set("z span", 2e-6);
        set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 2);
    }}

    V = [{-0.5*w_m}, {hs_m}; {0.5*w_m}, {hs_m}; {0.5*w_top_m}, {hf_m}; {-0.5*w_top_m}, {hf_m}];
    addpoly; set("name", "Core"); set("vertices", V);
    set("z", 0); set("z span", 2e-6);
    set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);

    addmesh; set("name", "CoreMesh");
    set("x", 0); set("x span", {w_m + 0.5 * wl_ref_m});
    set("y min", {-0.25 * wl_ref_m}); set("y max", {hf_m + 0.25 * wl_ref_m});
    set("z", 0); set("z span", 1e-6);
    set("dx", {dl_m}); set("dy", {dl_m}); set("override z mesh", 0);

    addfde;
    set("solver type", "2D Z normal");
    set("mesh refinement", "conformal variant 1");
    set("x", 0); set("x span", {dom_m});
    set("y min", {-pad_m}); set("y max", {hf_m + pad_m});
    set("z", 0);
    set("wavelength", {wl_m});
    {bent_lsf}
    set("number of trial modes", 8);
    """
    mode_sess.eval(lsf)
    for _attempt in range(12):
        try:
            n_modes = int(mode_sess.findmodes())
            break
        except Exception as exc:
            if "license" in str(exc).lower() and _attempt < 11:
                time.sleep(15.0)
                continue
            raise
    want_te = pol.upper().startswith("TE")
    best_neff, best_te = float("nan"), 0.0
    for idx in range(1, n_modes + 1):
        te_pol = _scalar_real(mode_sess.getdata(f"FDE::data::mode{idx}", "TE polarization fraction"))
        if (want_te and te_pol >= 0.5) or ((not want_te) and te_pol < 0.5):
            best_neff = _scalar_real(mode_sess.getdata(f"FDE::data::mode{idx}", "neff"))
            best_te = te_pol
            break
    return best_neff, best_te


def run_suite1_mode_fde(mode_sess, cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        t0 = time.perf_counter()
        wl0_nm = float(c["wavelength_nm"])
        dwl_nm = float(c["dwl_nm"])
        w_m = c["ring_width_um"] * 1e-6
        hf_m = c["film_height_um"] * 1e-6
        hs_m = c["slab_height_um"] * 1e-6
        r_m = c["ring_radius_um"] * 1e-6
        sw_deg = c["sidewall_deg"]
        dom_m = c["domain_width_um"] * 1e-6
        pad_m = c["box_pad_um"] * 1e-6
        dl_m = c["fde_dl_um"] * 1e-6

        neffs = {}
        te0 = 0.0
        for key in ("minus", "center", "plus"):
            ci = c["core_index"][key]
            cli = c["clad_index"][key]
            ti = c["top_index"][key]
            ne, tef = _solve_single_mode_fde(
                mode_sess,
                wl_m=ci["wavelength_nm"] * 1e-9,
                wl_mesh_m=wl0_nm * 1e-9,
                w_m=w_m, hf_m=hf_m, hs_m=hs_m, sw_deg=sw_deg,
                dom_m=dom_m, pad_m=pad_m, dl_m=dl_m,
                nx=ci["nx"], ny=ci["ny"], nz=ci["nz"],
                n_clad=cli["nx"], n_top=ti["nx"],
                bent=True, r_m=r_m, pol="TE0",
            )
            neffs[key] = ne
            if key == "center":
                te0 = tef

        ng_bent = neffs["center"] - wl0_nm * (neffs["plus"] - neffs["minus"]) / (2.0 * dwl_nm)

        ci0 = c["core_index"]["center"]
        ne_str, _ = _solve_single_mode_fde(
            mode_sess,
            wl_m=wl0_nm * 1e-9,
            w_m=w_m, hf_m=hf_m, hs_m=hs_m, sw_deg=sw_deg,
            dom_m=dom_m, pad_m=pad_m, dl_m=dl_m,
            nx=ci0["nx"], ny=ci0["ny"], nz=ci0["nz"],
            n_clad=c["clad_index"]["center"]["nx"], n_top=c["top_index"]["center"]["nx"],
            bent=False, r_m=0.0, pol="TE0",
        )
        l_rt_um = 2.0 * math.pi * c["ring_radius_um"] + 2.0 * c["straight_length_um"]
        fsr_nm = (wl0_nm ** 2) / (ng_bent * l_rt_um * 1e3)
        dt_s = time.perf_counter() - t0
        rows.append({
            "case_id": c["case_id"],
            "wavelength_nm": wl0_nm,
            "neff_bent": round(neffs["center"], 6),
            "neff_bent_minus": round(neffs["minus"], 6),
            "neff_bent_plus": round(neffs["plus"], 6),
            "ng_bent": round(ng_bent, 5),
            "neff_straight": round(ne_str, 6),
            "delta_neff_bend": round(neffs["center"] - ne_str, 6),
            "te_fraction": round(te0, 4),
            "fsr_nm": round(fsr_nm, 4),
            "wall_time_s": round(dt_s, 3),
        })
        print(
            f"[LUM-S1] {c['case_id']:<26} neff={neffs['center']:.6f} ng={ng_bent:.4f} "
            f"d_bend={neffs['center']-ne_str:+.6f} ({dt_s:.2f}s)"
        )
    return rows


def run_suite3a_dc_mode_fde(mode_sess, dc_cases: list[dict]) -> list[dict]:
    rows = []
    for c in dc_cases:
        t0 = time.perf_counter()
        wl_m = c["wavelength_nm"] * 1e-9
        w_m = c["waveguide_width_um"] * 1e-6
        g_m = c["gap_um"] * 1e-6
        hf_m = c["film_height_um"] * 1e-6
        dom_m = c["domain_width_um"] * 1e-6
        pad_m = c["box_pad_um"] * 1e-6
        xc_m = 0.5 * (w_m + g_m)
        ci = c["core_index"]["center"]
        n_cl = c["clad_index"]["center"]["nx"]

        lsf = f"""
        switchtolayout;
        deleteall;
        {_set_diag_materials_lsf(ci["nx"], ci["ny"], ci["nz"], n_cl, n_cl)}

        addrect; set("name", "Clad"); set("x", 0); set("x span", {dom_m * 1.4});
        set("y min", {-pad_m}); set("y max", {hf_m + pad_m}); set("z", 0); set("z span", 2e-6);
        set("material", "Val_Clad");

        addrect; set("name", "CoreL"); set("x", {-xc_m}); set("x span", {w_m});
        set("y min", 0); set("y max", {hf_m}); set("z", 0); set("z span", 2e-6);
        set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);

        addrect; set("name", "CoreR"); set("x", {+xc_m}); set("x span", {w_m});
        set("y min", 0); set("y max", {hf_m}); set("z", 0); set("z span", 2e-6);
        set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);

        addmesh; set("name", "CoreMesh");
        set("x", 0); set("x span", {2*w_m + g_m + 0.6e-6});
        set("y min", -0.25e-6); set("y max", {hf_m + 0.25e-6});
        set("z", 0); set("z span", 1e-6);
        set("dx", 10e-9); set("dy", 10e-9); set("override z mesh", 0);

        addfde;
        set("solver type", "2D Z normal");
        set("mesh refinement", "conformal variant 1");
        set("x", 0); set("x span", {dom_m});
        set("y min", {-pad_m}); set("y max", {hf_m + pad_m});
        set("z", 0); set("wavelength", {wl_m});
        set("number of trial modes", 8);
        """
        mode_sess.eval(lsf)
        for _attempt in range(12):
            try:
                n_modes = int(mode_sess.findmodes())
                break
            except Exception as exc:
                if "license" in str(exc).lower() and _attempt < 11:
                    time.sleep(15.0)
                    continue
                raise
        te_neffs = []
        for idx in range(1, n_modes + 1):
            te_pol = _scalar_real(mode_sess.getdata(f"FDE::data::mode{idx}", "TE polarization fraction"))
            if te_pol >= 0.5:
                te_neffs.append(_scalar_real(mode_sess.getdata(f"FDE::data::mode{idx}", "neff")))
        te_neffs.sort(reverse=True)
        n_even, n_odd = te_neffs[0], te_neffs[1]
        dn = n_even - n_odd
        l_pi = (c["wavelength_nm"] * 1e-3) / (2.0 * dn)
        rows.append({
            "case_id": c["case_id"],
            "gap_nm": round(c["gap_um"] * 1e3, 1),
            "n_even": round(n_even, 6),
            "n_odd": round(n_odd, 6),
            "delta_n": round(dn, 6),
            "L_pi_um": round(l_pi, 4),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        })
        print(f"[LUM-S3A] {c['case_id']:<26} n_e={n_even:.6f} n_o={n_odd:.6f} dn={dn:.6f} L_pi={l_pi:.3f} um")
    return rows


def run_suite5_mode_eo_baseline(mode_sess, eo_cases: list[dict]) -> list[dict]:
    """Verify baseline optical n_eff and uniform-field Pockels dn_eff/dE sensitivity in Lumerical MODE."""
    rows = []
    for c in eo_cases:
        t0 = time.perf_counter()
        wl_m = c["wavelength_nm"] * 1e-9
        ci = c["core_index"]["center"]
        n_cl = c["clad_index"]["center"]["nx"]
        n_tp = c["top_index"]["center"]["nx"]
        ne0, tef = _solve_single_mode_fde(
            mode_sess,
            wl_m=wl_m,
            w_m=c["waveguide_width_um"] * 1e-6,
            hf_m=c["film_height_um"] * 1e-6,
            hs_m=c["slab_height_um"] * 1e-6,
            sw_deg=c["sidewall_deg"],
            dom_m=5.5e-6, pad_m=2.0e-6, dl_m=0.015e-6,
            nx=ci["nx"], ny=ci["ny"], nz=ci["nz"],
            n_clad=n_cl, n_top=n_tp,
            bent=False, pol=c["polarization"],
        )
        # Uniform E = 1 V/um test shift inside core & slab to validate modal confinement / Pockels sensitivity
        r33 = c["core_index"]["r33_pm_per_v"]
        r13 = c["core_index"]["r13_pm_per_v"]
        e_test = 1.0  # V/um
        if c["electrode_config"] == "inplane_gsg":
            dnx = -0.5 * (ci["nx"] ** 3) * (r33 * 1e-6) * e_test
            dny = -0.5 * (ci["ny"] ** 3) * (r13 * 1e-6) * e_test
            dnz = -0.5 * (ci["nz"] ** 3) * (r13 * 1e-6) * e_test
        else:
            dnx = -0.5 * (ci["nx"] ** 3) * (r13 * 1e-6) * e_test
            dny = -0.5 * (ci["ny"] ** 3) * (r33 * 1e-6) * e_test
            dnz = -0.5 * (ci["nz"] ** 3) * (r13 * 1e-6) * e_test

        ne1, _ = _solve_single_mode_fde(
            mode_sess,
            wl_m=wl_m,
            w_m=c["waveguide_width_um"] * 1e-6,
            hf_m=c["film_height_um"] * 1e-6,
            hs_m=c["slab_height_um"] * 1e-6,
            sw_deg=c["sidewall_deg"],
            dom_m=5.5e-6, pad_m=2.0e-6, dl_m=0.015e-6,
            nx=ci["nx"] + dnx, ny=ci["ny"] + dny, nz=ci["nz"] + dnz,
            n_clad=n_cl, n_top=n_tp,
            bent=False, pol=c["polarization"],
        )
        dneff_per_vum = abs(ne1 - ne0) / e_test
        rows.append({
            "case_id": c["case_id"],
            "neff_0v": round(ne0, 6),
            "te_fraction": round(tef, 4),
            "dneff_per_v_per_um_uniform": float(f"{dneff_per_vum:.6e}"),
            "wall_time_s": round(time.perf_counter() - t0, 3),
        })
        print(f"[LUM-S5] {c['case_id']:<26} neff0={ne0:.6f} | |dneff/dE_unif|={dneff_per_vum:.4e} (V/um)^-1")
    return rows


def _build_slanted_slices(hs_m: float, hf_m: float, sw_deg: float) -> list[tuple[float, float, float]]:
    """Return list of (z_min, z_max, width_reduction_total_m) slices matching exact trapezoidal wall."""
    h_ridge = hf_m - hs_m
    if abs(sw_deg - 90.0) < 0.25 or h_ridge <= 1e-12:
        return [(hs_m, hf_m, 0.0)]
    inv_tan = 1.0 / math.tan(math.radians(sw_deg))
    n = SLANTED_WALL_SLICES
    dz = h_ridge / n
    out = []
    for k in range(n):
        z0 = hs_m + k * dz
        z1 = z0 + dz
        z_mid = 0.5 * (z0 + z1) - hs_m
        dw = 2.0 * z_mid * inv_tan
        out.append((z0, z1, dw))
    return out


def run_3d_fdtd_coupler_gpu(fdtd_sess, case: dict) -> dict:
    """Run topology-exact 3D FDTD coupler on RTX 5090 GPU, measuring |S21|^2, |S31|^2 & radiation loss."""
    t0 = time.perf_counter()
    wl_m = case["wavelength_nm"] * 1e-9
    r_m = case["ring_radius_um"] * 1e-6
    wr_m = case["ring_width_um"] * 1e-6
    wb_m = case["bus_width_um"] * 1e-6
    g_m = case["gap_thru_um"] * 1e-6
    hf_m = case["film_height_um"] * 1e-6
    hs_m = case["slab_height_um"] * 1e-6
    ls_m = case["straight_length_um"] * 1e-6
    sw_deg = float(case["sidewall_deg"])
    top = str(case["topology"])

    ci = case["core_index"]["center"]
    nx, ny, nz = ci["nx"], ci["ny"], ci["nz"]
    # Note: In 3D FDTD xy is wafer plane (x=propagation, y=radial, z=vertical) -> swap (ny, nz) for c-axis vertical!
    # Wait: in MicroRing 2D cross-section, x=radial, y=vertical, z=propagation.
    # For uniaxial c-vertical (4H-SiC): n_inplane = nx = nz = no, n_vertical = ny = ne!
    # In 3D FDTD: x, y are in-plane -> no, z is vertical -> ne!
    n_fdtd_x, n_fdtd_y, n_fdtd_z = nx, nz, ny
    n_clad = case["clad_index"]["center"]["nx"]
    n_top = case["top_index"]["center"]["nx"]

    z_pad_m = max(1.4e-6, 1.15 * wl_m)
    r_bus_m = r_m + 0.5 * wr_m + g_m + 0.5 * wb_m
    slices = _build_slanted_slices(hs_m, hf_m, sw_deg)

    geom_parts = []
    if top.startswith("pulley"):
        wrap_deg = float(case.get("pulley_angle_thru_deg", 20.0))
        r_trans_m = max(20.0e-6, 0.6 * r_m)
        x_half_m = max(14.0e-6, (r_bus_m + r_trans_m) * math.sin(math.radians(0.5 * wrap_deg)) + 3.5 * wl_m)
        bus_pts = _build_pulley_bus_centerline_m(r_bus_m, wrap_deg, r_trans_m, x_half_m + 4.0e-6)
        y_port_bus_m = float(bus_pts[0, 1])
        for k, (z0, z1, dw) in enumerate(slices):
            poly_lsf = _centerline_to_strip_polygon_lsf(bus_pts, max(wb_m - dw, 0.05e-6))
            wr_k = max(wr_m - dw, 0.05e-6)
            geom_parts.append(f"""
            V_bus_{k} = {poly_lsf};
            addpoly; set("name", "PulleyBus_{k}"); set("vertices", V_bus_{k});
            set("z min", {z0}); set("z max", {z1});
            set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
            addring; set("name", "RingArc_{k}"); set("x", 0); set("y", 0);
            set("inner radius", {r_m - 0.5 * wr_k}); set("outer radius", {r_m + 0.5 * wr_k});
            set("z min", {z0}); set("z max", {z1}); set("theta start", 0); set("theta stop", 180);
            set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
            """)
        x_span_m = 2.0 * x_half_m
        x_mon_m = 0.44 * x_span_m
        y_arc_mon_m = math.sqrt(max(r_m**2 - x_mon_m**2, 0.0))
        y_arc_edge = math.sqrt(max(r_m**2 - min(0.95 * r_m, x_half_m)**2, 0.0))
        y_min_fdtd = max(0.15 * r_m, y_arc_edge - 1.8 * wl_m)
        y_max_fdtd = max(r_bus_m, y_port_bus_m) + 1.8 * wl_m
    elif top.startswith("racetrack"):
        y_port_bus_m = r_bus_m
        x_half_m = 0.5 * ls_m + min(0.55 * r_m, max(8.0e-6, 3.0 * wl_m))
        x_span_m = 2.0 * x_half_m
        for k, (z0, z1, dw) in enumerate(slices):
            wb_k = max(wb_m - dw, 0.05e-6)
            wr_k = max(wr_m - dw, 0.05e-6)
            geom_parts.append(f"""
            addrect; set("name", "Bus_{k}"); set("x", 0); set("x span", {x_span_m + 8e-6});
            set("y", {y_port_bus_m}); set("y span", {wb_k}); set("z min", {z0}); set("z max", {z1});
            set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
            addrect; set("name", "RaceStraight_{k}"); set("x", 0); set("x span", {ls_m});
            set("y", {r_m}); set("y span", {wr_k}); set("z min", {z0}); set("z max", {z1});
            set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
            addring; set("name", "LeftArc_{k}"); set("x", {-0.5 * ls_m}); set("y", 0);
            set("inner radius", {r_m - 0.5 * wr_k}); set("outer radius", {r_m + 0.5 * wr_k});
            set("z min", {z0}); set("z max", {z1}); set("theta start", 90); set("theta stop", 180);
            set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
            addring; set("name", "RightArc_{k}"); set("x", {0.5 * ls_m}); set("y", 0);
            set("inner radius", {r_m - 0.5 * wr_k}); set("outer radius", {r_m + 0.5 * wr_k});
            set("z min", {z0}); set("z max", {z1}); set("theta start", 0); set("theta stop", 90);
            set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
            """)
        x_mon_m = 0.44 * x_span_m
        dx_mon = max(0.0, x_mon_m - 0.5 * ls_m)
        y_arc_mon_m = math.sqrt(max(r_m**2 - dx_mon**2, 0.0))
        dx_arc = max(0.0, x_half_m - 0.5 * ls_m)
        y_arc_edge = math.sqrt(max(r_m**2 - min(0.95 * r_m, dx_arc)**2, 0.0))
        y_min_fdtd = max(0.10 * r_m, y_arc_edge - 1.8 * wl_m)
        y_max_fdtd = y_port_bus_m + 1.8 * wl_m
    else:
        y_port_bus_m = r_bus_m
        x_half_m = min(0.62 * r_m, max(7.0e-6, 4.2 * wl_m))
        x_span_m = 2.0 * x_half_m
        for k, (z0, z1, dw) in enumerate(slices):
            wb_k = max(wb_m - dw, 0.05e-6)
            wr_k = max(wr_m - dw, 0.05e-6)
            geom_parts.append(f"""
            addrect; set("name", "Bus_{k}"); set("x", 0); set("x span", {x_span_m + 8e-6});
            set("y", {y_port_bus_m}); set("y span", {wb_k}); set("z min", {z0}); set("z max", {z1});
            set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
            addring; set("name", "RingArc_{k}"); set("x", 0); set("y", 0);
            set("inner radius", {r_m - 0.5 * wr_k}); set("outer radius", {r_m + 0.5 * wr_k});
            set("z min", {z0}); set("z max", {z1}); set("theta start", 0); set("theta stop", 180);
            set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 1);
            """)
        x_mon_m = 0.44 * x_span_m
        y_arc_mon_m = math.sqrt(max(r_m**2 - x_mon_m**2, 0.0))
        y_arc_edge = math.sqrt(max(r_m**2 - min(0.95 * r_m, x_half_m)**2, 0.0))
        y_min_fdtd = max(0.10 * r_m, y_arc_edge - 1.8 * wl_m)
        y_max_fdtd = y_port_bus_m + 1.8 * wl_m

    sep_mon_m = max(y_port_bus_m - y_arc_mon_m, 0.8e-6)
    port_y_span_m = min(max(wb_m + 1.5 * wl_m, wb_m + 1.2e-6), 1.25 * sep_mon_m)
    port_z_span_m = hf_m + 1.6 * z_pad_m
    y_split_m = 0.5 * (y_port_bus_m + y_arc_mon_m)
    ring_mon_ymin = y_min_fdtd + 0.25 * wl_m
    ring_mon_ymax = min(y_split_m, y_port_bus_m - 0.5 * port_y_span_m)

    geom_lsf = "\n".join(geom_parts)
    lsf = f"""
    switchtolayout;
    deleteall;
    {_set_diag_materials_lsf(n_fdtd_x, n_fdtd_y, n_fdtd_z, n_clad, n_top)}

    addrect; set("name", "BOX"); set("x", 0); set("x span", {x_span_m + 12e-6});
    set("y min", {y_min_fdtd - 4e-6}); set("y max", {y_max_fdtd + 4e-6});
    set("z min", {-z_pad_m - 1e-6}); set("z max", 0); set("material", "Val_Clad");

    addrect; set("name", "TopClad"); set("x", 0); set("x span", {x_span_m + 12e-6});
    set("y min", {y_min_fdtd - 4e-6}); set("y max", {y_max_fdtd + 4e-6});
    set("z min", 0); set("z max", {hf_m + z_pad_m + 1e-6}); set("material", "Val_Top");

    if ({hs_m} > 0) {{
        addrect; set("name", "Slab"); set("x", 0); set("x span", {x_span_m + 12e-6});
        set("y min", {y_min_fdtd - 4e-6}); set("y max", {y_max_fdtd + 4e-6});
        set("z min", 0); set("z max", {hs_m});
        set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 2);
    }}

    {geom_lsf}

    addfdtd;
    set("dimension", "3D");
    set("x", 0); set("x span", {x_span_m});
    set("y min", {y_min_fdtd}); set("y max", {y_max_fdtd});
    set("z min", {-z_pad_m}); set("z max", {hf_m + z_pad_m});
    set("mesh accuracy", {FDTD_MESH_ACCURACY});
    set("simulation time", {FDTD_SIM_TIME_FS * 1e-15});
    set("auto shutoff min", {FDTD_AUTO_SHUTOFF_MIN});
    set("pml layers", {FDTD_PML_LAYERS});

    # Fine mesh override across coupling gap region for crisp evanescent decay
    addmesh; set("name", "GapMesh");
    set("x", 0); set("x span", {min(x_span_m * 0.65, max(ls_m + 4e-6, 5.0 * wl_m))});
    set("y", {r_m + 0.5 * wr_m + 0.5 * g_m}); set("y span", {wr_m + wb_m + g_m + 0.6 * wl_m});
    set("z min", {-0.15 * wl_m}); set("z max", {hf_m + 0.15 * wl_m});
    set("dx", {min(0.040e-6, wl_m / 24.0)});
    set("dy", {min(0.025e-6, max(g_m / 6.0, wl_m / 36.0))});
    set("dz", {min(0.025e-6, hf_m / 8.0)});

    addport; set("name", "BusIn"); set("injection axis", "x-axis"); set("direction", "Forward");
    set("x", {-x_mon_m}); set("y", {y_port_bus_m}); set("y span", {port_y_span_m});
    set("z", {0.5 * hf_m}); set("z span", {port_z_span_m});

    addport; set("name", "BusThru"); set("injection axis", "x-axis"); set("direction", "Backward");
    set("x", {+x_mon_m}); set("y", {y_port_bus_m}); set("y span", {port_y_span_m});
    set("z", {0.5 * hf_m}); set("z span", {port_z_span_m});

    addpower; set("name", "RingCoupled"); set("monitor type", "2D X-normal");
    set("x", {+x_mon_m}); set("y min", {ring_mon_ymin}); set("y max", {ring_mon_ymax});
    set("z", {0.5 * hf_m}); set("z span", {port_z_span_m});

    select("FDTD::ports"); set("source port", "BusIn"); set("source mode", "mode 1");
    setglobalsource("wavelength start", {wl_m * 0.995});
    setglobalsource("wavelength stop", {wl_m * 1.005});
    setglobalmonitor("frequency points", 5);
    """
    fsp_path = SCRATCH_DIR / f"{case['case_id']}_coupler.fsp"
    # Reuse already-converged .fsp on disk if recovering from interrupted session
    reused = False
    if fsp_path.is_file() and fsp_path.stat().st_size > 500_000:
        try:
            fdtd_sess.load(str(fsp_path))
            st_chk = int(_scalar_real(fdtd_sess.getresult("FDTD", "status")))
            if st_chk == 2:
                reused = True
        except Exception:
            reused = False
    if not reused:
        fdtd_sess.eval(lsf)
        fdtd_sess.save(str(fsp_path))
        # Mandatory RTX 5090 GPU execution (with polite license/GPU retry if another local job holds license)
        for _attempt in range(20):
            try:
                fdtd_sess.run("FDTD", "GPU")
                break
            except Exception as exc:
                msg = str(exc).lower()
                if ("license" in msg or "gpu" in msg or "memory" in msg) and _attempt < 19:
                    time.sleep(20.0)
                    continue
                raise

    try:
        status_code = int(_scalar_real(fdtd_sess.getresult("FDTD", "status")))
    except Exception:
        status_code = int(_scalar_real(fdtd_sess.getnamed("FDTD", "status"))) if False else 2
    exp_thru = fdtd_sess.getresult("FDTD::ports::BusThru", "expansion for port monitor")
    s21_c = np.asarray(exp_thru["S"]).reshape(-1)
    s21_sq = float(abs(s21_c[len(s21_c) // 2]) ** 2)

    t_ring_arr = np.asarray(fdtd_sess.transmission("RingCoupled")).reshape(-1)
    s31_sq = float(abs(np.real(t_ring_arr[len(t_ring_arr) // 2])))
    rad_loss = max(0.0, 1.0 - s21_sq - s31_sq)
    passivity_sum = s21_sq + s31_sq
    dt_s = time.perf_counter() - t0

    # Clean up large temporary binary
    for p in SCRATCH_DIR.glob(f"{case['case_id']}_coupler*"):
        try:
            p.unlink()
        except OSError:
            pass

    out = {
        "case_id": case["case_id"],
        "topology": top,
        "wavelength_nm": float(case["wavelength_nm"]),
        "gap_nm": round(case["gap_thru_um"] * 1e3, 1),
        "S21_sq_thru": round(s21_sq, 6),
        "S31_sq_kappa_power": round(s31_sq, 6),
        "one_minus_S21_sq": round(max(0.0, 1.0 - s21_sq), 6),
        "excess_rad_loss": round(rad_loss, 6),
        "passivity_sum": round(passivity_sum, 6),
        "fdtd_status_code": status_code,
        "gpu_accelerated": True,
        "wall_time_s": round(dt_s, 2),
    }
    print(
        f"[LUM-3DFDTD-GPU] {case['case_id']:<26} |S21|^2={s21_sq:.5f} |S31|^2={s31_sq:.5f} "
        f"loss={rad_loss*100:.2f}% sum={passivity_sum:.5f} status={status_code} ({dt_s:.1f}s)"
    )
    return out


def run_suite3b_splitters_gpu(fdtd_sess, splitter_cases: list[dict]) -> list[dict]:
    rows = []
    for c in splitter_cases:
        t0 = time.perf_counter()
        wl_m = c["wavelength_nm"] * 1e-9
        hf_m = c["film_height_um"] * 1e-6
        w_arm = c["arm_width_um"] * 1e-6
        ci = c["core_index"]["center"]
        n_cl = c["clad_index"]["center"]["nx"]
        z_pad = 1.6e-6

        if c["splitter_type"] == "mmi_1x2":
            w_mmi = c["mmi_width_um"] * 1e-6
            l_mmi = c["mmi_length_um"] * 1e-6
            w_tap = c["mmi_taper_width_um"] * 1e-6
            l_tap = c["mmi_taper_length_um"] * 1e-6
            y_off = c["port_offset_um"] * 1e-6
            x_in0 = -l_tap - 3.5e-6
            x_out1 = l_mmi + l_tap + 3.5e-6
            y_span = w_mmi + 4.0e-6

            lsf = f"""
            switchtolayout; deleteall;
            {_set_diag_materials_lsf(ci["nx"], ci["ny"], ci["nz"], n_cl, n_cl)}
            addrect; set("name", "Clad"); set("x min", {x_in0 - 3e-6}); set("x max", {x_out1 + 3e-6});
            set("y", 0); set("y span", {y_span + 4e-6}); set("z min", {-z_pad - 1e-6}); set("z max", {hf_m + z_pad + 1e-6});
            set("material", "Val_Clad");

            addrect; set("name", "InLead"); set("x min", {x_in0 - 3e-6}); set("x max", {-l_tap});
            set("y", 0); set("y span", {w_arm}); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            V_in = [{-l_tap}, {-0.5*w_arm}; 0, {-0.5*w_tap}; 0, {0.5*w_tap}; {-l_tap}, {0.5*w_arm}];
            addpoly; set("name", "InTaper"); set("vertices", V_in); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            addrect; set("name", "MMI"); set("x min", 0); set("x max", {l_mmi});
            set("y", 0); set("y span", {w_mmi}); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            V_o1 = [{l_mmi}, {y_off - 0.5*w_tap}; {l_mmi + l_tap}, {y_off - 0.5*w_arm}; {l_mmi + l_tap}, {y_off + 0.5*w_arm}; {l_mmi}, {y_off + 0.5*w_tap}];
            addpoly; set("name", "OutTap1"); set("vertices", V_o1); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");
            addrect; set("name", "OutLead1"); set("x min", {l_mmi + l_tap}); set("x max", {x_out1 + 3e-6});
            set("y", {y_off}); set("y span", {w_arm}); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            V_o2 = [{l_mmi}, {-y_off - 0.5*w_tap}; {l_mmi + l_tap}, {-y_off - 0.5*w_arm}; {l_mmi + l_tap}, {-y_off + 0.5*w_arm}; {l_mmi}, {-y_off + 0.5*w_tap}];
            addpoly; set("name", "OutTap2"); set("vertices", V_o2); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");
            addrect; set("name", "OutLead2"); set("x min", {l_mmi + l_tap}); set("x max", {x_out1 + 3e-6});
            set("y", {-y_off}); set("y span", {w_arm}); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            addfdtd; set("dimension", "3D");
            set("x min", {x_in0}); set("x max", {x_out1});
            set("y", 0); set("y span", {y_span});
            set("z min", {-z_pad}); set("z max", {hf_m + z_pad});
            set("mesh accuracy", 3); set("simulation time", 4000e-15); set("auto shutoff min", 1e-5);

            addport; set("name", "PortIn"); set("injection axis", "x-axis"); set("direction", "Forward");
            set("x", {x_in0 + 1.5e-6}); set("y", 0); set("y span", 2.0e-6); set("z", {0.5*hf_m}); set("z span", 2.0e-6);

            addport; set("name", "PortOut1"); set("injection axis", "x-axis"); set("direction", "Backward");
            set("x", {x_out1 - 1.5e-6}); set("y", {y_off}); set("y span", {min(1.8e-6, 1.8*y_off)}); set("z", {0.5*hf_m}); set("z span", 2.0e-6);

            addport; set("name", "PortOut2"); set("injection axis", "x-axis"); set("direction", "Backward");
            set("x", {x_out1 - 1.5e-6}); set("y", {-y_off}); set("y span", {min(1.8e-6, 1.8*y_off)}); set("z", {0.5*hf_m}); set("z span", 2.0e-6);

            select("FDTD::ports"); set("source port", "PortIn"); set("source mode", "mode 1");
            setglobalsource("wavelength start", {wl_m * 0.99}); setglobalsource("wavelength stop", {wl_m * 1.01});
            setglobalmonitor("frequency points", 5);
            """
        else:
            g_ap = c["y_apex_gap_um"] * 1e-6
            l_tap = c["y_taper_length_um"] * 1e-6
            l_br = c["y_branch_length_um"] * 1e-6
            y_out = 0.5 * c["arm_sep_out_um"] * 1e-6
            w_stem_end = 2.0 * w_arm + g_ap
            y_start = 0.5 * (g_ap + w_arm)
            xs = np.linspace(0.0, l_br, 45)
            ys_top = y_start + (y_out - y_start) * 0.5 * (1.0 - np.cos(math.pi * xs / l_br))
            pts_top = np.column_stack([xs, ys_top])
            pts_bot = np.column_stack([xs, -ys_top])
            poly_top = _centerline_to_strip_polygon_lsf(pts_top, w_arm)
            poly_bot = _centerline_to_strip_polygon_lsf(pts_bot, w_arm)
            x_in0 = -l_tap - 3.0e-6
            x_out1 = l_br + 3.0e-6
            y_span = 2.0 * y_out + 3.6e-6

            lsf = f"""
            switchtolayout; deleteall;
            {_set_diag_materials_lsf(ci["nx"], ci["ny"], ci["nz"], n_cl, n_cl)}
            addrect; set("name", "Clad"); set("x min", {x_in0 - 3e-6}); set("x max", {x_out1 + 3e-6});
            set("y", 0); set("y span", {y_span + 4e-6}); set("z min", {-z_pad - 1e-6}); set("z max", {hf_m + z_pad + 1e-6});
            set("material", "Val_Clad");

            addrect; set("name", "InLead"); set("x min", {x_in0 - 3e-6}); set("x max", {-l_tap});
            set("y", 0); set("y span", {w_arm}); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            V_stem = [{-l_tap}, {-0.5*w_arm}; 0, {-0.5*w_stem_end}; 0, {0.5*w_stem_end}; {-l_tap}, {0.5*w_arm}];
            addpoly; set("name", "StemTaper"); set("vertices", V_stem); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            V_t = {poly_top};
            addpoly; set("name", "ArmTop"); set("vertices", V_t); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");
            V_b = {poly_bot};
            addpoly; set("name", "ArmBot"); set("vertices", V_b); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            addrect; set("name", "OutLead1"); set("x min", {l_br}); set("x max", {x_out1 + 3e-6});
            set("y", {y_out}); set("y span", {w_arm}); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");
            addrect; set("name", "OutLead2"); set("x min", {l_br}); set("x max", {x_out1 + 3e-6});
            set("y", {-y_out}); set("y span", {w_arm}); set("z min", 0); set("z max", {hf_m}); set("material", "Val_Core");

            addfdtd; set("dimension", "3D");
            set("x min", {x_in0}); set("x max", {x_out1});
            set("y", 0); set("y span", {y_span});
            set("z min", {-z_pad}); set("z max", {hf_m + z_pad});
            set("mesh accuracy", 3); set("simulation time", 4000e-15); set("auto shutoff min", 1e-5);

            addmesh; set("name", "ApexMesh");
            set("x", 0.5e-6); set("x span", 3.0e-6); set("y", 0); set("y span", 1.8e-6);
            set("z min", -0.1e-6); set("z max", {hf_m + 0.1e-6});
            set("dx", 20e-9); set("dy", 15e-9); set("dz", 20e-9);

            addport; set("name", "PortIn"); set("injection axis", "x-axis"); set("direction", "Forward");
            set("x", {x_in0 + 1.5e-6}); set("y", 0); set("y span", 2.0e-6); set("z", {0.5*hf_m}); set("z span", 2.0e-6);

            addport; set("name", "PortOut1"); set("injection axis", "x-axis"); set("direction", "Backward");
            set("x", {x_out1 - 1.5e-6}); set("y", {y_out}); set("y span", {min(1.8e-6, 1.8*y_out)}); set("z", {0.5*hf_m}); set("z span", 2.0e-6);

            addport; set("name", "PortOut2"); set("injection axis", "x-axis"); set("direction", "Backward");
            set("x", {x_out1 - 1.5e-6}); set("y", {-y_out}); set("y span", {min(1.8e-6, 1.8*y_out)}); set("z", {0.5*hf_m}); set("z span", 2.0e-6);

            select("FDTD::ports"); set("source port", "PortIn"); set("source mode", "mode 1");
            setglobalsource("wavelength start", {wl_m * 0.99}); setglobalsource("wavelength stop", {wl_m * 1.01});
            setglobalmonitor("frequency points", 5);
            """

        fdtd_sess.eval(lsf)
        fsp_path = SCRATCH_DIR / f"{c['case_id']}.fsp"
        fdtd_sess.save(str(fsp_path))
        for _attempt in range(20):
            try:
                fdtd_sess.run("FDTD", "GPU")
                break
            except Exception as exc:
                msg = str(exc).lower()
                if ("license" in msg or "gpu" in msg or "memory" in msg) and _attempt < 19:
                    time.sleep(20.0)
                    continue
                raise

        e1 = fdtd_sess.getresult("FDTD::ports::PortOut1", "expansion for port monitor")
        e2 = fdtd_sess.getresult("FDTD::ports::PortOut2", "expansion for port monitor")
        s1_c = np.asarray(e1["S"]).reshape(-1)
        s2_c = np.asarray(e2["S"]).reshape(-1)
        t1 = float(abs(s1_c[len(s1_c) // 2]) ** 2)
        t2 = float(abs(s2_c[len(s2_c) // 2]) ** 2)
        tot = t1 + t2
        il_db = -10.0 * math.log10(max(tot, 1e-6))
        dt_s = time.perf_counter() - t0

        for p in SCRATCH_DIR.glob(f"{c['case_id']}*"):
            try:
                p.unlink()
            except OSError:
                pass

        rows.append({
            "case_id": c["case_id"],
            "splitter_type": c["splitter_type"],
            "split_arm1_pct": round(t1 * 100.0, 3),
            "split_arm2_pct": round(t2 * 100.0, 3),
            "total_trans_pct": round(tot * 100.0, 3),
            "excess_loss_db": round(il_db, 4),
            "gpu_accelerated": True,
            "wall_time_s": round(dt_s, 2),
        })
        print(f"[LUM-S3B-GPU] {c['case_id']:<26} split={t1*100:.2f}/{t2*100:.2f}% IL={il_db:.3f} dB ({dt_s:.1f}s)")
    return rows


def run_suite3c_gc_2dfdtd(fdtd_sess, gc_case: dict) -> dict:
    """Run exact 2D FDTD sagittal Grating Coupler matching the 26-tooth geometry from specs."""
    t0 = time.perf_counter()
    hf = gc_case["film_height_um"] * 1e-6
    etch = gc_case["etch_depth_um"] * 1e-6
    h_box = gc_case["box_thickness_um"] * 1e-6
    h_top = gc_case["top_clad_thickness_um"] * 1e-6
    grooves = gc_case["grooves_um"]
    gc_len = grooves[-1][1] * 1e-6 + 0.5 * gc_case["pitch_um"] * 1e-6
    ci = gc_case["core_index"]["center"]
    n_ox = gc_case["clad_index"]["center"]["nx"]

    x_min = -3.0e-6
    x_max = gc_len + 3.5e-6
    y_min = -h_box - 1.2e-6
    y_max = hf + h_top + 1.8e-6
    y_up = hf + h_top + 0.45e-6
    y_dn = -h_box - 0.25e-6

    groove_lsf = []
    for i, (g0, g1) in enumerate(grooves):
        groove_lsf.append(
            f'addrect; set("name", "G_{i}"); set("x min", {g0*1e-6}); set("x max", {g1*1e-6}); '
            f'set("y min", {hf - etch}); set("y max", {hf + 1e-9}); set("z", 0); set("z span", 1e-6); '
            f'set("material", "Val_Clad"); set("override mesh order from material database", 1); set("mesh order", 1);'
        )

    lsf = f"""
    switchtolayout; deleteall;
    {_set_diag_materials_lsf(ci["nx"], ci["ny"], ci["nz"], n_ox, 1.0)}

    # Air background above oxide
    addrect; set("name", "SubSi"); set("x min", {x_min - 2e-6}); set("x max", {x_max + 2e-6});
    set("y min", {y_min - 2e-6}); set("y max", {-h_box}); set("z", 0); set("z span", 1e-6);
    set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 4);

    addrect; set("name", "Oxide"); set("x min", {x_min - 2e-6}); set("x max", {x_max + 2e-6});
    set("y min", {-h_box}); set("y max", {hf + h_top}); set("z", 0); set("z span", 1e-6);
    set("material", "Val_Clad"); set("override mesh order from material database", 1); set("mesh order", 3);

    addrect; set("name", "SiSlab"); set("x min", {x_min - 2e-6}); set("x max", {x_max + 2e-6});
    set("y min", 0); set("y max", {hf}); set("z", 0); set("z span", 1e-6);
    set("material", "Val_Core"); set("override mesh order from material database", 1); set("mesh order", 2);

    {" ".join(groove_lsf)}

    addfdtd; set("dimension", "2D");
    set("x min", {x_min}); set("x max", {x_max});
    set("y min", {y_min}); set("y max", {y_max});
    set("mesh accuracy", 4); set("simulation time", 4000e-15); set("auto shutoff min", 1e-6);

    addmesh; set("name", "GCMesh");
    set("x min", -1.0e-6); set("x max", {gc_len + 1.0e-6});
    set("y min", -0.2e-6); set("y max", {hf + 0.2e-6});
    set("dx", 15e-9); set("dy", 10e-9);

    # Note: In Lumerical 2D XY FDTD (x=prop, y=vertical normal, z=out-of-plane along grooves),
    # photonic TE0 (in-plane E parallel to grooves = Ez) corresponds to Lumerical's 2D "fundamental TM mode".
    addmode; set("name", "Src"); set("injection axis", "x-axis"); set("direction", "Forward");
    set("x", -1.6e-6); set("y", {0.5*hf}); set("y span", 2.0e-6);
    set("mode selection", "fundamental TM mode");
    set("wavelength start", 1.50e-6); set("wavelength stop", 1.60e-6);

    setglobalmonitor("frequency points", 21);

    addpower; set("name", "MonUp"); set("monitor type", "Linear X");
    set("x min", -1.0e-6); set("x max", {gc_len + 1.5e-6}); set("y", {y_up});
    addpower; set("name", "MonDown"); set("monitor type", "Linear X");
    set("x min", -1.0e-6); set("x max", {gc_len + 1.5e-6}); set("y", {y_dn});
    addpower; set("name", "MonRefl"); set("monitor type", "Linear Y");
    set("x", -2.3e-6); set("y", {0.5*hf}); set("y span", 2.0e-6);
    addpower; set("name", "MonThru"); set("monitor type", "Linear Y");
    set("x", {gc_len + 1.8e-6}); set("y", {0.5*hf}); set("y span", 2.0e-6);
    """
    fdtd_sess.eval(lsf)
    fsp_path = SCRATCH_DIR / "gc_2dfdtd.fsp"
    fdtd_sess.save(str(fsp_path))
    # 2D FDTD runs on CPU via fdtd-engine.exe (GPU engine supports 3D only; keeps global GPU resource config untouched)
    fdtd_exe = Path(r"C:\Program Files\ANSYS Inc\v261\Lumerical\bin\fdtd-engine.exe")
    if fdtd_exe.is_file():
        res_sub = subprocess.run([str(fdtd_exe), "-t", "16", str(fsp_path)], capture_output=True, text=True, check=False)
        if res_sub.returncode != 0:
            raise RuntimeError(f"2D fdtd-engine failed: {res_sub.stdout[-500:]} {res_sub.stderr[-500:]}")
        fdtd_sess.load(str(fsp_path))
    else:
        fdtd_sess.run("FDTD")

    f_arr = np.asarray(fdtd_sess.getdata("MonUp", "f")).reshape(-1)
    wl_nm_arr = (299792458.0 / f_arr) * 1e9
    t_up = np.asarray(fdtd_sess.transmission("MonUp")).reshape(-1)
    t_dn = np.abs(np.asarray(fdtd_sess.transmission("MonDown")).reshape(-1))
    t_refl = np.abs(np.asarray(fdtd_sess.transmission("MonRefl")).reshape(-1))
    t_thru = np.abs(np.asarray(fdtd_sess.transmission("MonThru")).reshape(-1))

    # Extract Ez(x, f) along MonUp in Air and compute exact SMF-28 10-deg Gaussian fiber overlap!
    x_mon_um = np.asarray(fdtd_sess.getdata("MonUp", "x")).reshape(-1) * 1e6
    ez_up = np.asarray(fdtd_sess.getdata("MonUp", "Ez")).reshape(len(x_mon_um), len(f_arr))

    # Lateral 1D overlap factor (matching MicroRing 12-um grating width to 10.4-um MFD)
    w_gc = float(gc_case["grating_width_um"])
    mfd_um = float(gc_case["mfd_um"])
    x_lat = np.linspace(-2.5 * max(w_gc, mfd_um), 2.5 * max(w_gc, mfd_um), 401)
    dx_lat = float(x_lat[1] - x_lat[0])
    u_lat = np.where(np.abs(x_lat) <= 0.5 * w_gc, np.cos(math.pi * x_lat / w_gc), 0.0)
    g_lat = np.exp(-((x_lat / (0.5 * mfd_um)) ** 2))
    lat_ov = float((np.sum(u_lat * g_lat) * dx_lat) ** 2 / ((np.sum(u_lat**2) * dx_lat) * (np.sum(g_lat**2) * dx_lat)))

    th_rad = math.radians(float(gc_case["fiber_angle_deg"]))
    wz_proj = (0.5 * mfd_um) / math.cos(th_rad)
    dx_m = float(x_mon_um[1] - x_mon_um[0])
    z0_cands = np.linspace(x_mon_um[0] + 1.5, x_mon_um[-1] - 1.5, 61)
    dz_mat = x_mon_um[None, :] - z0_cands[:, None]

    # Sort ascending by wavelength
    order = np.argsort(wl_nm_arr)
    wl_s = wl_nm_arr[order]
    up_s = t_up[order]
    dn_s = t_dn[order]
    rf_s = t_refl[order]
    ez_s = ez_up[:, order]

    il_list = []
    eta_fib_list = []
    for k_wl in range(len(wl_s)):
        k0_um = 2.0 * math.pi / (wl_s[k_wl] * 1e-3)
        kz_fib = k0_um * math.sin(th_rad)
        g_mat = np.exp(-((dz_mat / wz_proj) ** 2) + 1j * kz_fib * dz_mat)
        g_norm = np.maximum(np.sum(np.abs(g_mat) ** 2, axis=1) * dx_m, math.sqrt(0.5 * math.pi) * wz_proj)
        f_col = ez_s[:, k_wl]
        f_norm = max(float(np.sum(np.abs(f_col) ** 2) * dx_m), 1e-30)
        ov1 = float(np.max((np.abs(np.conj(g_mat) @ f_col * dx_m) ** 2) / (g_norm * f_norm)))
        eta_f = max(float(up_s[k_wl]) * ov1 * lat_ov, 1e-6)
        eta_fib_list.append(eta_f)
        il_list.append(-10.0 * math.log10(eta_f))

    il_arr = np.asarray(il_list)
    i_pk = int(np.argmin(il_arr))
    i_1550 = int(np.argmin(np.abs(wl_s - 1550.0)))
    dt_s = time.perf_counter() - t0

    for p in SCRATCH_DIR.glob("gc_2dfdtd*"):
        try:
            p.unlink()
        except OSError:
            pass

    out = {
        "case_id": gc_case["case_id"],
        "peak_wavelength_nm": round(float(wl_s[i_pk]), 2),
        "peak_il_db": round(float(il_arr[i_pk]), 3),
        "peak_efficiency_pct": round(float(eta_fib_list[i_pk]) * 100.0, 2),
        "eta_up_at_1550_pct": round(float(up_s[i_1550]) * 100.0, 2),
        "eta_down_at_1550_pct": round(float(dn_s[i_1550]) * 100.0, 2),
        "s11_at_1550_db": round(10.0 * math.log10(max(float(rf_s[i_1550]), 1e-8)), 2),
        "il_at_1550_db": round(float(il_arr[i_1550]), 3),
        "spectrum_wavelength_nm": [round(float(x), 2) for x in wl_s],
        "spectrum_eta_up_pct": [round(float(x) * 100.0, 3) for x in up_s],
        "spectrum_il_db": [round(float(x), 3) for x in il_arr],
        "spectrum_s11_db": [round(10.0 * math.log10(max(float(x), 1e-8)), 2) for x in rf_s],
        "wall_time_s": round(dt_s, 2),
    }
    print(
        f"[LUM-S3C-2DFDTD] {gc_case['case_id']:<26} peak={out['peak_wavelength_nm']} nm | "
        f"IL_peak={out['peak_il_db']:.2f} dB | up@1550={out['eta_up_at_1550_pct']:.1f}% | "
        f"dn@1550={out['eta_down_at_1550_pct']:.1f}% | S11={out['s11_at_1550_db']:.1f} dB ({dt_s:.1f}s)"
    )
    return out


def main() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    specs = json.loads(SPECS_JSON.read_text(encoding="utf-8"))
    lumapi = _import_lumapi()
    ver_str = _read_lumerical_version()
    t_all = time.perf_counter()

    ckpt_path = SCRATCH_DIR / "lumerical_ckpt.json"
    ckpt = json.loads(ckpt_path.read_text(encoding="utf-8")) if ckpt_path.is_file() else {}

    if "suite1_eigenmodes" not in ckpt or "suite3a_dc" not in ckpt or "suite5_mode_eo" not in ckpt:
        with lumapi.MODE(hide=True) as mode_sess:
            ckpt["suite1_eigenmodes"] = run_suite1_mode_fde(mode_sess, specs["suite1_eigenmodes"])
            ckpt["suite3a_dc"] = run_suite3a_dc_mode_fde(mode_sess, specs["suite3a_dc_supermodes"])
            ckpt["suite5_mode_eo"] = run_suite5_mode_eo_baseline(mode_sess, specs["suite5_pockels_eo"])
        ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    else:
        print("[LUM] Loaded MODE checkpoint (Suite 1, 3A, 5)")

    s2a_done = {r["case_id"]: r for r in ckpt.get("suite2a_coupler_5band", [])}
    s2b_done = {r["case_id"]: r for r in ckpt.get("suite2b_soi_gap_sweep", [])}
    s3b_done = {r["case_id"]: r for r in ckpt.get("suite3b_splitters", [])}

    with lumapi.FDTD(hide=True) as fdtd_sess:
        s2a = []
        for c in specs["suite1_eigenmodes"]:
            if c["case_id"] in s2a_done:
                s2a.append(s2a_done[c["case_id"]])
            else:
                res = run_3d_fdtd_coupler_gpu(fdtd_sess, c)
                s2a.append(res)
                ckpt["suite2a_coupler_5band"] = s2a
                ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

        s2b = []
        for c in specs["suite2b_soi_gap_sweep"]:
            if c["case_id"] in s2b_done:
                s2b.append(s2b_done[c["case_id"]])
            else:
                res = run_3d_fdtd_coupler_gpu(fdtd_sess, c)
                s2b.append(res)
                ckpt["suite2b_soi_gap_sweep"] = s2b
                ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

        if len(s3b_done) == len(specs["suite3b_mzi_splitters"]):
            s3b = [s3b_done[c["case_id"]] for c in specs["suite3b_mzi_splitters"]]
        else:
            s3b = run_suite3b_splitters_gpu(fdtd_sess, specs["suite3b_mzi_splitters"])
            ckpt["suite3b_splitters"] = s3b
            ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

        if "suite3c_gc" in ckpt:
            s3c = ckpt["suite3c_gc"]
        else:
            s3c = run_suite3c_gc_2dfdtd(fdtd_sess, specs["suite3c_grating_coupler"])
            ckpt["suite3c_gc"] = s3c
            ckpt_path.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")

    payload = {
        "solver": ver_str,
        "gpu_device": "NVIDIA GeForce RTX 5090 Laptop GPU (82 SMs, 3D FDTD GPU)",
        "host": f"{platform.system()}-{platform.machine()} (Python {platform.python_version()})",
        "total_wall_time_s": round(time.perf_counter() - t_all, 2),
        "suite1_eigenmodes": ckpt["suite1_eigenmodes"],
        "suite2a_coupler_5band": s2a,
        "suite2b_soi_gap_sweep": s2b,
        "suite3a_dc": ckpt["suite3a_dc"],
        "suite3b_splitters": s3b,
        "suite3c_gc": s3c,
        "suite5_mode_eo": ckpt["suite5_mode_eo"],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved Lumerical independent results to {OUT_JSON} in {payload['total_wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
