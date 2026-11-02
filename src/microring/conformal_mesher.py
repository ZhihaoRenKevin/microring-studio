"""Pure-BSD Conformal Waveguide & Multilayer Triangular Mesher (`scikit-fem`).

Eliminates GPL (`gmsh` / `femwell`) dependencies from the default eigenmode solver
stack so `MicroRing Studio` can be distributed under permissive MIT / BSD-3-Clause
or commercial licenses on both macOS and Windows with zero external C++ mesher
binaries required.

Key features:
  - Exact horizontal layer interfaces (`box`, `slab`, `core`, `interlayer`,
    `upper_core`, `cap`, `top_clad`).
  - Exact transfinite row-wise conformal mapping onto angled trapezoidal sidewalls
    (e.g., 80 deg `4H-SiC`, `ScAlN`, or `TFLN` ridges) — zero Yee-grid staircasing.
  - Strictly mirror-symmetric triangulation (`x <-> -x`) when radial domain bounds
    are symmetric, eliminating mesh-induced polarization/parity bias.
  - Smooth geometric grading (`1.18x`) from fine `core_mesh_res_um` inside active
    waveguide regions out to `clad_mesh_res_um` at outer cladding boundaries.
"""
from __future__ import annotations

import math
import numpy as np
import shapely
from skfem import MeshTri

from .eigenmode_solver import WaveguideCrossSectionSpec


def _graded_1d_segment(
    a: float,
    b: float,
    h_start: float,
    h_end: float,
    growth: float = 1.18,
) -> np.ndarray:
    """Generate 1D strictly ascending coordinates on `[a, b]` with smooth step grading.

    Args:
      a: Segment start coordinate [um].
      b: Segment end coordinate [um] (`b > a`).
      h_start: Target mesh spacing near `a` [um].
      h_end: Target mesh spacing near `b` [um].
      growth: Geometric progression factor (`> 1.0`).
    """
    span = float(b - a)
    if span <= 1e-12:
        return np.array([float(a)], dtype=float)
    h0 = max(float(h_start), 1e-4)
    h1 = max(float(h_end), 1e-4)

    # Uniform fine spacing when start & end resolutions are comparable
    if abs(h1 - h0) <= 0.15 * min(h0, h1):
        h_mid = 0.5 * (h0 + h1)
        n_seg = max(1, int(math.ceil(span / h_mid)))
        return np.linspace(float(a), float(b), n_seg + 1)

    # March from fine end toward coarse end with capped geometric growth
    reverse = h0 > h1
    h_fine = min(h0, h1)
    h_coarse = max(h0, h1)

    widths: list[float] = []
    rem = span
    cur = h_fine
    while rem > 1.35 * cur:
        widths.append(cur)
        rem -= cur
        cur = min(cur * growth, h_coarse)
    # Split final remainder cleanly if large
    if rem > 0:
        n_tail = max(1, int(round(rem / cur)))
        for _ in range(n_tail):
            widths.append(rem / n_tail)

    w_arr = np.asarray(widths, dtype=float)
    w_arr *= span / float(np.sum(w_arr))
    if reverse:
        w_arr = w_arr[::-1]
    coords = np.empty(len(w_arr) + 1, dtype=float)
    coords[0] = float(a)
    coords[1:] = float(a) + np.cumsum(w_arr)
    coords[-1] = float(b)
    return coords


def build_conformal_waveguide_mesh(
    spec: WaveguideCrossSectionSpec,
    *,
    core_mesh_res_um: float = 0.035,
    clad_mesh_res_um: float = 0.22,
) -> MeshTri:
    """Build a 100% BSD-licensed conformal `skfem.MeshTri` (in SI metres) for `spec`.

    All material interfaces (`box`, `slab`, `core`, `upper_core`, `cap`, `top_clad`)
    including angled trapezoidal sidewalls align with exact mesh edges.
    """
    xmin, xmax = spec.x_bounds_um()
    ymin, ymax = spec.y_bounds_um()

    h_core_res = max(float(core_mesh_res_um), 0.008)
    h_clad_res = max(float(clad_mesh_res_um), h_core_res * 1.2)
    h_cap_res = min(h_core_res * 0.5, 0.015)

    # ------------------------------------------------------------------
    # 1. Construct exact vertical (y) coordinate grid [um]
    # ------------------------------------------------------------------
    y_segments: list[np.ndarray] = []

    # BOX: coarse at ymin -> fine at y = 0
    y_segments.append(_graded_1d_segment(ymin, 0.0, h_clad_res, h_core_res * 1.15))

    # Slab (if present) + Primary Core ridge
    h_s = float(spec.h_slab_um)
    h_f = float(spec.h_film_um)
    if h_s > 1e-9:
        y_segments.append(_graded_1d_segment(0.0, h_s, h_core_res, h_core_res))
    if h_f > h_s + 1e-9:
        y_segments.append(_graded_1d_segment(h_s, h_f, h_core_res, h_core_res))

    y_top_active = h_f
    has_upper = spec.upper_core_h_um > 1e-9 and spec.upper_core_material_fn is not None
    has_cap = spec.cap_h_um > 1e-9 and spec.cap_material_fn is not None

    y0_up = h_f + float(spec.interlayer_h_um)
    y1_up = y0_up + float(spec.upper_core_h_um)
    if has_upper:
        if spec.interlayer_h_um > 1e-9:
            y_segments.append(
                _graded_1d_segment(h_f, y0_up, min(h_core_res * 0.75, 0.025), min(h_core_res * 0.75, 0.025))
            )
        y_segments.append(_graded_1d_segment(y0_up, y1_up, h_core_res, h_core_res))
        y_top_active = y1_up
    elif spec.interlayer_h_um > 1e-9 and has_cap:
        y_segments.append(_graded_1d_segment(h_f, y0_up, h_core_res * 0.8, h_core_res * 0.8))
        y_top_active = y0_up

    y_cap0 = y_top_active
    y_cap1 = y_cap0 + float(spec.cap_h_um)
    if has_cap:
        y_segments.append(_graded_1d_segment(y_cap0, y_cap1, h_cap_res, h_cap_res))
        y_top_active = y_cap1

    # Top cladding: fine at y_top_active -> coarse at ymax
    y_segments.append(_graded_1d_segment(y_top_active, ymax, h_core_res * 1.15, h_clad_res))

    y_rows = np.concatenate([seg if idx == 0 else seg[1:] for idx, seg in enumerate(y_segments)])
    ny = len(y_rows)

    # ------------------------------------------------------------------
    # 2. Construct transfinite conformal horizontal mapping x(i, y) [um]
    # ------------------------------------------------------------------
    hw_bot = 0.5 * float(spec.w_bottom_um)
    hw_top = 0.5 * float(spec.ridge_top_width_um())

    if has_upper:
        wb_up = float(spec.upper_w_bottom_um) if spec.upper_w_bottom_um is not None else 2.0 * hw_top
        inv_tu = 0.0 if spec.upper_sidewall_deg == 90.0 else 1.0 / math.tan(math.radians(spec.upper_sidewall_deg))
        wt_up = wb_up - 2.0 * float(spec.upper_core_h_um) * inv_tu
        hw_up_bot = 0.5 * wb_up
        hw_up_top = 0.5 * wt_up
    else:
        hw_up_bot = hw_top
        hw_up_top = hw_top

    if has_cap:
        default_cw = (2.0 * hw_up_top) if has_upper else (2.0 * hw_top)
        hw_cap = 0.5 * (float(spec.cap_w_um) if spec.cap_w_um is not None else default_cw)
    else:
        hw_cap = hw_up_top if has_upper else hw_top

    # Primary outer core half-width profile hw_main(y)
    def _hw_main_at_y(y_val: float) -> float:
        if y_val <= h_s:
            return hw_bot
        if y_val <= h_f:
            t = (y_val - h_s) / max(h_f - h_s, 1e-12)
            return hw_bot + t * (hw_top - hw_bot)
        if has_upper:
            if y_val <= y0_up:
                if y0_up - h_f > 1e-9:
                    t = (y_val - h_f) / (y0_up - h_f)
                    return hw_top + t * (hw_up_bot - hw_top)
                return hw_up_bot
            if y_val <= y1_up:
                t = (y_val - y0_up) / max(y1_up - y0_up, 1e-12)
                return hw_up_bot + t * (hw_up_top - hw_up_bot)
            return hw_up_top
        return hw_top

    # Check whether we need a second nested control curve (e.g. twin-core DC/Y-branch, narrow/wide cap, or step upper core)
    is_twin_core = float(getattr(spec, "twin_core_gap_um", 0.0)) > 0.0
    step_at_upper = has_upper and (spec.interlayer_h_um <= 1e-9) and (abs(hw_up_bot - hw_top) > 1e-5)
    ref_top_hw = hw_up_top if has_upper else hw_top
    step_at_cap = has_cap and (abs(hw_cap - ref_top_hw) > 1e-5)

    hw_ref_nom = max(0.5 * (hw_bot + hw_top), 0.05)

    if not is_twin_core and not step_at_upper and not step_at_cap:
        # Single smooth conformal control curve +-hw_main(y)
        x_core_pos = _graded_1d_segment(0.0, hw_ref_nom, h_core_res, h_core_res)
        x_clad_right = _graded_1d_segment(hw_ref_nom, xmax, h_core_res, h_clad_res)
        x_clad_left = -_graded_1d_segment(hw_ref_nom, abs(xmin), h_core_res, h_clad_res)[::-1]

        s_core_pos = x_core_pos / hw_ref_nom  # in [0, 1]
        s_core_neg = -s_core_pos[::-1][:-1]   # in [-1, 0)
        s_right = (x_clad_right[1:] - hw_ref_nom) / max(xmax - hw_ref_nom, 1e-9)  # in (0, 1]
        s_left = (x_clad_left[:-1] + hw_ref_nom) / max(abs(xmin) - hw_ref_nom, 1e-9)  # in [-1, 0)

        nx = len(s_left) + len(s_core_neg) + len(s_core_pos) + len(s_right)
        xx = np.empty((ny, nx), dtype=float)
        for j, yv in enumerate(y_rows):
            hw_y = _hw_main_at_y(float(yv))
            row_left = -hw_y + s_left * (abs(xmin) - hw_y)
            row_cneg = s_core_neg * hw_y
            row_cpos = s_core_pos * hw_y
            row_right = hw_y + s_right * (xmax - hw_y)
            xx[j, :] = np.concatenate([row_left, row_cneg, row_cpos, row_right])
    else:
        # Two non-crossing positive control curves: 0 < c_in(y) < c_out(y) < min(|xmin|, xmax)
        xc_twin = 0.5 * float(getattr(spec, "twin_core_gap_um", 0.0)) + hw_bot

        def _c_in_out(y_val: float) -> tuple[float, float]:
            if is_twin_core:
                hm = _hw_main_at_y(y_val)
                return max(xc_twin - hm, 0.004), xc_twin + hm
            if step_at_cap and not step_at_upper:
                hm = _hw_main_at_y(y_val)
                ratio_cap = min(hw_cap, ref_top_hw) / max(hw_cap, ref_top_hw)
                if hw_cap < ref_top_hw:
                    c_out = hm
                    c_in = hm * ratio_cap if y_val < y_cap0 else hw_cap
                else:
                    c_in = hm
                    c_out = (hm / ratio_cap) if y_val < y_cap0 else hw_cap
                return c_in, c_out
            # Step between lower core and upper core at y = h_f
            c1 = hw_bot + ((y_val - h_s) / max(h_f - h_s, 1e-12)) * (hw_top - hw_bot) if h_s <= y_val <= h_f else (hw_bot if y_val < h_s else hw_top)
            c2 = hw_up_bot + ((y_val - y0_up) / max(y1_up - y0_up, 1e-12)) * (hw_up_top - hw_up_bot) if y0_up <= y_val <= y1_up else (hw_up_bot if y_val < y0_up else hw_up_top)
            return min(c1, c2), max(c1, c2)

        cin_nom, cout_nom = _c_in_out(0.5 * (h_s + h_f))
        s01 = _graded_1d_segment(0.0, max(cin_nom, h_core_res), h_core_res, h_core_res) / max(cin_nom, h_core_res)
        s12 = _graded_1d_segment(cin_nom, max(cout_nom, cin_nom + h_core_res), h_core_res, h_core_res)[1:]
        s12 = (s12 - cin_nom) / max(cout_nom - cin_nom, h_core_res)
        x_clad_r = _graded_1d_segment(cout_nom, xmax, h_core_res, h_clad_res)[1:]
        sr = (x_clad_r - cout_nom) / max(xmax - cout_nom, 1e-9)
        x_clad_l = -_graded_1d_segment(cout_nom, abs(xmin), h_core_res, h_clad_res)[::-1][:-1]
        sl = (x_clad_l + cout_nom) / max(abs(xmin) - cout_nom, 1e-9)

        rows_list = []
        for yv in y_rows:
            ci, co = _c_in_out(float(yv))
            pos = np.concatenate([s01 * ci, ci + s12 * (co - ci), co + sr * (xmax - co)])
            neg = np.concatenate([-co + sl * (abs(xmin) - co), -(ci + s12 * (co - ci))[::-1], -(s01[1:] * ci)[::-1]])
            rows_list.append(np.concatenate([neg, pos]))
        xx = np.vstack(rows_list)
        nx = xx.shape[1]

    yy = np.broadcast_to(y_rows[:, np.newaxis], (ny, nx)).copy()

    # ------------------------------------------------------------------
    # 3. Mirror-symmetric quad triangulation & subdomain classification
    # ------------------------------------------------------------------
    p_um = np.vstack([xx.ravel(), yy.ravel()])
    p_m = np.ascontiguousarray(p_um * 1e-6)

    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    ii_f = ii.ravel()
    jj_f = jj.ravel()

    v00 = jj_f * nx + ii_f
    v10 = v00 + 1
    v01 = v00 + nx
    v11 = v01 + 1

    # Symmetric diagonal split around x = 0 so x <-> -x symmetry is exact
    x_quad_mid = 0.25 * (p_um[0, v00] + p_um[0, v10] + p_um[0, v01] + p_um[0, v11])
    right_mask = x_quad_mid >= 0.0

    n_quads = len(v00)
    tris = np.empty((2 * n_quads, 3), dtype=np.int64)

    # Left half: (v00, v10, v11) and (v00, v11, v01)
    left_mask = ~right_mask
    tris[0::2][left_mask] = np.column_stack([v00[left_mask], v10[left_mask], v11[left_mask]])
    tris[1::2][left_mask] = np.column_stack([v00[left_mask], v11[left_mask], v01[left_mask]])

    # Right half (mirror reflection): (v00, v10, v01) and (v10, v11, v01)
    tris[0::2][right_mask] = np.column_stack([v00[right_mask], v10[right_mask], v01[right_mask]])
    tris[1::2][right_mask] = np.column_stack([v10[right_mask], v11[right_mask], v01[right_mask]])

    t_mat = np.ascontiguousarray(tris.T)  # shape (3, N_tri)

    # Classify triangle centroids against exact ordered Shapely cross-section polygons
    xc_um = (p_um[0, t_mat[0]] + p_um[0, t_mat[1]] + p_um[0, t_mat[2]]) / 3.0
    yc_um = (p_um[1, t_mat[0]] + p_um[1, t_mat[1]] + p_um[1, t_mat[2]]) / 3.0

    polygons = spec.build_polygons_um()
    assigned = np.zeros(t_mat.shape[1], dtype=bool)
    subdomains: dict[str, np.ndarray] = {}

    for reg_name, poly in polygons.items():
        inside = (~assigned) & shapely.contains_xy(poly, xc_um, yc_um)
        if np.any(inside):
            subdomains[reg_name] = np.flatnonzero(inside).astype(np.int64)
            assigned |= inside

    # Fallback for any centroid sitting within floating-point epsilon of outer box boundary
    if not np.all(assigned):
        rem_idx = np.flatnonzero(~assigned)
        for idx in rem_idx:
            target = "box" if yc_um[idx] < 0.0 else "top_clad"
            subdomains[target] = np.sort(
                np.append(subdomains.get(target, np.empty(0, dtype=np.int64)), idx)
            )

    return MeshTri(p_m, t_mat).with_subdomains(subdomains)
