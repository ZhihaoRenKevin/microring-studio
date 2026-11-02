"""Bidirectional GDSII / OASIS Layout Bridge (`microring.gds_bridge`).

Supports BOTH directions of layout-driven integrated photonics design:
  1. **GDS -> Geometry Extraction -> Eigenmode/CMT Simulation (`analyze_uploaded_gds`)**:
     Reads any `.gds` or `.oas` file (`gdstk` + `shapely`), inspects cells and layers,
     automatically identifies annular ring/racetrack/dual-ring cavities and straight
     or concentric wrap-around pulley buses, and measures exact `R`, `W_ring`, `W_bus`,
     `gap_thru`, `gap_drop`, `L_s`, and `pulley_angle_deg` directly from polygons!
  2. **Design / Parameter Sweep -> Foundry GDS + KLayout `.lyp` (`export_mrr_gds_with_pdk`,
     `export_sweep_array_gds`, `write_klayout_layer_props`)**:
     Creates clean single-device layouts (with port pins, heater/GSG pads & labels)
     or MPW parameter-sweep test arrays ready for KLayout inspection and fabrication.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import gdstk
import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, unary_union

from .mrr_topologies import MRRDesignConfig, TOPOLOGY_CATALOG, export_mrr_gds

# Standard MicroRing role-to-GDS layer table (compatible with KLayout .lyp export)
MICRORING_LAYERS: dict[str, tuple[int, int, str, str]] = {
    "core": (1, 0, "#38bdf8", "Waveguide Core Etch"),
    "slab": (2, 0, "#0284c7", "Partial Rib Slab"),
    "shallow_etch": (4, 0, "#0ea5e9", "Grating Coupler Shallow Etch Grooves"),
    "heater": (10, 0, "#f59e0b", "TiN Thermo-Optic Heater"),
    "electrode": (11, 0, "#eab308", "GSG Electro-Optic Pads"),
    "facet_trench": (40, 0, "#f43f5e", "Deep Facet Dicing / Undercut Trench"),
    "pin": (65, 0, "#34d399", "Optical Port Pin"),
    "label": (66, 0, "#e2e8f0", "Annotation Text"),
}


def write_klayout_layer_props(output_lyp_path: str | Path) -> str:
    """Write a KLayout `.lyp` layer-properties file matching `MICRORING_LAYERS`."""
    out = Path(output_lyp_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for role, (lyr, dt, hex_col, desc) in MICRORING_LAYERS.items():
        blocks.append(
            f"""  <properties>
    <frame-color>{hex_col}</frame-color>
    <fill-color>{hex_col}</fill-color>
    <frame-brightness>0</frame-brightness>
    <fill-brightness>0</fill-brightness>
    <dither-pattern>I5</dither-pattern>
    <valid>true</valid>
    <visible>true</visible>
    <transparent>true</transparent>
    <width>1</width>
    <marked>false</marked>
    <name>{role} - {desc} ({lyr}/{dt})</name>
    <source>{lyr}/{dt}@1</source>
  </properties>"""
        )
    xml = '<?xml version="1.0" encoding="utf-8"?>\n<layer-properties>\n' + "\n".join(blocks) + "\n</layer-properties>\n"
    out.write_text(xml, encoding="utf-8")
    return str(out.resolve())


def export_sweep_array_gds(
    base_cfg: MRRDesignConfig,
    param_name: str,
    values: list[float],
    output_path: str | Path,
    pitch_x_um: float | None = None,
) -> str:
    """Generate a multi-device MPW parameter-sweep chiplet `.gds` (e.g. gap or pulley sweep)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    top = lib.new_cell(f"MicroRing_Sweep_{param_name}")

    default_span = 2.6 * base_cfg.ring_radius_um + base_cfg.straight_length_um + 55.0
    dx_pitch = float(pitch_x_um) if pitch_x_um is not None else max(default_span, 140.0)

    for idx, val in enumerate(values):
        cfg_i = replace(base_cfg, **{param_name: float(val)})
        # Build sub-device in temporary library and copy cell
        tmp_path = out.parent / f"._tmp_sub_{idx}.gds"
        try:
            export_mrr_gds(cfg_i, tmp_path)
            sub_lib = gdstk.read_gds(str(tmp_path))
            sub_cell = sub_lib.top_level()[0]
            sub_cell.name = f"MRR_{idx}_{param_name}_{val:.3f}"
            lib.add(sub_cell)
            top.add(gdstk.Reference(sub_cell, origin=(idx * dx_pitch, 0.0)))
            top.add(gdstk.Label(
                f"#{idx+1}: {param_name}={val:.3f}",
                (idx * dx_pitch, -(base_cfg.ring_radius_um + 22.0)),
                layer=66,
            ))
        finally:
            tmp_path.unlink(missing_ok=True)

    lib.write_gds(str(out))
    write_klayout_layer_props(out.with_suffix(".lyp"))
    return str(out.resolve())


def _measure_local_bus_width(bus_poly: Polygon, near_pt: Point) -> float:
    """Estimate waveguide width of `bus_poly` near coupling point `near_pt` [um]."""
    # Cast short orthogonal chords at 0 deg and 90 deg and take the narrower waveguide cross-width
    widths = []
    for dx, dy in ((1.0, 0.0), (0.0, 1.0), (0.7071, 0.7071), (-0.7071, 0.7071)):
        chord = LineString([
            (near_pt.x - 6.0 * dx, near_pt.y - 6.0 * dy),
            (near_pt.x + 6.0 * dx, near_pt.y + 6.0 * dy),
        ])
        inter = bus_poly.intersection(chord)
        if inter.is_empty:
            continue
        geoms = [inter] if isinstance(inter, LineString) else list(getattr(inter, "geoms", []))
        for seg in geoms:
            if isinstance(seg, LineString) and seg.distance(near_pt) < 0.25 and 0.12 <= seg.length <= 5.0:
                widths.append(float(seg.length))
    if widths:
        return round(float(min(widths)), 3)
    # Fallback: 2 * Area / Perimeter
    return round(float(2.0 * bus_poly.area / max(bus_poly.length, 1e-6)), 3)


def _detect_pulley_wrap_angle_deg(
    ring_ext_line: LineString,
    bus_poly: Polygon,
    cx: float,
    cy: float,
    gap_min_um: float,
    r_outer_um: float,
) -> tuple[bool, float]:
    """Distinguish a concentric wrap-around pulley bus from a straight point-coupled bus."""
    # Sample 360 points around ring outer circumference and check distance to bus_poly
    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    dists = np.array([
        bus_poly.distance(Point(cx + r_outer_um * math.cos(a), cy + r_outer_um * math.sin(a)))
        for a in angles
    ])
    # Angles where bus stays within gap_min + 22 nm of ring outer edge
    close_mask = dists <= (gap_min_um + 0.022)
    n_close = int(np.count_nonzero(close_mask))
    span_deg = n_close * (360.0 / 720.0)

    # For a straight bus tangent to radius R, g(theta) - g0 = R*(1/cos(theta) - 1) <= 0.022 um
    # gives 2*theta_straight_deg = 2 * degrees(acos(R / (R + 0.022))).
    theta_straight_deg = 2.0 * math.degrees(math.acos(r_outer_um / (r_outer_um + 0.022)))
    is_pulley = span_deg > max(theta_straight_deg * 1.45, 5.0)
    # Subtract slight corner transition overhang (~1.2 deg)
    wrap_deg = max(round(span_deg - 1.0, 2), 2.0) if is_pulley else 0.0
    return is_pulley, wrap_deg


def analyze_uploaded_gds(
    gds_path: str | Path,
    *,
    core_layer: tuple[int, int] | None = None,
    cell_name: str | None = None,
) -> dict:
    """Parse any `.gds` or `.oas` file and automatically extract MRR topology & dimensions."""
    p = Path(gds_path)
    if p.suffix.lower() == ".oas":
        lib = gdstk.read_oas(str(p))
    else:
        lib = gdstk.read_gds(str(p))

    if not lib.cells:
        raise ValueError("Uploaded GDS/OAS file contains no cells.")

    if cell_name:
        matches = [c for c in lib.cells if c.name == cell_name]
        if not matches:
            raise ValueError(f"Cell '{cell_name}' not found. Available: {[c.name for c in lib.cells]}")
        target_cell = matches[0]
    else:
        tops = lib.top_level()
        target_cell = tops[0] if tops else lib.cells[0]

    flat = target_cell.copy("._microring_flat_inspect")
    flat.flatten()
    scale_to_um = float(lib.unit) / 1e-6

    # Group polygons by (layer, datatype)
    by_layer: dict[tuple[int, int], list[np.ndarray]] = {}
    for poly in flat.polygons:
        key = (int(poly.layer), int(poly.datatype))
        by_layer.setdefault(key, []).append(np.asarray(poly.points, dtype=float) * scale_to_um)

    if not by_layer:
        raise ValueError(f"Cell '{target_cell.name}' contains no polygons.")

    layer_summary = [
        {"layer": k[0], "datatype": k[1], "polygon_count": len(v)}
        for k, v in sorted(by_layer.items())
    ]

    # Auto-pick core waveguide layer: explicit > (1,0) if present > layer with annular hole > largest count
    chosen_layer = core_layer
    if chosen_layer is None:
        if (1, 0) in by_layer:
            chosen_layer = (1, 0)
        else:
            for k, pts_list in sorted(by_layer.items()):
                merged_try = unary_union([Polygon(pts).buffer(0) for pts in pts_list if len(pts) >= 3])
                polys_try = [merged_try] if isinstance(merged_try, Polygon) else list(getattr(merged_try, "geoms", []))
                if any(len(g.interiors) >= 1 for g in polys_try):
                    chosen_layer = k
                    break
            if chosen_layer is None:
                chosen_layer = max(by_layer.items(), key=lambda kv: len(kv[1]))[0]

    if chosen_layer not in by_layer:
        raise ValueError(f"Requested core layer {chosen_layer} not present in GDS layers {list(by_layer)}")

    # Merge touching waveguide segments on core layer into topological components
    raw_shapely = [Polygon(pts).buffer(0.0005) for pts in by_layer[chosen_layer] if len(pts) >= 3]
    merged = unary_union(raw_shapely).buffer(-0.0005)
    geoms: list[Polygon] = (
        [merged] if isinstance(merged, Polygon)
        else list(merged.geoms) if isinstance(merged, MultiPolygon)
        else [g for g in getattr(merged, "geoms", []) if isinstance(g, Polygon)]
    )

    is_mzi_cell = "MZI" in target_cell.name.upper()
    annuli = [] if is_mzi_cell else [g for g in geoms if len(g.interiors) >= 1 and g.area > 1.0]
    buses = [g for g in geoms if (is_mzi_cell or len(g.interiors) == 0) and g.area > 0.5]

    if not annuli:
        # Non-resonant Photonic IC component: MZI, I/O coupler (Butt / Grating), or Coupler Cutout GDS
        viewer_layers = []
        for (lyr, dt), pts_list in sorted(by_layer.items()):
            polys_serialized = []
            for pts in pts_list[:160]:
                stride = max(1, len(pts) // 180)
                polys_serialized.append(np.round(pts[::stride], 3).tolist())
            viewer_layers.append({
                "layer": lyr,
                "datatype": dt,
                "is_core": (lyr, dt) == chosen_layer,
                "polygons": polys_serialized,
            })
        all_pts = np.vstack([pts for pts_list in by_layer.values() for pts in pts_list])
        bbox_um = [
            round(float(np.min(all_pts[:, 0])), 3),
            round(float(np.min(all_pts[:, 1])), 3),
            round(float(np.max(all_pts[:, 0])), 3),
            round(float(np.max(all_pts[:, 1])), 3),
        ]
        gap_markers = []
        gap_measured = 0.0
        w_bus_est = 0.50
        buses_sorted = sorted(buses, key=lambda g: -g.area)
        if buses_sorted:
            c0 = buses_sorted[0].centroid
            w_bus_est = _measure_local_bus_width(buses_sorted[0], c0)
        if len(buses_sorted) >= 2:
            d12 = float(buses_sorted[0].exterior.distance(buses_sorted[1].exterior))
            if d12 <= 5.0:
                p1, p2 = nearest_points(buses_sorted[0].exterior, buses_sorted[1].exterior)
                gap_measured = round(d12, 4)
                gap_markers.append({
                    "ring_pt": [round(p1.x, 4), round(p1.y, 4)],
                    "bus_pt": [round(p2.x, 4), round(p2.y, 4)],
                    "gap_um": gap_measured,
                })
        has_gc_etch = (40, 0) in by_layer
        if is_mzi_cell:
            disp_name = "Mach-Zehnder Interferometer (MZI)"
            top_code = "mzi_interferometer"
            n_ports = 2 if ("Y_BRANCH" in target_cell.name.upper() or "1X2" in target_cell.name.upper()) else 4
        elif has_gc_etch:
            disp_name = "Surface Grating Coupler (I/O)"
            top_code = "grating_coupler_io"
            n_ports = 2
        elif gap_measured > 0.0:
            disp_name = "Single-Pass Coupler Cutout"
            top_code = "coupler_cutout_4p"
            n_ports = 4
        else:
            disp_name = "Butt / Edge Taper Coupler (I/O)"
            top_code = "butt_coupler_io"
            n_ports = 2
        span_x = round(bbox_um[2] - bbox_um[0], 3)
        span_y = round(bbox_um[3] - bbox_um[1], 3)
        return {
            "file_name": p.name,
            "cell_name": target_cell.name,
            "available_cells": [c.name for c in lib.cells],
            "layers": layer_summary,
            "selected_core_layer": list(chosen_layer),
            "bbox_um": bbox_um,
            "extracted_geometry": {
                "is_non_ring": True,
                "topology": top_code,
                "topology_display": disp_name,
                "footprint_x_um": span_x,
                "footprint_y_um": span_y,
                "ring_radius_um": 0.0,
                "ring2_radius_um": 0.0,
                "ring_width_um": w_bus_est,
                "bus_width_um": w_bus_est,
                "gap_thru_um": gap_measured,
                "gap_drop_um": gap_measured,
                "inter_ring_gap_um": 0.0,
                "straight_length_um": span_x,
                "pulley_angle_thru_deg": 0.0,
                "ports_detected": n_ports,
            },
            "gap_markers": gap_markers,
            "viewer_layers": viewer_layers,
            "note": f"Non-resonant Photonic IC component ({disp_name}, footprint {span_x} x {span_y} um).",
        }

    # Order rings from top-to-bottom, left-to-right so Ring 1 is primary input-coupled cavity
    annuli.sort(key=lambda g: (-round(g.centroid.y, 2), round(g.centroid.x, 2)))
    ring1 = annuli[0]
    minx, miny, maxx, maxy = ring1.bounds
    h_minx, h_miny, h_maxx, h_maxy = ring1.interiors[0].bounds
    dx_out, dy_out = maxx - minx, maxy - miny
    dx_in, dy_in = h_maxx - h_minx, h_maxy - h_miny
    cx, cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)

    w_ring_um = round(0.25 * ((dx_out - dx_in) + (dy_out - dy_in)), 3)
    l_straight_um = round(abs(dx_out - dy_out), 3)
    is_racetrack = l_straight_um >= 0.40
    if is_racetrack:
        r1_um = round(0.5 * min(dx_out, dy_out) - 0.5 * w_ring_um, 3)
    else:
        l_straight_um = 0.0
        r1_um = round(0.25 * (dx_out + dy_out) - 0.5 * w_ring_um, 3)

    is_coupled_dual_ring = (
        len(annuli) >= 2 and float(ring1.exterior.distance(annuli[1].exterior)) <= 2.50
    )

    # Measure distances from Ring 1 (and Ring 2 if dual-ring) to buses
    bus_infos = []
    for bp in buses:
        d_r1 = float(ring1.exterior.distance(bp))
        d_min = d_r1
        coupled_ring_idx = 1
        if is_coupled_dual_ring:
            d_r2 = float(annuli[1].exterior.distance(bp))
            if d_r2 < d_r1:
                d_min = d_r2
                coupled_ring_idx = 2
        if d_min <= 2.50:
            p_ring, p_bus = nearest_points(
                (ring1 if coupled_ring_idx == 1 else annuli[1]).exterior, bp
            )
            wb = _measure_local_bus_width(bp, p_bus)
            bus_infos.append({
                "gap_um": round(d_min, 4),
                "bus_width_um": wb,
                "poly": bp,
                "ring_pt": [round(p_ring.x, 4), round(p_ring.y, 4)],
                "bus_pt": [round(p_bus.x, 4), round(p_bus.y, 4)],
            })
    bus_infos.sort(key=lambda d: d["gap_um"])

    gap_thru_um = bus_infos[0]["gap_um"] if bus_infos else 0.25
    w_bus_um = bus_infos[0]["bus_width_um"] if bus_infos else w_ring_um
    is_4port = len(bus_infos) >= 2
    gap_drop_um = bus_infos[1]["gap_um"] if is_4port else gap_thru_um

    r2_um = r1_um
    inter_gap_um = 0.35
    pulley_deg = 25.0

    if is_coupled_dual_ring:
        r2_poly = annuli[1]
        b2 = r2_poly.bounds
        r2_um = round(0.25 * ((b2[2] - b2[0]) + (b2[3] - b2[1])) - 0.5 * w_ring_um, 3)
        inter_gap_um = round(float(ring1.exterior.distance(r2_poly.exterior)), 4)
        detected_topology = "vernier_crow_ad"
    elif is_racetrack:
        detected_topology = "racetrack_ad" if is_4port else "racetrack_ap"
    else:
        is_pulley = False
        if bus_infos:
            is_pulley, pulley_deg = _detect_pulley_wrap_angle_deg(
                ring1.exterior, bus_infos[0]["poly"], cx, cy, gap_thru_um, r1_um + 0.5 * w_ring_um
            )
        if is_pulley:
            detected_topology = "pulley_ad" if is_4port else "pulley_ap"
        else:
            detected_topology = "point_circular_ad" if is_4port else "point_circular_ap"

    # Serialize decimated polygons for interactive 2D GDS canvas viewer
    viewer_layers = []
    for (lyr, dt), pts_list in sorted(by_layer.items()):
        polys_serialized = []
        for pts in pts_list[:120]:
            stride = max(1, len(pts) // 180)
            sub = pts[::stride]
            polys_serialized.append(np.round(sub, 3).tolist())
        viewer_layers.append({
            "layer": lyr,
            "datatype": dt,
            "is_core": (lyr, dt) == chosen_layer,
            "polygons": polys_serialized,
        })

    all_pts = np.vstack([pts for pts_list in by_layer.values() for pts in pts_list])
    bbox_um = [
        round(float(np.min(all_pts[:, 0])), 3),
        round(float(np.min(all_pts[:, 1])), 3),
        round(float(np.max(all_pts[:, 0])), 3),
        round(float(np.max(all_pts[:, 1])), 3),
    ]

    extracted = {
        "topology": detected_topology,
        "topology_display": TOPOLOGY_CATALOG[detected_topology]["name"],
        "ring_radius_um": r1_um,
        "ring2_radius_um": r2_um,
        "ring_width_um": w_ring_um,
        "bus_width_um": w_bus_um,
        "gap_thru_um": round(gap_thru_um, 4),
        "gap_drop_um": round(gap_drop_um, 4),
        "inter_ring_gap_um": round(inter_gap_um, 4),
        "straight_length_um": l_straight_um,
        "pulley_angle_thru_deg": pulley_deg,
        "ports_detected": 4 if is_4port else 2,
    }

    return {
        "file_name": p.name,
        "cell_name": target_cell.name,
        "available_cells": [c.name for c in lib.cells],
        "layers": layer_summary,
        "selected_core_layer": list(chosen_layer),
        "bbox_um": bbox_um,
        "extracted_geometry": extracted,
        "gap_markers": [
            {"ring_pt": b["ring_pt"], "bus_pt": b["bus_pt"], "gap_um": b["gap_um"]}
            for b in bus_infos[:2]
        ],
        "viewer_layers": viewer_layers,
        "note": (
            "In-plane XY dimensions (R, W_ring, W_bus, gap, L_s, pulley angle) were measured directly "
            "from GDS polygons. Please confirm vertical wafer stack (Core Material, H_film, H_slab, Sidewall) "
            "before running 2D eigenmode & CMT spectrum simulation."
        ),
    }
