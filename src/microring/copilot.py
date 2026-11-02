"""Guided Step-by-Step Conversational Photonics Copilot (`microring.copilot`).

Bilingual (`English` & `简体中文`) conversational assistant that guides users step-by-step
(one clear question at a time, or instant fast-forward when multiple specs are given):
  Step 1: Goal (New MRR Design / Extract Loss & Q from Lab .CSV / Analyze Uploaded .GDS)
  Step 2: Wavelength Band (`400 nm` Visible -> `1310 nm` -> `1550 nm` -> `14 um` Mid-IR)
  Step 3: Transparent Waveguide Core Material
  Step 4: Wafer Stack & Polarization -> 2D Full-Vector Eigenmode & Single-Mode Cutoff
  Step 5: Waveguide Propagation Loss (`dB/cm` or `Q_int`) -> Ring Topology & Target `FSR`
  Step 6: Critical Coupling Synthesis, Fabrication/Loss Sweeps & Foundry `.GDS` Export
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

import numpy as np

from .active_tuning import (
    export_lumerical_interconnect_bundle,
    run_active_tuning_analysis,
)
from .fab_tolerance import export_fab_bias_array_gds, run_fabrication_error_analysis
from .gds_bridge import (
    analyze_uploaded_gds,
    export_sweep_array_gds,
    write_klayout_layer_props,
)
from .io_couplers import (
    design_butt_edge_coupler,
    design_grating_coupler,
    export_io_coupler_gds,
    export_io_coupler_interconnect,
)
from .materials import C0_M_PER_S, MATERIAL_CATALOG
from .mrr_cmt_fit_predict import loss_db_per_cm_to_q_int, q_int_to_loss_db_per_cm
from .mrr_topologies import (
    MRRDesignConfig,
    TOPOLOGY_CATALOG,
    analyze_high_q_fdtd_convergence,
    export_coupler_cutout_gds,
    export_mrr_gds,
    simulate_mrr_topology,
)
from .mzi_interferometer import (
    MZI_SPLITTER_CATALOG,
    MZIDesignConfig,
    export_mzi_gds,
    export_mzi_interconnect_bundle,
    simulate_mzi,
)
from .sweeps import run_mrr_parameter_sweep, run_waveguide_width_sweep

# Foundry-standard starting presets per core material
PLATFORM_PRESETS: dict[str, dict] = {
    "Si": {
        "film_height_um": 0.22,
        "slab_height_um": 0.0,
        "sidewall_deg": 86.0,
        "ring_width_um": 0.50,
        "bus_width_um": 0.45,
        "ring_radius_um": 20.0,
        "gap_thru_um": 0.18,
        "gap_drop_um": 0.18,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.30, 0.85),
        "loss_choices": [0.8, 1.5, 2.2, 3.5],
    },
    "a-Si": {
        "film_height_um": 0.25,
        "slab_height_um": 0.0,
        "sidewall_deg": 85.0,
        "ring_width_um": 0.52,
        "bus_width_um": 0.46,
        "ring_radius_um": 25.0,
        "gap_thru_um": 0.20,
        "gap_drop_um": 0.20,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.32, 0.90),
        "loss_choices": [1.5, 3.0, 5.0, 8.0],
    },
    "Si3N4": {
        "film_height_um": 0.40,
        "slab_height_um": 0.0,
        "sidewall_deg": 85.0,
        "ring_width_um": 1.20,
        "bus_width_um": 1.00,
        "ring_radius_um": 50.0,
        "gap_thru_um": 0.25,
        "gap_drop_um": 0.25,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.60, 2.10),
        "loss_choices": [0.10, 0.35, 0.80, 1.50],
    },
    "SiRich_SiNx": {
        "film_height_um": 0.35,
        "slab_height_um": 0.0,
        "sidewall_deg": 84.0,
        "ring_width_um": 0.90,
        "bus_width_um": 0.75,
        "ring_radius_um": 35.0,
        "gap_thru_um": 0.22,
        "gap_drop_um": 0.22,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.45, 1.50),
        "loss_choices": [0.5, 1.2, 2.5, 4.0],
    },
    "4H-SiC": {
        "film_height_um": 0.60,
        "slab_height_um": 0.0,
        "sidewall_deg": 82.0,
        "ring_width_um": 1.10,
        "bus_width_um": 0.85,
        "ring_radius_um": 40.0,
        "gap_thru_um": 0.22,
        "gap_drop_um": 0.22,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.45, 1.70),
        "loss_choices": [0.25, 0.60, 1.20, 2.50],
    },
    "3C-SiC": {
        "film_height_um": 0.55,
        "slab_height_um": 0.0,
        "sidewall_deg": 82.0,
        "ring_width_um": 1.05,
        "bus_width_um": 0.85,
        "ring_radius_um": 40.0,
        "gap_thru_um": 0.22,
        "gap_drop_um": 0.22,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.45, 1.65),
        "loss_choices": [0.8, 1.5, 3.0, 5.0],
    },
    "AlN": {
        "film_height_um": 0.80,
        "slab_height_um": 0.20,
        "sidewall_deg": 78.0,
        "ring_width_um": 1.20,
        "bus_width_um": 1.00,
        "ring_radius_um": 60.0,
        "gap_thru_um": 0.25,
        "gap_drop_um": 0.25,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.60, 2.00),
        "loss_choices": [0.4, 0.8, 1.5, 3.0],
    },
    "ScAlN_20": {
        "film_height_um": 0.60,
        "slab_height_um": 0.15,
        "sidewall_deg": 76.0,
        "ring_width_um": 1.15,
        "bus_width_um": 0.95,
        "ring_radius_um": 55.0,
        "gap_thru_um": 0.24,
        "gap_drop_um": 0.24,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.55, 1.90),
        "loss_choices": [0.8, 1.8, 3.0, 5.0],
    },
    "LN_Xcut": {
        "topology": "racetrack_ap",
        "film_height_um": 0.60,
        "slab_height_um": 0.30,
        "sidewall_deg": 70.0,
        "ring_width_um": 1.40,
        "bus_width_um": 1.20,
        "ring_radius_um": 80.0,
        "straight_length_um": 30.0,
        "gap_thru_um": 0.30,
        "gap_drop_um": 0.30,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.75, 2.30),
        "loss_choices": [0.10, 0.25, 0.60, 1.20],
    },
    "LN_Zcut": {
        "film_height_um": 0.60,
        "slab_height_um": 0.30,
        "sidewall_deg": 70.0,
        "ring_width_um": 1.40,
        "bus_width_um": 1.20,
        "ring_radius_um": 80.0,
        "gap_thru_um": 0.28,
        "gap_drop_um": 0.28,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 1550.0,
        "w_sweep": (0.75, 2.30),
        "loss_choices": [0.10, 0.25, 0.60, 1.20],
    },
    "Ge": {
        "film_height_um": 1.40,
        "slab_height_um": 0.30,
        "sidewall_deg": 82.0,
        "ring_width_um": 2.10,
        "bus_width_um": 1.80,
        "ring_radius_um": 65.0,
        "gap_thru_um": 0.40,
        "gap_drop_um": 0.40,
        "clad_material": "CaF2",
        "top_clad_material": "Air",
        "wavelength_nm": 4200.0,
        "span_nm": 35.0,
        "w_sweep": (1.10, 3.20),
        "loss_choices": [1.0, 2.5, 4.0, 6.0],
    },
    "TiO2": {
        "film_height_um": 0.18,
        "slab_height_um": 0.0,
        "sidewall_deg": 85.0,
        "ring_width_um": 0.38,
        "bus_width_um": 0.32,
        "ring_radius_um": 20.0,
        "gap_thru_um": 0.14,
        "gap_drop_um": 0.14,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 633.0,
        "span_nm": 4.0,
        "w_sweep": (0.20, 0.65),
        "loss_choices": [0.8, 1.8, 3.5, 6.0],
    },
    "Ta2O5": {
        "film_height_um": 0.22,
        "slab_height_um": 0.0,
        "sidewall_deg": 85.0,
        "ring_width_um": 0.55,
        "bus_width_um": 0.48,
        "ring_radius_um": 30.0,
        "gap_thru_um": 0.18,
        "gap_drop_um": 0.18,
        "clad_material": "SiO2",
        "top_clad_material": "SiO2",
        "wavelength_nm": 780.0,
        "span_nm": 5.0,
        "w_sweep": (0.30, 0.95),
        "loss_choices": [0.25, 0.60, 1.20, 2.50],
    },
    "GaN": {
        "film_height_um": 0.30,
        "slab_height_um": 0.05,
        "sidewall_deg": 80.0,
        "ring_width_um": 0.52,
        "bus_width_um": 0.45,
        "ring_radius_um": 30.0,
        "gap_thru_um": 0.16,
        "gap_drop_um": 0.16,
        "clad_material": "Al2O3",
        "top_clad_material": "Air",
        "wavelength_nm": 780.0,
        "span_nm": 5.0,
        "w_sweep": (0.28, 0.90),
        "loss_choices": [1.0, 2.0, 4.0, 7.0],
    },
    "Diamond": {
        "film_height_um": 0.25,
        "slab_height_um": 0.0,
        "sidewall_deg": 85.0,
        "ring_width_um": 0.42,
        "bus_width_um": 0.36,
        "ring_radius_um": 25.0,
        "gap_thru_um": 0.15,
        "gap_drop_um": 0.15,
        "clad_material": "SiO2",
        "top_clad_material": "Air",
        "wavelength_nm": 737.0,
        "span_nm": 4.5,
        "w_sweep": (0.22, 0.78),
        "loss_choices": [0.5, 1.2, 2.5, 5.0],
    },
}


def scale_preset_for_wavelength(cfg_dict: dict, new_wl_nm: float, explicit_keys: set[str]) -> tuple[float, float]:
    """Scale waveguide cross-section, gap, cladding & span for Visible / 1310 / 1550 / SWIR / Mid-IR."""
    mat = cfg_dict.get("core_material", "Si3N4")
    preset_key = "ScAlN_20" if mat.startswith("ScAlN") else mat
    base = PLATFORM_PRESETS.get(preset_key, PLATFORM_PRESETS["Si3N4"])
    ref_wl = float(base.get("wavelength_nm", 1550.0))
    ratio = float(new_wl_nm) / max(ref_wl, 200.0)

    if abs(ratio - 1.0) > 0.06:
        if "film_height_um" not in explicit_keys:
            if mat == "Si" and 1200.0 <= new_wl_nm <= 1680.0:
                cfg_dict["film_height_um"] = 0.22
            else:
                cfg_dict["film_height_um"] = round(float(np.clip(base["film_height_um"] * (ratio ** 0.88), 0.10, 3.20)), 3)
            if base.get("slab_height_um", 0.0) > 0 and "slab_height_um" not in explicit_keys:
                cfg_dict["slab_height_um"] = round(cfg_dict["film_height_um"] * (base["slab_height_um"] / base["film_height_um"]), 3)
        if "ring_width_um" not in explicit_keys:
            cfg_dict["ring_width_um"] = round(float(np.clip(base["ring_width_um"] * (ratio ** 0.90), 0.24, 4.50)), 3)
        if "bus_width_um" not in explicit_keys:
            cfg_dict["bus_width_um"] = round(float(np.clip(base["bus_width_um"] * (ratio ** 0.90), 0.20, 4.00)), 3)
        if "gap_thru_um" not in explicit_keys:
            g_sc = round(float(np.clip(base["gap_thru_um"] * (ratio ** 0.72), 0.10, 1.20)), 3)
            cfg_dict["gap_thru_um"] = g_sc
            cfg_dict["gap_drop_um"] = g_sc
        if "ring_radius_um" not in explicit_keys:
            cfg_dict["ring_radius_um"] = round(float(np.clip(base["ring_radius_um"] * (ratio ** 0.65), 10.0, 200.0)), 1)

    if new_wl_nm > 3600.0:
        if "clad_material" not in explicit_keys and cfg_dict.get("clad_material") == "SiO2":
            cfg_dict["clad_material"] = "Si" if mat == "Ge" and new_wl_nm > 5500.0 else "CaF2"
        if "top_clad_material" not in explicit_keys and cfg_dict.get("top_clad_material") == "SiO2":
            cfg_dict["top_clad_material"] = "Air"

    if "span_nm" not in explicit_keys:
        est_span = 12.0 * ((new_wl_nm / 1550.0) ** 1.85)
        cfg_dict["span_nm"] = round(float(np.clip(est_span, 1.8, 90.0)), 2)

    w_lo, w_hi = base.get("w_sweep", (0.45, 1.80))
    return (
        round(max(0.16, w_lo * (ratio ** 0.90)), 3),
        round(min(6.00, w_hi * (ratio ** 0.90)), 3),
    )


def create_initial_session(lang: str = "en") -> dict:
    """Create a fresh Guided Conversational Photonics Copilot session (`en` or `zh`)."""
    zh = str(lang).lower().startswith("zh")
    cfg = asdict(MRRDesignConfig())
    welcome = (
        "👋 **欢迎使用 MicroRing Studio 微环光子设计助手！**\n\n"
        "请选择下方工作流，我将以**一问一答**方式逐步引导您完成微环设计（您也可以直接输入完整参数）："
        if zh
        else "👋 **Welcome to MicroRing Studio!**\n\n"
        "How would you like to get started? Pick a workflow below and I will guide you **step by step** "
        "(one question at a time), or type your full spec directly:"
    )
    return {
        "stage": "welcome",
        "lang": "zh" if zh else "en",
        "wizard_step": 1,
        "ui_view": "overview",
        "config": cfg,
        "confirmed": [],
        "target_fsr_ghz": None,
        "target_fsr_nm": None,
        "last_width_sweep": None,
        "last_mrr_metrics": None,
        "history": [{"role": "assistant", "content": welcome}],
    }


def get_welcome_chips(lang: str = "en") -> list[dict[str, str]]:
    """Clean Step-1 workflow choices + 1-click presets in English or Chinese."""
    if str(lang).lower().startswith("zh"):
        return [
            {"label": "✨ 开始逐步引导设计新微环", "prompt": "开始逐步设计微环"},
            {"label": "🔀 设计马赫-曾德尔调制器 MZI (DC / MMI / Y分支)", "prompt": "设计 220 nm SOI 2x2 MMI 马赫曾德尔推挽 PN 调制器"},
            {"label": "🔌 设计光栅耦合器 (GC) / 端面耦合器 (Butt)", "prompt": "设计 220 nm SOI 聚焦光栅耦合器 @ 1550 nm"},
            {"label": "🚀 超高 Q (Q > 10^6) 秒级收敛 vs FDTD 振铃诊断", "prompt": "分析超高 Q 氮化硅微环 FDTD 不收敛与振铃光谱"},
            {"label": "📈 从实测光谱提取损耗 (dB/cm) 与 Q 值", "prompt": "从实测光谱提取波导损耗"},
            {"label": "📐 上传已有 .GDS 版图提取并仿真", "prompt": "上传 GDS 版图", "ui_action": "open_gds_picker"},
            {"label": "⚡ 快速体验：400nm 氮化硅 @ 1550nm", "prompt": "400 nm Si3N4 条形波导, SiO2包层, TE0 @ 1550 nm"},
        ]
    return [
        {"label": "✨ Start Step-by-Step Ring Design", "prompt": "Start step-by-step microring design"},
        {"label": "🔀 Design MZI Modulator / Switch (DC / MMI / Y-Branch)", "prompt": "Design a 220 nm SOI 2x2 MMI push-pull PN MZI modulator at 1550 nm"},
        {"label": "🔌 Butt Coupler & 2D-FDFD Grating Coupler", "prompt": "Design a 220 nm SOI apodized focusing grating coupler at 1550 nm"},
        {"label": "🚀 High-Q (10^6+) Convergence vs Slow FDTD", "prompt": "Analyze High-Q Si3N4 microring FDTD non-convergence and ring-down"},
        {"label": "📈 Extract Loss (dB/cm) & Q from Spectrum", "prompt": "Extract waveguide loss from measured spectrum"},
        {"label": "📐 Upload & Analyze .GDS Layout", "prompt": "Upload GDS layout", "ui_action": "open_gds_picker"},
        {"label": "⚡ Quick Try: 400nm Si3N4 @ 1550nm", "prompt": "400 nm Si3N4 strip waveguide, SiO2 clad, TE0 at 1550 nm"},
    ]


def _get_waveband_chips(zh: bool = False) -> list[dict[str, str]]:
    if zh:
        return [
            {"label": "🌈 可见光 (532 nm)", "prompt": "波长 532 nm"},
            {"label": "⚛️ 量子/近红外 (780 nm)", "prompt": "波长 780 nm"},
            {"label": "📡 O波段数通 (1310 nm)", "prompt": "波长 1310 nm"},
            {"label": "🌐 C波段光通信 (1550 nm)", "prompt": "波长 1550 nm"},
            {"label": "🔦 短波红外 SWIR (2000 nm)", "prompt": "波长 2000 nm"},
            {"label": "🔥 中红外传感 (4200 nm)", "prompt": "波长 4200 nm"},
            {"label": "🌡️ 长波中红外 (7500 nm)", "prompt": "波长 7500 nm"},
        ]
    return [
        {"label": "🌈 Visible (532 nm)", "prompt": "Wavelength = 532 nm"},
        {"label": "⚛️ Quantum / Red-NIR (780 nm)", "prompt": "Wavelength = 780 nm"},
        {"label": "📡 Datacom O-Band (1310 nm)", "prompt": "Wavelength = 1310 nm"},
        {"label": "🌐 Telecom C-Band (1550 nm)", "prompt": "Wavelength = 1550 nm"},
        {"label": "🔦 SWIR (2000 nm)", "prompt": "Wavelength = 2000 nm"},
        {"label": "🔥 Mid-IR Sensing (4200 nm)", "prompt": "Wavelength = 4200 nm"},
        {"label": "🌡️ Long-Wave Mid-IR (7500 nm)", "prompt": "Wavelength = 7500 nm"},
    ]


def _get_material_chips_for_wavelength(wl_nm: float, zh: bool = False) -> list[dict[str, str]]:
    """Return only waveguide core platforms physically transparent at `wl_nm`."""
    if wl_nm < 1050.0:
        if zh:
            return [
                {"label": "Si3N4 氮化硅 (超低损耗)", "prompt": "材料 Si3N4"},
                {"label": "TiO2 二氧化钛 (可见光高折射率)", "prompt": "材料 TiO2"},
                {"label": "Ta2O5 五氧化二钽 (低损耗)", "prompt": "材料 Ta2O5"},
                {"label": "AlN 氮化铝 (压电/电光)", "prompt": "材料 AlN"},
                {"label": "4H-SiC 碳化硅 (量子/非线性)", "prompt": "材料 4H-SiC"},
                {"label": "Diamond 金刚石 (色心量子)", "prompt": "材料 Diamond"},
            ]
        return [
            {"label": "Si3N4 (Ultra-low loss Nitride)", "prompt": "Core material: Si3N4"},
            {"label": "TiO2 (High-index Visible)", "prompt": "Core material: TiO2"},
            {"label": "Ta2O5 (Low-loss Tantala)", "prompt": "Core material: Ta2O5"},
            {"label": "AlN (Piezo / Pockels)", "prompt": "Core material: AlN"},
            {"label": "4H-SiC (Quantum / Nonlinear)", "prompt": "Core material: 4H-SiC"},
            {"label": "Diamond (Color Centers)", "prompt": "Core material: Diamond"},
        ]
    if wl_nm > 3600.0:
        if wl_nm > 5500.0:
            return (
                [
                    {"label": "Ge-on-Si 硅基锗波导 (5–12 µm)", "prompt": "材料 Ge on Si"},
                    {"label": "Ge-on-CaF2 氟化钙基锗波导", "prompt": "材料 Ge on CaF2"},
                    {"label": "Diamond 金刚石宽谱波导", "prompt": "材料 Diamond"},
                ]
                if zh
                else [
                    {"label": "Ge on Si (5–12 µm LWIR)", "prompt": "Core material: Ge on Si"},
                    {"label": "Ge on CaF2 (3–8 µm Mid-IR)", "prompt": "Core material: Ge on CaF2"},
                    {"label": "Diamond (Broadband Mid-IR)", "prompt": "Core material: Diamond"},
                ]
            )
        return (
            [
                {"label": "Ge-on-CaF2 中红外锗波导", "prompt": "材料 Ge on CaF2"},
                {"label": "4H-SiC 碳化硅 (透光至 ~4.5 µm)", "prompt": "材料 4H-SiC"},
                {"label": "Si-on-CaF2 硅波导 (空气包层)", "prompt": "材料 Si on CaF2"},
                {"label": "AlN 氮化铝中红外波导", "prompt": "材料 AlN"},
                {"label": "Diamond 金刚石波导", "prompt": "材料 Diamond"},
            ]
            if zh
            else [
                {"label": "Ge on CaF2 (Mid-IR High-Index)", "prompt": "Core material: Ge on CaF2"},
                {"label": "4H-SiC (Transparent to ~4.5 µm)", "prompt": "Core material: 4H-SiC"},
                {"label": "Si on CaF2 / Air-clad", "prompt": "Core material: Si on CaF2"},
                {"label": "AlN (Mid-IR Piezo/EO)", "prompt": "Core material: AlN"},
                {"label": "Diamond (Broadband Mid-IR)", "prompt": "Core material: Diamond"},
            ]
        )
    if zh:
        return [
            {"label": "Si 绝缘体上硅 (220 nm SOI)", "prompt": "材料 SOI"},
            {"label": "Si3N4 氮化硅 (超低损耗)", "prompt": "材料 Si3N4"},
            {"label": "4H-SiC 碳化硅 (非线性/量子)", "prompt": "材料 4H-SiC"},
            {"label": "X-cut TFLN 薄膜铌酸锂 (高速电光)", "prompt": "材料 X-cut TFLN"},
            {"label": "AlN 氮化铝", "prompt": "材料 AlN"},
            {"label": "ScAlN 20% 氮化铝钪 (压电/电光)", "prompt": "材料 ScAlN 20%"},
        ]
    return [
        {"label": "Si (220 nm SOI Silicon)", "prompt": "Core material: SOI"},
        {"label": "Si3N4 (Low-loss Silicon Nitride)", "prompt": "Core material: Si3N4"},
        {"label": "4H-SiC (Nonlinear / Quantum)", "prompt": "Core material: 4H-SiC"},
        {"label": "X-cut TFLN (Electro-Optic LiNbO3)", "prompt": "Core material: X-cut TFLN"},
        {"label": "AlN (Aluminum Nitride)", "prompt": "Core material: AlN"},
        {"label": "ScAlN 20% (Piezo / Pockels)", "prompt": "Core material: ScAlN 20%"},
    ]


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _to_um(value: float, unit: str) -> float:
    u = unit.lower().strip()
    if u in ("nm", "纳米"):
        return round(value * 1e-3, 4)
    return round(value, 4)


def parse_photonics_intent(message: str) -> dict:
    """Extract guided step intents, material, wafer stack, propagation loss, Q_int, topology & actions."""
    text = message.strip()
    low = text.lower()
    out: dict = {
        "lang": "zh" if _has_cjk(text) else "en",
        "fields": {},
        "targets": {},
        "action": None,
        "sweep": None,
    }

    if re.search(r"\b(reset|start over|clear session)\b|重置|重新开始", low):
        out["action"] = "reset"
        return out

    if re.search(r"step[-_\s]*by[-_\s]*step|start\s+design|guided\s+design|逐步设计|逐步引导|开始设计|引导设计", low):
        out["action"] = "start_guided"
        return out

    if re.search(r"extract.*loss|fit.*spectrum|measured.*spectrum|lab.*spectrum|提取.*损耗|拟合.*光谱", low):
        out["action"] = "open_spectrum_fitter"
        return out

    if re.search(r"^upload\s+gds|i\s+want\s+to\s+upload.*gds|上传.*gds", low):
        out["action"] = "open_gds_uploader"
        return out

    # 1. Core Material Recognition
    mat = None
    if re.search(r"scaln|氮化铝钪", low):
        sc_match = re.search(r"(10|20|30)\s*%", low) or re.search(r"scaln[_\s]*(10|20|30)", low)
        pct = sc_match.group(1) if sc_match else "20"
        mat = f"ScAlN_{pct}"
        out["fields"]["sc_fraction"] = float(pct) / 100.0
    elif re.search(r"\b4h[-_\s]*sic\b|碳化硅|\bsic\b", low) and "3c" not in low:
        mat = "4H-SiC"
    elif re.search(r"\b3c[-_\s]*sic\b", low):
        mat = "3C-SiC"
    elif re.search(r"si[-_\s]*rich|srn|富硅氮化硅|sinx", low):
        mat = "SiRich_SiNx"
    elif re.search(r"\bsi3n4\b|\bsin\b|silicon nitride|氮化硅", low):
        mat = "Si3N4"
    elif re.search(r"\ba[-_\s]*si\b|amorphous silicon|非晶硅", low):
        mat = "a-Si"
    elif re.search(r"x[-_\s]*cut.*(ln|tfln|linbo3|niobate)|x切.*铌酸锂", low):
        mat = "LN_Xcut"
    elif re.search(r"z[-_\s]*cut.*(ln|tfln|linbo3|niobate)|z切.*铌酸锂", low):
        mat = "LN_Zcut"
    elif re.search(r"\b(tfln|linbo3|lithium niobate|ln)\b|铌酸锂", low):
        mat = "LN_Xcut"
    elif re.search(r"\btio2\b|titanium dioxide|二氧化钛", low):
        mat = "TiO2"
    elif re.search(r"\bta2o5\b|tantala|tantalum|五氧化二钽", low):
        mat = "Ta2O5"
    elif re.search(r"\bgan\b|gallium nitride|氮化镓", low):
        mat = "GaN"
    elif re.search(r"\bdiamond\b|金刚石", low):
        mat = "Diamond"
    elif re.search(r"\baln\b|aluminum nitride|aluminium nitride|氮化铝", low):
        mat = "AlN"
    elif re.search(r"\bgsst\b|phase[-_\s]*change|相变", low):
        mat = "GSST_crystalline" if ("cryst" in low or "晶态" in low) else "GSST_amorphous"
    elif re.search(r"\bgermanium\b|\bge\b|锗", low):
        mat = "Ge"
    elif re.search(r"\bsoi\b|\bc-si\b|\bsilicon\b|\bsi\b|硅波导|绝缘体上硅", low):
        mat = "Si"

    if mat:
        out["fields"]["core_material"] = mat

    # 2. Cladding & Substrate
    if re.search(r"air[-_\s]*(clad|top)|unclad|空气包层|无上包层", low):
        out["fields"]["top_clad_material"] = "Air"
    elif re.search(r"(sio2|oxide|silica)[-_\s]*(clad|top)|sio2\s*包层|二氧化硅包层|氧化物包层", low):
        out["fields"]["top_clad_material"] = "SiO2"
        out["fields"]["clad_material"] = "SiO2"
    if re.search(r"on\s+caf2|caf2|氟化钙", low):
        out["fields"]["clad_material"] = "CaF2"
    elif re.search(r"on\s+mgf2|mgf2|氟化镁", low):
        out["fields"]["clad_material"] = "MgF2"
    elif re.search(r"on\s+al2o3|sapphire|蓝宝石基底", low):
        out["fields"]["clad_material"] = "Al2O3"
    elif mat == "Ge" and re.search(r"on\s+si\b|ge[-_\s]*on[-_\s]*si|硅基底", low):
        out["fields"]["clad_material"] = "Si"

    # 3. Polarization
    if re.search(r"\btm0?\b|横磁", low):
        out["fields"]["polarization"] = "TM0"
    elif re.search(r"\bte0?\b|横电", low):
        out["fields"]["polarization"] = "TE0"

    # 4. Wavelength & Named Wavebands
    wl_m = (
        re.search(r"(?:wavelength|wl|lambda|@|\bat\b|波长)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)", low)
        or re.search(r"\b(405|450|488|532|633|637|736|737|750|780|850|980|1064|1270|1310|1330|1530|1550|1625|2000|2100|3300|3800|4200|4500|5200|7500|8000)(?:\.\d+)?\s*(nm)?\b", low)
    )
    if wl_m:
        val = float(wl_m.group(1))
        unit = wl_m.group(2) or "nm"
        wl_nm = val * 1000.0 if unit in ("um", "µm") else val
        if 350.0 <= wl_nm <= 14000.0:
            out["fields"]["wavelength_nm"] = round(wl_nm, 3)
    elif re.search(r"\bo[-_\s]*band\b|o波段", low):
        out["fields"]["wavelength_nm"] = 1310.0
    elif re.search(r"\bc[-_\s]*band\b|c波段", low):
        out["fields"]["wavelength_nm"] = 1550.0
    elif re.search(r"\blwir\b|长波红外", low):
        out["fields"]["wavelength_nm"] = 7500.0
    elif re.search(r"\bmid[-_\s]*ir\b|\bmir\b|中红外", low):
        out["fields"]["wavelength_nm"] = 4200.0
    elif re.search(r"\bswir\b|2[-_\s]*um\b|2[-_\s]*µm\b", low):
        out["fields"]["wavelength_nm"] = 2000.0
    elif re.search(r"\bvisible\b|可见光", low):
        out["fields"]["wavelength_nm"] = 532.0

    # 5. Film height H_film & Slab height H_slab & Sidewall angle
    h_explicit = re.search(
        r"(?:h_?film|film(?:\s*height|\s*thickness)?|thickness|core\s*height|膜厚|厚度)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)",
        low,
    )
    h_prefix = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(nm|um|µm)\s+(?:soi|si3n4|sin|tio2|ta2o5|gan|diamond|4h-sic|3c-sic|sic|aln|scaln|x-cut|z-cut|tfln|ln|ge|a-si|si\b)",
        low,
    )
    if h_explicit:
        out["fields"]["film_height_um"] = _to_um(float(h_explicit.group(1)), h_explicit.group(2))
    elif h_prefix:
        h_val = _to_um(float(h_prefix.group(1)), h_prefix.group(2))
        if 0.05 <= h_val <= 5.0:
            out["fields"]["film_height_um"] = h_val

    if re.search(r"strip|full[-_\s]*etch|slab\s*[=:]?\s*0\b|条形波导|全刻蚀", low):
        out["fields"]["slab_height_um"] = 0.0
    else:
        slab_m = (
            re.search(r"(?:h_?slab|slab(?:\s*height|\s*thickness)?|平板层|剩余厚度)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)", low)
            or re.search(r"(\d+(?:\.\d+)?)\s*(nm|um|µm)\s*(?:slab|平板)", low)
        )
        if slab_m:
            out["fields"]["slab_height_um"] = _to_um(float(slab_m.group(1)), slab_m.group(2))

    sw_m = (
        re.search(r"(?:sidewall|侧壁角?)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:deg|°|度)?", low)
        or re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|°|度)\s*sidewall", low)
    )
    if sw_m:
        ang = float(sw_m.group(1))
        if 55.0 <= ang <= 90.0:
            out["fields"]["sidewall_deg"] = ang

    # 6. Sweep Commands (passive geometry / loss sweeps)
    sweep_match = re.search(
        r"(?:sweep|scan|vary|扫描|扫参|扫)\s*(?:propagation\s*|传播)?(width|w\b|gap|loss|alpha|pulley|angle|theta|length|straight|l_s|radius|r\b|波导宽度|宽度|间距|损耗|包角|角度|直波导|耦合长度|半径)",
        low,
    )
    if sweep_match or re.search(r"single[-_\s]*mode\s*(?:check|sweep|cutoff|width)|单模(?:分析|截止|检查)", low):
        nums = [
            (float(m.group(1)), m.group(2))
            for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(nm|um|µm|deg|°|k|v|db)?", low[sweep_match.end():] if sweep_match else low)
        ]
        kind_tok = sweep_match.group(1) if sweep_match else "width"
        if kind_tok in ("width", "w", "波导宽度", "宽度"):
            w_pair = [(_to_um(v, u) if u in ("nm", "um", "µm") else v) for v, u in nums if v > 0]
            out["action"] = "sweep_width"
            if len(w_pair) >= 2:
                out["sweep"] = {"w_min_um": min(w_pair[0], w_pair[1]), "w_max_um": max(w_pair[0], w_pair[1])}
            return out

        param_map = {
            "gap": "gap_thru_um",
            "间距": "gap_thru_um",
            "loss": "alpha_loss_db_per_cm",
            "alpha": "alpha_loss_db_per_cm",
            "损耗": "alpha_loss_db_per_cm",
            "pulley": "pulley_angle_thru_deg",
            "angle": "pulley_angle_thru_deg",
            "theta": "pulley_angle_thru_deg",
            "包角": "pulley_angle_thru_deg",
            "角度": "pulley_angle_thru_deg",
            "length": "straight_length_um",
            "straight": "straight_length_um",
            "l_s": "straight_length_um",
            "直波导": "straight_length_um",
            "耦合长度": "straight_length_um",
            "radius": "ring_radius_um",
            "r": "ring_radius_um",
            "半径": "ring_radius_um",
            "temp": "delta_temp_k",
            "temperature": "delta_temp_k",
            "温度": "delta_temp_k",
            "voltage": "voltage_v",
            "bias": "voltage_v",
            "电压": "voltage_v",
        }
        p_name = param_map.get(kind_tok, "gap_thru_um")
        out["action"] = "sweep_mrr"
        vals_parsed = []
        for v, u in nums[:2]:
            if p_name.endswith("_um") and u == "nm":
                vals_parsed.append(v * 1e-3)
            elif p_name == "gap_thru_um" and u == "" and v > 10.0:
                vals_parsed.append(v * 1e-3)
            else:
                vals_parsed.append(v)
        out["sweep"] = {
            "param_name": p_name,
            "range": (vals_parsed[0], vals_parsed[1]) if len(vals_parsed) >= 2 else None,
        }
        return out

    # 7. Waveguide Widths, Radius, Gap, Pulley Angle, Straight Length, Propagation Loss (dB/cm) or Q_int, Tuning
    wr_m = re.search(r"(?:w_?ring|ring\s*width|环宽)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)", low)
    wb_m = re.search(r"(?:w_?bus|bus\s*width|总线宽)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)", low)
    wg_m = re.search(r"(?:\bwidth\b|波导宽)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)", low)
    if wr_m:
        out["fields"]["ring_width_um"] = _to_um(float(wr_m.group(1)), wr_m.group(2))
    if wb_m:
        out["fields"]["bus_width_um"] = _to_um(float(wb_m.group(1)), wb_m.group(2))
    if wg_m and not wr_m and not wb_m:
        w_u = _to_um(float(wg_m.group(1)), wg_m.group(2))
        out["fields"]["ring_width_um"] = w_u
        out["fields"]["bus_width_um"] = w_u

    rad_m = re.search(r"(?:\bradius\b|\br1?\s*=|半径)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(um|µm|nm)?", low)
    if rad_m:
        rv = float(rad_m.group(1))
        ru = rad_m.group(2) or "um"
        out["fields"]["ring_radius_um"] = _to_um(rv, ru)

    gap_m = re.search(r"(?:\bgap\b|间距)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)", low)
    if gap_m:
        gv = _to_um(float(gap_m.group(1)), gap_m.group(2))
        out["fields"]["gap_thru_um"] = gv
        out["fields"]["gap_drop_um"] = gv

    ang_m = re.search(r"(?:pulley\s*angle|wrap\s*angle|\btheta\b|包角)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:deg|°|度)?", low)
    if ang_m:
        out["fields"]["pulley_angle_thru_deg"] = float(ang_m.group(1))

    ls_m = re.search(r"(?:straight(?:\s*length)?|\bl_?s\b|直波导|耦合长度)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(um|µm|nm)", low)
    if ls_m:
        out["fields"]["straight_length_um"] = _to_um(float(ls_m.group(1)), ls_m.group(2))

    # Propagation loss in dB/cm
    loss_m = (
        re.search(r"(?:propagation\s*loss|loss|\balpha\b|传播损耗|损耗)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:db\s*/\s*cm|db/cm)?", low)
        or re.search(r"\b(\d+(?:\.\d+)?)\s*db\s*/\s*cm\b", low)
    )
    if loss_m:
        l_val = float(loss_m.group(1))
        if 0.001 <= l_val <= 200.0:
            out["fields"]["alpha_loss_db_per_cm"] = round(l_val, 4)
    elif re.search(r"default.*loss|keep.*loss|默认损耗", low):
        out["targets"]["confirm_default_loss"] = True

    # Intrinsic Q input
    qint_m = re.search(r"(?:q_?int|intrinsic\s*q|本征\s*q)\s*[=:≈~]?\s*(\d+(?:\.\d+)?(?:e\d+)?)\s*([km])?\b", low)
    if qint_m:
        qv = float(qint_m.group(1))
        suf = (qint_m.group(2) or "").lower()
        if suf == "k":
            qv *= 1e3
        elif suf == "m":
            qv *= 1e6
        if qv >= 500.0:
            out["targets"]["q_int_target"] = qv

    dt_m = re.search(r"(?:delta_?t|\bdt\b|temp(?:erature)?|升温|温升)\s*[=:+]?\s*(-?\d+(?:\.\d+)?)\s*(?:k|°c|c\b)", low)
    if dt_m:
        out["fields"]["delta_temp_k"] = float(dt_m.group(1))

    volt_m = re.search(r"(?:voltage|bias|电压)\s*[=:]?\s*(-?\d+(?:\.\d+)?)\s*v\b", low)
    if volt_m:
        out["fields"]["voltage_v"] = float(volt_m.group(1))

    # 8. Target FSR
    fsr_m = re.search(r"\bfsr\b\s*(?:of|=|:|约|~)?\s*(\d+(?:\.\d+)?)\s*(ghz|thz|nm)", low)
    if fsr_m:
        f_val, f_unit = float(fsr_m.group(1)), fsr_m.group(2)
        if f_unit == "thz":
            out["targets"]["fsr_ghz"] = f_val * 1000.0
        elif f_unit == "ghz":
            out["targets"]["fsr_ghz"] = f_val
        elif f_unit == "nm":
            out["targets"]["fsr_nm"] = f_val

    # 9. Topology Selection
    is_ad = bool(re.search(r"add[-_\s]*drop|4[-_\s]*port|四端口|下载", low))
    is_ap = bool(re.search(r"all[-_\s]*pass|2[-_\s]*port|两端口|全通", low))
    if re.search(r"vernier|crow|dual[-_\s]*ring|双环|游标", low):
        out["fields"]["topology"] = "vernier_crow_ad"
    elif re.search(r"pulley|wrap[-_\s]*around|滑轮", low):
        out["fields"]["topology"] = "pulley_ad" if is_ad else "pulley_ap"
    elif re.search(r"racetrack|跑道", low):
        out["fields"]["topology"] = "racetrack_ad" if is_ad else "racetrack_ap"
    elif re.search(r"point|circular|圆环|点耦合", low):
        out["fields"]["topology"] = "point_circular_ad" if is_ad else "point_circular_ap"
    elif is_ad or is_ap:
        out["targets"]["ports"] = 4 if is_ad else 2

    # 10. Direct Action Triggers
    if re.search(
        r"\bmzi\b|\bmzm\b|mach[-_\s]*zehnder|\bramzi\b|mmi[-_\s]*(?:splitter|coupler|2x2|1x2)|"
        r"y[-_\s]*(?:branch|junction|splitter)|马赫[-_\s]*曾德|马赫曾德|多模干涉|\by\s*分支",
        low,
    ):
        sp_det = "directional_coupler"
        if re.search(r"1\s*x\s*2\s*mmi|mmi[-_\s]*1x2|1x2\s*多模", low):
            sp_det = "mmi_1x2"
        elif re.search(r"\bmmi\b|2\s*x\s*2\s*mmi|多模干涉", low):
            sp_det = "mmi_2x2"
        elif re.search(r"y[-_\s]*(?:branch|junction|splitter)|\by\s*分支", low):
            sp_det = "y_branch"
        elif re.search(r"\bdc\b|directional[-_\s]*coupler|定向耦合", low):
            sp_det = "directional_coupler"

        mech_mzi = "auto"
        if re.search(r"\bpin\b|injection|注入", low):
            mech_mzi = "si_pin_injection"
        elif re.search(r"\bpn\b|depletion|耗尽", low):
            mech_mzi = "si_pn_depletion"
        elif re.search(r"\bpockels\b|electro[-_\s]*optic|\beo\b|tfln|linbo3|铌酸锂|电光", low):
            mech_mzi = "pockels_eo"
        elif re.search(r"thermo[-_\s]*optic|heater|\btin\b|热光|加热", low):
            mech_mzi = "thermo_optic"

        drv_mzi = "auto"
        if re.search(r"power|\bmw\b|功率", low):
            drv_mzi = "power_mw"
        elif re.search(r"current|\bma\b|电流", low):
            drv_mzi = "current_ma"
        elif re.search(r"delta_?t|temperature|温升|温度", low) and not re.search(r"heater|加热|\bmw\b|\bv\b", low):
            drv_mzi = "delta_temp_k"
        elif re.search(r"v\s*/\s*(?:um|µm)|e[-_\s]*field|电场", low):
            drv_mzi = "e_field_v_per_um"
        elif re.search(r"voltage|volt|\bv\b|电压", low):
            drv_mzi = "voltage_v"

        scheme_mzi = "auto"
        if re.search(r"single[-_\s]*arm|单臂", low):
            scheme_mzi = "single_arm"
        elif re.search(r"push[-_\s]*pull|推挽", low):
            scheme_mzi = "push_pull"

        larm_m = re.search(r"(?:arm[-_\s]*length|\bl_?arm\b|\bl1\b|臂长)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(mm|um|µm)", low)
        larm_um = None
        if larm_m:
            lv, lu = float(larm_m.group(1)), larm_m.group(2)
            larm_um = lv * 1000.0 if lu == "mm" else lv
        dl_m = re.search(r"(?:delta[-_\s]*l(?:ength)?|\bdl\b|Δl|臂长差|光程差)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(um|µm|nm)", low)
        dl_um = _to_um(float(dl_m.group(1)), dl_m.group(2)) if dl_m else None

        out["action"] = "design_mzi"
        out["mzi"] = {
            "splitter_type": sp_det,
            "arm1_length_um": larm_um,
            "delta_length_um": dl_um,
            "ramzi_enabled": bool(re.search(r"\bramzi\b|ring[-_\s]*assisted|微环辅助", low)),
            "mechanism": mech_mzi,
            "drive_type": drv_mzi,
            "drive_scheme": scheme_mzi,
            "include_active_tuning": not bool(re.search(r"passive[-_\s]*only|仅无源", low)),
            "export_gds": bool(re.search(r"\bgds\b|layout|mask|版图", low)),
            "export_interconnect": bool(re.search(r"\binterconnect\b|spar|compact[-_\s]*model", low)),
        }
    elif re.search(
        r"butt[-_\s]*coupl|edge[-_\s]*coupl|spot[-_\s]*size[-_\s]*convert|inverse[-_\s]*taper|lensed[-_\s]*fiber|"
        r"dfb[-_\s]*laser|grating[-_\s]*coupl|\bgc\s*(?:design|coupl|gds|loop|interconnect)|focusing[-_\s]*grating|"
        r"apodized[-_\s]*grating|fiber[-_\s]*array[-_\s]*loop|端面耦合|边缘耦合|模斑转换|倒锥|光栅耦合|聚焦光栅|光纤阵列回环",
        low,
    ):
        is_butt = bool(
            re.search(
                r"butt[-_\s]*coupl|edge[-_\s]*coupl|spot[-_\s]*size|inverse[-_\s]*taper|lensed[-_\s]*fiber|dfb[-_\s]*laser|端面耦合|边缘耦合|模斑转换|倒锥",
                low,
            )
            and not re.search(r"grating|光栅", low)
        )
        fib_p = "Lensed_2p5um" if is_butt else "SMF28_1550"
        if re.search(r"dfb|laser|激光器", low):
            fib_p = "DFB_Laser_Butt"
        elif re.search(r"uhna", low):
            fib_p = "UHNA_3"
        elif re.search(r"4(?:\.0)?\s*(?:um|µm)\s*lensed", low):
            fib_p = "Lensed_4p0um"
        elif re.search(r"2\.5\s*(?:um|µm)|lensed|透镜光纤", low) and is_butt:
            fib_p = "Lensed_2p5um"
        elif re.search(r"smf[-_\s]*28|10\.4\s*(?:um|µm)|标准单模光纤", low):
            fib_p = "SMF28_1550"

        etch_m = re.search(r"etch[-_\s]*(?:depth)?\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)", low)
        tip_m = re.search(r"(?:w_?tip|tip[-_\s]*width|尖端宽)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(nm|um|µm)", low)
        f_ang_m = re.search(r"(?:fiber[-_\s]*angle|tilt|光纤倾角|入射角)\s*[=:]?\s*(\d+(?:\.\d+)?)", low)
        out["action"] = "io_coupler"
        out["io"] = {
            "coupler_type": "butt_edge" if is_butt else "grating_coupler",
            "fiber_preset": fib_p,
            "apodized": not bool(re.search(r"\buniform\b|均匀光栅", low)),
            "focusing": not bool(re.search(r"\blinear\s+grating\b|直齿光栅", low)),
            "etch_depth_um": _to_um(float(etch_m.group(1)), etch_m.group(2)) if etch_m else None,
            "w_tip_um": _to_um(float(tip_m.group(1)), tip_m.group(2)) if tip_m else None,
            "fiber_angle_deg": float(f_ang_m.group(1)) if f_ang_m else 10.0,
            "export_gds": bool(re.search(r"\bgds\b|layout|mask|loopback|版图|回环", low)),
            "include_loopback": bool(re.search(r"loopback|fiber[-_\s]*array|127|回环|测试环", low)),
            "export_interconnect": bool(re.search(r"interconnect|spar|compact[-_\s]*model", low)),
        }
    elif re.search(
        r"high[-_\s]*q|ultra[-_\s]*high[-_\s]*q|ring[-_\s]*down|auto[-_\s]*shutoff|shutoff|"
        r"(?:fdtd|lumerical|mode).*(?:slow|converg)|(?:slow|not\s+converg|cannot\s+converg|non[-_\s]*converg).*(?:fdtd|mrr|ring|q\b)|"
        r"coupler[-_\s]*cutout|高\s*q|超高\s*q|不收敛|收敛慢|fdtd.*慢|振铃|耦合区切片",
        low,
    ):
        tsim_m = re.search(r"(\d+(?:\.\d+)?)\s*(ps|ns)\b", low)
        tsim_ps = 50.0
        if tsim_m:
            tsim_ps = float(tsim_m.group(1)) * (1000.0 if tsim_m.group(2) == "ns" else 1.0)
        out["action"] = "high_q_fdtd"
        out["high_q"] = {
            "fdtd_sim_time_ps": tsim_ps,
            "export_cutout_gds": bool(re.search(r"cutout|\bgds\b|lsf|切片|导出", low)),
            "ultra_low_loss": bool(re.search(r"ultra|10\^?6|10\^?7|million|超高", low)),
        }
    elif re.search(r"critical[-_\s]*coupl|apply\s+cc|临界耦合", low):
        out["action"] = "apply_critical_coupling"
    elif re.search(r"(?:sweep|array).*gds|gds.*(?:sweep|array)|阵列.*(?:gds|版图)", low):
        out["action"] = "export_sweep_gds"
    elif re.search(
        r"(?:pre[-_\s]*bias|opc|dose[-_\s]*(?:bracket|comp)).*(?:gds|array|mask)|(?:预偏置|补偿阵列|剂量阵列).*(?:gds|版图)",
        low,
    ):
        out["action"] = "export_fab_bias_gds"
    elif re.search(
        r"over[-_\s]*dos|under[-_\s]*dos|positive[-_\s]*(?:pr\b|resist)|negative[-_\s]*(?:pr\b|resist)|nagetive[-_\s]*(?:pr\b|resist)|\bhsq\b|\bzep520|\bcsar\b|fab(?:rication)?[-_\s]*(?:error|tol|analys|yield)|dose[-_\s]*(?:error|tol|sweep)|cd[-_\s]*(?:bias|error|tol)|过曝|欠曝|正胶|负胶|正性光刻胶|负性光刻胶|曝光剂量|工艺误差|工艺容差|线宽偏置|线宽误差",
        low,
    ):
        out["action"] = "fab_tolerance"
        resist_tone = "positive_pr"
        if re.search(r"negative|nagetive|\bhsq\b|\bma-?n\b|\bsu-?8\b|lift[-_\s]*off|负胶|负性|剥离", low):
            resist_tone = "negative_pr"
        elif re.search(r"direct[-_\s]*cd|cd[-_\s]*(?:bias|sweep)|线宽偏置", low) and not re.search(r"positive|negative|nagetive|正胶|负胶|dose|剂量|曝", low):
            resist_tone = "direct_cd"
        pct_m = re.search(r"(\d+(?:\.\d+)?)\s*%", low)
        nm_m = re.search(r"(?:cd|bias|±|\+/-)\s*(\d+(?:\.\d+)?)\s*nm", low)
        out["fab"] = {
            "resist_tone": resist_tone,
            "max_dose_error_pct": float(pct_m.group(1)) if pct_m else 15.0,
            "max_cd_bias_nm": float(nm_m.group(1)) if nm_m else 30.0,
        }
    elif re.search(
        r"\binterconnect\b|compact[-_\s]*model|sparams?\b|s[-_\s]*param|\.icp\b|导出\s*interconnect|"
        r"tuning[-_\s]*(?:eff|spec|slope|curve|analys)|active[-_\s]*tuning|\bpn[-_\s]*junction\b|\bsi[-_\s]*pn\b|"
        r"\bpin[-_\s]*(?:diode|junction|inject)|\bpockels\b|thermo[-_\s]*optic|micro[-_\s]*heater|\btin\s*heater\b|"
        r"e[-_\s]*field|electric[-_\s]*field|"
        r"(?:different|multi(?:ple)?|vary|sweep|scan)\s*(?:voltage|bias|current|temp(?:erature)?|power|e[-_\s]*field)|"
        r"(?:voltage|current|temperature|heater|eo|to)\s*(?:spectra|spectrum|tuning|sweep)|"
        r"\bv_?pi\b|\bp_?pi\b|调谐效率|调谐光谱|有源调谐|不同电压|不同电流|不同温度|多电压|多电流|温度光谱|电场|pn\s*结|pin\s*结|热光调谐|电光调谐|微加热器|扫描电压|扫描温度|扫描电流",
        low,
    ):
        want_ic = bool(
            re.search(r"\binterconnect\b|compact[-_\s]*model|sparams?\b|s[-_\s]*param|\.icp\b|导出\s*interconnect", low)
        )
        out["action"] = "export_interconnect" if want_ic else "active_tuning"
        out["fields"].pop("delta_temp_k", None)
        out["fields"].pop("voltage_v", None)

        mech_det = "auto"
        if re.search(r"\bpin\b|injection|注入", low):
            mech_det = "si_pin_injection"
        elif re.search(r"\bpn\b|depletion|耗尽", low):
            mech_det = "si_pn_depletion"
        elif re.search(r"\bpockels\b|electro[-_\s]*optic|\beo\b|e[-_\s]*field|electric[-_\s]*field|电光|电场", low):
            mech_det = "pockels_eo"
        elif re.search(r"thermo[-_\s]*optic|heater|\btin\b|热光|加热|温升|温度", low):
            mech_det = "thermo_optic"

        drv_det = "auto"
        if re.search(r"v\s*/\s*(?:um|µm)|e[-_\s]*field|electric[-_\s]*field|电场", low):
            drv_det = "e_field_v_per_um"
        elif re.search(r"current|\bma\b|电流", low):
            drv_det = "current_ma"
        elif re.search(r"power|\bmw\b|功率", low):
            drv_det = "power_mw"
        elif re.search(r"delta_?t|\bdt\b|temperature|\btemp\b|温度|温升", low) and not re.search(r"heater|加热|\bma\b|\bmw\b|\bv\b", low):
            drv_det = "delta_temp_k"
        elif re.search(r"voltage|volt|\bv\b|电压", low):
            drv_det = "voltage_v"

        # Parse optional explicit comma-separated bias list (e.g. "0, -2, -5 V" or "0, 20, 50 K")
        bias_list: list[float] | None = None
        list_m = re.search(
            r"(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){1,8})\s*(?:v\s*/\s*(?:um|µm)|ma|mw|k|°c|v\b)?",
            low,
        )
        if list_m:
            bias_list = [float(x.strip()) for x in list_m.group(1).split(",") if x.strip()]

        box_m = re.search(r"\bbox\b\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:um|µm)", low)
        hgap_m = re.search(r"heater[-_\s]*gap\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:um|µm)", low)
        egap_m = re.search(r"(?:eo|electrode)[-_\s]*gap\s*[=:]?\s*(\d+(?:\.\d+)?)\s*(?:um|µm)", low)

        out["tuning"] = {
            "mechanism": mech_det,
            "drive_type": drv_det,
            "bias_values": bias_list,
            "export_interconnect": want_ic,
            "box_thickness_um": float(box_m.group(1)) if box_m else 2.0,
            "heater_gap_um": float(hgap_m.group(1)) if hgap_m else 0.90,
            "electrode_gap_um": float(egap_m.group(1)) if egap_m else None,
        }
    elif re.search(r"(?:export|create|save|generate|download)\b.*(?:\.?\bgds\b|layout|mask)|(?:生成|导出|创建).*(?:gds|版图)", low):
        out["action"] = "export_single_gds"
    elif re.search(r"\b(simulate|calculate|compute|run\s+mrr|evaluate)\b|开始仿真|运行仿真|计算光谱|确认并仿真", low):
        out["action"] = "simulate_mrr"
    elif re.search(r"confirm\s+stack|check\s+mode|solve\s+2d\s+modes|确认叠层|求解模式", low):
        out["action"] = "sweep_width"

    return out


def _dict_to_cfg(cfg_dict: dict) -> MRRDesignConfig:
    valid = {f.name for f in fields(MRRDesignConfig)}
    return MRRDesignConfig(**{k: v for k, v in cfg_dict.items() if k in valid})


def _find_cli_binary(name: str) -> str | None:
    """Locate an installed CLI binary (`claude`, `codex`) even inside macOS .app / Windows .exe PATH."""
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    candidates = [
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
        home / ".local" / "bin" / name,
        home / ".npm-global" / "bin" / name,
    ]
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "npm" / f"{name}.cmd")
            candidates.append(Path(appdata) / "npm" / f"{name}.exe")
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def probe_ai_backends() -> dict:
    """Probe available Offline & Online (`Claude` / `Codex` / `Ollama`) AI backends locally."""
    ollama_ok = False
    for url in ("http://127.0.0.1:11434/api/tags", "http://127.0.0.1:1234/v1/models"):
        try:
            with urllib.request.urlopen(url, timeout=0.45) as resp:
                if resp.status == 200:
                    ollama_ok = True
                    break
        except Exception:
            pass
    claude_cli = _find_cli_binary("claude")
    codex_cli = _find_cli_binary("codex")
    return {
        "offline": True,
        "ollama_available": ollama_ok,
        "claude_cli_available": bool(claude_cli),
        "claude_env_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "codex_cli_available": bool(codex_cli),
        "codex_env_key": bool(os.environ.get("OPENAI_API_KEY")),
    }


def _maybe_ollama_commentary(compact_kpi: str, zh: bool = False) -> str | None:
    """Zero-cloud-token local LLM tip via Ollama (`127.0.0.1:11434`) or LM Studio MLX (`127.0.0.1:1234`)."""
    prompt = (
        f"请用1~2句简短中文回答或给出微环流片/设计建议：{compact_kpi}"
        if zh
        else f"Give a concise 1-2 sentence practical photonics design/fabrication insight for: {compact_kpi}"
    )
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=0.8) as tag_resp:
            models = [m.get("name", "") for m in json.loads(tag_resp.read().decode("utf-8")).get("models", [])]
        chat_models = [m for m in models if not any(eb in m.lower() for eb in ("bge", "embed", "nomic"))]
        chosen = next((m for m in chat_models if "14b" in m or "8b" in m), chat_models[0] if chat_models else "qwen3:14b")
        body = json.dumps({
            "model": chosen,
            "prompt": prompt,
            "think": False,
            "stream": False,
            "options": {"num_predict": 90, "temperature": 0.2},
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            tip = (json.loads(resp.read().decode("utf-8")).get("response") or "").strip()
            tip = re.sub(r"<think>.*?</think>", "", tip, flags=re.DOTALL).strip()
            if tip:
                return f"🖥 **Local LLM (`{chosen}`)**: {tip}"
    except Exception:
        pass

    try:
        lm_body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 90,
            "temperature": 0.2,
        }).encode("utf-8")
        req2 = urllib.request.Request(
            "http://127.0.0.1:1234/v1/chat/completions",
            data=lm_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=3.5) as resp2:
            d2 = json.loads(resp2.read().decode("utf-8"))
            tip2 = d2["choices"][0]["message"]["content"].strip()
            tip2 = re.sub(r"<think>.*?</think>", "", tip2, flags=re.DOTALL).strip()
            if tip2:
                return f"🖥 **Local MLX**: {tip2}"
    except Exception:
        pass
    return None


def query_ai_copilot(
    *,
    ai_mode: str,
    user_message: str,
    compact_state: str,
    zh: bool = False,
    api_key: str | None = None,
) -> str | None:
    """Query local (`ollama`) or online (`claude` / `codex` via CLI or API) using <150 tokens of context."""
    mode = (ai_mode or "offline").lower().strip()
    if mode == "offline":
        return None
    if mode == "ollama":
        return _maybe_ollama_commentary(f"User='{user_message[:120]}' | State: {compact_state}", zh=zh)

    sys_rule = (
        "你是集成光子微环芯片专家。本地2D矢量有限元与耦合模求解器已算出精确数值，请针对用户输入和当前器件状态，用2~3句简练专业中文回答问题或给出关键设计/流片建议（勿重复罗列全部数字）。"
        if zh
        else "You are an integrated photonics microring expert. Exact 2D Maxwell FEM & CMT physics already ran locally. Answer the user's message or give 2-3 concise sentences of actionable physics/tape-out guidance grounded in this state (do not repeat the whole table)."
    )
    compact_prompt = f"{sys_rule}\nUser: {user_message[:240]}\nSolved State: {compact_state}"

    if mode == "claude":
        eff_key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if eff_key:
            try:
                body = json.dumps({
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 200,
                    "temperature": 0.2,
                    "system": sys_rule,
                    "messages": [{"role": "user", "content": f"User: {user_message[:240]}\nSolved State: {compact_state}"}],
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": eff_key,
                        "anthropic-version": "2023-06-01",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=14.0) as resp:
                    blocks = json.loads(resp.read().decode("utf-8")).get("content", [])
                    txt = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                    if txt:
                        return f"🟣 **Claude**: {txt}"
            except Exception as exc:
                return f"🟣 **Claude API Notice**: `{exc}`"

        claude_bin = _find_cli_binary("claude")
        if claude_bin:
            try:
                env = os.environ.copy()
                env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
                proc = subprocess.run(
                    [claude_bin, "-p", compact_prompt],
                    capture_output=True,
                    text=True,
                    timeout=18.0,
                    env=env,
                )
                out = (proc.stdout or "").strip()
                if proc.returncode == 0 and out:
                    return f"🟣 **Claude Code**: {out}"
            except Exception:
                pass
        return (
            "🟣 *提示：如需启用在线 Claude 问答，请点击上方 `🔑 Key` 填入 Anthropic API Key，或在电脑终端登录 `claude` CLI。*"
            if zh
            else "🟣 *Tip: To enable Online Claude insights, click `🔑 Key` to enter an Anthropic API key (kept in memory only) or sign in to `claude` CLI locally.*"
        )

    if mode == "codex":
        eff_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        if eff_key:
            try:
                body = json.dumps({
                    "model": "gpt-4o-mini",
                    "max_tokens": 200,
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": sys_rule},
                        {"role": "user", "content": f"User: {user_message[:240]}\nSolved State: {compact_state}"},
                    ],
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {eff_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=14.0) as resp:
                    choices = json.loads(resp.read().decode("utf-8")).get("choices", [])
                    if choices:
                        txt = (choices[0].get("message", {}).get("content") or "").strip()
                        if txt:
                            return f"🟢 **Codex / OpenAI**: {txt}"
            except Exception as exc:
                return f"🟢 **OpenAI API Notice**: `{exc}`"

        codex_bin = _find_cli_binary("codex")
        if codex_bin:
            try:
                env = os.environ.copy()
                env["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
                proc = subprocess.run(
                    [codex_bin, "exec", "--skip-git-repo-check", compact_prompt],
                    capture_output=True,
                    text=True,
                    timeout=22.0,
                    env=env,
                )
                out = (proc.stdout or "").strip()
                if proc.returncode == 0 and out:
                    lines = [ln for ln in out.splitlines() if not ln.startswith(("[", "OpenAI Codex", "--------"))]
                    cleaned = "\n".join(lines[-6:]).strip() or out[-400:].strip()
                    return f"🟢 **Codex CLI**: {cleaned}"
            except Exception:
                pass
        return (
            "🟢 *提示：如需启用在线 Codex / OpenAI 问答，请点击上方 `🔑 Key` 填入 OpenAI API Key，或在电脑终端登录 `codex` CLI。*"
            if zh
            else "🟢 *Tip: To enable Online Codex / OpenAI insights, click `🔑 Key` to enter an OpenAI API key (kept in memory only) or sign in to `codex` CLI locally.*"
        )

    return None


def inject_uploaded_gds_into_session(session: dict, gds_info: dict, lang: str = "en") -> dict:
    """Update Copilot session with polygon-extracted dimensions from an uploaded `.gds`/`.oas`."""
    zh = str(lang or session.get("lang", "en")).lower().startswith("zh")
    ext = gds_info["extracted_geometry"]
    cfg = dict(session["config"])
    for key in (
        "topology",
        "ring_radius_um",
        "ring2_radius_um",
        "ring_width_um",
        "bus_width_um",
        "gap_thru_um",
        "gap_drop_um",
        "inter_ring_gap_um",
        "straight_length_um",
        "pulley_angle_thru_deg",
    ):
        if key in ext and ext[key] is not None:
            cfg[key] = ext[key]

    session["config"] = cfg
    session["stage"] = "gds_uploaded"
    session["wizard_step"] = 4
    session["ui_view"] = "gds"
    confirmed = set(session.get("confirmed", []))
    confirmed.update(["topology", "resonator_size", "uploaded_gds"])
    session["confirmed"] = sorted(confirmed)

    if zh:
        msg = (
            f"📐 **已成功解析版图 `{gds_info['file_name']}`**（单元 `{gds_info['cell_name']}`，图层 `{gds_info['selected_core_layer'][0]}/{gds_info['selected_core_layer'][1]}`）：\n"
            f"- **识别拓扑**：`{ext['topology_display']}`（{ext['ports_detected']} 端口）\n"
            f"- **多边形几何测量**：`R = {ext['ring_radius_um']:.3f} µm`，`W_ring = {ext['ring_width_um']:.3f} µm`，`W_bus = {ext['bus_width_um']:.3f} µm`，`耦合间距 gap = {ext['gap_thru_um']*1000:.1f} nm`"
            + (f"，`L_s = {ext['straight_length_um']:.2f} µm`" if ext["straight_length_um"] > 0 else "")
            + (f"，`包角 θ = {ext['pulley_angle_thru_deg']:.1f}°`" if "pulley" in ext["topology"] else "")
            + "\n\n**下一步**：请选择您的**晶圆叠层与波导传播损耗 (`dB/cm`)**，我将立即计算该版图的本征模与透射光谱："
        )
        chips = [
            {"label": "220 nm SOI (损耗 2.0 dB/cm) → 仿真", "prompt": "220 nm SOI 条形波导, SiO2包层, 损耗 2.0 dB/cm, TE0 @ 1550 nm, 开始仿真"},
            {"label": "400 nm Si3N4 (损耗 0.35 dB/cm) → 仿真", "prompt": "400 nm Si3N4 条形波导, SiO2包层, 损耗 0.35 dB/cm, TE0 @ 1550 nm, 开始仿真"},
            {"label": "600 nm 4H-SiC (损耗 0.6 dB/cm) → 仿真", "prompt": "600 nm 4H-SiC 条形波导, SiO2包层, 损耗 0.6 dB/cm, TE0 @ 1550 nm, 开始仿真"},
            {"label": "600 nm X切铌酸锂 (300nm平板) → 仿真", "prompt": "600 nm X切铌酸锂 300 nm 平板, SiO2包层, 损耗 0.25 dB/cm, TE0 @ 1550 nm, 开始仿真"},
        ]
    else:
        msg = (
            f"📐 **Parsed Uploaded Layout `{gds_info['file_name']}`** (cell `{gds_info['cell_name']}`, "
            f"layer `{gds_info['selected_core_layer'][0]}/{gds_info['selected_core_layer'][1]}`):\n"
            f"- **Detected Topology**: `{ext['topology_display']}` ({ext['ports_detected']}-port)\n"
            f"- **Measured XY Geometry**: `R = {ext['ring_radius_um']:.3f} µm`, `W_ring = {ext['ring_width_um']:.3f} µm`, "
            f"`W_bus = {ext['bus_width_um']:.3f} µm`, `gap = {ext['gap_thru_um']*1000:.1f} nm`"
            + (f", `L_s = {ext['straight_length_um']:.2f} µm`" if ext["straight_length_um"] > 0 else "")
            + (f", `θ = {ext['pulley_angle_thru_deg']:.1f}°`" if "pulley" in ext["topology"] else "")
            + "\n\n**Next Step**: Pick your **vertical wafer stack & propagation loss (`dB/cm`)** to compute the optical spectrum:"
        )
        chips = [
            {"label": "220 nm SOI (2.0 dB/cm) → Simulate", "prompt": "220 nm SOI strip, SiO2 clad, loss = 2.0 dB/cm, TE0 at 1550 nm, simulate"},
            {"label": "400 nm Si3N4 (0.35 dB/cm) → Simulate", "prompt": "400 nm Si3N4 strip, SiO2 clad, loss = 0.35 dB/cm, TE0 at 1550 nm, simulate"},
            {"label": "600 nm 4H-SiC (0.6 dB/cm) → Simulate", "prompt": "600 nm 4H-SiC strip, SiO2 clad, loss = 0.6 dB/cm, TE0 at 1550 nm, simulate"},
            {"label": "600 nm X-cut TFLN (300nm slab) → Simulate", "prompt": "600 nm X-cut TFLN with 300 nm slab, SiO2 clad, loss = 0.25 dB/cm, TE0 at 1550 nm, simulate"},
        ]

    session["history"] = (session.get("history", []) + [{"role": "assistant", "content": msg}])[-16:]
    return {
        "session": session,
        "assistant_message": msg,
        "chips": chips,
        "ui_view": "gds",
        "wizard_step": 4,
        "active_tab": "tab-gds",
    }


def _solve_radius_for_target_fsr(cfg: MRRDesignConfig, target_fsr_ghz: float | None, target_fsr_nm: float | None, ng_est: float) -> float:
    wl_m = cfg.wavelength_nm * 1e-9
    if target_fsr_ghz and target_fsr_ghz > 0:
        l_rt_m = C0_M_PER_S / (ng_est * target_fsr_ghz * 1e9)
    elif target_fsr_nm and target_fsr_nm > 0:
        l_rt_m = (wl_m * wl_m) / (ng_est * target_fsr_nm * 1e-9)
    else:
        return cfg.ring_radius_um

    ls_um = cfg.straight_length_um if cfg.topology.startswith("racetrack") else 0.0
    r_um = (l_rt_m * 1e6 - 2.0 * ls_um) / (2.0 * math.pi)
    return round(float(np.clip(r_um, 5.0, 800.0)), 2)


def _get_topology_fsr_chips(wl_nm: float, zh: bool = False) -> list[dict[str, str]]:
    if wl_nm > 3000.0:
        if zh:
            return [
                {"label": "⭕ 点耦合圆环 · FSR 150 GHz", "prompt": "FSR 150 GHz, 点耦合两端口"},
                {"label": "🎯 滑轮耦合圆环 · FSR 100 GHz", "prompt": "FSR 100 GHz, 滑轮两端口"},
                {"label": "🏁 跑道型微环 · R = 65 µm", "prompt": "半径 65 um, 直波导 20 um, 跑道两端口"},
                {"label": "🔀 四端口下载滤波环 · FSR 150 GHz", "prompt": "FSR 150 GHz, 四端口下载"},
            ]
        return [
            {"label": "⭕ Point Circle · FSR 150 GHz", "prompt": "FSR 150 GHz, point circular all-pass"},
            {"label": "🎯 Pulley Coupler · FSR 100 GHz", "prompt": "FSR 100 GHz, pulley all-pass"},
            {"label": "🏁 Racetrack · R = 65 µm", "prompt": "R = 65 um, L_s = 20 um, racetrack all-pass"},
            {"label": "🔀 4-Port Add-Drop · FSR 150 GHz", "prompt": "FSR 150 GHz, 4-port add-drop"},
        ]
    if zh:
        return [
            {"label": "⭕ 点耦合圆环 · FSR 200 GHz", "prompt": "FSR 200 GHz, 点耦合两端口"},
            {"label": "🎯 滑轮包角耦合 (高Q) · FSR 100 GHz", "prompt": "FSR 100 GHz, 滑轮两端口"},
            {"label": "🏁 跑道型微环 · FSR 100 GHz", "prompt": "FSR 100 GHz, 跑道两端口"},
            {"label": "🔀 四端口 Add-Drop · FSR 150 GHz", "prompt": "FSR 150 GHz, 四端口下载"},
            {"label": "🔬 双环游标滤波器 (Vernier)", "prompt": "半径 40 um, 双环游标四端口"},
        ]
    return [
        {"label": "⭕ Point Circle · FSR 200 GHz", "prompt": "FSR 200 GHz, point circular all-pass"},
        {"label": "🎯 Pulley (High-Q) · FSR 100 GHz", "prompt": "FSR 100 GHz, pulley all-pass"},
        {"label": "🏁 Racetrack · FSR 100 GHz", "prompt": "FSR 100 GHz, racetrack all-pass"},
        {"label": "🔀 4-Port Add-Drop · FSR 150 GHz", "prompt": "FSR 150 GHz, 4-port add-drop"},
        {"label": "🔬 Dual-Ring Vernier Filter", "prompt": "R = 40 um, dual-ring Vernier add-drop"},
    ]


def _build_compact_ai_state(session: dict) -> str:
    cfg = session.get("config") or {}
    parts = [
        f"stage={session.get('stage', 'welcome')}",
        f"top={cfg.get('topology')}",
        f"mat={cfg.get('core_material')}/{cfg.get('clad_material')}",
        f"wl={cfg.get('wavelength_nm')}nm({cfg.get('polarization')})",
        f"H={cfg.get('film_height_um')}um(slab={cfg.get('slab_height_um')}um)",
        f"W_ring={cfg.get('ring_width_um')}um,R={cfg.get('ring_radius_um')}um,gap={cfg.get('gap_thru_um')}um",
    ]
    if cfg.get("alpha_loss_db_per_cm") is not None:
        parts.append(f"loss={cfg.get('alpha_loss_db_per_cm')}dB/cm")
    ws = session.get("last_width_sweep") or {}
    if ws.get("single_mode_max_width_um"):
        parts.append(f"W_SM={ws['single_mode_max_width_um']:.3f}um,ng={ws.get('ng_at_rec', 0):.4f}")
    mm = (session.get("last_mrr_metrics") or {}).get("metrics") or {}
    cc = (session.get("last_mrr_metrics") or {}).get("critical_coupling") or {}
    if mm:
        parts.append(
            f"FSR={mm.get('FSR_GHz')}GHz,Q_int={mm.get('Q_int'):.0f},Q_L={mm.get('Q_loaded'):.0f},"
            f"ER={mm.get('ER_thru_dB')}dB,CC={cc.get('parameter')}={cc.get('recommended_value')}{cc.get('unit','')}"
        )
    return " | ".join(parts)


def process_chat_turn(
    session: dict | None,
    user_message: str,
    *,
    gds_export_dir: str | Path,
    use_ollama: bool = False,
    ai_mode: str = "offline",
    ai_api_key: str | None = None,
    lang: str | None = None,
) -> dict:
    """Process one user turn with local 2D FEM/CMT physics + optional Offline/Online AI (`Claude`/`Codex`/`Ollama`)."""
    eff_mode = (ai_mode or ("ollama" if use_ollama else "offline")).lower().strip()
    res = _process_chat_turn_core(
        session,
        user_message,
        gds_export_dir=gds_export_dir,
        lang=lang,
    )
    msg_clean = user_message.strip()
    if (
        eff_mode in ("ollama", "claude", "codex")
        and msg_clean
        and msg_clean.lower() not in ("reset", "__switch_lang__")
        and not res.get("silent_ui_sync")
    ):
        sess_out = res.get("session") or {}
        zh_flag = str(sess_out.get("lang", "en")).startswith("zh")
        compact_state = _build_compact_ai_state(sess_out)
        ai_note = query_ai_copilot(
            ai_mode=eff_mode,
            user_message=msg_clean,
            compact_state=compact_state,
            zh=zh_flag,
            api_key=ai_api_key,
        )
        if ai_note:
            res["assistant_message"] = f"{res['assistant_message']}\n\n---\n{ai_note}"
            if sess_out.get("history"):
                sess_out["history"][-1]["content"] = res["assistant_message"]
    res["ai_mode"] = eff_mode
    return res


def _process_chat_turn_core(
    session: dict | None,
    user_message: str,
    *,
    gds_export_dir: str | Path,
    lang: str | None = None,
) -> dict:
    """Core deterministic Guided Photonics Copilot (`en` or `zh`) + live local physics."""
    intent = parse_photonics_intent(user_message)
    if lang:
        zh = str(lang).lower().startswith("zh")
    elif intent["lang"] == "zh":
        zh = True
    else:
        zh = bool(session and str(session.get("lang", "en")).startswith("zh"))

    eff_lang = "zh" if zh else "en"
    if not session:
        session = create_initial_session(lang=eff_lang)
    session["lang"] = eff_lang

    if intent["action"] == "reset":
        fresh = create_initial_session(lang=eff_lang)
        return {
            "session": fresh,
            "assistant_message": fresh["history"][-1]["content"],
            "chips": get_welcome_chips(lang=eff_lang),
            "ui_view": "overview",
            "wizard_step": 1,
            "active_tab": "tab-mrr",
        }

    if user_message.strip() == "__switch_lang__":
        w_step = int(session.get("wizard_step") or 1)
        cfg_cur = dict(session.get("config") or {})
        conf_cur = set(session.get("confirmed") or [])
        wl_cur = float(cfg_cur.get("wavelength_nm") or 1550.0)
        mat_cur = str(cfg_cur.get("core_material") or "Si3N4")
        preset_cur = PLATFORM_PRESETS.get("ScAlN_20" if mat_cur.startswith("ScAlN") else mat_cur, PLATFORM_PRESETS["Si3N4"])
        if session.get("stage") == "welcome" or w_step <= 1:
            fresh_welcome = create_initial_session(lang=eff_lang)["history"][-1]["content"]
            if len(session.get("history") or []) <= 1:
                session["history"] = [{"role": "assistant", "content": fresh_welcome}]
            chips_lang = get_welcome_chips(lang=eff_lang)
        elif w_step == 2:
            chips_lang = _get_waveband_chips(zh)
        elif w_step == 3:
            chips_lang = _get_material_chips_for_wavelength(wl_cur, zh)
        elif w_step == 4:
            h_nm_c = round(float(cfg_cur.get("film_height_um", 0.40)) * 1000)
            chips_lang = (
                [
                    {"label": f"✅ 保持推荐叠层 (H={h_nm_c} nm, TE0) → 求解模式", "prompt": "确认叠层并求解模式"},
                    {"label": "🔄 切换为 TM0 偏振 → 求解模式", "prompt": "偏振 TM0, 确认叠层并求解模式"},
                ]
                if zh
                else [
                    {"label": f"✅ Keep Default Stack (H={h_nm_c} nm, TE0) → Solve Mode", "prompt": "Confirm stack and solve 2D modes"},
                    {"label": "🔄 Switch to TM0 Polarization → Solve Mode", "prompt": "Polarization TM0, confirm stack and solve 2D modes"},
                ]
            )
        elif w_step == 5 and "propagation_loss" not in conf_cur:
            l_vals = preset_cur.get("loss_choices", [0.3, 0.8, 1.5, 3.0])
            chips_lang = [
                {
                    "label": (f"损耗 α = {lv:g} dB/cm" if zh else f"Loss α = {lv:g} dB/cm"),
                    "prompt": (f"传播损耗 = {lv:g} dB/cm" if zh else f"Propagation loss = {lv:g} dB/cm"),
                }
                for lv in l_vals
            ]
        elif w_step == 5:
            chips_lang = _get_topology_fsr_chips(wl_cur, zh)
        else:
            cc_cur = (session.get("last_mrr_metrics") or {}).get("critical_coupling") or {}
            rec_str = f" ({cc_cur['recommended_value']} {cc_cur['unit']})" if "recommended_value" in cc_cur else ""
            chips_lang = (
                [
                    {"label": f"⚡ 一键应用临界耦合{rec_str}", "prompt": "临界耦合"},
                    {"label": "💾 导出单器件 .GDS + .LYP", "prompt": "导出单个 GDS 版图"},
                    {"label": "🗂️ 导出 5 环扫描阵列 .GDS", "prompt": "导出阵列 GDS 版图"},
                    {"label": "📏 扫描耦合间距 Gap", "prompt": "扫描间距"},
                    {"label": "📉 扫描传播损耗 (dB/cm)", "prompt": "扫描损耗"},
                ]
                if zh
                else [
                    {"label": f"⚡ Apply Critical Coupling{rec_str}", "prompt": "Apply critical coupling"},
                    {"label": "💾 Export Single .GDS + .LYP", "prompt": "Export single GDS layout"},
                    {"label": "🗂️ Export 5-Ring Sweep Array .GDS", "prompt": "Export 5-ring parameter sweep array GDS"},
                    {"label": "📏 Sweep Coupler Gap", "prompt": "Sweep gap"},
                    {"label": "📉 Sweep Loss (dB/cm)", "prompt": "Sweep propagation loss"},
                ]
            )
        return {
            "session": session,
            "assistant_message": session["history"][-1]["content"] if session.get("history") else "",
            "chips": chips_lang,
            "ui_view": session.get("ui_view", "overview"),
            "wizard_step": w_step,
            "silent_ui_sync": True,
        }

    cfg_dict = dict(session["config"])
    confirmed = set(session.get("confirmed", []))
    history = list(session.get("history", []))
    history.append({"role": "user", "content": user_message})

    # Guided Workflow Entry: Lab Spectrum Loss & Q Extractor
    if intent["action"] == "open_spectrum_fitter":
        session["stage"] = "spectrum_fitter"
        session["wizard_step"] = 1
        session["ui_view"] = "fit"
        msg = (
            "📈 **实测光谱波导传播损耗 (`dB/cm`) 与 Q 值提取**\n\n"
            "我已在右侧打开**实验光谱 CMT 拟合面板**并加载了一条高 Q 双峰示例数据：\n"
            "1. 点击右侧 **`📂 上传实验 .CSV / .TXT`**（自动兼容线性透过率或 `dB`/`dBm` 数据），或点击示例按钮。\n"
            "2. 填入群折射率 `n_g`（或选择 **`宽谱多谐振峰 Multi-FSR`**，直接从谐振峰间距同时自洽提取 `n_g` 与损耗 `α [dB/cm]`！）。\n"
            "3. 提取完成后，点击绿色按钮 **`🎯 使用该损耗设计新微环`** 即可一键带入临界耦合设计！"
            if zh
            else "📈 **Lab Spectrum Propagation Loss (`dB/cm`) & Q Extractor**\n\n"
            "I've opened the **Lab Spectrum CMT Fitter** on the right and loaded a sample high-Q doublet trace:\n"
            "1. Click **`📂 Upload Lab .CSV / .TXT`** on the right (accepts either linear power OR `dB`/`dBm` traces!), or try the demo spectra.\n"
            "2. Enter your **Group Index `n_g`** (or pick **`Broadband Multi-FSR`** to extract BOTH `n_g` and `α [dB/cm]` directly from resonance spacing!).\n"
            "3. Click **`🎯 Design Ring with α`** to feed your measured `dB/cm` straight into critical-coupling design!"
        )
        chips = (
            [{"label": "✨ 或开始逐步设计新微环", "prompt": "开始逐步设计微环"}]
            if zh
            else [{"label": "✨ Or Start Step-by-Step Ring Design", "prompt": "Start step-by-step microring design"}]
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "ui_view": "fit",
            "auto_demo_fit": True,
            "wizard_step": 1,
            "active_tab": "tab-fit",
        }

    # Guided Workflow Entry: Upload GDS
    if intent["action"] == "open_gds_uploader":
        session["stage"] = "awaiting_gds"
        session["wizard_step"] = 1
        session["ui_view"] = "gds"
        msg = (
            "📐 **上传 `.GDS` 或 `.OAS` 版图文件**\n\n"
            "请点击右侧面板顶部的 **`📂 上传 .GDS / .OAS`** 选择您的版图文件。我将自动从多边形提取半径 `R`、波导宽 `W`、耦合间距 `gap` 与包角，随后只需确认膜厚与损耗 (`dB/cm`) 即可计算透射光谱！"
            if zh
            else "📐 **Upload Your `.GDS` or `.OAS` Layout**\n\n"
            "Click **`📂 Upload .GDS / .OAS`** on the right panel. "
            "I will automatically measure `R`, `W_ring`, `W_bus`, `gap`, and coupler geometry from the polygons, "
            "then ask for your wafer thickness & propagation loss (`dB/cm`) to simulate its optical spectrum!"
        )
        chips = (
            [{"label": "✨ 或从零逐步设计新微环", "prompt": "开始逐步设计微环"}]
            if zh
            else [{"label": "✨ Or Design a New Ring from Scratch", "prompt": "Start step-by-step microring design"}]
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "ui_view": "gds",
            "wizard_step": 1,
            "active_tab": "tab-gds",
        }

    # Guided Step 1 -> Step 2: Ask Waveband
    if intent["action"] == "start_guided":
        session["stage"] = "ask_waveband"
        session["wizard_step"] = 2
        session["ui_view"] = "overview"
        msg = (
            "**第 2 / 6 步 — 选择工作波长波段** 🌈\n\n"
            "您的器件工作在哪个光学波段？\n"
            "*（可点击下方常用波段按钮，或直接输入 `400 nm – 14000 nm` 之间的任意精确波长，例如 `737 nm` 或 `1530 nm`。）*"
            if zh
            else "**Step 2 of 6 — Target Wavelength Band** 🌈\n\n"
            "Which optical wavelength band does your device operate in?\n"
            "*(Click a band below, or type any exact wavelength from `400 nm` to `14000 nm`, e.g. `737 nm` or `1530 nm`.)*"
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": _get_waveband_chips(zh),
            "ui_view": "overview",
            "wizard_step": 2,
            "active_tab": "tab-mrr",
        }

    # Apply platform preset FIRST if user switched core material (preserving prior explicit wavelength in guided mode!)
    explicit_keys = set(intent["fields"].keys())
    prev_wl = float(cfg_dict.get("wavelength_nm", 1550.0))
    keep_prior_wl = ("wavelength" in confirmed) and ("wavelength_nm" not in explicit_keys)

    new_mat = intent["fields"].get("core_material")
    if new_mat and (new_mat != cfg_dict.get("core_material") or "core_material" not in confirmed):
        preset_key = "ScAlN_20" if new_mat.startswith("ScAlN") else new_mat
        if preset_key in PLATFORM_PRESETS and "uploaded_gds" not in confirmed:
            for pk, pv in PLATFORM_PRESETS[preset_key].items():
                if pk not in ("w_sweep", "loss_choices"):
                    if pk == "wavelength_nm" and keep_prior_wl:
                        continue
                    cfg_dict[pk] = pv
        elif preset_key in PLATFORM_PRESETS and "uploaded_gds" in confirmed:
            for pk in ("film_height_um", "slab_height_um", "sidewall_deg", "clad_material", "top_clad_material", "wavelength_nm"):
                if pk in PLATFORM_PRESETS[preset_key]:
                    if pk == "wavelength_nm" and keep_prior_wl:
                        continue
                    cfg_dict[pk] = PLATFORM_PRESETS[preset_key][pk]
        confirmed.add("core_material")

    # Apply parsed fields
    for k, v in intent["fields"].items():
        cfg_dict[k] = v
        if k == "wavelength_nm":
            confirmed.add("wavelength")
        if k in ("film_height_um", "slab_height_um", "sidewall_deg", "top_clad_material", "clad_material"):
            confirmed.add("wafer_stack")
        if k == "polarization":
            confirmed.add("polarization")
        if k == "alpha_loss_db_per_cm":
            confirmed.add("propagation_loss")
        if k == "topology":
            confirmed.add("topology")
        if k in ("ring_radius_um", "gap_thru_um", "pulley_angle_thru_deg", "straight_length_um"):
            confirmed.add("resonator_size")
        if k in ("ring_width_um", "bus_width_um"):
            confirmed.add("explicit_widths")

    # Scale defaults if user selected a non-default waveband
    scaled_w_sweep: tuple[float, float] | None = None
    if ("wavelength_nm" in explicit_keys or (new_mat is not None and keep_prior_wl)) and "uploaded_gds" not in confirmed:
        cur_wl = float(cfg_dict["wavelength_nm"])
        if abs(cur_wl - prev_wl) > 25.0 or new_mat is not None:
            scaled_w_sweep = scale_preset_for_wavelength(cfg_dict, cur_wl, explicit_keys)
            session["last_width_sweep"] = None

    if "q_int_target" in intent["targets"]:
        q_target = float(intent["targets"]["q_int_target"])
        ng_ref = (session.get("last_width_sweep") or {}).get("ng_at_rec") or 2.15
        cfg_dict["alpha_loss_db_per_cm"] = round(
            q_int_to_loss_db_per_cm(q_target, float(ng_ref), float(cfg_dict["wavelength_nm"])), 4
        )
        confirmed.add("propagation_loss")
    elif intent["targets"].get("confirm_default_loss"):
        confirmed.add("propagation_loss")

    if "ports" in intent["targets"] and "topology" not in intent["fields"]:
        want_4 = intent["targets"]["ports"] == 4
        cur_t = cfg_dict["topology"]
        if cur_t.startswith("pulley"):
            cfg_dict["topology"] = "pulley_ad" if want_4 else "pulley_ap"
        elif cur_t.startswith("racetrack"):
            cfg_dict["topology"] = "racetrack_ad" if want_4 else "racetrack_ap"
        elif cur_t != "vernier_crow_ad":
            cfg_dict["topology"] = "point_circular_ad" if want_4 else "point_circular_ap"
        confirmed.add("topology")

    if "fsr_ghz" in intent["targets"]:
        session["target_fsr_ghz"] = intent["targets"]["fsr_ghz"]
        session["target_fsr_nm"] = None
        confirmed.add("resonator_size")
    elif "fsr_nm" in intent["targets"]:
        session["target_fsr_nm"] = intent["targets"]["fsr_nm"]
        session["target_fsr_ghz"] = None
        confirmed.add("resonator_size")

    session["config"] = cfg_dict
    session["confirmed"] = sorted(confirmed)
    cfg = _dict_to_cfg(cfg_dict)

    # ------------------------------------------------------------------
    # Guided Step 2 -> Step 3: Wavelength set, Core Material NOT yet set
    # ------------------------------------------------------------------
    if "core_material" not in confirmed and "wavelength" in confirmed:
        wl_now = float(cfg_dict["wavelength_nm"])
        session["stage"] = "ask_material"
        session["wizard_step"] = 3
        session["ui_view"] = "overview"
        msg = (
            f"**第 3 / 6 步 — 选择波导芯层材料** 💎\n\n"
            f"工作波长已设为 **`{wl_now:g} nm`**。请选择您的晶圆波导芯层材料：\n"
            f"*（下方已自动筛选在 `{wl_now:g} nm` 透明的材料平台，您也可直接输入材料名。）*"
            if zh
            else f"**Step 3 of 6 — Waveguide Core Material** 💎\n\n"
            f"Operating at **`{wl_now:g} nm`**. Which waveguide core material is on your wafer?\n"
            f"*(Showing platforms transparent around `{wl_now:g} nm` below, or type any material name.)*"
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": _get_material_chips_for_wavelength(wl_now, zh),
            "ui_view": "overview",
            "wizard_step": 3,
            "active_tab": "tab-mrr",
        }

    # ------------------------------------------------------------------
    # Direct Action 1: Apply Critical Coupling (`g_CC` / `theta_CC` / `L_s,CC`)
    # ------------------------------------------------------------------
    if intent["action"] == "apply_critical_coupling":
        last_m = session.get("last_mrr_metrics")
        if not last_m:
            res_init = simulate_mrr_topology(cfg)
            cc = res_init["critical_coupling"]
        else:
            cc = last_m["critical_coupling"]
        p_name = cc["parameter"]
        cfg_dict[p_name] = float(cc["recommended_value"])
        if cfg_dict["topology"].endswith("_ad") and p_name == "gap_thru_um":
            cfg_dict["gap_drop_um"] = round(float(cc["recommended_value"]) + 0.06, 4)
        session["config"] = cfg_dict
        cfg = _dict_to_cfg(cfg_dict)
        mrr_res = simulate_mrr_topology(cfg)
        session["last_mrr_metrics"] = {
            "metrics": mrr_res["metrics"],
            "critical_coupling": mrr_res["critical_coupling"],
        }
        session["stage"] = "mrr_synthesized"
        session["wizard_step"] = 6
        session["ui_view"] = "mrr"
        m = mrr_res["metrics"]
        if zh:
            msg = (
                f"⚡ **已针对损耗 `α = {m['alpha_loss_db_per_cm']:.2f} dB/cm` 应用临界耦合参数 (`{p_name} = {cc['recommended_value']} {cc['unit']}`)**：\n"
                f"- **直通端消光比**: **`{m['ER_thru_dB']:.1f} dB`**（`FSR = {m['FSR_GHz']:.2f} GHz` / `{m['FSR_nm']:.3f} nm`）\n"
                f"- **品质因数**: `Q_loaded = {m['Q_loaded']:,.0f}`（`Q_int = {m['Q_int']:,.0f}`，`Q_ext = {m['Q_ext_thru']:,.0f}`）\n\n"
                "您可以继续运行**间距/损耗容差扫描**，或直接**生成 `.GDS` 流片版图**："
            )
            chips = [
                {"label": "💾 导出单器件 .GDS + .LYP", "prompt": "导出单个 GDS 版图"},
                {"label": "🗂️ 导出 5 环参数扫描阵列 .GDS", "prompt": "导出阵列 GDS 版图"},
                {"label": "📏 扫描耦合间距 Gap", "prompt": "扫描间距"},
                {"label": "📉 扫描传播损耗 (dB/cm) 容差", "prompt": "扫描损耗"},
                {"label": "🌡️ 扫描热光温升 ΔT (0–40 K)", "prompt": "扫描温度 0 到 40 K"},
            ]
        else:
            msg = (
                f"⚡ **Applied Critical Coupling (`{p_name} = {cc['recommended_value']} {cc['unit']}`)** for **`α = {m['alpha_loss_db_per_cm']:.2f} dB/cm`**:\n"
                f"- **Extinction Ratio**: **`{m['ER_thru_dB']:.1f} dB`** (`FSR = {m['FSR_GHz']:.2f} GHz` / `{m['FSR_nm']:.3f} nm`)\n"
                f"- **Quality Factors**: `Q_loaded = {m['Q_loaded']:,.0f}` (`Q_int = {m['Q_int']:,.0f}`, `Q_ext = {m['Q_ext_thru']:,.0f}`)\n\n"
                "Would you like to **sweep gap / propagation loss tolerances** or **export your `.GDS` layout**?"
            )
            chips = [
                {"label": "💾 Export Single .GDS + .LYP", "prompt": "Export single GDS layout"},
                {"label": "🗂️ Export 5-Ring Sweep Array .GDS", "prompt": "Export 5-ring parameter sweep array GDS"},
                {"label": "📏 Sweep Coupler Gap", "prompt": "Sweep gap around current design"},
                {"label": "📉 Sweep Loss (dB/cm) Tolerance", "prompt": "Sweep propagation loss"},
                {"label": "🌡️ Sweep Heater ΔT (0 to 40 K)", "prompt": "Sweep temperature from 0 to 40 K"},
            ]
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "mrr_result": mrr_res,
            "ui_view": "mrr",
            "wizard_step": 6,
            "active_tab": "tab-mrr",
        }

    # ------------------------------------------------------------------
    # Direct Action 2: Export Single GDS or 5-Ring Sweep Array GDS
    # ------------------------------------------------------------------
    if intent["action"] in ("export_single_gds", "export_sweep_gds"):
        out_dir = Path(gds_export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if intent["action"] == "export_single_gds":
            fname = f"MicroRing_{cfg.topology}_{cfg.core_material}_R{int(cfg.ring_radius_um)}um.gds"
            gds_path = export_mrr_gds(cfg, out_dir / fname)
            lyp_path = write_klayout_layer_props(Path(gds_path).with_suffix(".lyp"))
            gds_view = analyze_uploaded_gds(gds_path)
            msg = (
                f"💾 **已生成流片级 `.GDS` 版图与 KLayout 图层文件**：\n"
                f"- **GDSII 文件**: `{Path(gds_path).name}`（`R = {cfg.ring_radius_um} µm`，`W_ring = {cfg.ring_width_um} µm`，`gap = {cfg.gap_thru_um*1000:.0f} nm`）\n"
                f"- **KLayout 配色文件**: `{Path(lyp_path).name}`\n\n"
                "请点击右侧绿色按钮下载 `.GDS` 文件，或在画布中缩放查看多边形与间距标尺！"
                if zh
                else f"💾 **Generated Foundry-Ready `.GDS` Mask & KLayout `.LYP`**:\n"
                f"- **GDSII File**: `{Path(gds_path).name}` (`R = {cfg.ring_radius_um} µm`, `W_ring = {cfg.ring_width_um} µm`, `gap = {cfg.gap_thru_um*1000:.0f} nm`)\n"
                f"- **KLayout Layer Map**: `{Path(lyp_path).name}`\n\n"
                "Click the green **Download `.GDS`** button on the right to save your mask!"
            )
        else:
            p_name = TOPOLOGY_CATALOG[cfg.topology]["coupler_param"]
            center = float(getattr(cfg, p_name))
            if p_name.endswith("_um"):
                vals = [round(max(0.10, center + d), 3) for d in (-0.04, -0.02, 0.0, 0.02, 0.04)]
            else:
                vals = [round(max(2.0, center * f), 2) for f in (0.7, 0.85, 1.0, 1.15, 1.30)]
            fname = f"MicroRing_SweepArray_{cfg.core_material}_{p_name}.gds"
            gds_path = export_sweep_array_gds(cfg, p_name, vals, out_dir / fname)
            lyp_path = str(Path(gds_path).with_suffix(".lyp"))
            gds_view = analyze_uploaded_gds(gds_path)
            msg = (
                f"🗂️ **已生成 5 器件 MPW 参数扫描阵列版图 `{Path(gds_path).name}`**\n"
                f"- 扫描 `{p_name}`：`{vals}`，用于覆盖流片刻蚀与损耗波动以锁定片上临界耦合。\n"
                f"- 含端口引脚 (`65/0`)、热光/电光焊盘 (`10/0`, `11/0`) 及 KLayout `{Path(lyp_path).name}`。"
                if zh
                else f"🗂️ **Generated 5-Ring MPW Parameter-Sweep Array `{Path(gds_path).name}`**\n"
                f"- Sweeps `{p_name}` across `{vals}` to guarantee critical coupling on-wafer despite etch/loss variations.\n"
                f"- Includes port pins (`65/0`), heater/GSG pads (`10/0`, `11/0`), and KLayout `{Path(lyp_path).name}`."
            )
        session["wizard_step"] = 6
        session["ui_view"] = "gds"
        chips = (
            [
                {"label": "📊 返回微环透射光谱", "prompt": "开始仿真"},
                {"label": "📏 扫描耦合间距", "prompt": "扫描间距"},
            ]
            if zh
            else [
                {"label": "📊 Back to MRR Spectrum", "prompt": "Simulate current MRR"},
                {"label": "📏 Sweep Coupler Gap", "prompt": "Sweep gap"},
            ]
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "gds_export": {"gds_path": gds_path, "lyp_path": lyp_path, "gds_view": gds_view},
            "ui_view": "gds",
            "wizard_step": 6,
            "active_tab": "tab-gds",
        }

    # ------------------------------------------------------------------
    # Direct Action 2B: Lithography Dose / Positive-Negative PR Error Analysis & Pre-Bias GDS
    # ------------------------------------------------------------------
    if intent["action"] == "export_fab_bias_gds":
        out_dir = Path(gds_export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"MicroRing_OPC_DoseBracket_{cfg.core_material}_{cfg.topology}.gds"
        bias_info = export_fab_bias_array_gds(cfg, out_dir / fname)
        gds_path = bias_info["gds_path"]
        lyp_path = bias_info["lyp_path"]
        gds_view = analyze_uploaded_gds(gds_path)
        msg = (
            f"🗂️ **已生成 5 单元光刻剂量/线宽预偏置补偿阵列 `{Path(gds_path).name}`**\n"
            f"- 版图预偏置 `ΔCD_drawn = [-20, -10, 0, +10, +20] nm`（同步反向补偿 `W - δ` 与 `gap + δ`），确保超净间无论过曝或欠曝，片上至少有一颗微环精准命中临界耦合！\n"
            f"- 配套 KLayout 图层文件：`{Path(lyp_path).name}`。"
            if zh
            else f"🗂️ **Generated 5-Cell Lithography Pre-Bias Compensation Array `{Path(gds_path).name}`**\n"
            f"- Brackets drawn mask `ΔCD_drawn = [-20, -10, 0, +10, +20] nm` (`W_drawn + δ`, `gap_drawn - δ`) so even if cleanroom exposure over-doses or under-doses by `±20 nm`, one cell on-chip lands right at critical coupling!\n"
            f"- Includes KLayout layer map `{Path(lyp_path).name}`."
        )
        session["wizard_step"] = 6
        session["ui_view"] = "gds"
        chips = (
            [
                {"label": "🧪 返回工艺/光刻胶容差面板", "prompt": "正胶过曝与欠曝工艺误差分析"},
                {"label": "📊 返回微环透射光谱", "prompt": "开始仿真"},
            ]
            if zh
            else [
                {"label": "🧪 Back to Fab / PR Dose Panel", "prompt": "Check positive PR over-dose and under-dose error"},
                {"label": "📊 Back to MRR Spectrum", "prompt": "Simulate current MRR"},
            ]
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "gds_export": {"gds_path": gds_path, "lyp_path": lyp_path, "gds_view": gds_view},
            "ui_view": "gds",
            "wizard_step": 6,
            "active_tab": "tab-gds",
        }

    if intent["action"] == "fab_tolerance":
        fab_opts = intent.get("fab") or {}
        resist_tone = str(fab_opts.get("resist_tone") or "positive_pr")
        dose_pct = float(fab_opts.get("max_dose_error_pct") or 15.0)
        cd_nm = float(fab_opts.get("max_cd_bias_nm") or 30.0)
        fab_res = run_fabrication_error_analysis(
            cfg,
            resist_tone=resist_tone,
            max_dose_error_pct=dose_pct,
            max_cd_bias_nm=cd_nm,
            num_points=7,
        )
        s = fab_res["summary"]
        uc = s["under_dose_corner"]
        oc = s["over_dose_corner"]
        session["wizard_step"] = 6
        session["ui_view"] = "fab"
        if zh:
            msg = (
                f"🧪 **已完成光刻曝光与刻蚀容差分析（`{fab_res['resist_name_zh']}`）**\n"
                f"- **物理机制**：\n"
                f"  - **过曝光 (`{oc['dose_error_pct']:+.0f}%`)**：`ΔCD = {oc['delta_cd_nm']:+.1f} nm`，实际 `gap = {oc['gap_fab_um']*1000:.0f} nm` → `ER = {oc['extinction_ratio_dB']:.1f} dB`（`{oc['coupling_regime']}`，`Δλ = {oc['delta_wl_res_nm']:+.2f} nm`）\n"
                f"  - **欠曝光 (`{uc['dose_error_pct']:+.0f}%`)**：`ΔCD = {uc['delta_cd_nm']:+.1f} nm`，实际 `gap = {uc['gap_fab_um']*1000:.0f} nm` → `ER = {uc['extinction_ratio_dB']:.1f} dB`（`{uc['coupling_regime']}`，`Δλ = {uc['delta_wl_res_nm']:+.2f} nm`）\n"
                f"- **`ER ≥ 15 dB` 工艺良率窗口**：线宽窗口 **`{s['cd_window_er15_span_nm']:.1f} nm`**（`{s['cd_window_er15_nm']}` nm）｜曝光剂量窗口 **`{s['dose_window_er15_span_pct']:.1f}%`**\n"
                f"- **谐振波长线宽灵敏度**：**`dλ/dCD = {s['sensitivity_dl_dcd_pm_per_nm']:+.1f} pm/nm`**"
            )
            if fab_res["drc_warnings"]:
                msg += "\n- ⚠️ **超净间工艺提示**：" + "；".join(fab_res["drc_warnings"])
            chips = [
                {
                    "label": "🔄 切换为负性光刻胶 (HSQ / ma-N)" if resist_tone != "negative_pr" else "🔄 切换为正性光刻胶 (ZEP / DUV)",
                    "prompt": "负胶 ±15% 曝光剂量误差分析" if resist_tone != "negative_pr" else "正胶 ±15% 曝光剂量误差分析",
                },
                {"label": "🗂️ 导出 5 单元预偏置补偿阵列 .GDS", "prompt": "导出预偏置补偿阵列 GDS"},
                {"label": "⚡ 先优化至临界耦合再测容差", "prompt": "临界耦合"},
            ]
        else:
            msg = (
                f"🧪 **Lithography & Photoresist Fabrication Error Analysis (`{fab_res['resist_name_en']}`)**\n"
                f"- **Over-Dose (`{oc['dose_error_pct']:+.0f}%`)**: `ΔCD = {oc['delta_cd_nm']:+.1f} nm`, `gap_fab = {oc['gap_fab_um']*1000:.0f} nm` -> `ER = {oc['extinction_ratio_dB']:.1f} dB` (`{oc['coupling_regime']}`, `Δλ = {oc['delta_wl_res_nm']:+.2f} nm`)\n"
                f"- **Under-Dose (`{uc['dose_error_pct']:+.0f}%`)**: `ΔCD = {uc['delta_cd_nm']:+.1f} nm`, `gap_fab = {uc['gap_fab_um']*1000:.0f} nm` -> `ER = {uc['extinction_ratio_dB']:.1f} dB` (`{uc['coupling_regime']}`, `Δλ = {uc['delta_wl_res_nm']:+.2f} nm`)\n"
                f"- **`ER ≥ 15 dB` Process Yield Window**: CD window **`{s['cd_window_er15_span_nm']:.1f} nm`** (`{s['cd_window_er15_nm']}` nm) | Dose window **`{s['dose_window_er15_span_pct']:.1f}%`**\n"
                f"- **Resonance CD Sensitivity**: **`dλ/dCD = {s['sensitivity_dl_dcd_pm_per_nm']:+.1f} pm/nm`**"
            )
            if fab_res["drc_warnings"]:
                msg += "\n- ⚠️ **Cleanroom DRC Notes**: " + " ".join(fab_res["drc_warnings"])
            chips = [
                {
                    "label": "🔄 Compare Negative PR (HSQ / ma-N)" if resist_tone != "negative_pr" else "🔄 Compare Positive PR (ZEP / DUV)",
                    "prompt": "Analyze negative PR ±15% over-dose and under-dose" if resist_tone != "negative_pr" else "Analyze positive PR ±15% over-dose and under-dose",
                },
                {"label": "🗂️ Export 5-Cell Pre-Biased Compensation .GDS", "prompt": "Export pre-biased dose compensation array GDS"},
                {"label": "⚡ Apply Critical Coupling First", "prompt": "Apply critical coupling"},
            ]
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "fab_result": fab_res,
            "ui_view": "fab",
            "wizard_step": 6,
            "active_tab": "tab-fab",
        }

    # ------------------------------------------------------------------
    # Direct Action 2C: 2D-FEM TO / EO / Si-PN Active Tuning & Lumerical INTERCONNECT Export
    # ------------------------------------------------------------------
    if intent["action"] in ("active_tuning", "export_interconnect"):
        t_opts = intent.get("tuning") or {}
        mech_req = str(t_opts.get("mechanism") or "auto")
        if mech_req in ("si_pn_depletion", "si_pin_injection") and cfg.core_material != "Si":
            # Auto-switch platform to SOI if user explicitly requested Si PN/PIN from default Si3N4
            preset = PLATFORM_PRESETS["Si"]
            cfg_dict.update({
                "core_material": "Si",
                "film_height_um": preset["film_height_um"],
                "slab_height_um": 0.09,
                "ring_width_um": preset["ring_width_um"],
                "bus_width_um": preset["bus_width_um"],
                "ring_radius_um": preset["ring_radius_um"],
                "gap_thru_um": preset["gap_thru_um"],
                "gap_drop_um": preset["gap_drop_um"],
            })
            session["config"] = cfg_dict
            cfg = _dict_to_cfg(cfg_dict)

        tune_res = run_active_tuning_analysis(
            cfg,
            mechanism=mech_req,
            drive_type=str(t_opts.get("drive_type") or "auto"),
            bias_values=t_opts.get("bias_values"),
            box_thickness_um=float(t_opts.get("box_thickness_um") or 2.0),
            heater_gap_um=float(t_opts.get("heater_gap_um") or 0.90),
            electrode_gap_um=t_opts.get("electrode_gap_um"),
        )
        eff = tune_res["efficiency"]
        gp = tune_res["geometry_physics"]
        mech_used = eff["mechanism"]
        session["last_tuning_result"] = tune_res
        session["wizard_step"] = 6
        session["ui_view"] = "tuning"

        ic_res = None
        if intent["action"] == "export_interconnect" or t_opts.get("export_interconnect"):
            out_dir = Path(gds_export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_mat = str(cfg.core_material).replace("-", "")
            ic_res = export_lumerical_interconnect_bundle(
                cfg, tune_res, out_dir / f"interconnect_{safe_mat}_{mech_used}", make_zip=True
            )

        if mech_used == "thermo_optic":
            geom_line_en = (
                f"2D Fourier FEM: `R_th = {gp.get('r_th_core_k_per_mw', '--')} K/mW`, "
                f"`R_elec = {gp.get('heater_resistance_ohm', '--')} Ω`, `BOX = {gp.get('box_thickness_um', 2.0)} µm`"
            )
            geom_line_zh = (
                f"2D 热传导有限元：`R_th = {gp.get('r_th_core_k_per_mw', '--')} K/mW`，"
                f"`R_elec = {gp.get('heater_resistance_ohm', '--')} Ω`，`BOX = {gp.get('box_thickness_um', 2.0)} µm`"
            )
        elif mech_used == "pockels_eo":
            geom_line_en = (
                f"2D Laplace FEM: `E_x = {gp.get('ex_core_v_per_um_per_v', '--')} (V/µm)/V`, "
                f"`C_elec = {gp.get('c_elec_ff_per_um', '--')} fF/µm`, `G = {gp.get('electrode_gap_um', '--')} µm`"
            )
            geom_line_zh = (
                f"2D 静电场有限元：`E_x = {gp.get('ex_core_v_per_um_per_v', '--')} (V/µm)/V`，"
                f"`C_elec = {gp.get('c_elec_ff_per_um', '--')} fF/µm`，电极间距 `G = {gp.get('electrode_gap_um', '--')} µm`"
            )
        else:
            geom_line_en = (
                f"2D Si Junction + Maxwell FEM: `W_dep(0V) = {gp.get('w_dep_0v_nm', '--')} nm`, "
                f"`C_j(0V) = {gp.get('cj_0v_ff_per_um', '--')} fF/µm`, `R_s = {gp.get('r_series_ohm', '--')} Ω`"
            )
            geom_line_zh = (
                f"2D 硅结耗尽/注入 + 本征模有限元：`W_dep(0V) = {gp.get('w_dep_0v_nm', '--')} nm`，"
                f"`C_j(0V) = {gp.get('cj_0v_ff_per_um', '--')} fF/µm`，串联电阻 `R_s = {gp.get('r_series_ohm', '--')} Ω`"
            )

        if zh:
            msg = (
                f"⚡ **已完成纯自研 2D-FEM 多物理场有源调谐与多态光谱计算（零 COMSOL 依赖）**\n"
                f"- **平台与机制**：`{cfg.core_material}` · `{gp['mechanism_name_zh']}`（输入激励：`{gp['drive_type']}`，共 `{len(tune_res['states'])}` 个状态）\n"
                f"- **器件调谐斜率**：**`{eff['wavelength_slope']} {eff['wavelength_slope_unit']}`**（`{eff['freq_slope']} {eff['freq_slope_unit']}`）\n"
                f"- **π 相移品质因数**：**`{eff['pi_figure_label']} = {eff['pi_figure']} {eff['pi_figure_unit']}`**｜**3-dB 调制带宽**：**`{eff['bandwidth_3db']} {eff['bandwidth_unit']}`**\n"
                f"- **几何截面耦合参数**：{geom_line_zh}"
            )
            if ic_res:
                msg += (
                    f"\n- 📦 **已生成 Ansys Lumerical INTERCONNECT 紧凑模型包**：`{ic_res['zip_name']}`"
                    f"（含无源 S 参数 `.dat`、有源复折射率表 `.txt`、电路自动生成 `.lsf` 与 `lumapi` 脚本，无源性校验 = `{ic_res['passivity_verified']}`）"
                )
            chips = [
                {"label": "📦 导出 Lumerical INTERCONNECT 模型包 (.ZIP)", "prompt": "导出 Lumerical INTERCONNECT 紧凑模型"},
                {"label": "🌡️ 按直接温度 ΔT (0, 20, 50 K) 计算光谱", "prompt": "温度光谱 0, 20, 50 K"},
                {"label": "⚡ 按直接电场 E (-10, 0, 10 V/um) 计算光谱", "prompt": "电场调谐光谱 -10, 0, 10 V/um"},
                {"label": "📊 返回无源微环光谱", "prompt": "开始仿真"},
            ]
        else:
            msg = (
                f"⚡ **Completed Native 2D-FEM Active Tuning & Multi-State Spectra (Zero COMSOL)**\n"
                f"- **Platform & Mechanism**: `{cfg.core_material}` · `{gp['mechanism_name_en']}` (Drive: `{gp['drive_type']}`, `{len(tune_res['states'])}` states)\n"
                f"- **Device Tuning Efficiency**: **`{eff['wavelength_slope']} {eff['wavelength_slope_unit']}`** (`{eff['freq_slope']} {eff['freq_slope_unit']}`)\n"
                f"- **π-Phase-Shift Figure**: **`{eff['pi_figure_label']} = {eff['pi_figure']} {eff['pi_figure_unit']}`** | **3-dB Bandwidth**: **`{eff['bandwidth_3db']} {eff['bandwidth_unit']}`**\n"
                f"- **Cross-Section Geometry Coupling**: {geom_line_en}"
            )
            if ic_res:
                msg += (
                    f"\n- 📦 **Exported Ansys Lumerical INTERCONNECT Compact Model**: `{ic_res['zip_name']}` "
                    f"(`SPAR .dat` + `OMM n_eff .txt` + `.lsf` circuit + `lumapi` runner; passivity verified = `{ic_res['passivity_verified']}`)"
                )
            chips = [
                {"label": "📦 Export Lumerical INTERCONNECT (.ZIP)", "prompt": "Export Lumerical INTERCONNECT compact model"},
                {"label": "🌡️ Direct ΔT Spectra (0, 20, 50 K)", "prompt": "Temperature tuning spectra at 0, 20, 50 K"},
                {"label": "⚡ Direct E-Field Spectra (-10, 0, 10 V/um)", "prompt": "E-field tuning spectra at -10, 0, 10 V/um"},
                {"label": "📊 Back to Passive MRR Spectrum", "prompt": "Simulate current MRR"},
            ]

        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        ret_payload = {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "tuning_result": tune_res,
            "ui_view": "tuning",
            "wizard_step": 6,
            "active_tab": "tab-tuning",
        }
        if ic_res is not None:
            ret_payload["interconnect_export"] = ic_res
        return ret_payload

    # ------------------------------------------------------------------
    # Direct Action 2D: License-Free Butt / Edge Coupler & 2D-FDFD Grating Coupler
    # ------------------------------------------------------------------
    if intent["action"] == "io_coupler":
        io_opts = intent.get("io") or {}
        ctype = str(io_opts.get("coupler_type") or "grating_coupler")
        fib_preset = str(io_opts.get("fiber_preset") or ("Lensed_2p5um" if ctype == "butt_edge" else "SMF28_1550"))
        if ctype == "butt_edge":
            io_res = design_butt_edge_coupler(
                core_material=cfg.core_material,
                clad_material=cfg.clad_material,
                top_clad_material=cfg.top_clad_material,
                wavelength_nm=cfg.wavelength_nm,
                polarization=cfg.polarization,
                film_height_um=cfg.film_height_um,
                slab_height_um=cfg.slab_height_um,
                sidewall_deg=cfg.sidewall_deg,
                w_bus_um=cfg.bus_width_um,
                w_tip_um=io_opts.get("w_tip_um"),
                fiber_preset=fib_preset,
            )
            mt = io_res["metrics"]
            if zh:
                msg = (
                    f"🔌 **已完成纯自研二维矢量有限元端面耦合器 (`Butt / Edge Coupler`) 设计（零商业软件依赖）**\n"
                    f"- **对接光纤 / 激光器**：`{io_res['fiber_label']}`（`{cfg.core_material}`，`H = {cfg.film_height_um*1000:.0f} nm`，`{cfg.polarization}` @ `{cfg.wavelength_nm:g} nm`）\n"
                    f"- **最优倒锥尖端宽度 (`W_tip`)**：**`{io_res['w_tip_um']*1000:.0f} nm`** → **总插入损耗 `IL = {mt['total_il_db']:.2f} dB`**（耦合效率 **`{mt['total_efficiency_pct']:.1f}%`**，回波损耗 `{mt['return_loss_db']:.1f} dB`）\n"
                    f"- **光模块封装对准容差**：**`1-dB = (±{mt['tol_1db_x_um']:.2f}, ±{mt['tol_1db_y_um']:.2f}) µm`**｜`3-dB = (±{mt['tol_3db_x_um']:.2f}, ±{mt['tol_3db_y_um']:.2f}) µm`\n"
                    f"- **推荐绝热锥长**：`L_taper ≥ {io_res['recommended_taper_length_um']:.0f} µm`"
                )
            else:
                msg = (
                    f"🔌 **Solved 2D Vector FEM Butt / Edge Coupler (`Zero Commercial Licenses`)**\n"
                    f"- **Target Fiber / Laser**: `{io_res['fiber_label']}` (`{cfg.core_material}`, `H = {cfg.film_height_um*1000:.0f} nm`, `{cfg.polarization}` @ `{cfg.wavelength_nm:g} nm`)\n"
                    f"- **Optimal Tip Width (`W_tip`)**: **`{io_res['w_tip_um']*1000:.0f} nm`** -> **Total `IL = {mt['total_il_db']:.2f} dB`** (**`{mt['total_efficiency_pct']:.1f}%`**, Return Loss `{mt['return_loss_db']:.1f} dB`)\n"
                    f"- **Optical-Module Alignment Tolerance**: **`1-dB = (±{mt['tol_1db_x_um']:.2f}, ±{mt['tol_1db_y_um']:.2f}) µm`** | `3-dB = (±{mt['tol_3db_x_um']:.2f}, ±{mt['tol_3db_y_um']:.2f}) µm`\n"
                    f"- **Recommended Adiabatic Taper Length**: `L_taper ≥ {io_res['recommended_taper_length_um']:.0f} µm`"
                )
        else:
            io_res = design_grating_coupler(
                core_material=cfg.core_material,
                clad_material=cfg.clad_material,
                top_clad_material=cfg.top_clad_material,
                wavelength_nm=cfg.wavelength_nm,
                polarization=cfg.polarization,
                film_height_um=cfg.film_height_um,
                etch_depth_um=io_opts.get("etch_depth_um"),
                fiber_angle_deg=float(io_opts.get("fiber_angle_deg") or 10.0),
                apodized=bool(io_opts.get("apodized", True)),
                focusing=bool(io_opts.get("focusing", True)),
                w_bus_um=cfg.bus_width_um,
                fiber_preset=fib_preset,
            )
            mt = io_res["metrics"]
            sy = io_res["synthesis"]
            if zh:
                msg = (
                    f"🔌 **已完成全波二维频域有限差分 (`2D Full-Wave Maxwell FDFD`) 光栅耦合器仿真（零 Lumerical / RSoft 依赖）**\n"
                    f"- **结构参数**：`{cfg.core_material}`（膜厚 `{io_res['film_height_um']*1000:.0f} nm`，浅刻蚀 `{io_res['etch_depth_um']*1000:.0f} nm`，平均周期 **`Λ = {sy['pitch_mean_um']*1000:.1f} nm`**，占空比 `{sy['ff_start']:.2f}→{sy['ff_end']:.2f}`）\n"
                    f"- **峰值耦合损耗与带宽**：**`IL = {mt['peak_il_db']:.2f} dB`**（效率 **`{mt['peak_efficiency_pct']:.1f}%`** @ `{mt['peak_wavelength_nm']:.1f} nm`）｜**`1-dB / 3-dB BW = {mt['bw_1db_nm']:.1f} / {mt['bw_3db_nm']:.1f} nm`**\n"
                    f"- **方向性与回波反射**：向上辐射 `{mt['upward_directionality_pct']:.1f}%`，基底泄漏 `{mt['substrate_loss_pct']:.1f}%`，波导反射 **`S11 = {mt['s11_db']:.1f} dB`**"
                )
            else:
                msg = (
                    f"🔌 **Solved Native 2D Full-Wave Maxwell `FDFD` Grating Coupler (`Zero Lumerical / RSoft`)**\n"
                    f"- **Synthesized Teeth**: `{cfg.core_material}` (`H = {io_res['film_height_um']*1000:.0f} nm`, Etch `{io_res['etch_depth_um']*1000:.0f} nm`, Mean Pitch **`Λ = {sy['pitch_mean_um']*1000:.1f} nm`**, Fill `{sy['ff_start']:.2f} -> {sy['ff_end']:.2f}`)\n"
                    f"- **Peak Insertion Loss & Bandwidth**: **`IL = {mt['peak_il_db']:.2f} dB`** (**`{mt['peak_efficiency_pct']:.1f}%`** @ `{mt['peak_wavelength_nm']:.1f} nm`) | **`1-dB / 3-dB BW = {mt['bw_1db_nm']:.1f} / {mt['bw_3db_nm']:.1f} nm`**\n"
                    f"- **Directionality & Reflection**: Upward `{mt['upward_directionality_pct']:.1f}%`, Substrate `{mt['substrate_loss_pct']:.1f}%`, Back-Reflection **`S11 = {mt['s11_db']:.1f} dB`**"
                )

        gds_exp = None
        if io_opts.get("export_gds"):
            out_dir = Path(gds_export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            loop_flag = bool(io_opts.get("include_loopback"))
            g_name = f"MicroRing_IO_{ctype}_{cfg.core_material}{'_Loopback' if loop_flag else ''}.gds"
            gds_info = export_io_coupler_gds(
                io_res, out_dir / g_name, ring_config=cfg if loop_flag else None, include_ring_loopback=loop_flag
            )
            gds_exp = {
                "gds_path": gds_info["gds_path"],
                "lyp_path": gds_info["lyp_path"],
                "gds_view": analyze_uploaded_gds(gds_info["gds_path"]),
            }
            msg += (
                f"\n- 💾 **已导出 GDSII 版图**：`{Path(gds_info['gds_path']).name}`"
                if zh
                else f"\n- 💾 **Exported GDSII Mask**: `{Path(gds_info['gds_path']).name}`"
            )

        ic_io = None
        if io_opts.get("export_interconnect"):
            out_dir = Path(gds_export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ic_io = export_io_coupler_interconnect(io_res, out_dir / f"io_interconnect_{ctype}_{cfg.core_material}")
            ic_io["zip_download_url"] = f"/api/download_file?name={ic_io['zip_name']}"
            msg += (
                f"\n- 📦 **已导出 Lumerical INTERCONNECT 端口 S 参数包**：`{ic_io['zip_name']}`"
                if zh
                else f"\n- 📦 **Exported Lumerical INTERCONNECT 2-Port `SPAR`**: `{ic_io['zip_name']}`"
            )

        session["last_io_result"] = io_res
        session["wizard_step"] = 6
        session["ui_view"] = "io"
        chips = (
            [
                {"label": "🔌 切换为端面耦合器 (Butt -> 2.5µm 透镜光纤)", "prompt": "设计端面耦合器对接 2.5 um 透镜光纤"}
                if ctype != "butt_edge"
                else {"label": "🔌 切换为切趾聚焦光栅耦合器 (2D-FDFD)", "prompt": "设计切趾聚焦光栅耦合器"},
                {"label": "🔬 端面直耦 III-V DFB 激光器 (2.4×1.2 µm)", "prompt": "设计端面耦合器对接 DFB 激光器"},
                {"label": "💾 导出 127µm 光纤阵列 + 微环回环测试 .GDS", "prompt": f"导出 {'端面耦合器' if ctype == 'butt_edge' else '聚焦光栅耦合器'} 127um 光纤阵列回环 GDS"},
                {"label": "📦 导出耦合器 INTERCONNECT S 参数包", "prompt": f"导出 {'端面耦合器' if ctype == 'butt_edge' else '光栅耦合器'} INTERCONNECT S 参数"},
            ]
            if zh
            else [
                {"label": "🔌 Switch to Butt Coupler (2.5 µm Lensed Fiber)", "prompt": "Design butt coupler to 2.5 um lensed fiber"}
                if ctype != "butt_edge"
                else {"label": "🔌 Switch to Apodized Focusing GC (2D-FDFD)", "prompt": "Design apodized focusing grating coupler"},
                {"label": "🔬 Butt-Couple III-V DFB Laser (2.4×1.2 µm)", "prompt": "Design butt coupler to DFB laser diode"},
                {"label": "💾 Export 127 µm FA + Ring Loopback .GDS", "prompt": f"Export {'butt coupler' if ctype == 'butt_edge' else 'focusing grating coupler'} 127um fiber array loopback GDS"},
                {"label": "📦 Export Coupler INTERCONNECT SPAR", "prompt": f"Export {'butt coupler' if ctype == 'butt_edge' else 'grating coupler'} INTERCONNECT compact model"},
            ]
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        ret_io = {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "io_result": io_res,
            "ui_view": "io",
            "wizard_step": 6,
            "active_tab": "tab-io",
        }
        if gds_exp is not None:
            ret_io["gds_export"] = gds_exp
        if ic_io is not None:
            ret_io["interconnect_export"] = ic_io
        return ret_io

    # ------------------------------------------------------------------
    # Direct Action 2E: Ultra-High-Q Convergence vs Slow Lumerical FDTD Ring-Down
    # ------------------------------------------------------------------
    if intent["action"] == "high_q_fdtd":
        hq_opts = intent.get("high_q") or {}
        cfg_hq = cfg
        if hq_opts.get("ultra_low_loss") and (cfg_hq.alpha_loss_db_per_cm is None or cfg_hq.alpha_loss_db_per_cm > 0.15):
            cfg_hq = replace(cfg_hq, alpha_loss_db_per_cm=0.08, gap_thru_um=max(cfg_hq.gap_thru_um, 0.36))
            cfg_dict["alpha_loss_db_per_cm"] = 0.08
            cfg_dict["gap_thru_um"] = cfg_hq.gap_thru_um
            session["config"] = cfg_dict

        tsim_ps = float(hq_opts.get("fdtd_sim_time_ps") or 50.0)
        hq_res = analyze_high_q_fdtd_convergence(
            cfg_hq,
            sim_times_ps=[10.0, tsim_ps, 250.0, 1000.0] if tsim_ps not in (10.0, 250.0, 1000.0) else [10.0, 50.0, 250.0, 1000.0],
        )
        traces = hq_res["truncated_fdtd_traces"]
        sel_tr = next(
            (r for r in traces if abs(r["fdtd_sim_time_ps"] - tsim_ps) < 1e-3),
            traces[1],
        )
        cut_exp = None
        if hq_opts.get("export_cutout_gds"):
            out_dir = Path(gds_export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            cut_exp = export_coupler_cutout_gds(
                cfg_hq, out_dir / f"MicroRing_CouplerCutout_{cfg_hq.topology}_{cfg_hq.core_material}.gds"
            )

        if zh:
            msg = (
                f"🚀 **已完成超高 Q 微环谐振腔精确收敛 vs Lumerical 3D-FDTD 截断振铃诊断**\n"
                f"- **精确稳态解 ($M \\to \\infty$)**：**`Q_loaded = {hq_res['Q_loaded']:,.0f}`**（线宽 `FWHM = {hq_res['FWHM_pm']:.3f} pm`，光子寿命 `τ_ph = {hq_res['photon_lifetime_ps']:.1f} ps`，真实消光比 **`ER = {hq_res['true_converged_er_db']:.2f} dB`**）\n"
                f"- **为何 3D-FDTD 无法收敛**：达到 `1e-5` 能量关断阈值需要仿真时间 **`{hq_res['required_fdtd_sim_time_1e5_ns']:.2f} ns`**（足足 **`{hq_res['required_fdtd_roundtrips_1e5']:,.0f}` 圈**往返！）；若在 `{sel_tr['fdtd_sim_time_ps']:.0f} ps` 截断，剩余能量高达 `{sel_tr['residual_energy_shutoff']:.3f}`，表观消光比坍塌至 **`{sel_tr['apparent_er_db']:.2f} dB`**（误差 `{sel_tr['er_error_db']:.1f} dB`！）\n"
                f"- **混合加速方案**：仅对耦合区导出单程切片（仅需 **`{hq_res['coupler_only_fdtd_time_ps']} ps`**，提速 **`{hq_res['time_domain_speedup_factor']:,.0f}×`**），再由 MicroRing Studio 解析闭合无限圈谐振！"
            )
            chips = [
                {"label": "✂️ 导出单程耦合区 FDTD 切片 (.GDS + .lsf)", "prompt": "导出高Q微环耦合区切片 GDS 与 lsf"},
                {"label": "⏱️ 查看 t_sim = 500 ps 截断光谱对比", "prompt": "分析高Q微环 500 ps FDTD 截断光谱"},
                {"label": "⚡ 一键应用临界耦合再测高Q收敛", "prompt": "临界耦合"},
                {"label": "📊 返回微环透射光谱面板", "prompt": "开始仿真"},
            ]
        else:
            msg = (
                f"🚀 **Solved High-Q Infinite-Roundtrip Convergence vs Truncated 3D-FDTD Ring-Down**\n"
                f"- **Exact $M \\to \\infty$ Steady State**: **`Q_loaded = {hq_res['Q_loaded']:,.0f}`** (`FWHM = {hq_res['FWHM_pm']:.3f} pm`, `τ_ph = {hq_res['photon_lifetime_ps']:.1f} ps`, True Converged **`ER = {hq_res['true_converged_er_db']:.2f} dB`**)\n"
                f"- **Why 3D-FDTD Stalls**: Reaching `1e-5` auto-shutoff requires **`{hq_res['required_fdtd_sim_time_1e5_ns']:.2f} ns`** (**`{hq_res['required_fdtd_roundtrips_1e5']:,.0f}` round-trips**!). Stopping FDTD at `{sel_tr['fdtd_sim_time_ps']:.0f} ps` leaves `shutoff = {sel_tr['residual_energy_shutoff']:.3f}` and collapses apparent `ER` to **`{sel_tr['apparent_er_db']:.2f} dB`** (`{sel_tr['er_error_db']:.1f} dB` error + sinc ringing!)\n"
                f"- **Hybrid Speedup**: Export our non-resonant Coupler-Only Cutout (`{hq_res['coupler_only_fdtd_time_ps']} ps` single pass, **`{hq_res['time_domain_speedup_factor']:,.0f}×` faster**) + let MicroRing Studio close the infinite ring analytically!"
            )
            chips = [
                {"label": "✂️ Export Coupler Cutout (.GDS + .lsf)", "prompt": "Export high-Q coupler cutout GDS and lsf"},
                {"label": "⏱️ Compare Truncated FDTD @ 500 ps", "prompt": "Analyze high-Q FDTD convergence at 500 ps"},
                {"label": "⚡ Apply Critical Coupling First", "prompt": "Apply critical coupling"},
                {"label": "📊 Back to MRR Spectrum", "prompt": "Simulate current MRR"},
            ]
        if cut_exp is not None:
            msg += (
                f"\n- ✂️ **已生成单程耦合区切片**：`{Path(cut_exp['gds_path']).name}` + `{Path(cut_exp['lsf_path']).name}`（建议 FDTD 时间 `{cut_exp['recommended_fdtd_time_ps']} ps`，提速 `{hq_res['time_domain_speedup_factor']:,.0f}×`）"
                if zh
                else f"\n- ✂️ **Exported Coupler-Only Cutout**: `{Path(cut_exp['gds_path']).name}` + `{Path(cut_exp['lsf_path']).name}` (`t_sim = {cut_exp['recommended_fdtd_time_ps']} ps`, `{hq_res['time_domain_speedup_factor']:,.0f}×` faster)"
            )

        session["last_high_q_result"] = hq_res
        session["wizard_step"] = 6
        session["ui_view"] = "io"
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        ret_hq = {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "high_q_result": hq_res,
            "ui_view": "io",
            "wizard_step": 6,
            "active_tab": "tab-io",
        }
        if cut_exp is not None:
            ret_hq["gds_export"] = {
                "gds_path": cut_exp["gds_path"],
                "lyp_path": cut_exp["lyp_path"],
                "gds_view": analyze_uploaded_gds(cut_exp["gds_path"]),
            }
        return ret_hq

    # ------------------------------------------------------------------
    # Direct Action 2F: Mach-Zehnder Interferometer (MZI) & Active Modulator
    # ------------------------------------------------------------------
    if intent["action"] == "design_mzi":
        mzi_opts = intent.get("mzi") or {}
        mech_req = str(mzi_opts.get("mechanism") or "auto")
        if mech_req in ("si_pn_depletion", "si_pin_injection") and cfg.core_material != "Si":
            preset = PLATFORM_PRESETS["Si"]
            cfg_dict.update({
                "core_material": "Si",
                "film_height_um": preset["film_height_um"],
                "slab_height_um": 0.09,
                "ring_width_um": 0.48,
                "bus_width_um": 0.48,
            })
            session["config"] = cfg_dict
            cfg = _dict_to_cfg(cfg_dict)

        default_l1 = 1500.0 if mech_req in ("si_pn_depletion", "pockels_eo") or cfg.core_material == "LN_Xcut" else 600.0
        l1_use = float(mzi_opts["arm1_length_um"]) if mzi_opts.get("arm1_length_um") is not None else default_l1
        dl_use = float(mzi_opts["delta_length_um"]) if mzi_opts.get("delta_length_um") is not None else 80.0

        mzi_cfg = MZIDesignConfig(
            core_material=cfg.core_material,
            clad_material=cfg.clad_material,
            top_clad_material=cfg.top_clad_material,
            wavelength_nm=cfg.wavelength_nm,
            polarization=cfg.polarization,
            film_height_um=cfg.film_height_um,
            slab_height_um=0.09 if (mech_req in ("si_pn_depletion", "si_pin_injection") and cfg.slab_height_um <= 0.01) else cfg.slab_height_um,
            sidewall_deg=cfg.sidewall_deg,
            arm1_width_um=cfg.ring_width_um,
            arm1_length_um=l1_use,
            delta_length_um=dl_use,
            target_fsr_nm=intent.get("targets", {}).get("fsr_nm"),
            target_fsr_ghz=intent.get("targets", {}).get("fsr_ghz"),
            splitter_type=str(mzi_opts.get("splitter_type") or "directional_coupler"),
            ramzi_enabled=bool(mzi_opts.get("ramzi_enabled", False)),
            alpha_loss_db_per_cm=cfg.alpha_loss_db_per_cm,
            span_nm=max(float(cfg.span_nm), 20.0),
        )
        mzi_res = simulate_mzi(
            mzi_cfg,
            include_active_tuning=bool(mzi_opts.get("include_active_tuning", True)),
            mechanism=mech_req,
            drive_type=str(mzi_opts.get("drive_type") or "auto"),
            drive_scheme=str(mzi_opts.get("drive_scheme") or "auto"),
        )
        sp = mzi_res["splitter"]
        sp_s = sp["summary"]
        mm = mzi_res["metrics"]
        act = mzi_res.get("active_tuning")

        gds_mzi = None
        if mzi_opts.get("export_gds"):
            out_dir = Path(gds_export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            g_raw = export_mzi_gds(mzi_res, out_dir / f"MicroRing_MZI_{sp['splitter_type']}_{cfg.core_material}.gds")
            gds_mzi = {
                "gds_path": g_raw["gds_path"],
                "lyp_path": g_raw["lyp_path"],
                "saved_path": g_raw["gds_path"],
                "gds_download_url": f"/api/download_file?name={Path(g_raw['gds_path']).name}",
                "lyp_download_url": f"/api/download_file?name={Path(g_raw['lyp_path']).name}",
                "gds_view": analyze_uploaded_gds(g_raw["gds_path"]),
            }

        ic_mzi = None
        if mzi_opts.get("export_interconnect"):
            out_dir = Path(gds_export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ic_mzi = export_mzi_interconnect_bundle(mzi_res, out_dir / f"mzi_ic_{sp['splitter_type']}_{cfg.core_material}")
            ic_mzi["zip_download_url"] = f"/api/download_file?name={Path(ic_mzi['zip_path']).name}"

        if zh:
            msg = (
                f"🔀 **已完成无 3D-FDTD 马赫-曾德尔干涉仪 (`MZI`) 全本征模求解**（`{cfg.core_material}` @ `{cfg.wavelength_nm:g} nm`）\n"
                f"- **分束/合束器 (`{sp['name_zh']}`)**：分光比 **`{sp_s['split_bar_pct_at_wl0']:.1f}% : {sp_s['split_cross_pct_at_wl0']:.1f}%`**，附加损耗 **`IL = {sp_s['excess_loss_db']:.2f} dB`**，不平衡度 `{sp_s['imbalance_at_wl0_db']:+.2f} dB`\n"
                f"- **干涉臂与无源光谱**：`L_1 = {mzi_res['arm_eigenmodes']['arm1']['length_um']:.0f} µm`，`ΔL = {mzi_res['arm_eigenmodes']['delta_length_um']:.1f} µm` → **`FSR = {mm['FSR_nm'] or '宽带'} nm`**，无源消光比 **`ER = {mm['ER_bar_passive_dB']:.1f} dB`**"
            )
            if act:
                fom_str = (
                    f"`V_π·L = {act['v_pi_l_vcm']} V·cm`（`V_π = {act['v_pi_v']} V`）"
                    if act.get("v_pi_l_vcm") is not None
                    else f"`P_π = {act['p_pi_mw']} mW`"
                )
                msg += (
                    f"\n- **有源调制性能 (`{act['mechanism_name_zh']}`, `{act['drive_scheme']}`)**：{fom_str}，"
                    f"非谐振 **3-dB 带宽 = `{act['bandwidth_3db']} {act['bandwidth_unit']}`**，动态消光比 **`ER = {act['dynamic_er_bar_db']:.1f} dB`**"
                )
            chips = [
                {"label": "🔀 切换为 2x2 DC 定向耦合器 MZI", "prompt": "设计 2x2 DC 马赫曾德尔调制器"},
                {"label": "🔀 切换为 1x2 绝热 Y 分支 MZI", "prompt": "设计 Y 分支马赫曾德尔调制器"},
                {"label": "⚡ X-cut TFLN 铌酸锂推挽电光 MZI", "prompt": "600 nm X切铌酸锂 Y分支推挽电光 MZI 臂长 5 mm"},
                {"label": "💾 导出 MZI .GDS + INTERCONNECT 模型包", "prompt": f"导出 {sp['splitter_type']} MZI GDS 版图与 INTERCONNECT 模型"},
            ]
        else:
            msg = (
                f"🔀 **Solved Mach-Zehnder Interferometer (`MZI`) via 2D Vector FEM + EME (Zero 3D FDTD)** on **`{cfg.core_material}`** (`{cfg.wavelength_nm:g} nm`)\n"
                f"- **Splitter / Combiner (`{sp['name_en']}`)**: Split **`{sp_s['split_bar_pct_at_wl0']:.1f}% : {sp_s['split_cross_pct_at_wl0']:.1f}%`**, Excess Loss **`IL = {sp_s['excess_loss_db']:.2f} dB`**, Imbalance `{sp_s['imbalance_at_wl0_db']:+.2f} dB`\n"
                f"- **Interferometer Arms & Spectrum**: `L_1 = {mzi_res['arm_eigenmodes']['arm1']['length_um']:.0f} µm`, `ΔL = {mzi_res['arm_eigenmodes']['delta_length_um']:.1f} µm` → **`FSR = {mm['FSR_nm'] or 'Broadband'} nm`**, Passive **`ER = {mm['ER_bar_passive_dB']:.1f} dB`**"
            )
            if act:
                fom_str = (
                    f"**`V_π·L = {act['v_pi_l_vcm']} V·cm`** (`V_π = {act['v_pi_v']} V`)"
                    if act.get("v_pi_l_vcm") is not None
                    else f"**`P_π = {act['p_pi_mw']} mW`**"
                )
                msg += (
                    f"\n- **Active Tuning (`{act['mechanism_name_en']}`, `{act['drive_scheme']}`)**: {fom_str}, "
                    f"Non-Cavity **3-dB BW = `{act['bandwidth_3db']} {act['bandwidth_unit']}`**, Dynamic **`ER = {act['dynamic_er_bar_db']:.1f} dB`**"
                )
            chips = [
                {"label": "🔀 Switch to 2x2 DC MZI", "prompt": "Design a 2x2 directional coupler MZI modulator"},
                {"label": "🔀 Switch to 1x2 Y-Branch MZI", "prompt": "Design a 1x2 adiabatic Y-branch MZI modulator"},
                {"label": "⚡ X-Cut TFLN Push-Pull EO MZI (5 mm)", "prompt": "Design a 600 nm X-cut TFLN Y-branch push-pull EO MZI arm length 5 mm"},
                {"label": "💾 Export MZI .GDS + INTERCONNECT (.ZIP)", "prompt": f"Export {sp['splitter_type']} MZI GDS and INTERCONNECT compact model"},
            ]

        session["last_mzi_result"] = mzi_res
        session["wizard_step"] = 6
        session["ui_view"] = "mzi"
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        ret_mzi: dict = {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "mzi_result": mzi_res,
            "ui_view": "mzi",
            "wizard_step": 6,
            "active_tab": "tab-mzi",
        }
        if gds_mzi is not None:
            ret_mzi["gds_export"] = gds_mzi
        if ic_mzi is not None:
            ret_mzi["interconnect_export"] = ic_mzi
        return ret_mzi

    # ------------------------------------------------------------------
    # Guided Step 3 -> Step 4 Pre-Check: Material chosen, ask to confirm Wafer Stack
    # ------------------------------------------------------------------
    if (
        "core_material" in confirmed
        and "wafer_stack" not in confirmed
        and session.get("last_width_sweep") is None
        and "resonator_size" not in confirmed
        and "topology" not in confirmed
        and intent["action"] not in (
            "sweep_width",
            "simulate_mrr",
            "sweep_mrr",
            "fab_tolerance",
            "export_fab_bias_gds",
            "active_tuning",
            "export_interconnect",
            "io_coupler",
            "high_q_fdtd",
        )
        and "uploaded_gds" not in confirmed
    ):
        session["stage"] = "ask_stack"
        session["wizard_step"] = 4
        session["ui_view"] = "overview"
        h_nm = round(cfg.film_height_um * 1000)
        s_nm = round(cfg.slab_height_um * 1000)
        rib_sug_nm = max(40, round(0.35 * h_nm / 10) * 10)
        if zh:
            etch_desc = "全刻蚀条形波导 (`0 nm` 平板层)" if s_nm == 0 else f"浅刻蚀脊形波导 (`{s_nm} nm` 平板层)"
            msg = (
                f"**第 4 / 6 步 — 确认晶圆叠层与偏振态** 🧱\n\n"
                f"针对 **`{cfg.core_material}`**（`@ {cfg.wavelength_nm:g} nm`），已为您预填标准工艺截面：\n"
                f"- **芯层总膜厚 (`H_film`)**：`{h_nm} nm` — {etch_desc}\n"
                f"- **包层与基底**：上包层 `{cfg.top_clad_material}` / 下包层 `{cfg.clad_material}`（侧壁角 `{cfg.sidewall_deg:.0f}°`）\n"
                f"- **偏振模式**：`{cfg.polarization}`\n\n"
                "是否符合您的晶圆规格？点击**确认**即可求解二维矢量本征模与单模截止宽度（也可直接输入如 `膜厚 500 nm, 空气包层`）："
            )
            chips = [
                {"label": "✅ 确认叠层并求解二维模式", "prompt": "确认叠层并求解模式"},
                {"label": "🌤️ 改为空气上包层 + 求解模式", "prompt": "空气包层, 确认叠层"},
                {
                    "label": f"🧱 改为脊形波导 ({rib_sug_nm} nm 平板) + 求解" if s_nm == 0 else "🔪 改为全刻蚀条形波导 + 求解",
                    "prompt": f"{rib_sug_nm} nm 平板, 确认叠层" if s_nm == 0 else "条形波导, 确认叠层",
                },
                {"label": "🔄 切换为 TM0 偏振 + 求解模式", "prompt": "TM0, 确认叠层"},
            ]
        else:
            etch_desc = "Full-etch strip (`0 nm` slab)" if s_nm == 0 else f"Partial-etch rib (`{s_nm} nm` slab)"
            msg = (
                f"**Step 4 of 6 — Confirm Wafer Stack & Polarization** 🧱\n\n"
                f"For **`{cfg.core_material}`** at **`{cfg.wavelength_nm:g} nm`**, I pre-filled this standard cross-section:\n"
                f"- **Film Thickness (`H_film`)**: `{h_nm} nm` — {etch_desc}\n"
                f"- **Cladding**: `{cfg.top_clad_material}` top / `{cfg.clad_material}` bottom (`{cfg.sidewall_deg:.0f}°` sidewall)\n"
                f"- **Polarization**: `{cfg.polarization}`\n\n"
                "Does this match your wafer? Click **Confirm** to solve 2D Maxwell eigenmodes, or type custom thickness/cladding (e.g. `H_film = 500 nm, air clad`):"
            )
            chips = [
                {"label": "✅ Confirm Stack & Solve 2D Modes", "prompt": "Confirm stack and check mode"},
                {"label": "🌤️ Air Top Cladding + Solve Modes", "prompt": "Air top cladding, confirm stack"},
                {
                    "label": f"🧱 Rib ({rib_sug_nm} nm slab) + Solve" if s_nm == 0 else "🔪 Full-Etch Strip (0 nm slab) + Solve",
                    "prompt": f"{rib_sug_nm} nm slab, confirm stack" if s_nm == 0 else "Full-etch strip, confirm stack",
                },
                {"label": "🔄 Switch to TM0 + Solve Modes", "prompt": "TM0 polarization, confirm stack"},
            ]
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "ui_view": "overview",
            "wizard_step": 4,
            "active_tab": "tab-mrr",
        }

    # ------------------------------------------------------------------
    # Step 4 Action: Waveguide Width Dispersion & Single-Mode Check
    # ------------------------------------------------------------------
    preset_key = "ScAlN_20" if cfg.core_material.startswith("ScAlN") else cfg.core_material
    w_min_def, w_max_def = scaled_w_sweep or PLATFORM_PRESETS.get(preset_key, {"w_sweep": (0.45, 1.80)})["w_sweep"]

    should_run_width_sweep = (
        intent["action"] == "sweep_width"
        or (
            "core_material" in confirmed
            and session.get("last_width_sweep") is None
            and "resonator_size" not in confirmed
            and "topology" not in confirmed
            and intent["action"] != "simulate_mrr"
            and "uploaded_gds" not in confirmed
        )
    )

    if should_run_width_sweep:
        sw_opts = intent.get("sweep") or {}
        w_min = float(sw_opts.get("w_min_um", w_min_def))
        w_max = float(sw_opts.get("w_max_um", w_max_def))
        w_res = run_waveguide_width_sweep(
            core_material=cfg.core_material,
            clad_material=cfg.clad_material,
            top_clad_material=cfg.top_clad_material,
            h_film_um=cfg.film_height_um,
            h_slab_um=cfg.slab_height_um,
            sidewall_deg=cfg.sidewall_deg,
            wavelength_nm=cfg.wavelength_nm,
            w_min_um=w_min,
            w_max_um=w_max,
            num_points=7,
            polarization=cfg.polarization,
            sc_fraction=cfg.sc_fraction,
        )
        if "explicit_widths" not in confirmed:
            cfg_dict["bus_width_um"] = w_res["recommended_bus_width_um"]
            cfg_dict["ring_width_um"] = w_res["recommended_ring_width_um"]
            session["config"] = cfg_dict
            cfg = _dict_to_cfg(cfg_dict)

        idx_mid = int(np.argmin(np.abs(np.asarray(w_res["widths_um"]) - cfg.ring_width_um)))
        ng_est = float(w_res["ng_fundamental"][idx_mid])
        neff_est = float(w_res["neff_fundamental"][idx_mid])
        session["last_width_sweep"] = {**w_res, "ng_at_rec": ng_est, "neff_at_rec": neff_est}
        session["stage"] = "mode_confirmed"
        session["wizard_step"] = 5
        session["ui_view"] = "mode"

        def_loss = float(
            cfg.alpha_loss_db_per_cm
            if cfg.alpha_loss_db_per_cm is not None
            else MATERIAL_CATALOG.get(cfg.core_material, {}).get("typical_loss_db_cm", 1.0)
        )
        loss_opts = PLATFORM_PRESETS.get(preset_key, {}).get("loss_choices", [0.2, 0.5, 1.0, 2.0])

        if zh:
            msg = (
                f"✅ **二维矢量本征模与单模截止计算完成**（`{cfg.core_material}`，`H = {cfg.film_height_um*1000:.0f} nm`，`{cfg.polarization}` @ `{cfg.wavelength_nm:g} nm`）：\n"
                f"- **单模最大波导宽度 (`TE1`/`TM1` 截止)**：`W_SM,max = {w_res['single_mode_max_width_um']:.3f} µm`\n"
                f"- **推荐波导宽度**：总线 `W_bus = {cfg.bus_width_um:.3f} µm`，环宽 `W_ring = {cfg.ring_width_um:.3f} µm`（`n_eff = {neff_est:.4f}`，`n_g = {ng_est:.4f}`）\n\n"
                f"**第 5 / 6 步 — 波导传播损耗 (`dB/cm`)** 📉\n"
                f"您的波导传播损耗 `α [dB/cm]` 大约是多少？（临界耦合间距/包角直接取决于波导损耗！您也可直接输入如 `损耗 0.8 dB/cm` 或 `Q_int = 300k`）："
            )
            chips = [
                {"label": f"α = {loss_opts[0]:g} dB/cm (低损耗)", "prompt": f"传播损耗 = {loss_opts[0]:g} dB/cm"},
                {"label": f"α = {loss_opts[1]:g} dB/cm (典型值)", "prompt": f"传播损耗 = {loss_opts[1]:g} dB/cm"},
                {"label": f"α = {loss_opts[2]:g} dB/cm", "prompt": f"传播损耗 = {loss_opts[2]:g} dB/cm"},
                {"label": f"α = {loss_opts[3]:g} dB/cm", "prompt": f"传播损耗 = {loss_opts[3]:g} dB/cm"},
                {"label": "📈 先从实测光谱 .CSV 提取损耗", "prompt": "从实测光谱提取波导损耗"},
            ]
        else:
            msg = (
                f"✅ **2D Maxwell Mode & Single-Mode Cutoff Solved** (`{cfg.core_material}`, `H = {cfg.film_height_um*1000:.0f} nm`, `{cfg.polarization}` @ `{cfg.wavelength_nm:g} nm`):\n"
                f"- **Single-Mode Cutoff (`TE1`/`TM1` onset)**: `W_SM,max = {w_res['single_mode_max_width_um']:.3f} µm`\n"
                f"- **Selected Widths**: `W_bus = {cfg.bus_width_um:.3f} µm` & `W_ring = {cfg.ring_width_um:.3f} µm` (`n_eff = {neff_est:.4f}`, `n_g = {ng_est:.4f}`)\n\n"
                f"**Step 5 of 6 — Waveguide Propagation Loss (`dB/cm`)** 📉\n"
                f"What propagation loss `α [dB/cm]` do you know for your waveguide (or type `Q_int = ...`)?\n"
                f"*(Critical-coupling gap/angle depends directly on loss — or pick a topology/FSR directly to use default `{def_loss:g} dB/cm`.)*"
            )
            chips = [
                {"label": f"α = {loss_opts[0]:g} dB/cm (Low Loss)", "prompt": f"Propagation loss = {loss_opts[0]:g} dB/cm"},
                {"label": f"α = {loss_opts[1]:g} dB/cm (Typical)", "prompt": f"Propagation loss = {loss_opts[1]:g} dB/cm"},
                {"label": f"α = {loss_opts[2]:g} dB/cm", "prompt": f"Propagation loss = {loss_opts[2]:g} dB/cm"},
                {"label": f"α = {loss_opts[3]:g} dB/cm", "prompt": f"Propagation loss = {loss_opts[3]:g} dB/cm"},
                {"label": "📈 Extract Loss from Lab .CSV First", "prompt": "Extract waveguide loss from measured spectrum"},
            ]
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "width_sweep_result": w_res,
            "ui_view": "mode",
            "wizard_step": 5,
            "active_tab": "tab-mode",
        }

    # ------------------------------------------------------------------
    # Guided Step 5 -> Step 6 Question: Propagation Loss confirmed, ask Ring Topology & FSR
    # ------------------------------------------------------------------
    if (
        "propagation_loss" in confirmed
        and "resonator_size" not in confirmed
        and "topology" not in confirmed
        and intent["action"] not in ("simulate_mrr", "sweep_mrr")
        and session.get("stage") in ("mode_confirmed", "loss_confirmed")
    ):
        session["stage"] = "loss_confirmed"
        session["wizard_step"] = 5
        session["ui_view"] = "mode"
        ng_est = (session.get("last_width_sweep") or {}).get("ng_at_rec") or 2.15
        alpha_now = float(
            cfg.alpha_loss_db_per_cm
            if cfg.alpha_loss_db_per_cm is not None
            else MATERIAL_CATALOG.get(cfg.core_material, {}).get("typical_loss_db_cm", 1.0)
        )
        q_int_est = loss_db_per_cm_to_q_int(alpha_now, float(ng_est), cfg.wavelength_nm)
        msg = (
            f"📌 **已锁定传播损耗**：**`α = {alpha_now:.3g} dB/cm`** "
            f"⇒ **本征品质因数 `Q_int ≈ {q_int_est:,.0f}`**（`n_g = {ng_est:.4f}` @ `{cfg.wavelength_nm:g} nm`）。\n\n"
            "**最后一步 — 选择微环拓扑与目标自由光谱范围 (`FSR` 或半径 `R`)** 🎯\n"
            "请选择您需要的微环架构与目标 `FSR`（也可直接输入如 `R = 35 um, 滑轮耦合`）："
            if zh
            else f"📌 **Propagation Loss Set**: **`α = {alpha_now:.3g} dB/cm`** "
            f"⇒ **Intrinsic `Q_int ≈ {q_int_est:,.0f}`** (`n_g = {ng_est:.4f}` @ `{cfg.wavelength_nm:g} nm`).\n\n"
            "**Final Design Step — Ring Topology & Target FSR (or Radius)** 🎯\n"
            "Which resonator topology and Free Spectral Range (`FSR`) or radius (`R`) do you need?"
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": _get_topology_fsr_chips(cfg.wavelength_nm, zh),
            "ui_view": "mode",
            "wizard_step": 5,
            "active_tab": "tab-mode",
        }

    # ------------------------------------------------------------------
    # Direct Action 4: MRR Parameter Sweep
    # ------------------------------------------------------------------
    if intent["action"] == "sweep_mrr":
        sw = intent["sweep"]
        p_name = sw["param_name"]
        rng = sw.get("range")
        if rng:
            vals = np.linspace(rng[0], rng[1], 7).tolist()
        else:
            if p_name == "alpha_loss_db_per_cm":
                cur_l = float(
                    cfg.alpha_loss_db_per_cm
                    if cfg.alpha_loss_db_per_cm is not None
                    else MATERIAL_CATALOG.get(cfg.core_material, {}).get("typical_loss_db_cm", 1.0)
                )
                vals = np.linspace(max(0.05, cur_l * 0.25), max(cur_l * 2.5, 2.0), 7).tolist()
            else:
                cur = float(getattr(cfg, p_name))
                if p_name == "gap_thru_um":
                    vals = np.linspace(max(0.10, cur - 0.10), cur + 0.12, 7).tolist()
                elif p_name == "pulley_angle_thru_deg":
                    vals = np.linspace(max(3.0, cur * 0.35), min(120.0, max(cur * 1.8, 35.0)), 7).tolist()
                elif p_name == "straight_length_um":
                    vals = np.linspace(2.0, max(cur * 2.0, 30.0), 7).tolist()
                elif p_name == "ring_radius_um":
                    vals = np.linspace(max(10.0, cur * 0.6), cur * 1.5, 6).tolist()
                elif p_name == "delta_temp_k":
                    vals = np.linspace(0.0, 50.0, 6).tolist()
                else:
                    vals = np.linspace(-15.0, 15.0, 7).tolist()

        sw_res = run_mrr_parameter_sweep(cfg, p_name, vals)
        best_i = sw_res["best_critical_coupling_index"]
        best_v = sw_res["values"][best_i]
        best_er = sw_res["ER_thru_dB"][best_i]
        best_ql = sw_res["Q_loaded"][best_i]
        session["wizard_step"] = 6
        session["ui_view"] = "mrr"
        if zh:
            msg = (
                f"📊 **已完成 {len(vals)} 点 `{p_name}` 参数扫描**（`{vals[0]:.3g}` → `{vals[-1]:.3g} {sw_res['unit']}`）：\n"
                f"- **最大消光比工作点**：`{p_name} = {best_v:.4g} {sw_res['unit']}` → **`ER = {best_er:.1f} dB`**，`Q_loaded = {best_ql:,.0f}`\n"
                + (
                    f"- **对应本征 Q 范围**：`Q_int = {sw_res['Q_int'][0]:,.0f}` → `{sw_res['Q_int'][-1]:,.0f}`\n"
                    if p_name == "alpha_loss_db_per_cm" and "Q_int" in sw_res
                    else f"- **谐振波长漂移**：`{sw_res['resonance_shift_pm'][0]:+.1f} pm` → `{sw_res['resonance_shift_pm'][-1]:+.1f} pm`\n"
                )
            )
            set_prompt = (
                f"传播损耗 = {best_v:.4g} dB/cm, 开始仿真"
                if p_name == "alpha_loss_db_per_cm"
                else f"Set {p_name.replace('_thru_um','').replace('_thru_deg',' angle')} = {best_v:.4g} {sw_res['unit']} and simulate"
            )
            chips = [
                {"label": f"采用 {best_v:.3g} {sw_res['unit']} 并仿真", "prompt": set_prompt},
                {"label": "💾 导出单器件 .GDS", "prompt": "导出单个 GDS 版图"},
                {"label": "🗂️ 导出 5 环阵列 .GDS", "prompt": "导出阵列 GDS 版图"},
            ]
        else:
            msg = (
                f"📊 **Completed {len(vals)}-Point `{p_name}` Sweep** (`{vals[0]:.3g}` → `{vals[-1]:.3g} {sw_res['unit']}`):\n"
                f"- **Peak Extinction Point**: `{p_name} = {best_v:.4g} {sw_res['unit']}` → **`ER = {best_er:.1f} dB`**, `Q_loaded = {best_ql:,.0f}`\n"
                + (
                    f"- **Intrinsic Q Range Across Loss**: `Q_int = {sw_res['Q_int'][0]:,.0f}` → `{sw_res['Q_int'][-1]:,.0f}`\n"
                    if p_name == "alpha_loss_db_per_cm" and "Q_int" in sw_res
                    else f"- **Resonance Shift**: `{sw_res['resonance_shift_pm'][0]:+.1f} pm` → `{sw_res['resonance_shift_pm'][-1]:+.1f} pm`\n"
                )
            )
            set_prompt = (
                f"Propagation loss = {best_v:.4g} dB/cm, simulate"
                if p_name == "alpha_loss_db_per_cm"
                else f"Set {p_name.replace('_thru_um','').replace('_thru_deg',' angle')} = {best_v:.4g} {sw_res['unit']} and simulate"
            )
            chips = [
                {"label": f"Use {best_v:.3g} {sw_res['unit']} & Simulate", "prompt": set_prompt},
                {"label": "💾 Export Single .GDS", "prompt": "Export single GDS layout"},
                {"label": "🗂️ Export 5-Ring Sweep Array .GDS", "prompt": "Export 5-ring parameter sweep array GDS"},
            ]
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": chips,
            "mrr_sweep_result": sw_res,
            "ui_view": "mrr",
            "wizard_step": 6,
            "active_tab": "tab-mrr",
        }

    # ------------------------------------------------------------------
    # Fallback: If material still unknown, guide user into Step 2 (Waveband)
    # ------------------------------------------------------------------
    if "core_material" not in confirmed:
        session["wizard_step"] = 2
        session["ui_view"] = "overview"
        msg = (
            "让我们**一步一步**来完成设计！\n\n"
            "**第 2 / 6 步 — 选择工作波长波段**：您的器件工作在哪个波段？（可点击下方按钮、上传 `.GDS` 版图，或从实测 `.CSV` 光谱提取损耗。）"
            if zh
            else "Let's take it **one simple step at a time**!\n\n"
            "**Step 2 of 6 — Target Wavelength Band**: Which optical wavelength band does your device use? "
            "*(Click a band below, upload a `.GDS`, or extract `dB/cm` loss from a lab `.CSV` spectrum.)*"
        )
        session["history"] = (history + [{"role": "assistant", "content": msg}])[-16:]
        return {
            "session": session,
            "assistant_message": msg,
            "chips": _get_waveband_chips(zh),
            "ui_view": "overview",
            "wizard_step": 2,
            "active_tab": "tab-mrr",
        }

    # ------------------------------------------------------------------
    # Stage 6: Full 2D Eigenmode + Spatial CMT + Broadband MRR Synthesis
    # ------------------------------------------------------------------
    if (session.get("target_fsr_ghz") or session.get("target_fsr_nm")) and "ring_radius_um" not in intent["fields"]:
        ng_est = (session.get("last_width_sweep") or {}).get("ng_at_rec")
        if not ng_est:
            probe_res = simulate_mrr_topology(replace(cfg, num_spectrum_points=400))
            ng_est = float(probe_res["metrics"]["ng_ring"])
        cfg_dict["ring_radius_um"] = _solve_radius_for_target_fsr(
            cfg, session.get("target_fsr_ghz"), session.get("target_fsr_nm"), ng_est
        )
        fsr_nm_approx = (cfg.wavelength_nm ** 2) * 1e-3 / (2.0 * math.pi * cfg_dict["ring_radius_um"] * ng_est)
        cfg_dict["span_nm"] = round(float(np.clip(2.4 * fsr_nm_approx, 2.0, 90.0)), 2)
        session["config"] = cfg_dict
        cfg = _dict_to_cfg(cfg_dict)
        session["target_fsr_ghz"] = None
        session["target_fsr_nm"] = None

    mrr_res = simulate_mrr_topology(cfg)
    m = mrr_res["metrics"]
    cc = mrr_res["critical_coupling"]
    session["last_mrr_metrics"] = {"metrics": m, "critical_coupling": cc}
    session["stage"] = "mrr_synthesized"
    session["wizard_step"] = 6
    session["ui_view"] = "mrr"

    if zh:
        summary = (
            f"🎯 **已完成 `{TOPOLOGY_CATALOG[cfg.topology]['name']}` 微环综合**（**`{cfg.core_material}`**，"
            f"`H = {cfg.film_height_um*1000:.0f} nm`，`W_ring = {cfg.ring_width_um:.2f} µm`，`R = {cfg.ring_radius_um:.2f} µm`，`{cfg.polarization}` @ `{cfg.wavelength_nm:g} nm`）：\n"
            f"- **波导传播损耗与本征 Q**：**`α = {m['alpha_loss_db_per_cm']:.2f} dB/cm`** ⇔ **`Q_int = {m['Q_int']:,.0f}`**（`n_eff = {m['neff_ring']:.4f}`，`n_g = {m['ng_ring']:.4f}`）\n"
            f"- **自由光谱范围 (`FSR`)**：**`{m['FSR_GHz']:.2f} GHz`**（`{m['FSR_nm']:.3f} nm`）\n"
            f"- **有载 Q 与耦合状态**：`Q_loaded = {m['Q_loaded']:,.0f}` | `{cc['regime']}`（消光比 `ER = {m['ER_thru_dB']:.1f} dB`）\n"
            f"- **对应 `{m['alpha_loss_db_per_cm']:.2f} dB/cm` 的临界耦合目标**：**`{cc['parameter']} = {cc['recommended_value']} {cc['unit']}`**"
        )
    else:
        summary = (
            f"🎯 **Synthesized `{TOPOLOGY_CATALOG[cfg.topology]['name']}`** on **`{cfg.core_material}`** "
            f"(`H = {cfg.film_height_um*1000:.0f} nm`, `W_ring = {cfg.ring_width_um:.2f} µm`, `R = {cfg.ring_radius_um:.2f} µm`, `{cfg.polarization}` @ `{cfg.wavelength_nm:g} nm`):\n"
            f"- **Propagation Loss & Intrinsic Q**: **`α = {m['alpha_loss_db_per_cm']:.2f} dB/cm`** ⇔ **`Q_int = {m['Q_int']:,.0f}`** (`n_eff = {m['neff_ring']:.4f}`, `n_g = {m['ng_ring']:.4f}`)\n"
            f"- **Free Spectral Range**: **`{m['FSR_GHz']:.2f} GHz`** (`{m['FSR_nm']:.3f} nm`)\n"
            f"- **Loaded Q & Coupling**: `Q_loaded = {m['Q_loaded']:,.0f}` | `{cc['regime']}` (`ER = {m['ER_thru_dB']:.1f} dB`)\n"
            f"- **Critical-Coupling Target (for `{m['alpha_loss_db_per_cm']:.2f} dB/cm`)**: **`{cc['parameter']} = {cc['recommended_value']} {cc['unit']}`**"
        )
    if mrr_res.get("wavelength_warnings"):
        summary += ("\n\n⚠️ **材料与波段兼容性提示**：\n" if zh else "\n\n⚠️ **Material / Wavelength Notice**:\n") + "\n".join(
            f"- {w}" for w in mrr_res["wavelength_warnings"]
        )

    chips = (
        [
            {"label": f"⚡ 一键应用临界耦合 ({cc['recommended_value']} {cc['unit']})", "prompt": "临界耦合"},
            {"label": "🔌 光栅耦合器 (GC) / 端面耦合器 (Butt)", "prompt": "设计切趾聚焦光栅耦合器"},
            {"label": "🚀 高Q收敛 vs FDTD 振铃诊断", "prompt": "分析高Q微环 FDTD 收敛与振铃"},
            {"label": "🎛️ 有源调谐光谱 (V / I / ΔT / E) + INTERCONNECT", "prompt": "计算有源调谐效率与不同电压电流光谱"},
            {"label": "🧪 正/负胶过曝与欠曝容差分析", "prompt": "正胶 ±15% 过曝与欠曝工艺误差分析"},
            {"label": "💾 导出单器件 .GDS + .LYP", "prompt": "导出单个 GDS 版图"},
            {"label": "🗂️ 导出 5 环扫描阵列 .GDS", "prompt": "导出阵列 GDS 版图"},
        ]
        if zh
        else [
            {"label": f"⚡ Apply Critical Coupling ({cc['recommended_value']} {cc['unit']})", "prompt": "Apply critical coupling"},
            {"label": "🔌 Butt Coupler & 2D-FDFD Grating Coupler", "prompt": "Design apodized focusing grating coupler"},
            {"label": "🚀 High-Q Convergence vs FDTD Ring-Down", "prompt": "Analyze high-Q FDTD convergence and ring-down"},
            {"label": "🎛️ Active Tuning (V / I / ΔT / E) + INTERCONNECT", "prompt": "Calculate tuning efficiency and multi-bias spectra"},
            {"label": "🧪 Fab Error (Over/Under-Dose PR)", "prompt": "Analyze positive PR ±15% over-dose and under-dose"},
            {"label": "💾 Export Single .GDS + .LYP", "prompt": "Export single GDS layout"},
            {"label": "🗂️ Export 5-Ring Sweep Array .GDS", "prompt": "Export 5-ring parameter sweep array GDS"},
        ]
    )
    session["history"] = (history + [{"role": "assistant", "content": summary}])[-16:]
    return {
        "session": session,
        "assistant_message": summary,
        "chips": chips,
        "mrr_result": mrr_res,
        "ui_view": "mrr",
        "wizard_step": 6,
        "active_tab": "tab-mrr",
    }
