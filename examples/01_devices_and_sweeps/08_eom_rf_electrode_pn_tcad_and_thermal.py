"""Cookbook 08: License-Free 2D-FEM EOM RF Electrodes, Si PN Doping TCAD & Thermal FEM.

Demonstrates all three multiphysics & cleanroom layout engines in MicroRing Studio v3.7.1
without Ansys HFSS, Lumerical CHARGE/HEAT/MODE, or COMSOL Multiphysics:
  1. Push-Pull GSG CPW vs Velocity-Matched Slow-Wave T-Rail CPW on X-Cut TFLN (.GDS + .S2P + TWMOD)
  2. 2D Nonlinear Poisson-Boltzmann Silicon Lateral PN vs L-Shaped PN TCAD (.GDS + OMM)
  3. 2D Steady-State + Crank-Nicolson Transient Thermal FEM across 4 Heater Architectures (.GDS + CSV)
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from microring import (
    EOMElectrodeConfig,
    PNDopingTCADConfig,
    ThermalSimConfig,
    analyze_eo_wafer_orientation,
    design_eom_electrode,
    export_eom_electrode_gds,
    export_eom_rf_interconnect_bundle,
    export_pn_doping_gds,
    export_tcad_data_bundle,
    export_thermal_shifter_gds,
    simulate_pn_doping_tcad,
    simulate_thermal_phase_shifter,
)


def main() -> None:
    out_dir = REPO_ROOT / "outputs" / "cookbook_08_eom_pn_thermal"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 0. 3D Crystal-Tensor EO Wafer Orientation & Azimuth Finder ===")
    for mat, wcut, pol in (
        ("LN_Xcut", "x_cut_ln", "TE0"),
        ("LT_Xcut", "x_cut_lt", "TE0"),
        ("BTO", "a_cut_bto_si", "TE0"),
        ("PZT", "inplane_poled_pzt", "TE0"),
        ("LN_Zcut", "z_cut_ln", "TM0"),
        ("ScAlN_20", "c_plane_6mm", "TM0"),
        ("3C-SiC", "cubic_001_43m", "TE0"),
    ):
        wo = analyze_eo_wafer_orientation(core_material=mat, wafer_cut=wcut, polarization=pol)
        wm = wo["metrics"]
        print(
            f"  [{mat:8s} | {wcut:17s} | {pol}] phi_opt={wm['optimal_azimuth_deg']:5.1f} deg | "
            f"r_eff={wm['r_eff_optimal_pm_per_v']:+7.2f} pm/V | V_pi*L={wm['V_pi_L_optimal_Vcm']:5.2f} V*cm | "
            f"Δn_biref={wm['birefringence_span_dn']:.4f} | "
            f"Ring Unsplit={wm['ring_unsplit_efficiency_pct']:5.1f}% vs Split={wm['ring_split_pushpull_efficiency_pct']:5.1f}%"
        )

    print("\n=== 1. EOM RF Traveling-Wave & Slow-Wave T-Rail Electrode Synthesis ===")
    for top, mat, sub, hf, hs, wwg, gap, alen in (
        ("gsg_cpw", "LN_Xcut", "hr_si", 0.60, 0.30, 1.10, 5.0, 5000.0),
        ("t_rail_cpw", "LN_Xcut", "quartz", 0.60, 0.30, 1.10, 5.0, 5000.0),
        ("gsg_cpw", "LT_Xcut", "hr_si", 0.60, 0.30, 1.15, 5.0, 5000.0),
        ("gsg_cpw", "BTO", "hr_si", 0.35, 0.12, 0.90, 4.0, 1500.0),
        ("gsg_cpw", "PZT", "hr_si", 0.32, 0.10, 0.85, 4.0, 1500.0),
    ):
        rf = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology=top,
                core_material=mat,
                substrate_type=sub,
                wavelength_nm=1550.0,
                film_height_um=hf,
                slab_height_um=hs,
                waveguide_width_um=wwg,
                electrode_gap_um=gap,
                active_length_um=alen,
                probe_pitch_um=150.0,
                run_gap_sweep=False,
            )
        )
        dm, mm, bwa = rf["dimensions"], rf["metrics"], rf["bandwidth_analysis"]
        print(
            f"  [{mat:7s} | {top:10s} | {sub:6s}] W_S={dm['signal_width_um']:.2f} um, "
            f"L_rail={dm['t_rail_length_um']:.2f} um | Z0={mm['Z0_ohm']:.2f} Ohm, "
            f"n_RF={mm['n_RF']:.4f} vs n_g={mm['n_g_opt']:.4f} | "
            f"f_3dBe={mm['f_3dBe_GHz']:.1f} GHz, f_3dBo={mm['f_3dBo_GHz']:.1f} GHz "
            f"({bwa['limiting_mechanism']}) | V_pi*L={mm['V_pi_L_Vcm']:.3f} V*cm"
        )
        if top == "t_rail_cpw":
            gds_rf = export_eom_electrode_gds(rf, out_dir / "tfln_slow_wave_trail_eom.gds")
            bun_rf = export_eom_rf_interconnect_bundle(rf, out_dir / "tfln_rf_bundle")
            print(f"    -> Exported GDS: {gds_rf['gds_path']}")
            print(f"    -> Exported .S2P + INTERCONNECT Bundle: {bun_rf['zip_path']}")

    print("\n=== 1A. Sub-THz 300-GHz X-Cut TFLN Slow-Wave T-Rail Modulator (Quartz vs Undercut-Si vs HR-Si) ===")
    for sub_thz, t_per in (("quartz", 25.0), ("quartz", 220.0), ("undercut_si", 25.0), ("hr_si", 25.0)):
        rf_300 = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology="t_rail_cpw",
                core_material="LN_Xcut",
                substrate_type=sub_thz,
                metal_thickness_um=1.50,
                t_rail_period_um=t_per,
                active_length_um=4500.0,
                max_freq_ghz=360.0,
                run_gap_sweep=False,
            )
        )
        d3, m3 = rf_300["dimensions"], rf_300["metrics"]
        print(
            f"  [LN_Xcut | {sub_thz:11s} | Pitch={t_per:5.1f} um] "
            f"W_S={d3['signal_width_um']:5.2f} um, L_rail={d3['t_rail_length_um']:5.2f} um | "
            f"Z0={m3['Z0_ohm']:.2f} Ohm, n_RF={m3['n_RF']:.4f} | "
            f"a_RF(40G/300G)={m3['alpha_RF_40GHz_dB_per_cm']:.2f}/{m3['alpha_RF_300GHz_dB_per_cm']:.2f} dB/cm | "
            f"f_Bragg={m3['f_bragg_cutoff_GHz']:.0f} GHz | "
            f"S21_EO(300G)={m3['S21_EO_300GHz_dBe']:+5.2f} dBe, f_3dBe={m3['f_3dBe_GHz']:.1f} GHz ({m3['limiting_mechanism']})"
        )

    print("\n=== 1B. Microring Modulator (MRM) Two-Sideband CMT Optical-Peaking Bandwidth ===")
    for det_norm in (0.577, 1.45):
        mrm = design_eom_electrode(
            EOMElectrodeConfig(
                electrode_topology="lumped_ring_gsg",
                core_material="BTO",
                ring_radius_um=25.0,
                racetrack_straight_um=0.0,
                active_length_um=157.1,
                ring_loaded_q=12000.0,
                ring_detuning_norm=det_norm,
                run_gap_sweep=False,
            )
        )
        mm_r, bwa_r = mrm["metrics"], mrm["bandwidth_analysis"]
        print(
            f"  [BTO MRM | Q_L=12,000 | delta={det_norm:.3f}] "
            f"f_3dBe={mm_r['f_3dBe_GHz']:.2f} GHz, f_3dBo={mm_r['f_3dBo_GHz']:.2f} GHz, "
            f"peaking=+{bwa_r['optical_peaking_dBe']:.2f} dBe ({bwa_r['limiting_mechanism']})"
        )

    print("\n=== 2. 2D Nonlinear Poisson-Boltzmann Silicon PN Doping TCAD ===")
    for jtype in ("lateral_pn", "l_shaped_pn", "u_shaped_pn"):
        pn = simulate_pn_doping_tcad(
            PNDopingTCADConfig(
                junction_type=jtype,
                wavelength_nm=1550.0,
                film_height_um=0.22,
                slab_height_um=0.09,
                waveguide_width_um=0.45,
                active_length_um=1500.0,
                na_core_cm3=5.0e17,
                nd_core_cm3=5.0e17,
                pn_offset_nm=25.0,
                map_bias_v=-3.0,
            )
        )
        pm = pn["metrics"]
        print(
            f"  [{jtype:12s}] V_pi*L={pm['V_pi_L_Vcm']:.3f} V*cm, "
            f"FCA(-3V)={pm['alpha_FCA_bias_dB_per_cm']:.2f} dB/cm, "
            f"V_pi*L*alpha={pm['V_pi_L_alpha_V_dB']:.2f} V*dB | "
            f"Cj(0->-3V)={pm['Cj_0V_fF_per_um']:.3f}->{pm['Cj_bias_fF_per_um']:.3f} fF/um, "
            f"f_RC,intr={pm['f_RC_intrinsic_GHz']:.1f} GHz, f_TW,3dBo={pm['f_TW_3dBo_GHz']:.1f} GHz"
        )
        if jtype == "l_shaped_pn":
            gds_pn = export_pn_doping_gds(pn, out_dir / "soi_l_shaped_pn_modulator.gds")
            bun_pn = export_tcad_data_bundle(pn, out_dir / "soi_pn_tcad_bundle")
            print(f"    -> Exported Multi-Implant GDS: {gds_pn['gds_path']}")
            print(f"    -> Exported TCAD + INTERCONNECT OMM Bundle: {bun_pn['zip_path']}")

    print("\n=== 3. 2D Steady-State + Crank-Nicolson Transient Thermal FEM ===")
    for tstruct in ("top_heater", "deep_trench", "undercut_suspended", "doped_si_rib"):
        th = simulate_thermal_phase_shifter(
            ThermalSimConfig(
                thermal_structure=tstruct,
                core_material="Si",
                wavelength_nm=1550.0,
                active_length_um=200.0,
            )
        )
        tm = th["metrics"]
        print(
            f"  [{tstruct:18s}] P_pi={tm['P_pi_mW']:6.3f} mW, R_th={tm['R_th_core_K_per_mW']:6.3f} K/mW | "
            f"t_10-90={tm['t_rise_10_90_us']:6.2f}/{tm['t_fall_90_10_us']:6.2f} us, "
            f"f_3dB={tm['f_3dB_thermal_kHz']:5.2f} kHz | Xtalk@20um={tm['crosstalk_at_neighbor_pct']:.2f}%"
        )
        if tstruct == "deep_trench":
            gds_th = export_thermal_shifter_gds(th, out_dir / "soi_deep_trench_heater.gds")
            print(f"    -> Exported Heater + Air-Trench GDS: {gds_th['gds_path']}")


if __name__ == "__main__":
    main()
