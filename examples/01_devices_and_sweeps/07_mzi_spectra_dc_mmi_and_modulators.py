"""Recipe 07: Mach-Zehnder Interferometer (MZI) Spectra, Splitters (DC / MMI / Y-Branch) & Modulators.

Demonstrates MicroRing Studio v3.6.0 license-free MZI design & tuning without 3D FDTD or COMSOL:
  1. Comparing all 4 selectable 50:50 splitters solved directly via 2D Conformal Vector FEM + EME:
     - 2x2 Directional Coupler (`directional_coupler`): Exact mirror-symmetric Twin-Core Even/Odd Supermodes
     - 2x2 Multimode Interference (`mmi_2x2`): Multi-Mode 2D-FEM Eigenmode Expansion (EME) + 2D |E(x,z)| map
     - 1x2 Symmetric MMI (`mmi_1x2`): Symmetric Self-Imaging Multi-Mode 2D-FEM EME
     - 1x2 Adiabatic Y-Branch (`y_branch`): 2D-FEM Stem-to-Twin-Core Supermode Overlap across lithographic apex
  2. High-Speed Silicon Lateral PN Push-Pull MZI Modulator (V_pi*L, V_pi, RC 3-dB BW, dynamic ER)
  3. X-Cut Thin-Film Lithium Niobate (TFLN) GSG Push-Pull Electro-Optic MZI Modulator
  4. Low-Loss Si3N4 Thermo-Optic TiN Heater MZI Switch (P_pi, thermal 3-dB BW) + Ring-Assisted MZI (RAMZI)
  5. Foundry .GDS mask export + Ansys Lumerical INTERCONNECT (.dat SPAR + .txt OMM + .lsf) bundle.

Run:
    python examples/01_devices_and_sweeps/07_mzi_spectra_dc_mmi_and_modulators.py
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from microring import (
    MZIDesignConfig,
    analyze_uploaded_gds,
    export_mzi_gds,
    export_mzi_interconnect_bundle,
    simulate_mzi,
)


def main() -> None:
    out_dir = REPO_ROOT / "outputs" / "cookbook_07_mzi_suite"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("1. 50:50 Splitter Comparison on 220 nm SOI (dL = 80 um, Zero 3D FDTD)")
    print("=" * 84)
    for sp_key in ("directional_coupler", "mmi_2x2", "mmi_1x2", "y_branch"):
        cfg_sp = MZIDesignConfig(
            core_material="Si",
            wavelength_nm=1550.0,
            film_height_um=0.22,
            slab_height_um=0.0,
            arm1_width_um=0.48,
            arm1_length_um=300.0,
            delta_length_um=80.0,
            splitter_type=sp_key,
            span_nm=30.0,
        )
        res = simulate_mzi(cfg_sp, include_active_tuning=False)
        m = res["metrics"]
        s = res["splitter"]["summary"]
        print(
            f"  [{sp_key:19s}] Split={s['split_bar_pct_at_wl0']:5.2f}/{s['split_cross_pct_at_wl0']:5.2f}% | "
            f"IL={s['excess_loss_db']:.3f} dB | Imb={s['imbalance_at_wl0_db']:+.3f} dB | "
            f"1dB-BW={s['bw_1db_imbalance_nm']:5.1f} nm | MZI FSR={m['FSR_nm']:.3f} nm | "
            f"ER_bar={m['ER_bar_passive_dB']:.1f} dB ({res['elapsed_ms']:.0f} ms)"
        )

    print("\n" + "=" * 84)
    print("2. Silicon Lateral PN Push-Pull MZI Modulator (2x2 MMI, L_arm = 1500 um)")
    print("=" * 84)
    si_pn_cfg = MZIDesignConfig(
        core_material="Si",
        wavelength_nm=1550.0,
        film_height_um=0.22,
        slab_height_um=0.09,
        arm1_width_um=0.48,
        arm1_length_um=1500.0,
        delta_length_um=60.0,
        splitter_type="mmi_2x2",
        mmi_width_um=6.0,
        span_nm=20.0,
    )
    si_pn = simulate_mzi(
        si_pn_cfg,
        include_active_tuning=True,
        mechanism="si_pn_depletion",
        drive_type="voltage_v",
        drive_scheme="push_pull",
        bias_values=[-1.0, -2.0, -3.0, -4.0],
    )
    act_pn = si_pn["active_tuning"]
    print(
        f"  Scheme={act_pn['drive_scheme']} | V_pi*L={act_pn['v_pi_l_vcm']} V*cm | "
        f"V_pi={act_pn['v_pi_v']} V | f_3dB={act_pn['bandwidth_3db']} {act_pn['bandwidth_unit']} | "
        f"Dyn ER={act_pn['dynamic_er_bar_db']} dB | IL_on={act_pn['on_state_il_db']} dB ({si_pn['elapsed_ms']:.0f} ms)"
    )

    print("\n" + "=" * 84)
    print("3. X-Cut Thin-Film LiNbO3 (TFLN) GSG Push-Pull EO MZI (L_arm = 5000 um)")
    print("=" * 84)
    ln_cfg = MZIDesignConfig(
        core_material="LN_Xcut",
        wavelength_nm=1550.0,
        film_height_um=0.60,
        slab_height_um=0.30,
        sidewall_deg=70.0,
        arm1_width_um=1.20,
        arm1_length_um=5000.0,
        delta_length_um=150.0,
        splitter_type="y_branch",
        y_apex_gap_um=0.18,
        span_nm=24.0,
    )
    ln_eo = simulate_mzi(
        ln_cfg,
        include_active_tuning=True,
        mechanism="pockels_eo",
        drive_type="voltage_v",
        drive_scheme="push_pull",
        electrode_gap_um=5.0,
        bias_values=[-4.0, -2.0, 2.0, 4.0],
    )
    act_ln = ln_eo["active_tuning"]
    print(
        f"  Scheme={act_ln['drive_scheme']} | V_pi*L={act_ln['v_pi_l_vcm']} V*cm | "
        f"V_pi={act_ln['v_pi_v']} V | RC f_3dB={act_ln['bandwidth_3db']} {act_ln['bandwidth_unit']} | "
        f"Azimuthal factor={act_ln['geometry_physics']['crystal_azimuthal_factor']} ({ln_eo['elapsed_ms']:.0f} ms)"
    )

    print("\n" + "=" * 84)
    print("4. Si3N4 Thermo-Optic TiN MZI Switch (2x2 DC) + GDS & INTERCONNECT Export")
    print("=" * 84)
    sin_cfg = MZIDesignConfig(
        core_material="Si3N4",
        wavelength_nm=1550.0,
        film_height_um=0.40,
        slab_height_um=0.0,
        arm1_width_um=1.00,
        arm1_length_um=800.0,
        delta_length_um=120.0,
        splitter_type="directional_coupler",
        dc_gap_um=0.35,
        ramzi_enabled=True,
        ramzi_ring_radius_um=45.0,
        ramzi_ring_gap_um=0.35,
        span_nm=20.0,
    )
    sin_to = simulate_mzi(
        sin_cfg,
        include_active_tuning=True,
        mechanism="thermo_optic",
        drive_type="power_mw",
        drive_scheme="single_arm",
        bias_values=[25.0, 50.0, 100.0, 150.0],
    )
    act_to = sin_to["active_tuning"]
    print(
        f"  P_pi={act_to['p_pi_mw']} mW | Thermal f_3dB={act_to['bandwidth_3db']} {act_to['bandwidth_unit']} | "
        f"RAMZI Ring Q_L={sin_to['ramzi']['ring_Q_loaded']:.0f} ({sin_to['elapsed_ms']:.0f} ms)"
    )

    gds_out = export_mzi_gds(si_pn, out_dir / "Si_PN_MMI2x2_MZI.gds")
    ic_out = export_mzi_interconnect_bundle(si_pn, out_dir / "interconnect_si_pn_mzi")
    gds_cal = analyze_uploaded_gds(gds_out["gds_path"])
    print(f"  Exported GDS : {gds_out['gds_path']} (cell={gds_cal['cell_name']}, bbox={gds_out['footprint_um']} um)")
    print(f"  Exported IC  : {ic_out['zip_path']} (passivity_verified={ic_out['passivity_verified']}, ports={ic_out['ports']})")


if __name__ == "__main__":
    main()
