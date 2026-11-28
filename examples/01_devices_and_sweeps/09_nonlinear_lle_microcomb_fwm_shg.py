"""Cookbook 09: First-Principles Nonlinear Photonics, LLE Microcombs, FWM-BS, SHG & Sidewall Scattering.

Demonstrates the complete license-free nonlinear microresonator suite in MicroRing Studio:
  1. Broadband 2D curved-FEM integrated dispersion D_int(mu), vector Afshar-Monro gamma_NL,
     dissipative Kerr soliton (DKS), and causal Raman self-frequency shift (Brasch Science 2016 Si3N4).
  2. Cross-band 4H-SiC Four-Wave Mixing Bragg Scattering (FWM-BS / quantum frequency conversion).
  3. X-cut TFLN PPLN Chi(2) Second-Harmonic Generation (g2, QPM period & pump-depleted curve).
  4. High-power Photothermal + Kerr Bistability (steady S-curve & dynamic shark-fin hysteresis).
  5. Barwicz-Haus 3D Volume-Current sidewall roughness scattering loss & CW/CCW doublet splitting.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from microring import (
    FWMBSConfig,
    LLEConfig,
    MRRDesignConfig,
    SHGConfig,
    ThermalKerrBistabilityConfig,
    compute_resonator_dispersion_and_nl,
    simulate_chi2_shg_resonator,
    simulate_fwm_bragg_scattering,
    simulate_lle_microcomb,
    simulate_mrr_topology,
    simulate_thermal_kerr_bistability,
)


def main() -> None:
    out_dir = REPO_ROOT / "outputs" / "cookbook_09_nonlinear_lle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Broadband 2D Bent-FEM Dispersion + Generalized SSFM LLE Soliton Microcomb ===")
    cfg_sin = LLEConfig(
        core_material="Si3N4",
        film_height_um=0.80,
        ring_width_um=1.60,
        ring_radius_um=23.0,
        gap_thru_um=0.45,
        q_int_override=1.5e6,
        q_ext_override=1.5e6,
        pump_power_mw=80.0,
        detuning_end_kappa=17.0,
        mode="single_soliton",
        num_comb_modes=256,
        num_steps=2500,
    )
    disp = compute_resonator_dispersion_and_nl(cfg_sin)
    dm = disp["metrics"]
    print(
        f"  Si3N4 800x1600nm (R=23um): FSR={dm['FSR_GHz']:.1f} GHz | D2/2pi={dm['D2_2pi_MHz']:.2f} MHz "
        f"({dm['dispersion_regime']}) | gamma_NL={dm['gamma_NL_per_W_m']:.2f} /Wm | P_th={dm['P_th_OPO_mW']:.2f} mW"
    )
    r_dks = simulate_lle_microcomb(cfg_sin, disp_cache=disp)
    md = r_dks["metrics"]
    print(
        f"  DKS ({r_dks['compute_backend']}): state={md['comb_state']} | peak={md['soliton_peak_power_W']:.1f} W "
        f"(analytic={md['analytic_soliton_peak_W']:.1f} W) | FWHM={md['pulse_fwhm_fs']:.1f} fs "
        f"(analytic={md['analytic_pulse_fwhm_fs']:.1f} fs) | 3dB BW={md['comb_3db_bandwidth_THz']:.2f} THz"
    )
    r_ram = simulate_lle_microcomb(
        LLEConfig(**{**cfg_sin.__dict__, "include_raman": True}), disp_cache=disp
    )
    print(
        f"  + Causal Raman: centroid mu={r_ram['metrics']['spectral_centroid_mu']:+.2f} "
        f"(self-frequency shift={r_ram['metrics']['raman_shift_THz']:+.3f} THz)"
    )

    comb_csv = out_dir / "sin_dks_comb_spectrum.csv"
    sp = r_dks["spectrum"]
    with comb_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mode_index_mu", "wavelength_nm", "D_int_2pi_GHz", "comb_power_dBm", "intracavity_power_dBm"])
        for row in zip(sp["mu"], sp["wavelength_nm"], sp["D_int_2pi_GHz"], sp["comb_power_dBm"], sp["intracavity_power_dBm"]):
            w.writerow(row)
    print(f"  Saved comb CSV -> {comb_csv}")

    print("\n=== 2. Cross-Band 4H-SiC Four-Wave Mixing Bragg Scattering (FWM-BS / QFC) ===")
    r_fwm = simulate_fwm_bragg_scattering(
        FWMBSConfig(
            core_material="4H-SiC",
            film_height_um=0.50,
            ring_width_um=1.15,
            ring_radius_um=35.0,
            pump1_wl_nm=1550.0,
            pump2_wl_nm=1510.0,
            signal_wl_nm=750.0,
            pump1_power_mw=20.0,
            pump2_power_mw=20.0,
        )
    )
    mf = r_fwm["metrics"]
    print(
        f"  4H-SiC 750nm QFC: gamma_BS={mf['gamma_BS_per_W_m']:.2f} /Wm | C_BS={mf['cooperativity_C_BS']:.2f} | "
        f"eta_CE(PM)={mf['conversion_eff_phase_matched_pct']:.1f}% | P_opt={mf['optimal_pump_geom_mean_mW']:.1f} mW | "
        f"Spurious suppression={mf['spurious_suppression_dB']:.1f} dB"
    )

    print("\n=== 3. X-Cut TFLN PPLN Chi(2) Second-Harmonic Generation (SHG) ===")
    r_shg = simulate_chi2_shg_resonator(
        SHGConfig(
            core_material="LN_Xcut",
            film_height_um=0.60,
            slab_height_um=0.30,
            ring_width_um=1.40,
            ring_radius_um=50.0,
            fundamental_wl_nm=1550.0,
            q_int_fh=1e6,
            q_ext_fh=1e6,
            q_int_sh=5e5,
            q_ext_sh=5e5,
        )
    )
    ms = r_shg["metrics"]
    print(
        f"  X-TFLN SHG: Lambda_QPM={ms['QPM_period_um']:.3f} um (M={ms['QPM_azimuthal_order_M']:.1f}) | "
        f"g2/2pi={ms['g2_2pi_MHz']:.3f} MHz | eta_0={ms['eta_SHG_undepleted_pct_per_W']:.1e} %/W | "
        f"Max depleted eff={ms['max_depleted_eff_pct']:.1f}%"
    )

    print("\n=== 4. High-Power Photothermal + Kerr Shark-Fin Bistability ===")
    r_bis = simulate_thermal_kerr_bistability(
        ThermalKerrBistabilityConfig(
            core_material="Si3N4", ring_radius_um=40.0, q_int=1e6, q_ext=1e6, input_power_mw=25.0
        )
    )
    mb = r_bis["metrics"]
    print(
        f"  Si3N4 25mW: bistable={mb['is_bistable']} | P_bist={mb['bistability_threshold_mW']:.3f} mW | "
        f"broaden={mb['thermal_broadening_factor']:.2f}x | hysteresis={mb['scan_hysteresis_GHz']:.3f} GHz | "
        f"dT_peak={mb['peak_delta_T_K']:.2f} K"
    )

    print("\n=== 5. First-Principles Barwicz-Haus Sidewall Roughness Scattering & Backscattering ===")
    for lbl, mat, h, w, r, sig in (
        ("SOI 220x450nm", "Si", 0.22, 0.45, 15.0, 2.0),
        ("SOI 220x500nm", "Si", 0.22, 0.50, 15.0, 2.0),
        ("SOI 220x1000nm", "Si", 0.22, 1.00, 15.0, 2.0),
        ("Si3N4 800x1600nm", "Si3N4", 0.80, 1.60, 23.0, 1.0),
    ):
        rs = simulate_mrr_topology(
            MRRDesignConfig(
                core_material=mat,
                film_height_um=h,
                ring_width_um=w,
                bus_width_um=w,
                ring_radius_um=r,
                gap_thru_um=0.22,
                alpha_loss_db_per_cm=0.1,
                sidewall_rms_roughness_nm=sig,
                sidewall_corr_length_nm=50.0,
                span_nm=4.0,
            )
        )
        print(
            f"  {lbl:17s} (sigma={sig:.1f}nm, Lc=50nm): alpha_scat={rs['metrics']['alpha_scattering_db_per_cm']:.3f} dB/cm | "
            f"CW/CCW split_rms={rs['metrics']['backscatter_split_rms_mhz']:.1f} MHz"
        )


if __name__ == "__main__":
    main()
