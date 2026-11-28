"""Experimental & Synthetic MRR Coupled-Mode Theory (CMT) Fitting and Prediction Engine.

Physical & experimental safeguards:
1. Never modifies raw experimental spectra; models linear/Fabry-Perot baseline envelopes explicitly.
2. Explicitly distinguishes under-coupled vs. over-coupled branch ambiguity in intensity-only
   all-pass measurements, and automatically resolves coupling regime when phase, drop-port, or
   loss/coupling sweep series are provided.
3. Supports four physical CMT lineshape families via `lmfit`:
   - `all_pass_tmm`: Exact periodic transfer-matrix / CMT all-pass resonator (wide-FSR + narrow-Q)
   - `all_pass_lorentzian`: Temporal CMT single-mode resonance with linear baseline
   - `split_doublet_cmt`: Counter-propagating CW/CCW backscattering doublet CMT (essential for
     high-Q 4H-SiC and Si/AlN rings where mode splitting causes single-Lorentzian residuals)
   - `fano_etalon_cmt`: Temporal CMT interfering with a coherent bus/facet reflection background
4. Multi-resonance FSR -> n_g(lambda), D_2 dispersion extraction & forward spectrum prediction
   with strict calibration vs. blind-holdout device evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.signal import find_peaks
import lmfit

from .materials import C0_M_PER_S

_DB_PER_NEPER = 10.0 / math.log(10.0)  # ~4.342944819 dB


def q_int_to_loss_db_per_cm(q_int: float, ng: float, wavelength_nm: float) -> float:
    """Convert intrinsic quality factor `Q_int` to waveguide power propagation loss `alpha` [dB/cm].

    Exact relation:
      alpha [1/m] = (2 * pi * n_g) / (Q_int * lambda_0 [m])
      alpha [dB/cm] = alpha [1/m] * (10 / ln(10)) / 100
    """
    wl_m = max(float(wavelength_nm) * 1e-9, 1e-12)
    alpha_per_m = (2.0 * math.pi * float(ng)) / (max(float(q_int), 1.0) * wl_m)
    return float(alpha_per_m * _DB_PER_NEPER / 100.0)


def loss_db_per_cm_to_q_int(alpha_db_per_cm: float, ng: float, wavelength_nm: float) -> float:
    """Convert waveguide power propagation loss `alpha` [dB/cm] to intrinsic `Q_int`."""
    wl_m = max(float(wavelength_nm) * 1e-9, 1e-12)
    alpha_per_m = max(float(alpha_db_per_cm), 1e-9) * 100.0 / _DB_PER_NEPER
    return float((2.0 * math.pi * float(ng)) / (alpha_per_m * wl_m))


# ==============================================================================
# 1. Exact Analytical CMT / Transfer-Matrix Forward Models
# ==============================================================================

def cmt_all_pass_complex_amplitude(
    wavelength_nm: np.ndarray,
    lambda0_nm: float,
    kappa_int_ghz: float,
    kappa_ext_ghz: float,
) -> np.ndarray:
    """Temporal CMT complex through-port transmission amplitude t(lambda).

    Convention: kappa_int_ghz, kappa_ext_ghz are full-width energy decay rates
    kappa / (2*pi) in GHz, so loaded FWHM in frequency is:
      Delta_nu_FWHM = kappa_int_ghz + kappa_ext_ghz  [GHz].
    At resonance (Delta_nu = 0):
      t(lambda0) = (kappa_int - kappa_ext) / (kappa_int + kappa_ext)
      - Under-coupled (kappa_ext < kappa_int): t(lambda0) > 0, phase swing < pi
      - Critical coupling (kappa_ext == kappa_int): t(lambda0) = 0
      - Over-coupled (kappa_ext > kappa_int): t(lambda0) < 0, full 2*pi phase swing
    """
    wl_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    wl0_m = float(lambda0_nm) * 1e-9
    nu_ghz = (C0_M_PER_S / wl_m) * 1e-9
    nu0_ghz = (C0_M_PER_S / wl0_m) * 1e-9
    detuning_ghz = nu_ghz - nu0_ghz
    kappa_tot_ghz = kappa_int_ghz + kappa_ext_ghz
    return 1.0 - kappa_ext_ghz / (0.5 * kappa_tot_ghz - 1j * detuning_ghz)


def cmt_add_drop_spectra(
    wavelength_nm: np.ndarray,
    lambda0_nm: float,
    kappa_int_ghz: float,
    kappa_ext_thru_ghz: float,
    kappa_ext_drop_ghz: float,
    baseline_thru: float = 1.0,
    baseline_drop: float = 1.0,
) -> dict[str, np.ndarray]:
    """Temporal CMT 4-port add-drop microring through & drop power spectra."""
    wl_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    wl0_m = float(lambda0_nm) * 1e-9
    detuning_ghz = (C0_M_PER_S / wl_m - C0_M_PER_S / wl0_m) * 1e-9
    kappa_tot_ghz = kappa_int_ghz + kappa_ext_thru_ghz + kappa_ext_drop_ghz
    denom = 0.5 * kappa_tot_ghz - 1j * detuning_ghz
    t_amp = 1.0 - kappa_ext_thru_ghz / denom
    d_amp = -math.sqrt(max(kappa_ext_thru_ghz * kappa_ext_drop_ghz, 0.0)) / denom
    return {
        "thru_power": baseline_thru * (np.abs(t_amp) ** 2),
        "drop_power": baseline_drop * (np.abs(d_amp) ** 2),
        "thru_phase_rad": np.unwrap(np.angle(t_amp)),
        "drop_phase_rad": np.unwrap(np.angle(d_amp)),
    }


def cmt_split_doublet_power(
    wavelength_nm: np.ndarray,
    lambda0_nm: float,
    kappa_int_ghz: float,
    kappa_ext_ghz: float,
    mu_split_ghz: float,
    baseline_level: float = 1.0,
    baseline_slope_per_nm: float = 0.0,
) -> np.ndarray:
    """Counter-propagating CW/CCW backscattering split-doublet temporal CMT model.

    Standing-wave symmetric/antisymmetric eigenmodes split by 2 * mu_split_ghz [GHz]:
      a_+ , a_- have resonance frequencies nu_0 +/- mu_split_ghz
      t(Delta_nu) = 1 - (kappa_ext / 2) * [ 1/(kappa_tot/2 - i(Delta_nu - mu))
                                          + 1/(kappa_tot/2 - i(Delta_nu + mu)) ]
    """
    wl_nm = np.asarray(wavelength_nm, dtype=float)
    detuning_ghz = (C0_M_PER_S / (wl_nm * 1e-9) - C0_M_PER_S / (lambda0_nm * 1e-9)) * 1e-9
    ktot = kappa_int_ghz + kappa_ext_ghz
    half_k = 0.5 * ktot
    t_amp = 1.0 - 0.5 * kappa_ext_ghz * (
        1.0 / (half_k - 1j * (detuning_ghz - mu_split_ghz))
        + 1.0 / (half_k - 1j * (detuning_ghz + mu_split_ghz))
    )
    env = baseline_level + baseline_slope_per_nm * (wl_nm - lambda0_nm)
    return env * (np.abs(t_amp) ** 2)


def cmt_fano_background_power(
    wavelength_nm: np.ndarray,
    lambda0_nm: float,
    kappa_int_ghz: float,
    kappa_ext_ghz: float,
    fano_phase_rad: float = 0.0,
    bg_refl_amp: float = 0.0,
    baseline_level: float = 1.0,
    baseline_slope_per_nm: float = 0.0,
) -> np.ndarray:
    """Temporal CMT with coherent Fano / weak facet-reflection background."""
    wl_nm = np.asarray(wavelength_nm, dtype=float)
    detuning_ghz = (C0_M_PER_S / (wl_nm * 1e-9) - C0_M_PER_S / (lambda0_nm * 1e-9)) * 1e-9
    ktot = kappa_int_ghz + kappa_ext_ghz
    t_ring = np.exp(1j * fano_phase_rad) * bg_refl_amp + (
        1.0 - kappa_ext_ghz / (0.5 * ktot - 1j * detuning_ghz)
    )
    norm = abs(np.exp(1j * fano_phase_rad) * bg_refl_amp + 1.0) ** 2
    env = baseline_level + baseline_slope_per_nm * (wl_nm - lambda0_nm)
    return env * (np.abs(t_ring) ** 2) / max(norm, 1e-12)


def tmm_broadband_mrr_spectrum(
    wavelength_nm: np.ndarray,
    *,
    radius_um: float,
    racetrack_straight_um: float = 0.0,
    neff_at_ref: float,
    ng_at_ref: float,
    lambda_ref_nm: float,
    dispersion_d2_per_nm2: float = 0.0,
    alpha_loss_db_per_cm: float = 1.0,
    kappa_power_thru: float = 0.05,
    kappa_power_drop: float = 0.0,
    delta_neff_tuning: float = 0.0,
) -> dict[str, np.ndarray]:
    """Broadband multi-FSR Transfer-Matrix + Dispersive Eigenmode MRR Spectrum Predictor.

    Predicts exact multi-resonance periodic through/drop transmission, phase, FSR,
    and resonance wavelengths from waveguide eigenmode dispersion (n_eff, n_g, D2),
    propagation loss (alpha [dB/cm]), coupler power coupling (kappa^2), and
    EO / TO / PCM modal index shift (`delta_neff_tuning`, which may be complex
    Delta_n + i * Delta_k for PCM state switching!).
    """
    wl_nm = np.asarray(wavelength_nm, dtype=float)
    wl_m = wl_nm * 1e-9
    l_rt_m = 2.0 * math.pi * (radius_um * 1e-6) + 2.0 * (racetrack_straight_um * 1e-6)

    # First-order group-index dispersion: n_eff(lambda) = n_eff(lam0) - (n_g - n_eff)/lam0 * dlam + 0.5*d2*dlam^2
    dlam_nm = wl_nm - lambda_ref_nm
    dn_dlam_per_nm = -(ng_at_ref - neff_at_ref) / lambda_ref_nm
    neff_real = neff_at_ref + dn_dlam_per_nm * dlam_nm + 0.5 * dispersion_d2_per_nm2 * (dlam_nm ** 2) + np.real(delta_neff_tuning)
    neff_imag = np.imag(delta_neff_tuning)

    # Field attenuation: alpha_power [1/m] = alpha_db_per_cm * 100 * ln(10)/10 + 4*pi*neff_imag / wl_m
    alpha_power_per_m = alpha_loss_db_per_cm * 100.0 * (math.log(10.0) / 10.0) + (4.0 * math.pi * neff_imag / wl_m)
    a_rt = np.exp(-0.5 * alpha_power_per_m * l_rt_m)
    phi_rt = (2.0 * math.pi * neff_real / wl_m) * l_rt_m

    t1 = math.sqrt(max(1.0 - kappa_power_thru, 0.0))
    k1 = math.sqrt(max(kappa_power_thru, 0.0))
    t2 = math.sqrt(max(1.0 - kappa_power_drop, 0.0))
    k2 = math.sqrt(max(kappa_power_drop, 0.0))

    denom = 1.0 - t1 * t2 * a_rt * np.exp(1j * phi_rt)
    thru_amp = (t1 - t2 * a_rt * np.exp(1j * phi_rt)) / denom
    drop_amp = (-k1 * k2 * np.sqrt(a_rt) * np.exp(0.5j * phi_rt)) / denom

    return {
        "wavelength_nm": wl_nm,
        "thru_T": np.abs(thru_amp) ** 2,
        "drop_T": np.abs(drop_amp) ** 2,
        "thru_phase_rad": np.unwrap(np.angle(thru_amp)),
        "roundtrip_amplitude_a": a_rt,
        "neff_real": neff_real,
        "L_rt_um": np.full_like(wl_nm, l_rt_m * 1e6),
    }


# ==============================================================================
# 2. Single-Resonance & Multi-Resonance CMT Spectrum Fitting (`lmfit`)
# ==============================================================================

@dataclass
class ResonanceFitResult:
    model_type: str
    lambda0_nm: float
    lambda0_stderr_nm: float | None
    fwhm_pm: float
    fwhm_ghz: float
    Q_loaded: float
    extinction_ratio_db: float
    # Branch-disambiguated or explicitly paired solutions:
    coupling_regime_status: str
    Q_int_under_branch: float
    Q_ext_under_branch: float
    Q_int_over_branch: float
    Q_ext_over_branch: float
    Q_int_selected: float | None
    Q_ext_selected: float | None
    mode_splitting_ghz: float
    fano_phase_rad: float
    r_squared: float
    nrmse_pct: float
    aic: float
    bic: float
    # Extracted waveguide propagation loss [dB/cm] & coupler power coupling (given n_g / L_rt)
    ng_used: float = 2.10
    alpha_loss_under_db_per_cm: float = 0.0
    alpha_loss_over_db_per_cm: float = 0.0
    alpha_loss_selected_db_per_cm: float | None = None
    roundtrip_loss_db: float | None = None
    kappa_power_ext: float | None = None

    @property
    def Q_int(self) -> float:
        return float(self.Q_int_selected if self.Q_int_selected is not None else self.Q_int_under_branch)

    @property
    def Q_ext(self) -> float:
        return float(self.Q_ext_selected if self.Q_ext_selected is not None else self.Q_ext_under_branch)

    @property
    def alpha_loss_db_per_cm(self) -> float:
        return float(
            self.alpha_loss_selected_db_per_cm
            if self.alpha_loss_selected_db_per_cm is not None
            else self.alpha_loss_under_db_per_cm
        )

    @property
    def mu_split_ghz(self) -> float:
        return float(self.mode_splitting_ghz)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["Q_int"] = self.Q_int
        d["Q_ext"] = self.Q_ext
        d["alpha_loss_db_per_cm"] = round(self.alpha_loss_db_per_cm, 4)
        d["mu_split_ghz"] = self.mu_split_ghz
        return d


CMTFitSummary = ResonanceFitResult


def fit_single_resonance_cmt(
    wavelength_nm: np.ndarray,
    transmission: np.ndarray,
    *,
    model_type: str = "auto",
    known_regime: str | None = None,
    measured_phase_rad: np.ndarray | None = None,
    ng_assumed: float | None = None,
    ring_length_um: float | None = None,
) -> tuple[ResonanceFitResult, np.ndarray, lmfit.model.ModelResult]:
    """Fit a single MRR resonance window with rigorous CMT using `lmfit`.

    Args:
      wavelength_nm: 1D wavelength vector in nm (raw, unmodified).
      transmission: 1D linear power transmission (or dB/dBm if negative, auto-converted).
      model_type: 'lorentzian', 'split_doublet', 'fano', or 'auto' (compares BIC to
                  detect genuine backscattering doublet splitting vs single Lorentzian).
      known_regime: None (intensity-only -> reports both under/over branches honestly!),
                    'under', or 'over'. If `measured_phase_rad` is supplied, regime is
                    determined automatically from the phase swing!
      ng_assumed: Group index `n_g` used to convert extracted `Q_int` into waveguide
                  propagation loss `alpha [dB/cm]` (defaults to `2.10` if omitted).
      ring_length_um: Optional cavity round-trip length `L_rt` [um] to also compute
                      round-trip loss [dB] and bus-ring power coupling `kappa^2`.
    """
    wl = np.asarray(wavelength_nm, dtype=float).ravel()
    tr = np.asarray(transmission, dtype=float).ravel()
    if len(wl) != len(tr) or len(wl) < 7:
        raise ValueError("Need matching wavelength_nm and transmission vectors with >= 7 points")
    if float(np.min(tr)) < -0.25:
        # Auto-convert dB / dBm transmission trace into normalized linear power
        tr = 10.0 ** ((tr - float(np.percentile(tr, 95))) / 10.0)

    # Initial guesses from raw data (without smoothing or altering tr)
    edge_n = max(2, len(tr) // 10)
    b0_guess = float(0.5 * (np.median(tr[:edge_n]) + np.median(tr[-edge_n:])))
    b0_guess = max(b0_guess, float(np.percentile(tr, 90)), 1e-6)
    slope_guess = float(( np.median(tr[-edge_n:]) - np.median(tr[:edge_n]) ) / max(wl[-1] - wl[0], 1e-6))

    dip_idx = int(np.argmin(tr))
    lam0_guess = float(wl[dip_idx])
    min_tr = max(float(tr[dip_idx]) / b0_guess, 1e-6)

    half_level = b0_guess * (0.5 * (1.0 + min_tr))
    below = wl[tr <= half_level]
    fwhm_nm_guess = float(below[-1] - below[0]) if len(below) >= 2 else float((wl[-1] - wl[0]) * 0.15)
    fwhm_nm_guess = float(np.clip(fwhm_nm_guess, 2.0 * float(np.median(np.abs(np.diff(wl)))), (wl[-1] - wl[0]) * 0.8))

    nu0_ghz = (C0_M_PER_S / (lam0_guess * 1e-9)) * 1e-9
    ktot_ghz_guess = max(nu0_ghz * (fwhm_nm_guess / lam0_guess), 1e-4)

    # From T_min = ((k_int - k_ext)/(k_int + k_ext))^2 => k_under, k_over
    r_amp = math.sqrt(min(max(min_tr, 0.0), 0.9999))
    k_small = 0.5 * ktot_ghz_guess * (1.0 - r_amp)
    k_large = 0.5 * ktot_ghz_guess * (1.0 + r_amp)

    def _run_lorentzian() -> lmfit.model.ModelResult:
        def _lorentz_fn(wavelength_nm, lambda0_nm, kappa_a_ghz, kappa_b_ghz, baseline_level, baseline_slope_per_nm):
            t_c = cmt_all_pass_complex_amplitude(wavelength_nm, lambda0_nm, kappa_a_ghz, kappa_b_ghz)
            env = baseline_level + baseline_slope_per_nm * (wavelength_nm - lambda0_nm)
            return env * (np.abs(t_c) ** 2)

        mod = lmfit.Model(_lorentz_fn)
        p = mod.make_params(
            lambda0_nm=dict(value=lam0_guess, min=float(wl.min()), max=float(wl.max())),
            kappa_a_ghz=dict(value=max(k_large, 1e-5), min=1e-6, max=ktot_ghz_guess * 20.0),
            kappa_b_ghz=dict(value=max(k_small, 1e-5), min=1e-6, max=ktot_ghz_guess * 20.0),
            baseline_level=dict(value=b0_guess, min=b0_guess * 0.2, max=b0_guess * 3.0),
            baseline_slope_per_nm=dict(value=slope_guess),
        )
        return mod.fit(tr, p, wavelength_nm=wl)

    def _run_doublet() -> lmfit.model.ModelResult:
        mod = lmfit.Model(cmt_split_doublet_power)
        p = mod.make_params(
            lambda0_nm=dict(value=lam0_guess, min=float(wl.min()), max=float(wl.max())),
            kappa_int_ghz=dict(value=max(k_large * 0.8, 1e-5), min=1e-6, max=ktot_ghz_guess * 20.0),
            kappa_ext_ghz=dict(value=max(k_small * 1.1, 1e-5), min=1e-6, max=ktot_ghz_guess * 20.0),
            mu_split_ghz=dict(value=0.35 * ktot_ghz_guess, min=0.0, max=ktot_ghz_guess * 5.0),
            baseline_level=dict(value=b0_guess, min=b0_guess * 0.2, max=b0_guess * 3.0),
            baseline_slope_per_nm=dict(value=slope_guess),
        )
        return mod.fit(tr, p, wavelength_nm=wl)

    def _run_fano() -> lmfit.model.ModelResult:
        mod = lmfit.Model(cmt_fano_background_power)
        p = mod.make_params(
            lambda0_nm=dict(value=lam0_guess, min=float(wl.min()), max=float(wl.max())),
            kappa_int_ghz=dict(value=max(k_large, 1e-5), min=1e-6, max=ktot_ghz_guess * 20.0),
            kappa_ext_ghz=dict(value=max(k_small, 1e-5), min=1e-6, max=ktot_ghz_guess * 20.0),
            fano_phase_rad=dict(value=0.2, min=-math.pi, max=math.pi),
            bg_refl_amp=dict(value=0.05, min=0.0, max=0.8),
            baseline_level=dict(value=b0_guess, min=b0_guess * 0.2, max=b0_guess * 3.0),
            baseline_slope_per_nm=dict(value=slope_guess),
        )
        return mod.fit(tr, p, wavelength_nm=wl)

    if model_type == "lorentzian":
        chosen_name = "lorentzian"
        res = _run_lorentzian()
    elif model_type == "split_doublet":
        chosen_name = "split_doublet"
        res = _run_doublet()
    elif model_type == "fano":
        chosen_name = "fano"
        res = _run_fano()
    elif model_type == "auto":
        res_lor = _run_lorentzian()
        res_dbl = _run_doublet()
        res_fano = _run_fano()
        # Require strong BIC improvement and genuine physical signature before selecting higher-DOF models
        mu_fit = float(res_dbl.best_values["mu_split_ghz"])
        kt_dbl = float(res_dbl.best_values["kappa_int_ghz"] + res_dbl.best_values["kappa_ext_ghz"])
        fano_asym = abs(math.sin(float(res_fano.best_values["fano_phase_rad"]))) * float(res_fano.best_values["bg_refl_amp"])
        if (res_lor.bic - res_fano.bic) > 14.0 and (res_dbl.bic - res_fano.bic) > 6.0 and fano_asym > 0.04:
            chosen_name = "fano"
            res = res_fano
        elif (res_lor.bic - res_dbl.bic) > 10.0 and mu_fit > 0.12 * kt_dbl:
            chosen_name = "split_doublet"
            res = res_dbl
        else:
            chosen_name = "lorentzian"
            res = res_lor
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    bv = res.best_values
    lam0 = float(bv["lambda0_nm"])
    lam0_err = res.params["lambda0_nm"].stderr
    nu0 = (C0_M_PER_S / (lam0 * 1e-9)) * 1e-9  # GHz

    if chosen_name == "lorentzian":
        ka, kb = float(bv["kappa_a_ghz"]), float(bv["kappa_b_ghz"])
        k_high, k_low = max(ka, kb), min(ka, kb)
        mu_ghz = 0.0
        fano_phi = 0.0
    elif chosen_name == "split_doublet":
        # Note: in a symmetric CW/CCW doublet, T(Delta_nu) is NOT invariant under swapping k_int and k_ext
        # once mu_split > 0 because the central peak/dip contrast depends distinctly on k_ext vs k_int!
        ki, ke = float(bv["kappa_int_ghz"]), float(bv["kappa_ext_ghz"])
        k_high, k_low = max(ki, ke), min(ki, ke)
        mu_ghz = float(bv["mu_split_ghz"])
        fano_phi = 0.0
    else:
        ki, ke = float(bv["kappa_int_ghz"]), float(bv["kappa_ext_ghz"])
        k_high, k_low = max(ki, ke), min(ki, ke)
        mu_ghz = 0.0
        fano_phi = float(bv["fano_phase_rad"])

    ktot_ghz = k_high + k_low
    fwhm_pm = (lam0 * (ktot_ghz / nu0)) * 1e3
    q_loaded = nu0 / ktot_ghz

    q_int_under = nu0 / k_high
    q_ext_under = nu0 / k_low
    q_int_over = nu0 / k_low
    q_ext_over = nu0 / k_high

    # Resolve coupling regime if phase or prior regime or non-degenerate doublet is available
    regime_status = "AMBIGUOUS_INTENSITY_ONLY (both under/over branches reported)"
    q_int_sel: float | None = None
    q_ext_sel: float | None = None

    if measured_phase_rad is not None:
        ph_unwrap = np.unwrap(np.asarray(measured_phase_rad, dtype=float))
        swing = float(np.max(ph_unwrap) - np.min(ph_unwrap))
        if swing > math.pi:
            regime_status = "OVER_COUPLED (resolved from >pi phase swing)"
            q_int_sel, q_ext_sel = q_int_over, q_ext_over
        else:
            regime_status = "UNDER_COUPLED (resolved from <pi phase swing)"
            q_int_sel, q_ext_sel = q_int_under, q_ext_under
    elif known_regime in ("under", "undercoupled"):
        regime_status = "UNDER_COUPLED (user/sweep specified)"
        q_int_sel, q_ext_sel = q_int_under, q_ext_under
    elif known_regime in ("over", "overcoupled"):
        regime_status = "OVER_COUPLED (user/sweep specified)"
        q_int_sel, q_ext_sel = q_int_over, q_ext_over
    elif chosen_name == "split_doublet" and mu_ghz > 0.15 * ktot_ghz:
        # In standing-wave split doublet, k_ext vs k_int breaks exchange symmetry!
        ki_fit, ke_fit = float(bv["kappa_int_ghz"]), float(bv["kappa_ext_ghz"])
        q_int_sel, q_ext_sel = nu0 / ki_fit, nu0 / ke_fit
        regime_status = (
            "OVER_COUPLED (resolved by doublet contrast)" if ke_fit > ki_fit
            else "UNDER_COUPLED (resolved by doublet contrast)"
        )

    y_fit = res.best_fit
    env_at_res = float(bv["baseline_level"])
    min_norm = max(float(np.min(y_fit)) / max(env_at_res, 1e-12), 1e-12)
    er_db = -10.0 * math.log10(min_norm)

    ss_res = float(np.sum((tr - y_fit) ** 2))
    ss_tot = float(np.sum((tr - np.mean(tr)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-18)
    rmse = math.sqrt(ss_res / len(tr))
    nrmse_pct = (rmse / max(float(np.max(tr) - np.min(tr)), 1e-12)) * 100.0

    ng_val = float(ng_assumed) if (ng_assumed is not None and ng_assumed > 0.5) else 2.10
    loss_under = q_int_to_loss_db_per_cm(q_int_under, ng_val, lam0)
    loss_over = q_int_to_loss_db_per_cm(q_int_over, ng_val, lam0)
    loss_sel = q_int_to_loss_db_per_cm(q_int_sel, ng_val, lam0) if q_int_sel is not None else None

    rt_loss_db: float | None = None
    kp_ext: float | None = None
    if ring_length_um is not None and ring_length_um > 0:
        eff_loss = loss_sel if loss_sel is not None else loss_under
        rt_loss_db = round(eff_loss * (float(ring_length_um) * 1e-4), 5)
        eff_q_ext = q_ext_sel if q_ext_sel is not None else q_ext_under
        kp_ext = round(
            float(1.0 - math.exp(-(2.0 * math.pi * ng_val * (float(ring_length_um) * 1e-6)) / max(eff_q_ext * (lam0 * 1e-9), 1e-18))),
            6,
        )

    summary = ResonanceFitResult(
        model_type=chosen_name,
        lambda0_nm=lam0,
        lambda0_stderr_nm=float(lam0_err) if lam0_err is not None else None,
        fwhm_pm=fwhm_pm,
        fwhm_ghz=ktot_ghz,
        Q_loaded=q_loaded,
        extinction_ratio_db=er_db,
        coupling_regime_status=regime_status,
        Q_int_under_branch=q_int_under,
        Q_ext_under_branch=q_ext_under,
        Q_int_over_branch=q_int_over,
        Q_ext_over_branch=q_ext_over,
        Q_int_selected=q_int_sel,
        Q_ext_selected=q_ext_sel,
        mode_splitting_ghz=mu_ghz,
        fano_phase_rad=fano_phi,
        r_squared=r2,
        nrmse_pct=nrmse_pct,
        aic=float(res.aic),
        bic=float(res.bic),
        ng_used=round(ng_val, 5),
        alpha_loss_under_db_per_cm=round(loss_under, 4),
        alpha_loss_over_db_per_cm=round(loss_over, 4),
        alpha_loss_selected_db_per_cm=round(loss_sel, 4) if loss_sel is not None else None,
        roundtrip_loss_db=rt_loss_db,
        kappa_power_ext=kp_ext,
    )
    return summary, y_fit, res


def fit_multi_resonance_spectrum(
    wavelength_nm: np.ndarray,
    transmission: np.ndarray,
    *,
    ring_length_um: float,
    min_prominence: float = 0.05,
    window_fwhm_factor: float = 8.0,
    known_regime: str | None = None,
    model_type: str = "auto",
) -> dict:
    """Detect and fit multiple resonances across a broadband MRR scan; extract FSR, n_g & propagation loss [dB/cm]."""
    wl = np.asarray(wavelength_nm, dtype=float).ravel()
    tr = np.asarray(transmission, dtype=float).ravel()
    if float(np.min(tr)) < -0.25:
        tr = 10.0 ** ((tr - float(np.percentile(tr, 95))) / 10.0)
    step_nm = float(np.median(np.diff(wl)))
    if step_nm <= 0:
        raise ValueError("wavelength_nm must be strictly ascending")

    inverted = np.max(tr) - tr
    peaks, props = find_peaks(inverted, prominence=min_prominence * (np.max(tr) - np.min(tr)), width=1)
    if len(peaks) == 0:
        raise ValueError("No resonance dips detected with given min_prominence")

    # Estimate spacing to avoid overlapping windows
    if len(peaks) >= 2:
        est_fsr_nm = float(np.median(np.diff(wl[peaks])))
    else:
        est_fsr_nm = float(wl[-1] - wl[0])

    fits: list[dict] = []
    for p_idx, w_pts in zip(peaks, props["widths"]):
        est_fwhm_nm = max(float(w_pts) * step_nm, 3.0 * step_nm)
        half_win = min(0.42 * est_fsr_nm, max(window_fwhm_factor * est_fwhm_nm, 15.0 * step_nm))
        c_wl = wl[p_idx]
        mask = (wl >= c_wl - half_win) & (wl <= c_wl + half_win)
        if np.count_nonzero(mask) < 9:
            continue
        fit_res, _, _ = fit_single_resonance_cmt(
            wl[mask], tr[mask], model_type=model_type, known_regime=known_regime, ring_length_um=ring_length_um
        )
        fits.append(fit_res.to_dict())

    lam_centers = np.array([f["lambda0_nm"] for f in fits], dtype=float)
    fsr_nm = np.diff(lam_centers)
    lam_mid = 0.5 * (lam_centers[:-1] + lam_centers[1:])
    l_rt_m = ring_length_um * 1e-6
    ng_extracted = ((lam_mid * 1e-9) ** 2) / (l_rt_m * (fsr_nm * 1e-9)) if len(fsr_nm) > 0 else np.array([])
    fsr_ghz = (C0_M_PER_S / (l_rt_m * ng_extracted)) * 1e-9 if len(ng_extracted) > 0 else np.array([])

    mean_q_loaded = float(np.mean([f["Q_loaded"] for f in fits])) if fits else 0.0
    mean_q_int = float(np.mean([f["Q_int"] for f in fits])) if fits else 0.0
    mean_q_int_under = float(np.mean([f["Q_int_under_branch"] for f in fits])) if fits else 0.0
    mean_q_int_over = float(np.mean([f["Q_int_over_branch"] for f in fits])) if fits else 0.0
    ng_mean_val = float(np.mean(ng_extracted)) if len(ng_extracted) > 0 else None

    # Self-contained propagation loss [dB/cm] using FSR-extracted n_g + fitted Q_int!
    ng_for_loss = ng_mean_val if ng_mean_val is not None else 2.10
    lam_mean = float(np.mean(lam_centers)) if len(lam_centers) > 0 else float(np.mean(wl))
    loss_under = round(q_int_to_loss_db_per_cm(mean_q_int_under, ng_for_loss, lam_mean), 4) if fits else None
    loss_over = round(q_int_to_loss_db_per_cm(mean_q_int_over, ng_for_loss, lam_mean), 4) if fits else None
    loss_sel = round(q_int_to_loss_db_per_cm(mean_q_int, ng_for_loss, lam_mean), 4) if fits else None
    rt_loss = round(loss_sel * (ring_length_um * 1e-4), 5) if loss_sel is not None else None

    return {
        "resonance_count": len(fits),
        "num_resonances": len(fits),
        "resonances": fits,
        "FSR_mid_wavelength_nm": lam_mid.tolist(),
        "FSR_nm": fsr_nm.tolist(),
        "FSR_GHz": fsr_ghz.tolist(),
        "mean_fsr_nm": float(np.mean(fsr_nm)) if len(fsr_nm) > 0 else 0.0,
        "mean_fsr_ghz": float(np.mean(fsr_ghz)) if len(fsr_ghz) > 0 else 0.0,
        "mean_Q_loaded": mean_q_loaded,
        "mean_Q_int": mean_q_int,
        "alpha_loss_db_per_cm": loss_sel,
        "alpha_loss_under_db_per_cm": loss_under,
        "alpha_loss_over_db_per_cm": loss_over,
        "roundtrip_loss_db": rt_loss,
        "ng_extracted_list": ng_extracted.tolist(),
        "ng_extracted": ng_mean_val,
        "ng_mean": ng_mean_val,
        "ng_std": float(np.std(ng_extracted)) if len(ng_extracted) > 0 else None,
    }
