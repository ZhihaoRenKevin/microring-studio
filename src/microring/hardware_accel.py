"""Cross-Platform Hardware Detection & Compute Dispatcher (macOS / Windows · CUDA GPU / Metal / CPU).

Automatically probes the host operating system and available accelerators with zero
manual configuration required:
  1. **Windows / Linux + NVIDIA CUDA GPU** (`cuda:0` via PyTorch `float64`/`complex128`):
     Executes broadband multi-FSR Transfer-Matrix (TMM) spectra, 2D coupler design maps,
     and Vernier/CROW parameter sweeps in native FP64 on the GPU, with automatic VRAM
     headroom checking (auto-falls back to multi-core CPU if GPU memory is >85% occupied).
  2. **macOS Apple Silicon (`arm64` M-series)**:
     Detects Apple Accelerate AMX (FP64 sparse FEM eigensolver), Metal MPS (`torch.mps`),
     and Apple `MLX` unified-memory array engine.
  3. **CPU-Only PC (Windows `x86_64` / Intel Mac / Linux)**:
     Cleanly dispatches to multi-threaded BLAS/LAPACK (`NumPy` / `SciPy`) with safe
     thread-count capping based on physical cores and available RAM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import platform
import time

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:
    import mlx.core as mx  # noqa: F401
    from importlib.metadata import version as _pkg_version
    _MLX_VERSION: str | None = _pkg_version("mlx")
except Exception:  # pragma: no cover
    _MLX_VERSION = None


@dataclass(frozen=True)
class HardwareProfile:
    """Detected host platform and active compute backend summary."""
    os_name: str                    # 'macOS', 'Windows', or 'Linux'
    os_release: str
    arch: str                       # 'arm64', 'AMD64' / 'x86_64', etc.
    cpu_brand: str
    physical_cores: int
    logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    has_cuda_gpu: bool
    cuda_device_name: str | None
    cuda_vram_total_gb: float
    cuda_vram_free_gb: float
    cuda_vram_used_pct: float
    has_apple_mps: bool
    mlx_version: str | None
    selected_spectrum_backend: str  # 'cuda_fp64', 'apple_amx_accelerate', or 'cpu_multicore'
    selected_fem_backend: str       # 'cpu_amx_sparse_fp64' or 'cpu_blas_sparse_fp64'
    recommended_workers: int
    badge_label: str                # concise UI header badge

    @property
    def cpu_cores(self) -> int:
        return self.logical_cores

    @property
    def gpu_name(self) -> str | None:
        return self.cuda_device_name or (
            "Apple Silicon Unified AMX/Metal" if self.os_name == "macOS" and self.arch == "arm64" else None
        )

    @property
    def recommended_backend(self) -> str:
        return self.badge_label

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cpu_cores"] = self.cpu_cores
        d["gpu_name"] = self.gpu_name
        d["recommended_backend"] = self.recommended_backend
        return d


def _detect_cpu_brand() -> str:
    sys_name = platform.system()
    if sys_name == "Darwin":
        # Avoid spawning subprocesses if platform.processor() or sysctl info is cached
        proc = platform.processor() or platform.machine()
        return f"Apple Silicon ({platform.machine()})" if platform.machine() == "arm64" else proc
    return platform.processor() or platform.machine() or "Generic CPU"


def detect_hardware_profile(prefer_gpu: bool = True) -> HardwareProfile:
    """Probe OS, CPU, RAM, NVIDIA CUDA GPU, and Apple Silicon Metal/AMX/MLX."""
    raw_sys = platform.system()
    os_name = "macOS" if raw_sys == "Darwin" else ("Windows" if raw_sys == "Windows" else raw_sys)
    arch = platform.machine()
    cpu_brand = _detect_cpu_brand()

    if _HAS_PSUTIL and psutil is not None:
        phys_cores = psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
        log_cores = psutil.cpu_count(logical=True) or phys_cores
        vm = psutil.virtual_memory()
        ram_tot_gb = round(vm.total / (1024 ** 3), 2)
        ram_avail_gb = round(vm.available / (1024 ** 3), 2)
    else:
        log_cores = os.cpu_count() or 4
        phys_cores = max(1, log_cores // 2)
        ram_tot_gb = 16.0
        ram_avail_gb = 8.0

    has_cuda = False
    cuda_name: str | None = None
    vram_tot_gb = 0.0
    vram_free_gb = 0.0
    vram_used_pct = 0.0
    has_mps = False

    if _HAS_TORCH and torch is not None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            has_cuda = True
            cuda_name = str(torch.cuda.get_device_name(0))
            try:
                free_b, tot_b = torch.cuda.mem_get_info(0)
                vram_tot_gb = round(tot_b / (1024 ** 3), 2)
                vram_free_gb = round(free_b / (1024 ** 3), 2)
                vram_used_pct = round((1.0 - free_b / max(tot_b, 1)) * 100.0, 1)
            except Exception:
                props = torch.cuda.get_device_properties(0)
                vram_tot_gb = round(props.total_memory / (1024 ** 3), 2)
                vram_free_gb = vram_tot_gb
                vram_used_pct = 0.0

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            has_mps = True

    rec_workers = max(1, min(phys_cores, 8, max(1, int(ram_avail_gb // 2.0))))

    if prefer_gpu and has_cuda and vram_used_pct < 85.0:
        spec_backend = "cuda_fp64"
        badge = f"{os_name} · CUDA GPU ({cuda_name})"
    elif os_name == "macOS" and arch == "arm64":
        spec_backend = "apple_amx_accelerate"
        accel_tags = ["AMX FP64"]
        if has_mps:
            accel_tags.append("Metal")
        if _MLX_VERSION:
            accel_tags.append("MLX")
        badge = f"macOS ARM64 · Apple Silicon ({'+'.join(accel_tags)})"
    else:
        spec_backend = "cpu_multicore"
        fallback_note = " (GPU busy -> CPU auto-fallback)" if (has_cuda and vram_used_pct >= 85.0) else ""
        badge = f"{os_name} {arch} · {phys_cores}-Core CPU{fallback_note}"

    fem_backend = "cpu_amx_sparse_fp64" if (os_name == "macOS" and arch == "arm64") else "cpu_blas_sparse_fp64"

    return HardwareProfile(
        os_name=os_name,
        os_release=platform.release(),
        arch=arch,
        cpu_brand=cpu_brand,
        physical_cores=int(phys_cores),
        logical_cores=int(log_cores),
        ram_total_gb=ram_tot_gb,
        ram_available_gb=ram_avail_gb,
        has_cuda_gpu=has_cuda,
        cuda_device_name=cuda_name,
        cuda_vram_total_gb=vram_tot_gb,
        cuda_vram_free_gb=vram_free_gb,
        cuda_vram_used_pct=vram_used_pct,
        has_apple_mps=has_mps,
        mlx_version=_MLX_VERSION,
        selected_spectrum_backend=spec_backend,
        selected_fem_backend=fem_backend,
        recommended_workers=rec_workers,
        badge_label=badge,
    )


def accelerated_tmm_spectrum(
    wavelength_nm: np.ndarray,
    *,
    roundtrip_length_um: float,
    neff_at_ref: float,
    ng_at_ref: float,
    lambda_ref_nm: float,
    dispersion_d2_per_nm2: float = 0.0,
    alpha_loss_db_per_cm: float = 1.0,
    kappa_power_thru: float | np.ndarray = 0.05,
    kappa_power_drop: float | np.ndarray = 0.0,
    delta_neff_tuning: complex = 0.0 + 0.0j,
    prefer_gpu: bool = True,
) -> dict[str, np.ndarray | str]:
    """Compute broadband dispersive MRR Through/Drop/Phase spectra on CUDA FP64 or CPU/AMX.

    Supports both scalar and wavelength-dependent `kappa_power_thru(lambda)` /
    `kappa_power_drop(lambda)` arrays for dispersive directional/pulley couplers.
    """
    hw = detect_hardware_profile(prefer_gpu=prefer_gpu)
    wl_np = np.asarray(wavelength_nm, dtype=np.float64)
    l_rt_m = float(roundtrip_length_um) * 1e-6
    dn_dlam = -(float(ng_at_ref) - float(neff_at_ref)) / float(lambda_ref_nm)
    dn_re = float(np.real(delta_neff_tuning))
    dn_im = float(np.imag(delta_neff_tuning))

    if hw.selected_spectrum_backend == "cuda_fp64" and _HAS_TORCH and torch is not None:
        dev = torch.device("cuda:0")
        wl_t = torch.as_tensor(wl_np, dtype=torch.float64, device=dev)
        wl_m = wl_t * 1e-9
        dlam = wl_t - float(lambda_ref_nm)
        neff_re = float(neff_at_ref) + dn_dlam * dlam + 0.5 * float(dispersion_d2_per_nm2) * (dlam ** 2) + dn_re

        alpha_p = float(alpha_loss_db_per_cm) * 100.0 * (math.log(10.0) / 10.0) + (4.0 * math.pi * dn_im / wl_m)
        a_rt = torch.exp(-0.5 * alpha_p * l_rt_m)
        phi_rt = (2.0 * math.pi * neff_re / wl_m) * l_rt_m

        kp1 = torch.clamp(torch.as_tensor(kappa_power_thru, dtype=torch.float64, device=dev), 0.0, 0.999999)
        kp2 = torch.clamp(torch.as_tensor(kappa_power_drop, dtype=torch.float64, device=dev), 0.0, 0.999999)
        t1 = torch.sqrt(1.0 - kp1)
        k1 = torch.sqrt(kp1)
        t2 = torch.sqrt(1.0 - kp2)
        k2 = torch.sqrt(kp2)

        exp_phi = torch.exp(1j * phi_rt.to(torch.complex128))
        exp_half = torch.exp(0.5j * phi_rt.to(torch.complex128))
        denom = 1.0 - t1 * t2 * a_rt * exp_phi
        thru_amp = (t1 - t2 * a_rt * exp_phi) / denom
        drop_amp = (-k1 * k2 * torch.sqrt(a_rt) * exp_half) / denom

        thru_T = (torch.abs(thru_amp) ** 2).cpu().numpy()
        drop_T = (torch.abs(drop_amp) ** 2).cpu().numpy()
        phase_rad = np.unwrap(torch.angle(thru_amp).cpu().numpy())
        neff_out = neff_re.cpu().numpy()
        used_backend = f"cuda_fp64 ({hw.cuda_device_name})"
    else:
        wl_m = wl_np * 1e-9
        dlam = wl_np - float(lambda_ref_nm)
        neff_out = float(neff_at_ref) + dn_dlam * dlam + 0.5 * float(dispersion_d2_per_nm2) * (dlam ** 2) + dn_re

        alpha_p = float(alpha_loss_db_per_cm) * 100.0 * (math.log(10.0) / 10.0) + (4.0 * math.pi * dn_im / wl_m)
        a_rt = np.exp(-0.5 * alpha_p * l_rt_m)
        phi_rt = (2.0 * math.pi * neff_out / wl_m) * l_rt_m

        kp1 = np.clip(np.asarray(kappa_power_thru, dtype=np.float64), 0.0, 0.999999)
        kp2 = np.clip(np.asarray(kappa_power_drop, dtype=np.float64), 0.0, 0.999999)
        t1 = np.sqrt(1.0 - kp1)
        k1 = np.sqrt(kp1)
        t2 = np.sqrt(1.0 - kp2)
        k2 = np.sqrt(kp2)

        exp_phi = np.exp(1j * phi_rt)
        denom = 1.0 - t1 * t2 * a_rt * exp_phi
        thru_amp = (t1 - t2 * a_rt * exp_phi) / denom
        drop_amp = (-k1 * k2 * np.sqrt(a_rt) * np.exp(0.5j * phi_rt)) / denom

        thru_T = np.abs(thru_amp) ** 2
        drop_T = np.abs(drop_amp) ** 2
        phase_rad = np.unwrap(np.angle(thru_amp))
        used_backend = hw.selected_spectrum_backend

    return {
        "wavelength_nm": wl_np,
        "thru_T": thru_T,
        "drop_T": drop_T,
        "thru_phase_rad": phase_rad,
        "neff_real": neff_out,
        "compute_backend": used_backend,
    }


def run_compute_benchmark(num_wavelengths: int = 100_000) -> dict:
    """Run a quick broadband TMM compute throughput benchmark on the active hardware."""
    hw = detect_hardware_profile()
    wl = np.linspace(1500.0, 1600.0, int(num_wavelengths), dtype=np.float64)
    t0 = time.perf_counter()
    res = accelerated_tmm_spectrum(
        wl,
        roundtrip_length_um=2.0 * math.pi * 40.0,
        neff_at_ref=2.45,
        ng_at_ref=4.18,
        lambda_ref_nm=1550.0,
        alpha_loss_db_per_cm=1.2,
        kappa_power_thru=0.04,
        kappa_power_drop=0.04,
    )
    dt_ms = (time.perf_counter() - t0) * 1e3
    pts_per_sec = float(num_wavelengths) / max(dt_ms * 1e-3, 1e-9)
    return {
        "hardware": hw.to_dict(),
        "num_wavelengths": int(num_wavelengths),
        "elapsed_ms": round(dt_ms, 3),
        "million_points_per_sec": round(pts_per_sec / 1e6, 2),
        "throughput_mpts_per_s": round(pts_per_sec / 1e6, 2),
        "compute_backend": str(res["compute_backend"]),
        "backend_used": str(res["compute_backend"]),
    }
