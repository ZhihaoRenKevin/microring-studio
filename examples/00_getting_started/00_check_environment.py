"""Recipe 00a: Verify MicroRing Studio environment, GPU/AMX/CPU compute & 29 materials.

Run:
    uv run python examples/00_getting_started/00_check_environment.py
"""
from __future__ import annotations

from pathlib import Path
import sys

# Allow running directly without editable install
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from microring import MATERIAL_CATALOG, T_REF_K, __version__, detect_hardware_profile, run_compute_benchmark


def main() -> None:
    prof = detect_hardware_profile()
    bench = run_compute_benchmark(100_000)

    print("=" * 74)
    print(f"MicroRing Studio v{__version__} - Environment & Hardware Verification")
    print("=" * 74)
    print(f"OS / Arch          : {prof.os_name} ({prof.arch})")
    print(f"CPU                : {prof.cpu_brand} ({prof.cpu_cores} cores)")
    print(f"GPU / Accelerator  : {prof.gpu_name or 'CPU Vectorized'}")
    print(f"Selected Backend   : {prof.recommended_backend}")
    print(
        f"100k-pt TMM Speed  : {bench['elapsed_ms']:.2f} ms "
        f"({bench['throughput_mpts_per_s']:.2f} M pts/s)"
    )
    print("-" * 74)
    print(f"{'Key':<18} {'Category':<15} {'Ref Band [nm]':<14} {'n_x (300 K)':<12} {'r_33 [pm/V]'}")
    print("-" * 74)
    for key, meta in MATERIAL_CATALOG.items():
        wl_m = float(meta["default_band_nm"]) * 1e-9
        t = meta["fn"](wl_m, T_REF_K)
        print(f"{key:<18} {meta['category']:<15} {meta['default_band_nm']:<14.0f} {t.nx:<12.4f} {t.r33_pm_per_v:.1f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
