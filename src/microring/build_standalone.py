"""Cross-Platform Consumer Installer Builder (`macOS .dmg` & `Windows Setup .exe` / `.zip`).

Builds native standalone `MicroRing Studio` desktop installers with zero Python setup
and zero GPL dependencies required on end-user machines:
  - macOS Apple Silicon / Intel:
      python -m microring.build_standalone --target macos
      -> dist/MicroRing-Studio-v3.6.0-macOS-AppleSilicon.dmg (Drag-to-Applications .dmg)
  - Windows 10/11 (CPU or CUDA):
      python -m microring.build_standalone --target windows-cpu
      -> dist/MicroRing-Studio-v3.6.0-Windows-Setup-x64.exe (Inno Setup Wizard Installer)
      -> dist/MicroRing-Studio-v3.6.0-Windows-Portable-x64.zip (Portable unzip-and-run)
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from microring import __version__ as APP_VERSION
except Exception:
    APP_VERSION = "3.6.0"


def generate_app_icons(pkg_dir: Path) -> tuple[Path | None, Path | None]:
    """Generate crisp vector-styled microring resonator icons (.icns & .ico) via Pillow."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    ico_path = pkg_dir / "microring.ico"
    icns_path = pkg_dir / "microring.icns"
    png_path = pkg_dir / "microring_icon_1024.png"

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return (icns_path if icns_path.is_file() else None, ico_path if ico_path.is_file() else None)

    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded dark-obsidian squircle background
    pad = 56
    draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=210,
        fill=(11, 18, 32, 255),
        outline=(56, 189, 248, 190),
        width=12,
    )

    # Concentric pulley bus waveguide arc + horizontal leads (amber-gold)
    cx, cy = size // 2, size // 2 + 28
    r_ring = 248
    r_bus = 316
    bus_col = (245, 158, 11, 255)
    ring_glow = (14, 165, 233, 90)
    ring_col = (56, 189, 248, 255)

    # Outer glow around microring
    draw.ellipse(
        [cx - r_ring - 28, cy - r_ring - 28, cx + r_ring + 28, cy + r_ring + 28],
        outline=ring_glow,
        width=56,
    )
    # Microring cavity
    draw.ellipse(
        [cx - r_ring, cy - r_ring, cx + r_ring, cy + r_ring],
        outline=ring_col,
        width=36,
    )

    # Top wrap-around pulley bus waveguide
    wrap_y = cy - r_bus
    draw.line([(145, wrap_y - 26), (cx - 170, wrap_y - 26)], fill=bus_col, width=26)
    draw.line([(cx + 170, wrap_y - 26), (size - 145, wrap_y - 26)], fill=bus_col, width=26)
    draw.arc(
        [cx - r_bus, cy - r_bus, cx + r_bus, cy + r_bus],
        start=236,
        end=304,
        fill=bus_col,
        width=26,
    )

    # Sharp critical-coupling Lorentzian resonance dip inside the ring
    pts = []
    for i in range(121):
        u = (i / 120.0) * 2.0 - 1.0
        x = cx + int(u * 145)
        lorentz = 1.0 / (1.0 + (u / 0.16) ** 2)
        y = (cy - 52) + int(lorentz * 128)
        pts.append((x, y))
    draw.line(pts, fill=(52, 211, 153, 255), width=18, joint="curve")

    img.save(png_path, format="PNG")
    img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

    if sys.platform == "darwin" and shutil.which("iconutil"):
        iconset = pkg_dir / "microring.iconset"
        if iconset.exists():
            shutil.rmtree(iconset)
        iconset.mkdir(parents=True)
        for pt in (16, 32, 128, 256, 512):
            img.resize((pt, pt), Image.Resampling.LANCZOS).save(iconset / f"icon_{pt}x{pt}.png")
            img.resize((pt * 2, pt * 2), Image.Resampling.LANCZOS).save(iconset / f"icon_{pt}x{pt}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns_path)], check=False)
        shutil.rmtree(iconset, ignore_errors=True)
    else:
        try:
            img.save(icns_path, format="ICNS")
        except Exception:
            pass

    return (icns_path if icns_path.is_file() else None, ico_path if ico_path.is_file() else None)


def _package_macos_dmg(repo_root: Path, app_bundle: Path) -> Path:
    arch = "AppleSilicon" if platform.machine() == "arm64" else "Intel"
    dmg_name = f"MicroRing-Studio-v{APP_VERSION}-macOS-{arch}.dmg"
    dmg_path = repo_root / "dist" / dmg_name
    stage_dir = repo_root / "build" / "dmg_stage"

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Ad-hoc sign .app bundle cleanly for Apple Silicon Gatekeeper compatibility
    if shutil.which("codesign"):
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", str(app_bundle)],
            check=False,
        )

    staged_app = stage_dir / app_bundle.name
    shutil.copytree(app_bundle, staged_app, symlinks=True)
    os.symlink("/Applications", stage_dir / "Applications")

    readme_txt = stage_dir / "HOW_TO_OPEN_ON_MAC.txt"
    readme_txt.write_text(
        "MicroRing Studio - macOS Installation (Zero Python Setup Needed)\n"
        "==============================================================\n\n"
        "1. Drag 'MicroRing Studio.app' into the 'Applications' folder.\n"
        "2. First launch tip (unsigned open-source app on macOS):\n"
        "   - Right-click (or Control-click) 'MicroRing Studio.app' in Applications -> click 'Open'.\n"
        "   - Or if macOS Sequoia/Tahoe blocks an unsigned app, open System Settings ->\n"
        "     Privacy & Security -> scroll down and click 'Open Anyway' once.\n"
        "   - Optional Terminal 1-liner to clear quarantine attribute:\n"
        "     xattr -dr com.apple.quarantine \"/Applications/MicroRing Studio.app\"\n\n"
        "Exported .GDS masks, .LYP files, and .CSV spectra are saved in:\n"
        "  ~/Documents/MicroRing_Studio/\n",
        encoding="utf-8",
    )

    if dmg_path.exists():
        dmg_path.unlink()

    subprocess.run(
        [
            "hdiutil",
            "create",
            "-volname",
            "MicroRing Studio",
            "-srcfolder",
            str(stage_dir),
            "-ov",
            "-format",
            "UDZO",
            str(dmg_path),
        ],
        check=True,
    )
    shutil.rmtree(stage_dir, ignore_errors=True)
    # Clean duplicate unpacked PyInstaller directory, .app staging bundle, and intermediate build/ folder to save ~600 MB disk space
    shutil.rmtree(repo_root / "dist" / "MicroRing Studio", ignore_errors=True)
    shutil.rmtree(app_bundle, ignore_errors=True)
    shutil.rmtree(repo_root / "build", ignore_errors=True)
    print(f"[MicroRing Builder] Created macOS installer DMG: {dmg_path}")
    return dmg_path


def _package_windows_artifacts(repo_root: Path, dist_folder: Path, suffix: str) -> list[Path]:
    created: list[Path] = []
    zip_base = repo_root / "dist" / f"MicroRing-Studio-v{APP_VERSION}-Windows-Portable-{suffix}"
    zip_file = Path(shutil.make_archive(str(zip_base), "zip", root_dir=str(dist_folder.parent), base_dir=dist_folder.name))
    created.append(zip_file)
    print(f"[MicroRing Builder] Created portable Windows ZIP: {zip_file}")

    iss_file = repo_root / "packaging" / "windows_installer.iss"
    iscc_candidates = [
        shutil.which("ISCC") or "",
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
    ]
    iscc = next((p for p in iscc_candidates if p and Path(p).is_file()), None)
    if iscc and iss_file.is_file():
        setup_base = f"MicroRing-Studio-v{APP_VERSION}-Windows-Setup-{suffix}"
        rc = subprocess.call([
            iscc,
            f"/DMyAppVersion={APP_VERSION}",
            f"/DOutputBaseFilename={setup_base}",
            str(iss_file),
        ], cwd=str(repo_root))
        exe_out = repo_root / "dist" / f"{setup_base}.exe"
        if rc == 0 and exe_out.is_file():
            created.append(exe_out)
            print(f"[MicroRing Builder] Created Windows Setup Installer EXE: {exe_out}")
    return created


def build_package(target: str) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    entry = repo_root / "src" / "microring" / "desktop_app.py"
    pkg_dir = repo_root / "packaging"
    icns_path, ico_path = generate_app_icons(pkg_dir)

    app_name = "MicroRing Studio"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        app_name,
        "--specpath",
        str(repo_root / "build"),
        "--paths",
        str(repo_root / "src"),
        "--hidden-import=scipy.sparse.linalg._eigen.arpack",
        "--hidden-import=scipy.sparse.linalg._dsolve.linsolve",
        "--hidden-import=scipy.special._cdflib",
        "--hidden-import=scipy.spatial._qhull",
        "--hidden-import=skfem",
        "--hidden-import=shapely",
        "--hidden-import=gdstk",
        "--hidden-import=lmfit",
        "--hidden-import=uvicorn",
        "--hidden-import=starlette",
        "--hidden-import=anyio._backends._asyncio",
        "--collect-submodules=microring",
        "--collect-submodules=skfem",
        "--collect-submodules=uvicorn",
    ]

    if target == "macos":
        cmd.extend(["--osx-bundle-identifier", "org.microring.studio"])
        if icns_path:
            cmd.extend(["--icon", str(icns_path)])
    else:
        if ico_path:
            cmd.extend(["--icon", str(ico_path)])

    # Exclude optional/dev heavy packages so consumer .dmg/.exe stays compact (~100 MB) and starts in ~1.5 s
    excludes = [
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "tidy3d",
        "gdsfactory",
        "klayout",
        "numba",
        "llvmlite",
        "zmq",
    ]
    if target in ("macos", "windows-cpu"):
        excludes.extend(["torch", "torchaudio", "torchvision", "cupy", "mlx", "triton"])
    for mod in excludes:
        cmd.extend(["--exclude-module", mod])

    cmd.append(str(entry))

    env = os.environ.copy()
    if sys.platform == "darwin":
        # Provide a pure-Python Mach-O universal2 -> thin slice shim if Xcode Command Line Tools (lipo) are absent
        probe = subprocess.run(["lipo", "-info", sys.executable], capture_output=True, text=True)
        if probe.returncode != 0:
            shim_dir = repo_root / "build" / "bin_shims"
            shim_dir.mkdir(parents=True, exist_ok=True)
            lipo_shim = shim_dir / "lipo"
            lipo_shim.write_text(
                f"#!{sys.executable}\n"
                "import os, shutil, stat, struct, sys\n"
                "ARCH = {'arm64': 0x0100000C, 'x86_64': 0x01000007}\n"
                "a = sys.argv[1:]\n"
                "ti = a.index('-thin'); arch, inf = a[ti+1], a[ti+2]\n"
                "outf = a[a.index('-output')+1]\n"
                "d = open(inf, 'rb').read()\n"
                "magic, n = struct.unpack('>II', d[:8])\n"
                "if magic != 0xCAFEBABE:\n"
                "    if inf != outf: shutil.copy2(inf, outf)\n"
                "    raise SystemExit(0)\n"
                "for i in range(n):\n"
                "    cpu, _, off, sz, _ = struct.unpack('>IIIII', d[8+i*20:28+i*20])\n"
                "    if cpu == ARCH[arch]:\n"
                "        m = os.stat(inf).st_mode\n"
                "        open(outf, 'wb').write(d[off:off+sz])\n"
                "        os.chmod(outf, m | stat.S_IXUSR)\n"
                "        raise SystemExit(0)\n"
                "raise SystemExit(1)\n",
                encoding="utf-8",
            )
            lipo_shim.chmod(0o755)
            int_shim = shim_dir / "install_name_tool"
            int_shim.write_text(
                f"#!{sys.executable}\n"
                "import os, stat, sys, lief\n"
                "lief.logging.disable()\n"
                "args = sys.argv[1:]\n"
                "target = args[-1]\n"
                "new_id = None; changes = []; del_rp = set(); add_rp = []\n"
                "i = 0\n"
                "while i < len(args) - 1:\n"
                "    k = args[i]\n"
                "    if k == '-id': new_id = args[i+1]; i += 2\n"
                "    elif k == '-change': changes.append((args[i+1], args[i+2])); i += 3\n"
                "    elif k == '-delete_rpath': del_rp.add(args[i+1]); i += 2\n"
                "    elif k == '-add_rpath': add_rp.append(args[i+1]); i += 2\n"
                "    else: i += 1\n"
                "mode = os.stat(target).st_mode\n"
                "os.chmod(target, mode | stat.S_IWUSR)\n"
                "fat = lief.MachO.parse(target)\n"
                "for b in fat:\n"
                "    if new_id is not None:\n"
                "        for cmd in b.commands:\n"
                "            if cmd.command == lief.MachO.LoadCommand.TYPE.ID_DYLIB:\n"
                "                cmd.name = new_id\n"
                "    for old_p, new_p in changes:\n"
                "        for lib in b.libraries:\n"
                "            if lib.name == old_p: lib.name = new_p\n"
                "    for rp in list(b.rpaths):\n"
                "        if rp.path in del_rp: b.remove(rp)\n"
                "    existing_rp = {rp.path for rp in b.rpaths}\n"
                "    for rp_str in add_rp:\n"
                "        if rp_str not in existing_rp:\n"
                "            b.add(lief.MachO.RPathCommand.create(rp_str))\n"
                "fat.write(target)\n"
                "os.chmod(target, mode)\n",
                encoding="utf-8",
            )
            int_shim.chmod(0o755)
            env["PATH"] = f"{shim_dir}:{env.get('PATH', '')}"

    print("[MicroRing Builder] Running PyInstaller:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(repo_root), env=env)
    if rc != 0:
        return rc

    # Smoke-test the compiled binary (starts local ASGI server + solves 2D Maxwell FEM mode!)
    if target == "macos":
        app_bundle = repo_root / "dist" / f"{app_name}.app"
        bin_path = app_bundle / "Contents" / "MacOS" / app_name
        print(f"[MicroRing Builder] Smoke-testing compiled macOS binary: {bin_path}")
        chk = subprocess.call([str(bin_path), "--headless-check"])
        if chk != 0:
            print("[MicroRing Builder] ERROR: Compiled macOS app failed --headless-check!")
            return chk
        _package_macos_dmg(repo_root, app_bundle)
    else:
        dist_folder = repo_root / "dist" / app_name
        bin_path = dist_folder / f"{app_name}.exe"
        print(f"[MicroRing Builder] Smoke-testing compiled Windows binary: {bin_path}")
        chk = subprocess.call([str(bin_path), "--headless-check"])
        if chk != 0:
            print("[MicroRing Builder] ERROR: Compiled Windows .exe failed --headless-check!")
            return chk
        suffix = "x64-CUDA" if target == "windows-cuda" else "x64"
        _package_windows_artifacts(repo_root, dist_folder, suffix)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consumer .dmg / .exe installers for MicroRing Studio")
    parser.add_argument("--target", choices=["macos", "windows-cpu", "windows-cuda"], required=True)
    args = parser.parse_args()
    raise SystemExit(build_package(args.target))


if __name__ == "__main__":
    main()
