# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller ONEDIR build of the Meloie RVC GUI.

Build (from RVC/, inside .venv-applio with caches redirected via setup_env_applio.ps1):
    pyinstaller meloie.spec --noconfirm --distpath dist --workpath build

Output: dist/Meloie/Meloie.exe (+ _internal/). torch+CUDA make this ~8-10 GB; ONEFILE
is impractical, so this is ONEDIR.

EXTERNAL data the user stages NEXT TO Meloie.exe (NOT bundled — large / writable):
    <exe>/rvc/configs/{48000,40000,32000,24000}.json
    <exe>/rvc/models/predictors/{rmvpe.pt,fcpe.pt}
    <exe>/rvc/models/embedders/contentvec/{pytorch_model.bin,config.json}
    <exe>/models/<your>.pth (+ .index)
    <exe>/config/{model_profiles,precise_maps}   (auto-created on first save)
    <exe>/icon.svg
The vendored loaders resolve these against CWD, which app.main() chdir's to the exe dir.

Console is TRUE for this build so the SELFTEST line + any traceback is visible. For a
shipping build flip `console=False` (windowed).
"""
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# Make the vendored `rvc` package importable at SPEC-EVAL time so collect_submodules
# can walk it, and add it to pathex so Analysis compiles those modules into the bundle.
VENDOR = os.path.abspath(os.path.join("src", "vendor", "applio"))
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

datas = []
binaries = []
hiddenimports = []

# QML graph (15 .qml) — not Python, so bundle as data at the package-relative path the
# frozen _QML_DIR (= <_MEIPASS>/src/ui/qml) expects.
datas += [(os.path.join("src", "ui", "qml"), os.path.join("src", "ui", "qml"))]

# Vendored Applio `rvc` package: imported lazily at runtime (sys.path trick), so the
# static graph never sees it — force-collect every submodule.
hiddenimports += collect_submodules("rvc")
# AND ship the rvc .py SOURCE on disk: the synthesizer uses @torch.jit.script
# (fused_add_tanh_sigmoid_multiply), and TorchScript does inspect.getsource at import
# -> it needs the original .py, not just the PYZ bytecode. 0.6 MB of code.
datas += [(os.path.join("src", "vendor", "applio", "rvc"), "rvc")]

# Heavy native stacks (lazy and/or data-dependent). collect_all grabs py + dlls + data
# (e.g. torch/lib CUDA dlls, faiss.dll, libportaudio, llvmlite, torchfcpe/torchcrepe
# asset weights, libsndfile). Each wrapped so an absent optional pkg is skipped.
for pkg in [
    "torch", "torchaudio", "transformers", "faiss", "librosa", "numba", "llvmlite",
    "torchfcpe", "torchcrepe", "sounddevice", "soundfile", "soxr", "scipy",
    "noisereduce", "stftpitchshift", "tokenizers", "safetensors", "huggingface_hub",
    "regex", "audioread", "pooch", "lazy_loader", "joblib", "sklearn",
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:                       # optional / not installed -> skip
        print(f"[meloie.spec] collect_all({pkg!r}) skipped: {exc}")

hiddenimports += ["wget", "PySide6.QtSvg"]

a = Analysis(
    ["run_meloie.py"],
    pathex=[VENDOR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "PyQt5", "PyQt6",
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
        "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
        "IPython", "pytest", "notebook", "jupyter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Meloie",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,                 # TEST build: show SELFTEST output + tracebacks
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Meloie",
)
