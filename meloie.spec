# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller ONEDIR build of the Meloie GUI.

Build (from RVC/, inside .venv-applio with caches redirected via setup_env_applio.ps1):
    pyinstaller meloie.spec --noconfirm --distpath dist --workpath build

Output: dist/Meloie/Meloie.exe (+ _internal/). torch+CUDA make this ~8-10 GB; ONEFILE
is impractical, so this is ONEDIR.

EXTERNAL data the user stages NEXT TO Meloie.exe (NOT bundled — large / writable):
    <exe>/models/<your>.pth (+ .index)
    <exe>/models/predictors/{rmvpe.pt,fcpe.pt}
    <exe>/models/embedders/contentvec/{pytorch_model.bin,config.json}
    <exe>/config/{model_profiles,precise_maps}   (auto-created on first save)
    <exe>/icon.svg
The engine resolves these via app_base_dir() (= the exe dir when frozen).

Console is TRUE for this build so the SELFTEST line + any traceback is visible. For a
shipping build flip `console=False` (windowed).
"""
import os

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# QML graph — not Python, so bundle as data at the package-relative path the
# frozen _QML_DIR (= <_MEIPASS>/meloie/ui/qml) expects.
datas += [(os.path.join("meloie", "ui", "qml"), os.path.join("meloie", "ui", "qml"))]

# Ship meloie/core .py SOURCE on disk: the synthesizer uses @torch.jit.script
# (synth/commons.py fused_add_tanh_sigmoid_multiply), and TorchScript does
# inspect.getsource at import -> it needs the original .py at the module's
# import path, not just the PYZ bytecode.
for root, _dirs, files in os.walk(os.path.join("meloie", "core")):
    if "__pycache__" in root:
        continue
    for f in files:
        if f.endswith(".py"):
            datas += [(os.path.join(root, f), root)]

# Heavy native stacks (lazy and/or data-dependent). collect_all grabs py + dlls + data
# (e.g. torch/lib CUDA dlls, faiss.dll, libportaudio, llvmlite, torchfcpe asset
# weights, libsndfile). Each wrapped so an absent optional pkg is skipped.
for pkg in [
    "torch", "torchaudio", "transformers", "faiss", "librosa", "numba", "llvmlite",
    "torchfcpe", "sounddevice", "soundfile", "soxr", "scipy",
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

hiddenimports += ["PySide6.QtSvg"]

a = Analysis(
    ["run_meloie.py"],
    pathex=[],
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
