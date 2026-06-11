"""Import-safety guard for the validation tools.

``tools.verify_cable_route`` and ``tools.offline_infer`` are CLI scripts
that lazy-import ``sounddevice`` / the RVC stack inside their ``main()``.
Importing the module by itself must NOT pull sounddevice in, so the
unit-test process can load them on machines where sounddevice is not
installed.
"""

from __future__ import annotations

import builtins
import importlib
import sys


def _import_under_trip_wire(module_name: str, monkeypatch) -> None:
    for name in list(sys.modules):
        if name == module_name or name.startswith(module_name + "."):
            del sys.modules[name]
    had_sounddevice_before = "sounddevice" in sys.modules

    orig_import = builtins.__import__

    def _forbidden_import(name, *args, **kwargs):
        if name == "sounddevice" or name.startswith("sounddevice."):
            raise AssertionError(
                f"{module_name} triggered an import of sounddevice at "
                "import time; it must stay lazy."
            )
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _forbidden_import)
    module = importlib.import_module(module_name)
    if not had_sounddevice_before:
        assert "sounddevice" not in sys.modules
    assert hasattr(module, "main")


def test_importing_verify_cable_route_is_lazy(monkeypatch):
    _import_under_trip_wire("tools.verify_cable_route", monkeypatch)


def test_importing_offline_infer_is_lazy(monkeypatch):
    _import_under_trip_wire("tools.offline_infer", monkeypatch)


def test_importing_merge_models_is_lazy(monkeypatch):
    _import_under_trip_wire("tools.merge_models", monkeypatch)


def test_importing_analyze_model_f0_is_lazy(monkeypatch):
    _import_under_trip_wire("tools.analyze_model_f0", monkeypatch)


def test_importing_measure_formant_is_lazy(monkeypatch):
    _import_under_trip_wire("tools.measure_formant", monkeypatch)
