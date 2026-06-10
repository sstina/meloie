"""Pure tests for the GUI model-discovery + recipe-lookup (no torch, no Qt).

Covers the models/ folder discovery and the per-model recipe (a matching
``<stem>.json`` profile supplies pitch/index/...; otherwise neutral defaults).
Uses tmp dirs + monkeypatch so it never depends on real model files.
"""

from __future__ import annotations

import json

from src.engine.streaming_engine import StreamingEngineConfig
from src.ui import config_assembly as ca


def test_list_model_files_lists_pth_by_stem(tmp_path, monkeypatch):
    (tmp_path / "B.pth").write_bytes(b"x")
    (tmp_path / "A.pth").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("nope", encoding="utf-8")
    monkeypatch.setattr(ca, "models_dir", lambda: str(tmp_path))
    ms = ca.list_model_files()
    assert [m["name"] for m in ms] == ["A", "B"]      # sorted, .pth only, stem = name
    assert ms[0]["path"].endswith("A.pth")


def test_list_model_files_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "models_dir", lambda: str(tmp_path / "nope"))
    assert ca.list_model_files() == []


def test_list_model_files_recurses_subfolders(tmp_path, monkeypatch):
    # .pth at any depth must be found; nested ones get a forward-slashed subpath
    # name so same-filename models in different folders stay distinguishable.
    (tmp_path / "A.pth").write_bytes(b"x")
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "C.pth").write_bytes(b"x")
    deep = tmp_path / "voices" / "deep"
    deep.mkdir()
    (deep / "Nested.pth").write_bytes(b"x")
    (deep / "skip.bin").write_bytes(b"x")          # non-.pth ignored at any depth
    monkeypatch.setattr(ca, "models_dir", lambda: str(tmp_path))

    ms = ca.list_model_files()
    names = [m["name"] for m in ms]
    assert names == ["A", "voices/C", "voices/deep/Nested"]   # sorted, recursive
    by_name = {m["name"]: m["path"] for m in ms}
    assert by_name["voices/deep/Nested"].endswith("Nested.pth")


def _write_profile(tmp_path, stem, **fields):
    payload = {"name": stem, "model_path": f"models/{stem}.pth", **fields}
    (tmp_path / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_model_default_params_from_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    _write_profile(tmp_path, "X", pitch_shift=12, index_rate=0.5,
                   index_path="models/V2.index", protect=0.33)
    p = ca.model_default_params("anywhere/X.pth")
    assert p["pitch_shift"] == 12 and p["index_rate"] == 0.5 and p["has_index"] is True


def test_model_default_params_defaults_without_profile(tmp_path, monkeypatch):
    # No profile AND no .index anywhere -> has_index False (hermetic empty models dir).
    mdir = tmp_path / "models"; mdir.mkdir()
    (mdir / "NOPROFILE.pth").write_bytes(b"x")
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr(ca, "models_dir", lambda: str(mdir))
    p = ca.model_default_params(str(mdir / "NOPROFILE.pth"))
    assert p["pitch_shift"] == 0 and p["index_rate"] == 0.0 and p["has_index"] is False
    assert p["formant_on"] is False                 # neutral default -> formant off


def test_model_default_params_has_index_from_sibling_without_profile(tmp_path, monkeypatch):
    # No profile but a sibling .index exists -> has_index True so the slider is live.
    mdir = tmp_path / "models"; mdir.mkdir()
    (mdir / "NOPROF.pth").write_bytes(b"x")
    (mdir / "NOPROF.index").write_bytes(b"x")
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr(ca, "models_dir", lambda: str(mdir))
    assert ca.model_default_params(str(mdir / "NOPROF.pth"))["has_index"] is True


def test_model_default_params_derives_formant_on(tmp_path, monkeypatch):
    # a saved gender shift (timbre != 1.0) must surface as formant_on=True so the GUI
    # checkbox reflects it; the dead formant_qfrency key is no longer emitted.
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    _write_profile(tmp_path, "F", formant_timbre=1.25)
    p = ca.model_default_params("anywhere/F.pth")
    assert p["formant_on"] is True and p["formant_timbre"] == 1.25
    assert "formant_qfrency" not in p
    _write_profile(tmp_path, "N", formant_timbre=1.0)
    assert ca.model_default_params("anywhere/N.pth")["formant_on"] is False


def test_build_configs_for_model_applies_profile_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    _write_profile(tmp_path, "X", pitch_shift=12, index_rate=0.5,
                   index_path="models/V2.index")
    scfg, acfg = ca.build_configs_for_model("models/X.pth", None, None, "fcpe")
    assert scfg.model_path == "models/X.pth"          # the discovered .pth, not the profile's
    assert scfg.pitch_shift == 12
    assert scfg.index_path == "models/V2.index"       # recipe from the profile
    assert scfg.f0_method == "fcpe"
    assert acfg.channels == 1


def test_build_configs_for_model_defaults_without_profile(tmp_path, monkeypatch):
    # No profile and no .index anywhere -> empty index_path (hermetic).
    mdir = tmp_path / "models"; mdir.mkdir()
    (mdir / "NOPROF.pth").write_bytes(b"x")
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr(ca, "models_dir", lambda: str(mdir))
    scfg, _ = ca.build_configs_for_model(str(mdir / "NOPROF.pth"), None, None, "rmvpe")
    assert scfg.pitch_shift == 0 and scfg.index_path == "" and scfg.index_rate == 0.0


def test_build_configs_for_model_defaults_to_sibling_index(tmp_path, monkeypatch):
    # No profile -> the index defaults to the .pth's own .index. Mirrors the real
    # case: models/1/GentleF_40k_ep15.pth picks up models/1/GentleF_40k.index.
    sub = tmp_path / "models" / "1"; sub.mkdir(parents=True)
    (sub / "GentleF_40k_ep15.pth").write_bytes(b"x")
    (sub / "GentleF_40k.index").write_bytes(b"x")   # different stem, same folder
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr(ca, "models_dir", lambda: str(tmp_path / "models"))
    scfg, _ = ca.build_configs_for_model(
        str(sub / "GentleF_40k_ep15.pth"), None, None, "fcpe")
    assert scfg.index_path.endswith("GentleF_40k.index")
    assert scfg.index_rate == 0.0     # loaded so the slider is live, but inert until raised


def test_build_configs_for_model_profile_index_wins_over_default(tmp_path, monkeypatch):
    # An explicit profile index_path is honoured even if a sibling .index exists.
    mdir = tmp_path / "models"; mdir.mkdir()
    (mdir / "X.pth").write_bytes(b"x")
    (mdir / "X.index").write_bytes(b"x")            # sibling that would otherwise win
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    monkeypatch.setattr(ca, "models_dir", lambda: str(mdir))
    _write_profile(tmp_path, "X", index_rate=0.5, index_path="models/V2.index")
    scfg, _ = ca.build_configs_for_model(str(mdir / "X.pth"), None, None, "fcpe")
    assert scfg.index_path == "models/V2.index"     # explicit profile value, not X.index


# ----------------------------------------------------------------- game mode
def _base_scfg():
    return StreamingEngineConfig(
        model_path="m.pth", index_path="i.index", f0_method="fcpe",
        pitch_shift=12, index_rate=0.5, block_ms=250.0, context_ms=2500.0,
        device="cuda", cpu_threads=0,
    )


def test_game_modes_table_has_the_three_modes():
    assert set(ca.GAME_MODES) == {"off", "dgpu_light", "cpu_zero"}
    assert ca.GAME_MODES["off"] == {}                # off never overrides


def test_apply_game_mode_off_is_identity():
    base = _base_scfg()
    out = ca.apply_game_mode(base, "off")
    assert out is base                               # off returns the same object, untouched
    assert out.device == "cuda" and out.block_ms == 250.0 and out.cpu_threads == 0


def test_apply_game_mode_unknown_is_identity():
    base = _base_scfg()
    assert ca.apply_game_mode(base, "garbage") is base
    assert ca.apply_game_mode(base, None) is base


def test_apply_game_mode_cpu_zero_overrides_only_loadtime_fields():
    base = _base_scfg()
    out = ca.apply_game_mode(base, "cpu_zero")
    assert out is not base                           # active mode returns a NEW config
    assert base.device == "cuda" and base.block_ms == 250.0   # input untouched (pure)
    assert out.device == "cpu"
    assert out.block_ms == 500.0 and out.context_ms == 1000.0
    assert out.cpu_threads == 8
    # recipe preserved (only device/block/context/cpu_threads change)
    assert out.model_path == "m.pth" and out.pitch_shift == 12 and out.index_rate == 0.5
    assert out.index_path == "i.index" and out.f0_method == "fcpe"


def test_apply_game_mode_dgpu_light_stays_on_cuda():
    out = ca.apply_game_mode(_base_scfg(), "dgpu_light")
    assert out.device == "cuda"
    assert out.block_ms == 500.0 and out.context_ms == 1500.0
    assert out.cpu_threads == 0                      # GPU path: no CPU thread pin
    assert out.pitch_shift == 12                     # recipe preserved


def test_build_configs_for_model_threads_game_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "PROFILES_DIR", str(tmp_path))
    _write_profile(tmp_path, "X", pitch_shift=7, index_rate=0.3,
                   index_path="models/V2.index")
    scfg, _ = ca.build_configs_for_model("models/X.pth", None, None, "fcpe",
                                         game_mode="cpu_zero")
    assert scfg.device == "cpu" and scfg.block_ms == 500.0 and scfg.cpu_threads == 8
    assert scfg.pitch_shift == 7 and scfg.index_rate == 0.3   # recipe still applied
    # default game_mode is off -> normal GPU bundle
    scfg2, _ = ca.build_configs_for_model("models/X.pth", None, None, "fcpe")
    assert scfg2.device == "cuda" and scfg2.block_ms == 250.0 and scfg2.cpu_threads == 0
