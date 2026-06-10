"""Pure tests for the saved precise-mapping store (meloie/ui/precise_store.py).

No Qt — json + numpy round-trips in a tmp dir. Pins: save/list/load round-trip,
filename sanitization (no path traversal, display name kept), tolerant listing
(skips junk / array-less files), and malformed-file rejection on load.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from meloie.ui import precise_store as ps


def _q():
    return np.linspace(6.0, 8.0, 48), np.linspace(7.0, 9.0, 48)


def test_save_list_load_roundtrip(tmp_path):
    src, tgt = _q()
    p = ps.save_precise_map("我的→模型A", "rmvpe", "me.wav", "A.wav", src, tgt, str(tmp_path))
    assert p.endswith(".json")

    lst = ps.list_precise_maps(str(tmp_path))
    assert len(lst) == 1
    m0 = lst[0]
    assert m0["name"] == "我的→模型A" and m0["method"] == "rmvpe"
    assert m0["voice_name"] == "me.wav" and m0["target_name"] == "A.wav"
    assert m0["file"] == p

    m = ps.load_precise_map(p)
    assert np.allclose(m["src_q"], src) and np.allclose(m["tgt_q"], tgt)
    assert m["method"] == "rmvpe" and m["name"] == "我的→模型A"


def test_sanitize_illegal_filename_chars(tmp_path):
    src, tgt = _q()
    p = ps.save_precise_map('a/b:c*?', "fcpe", "", "", src, tgt, str(tmp_path))
    # file stays inside maps_dir (illegal chars replaced -> no traversal)
    assert os.path.dirname(os.path.abspath(p)) == os.path.abspath(str(tmp_path))
    d = json.loads(open(p, encoding="utf-8").read())
    assert d["name"] == 'a/b:c*?'                    # display name kept verbatim in payload


def test_list_skips_junk_and_arrayless(tmp_path):
    (tmp_path / "bad.json").write_text("{ not json", encoding="utf-8")
    (tmp_path / "meta_only.json").write_text(
        json.dumps({"name": "x", "method": "rmvpe"}), encoding="utf-8")   # no arrays
    (tmp_path / "notjson.txt").write_text("nope", encoding="utf-8")
    src, tgt = _q()
    ps.save_precise_map("good", "rmvpe", "", "", src, tgt, str(tmp_path))
    assert [m["name"] for m in ps.list_precise_maps(str(tmp_path))] == ["good"]


def test_list_empty_when_dir_missing(tmp_path):
    assert ps.list_precise_maps(str(tmp_path / "nope")) == []


def test_load_rejects_malformed(tmp_path):
    mismatch = tmp_path / "m.json"
    mismatch.write_text(json.dumps({"name": "x", "src_q": [1, 2], "tgt_q": [1]}),
                        encoding="utf-8")
    with pytest.raises(ValueError):
        ps.load_precise_map(str(mismatch))
    tooshort = tmp_path / "s.json"
    tooshort.write_text(json.dumps({"src_q": [1], "tgt_q": [1]}), encoding="utf-8")
    with pytest.raises(ValueError):
        ps.load_precise_map(str(tooshort))
