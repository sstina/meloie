"""Tests for default .index resolution (find_default_index / models_root_for).

Pure filesystem logic, exercised with tmp dirs — no torch, no model files. Pins
the priority: same-stem (own dir -> recursive) before first-.index (own dir ->
recursive), with case-SENSITIVE stem matching and Unicode names.
"""

from __future__ import annotations

import os

from meloie.engine.model_profile import find_default_index, models_root_for


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def test_tier1_same_stem_in_own_dir_wins(tmp_path):
    root = tmp_path / "models"
    _touch(root / "Foo.pth")
    _touch(root / "Foo.index")        # same stem, own dir
    _touch(root / "aaa.index")        # alphabetically first, but not same-stem
    got = find_default_index(str(root / "Foo.pth"), str(root))
    assert os.path.basename(got) == "Foo.index"


def test_tier2_same_stem_recursive_beats_first_in_own_dir(tmp_path):
    root = tmp_path / "models"
    sub = root / "m"
    _touch(sub / "Foo.pth")
    _touch(sub / "bar.index")         # own-dir, different stem (would be tier 3)
    _touch(root / "Foo.index")        # same stem, elsewhere under models/ (tier 2)
    got = find_default_index(str(sub / "Foo.pth"), str(root))
    # name match (recursive) is preferred over a different-name index in own dir
    assert os.path.basename(got) == "Foo.index"
    assert os.path.samefile(got, root / "Foo.index")


def test_tier3_first_index_in_own_dir_when_no_same_stem(tmp_path):
    # The real GentleF case: stems differ, so fall back to first .index in own dir.
    root = tmp_path / "models"
    sub = root / "1"
    _touch(sub / "GentleF_40k_ep15.pth")
    _touch(sub / "GentleF_40k.index")
    got = find_default_index(str(sub / "GentleF_40k_ep15.pth"), str(root))
    assert os.path.basename(got) == "GentleF_40k.index"


def test_tier3_first_is_alphabetical(tmp_path):
    root = tmp_path / "models"
    _touch(root / "Z.pth")
    _touch(root / "bbb.index")
    _touch(root / "aaa.index")
    got = find_default_index(str(root / "Z.pth"), str(root))
    assert os.path.basename(got) == "aaa.index"


def test_tier4_first_index_recursive_when_own_dir_empty(tmp_path):
    root = tmp_path / "models"
    sub = root / "deep"
    _touch(sub / "Solo.pth")          # own dir has NO .index
    _touch(root / "elsewhere" / "only.index")
    got = find_default_index(str(sub / "Solo.pth"), str(root))
    assert os.path.basename(got) == "only.index"


def test_returns_empty_when_no_index_anywhere(tmp_path):
    root = tmp_path / "models"
    _touch(root / "Lonely.pth")
    assert find_default_index(str(root / "Lonely.pth"), str(root)) == ""


def test_same_stem_match_is_case_sensitive(tmp_path):
    # 'foo' must NOT match a .pth stem 'Foo' (区分大小写). With only foo.index and
    # bar.index present, neither is a same-stem match, so tier 3 (first) -> bar.index.
    root = tmp_path / "models"
    _touch(root / "Foo.pth")
    _touch(root / "foo.index")        # case differs -> NOT a same-stem match
    _touch(root / "bar.index")
    got = find_default_index(str(root / "Foo.pth"), str(root))
    assert os.path.basename(got) == "bar.index"


def test_unicode_same_stem_match(tmp_path):
    root = tmp_path / "models"
    _touch(root / "中文模型.pth")
    _touch(root / "中文模型.index")
    _touch(root / "aaa.index")
    got = find_default_index(str(root / "中文模型.pth"), str(root))
    assert os.path.basename(got) == "中文模型.index"


def test_models_root_for_finds_models_ancestor(tmp_path):
    sub = tmp_path / "models" / "1" / "deep"
    sub.mkdir(parents=True)
    root = models_root_for(str(sub / "x.pth"))
    assert os.path.basename(root).lower() == "models"
    assert os.path.samefile(root, tmp_path / "models")


def test_models_root_for_falls_back_to_own_dir(tmp_path):
    d = tmp_path / "loose"
    d.mkdir()
    root = models_root_for(str(d / "x.pth"))
    assert os.path.samefile(root, d)


def test_find_default_index_derives_root_when_omitted(tmp_path):
    # With models_root omitted, recursion is bounded by the nearest 'models' ancestor.
    sub = tmp_path / "models" / "1"
    _touch(sub / "GentleF_40k_ep15.pth")
    _touch(sub / "GentleF_40k.index")
    got = find_default_index(str(sub / "GentleF_40k_ep15.pth"))
    assert os.path.basename(got) == "GentleF_40k.index"
