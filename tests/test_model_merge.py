"""Pure tests for the offline model-merge logic (NO torch).

Exercises the validation + bookkeeping (strength normalization, architecture
compatibility, enc_q exclusion, metadata carry-over) on fabricated checkpoint
dicts whose "tensors" are plain objects. The actual tensor blend (blend_weights,
which imports torch) is covered by the .venv-applio smoke, not here.
"""

from __future__ import annotations

import pytest

from src.engine.model_merge import (
    MergeError,
    blend_weights,
    build_merged_checkpoint,
    check_mergeable,
    checkpoint_meta,
    normalize_strengths,
    weight_dict,
)


def _cpt(keys, sr=40000, version="v2", f0=1, vocoder="HiFi-GAN", config=None):
    if config is None:
        config = [1025, 32, 192, 109, sr]   # config[-1] = sr
    return {
        "weight": {k: object() for k in keys},
        "config": config,
        "version": version,
        "f0": f0,
        "vocoder": vocoder,
    }


# ---- normalize_strengths ----

def test_normalize_strengths_equal():
    assert normalize_strengths([1, 1]) == [0.5, 0.5]


def test_normalize_strengths_weighted():
    assert normalize_strengths([3, 1]) == [0.75, 0.25]


def test_normalize_strengths_rejects_negative():
    with pytest.raises(MergeError):
        normalize_strengths([1, -1])


def test_normalize_strengths_rejects_all_zero():
    with pytest.raises(MergeError):
        normalize_strengths([0, 0])


# ---- checkpoint_meta ----

def test_checkpoint_meta_excludes_enc_q():
    meta = checkpoint_meta(_cpt(["emb_g.weight", "enc_q.enc.0", "dec.conv"]))
    assert meta["keys"] == frozenset({"emb_g.weight", "dec.conv"})
    assert meta["sr"] == 40000 and meta["version"] == "v2" and meta["f0"] == 1


def test_weight_dict_rejects_missing():
    with pytest.raises(MergeError):
        weight_dict({"config": [1]})


def test_checkpoint_meta_rejects_no_config():
    with pytest.raises(MergeError):
        checkpoint_meta({"weight": {"a": object()}})


# ---- check_mergeable ----

def test_check_mergeable_accepts_matching():
    keys = ["emb_g.weight", "dec.conv"]
    common = check_mergeable([checkpoint_meta(_cpt(keys)), checkpoint_meta(_cpt(keys))])
    assert common["sr"] == 40000


def test_check_mergeable_needs_two():
    with pytest.raises(MergeError):
        check_mergeable([checkpoint_meta(_cpt(["a"]))])


def test_check_mergeable_rejects_non_v2():
    keys = ["emb_g.weight"]
    with pytest.raises(MergeError):
        check_mergeable([checkpoint_meta(_cpt(keys, version="v1")),
                         checkpoint_meta(_cpt(keys))])


def test_check_mergeable_rejects_sr_mismatch():
    keys = ["emb_g.weight"]
    with pytest.raises(MergeError):
        check_mergeable([checkpoint_meta(_cpt(keys, sr=40000)),
                         checkpoint_meta(_cpt(keys, sr=48000))])


def test_check_mergeable_rejects_key_mismatch():
    with pytest.raises(MergeError):
        check_mergeable([checkpoint_meta(_cpt(["a", "b"])),
                         checkpoint_meta(_cpt(["a", "c"]))])


def test_check_mergeable_rejects_vocoder_mismatch():
    keys = ["a"]
    with pytest.raises(MergeError):
        check_mergeable([checkpoint_meta(_cpt(keys, vocoder="HiFi-GAN")),
                         checkpoint_meta(_cpt(keys, vocoder="RefineGAN"))])


# ---- blend_weights input guards (validation runs before the lazy torch import) ----

def test_blend_weights_rejects_length_mismatch():
    with pytest.raises(MergeError):
        blend_weights([{"a": object()}, {"a": object()}], [1.0])


def test_blend_weights_needs_two():
    with pytest.raises(MergeError):
        blend_weights([{"a": object()}], [1.0])


def test_blend_weights_rejects_asymmetric_keys():
    # a key present ONLY in a later model must raise, not be silently dropped
    with pytest.raises(MergeError):
        blend_weights([{"a": object()}, {"a": object(), "b": object()}], [0.5, 0.5])


# ---- build_merged_checkpoint ----

def test_build_merged_checkpoint_carries_metadata():
    base = _cpt(["a"], config=[1, 2, 40000])
    base["extra"] = "keep"
    merged_w = {"a": object()}
    out = build_merged_checkpoint(base, merged_w)
    assert out["weight"] is merged_w
    assert out["config"] == [1, 2, 40000]
    assert out["version"] == "v2"
    assert out["extra"] == "keep"
    assert "model" not in out
