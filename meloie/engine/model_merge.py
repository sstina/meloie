"""Offline RVC model merge — blend N v2 .pth checkpoints into one hybrid voice.

捏脸 / voice-morphing: a convex weighted average of the model state dicts. The
merged MODEL defines the voice; the realtime runtime stays a faithful courier (no
output shaping) -> this is contract-safe. Only same-architecture v2 models merge
(identical sampling rate, f0 flag, vocoder, weight key-set and per-key shape);
``enc_q.*`` (the training-only posterior encoder, unused at inference -- our loader
is ``strict=False``) is dropped.

Pure + torch-free at import: torch is imported lazily inside :func:`blend_weights`
only, so this module (and the CLI that wraps it) imports without the heavy stack.
The blend logic lives here (not in the CLI) so it is unit-testable on plain dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List

ENC_Q_PREFIX = "enc_q."


class MergeError(Exception):
    """Models cannot be merged (incompatible) or the inputs are invalid."""


def normalize_strengths(strengths: List[float]) -> List[float]:
    """Scale per-model blend strengths to a convex weighting (sum 1)."""
    vals = [float(s) for s in strengths]
    if any(v < 0 for v in vals):
        raise MergeError(f"merge strengths must be >= 0, got {vals}")
    total = sum(vals)
    if total <= 0:
        raise MergeError("merge strengths sum to 0; give at least one positive weight")
    return [v / total for v in vals]


def weight_dict(cpt: Dict[str, Any]) -> Dict[str, Any]:
    """The trainable state dict inside a checkpoint. Our v2 .pth stores it under
    ``weight`` (what the vendored loader reads); ``model`` is accepted as a
    fallback (RVC training-format checkpoints)."""
    if isinstance(cpt, dict) and isinstance(cpt.get("weight"), dict):
        return cpt["weight"]
    if isinstance(cpt, dict) and isinstance(cpt.get("model"), dict):
        return cpt["model"]
    raise MergeError("checkpoint has no 'weight' (or 'model') state dict")


def _merge_keys(weight: Dict[str, Any]) -> frozenset:
    return frozenset(k for k in weight.keys() if not k.startswith(ENC_Q_PREFIX))


def checkpoint_meta(cpt: Dict[str, Any]) -> Dict[str, Any]:
    """Architecture fingerprint used to decide mergeability (no torch needed)."""
    weight = weight_dict(cpt)
    config = cpt.get("config")
    if not isinstance(config, (list, tuple)) or len(config) == 0:
        raise MergeError("checkpoint has no usable 'config'")
    return {
        "version": str(cpt.get("version", "v1")).lower(),
        "sr": int(config[-1]),
        "f0": int(cpt.get("f0", 1)),
        "vocoder": str(cpt.get("vocoder", "HiFi-GAN")),
        "config": list(config),
        "keys": _merge_keys(weight),
    }


def check_mergeable(metas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify every model shares architecture; return the common meta or raise."""
    if len(metas) < 2:
        raise MergeError("need at least 2 models to merge")
    base = metas[0]
    if base["version"] != "v2":
        raise MergeError(f"v2-only merge: model 0 is {base['version']!r} (not v2)")
    for i, m in enumerate(metas[1:], start=1):
        if m["version"] != "v2":
            raise MergeError(f"v2-only merge: model {i} is {m['version']!r} (not v2)")
        for field in ("sr", "f0", "vocoder", "config"):
            if m[field] != base[field]:
                raise MergeError(
                    f"model {i} {field}={m[field]!r} != model 0 {field}={base[field]!r}; "
                    "only identical-architecture v2 models can merge"
                )
        if m["keys"] != base["keys"]:
            only0 = sorted(base["keys"] - m["keys"])[:3]
            onlyi = sorted(m["keys"] - base["keys"])[:3]
            raise MergeError(
                f"model {i} has a different weight key-set (only-in-0={only0}, "
                f"only-in-{i}={onlyi}); models are not the same architecture"
            )
    return base


def blend_weights(weights: List[Dict[str, Any]], alphas: List[float]) -> Dict[str, Any]:
    """Convex weighted average of N state dicts (keys excl ``enc_q.*``). Floating-
    point tensors are blended in float32 then cast back; non-float buffers are
    copied from the first model. Per-key shapes must match across all models.
    torch is imported lazily (after validation) so the module import stays torch-free
    and the input guards below are unit-testable without torch."""
    if len(weights) != len(alphas):
        raise MergeError("weights/alphas length mismatch")
    if len(weights) < 2:
        raise MergeError("need at least 2 models to merge")
    # self-guard: the blend iterates model 0's keys only, so a key present ONLY in a
    # later model would be silently dropped. check_mergeable enforces equal key-sets
    # upstream, but blend_weights is public/unit-testable -> guard regardless of caller.
    base_keys = _merge_keys(weights[0])
    for i, w in enumerate(weights[1:], start=1):
        wk = _merge_keys(w)
        if wk != base_keys:
            only0 = sorted(base_keys - wk)[:3]
            onlyi = sorted(wk - base_keys)[:3]
            raise MergeError(
                f"model {i} weight key-set differs (only-in-0={only0}, "
                f"only-in-{i}={onlyi}); cannot blend"
            )
    keys = sorted(base_keys)

    import torch
    merged: Dict[str, Any] = {}
    for key in keys:
        base_t = weights[0][key]
        shape = tuple(base_t.shape)
        for i, w in enumerate(weights):
            if key not in w:
                raise MergeError(f"model {i} is missing key {key!r}")
            if tuple(w[key].shape) != shape:
                raise MergeError(
                    f"key {key!r} shape {tuple(w[key].shape)} in model {i} "
                    f"!= {shape} in model 0 (different speaker count / architecture)"
                )
        if base_t.is_floating_point():
            acc = base_t.detach().to(torch.float32) * float(alphas[0])
            for i in range(1, len(weights)):
                acc = acc + weights[i][key].detach().to(torch.float32) * float(alphas[i])
            merged[key] = acc.to(base_t.dtype)
        else:
            merged[key] = base_t.clone()
    return merged


def build_merged_checkpoint(
    base_cpt: Dict[str, Any], merged_weight: Dict[str, Any]
) -> Dict[str, Any]:
    """Carry all non-state-dict metadata (config / version / f0 / vocoder / ...)
    from ``base_cpt`` and attach the blended weight under ``weight``."""
    out: Dict[str, Any] = {}
    for k, v in base_cpt.items():
        if k in ("weight", "model"):
            continue
        out[k] = v
    out["weight"] = merged_weight
    return out


def merge_checkpoints(model_paths, strengths):
    """Load N checkpoints, validate compatibility, and blend them in one call.

    Returns ``(merged_cpt, common_meta, normalized_alphas)``. The shared core for
    both the CLI (``tools.merge_models``) and the GUI merge worker. Lazy torch.
    Raises :class:`MergeError` on incompatibility; other exceptions (a missing /
    corrupt file) propagate from ``torch.load``."""
    import torch

    if len(model_paths) < 2:
        raise MergeError("need at least 2 models to merge")
    cpts = [torch.load(p, map_location="cpu", weights_only=True) for p in model_paths]
    common = check_mergeable([checkpoint_meta(c) for c in cpts])
    alphas = normalize_strengths(strengths)
    merged_weight = blend_weights([weight_dict(c) for c in cpts], alphas)
    return build_merged_checkpoint(cpts[0], merged_weight), common, alphas
