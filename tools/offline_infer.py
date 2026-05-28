"""Placeholder CLI for the offline RVC sanity check.

The offline tool will run ``rvc_engine`` on a wav file end-to-end so a
model can be sanity-checked without the realtime loop. It is Stage 2
work and intentionally not implemented here: this file must not import
torch, infer_rvc_python, or rvc-python.
"""

from __future__ import annotations


MESSAGE = (
    "Offline RVC sanity check is Stage 2 and not implemented yet."
)


def main() -> int:
    print(MESSAGE)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
