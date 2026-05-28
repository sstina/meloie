"""Placeholder CLI for the Stage 1+ click-test latency tool.

The click-test injects a sharp spike into the input path, captures the
loopback through VB-CABLE, and finds the max-abs sample index to derive
honest one-way latency. It cannot exist before the realtime identity
stream exists — there is nothing to inject into yet.
"""

from __future__ import annotations

import sys


MESSAGE = (
    "Click-test will be implemented after Stage 1 identity streaming exists."
)


def main() -> int:
    print(MESSAGE)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
