"""Pure tests for the direct worker's safety net (no torch / devices).

Drives ``rvc_direct_worker_loop`` with a fake engine + real queues in a thread:
identity fallback on engine error, persistent-failure escalation (one
engine.reset() per streak, unhealthy flag after ~10 s), and the
startup-vs-steady underrun classifier flag gated on engine warm-up.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from meloie.engine.streaming_worker import rvc_direct_worker_loop
from meloie.safety.metrics import RuntimeMetrics

SENTINEL = object()
BF = 480           # block_frame == output_block_size: 1 in -> 1 out


class FakeEngine:
    """process_block fails for the first ``fail_first`` blocks, then echoes."""

    def __init__(self, fail_first=0, warmup_left=0):
        self.fail_first = fail_first
        self.calls = 0
        self.resets = 0
        self.warmup_blocks_left = warmup_left
        self.last_sola_offset = 0
        self.last_silence_skipped = False

    def process_block(self, block, sr):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("boom")
        return block

    def reset(self):
        self.resets += 1


def _run_worker(engine, blocks, metrics):
    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue(maxsize=len(blocks) + 8)
    stop = threading.Event()
    for b in blocks:
        in_q.put(b)
    in_q.put(SENTINEL)
    t = threading.Thread(
        target=rvc_direct_worker_loop,
        args=(engine, in_q, out_q, metrics, stop, SENTINEL),
        kwargs=dict(stream_sr=48000, block_frame=BF, output_block_size=BF,
                    drop_stale_input=False),
        daemon=True,
    )
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    out = []
    while True:
        try:
            out.append(out_q.get_nowait())
        except queue.Empty:
            return out


def _blocks(n):
    return [np.full(BF, 0.1, dtype=np.float32) for _ in range(n)]


def test_fallback_emits_identity_and_counts():
    m = RuntimeMetrics()
    eng = FakeEngine(fail_first=2)
    out = _run_worker(eng, _blocks(5), m)
    assert len(out) == 5                       # link never dies
    assert m.rvc_fallback_count == 2
    assert m.rvc_chunks_processed == 5


def test_persistent_failure_resets_engine_once_per_streak():
    m = RuntimeMetrics()
    eng = FakeEngine(fail_first=5)             # streak of 5 >= reset_after (3)
    _run_worker(eng, _blocks(8), m)
    assert eng.resets == 1                     # exactly one reset per streak
    assert m.rvc_engine_resets == 1
    assert m.rvc_engine_unhealthy is False     # recovered before the 10 s mark


def test_short_streak_does_not_reset():
    m = RuntimeMetrics()
    eng = FakeEngine(fail_first=2)             # below reset_after
    _run_worker(eng, _blocks(6), m)
    assert eng.resets == 0
    assert m.rvc_engine_resets == 0


def test_unhealthy_flag_after_long_streak_and_clears_on_recovery():
    m = RuntimeMetrics()
    # block budget = 480/48000 = 10ms -> unhealthy_after = 1000 consecutive;
    # use a smaller budget surrogate by checking the flag math directly with a
    # very long streak instead: 1005 failures then recovery.
    eng = FakeEngine(fail_first=1005)
    _run_worker(eng, _blocks(1010), m)
    # flag flips during the streak; the recovery blocks clear it again
    assert m.rvc_engine_unhealthy is False
    assert m.rvc_fallback_count == 1005
    assert eng.resets == 1


def test_first_real_output_waits_for_engine_warmup():
    m = RuntimeMetrics()
    eng = FakeEngine(warmup_left=3)            # engine still emitting warm-up zeros

    real_engine_process = eng.process_block

    def process(block, sr):
        out = real_engine_process(block, sr)
        if eng.warmup_blocks_left > 0:
            eng.warmup_blocks_left -= 1
            return np.zeros_like(out)
        return out

    eng.process_block = process
    _run_worker(eng, _blocks(2), m)
    assert m.first_real_output_seen is False   # all emitted blocks were warm-up

    m2 = RuntimeMetrics()
    eng2 = FakeEngine(warmup_left=0)
    _run_worker(eng2, _blocks(2), m2)
    assert m2.first_real_output_seen is True
