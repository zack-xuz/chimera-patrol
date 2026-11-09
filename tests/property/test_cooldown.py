"""Property 9: 冷却去重（CooldownTable）。

**Validates: Requirements 3.9**

对任意同一 ``(event_type, target_key)`` 的升序触发时间序列与 ``cooldown_s ≥ 0``：
首个触发必被接受；相邻两个被接受事件的时间差 ``≥ cooldown_s``；每个被丢弃的触发与
其前一个被接受事件的时间差 ``< cooldown_s``；不同键互不影响。

冷却时长以整毫秒给出（``cooldown_s = cooldown_ms / 1000``），避免浮点歧义。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from hypothesis import given, settings, strategies as st

from chimera_patrol.rules.cooldown import CooldownTable

EVENT_TYPE = "person_intrusion"

# (key_index, delta_ms)；全局时间为 delta 的前缀和，因此每个键的时间戳非递减。
_steps = st.lists(
    st.tuples(st.integers(min_value=0, max_value=3), st.integers(min_value=0, max_value=3000)),
    min_size=0,
    max_size=60,
)
_cooldown_ms = st.integers(min_value=0, max_value=5000)


def _to_triggers(steps: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """把 (key, delta) 序列转为 (key, ts_ms) 的全局升序触发序列。"""
    triggers: List[Tuple[int, int]] = []
    t = 0
    for key, delta in steps:
        t += delta
        triggers.append((key, t))
    return triggers


def _reference_decisions(
    triggers: List[Tuple[int, int]], cooldown_ms: int
) -> List[bool]:
    """贪心参考模型：无上次接受记录或 ts - last ≥ cooldown_ms 时接受。"""
    last: dict = {}
    out: List[bool] = []
    for key, ts in triggers:
        prev: Optional[int] = last.get(key)
        if prev is None or ts - prev >= cooldown_ms:
            last[key] = ts
            out.append(True)
        else:
            out.append(False)
    return out


def _impl_decisions(
    triggers: List[Tuple[int, int]], cooldown_ms: int
) -> List[bool]:
    table = CooldownTable(cooldown_s=cooldown_ms / 1000.0)
    return [table.try_accept(EVENT_TYPE, f"track:{key}", ts) for key, ts in triggers]


@settings(max_examples=200, deadline=None)
@given(steps=_steps, cooldown_ms=_cooldown_ms)
def test_decisions_match_reference_model(steps, cooldown_ms):
    """接受/丢弃序列与贪心参考模型逐项一致（多键交错、升序时间）。"""
    triggers = _to_triggers(steps)
    assert _impl_decisions(triggers, cooldown_ms) == _reference_decisions(
        triggers, cooldown_ms
    )


@settings(max_examples=200, deadline=None)
@given(steps=_steps, cooldown_ms=_cooldown_ms)
def test_accepted_and_discarded_gap_invariants(steps, cooldown_ms):
    """逐条检查 Property 9 的三个子句。"""
    triggers = _to_triggers(steps)
    decisions = _impl_decisions(triggers, cooldown_ms)

    last_accepted: dict = {}
    for (key, ts), accepted in zip(triggers, decisions):
        prev = last_accepted.get(key)
        if prev is None:
            # 首个触发必被接受
            assert accepted, f"first trigger for key {key} at {ts} was discarded"
        elif accepted:
            # 相邻被接受事件时间差 >= cooldown
            assert ts - prev >= cooldown_ms
        else:
            # 被丢弃的触发与前一个被接受事件时间差 < cooldown
            assert ts - prev < cooldown_ms
        if accepted:
            last_accepted[key] = ts


@settings(max_examples=200, deadline=None)
@given(steps=_steps, cooldown_ms=_cooldown_ms)
def test_keys_are_independent(steps, cooldown_ms):
    """某键的决策只取决于该键自身的触发，不受其他键影响。"""
    triggers = _to_triggers(steps)
    full = _impl_decisions(triggers, cooldown_ms)

    for key in {k for k, _ in triggers}:
        only_key = [(k, ts) for k, ts in triggers if k == key]
        solo = _impl_decisions(only_key, cooldown_ms)
        in_mixed = [d for (k, _), d in zip(triggers, full) if k == key]
        assert in_mixed == solo


@settings(max_examples=200, deadline=None)
@given(
    deltas=st.lists(st.integers(min_value=0, max_value=3000), min_size=0, max_size=60),
    cooldown_ms=_cooldown_ms,
    threshold=st.floats(min_value=0.1, max_value=1.0),
)
def test_spot_same_box_follows_cooldown(deltas, cooldown_ms, threshold):
    """同一点位（相同框，IoU=1）的 try_accept_spot 与单键冷却规则一致。"""
    box = (10.0, 10.0, 50.0, 60.0)
    triggers: List[Tuple[int, int]] = []
    t = 0
    for d in deltas:
        t += d
        triggers.append((0, t))
    expected = _reference_decisions(triggers, cooldown_ms)

    table = CooldownTable(cooldown_s=cooldown_ms / 1000.0)
    actual = [table.try_accept_spot(EVENT_TYPE, box, ts, threshold) for _, ts in triggers]
    assert actual == expected
