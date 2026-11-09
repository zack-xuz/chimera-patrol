"""区域闯入与人员滞留的属性测试（Property 4）。

验证 ``IntrusionRule`` 在冷却关闭时满足：
(a) 每次参考点从“不在 ROI 内”（经 ``roi_exit_grace_s`` 确认）变为“在 ROI 内”，恰产生一条 ``intrusion`` 事件；
(b) 当且仅当存在一段连续停留时长 ``>= dwell_threshold_s``，才对该段产生一条（且仅一条）``loitering`` 事件。

停留时长按在内帧的时间戳计算：``最后在内帧 - 进入帧``。模块只在在内帧上判定滞留，宽限期内的空档不计入判定。

**Validates: Requirements 3.2, 3.3**
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from chimera_patrol.rules.intrusion import IntrusionRule, RoiSpec, TrackObservation, ZoneEvent

ROI_ID = "roi-1"
POLYGON = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
BOX_HALF_WIDTH = 5.0
BOX_HEIGHT = 10.0

Label = str  # "in" | "out" | "missing"
Entry = Tuple[Label, float, float]

# 参考点取框底边中点，因此 box 的底边中点即为 (x, y)。
# 在内点位于多边形内部，在外点位于多边形右侧，二者与边界保持足够距离。
MISSING: Entry = ("missing", 0.0, 0.0)
INSIDE_ENTRY = st.tuples(st.just("in"), st.floats(1.0, 99.0), st.floats(1.0, 99.0))
OUTSIDE_ENTRY = st.tuples(st.just("out"), st.floats(110.0, 200.0), st.floats(1.0, 99.0))
ENTRY = st.one_of(st.just(MISSING), INSIDE_ENTRY, OUTSIDE_ENTRY)


@dataclass(frozen=True)
class Scenario:
    """一组随机输入：严格递增的帧时间戳、每路轨迹的逐帧观测、宽限与阈值（毫秒）。"""

    timestamps: List[int]
    columns: List[List[Entry]]  # columns[k][f] 为第 k 路轨迹在第 f 帧的观测
    grace_ms: int
    dwell_ms: int


@st.composite
def scenarios(draw) -> Scenario:
    n = draw(st.integers(min_value=1, max_value=40))
    gaps = draw(st.lists(st.integers(min_value=1, max_value=2500), min_size=n, max_size=n))
    start = draw(st.integers(min_value=0, max_value=1_000_000))
    timestamps = [start + total for total in itertools.accumulate(gaps)]
    grace_ms = draw(st.integers(min_value=0, max_value=3000))
    dwell_ms = draw(st.integers(min_value=0, max_value=8000))
    n_tracks = draw(st.integers(min_value=1, max_value=2))
    columns = [draw(st.lists(ENTRY, min_size=n, max_size=n)) for _ in range(n_tracks)]
    return Scenario(timestamps, columns, grace_ms, dwell_ms)


def _track_id(k: int) -> int:
    return k + 1


def _run_rule(s: Scenario) -> List[ZoneEvent]:
    rule = IntrusionRule(
        reference_point="bottom_center",
        roi_exit_grace_s=s.grace_ms / 1000,
        dwell_threshold_s=s.dwell_ms / 1000,
        rois=[RoiSpec(id=ROI_ID, type="intrusion", polygon=POLYGON)],
    )
    events: List[ZoneEvent] = []
    for f, ts in enumerate(s.timestamps):
        observations = []
        for k, column in enumerate(s.columns):
            label, x, y = column[f]
            if label == "missing":
                continue
            observations.append(
                TrackObservation(
                    track_id=_track_id(k),
                    box=(x - BOX_HALF_WIDTH, y - BOX_HEIGHT, x + BOX_HALF_WIDTH, y),
                )
            )
        events.extend(rule.process(ts, observations))
    return events


def _inside_timestamps(s: Scenario, k: int) -> List[int]:
    return [ts for ts, (label, _, _) in zip(s.timestamps, s.columns[k]) if label == "in"]


def _split_stays(inside_ts: Sequence[int], grace_ms: int) -> List[List[int]]:
    """把某轨迹的在内时间戳切分为连续停留：相邻在内帧间隔超过宽限则开启新停留。"""
    stays: List[List[int]] = []
    for ts in inside_ts:
        if stays and ts - stays[-1][-1] <= grace_ms:
            stays[-1].append(ts)
        else:
            stays.append([ts])
    return stays


def _reference_events(s: Scenario) -> List[ZoneEvent]:
    """独立的穷举参考模型：对每路轨迹切分停留，逐段生成事件。"""
    events: List[ZoneEvent] = []
    for k in range(len(s.columns)):
        track_id = _track_id(k)
        for stay in _split_stays(_inside_timestamps(s, k), s.grace_ms):
            entry = stay[0]
            events.append(ZoneEvent("intrusion", ROI_ID, track_id, entry, entry))
            for ts in stay:
                if ts - entry >= s.dwell_ms:
                    events.append(ZoneEvent("loitering", ROI_ID, track_id, entry, ts))
                    break
    return events


def _sort_key(e: ZoneEvent) -> Tuple[int, int, str, int]:
    return (e.end_ms, e.track_id, e.type, e.start_ms)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_events_match_reference_model(s: Scenario) -> None:
    """规则输出的事件列表与穷举参考模型完全一致。"""
    actual = sorted(_run_rule(s), key=_sort_key)
    expected = sorted(_reference_events(s), key=_sort_key)
    assert actual == expected


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_property_4a_one_intrusion_per_confirmed_entry(s: Scenario) -> None:
    """(a) 每次经宽限确认的“外→内”转换恰好产生一条 intrusion 事件。"""
    events = _run_rule(s)

    expected: Counter = Counter()
    for k in range(len(s.columns)):
        inside_ts = _inside_timestamps(s, k)
        for i, ts in enumerate(inside_ts):
            # 前一在内帧不存在，或距其已超过宽限，即视为确认离开后再次进入。
            if i == 0 or ts - inside_ts[i - 1] > s.grace_ms:
                expected[(_track_id(k), ts)] += 1

    actual = Counter(
        (e.track_id, e.start_ms) for e in events if e.type == "intrusion"
    )
    assert all(e.roi_id == ROI_ID for e in events if e.type == "intrusion")
    assert actual == expected


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(scenarios())
def test_property_4b_loitering_iff_stay_reaches_dwell(s: Scenario) -> None:
    """(b) 当且仅当某段连续停留时长 >= dwell_threshold_s 时，该段恰有一条 loitering 事件。"""
    events = _run_rule(s)

    expected_keys: Counter = Counter()
    for k in range(len(s.columns)):
        for stay in _split_stays(_inside_timestamps(s, k), s.grace_ms):
            if stay[-1] - stay[0] >= s.dwell_ms:
                expected_keys[(_track_id(k), stay[0])] += 1

    actual_keys = Counter(
        (e.track_id, e.start_ms) for e in events if e.type == "loitering"
    )
    # 每段停留至多一条，且集合与满足时长条件的停留一一对应。
    assert all(count == 1 for count in actual_keys.values())
    assert actual_keys == expected_keys
