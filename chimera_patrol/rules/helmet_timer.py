"""未佩戴安全帽连续计时器（D3）。

按 person track 逐帧计时。时长参数均为整数毫秒，由调用方在构造前把配置秒数一次性换算
（见 rules/engine.py 的 seconds_to_ms）。计时全程只做整数比较，不做浮点运算。

规则摘要：
- 作用域内且无关联安全帽的观测为违规帧。首个违规帧记录 t0，之后保持不变。
- 关联到安全帽的观测清零计时，任意已处理帧均适用。
- 违规帧满足 ts - t0 >= no_helmet_duration_ms 时产生一次事件（start_ms = t0，end_ms = ts）。
  同一段连续违规至多触发一次。
- track 相对上次观测的间隔严格大于 track_gap_reset_ms 时清零（等于限值不清零）。
- track 离开作用域且超过 roi_exit_grace_ms 时清零；宽限期内保留 t0。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

__all__ = ["HelmetObservation", "NoHelmetEvent", "HelmetTimer"]


@dataclass(frozen=True)
class HelmetObservation:
    """单帧内某个 person track 的观测。"""

    track_id: int
    in_scope: bool
    has_associated_helmet: bool


@dataclass(frozen=True)
class NoHelmetEvent:
    """未佩戴安全帽事件，时间单位为毫秒。"""

    track_id: int
    start_ms: int
    end_ms: int


@dataclass
class _TrackState:
    last_seen_ms: int
    t0_ms: Optional[int] = None
    fired: bool = False
    out_since_ms: Optional[int] = None


def _check_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int (milliseconds), got {value!r}")
    return value


@dataclass
class HelmetTimer:
    """按 track 计算连续未佩戴安全帽时长（整数毫秒）。"""

    no_helmet_duration_ms: int
    track_gap_reset_ms: int
    roi_exit_grace_ms: int
    _states: Dict[int, _TrackState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.no_helmet_duration_ms = _check_int("no_helmet_duration_ms", self.no_helmet_duration_ms)
        self.track_gap_reset_ms = _check_int("track_gap_reset_ms", self.track_gap_reset_ms)
        self.roi_exit_grace_ms = _check_int("roi_exit_grace_ms", self.roi_exit_grace_ms)
        if self.no_helmet_duration_ms <= 0:
            raise ValueError("no_helmet_duration_ms must be > 0")
        if self.track_gap_reset_ms <= 0:
            raise ValueError("track_gap_reset_ms must be > 0")
        if self.roi_exit_grace_ms < 0:
            raise ValueError("roi_exit_grace_ms must be >= 0")

    def update(self, ts_ms: int, observations: Iterable[HelmetObservation]) -> List[NoHelmetEvent]:
        """处理一个已处理帧，返回本帧产生的事件。ts_ms 须按处理顺序单调不减。"""
        # 清理相对上次观测已超过 track_gap_reset_ms 的 track（含本帧即将重新出现的 track）。
        for track_id in list(self._states):
            if ts_ms - self._states[track_id].last_seen_ms > self.track_gap_reset_ms:
                del self._states[track_id]

        events: List[NoHelmetEvent] = []
        for obs in observations:
            state = self._states.get(obs.track_id)
            if state is None:
                state = _TrackState(last_seen_ms=ts_ms)
                self._states[obs.track_id] = state
            state.last_seen_ms = ts_ms

            if not obs.in_scope:
                if state.out_since_ms is None:
                    state.out_since_ms = ts_ms
                if ts_ms - state.out_since_ms > self.roi_exit_grace_ms:
                    state.t0_ms = None
                    state.fired = False
                continue

            state.out_since_ms = None

            if obs.has_associated_helmet:
                state.t0_ms = None
                state.fired = False
                continue

            if state.t0_ms is None:
                state.t0_ms = ts_ms
                state.fired = False
            if not state.fired and ts_ms - state.t0_ms >= self.no_helmet_duration_ms:
                state.fired = True
                events.append(NoHelmetEvent(track_id=obs.track_id, start_ms=state.t0_ms, end_ms=ts_ms))

        return events