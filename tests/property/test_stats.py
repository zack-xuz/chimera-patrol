"""Property 17: 统计口径。

# Feature: chimera-patrol-inspection, Property 17: 统计口径
**Validates: Requirements 8.1, 8.2**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from chimera_patrol.api.stats import COUNT_KEYS, compute_stats

CAMERAS = ["cam_a", "cam_b", "cam_c"]
TYPES = ["intrusion", "loitering", "smoke_fire", "no_helmet", "sop_violation"]
STATUSES = list(COUNT_KEYS)

event_strategy = st.fixed_dictionaries(
    {
        "camera_id": st.sampled_from(CAMERAS),
        "event_type": st.sampled_from(TYPES),
        "status": st.sampled_from(STATUSES),
    }
)


@settings(max_examples=100)
@given(events=st.lists(event_strategy, max_size=60), include_sop=st.booleans())
def test_statistics_caliber(events: list[dict[str, str]], include_sop: bool) -> None:
    stats = compute_stats(events, include_sop)
    kept = [e for e in events if include_sop or e["event_type"] != "sop_violation"]

    totals = stats["totals"]
    # 总数等于各统计状态计数之和，且等于纳入统计的事件数。
    assert totals["total_alarms"] == sum(totals[k] for k in COUNT_KEYS) == len(kept)
    # 待人工数 = needs_human + review_failed。
    assert totals["pending_human"] == totals["needs_human"] + totals["review_failed"]
    # 误报下降率 = dismissed / total_alarms，总数为 0 时为 None。
    if totals["total_alarms"] == 0:
        assert totals["false_positive_reduction_rate"] is None
    else:
        assert totals["false_positive_reduction_rate"] == totals["dismissed"] / totals["total_alarms"]

    # 分桶求和等于总计。
    for key in COUNT_KEYS:
        assert sum(item[key] for item in stats["by_camera"]) == totals[key]
        assert sum(item[key] for item in stats["by_type"]) == totals[key]

    # include_sop 为 False 时不含 sop_violation 桶。
    if not include_sop:
        assert all(item["event_type"] != "sop_violation" for item in stats["by_type"])
