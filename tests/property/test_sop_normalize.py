"""Property 21: SOP 结果归一化与近似标注。

# Feature: chimera-patrol-inspection, Property 21: SOP 结果归一化与近似标注
**Validates: Requirements 11.4, 11.8**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from chimera_patrol.sop.loader import SopDefinition, SopStep
from chimera_patrol.sop.normalize import (
    EMPTY_EVIDENCE,
    MISSING_EVIDENCE,
    SopResultError,
    normalize_sop_result,
)
from chimera_patrol.vlm.protocol import SopCheckResult

VALID = ["pass", "fail", "unknown"]


@st.composite
def sop_and_result(draw):
    n_steps = draw(st.integers(min_value=1, max_value=8))
    steps = [
        SopStep(
            id=f"step_{i}",
            description=f"描述{i}",
            criteria=f"标准{i}",
            requires_sequence=draw(st.booleans()),
        )
        for i in range(n_steps)
    ]
    sop = SopDefinition(id="entry_prep", name="入场作业准备", camera_ids=["*"], steps=steps)

    # VLM 返回：部分 SOP 步骤、若干多余步骤；status 合法；evidence 可能为空。
    chosen = draw(st.lists(st.sampled_from([s.id for s in steps]), unique=True, max_size=n_steps))
    extras = [f"extra_{i}" for i in range(draw(st.integers(min_value=0, max_value=3)))]
    raw = []
    for step_id in chosen + extras:
        status = draw(st.sampled_from(VALID))
        evidence = draw(st.one_of(st.just(""), st.just("   "), st.text(min_size=1, max_size=20)))
        frame = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=15)))
        raw.append((step_id, status, evidence, frame))
    return sop, SopCheckResult(steps=raw), set(chosen), set(extras)


@settings(max_examples=100)
@given(data=sop_and_result())
def test_normalization_matches_sop(data) -> None:
    sop, result, returned, extras = data
    normalized = normalize_sop_result(sop, result)

    # 每个 SOP 步骤恰好一条，顺序与定义一致，多余步骤被忽略。
    assert [n.step_id for n in normalized] == [s.id for s in sop.steps]
    assert [n.index for n in normalized] == list(range(len(sop.steps)))
    assert not any(n.step_id in extras for n in normalized)

    for step, out in zip(sop.steps, normalized):
        # 依据非空。
        assert out.evidence.strip() != ""
        # 近似标注取自 requires_sequence。
        assert out.approximate == step.requires_sequence
        assert out.status in VALID
        if step.id not in returned:
            # 缺失步骤补为 unknown。
            assert out.status == "unknown"
            assert out.evidence == MISSING_EVIDENCE
            assert out.evidence_frame is None


@settings(max_examples=100)
@given(
    status=st.text(min_size=1, max_size=10).filter(lambda s: s not in VALID),
    n_steps=st.integers(min_value=1, max_value=5),
)
def test_invalid_status_is_rejected(status: str, n_steps: int) -> None:
    steps = [SopStep(id=f"s{i}", description="d", criteria="c") for i in range(n_steps)]
    sop = SopDefinition(id="x", name="x", camera_ids=["*"], steps=steps)
    result = SopCheckResult(steps=[("s0", status, "依据", None)])
    try:
        normalize_sop_result(sop, result)
    except SopResultError:
        return
    raise AssertionError("invalid status must raise SopResultError")


def test_empty_evidence_is_filled() -> None:
    steps = [SopStep(id="s0", description="d", criteria="c")]
    sop = SopDefinition(id="x", name="x", camera_ids=["*"], steps=steps)
    out = normalize_sop_result(sop, SopCheckResult(steps=[("s0", "pass", "  ", None)]))
    assert out[0].evidence == EMPTY_EVIDENCE
