"""SOP 检查结果归一化（任务 15.5；需求 11.4、11.8；假设 A15）。

VLM 返回的步骤结果按 SOP 定义做如下处理：

- 结构非法（结果不是 SopCheckResult、步骤不是三元组等）、status 不在 pass/fail/unknown 中，
  抛出 ``SopResultError``。调用方将其视为 ``response_invalid``，整次检查为检查失败待重试。
- 每个 SOP 步骤恰好输出一条结果，顺序与 SOP 定义一致。
- VLM 缺失的步骤补为 ``unknown``，依据为固定说明文字。
- 不属于该 SOP 的多余步骤被忽略。
- ``evidence`` 为空时补为固定说明文字，保证输出的依据非空。
- ``approximate`` 取自步骤定义的 ``requires_sequence``（需求 11.8）。

本模块为纯函数，不访问 I/O。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chimera_patrol.sop.loader import SopDefinition
from chimera_patrol.vlm.protocol import SopCheckResult

VALID_STATUSES: tuple[str, ...] = ("pass", "fail", "unknown")
MISSING_EVIDENCE = "VLM 未返回该步骤的判定依据，记为未知"
EMPTY_EVIDENCE = "VLM 未给出判定依据"


class SopResultError(ValueError):
    """VLM 返回的 SOP 结果结构非法。"""


@dataclass(frozen=True)
class NormalizedStep:
    index: int
    step_id: str
    description: str
    status: str
    evidence: str
    approximate: bool
    evidence_frame: int | None


def _check_entry(entry: Any, position: int) -> tuple[str, str, str, int | None]:
    if not isinstance(entry, (tuple, list)) or len(entry) != 4:
        raise SopResultError(f"step result {position} must be a 4-item tuple")
    step_id, status, evidence, evidence_frame = entry
    if not isinstance(step_id, str) or not step_id:
        raise SopResultError(f"step result {position} has an invalid step_id")
    if status not in VALID_STATUSES:
        raise SopResultError(f"step {step_id!r} has invalid status {status!r}")
    if not isinstance(evidence, str):
        raise SopResultError(f"step {step_id!r} evidence must be a string")
    if evidence_frame is not None and (isinstance(evidence_frame, bool) or not isinstance(evidence_frame, int)):
        raise SopResultError(f"step {step_id!r} evidence_frame must be an int or None")
    return step_id, status, evidence, evidence_frame


def normalize_sop_result(sop: SopDefinition, result: SopCheckResult) -> list[NormalizedStep]:
    """把 VLM 结果归一化为与 SOP 步骤一一对应的结果列表。

    Raises:
        SopResultError: 结构非法或 status 不在枚举中。
    """
    if not isinstance(result, SopCheckResult):
        raise SopResultError("result must be a SopCheckResult")
    raw_steps: Any = result.steps
    if not isinstance(raw_steps, (list, tuple)):
        raise SopResultError("result.steps must be a list")

    by_id: dict[str, tuple[str, str, int | None]] = {}
    for position, entry in enumerate(raw_steps):
        step_id, status, evidence, evidence_frame = _check_entry(entry, position)
        # 同一步骤返回多次时以首次为准。
        by_id.setdefault(step_id, (status, evidence, evidence_frame))

    normalized: list[NormalizedStep] = []
    for index, step in enumerate(sop.steps):
        if step.id in by_id:
            status, evidence, evidence_frame = by_id[step.id]
            text = evidence.strip() or EMPTY_EVIDENCE
        else:
            status, text, evidence_frame = "unknown", MISSING_EVIDENCE, None
        normalized.append(
            NormalizedStep(
                index=index,
                step_id=step.id,
                description=step.description,
                status=status,
                evidence=text,
                approximate=bool(step.requires_sequence),
                evidence_frame=evidence_frame,
            )
        )
    return normalized


def failed_step_ids(steps: Sequence[NormalizedStep]) -> list[str]:
    """返回 status 为 fail 的步骤 ID，保持 SOP 顺序。"""
    return [step.step_id for step in steps if step.status == "fail"]
