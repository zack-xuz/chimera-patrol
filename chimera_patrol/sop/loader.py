"""sops.yaml 的加载与校验。

结构与校验规则依据 design.md 的「sops.yaml 结构」一节。
加载失败时抛出 SopConfigError，错误信息包含出错字段的路径（如 ``sops[1].steps[2].criteria``）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_log = logging.getLogger(__name__)
MAX_STEPS_PER_SOP = 20
NAME_MAX_LEN = 50
TEXT_MAX_LEN = 200

NonEmptyStr = Annotated[str, Field(min_length=1)]
NameStr = Annotated[str, Field(min_length=1, max_length=NAME_MAX_LEN)]
TextStr = Annotated[str, Field(min_length=1, max_length=TEXT_MAX_LEN)]


@dataclass(frozen=True)
class SopStep:
    id: str
    description: str
    criteria: str
    requires_sequence: bool = False


@dataclass(frozen=True)
class SopDefinition:
    id: str
    name: str
    camera_ids: list[str] = field(default_factory=list)
    steps: list[SopStep] = field(default_factory=list)


@dataclass(frozen=True)
class SopIssue:
    path: str
    message: str


class SopConfigError(Exception):
    """sops.yaml 非法。``issues`` 列出每个出错字段的路径与原因。"""

    def __init__(self, issues: list[SopIssue]) -> None:
        self.issues = issues
        lines = [f"  - {issue.path}: {issue.message}" for issue in issues]
        super().__init__("SOP 配置校验失败:\n" + "\n".join(lines))


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _StepModel(_Model):
    id: NonEmptyStr
    description: TextStr
    criteria: TextStr
    requires_sequence: bool = False


class _SopModel(_Model):
    id: NonEmptyStr
    name: NameStr
    camera_ids: list[NonEmptyStr] = Field(min_length=1)
    steps: list[_StepModel] = Field(min_length=1, max_length=MAX_STEPS_PER_SOP)


class _SopFileModel(_Model):
    sops: list[_SopModel]


def _format_loc(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += ("." if out else "") + str(part)
    return out or "(root)"


def _cross_checks(sops: list[_SopModel]) -> list[SopIssue]:
    issues: list[SopIssue] = []
    seen_sops: dict[str, int] = {}
    for i, sop in enumerate(sops):
        base = f"sops[{i}]"
        if sop.id in seen_sops:
            issues.append(SopIssue(f"{base}.id", f"与 sops[{seen_sops[sop.id]}] 的 id 重复"))
        else:
            seen_sops[sop.id] = i

        seen_steps: dict[str, int] = {}
        for j, step in enumerate(sop.steps):
            if step.id in seen_steps:
                issues.append(
                    SopIssue(
                        f"{base}.steps[{j}].id",
                        f"同一 SOP 内与 steps[{seen_steps[step.id]}] 的 id 重复",
                    )
                )
            else:
                seen_steps[step.id] = j
    return issues


def parse_sops(data: Any) -> list[SopDefinition]:
    """校验已解析的 sops.yaml 内容，返回 SopDefinition 列表。"""
    if not isinstance(data, dict):
        raise SopConfigError([SopIssue("(root)", "顶层必须是映射（mapping），且包含 sops 列表")])

    try:
        parsed = _SopFileModel.model_validate(data)
    except ValidationError as exc:
        issues = [
            SopIssue(_format_loc(tuple(err["loc"])), str(err["msg"])) for err in exc.errors()
        ]
        raise SopConfigError(issues) from exc

    issues = _cross_checks(parsed.sops)
    if issues:
        raise SopConfigError(issues)
    if not parsed.sops:
        _log.warning("sops.yaml 中没有定义任何 SOP，SOP 检查功能不可用")

    return [
        SopDefinition(
            id=sop.id,
            name=sop.name,
            camera_ids=list(sop.camera_ids),
            steps=[
                SopStep(
                    id=step.id,
                    description=step.description,
                    criteria=step.criteria,
                    requires_sequence=step.requires_sequence,
                )
                for step in sop.steps
            ],
        )
        for sop in parsed.sops
    ]


def load_sops(path: str | Path) -> list[SopDefinition]:
    """读取 sops.yaml 并校验，非法时抛出 SopConfigError。"""
    sops_path = Path(path)
    if not sops_path.is_file():
        raise SopConfigError([SopIssue("(file)", f"SOP 文件不存在: {sops_path}")])
    try:
        with sops_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise SopConfigError([SopIssue("(yaml)", f"YAML 解析失败: {exc}")]) from exc
    return parse_sops(data)
