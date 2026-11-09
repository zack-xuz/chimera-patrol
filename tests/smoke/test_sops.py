"""sops.yaml 冒烟测试：示例文件可加载，重复 id、空步骤、超长名称等非法输入被拒绝并给出字段路径。"""

import copy
from pathlib import Path

import pytest
import yaml

from chimera_patrol.sop.loader import SopConfigError, load_sops

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "sops.yaml"

BASE_SOP = {
    "id": "entry_prep",
    "name": "入场作业准备",
    "camera_ids": ["*"],
    "steps": [
        {"id": "wear_helmet", "description": "佩戴安全帽", "criteria": "头部可见且戴有安全帽"},
        {"id": "wear_gloves", "description": "佩戴手套", "criteria": "双手可见时均戴有手套"},
    ],
}


def _write(tmp_path: Path, sops: list[dict]) -> Path:
    path = tmp_path / "sops.yaml"
    path.write_text(yaml.safe_dump({"sops": sops}, allow_unicode=True), encoding="utf-8")
    return path


def _paths(excinfo: pytest.ExceptionInfo[SopConfigError]) -> list[str]:
    return [issue.path for issue in excinfo.value.issues]


def test_root_sops_yaml_loads_with_example_sop() -> None:
    sops = load_sops(EXAMPLE)

    assert len(sops) == 1
    sop = sops[0]
    assert sop.id == "entry_prep"
    assert sop.name == "入场作业准备"
    assert sop.camera_ids == ["*"]
    assert [step.id for step in sop.steps] == [
        "wear_helmet",
        "wear_gloves",
        "wear_vest",
        "barrier_in_place",
    ]
    assert all(step.requires_sequence is False for step in sop.steps)
    assert sop.steps[0].criteria == "头部可见且戴有安全帽，帽带系好"


def test_valid_sop_loads_with_requires_sequence_default_false(tmp_path: Path) -> None:
    sops = load_sops(_write(tmp_path, [BASE_SOP]))

    assert sops[0].steps[0].requires_sequence is False


def test_duplicate_sop_id_is_rejected_with_field_path(tmp_path: Path) -> None:
    path = _write(tmp_path, [BASE_SOP, copy.deepcopy(BASE_SOP)])

    with pytest.raises(SopConfigError) as excinfo:
        load_sops(path)

    assert _paths(excinfo) == ["sops[1].id"]


def test_duplicate_step_id_within_sop_is_rejected(tmp_path: Path) -> None:
    sop = copy.deepcopy(BASE_SOP)
    sop["steps"][1]["id"] = "wear_helmet"

    with pytest.raises(SopConfigError) as excinfo:
        load_sops(_write(tmp_path, [sop]))

    assert _paths(excinfo) == ["sops[0].steps[1].id"]


def test_same_step_id_across_different_sops_is_allowed(tmp_path: Path) -> None:
    other = copy.deepcopy(BASE_SOP)
    other["id"] = "hot_work"

    sops = load_sops(_write(tmp_path, [BASE_SOP, other]))

    assert [sop.id for sop in sops] == ["entry_prep", "hot_work"]


def test_empty_steps_is_rejected(tmp_path: Path) -> None:
    sop = copy.deepcopy(BASE_SOP)
    sop["steps"] = []

    with pytest.raises(SopConfigError) as excinfo:
        load_sops(_write(tmp_path, [sop]))

    assert _paths(excinfo) == ["sops[0].steps"]


def test_over_long_name_is_rejected_with_field_path(tmp_path: Path) -> None:
    sop = copy.deepcopy(BASE_SOP)
    sop["name"] = "长" * 51

    with pytest.raises(SopConfigError) as excinfo:
        load_sops(_write(tmp_path, [sop]))

    assert _paths(excinfo) == ["sops[0].name"]


def test_criteria_over_200_chars_is_rejected_with_nested_path(tmp_path: Path) -> None:
    sop = copy.deepcopy(BASE_SOP)
    sop["steps"][1]["criteria"] = "字" * 201

    with pytest.raises(SopConfigError) as excinfo:
        load_sops(_write(tmp_path, [sop]))

    assert _paths(excinfo) == ["sops[0].steps[1].criteria"]


def test_more_than_20_steps_is_rejected(tmp_path: Path) -> None:
    sop = copy.deepcopy(BASE_SOP)
    sop["steps"] = [
        {"id": f"s{i}", "description": "d", "criteria": "c"} for i in range(21)
    ]

    with pytest.raises(SopConfigError) as excinfo:
        load_sops(_write(tmp_path, [sop]))

    assert _paths(excinfo) == ["sops[0].steps"]


def test_empty_camera_ids_is_rejected(tmp_path: Path) -> None:
    sop = copy.deepcopy(BASE_SOP)
    sop["camera_ids"] = []

    with pytest.raises(SopConfigError) as excinfo:
        load_sops(_write(tmp_path, [sop]))

    assert _paths(excinfo) == ["sops[0].camera_ids"]


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SopConfigError) as excinfo:
        load_sops(tmp_path / "missing.yaml")

    assert _paths(excinfo) == ["(file)"]
