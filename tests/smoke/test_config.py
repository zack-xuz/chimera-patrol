"""配置加载冒烟测试：示例 config.yaml 可加载，演示配置无需密钥，非法值被拒绝并给出字段路径。"""

from pathlib import Path

import pytest
import yaml

from chimera_patrol.config import ConfigError, load_config, parse_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "config.yaml"
DEMO = ROOT / "config.demo.yaml"
TEST_ENV = {"VLM_API_KEY": "dummy-for-test"}


def test_example_config_loads_with_design_defaults() -> None:
    cfg = load_config(EXAMPLE, env=TEST_ENV)

    assert cfg.detector.frame_skip_n == 3
    assert cfg.vlm.provider == "openai_compatible"
    assert cfg.vlm.timeout_s == 20
    assert cfg.vlm.description_max_chars == 50
    assert cfg.sop.sample_count == 8
    assert cfg.sop.window_before_s == 5
    assert cfg.rules.roi.max_per_camera == 10


def test_demo_config_uses_sensenova_and_requires_key() -> None:
    cfg = load_config(DEMO, env=TEST_ENV)

    assert cfg.vlm.provider == "openai_compatible"
    assert cfg.vlm.openai_compatible.base_url == "https://token.sensenova.cn/v1"
    assert cfg.vlm.openai_compatible.model == "sensenova-6.8-flash"
    assert cfg.vlm.timeout_s == 20
    assert cfg.vlm.review_enabled is True

    with pytest.raises(ConfigError) as excinfo:
        load_config(DEMO, env={})
    assert "vlm.openai_compatible.api_key_env" in [i.path for i in excinfo.value.issues]


def test_example_config_without_api_key_fails_on_api_key_env_path() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(EXAMPLE, env={})

    paths = [issue.path for issue in excinfo.value.issues]
    assert "vlm.openai_compatible.api_key_env" in paths


def test_out_of_range_frame_skip_is_rejected_with_field_path() -> None:
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data["detector"]["frame_skip_n"] = 31

    with pytest.raises(ConfigError) as excinfo:
        parse_config(data, env=TEST_ENV)

    assert [issue.path for issue in excinfo.value.issues] == ["detector.frame_skip_n"]
