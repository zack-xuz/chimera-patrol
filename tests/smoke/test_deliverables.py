"""交付物完整性冒烟测试（任务 19.3；需求 13.1–13.5）。

只检查文件是否存在与基本结构，不运行服务。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / ".kiro" / "specs" / "chimera-patrol-inspection"


def test_spec_documents_present() -> None:
    for name in ("requirements.md", "design.md", "tasks.md"):  # 13.1
        assert (SPEC_DIR / name).is_file(), f"missing spec file {name}"


def test_steering_documents_present() -> None:
    steering = sorted((ROOT / ".kiro" / "steering").glob("*.md"))  # 13.2
    assert len(steering) >= 2, f"need at least 2 steering documents, found {len(steering)}"


def test_agent_hooks_present() -> None:
    hooks = sorted((ROOT / ".kiro" / "hooks").glob("*.json"))  # 13.3
    assert len(hooks) >= 2, f"need at least 2 agent hooks, found {len(hooks)}"
    for path in hooks:
        data = json.loads(path.read_text(encoding="utf-8"))
        # 接受 Kiro IDE 的 when/then 格式，或 createHook 写入的 hooks 数组格式。
        ide_format = isinstance(data.get("when"), dict) and isinstance(data.get("then"), dict)
        hook_array = isinstance(data.get("hooks"), list) and len(data["hooks"]) > 0
        assert ide_format or hook_array, f"{path.name} is neither an IDE hook nor a hooks array"


def test_mcp_registration_present() -> None:
    mcp = json.loads((ROOT / ".kiro" / "settings" / "mcp.json").read_text(encoding="utf-8"))  # 13.4
    assert "chimera-vlm" in mcp["mcpServers"]


def test_kiro_directory_at_root() -> None:
    assert (ROOT / ".kiro").is_dir()  # 13.5


def test_config_sops_and_samples_present() -> None:
    assert (ROOT / "config.yaml").is_file()
    assert (ROOT / "config.demo.yaml").is_file()
    assert (ROOT / "sops.yaml").is_file()
    videos = [p for p in (ROOT / "samples" / "videos").glob("*.mp4")]
    assert len(videos) >= 2, "need at least 2 sample videos"


def test_web_frontend_and_readme_present() -> None:
    assert (ROOT / "web" / "index.html").is_file()
    assert (ROOT / "web" / "event.html").is_file()
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "scripts" / "fetch_weights.py").is_file()
