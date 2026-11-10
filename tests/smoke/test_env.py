"""项目 .env 加载规则：读取 .env、已有环境变量优先、文件缺失时不报错。"""

import os
from pathlib import Path

from chimera_patrol.env import load_project_env

VAR = "CHIMERA_TEST_DOTENV_VAR"


def test_dotenv_value_is_loaded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(VAR, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{VAR}=from-file\n", encoding="utf-8")

    assert load_project_env(dotenv) is True
    assert os.environ[VAR] == "from-file"


def test_existing_environment_variable_wins(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(VAR, "from-shell")
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"{VAR}=from-file\n", encoding="utf-8")

    load_project_env(dotenv)
    assert os.environ[VAR] == "from-shell"


def test_missing_file_is_not_an_error(tmp_path: Path) -> None:
    assert load_project_env(tmp_path / "does-not-exist.env") is False
