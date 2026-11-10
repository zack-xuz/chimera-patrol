"""项目根目录 .env 的加载（所有入口共用）。

程序启动时先调用 ``load_project_env``，把 ``.env`` 中的变量写入进程环境。
规则：

- 只读取项目根目录下的 ``.env``，路径固定，不向上查找。
- ``override=False``：已经在系统或 shell 中设置的变量优先，``.env`` 不覆盖它们。
- 文件不存在时静默返回 False，不报错（运行时仍可只靠系统环境变量）。

``load_config``、``parse_config`` 与 ``__main__`` 都经过这里，``vlm_probe`` 等外部脚本也应调用它，
保证所有入口的环境变量来源一致。
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_project_env(path: Path = DOTENV_PATH) -> bool:
    """加载 .env 到 os.environ，已存在的变量不覆盖。返回文件是否存在并被读取。"""
    return load_dotenv(dotenv_path=path, override=False)
