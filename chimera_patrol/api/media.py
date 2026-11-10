"""证据静态文件端点的路径解析（任务 13.3；需求 7.5、design 错误处理「路径穿越」）。

``/media/{path}`` 只能访问 ``evidence.base_dir`` 之下的文件。解析规则：

- 拒绝绝对路径、盘符、空路径与包含 ``..`` 的路径。
- 解析符号链接后，结果必须仍位于 ``base_dir`` 之内。
- 目标不存在或不是普通文件时返回 None。

任何拒绝都返回 None，由调用方统一返回 404，不泄露越界细节。
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def resolve_media_path(base_dir: Path, rel_path: str) -> Path | None:
    """把请求路径解析为 base_dir 下的真实文件路径。越界或不存在时返回 None。"""
    if not rel_path or not rel_path.strip():
        return None
    normalized = rel_path.replace("\\", "/")
    if "\x00" in normalized:
        return None
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(rel_path).drive:
        return None
    if ".." in normalized.split("/"):
        return None

    root = base_dir.resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate
