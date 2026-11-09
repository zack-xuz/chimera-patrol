"""Camera 数据结构（design.md「视频接入（Video_Ingest）」）。

``source`` 可能是 RTSP URL，其中可能含 ``user:pass@`` 凭据。``repr`` 与 ``str``
只输出脱敏后的地址，原始值仅通过 ``source`` 字段访问。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from chimera_patrol.config import RoiConfig

SourceType = Literal["file", "rtsp"]

# 匹配 URL 的 userinfo 部分（scheme://...@）。使用贪婪匹配，使含未转义 "@" 的密码也被整段遮盖。
_USERINFO_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)[^/\s]*@")


def mask_source(source: str) -> str:
    """把 URL 中的 ``user:pass@`` 替换为 ``***@``；无凭据的字符串原样返回。"""
    return _USERINFO_RE.sub(lambda m: f"{m.group('scheme')}***@", source)


@dataclass(frozen=True, repr=False)
class Camera:
    """一路虚拟摄像头。``source_type`` 为 ``file`` 时 ``source`` 是文件路径，为 ``rtsp`` 时是 RTSP URL。"""

    id: str
    name: str
    source_type: SourceType
    source: str
    rois: tuple[RoiConfig, ...] = ()

    def __repr__(self) -> str:
        return (
            f"Camera(id={self.id!r}, name={self.name!r}, source_type={self.source_type!r}, "
            f"source={mask_source(self.source)!r}, rois={self.rois!r})"
        )
