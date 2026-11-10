"""统计导出：JSON 与 CSV（任务 14.5；需求 8.3；design「CSV 导出表头」）。

CSV 表头：``dimension,key,total_alarms,confirmed,dismissed,needs_human,review_failed,pending_review,pending_human,false_positive_reduction_rate``。
``dimension`` 取 total、camera、event_type 之一；total 行的 key 为空。
比率为 null 时单元格留空。
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

CSV_COLUMNS: tuple[str, ...] = (
    "dimension",
    "key",
    "total_alarms",
    "confirmed",
    "dismissed",
    "needs_human",
    "review_failed",
    "pending_review",
    "pending_human",
    "false_positive_reduction_rate",
)


def export_json(stats: dict[str, Any]) -> bytes:
    """统计结果的 JSON 文本（UTF-8，保留中文）。"""
    return json.dumps(stats, ensure_ascii=False, indent=2).encode("utf-8")


def _format_rate(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _row(dimension: str, key: str, bucket: dict[str, Any]) -> list[str]:
    return [
        dimension,
        key,
        str(bucket["total_alarms"]),
        str(bucket["confirmed"]),
        str(bucket["dismissed"]),
        str(bucket["needs_human"]),
        str(bucket["review_failed"]),
        str(bucket["pending_review"]),
        str(bucket["pending_human"]),
        _format_rate(bucket["false_positive_reduction_rate"]),
    ]


def export_csv(stats: dict[str, Any]) -> bytes:
    """统计结果的 CSV 文本（UTF-8 带 BOM，便于 Excel 识别中文）。"""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    writer.writerow(_row("total", "", stats["totals"]))
    for item in stats["by_camera"]:
        writer.writerow(_row("camera", item["camera_id"], item))
    for item in stats["by_type"]:
        writer.writerow(_row("event_type", item["event_type"], item))
    return ("\ufeff" + buffer.getvalue()).encode("utf-8")
