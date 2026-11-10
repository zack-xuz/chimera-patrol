"""数据库行与内部记录到 API DTO 的转换（event-contract 第 3、4、5 节；design EventDTO / CameraDTO / SopRunDTO）。

所有函数都是同步函数，可能访问数据库。API 层在 ``asyncio.to_thread`` 中调用它们。
对外不返回 RTSP 凭据：Camera 的 source 字段从不输出。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chimera_patrol.config import RoiConfig
from chimera_patrol.store.repository import EventRepository
from chimera_patrol.store.sop_repo import SopRepository, SopRunRecord

SOP_EVENT_TYPE = "sop_violation"


def media_url(rel_path: str | None) -> str | None:
    """把证据的相对路径转为 ``/media/`` URL。路径为空时返回 None。"""
    if not rel_path:
        return None
    return f"/media/{rel_path.lstrip('/')}"


def vlm_block(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """VLM 复核块（event-contract 4.3）。未复核时为 None。"""
    if row.get("review_skipped"):
        return {
            "verdict": None,
            "confidence": None,
            "description": None,
            "suggestion": None,
            "reviewed_ms": None,
            "error": None,
            "skipped": True,
        }
    if row.get("vlm_error"):
        return {
            "verdict": None,
            "confidence": None,
            "description": None,
            "suggestion": None,
            "reviewed_ms": None,
            "error": row["vlm_error"],
            "skipped": False,
        }
    if row.get("vlm_verdict") is None:
        return None
    return {
        "verdict": row["vlm_verdict"],
        "confidence": row["vlm_confidence"],
        "description": row["vlm_description"],
        "suggestion": row["vlm_suggestion"] or "",
        "reviewed_ms": row["vlm_reviewed_ms"],
        "error": None,
        "skipped": False,
    }


def sop_block(record: SopRunRecord | None) -> dict[str, Any] | None:
    """SOP 信息块。仅 sop_violation 事件有值。"""
    if record is None:
        return None
    failed = [
        {"step_id": step["step_id"], "description": step["description"], "evidence": step["evidence"]}
        for step in record.steps
        if step["status"] == "fail"
    ]
    return {"run_id": record.row["id"], "sop_id": record.row["sop_id"], "failed_steps": failed}


def event_dto(
    row: Mapping[str, Any],
    *,
    camera_names: Mapping[str, str],
    sop_record: SopRunRecord | None = None,
    include_trajectory: bool = False,
) -> dict[str, Any]:
    """事件 DTO。列表与 WebSocket 不含 ``cv.trajectory``，详情接口通过 include_trajectory 返回。"""
    cv: dict[str, Any] = {
        "confidence": row["cv_confidence"],
        "track_id": row["track_id"],
        "roi_id": row["roi_id"],
        "roi_type": row["roi_type"],
        "detail": row["cv_detail"],
    }
    if include_trajectory:
        cv["trajectory"] = row["trajectory"]

    clip_ready = row["clip_status"] == "ready" and bool(row["clip_path"])
    camera_id = row["camera_id"]
    return {
        "id": row["id"],
        "camera_id": camera_id,
        "camera_name": camera_names.get(camera_id, camera_id),
        "event_type": row["event_type"],
        "detection_source": row["detection_source"],
        "status": row["status"],
        "start_ms": row["start_ms"],
        "end_ms": row["end_ms"],
        "cv": cv,
        "evidence": {
            "keyframe_annotated_url": media_url(row["keyframe_annotated_path"]),
            "keyframe_raw_url": media_url(row["keyframe_raw_path"]),
            "clip_url": media_url(row["clip_path"]) if clip_ready else None,
            "clip_status": row["clip_status"],
        },
        "vlm": vlm_block(row),
        "sop": sop_block(sop_record) if row["event_type"] == SOP_EVENT_TYPE else None,
        "created_ms": row["created_ms"],
        "updated_ms": row["updated_ms"],
    }


def load_event_dto(
    repo: EventRepository,
    sop_repo: SopRepository,
    event_id: str,
    *,
    camera_names: Mapping[str, str],
    include_detail: bool = False,
) -> dict[str, Any] | None:
    """读取单个事件并转为 DTO。``include_detail`` 为 True 时附加 trajectory 与 status_log。"""
    row = repo.get_event(event_id)
    if row is None:
        return None
    sop_record = sop_repo.get_run(row["sop_run_id"]) if row["sop_run_id"] else None
    dto = event_dto(
        row,
        camera_names=camera_names,
        sop_record=sop_record,
        include_trajectory=include_detail,
    )
    if include_detail:
        dto["status_log"] = [
            {
                "from_status": entry["from_status"],
                "to_status": entry["to_status"],
                "actor_type": entry["actor_type"],
                "actor": entry["actor"],
                "reason": entry["reason"],
                "created_ms": entry["created_ms"],
            }
            for entry in repo.list_status_log(event_id)
        ]
    return dto


def list_event_dtos(
    repo: EventRepository,
    sop_repo: SopRepository,
    rows: Sequence[Mapping[str, Any]],
    *,
    camera_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    """批量转换事件行（列表接口）。"""
    out: list[dict[str, Any]] = []
    for row in rows:
        sop_record = sop_repo.get_run(row["sop_run_id"]) if row["sop_run_id"] else None
        out.append(event_dto(row, camera_names=camera_names, sop_record=sop_record))
    return out


def camera_dto(
    camera: Mapping[str, Any],
    *,
    rois: Sequence[RoiConfig],
    latest_event: Mapping[str, Any] | None,
    total: int,
    pending_human: int,
) -> dict[str, Any]:
    """CameraDTO（design HTTP API 一节）。不含 source，避免泄露 RTSP 凭据。"""
    camera_id = camera["id"]
    return {
        "id": camera_id,
        "name": camera["name"],
        "source_type": camera["source_type"],
        "status": camera["status"],
        "status_changed_ms": camera["status_changed_ms"],
        "stream_url": f"/api/cameras/{camera_id}/stream.mjpg",
        "rois": [
            {
                "id": roi.id,
                "type": roi.type,
                "name": roi.name,
                "polygon": [[point[0], point[1]] for point in roi.polygon],
            }
            for roi in rois
        ],
        "latest_event": (
            {
                "id": latest_event["id"],
                "event_type": latest_event["event_type"],
                "status": latest_event["status"],
                "created_ms": latest_event["created_ms"],
            }
            if latest_event is not None
            else None
        ),
        "event_counts": {"total": total, "pending_human": pending_human},
    }


def load_camera_dtos(
    repo: EventRepository,
    rois_by_camera: Mapping[str, Sequence[RoiConfig]],
) -> list[dict[str, Any]]:
    """读取全部 Camera 并附加最近事件与计数。"""
    items: list[dict[str, Any]] = []
    for camera in repo.list_cameras():
        camera_id = camera["id"]
        latest = repo.read_rows(
            "SELECT id, event_type, status, created_ms FROM events WHERE camera_id = ? "
            "ORDER BY created_ms DESC, id DESC LIMIT 1",
            [camera_id],
        )
        counts = repo.read_rows(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(CASE WHEN status IN ('needs_human', 'review_failed') THEN 1 ELSE 0 END), 0) "
            "AS pending_human FROM events WHERE camera_id = ?",
            [camera_id],
        )[0]
        items.append(
            camera_dto(
                camera,
                rois=list(rois_by_camera.get(camera_id, ())),
                latest_event=latest[0] if latest else None,
                total=int(counts["total"]),
                pending_human=int(counts["pending_human"]),
            )
        )
    return items


def sop_run_dto(record: SopRunRecord) -> dict[str, Any]:
    """SopRunDTO（design SopRunDTO 一节）。"""
    row = record.row
    failure_reason = row["failure_reason"] if row["status"] == "check_failed_retry" else None
    return {
        "id": row["id"],
        "sop_id": row["sop_id"],
        "sop_name": row["sop_name"],
        "camera_id": row["camera_id"],
        "status": row["status"],
        "triggered_ms": row["triggered_ms"],
        "window_start_ms": row["window_start_ms"],
        "window_end_ms": row["window_end_ms"],
        "sample_count": row["sample_count"],
        "sample_times_ms": row["sample_times_ms"],
        "failure_reason": failure_reason,
        "retry_of": row["retry_of"],
        "operator": row["operator"],
        "steps": [
            {
                "index": step["step_index"],
                "step_id": step["step_id"],
                "description": step["description"],
                "status": step["status"],
                "evidence": step["evidence"],
                "approximate": step["approximate"],
            }
            for step in record.steps
        ],
        "event_id": record.event_id,
        "created_ms": row["created_ms"],
        "finished_ms": row["finished_ms"],
    }
