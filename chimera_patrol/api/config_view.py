"""只读的“当前生效配置”视图（GET /api/config/effective）。

纯函数，不修改配置、不做热加载。输出只包含检测阈值、ROI、冷却时间、规则参数与 VLM 的
提供方、模型、超时和复核开关。

脱敏规则：API Key 只返回“是否已设置”（``configured``）和存放它的环境变量名，绝不返回取值。
本模块只判断环境变量是否非空，不读取其内容。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chimera_patrol.config import AppConfig, RoiConfig
from chimera_patrol.ingest.camera import Camera


def _rois(rois: Sequence[RoiConfig]) -> list[dict[str, Any]]:
    return [
        {
            "id": roi.id,
            "type": roi.type,
            "name": roi.name,
            "polygon": [[point[0], point[1]] for point in roi.polygon],
        }
        for roi in rois
    ]


def effective_config_view(
    cfg: AppConfig,
    cameras: Sequence[Camera],
    paused: set[str],
    env: Mapping[str, str],
) -> dict[str, Any]:
    """组装只读配置视图。``env`` 仅用于判断密钥是否已设置。"""
    rules = cfg.rules
    thresholds = {
        "person_conf": cfg.detector.person.conf,
        "smoke_fire_conf": rules.smoke_fire.conf_threshold,
        "helmet_conf": rules.helmet.conf_threshold,
    }
    camera_items = [
        {
            "id": cam.id,
            "name": cam.name,
            "source_type": cam.source_type,
            "paused": cam.id in paused,
            "thresholds": dict(thresholds),
            "cooldown_s": rules.cooldown_s,
            "rois": _rois(cam.rois),
        }
        for cam in cameras
    ]

    vlm = cfg.vlm
    key_env = vlm.openai_compatible.api_key_env
    provider_is_remote = vlm.provider == "openai_compatible"

    return {
        "cameras": camera_items,
        "rules": {
            "reference_point": rules.reference_point,
            "roi_exit_grace_s": rules.roi_exit_grace_s,
            "dwell_threshold_s": rules.dwell_threshold_s,
            "cooldown_s": rules.cooldown_s,
            "trajectory_max_points": rules.trajectory_max_points,
            "roi_limits": {
                "max_per_camera": rules.roi.max_per_camera,
                "min_vertices": rules.roi.min_vertices,
                "max_vertices": rules.roi.max_vertices,
            },
            "smoke_fire": {
                "conf_threshold": rules.smoke_fire.conf_threshold,
                "consecutive_frames": rules.smoke_fire.consecutive_frames,
                "same_spot_iou_threshold": rules.smoke_fire.same_spot_iou_threshold,
            },
            "helmet": {
                "conf_threshold": rules.helmet.conf_threshold,
                "no_helmet_duration_s": rules.helmet.no_helmet_duration_s,
                "head_region_ratio": rules.helmet.head_region_ratio,
                "assoc_overlap_metric": rules.helmet.assoc_overlap_metric,
                "assoc_overlap_threshold": rules.helmet.assoc_overlap_threshold,
                "min_person_height_px": rules.helmet.min_person_height_px,
                "track_gap_reset_s": rules.helmet.track_gap_reset_s,
                "vlm_fallback": {
                    "enabled": rules.helmet.vlm_fallback.enabled,
                    "interval_s": rules.helmet.vlm_fallback.interval_s,
                    "min_confidence": rules.helmet.vlm_fallback.min_confidence,
                },
            },
        },
        "detector": {
            "frame_skip_n": cfg.detector.frame_skip_n,
            "imgsz": cfg.detector.imgsz,
            "device": cfg.detector.device,
            "person_conf": cfg.detector.person.conf,
            "tracker_lost_buffer_s": cfg.detector.tracker.lost_buffer_s,
        },
        "vlm": {
            "review_enabled": vlm.review_enabled,
            "provider": vlm.provider,
            "model": vlm.openai_compatible.model if provider_is_remote else None,
            "base_url": vlm.openai_compatible.base_url if provider_is_remote else None,
            "timeout_s": vlm.timeout_s,
            "max_concurrency": vlm.max_concurrency,
            "description_max_chars": vlm.description_max_chars,
            "auto_decision_threshold": vlm.auto_decision_threshold,
            "api_key": {
                "env_var": key_env,
                "configured": bool(env.get(key_env)),
            },
        },
    }
