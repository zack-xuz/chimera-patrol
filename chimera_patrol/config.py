"""Config_File（config.yaml）的 pydantic 模型与加载校验。

字段、默认值与取值范围依据 design.md 的「config.yaml 完整字段」表。
加载失败时抛出 ConfigError，错误信息包含出错字段的路径（如 ``detector.frame_skip_n``）。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Unit = Annotated[float, Field(ge=0, le=1)]
Point = tuple[Unit, Unit]
Verdict = Literal["true_positive", "false_positive", "uncertain"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- app / db / logging / ws


class AppSection(_Model):
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    data_dir: Path = Path("./data")


class DatabaseSection(_Model):
    path: Path = Path("./data/chimera.db")


class LoggingSection(_Model):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    dir: Path = Path("./data/logs")
    max_bytes: int = 10485760
    backup_count: int = 5


class WsSection(_Model):
    heartbeat_interval_s: int = Field(15, ge=1, le=120)


# ---------------------------------------------------------------- video / cameras


class MjpegSection(_Model):
    fps: int = Field(10, ge=1, le=30)
    jpeg_quality: int = Field(70, ge=10, le=95)
    max_width: int = Field(960, ge=160, le=1920)


class VideoSection(_Model):
    source_dir: Path = Path("./samples/videos")
    file_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".avi", ".mkv", ".mov"])
    max_cameras: int = Field(4, ge=2, le=4)
    loop: bool = True
    default_fps: float = Field(25, ge=1, le=60)
    offline_after_failures: int = Field(3, ge=1, le=100)
    reconnect_interval_s: float = Field(5, ge=1, le=300)
    mjpeg: MjpegSection = Field(default_factory=MjpegSection)


class RoiConfig(_Model):
    id: str
    type: Literal["intrusion", "work_zone"] = "intrusion"
    name: str | None = None
    # 归一化顶点坐标 (x, y)，取值 0~1；顶点数范围由 rules.roi 约束（跨字段校验）
    polygon: list[Point]

    @model_validator(mode="after")
    def _default_name(self) -> RoiConfig:
        if self.name is None:
            self.name = self.id
        return self


class CameraConfig(_Model):
    id: str
    name: str | None = None
    rtsp_url: str | None = None
    rois: list[RoiConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _default_name(self) -> CameraConfig:
        if self.name is None:
            self.name = self.id
        return self


# ---------------------------------------------------------------- detector


class PersonDetectorSpec(_Model):
    weights: Path = Path("yolov8n.pt")
    classes: list[str] = Field(default_factory=lambda: ["person"])
    conf: Unit = 0.4


class FireSmokeDetectorSpec(_Model):
    weights: Path = Path("weights/fire_smoke.pt")
    classes: list[str] = Field(default_factory=lambda: ["smoke", "fire"])


class HelmetDetectorSpec(_Model):
    weights: Path = Path("weights/helmet.pt")
    helmet_classes: list[str] = Field(default_factory=lambda: ["helmet"])
    no_helmet_classes: list[str] = Field(default_factory=lambda: ["no-helmet", "head"])


class TrackerSection(_Model):
    lost_buffer_s: float = Field(2.0, ge=0.5, le=30)


class DetectorSection(_Model):
    frame_skip_n: int = Field(3, ge=1, le=30)
    device: str = "auto"
    imgsz: int = 640
    person: PersonDetectorSpec = Field(default_factory=PersonDetectorSpec)
    fire_smoke: FireSmokeDetectorSpec = Field(default_factory=FireSmokeDetectorSpec)
    helmet: HelmetDetectorSpec = Field(default_factory=HelmetDetectorSpec)
    tracker: TrackerSection = Field(default_factory=TrackerSection)


# ---------------------------------------------------------------- rules


class RoiRules(_Model):
    max_per_camera: int = Field(10, ge=1, le=10)
    min_vertices: int = Field(3, ge=3)
    max_vertices: int = Field(20, le=20)


class SmokeFireRules(_Model):
    conf_threshold: Unit = 0.5
    consecutive_frames: int = Field(3, ge=1)
    same_spot_iou_threshold: Unit = 0.3


class HelmetVlmFallback(_Model):
    enabled: bool = True
    interval_s: float = Field(10, ge=1)
    min_confidence: Unit = 0.6
    crop_padding_ratio: Unit = 0.2


class HelmetRules(_Model):
    conf_threshold: Unit = 0.5
    no_helmet_duration_s: float = Field(5, gt=0)
    # design 性质 6 要求 (0, 1]
    head_region_ratio: float = Field(0.25, gt=0, le=1)
    assoc_overlap_metric: Literal["ioh", "iou"] = "ioh"
    assoc_overlap_threshold: Unit = 0.5
    require_no_helmet_evidence: bool = False
    min_person_height_px: int = Field(60, ge=1)
    track_gap_reset_s: float = Field(2.0, gt=0)
    vlm_fallback: HelmetVlmFallback = Field(default_factory=HelmetVlmFallback)


class RulesSection(_Model):
    reference_point: Literal["bottom_center", "center"] = "bottom_center"
    roi: RoiRules = Field(default_factory=RoiRules)
    roi_exit_grace_s: float = Field(1.0, ge=0, le=10)
    dwell_threshold_s: float = Field(10, gt=0)
    cooldown_s: float = Field(30, ge=0)
    trajectory_max_points: int = Field(100, ge=2, le=1000)
    smoke_fire: SmokeFireRules = Field(default_factory=SmokeFireRules)
    helmet: HelmetRules = Field(default_factory=HelmetRules)


# ---------------------------------------------------------------- evidence


class EvidenceSection(_Model):
    base_dir: Path = Path("./data/evidence")
    ring_buffer_s: float = 20
    ring_fps: int = Field(10, ge=1, le=30)
    ring_jpeg_quality: int = Field(90, ge=50, le=100)
    clip_pre_s: float = Field(3, ge=0)
    clip_post_s: float = Field(3, ge=0)
    clip_fps: int = Field(10, ge=1, le=30)


# ---------------------------------------------------------------- vlm


class OpenAICompatibleSettings(_Model):
    # 复核开启且 provider 为 openai_compatible 时必填（跨字段校验）
    base_url: str | None = None
    model: str | None = None
    # 密钥不写入配置文件，只记录存放密钥的环境变量名
    api_key_env: str = "VLM_API_KEY"
    temperature: float = Field(0, ge=0, le=2)
    max_retries: int = Field(0, ge=0, le=3)


class MockSettings(_Model):
    default_verdict: Verdict = "uncertain"
    by_camera: dict[str, Verdict] = Field(default_factory=dict)


class VlmSection(_Model):
    review_enabled: bool = True
    timeout_s: float = Field(20, gt=0)
    max_concurrency: int = Field(2, ge=1, le=16)
    description_max_chars: int = Field(50, ge=1, le=200)
    auto_decision_threshold: Unit = 0.7
    image_max_side: int = Field(1280, ge=256, le=4096)
    image_jpeg_quality: int = Field(85, ge=50, le=100)
    provider: Literal["openai_compatible", "mock"] = "openai_compatible"
    openai_compatible: OpenAICompatibleSettings = Field(default_factory=OpenAICompatibleSettings)
    mock: MockSettings = Field(default_factory=MockSettings)


# ---------------------------------------------------------------- sop


class SopSection(_Model):
    file: Path = Path("./sops.yaml")
    sample_count: int = Field(8, ge=1, le=16)
    window_before_s: float = Field(5, ge=0)
    window_after_s: float = Field(5, ge=0)
    endpoints_inclusive: bool = True


# ---------------------------------------------------------------- root


class AppConfig(_Model):
    app: AppSection = Field(default_factory=AppSection)
    database: DatabaseSection = Field(default_factory=DatabaseSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    ws: WsSection = Field(default_factory=WsSection)
    video: VideoSection = Field(default_factory=VideoSection)
    cameras: list[CameraConfig] = Field(default_factory=list)
    detector: DetectorSection = Field(default_factory=DetectorSection)
    rules: RulesSection = Field(default_factory=RulesSection)
    evidence: EvidenceSection = Field(default_factory=EvidenceSection)
    vlm: VlmSection = Field(default_factory=VlmSection)
    sop: SopSection = Field(default_factory=SopSection)


# ---------------------------------------------------------------- errors and loading


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    message: str


class ConfigError(Exception):
    """配置非法。``issues`` 列出每个出错字段的路径与原因。"""

    def __init__(self, issues: list[ConfigIssue]) -> None:
        self.issues = issues
        lines = [f"  - {issue.path}: {issue.message}" for issue in issues]
        super().__init__("配置校验失败:\n" + "\n".join(lines))


def _format_loc(loc: tuple[Any, ...]) -> str:
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += ("." if out else "") + str(part)
    return out or "(root)"


def _orient(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _segments_intersect(
    p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float], p4: tuple[float, float]
) -> bool:
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
        return True
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


def polygon_self_intersects(polygon: list[tuple[float, float]]) -> bool:
    """判断多边形是否自交（仅检查不相邻的边）。"""
    n = len(polygon)
    if n < 4:
        return False
    edges = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(*edges[i], *edges[j]):
                return True
    return False


def _cross_checks(cfg: AppConfig, env: Mapping[str, str]) -> list[ConfigIssue]:
    issues: list[ConfigIssue] = []
    roi = cfg.rules.roi

    if roi.min_vertices > roi.max_vertices:
        issues.append(
            ConfigIssue(
                "rules.roi.max_vertices",
                f"不能小于 rules.roi.min_vertices（{roi.min_vertices}）",
            )
        )

    seen_cameras: dict[str, int] = {}
    for i, cam in enumerate(cfg.cameras):
        base = f"cameras[{i}]"
        if cam.id in seen_cameras:
            issues.append(
                ConfigIssue(f"{base}.id", f"与 cameras[{seen_cameras[cam.id]}] 的 id 重复")
            )
        else:
            seen_cameras[cam.id] = i

        if len(cam.rois) > roi.max_per_camera:
            issues.append(
                ConfigIssue(
                    f"{base}.rois",
                    f"ROI 数量 {len(cam.rois)} 超过 rules.roi.max_per_camera（{roi.max_per_camera}）",
                )
            )

        seen_rois: dict[str, int] = {}
        for j, region in enumerate(cam.rois):
            rbase = f"{base}.rois[{j}]"
            if region.id in seen_rois:
                issues.append(
                    ConfigIssue(f"{rbase}.id", f"同一 Camera 内与 rois[{seen_rois[region.id]}] 的 id 重复")
                )
            else:
                seen_rois[region.id] = j

            n = len(region.polygon)
            if not roi.min_vertices <= n <= roi.max_vertices:
                issues.append(
                    ConfigIssue(
                        f"{rbase}.polygon",
                        f"顶点数 {n} 不在 [{roi.min_vertices}, {roi.max_vertices}] 范围内",
                    )
                )
            elif polygon_self_intersects(region.polygon):
                issues.append(ConfigIssue(f"{rbase}.polygon", "多边形自交，不是有效多边形"))

    ev = cfg.evidence
    dwell_or_helmet = max(cfg.rules.dwell_threshold_s, cfg.rules.helmet.no_helmet_duration_s)
    min_ring_a = max(ev.clip_pre_s, dwell_or_helmet / 2) + 2
    if ev.ring_buffer_s < min_ring_a:
        issues.append(
            ConfigIssue(
                "evidence.ring_buffer_s",
                f"至少为 {min_ring_a:g} 秒（由 clip_pre_s、dwell_threshold_s、"
                f"helmet.no_helmet_duration_s 推导），当前为 {ev.ring_buffer_s:g}",
            )
        )
    min_ring_b = cfg.sop.window_before_s + cfg.sop.window_after_s + 1
    if ev.ring_buffer_s < min_ring_b:
        issues.append(
            ConfigIssue(
                "evidence.ring_buffer_s",
                f"至少为 {min_ring_b:g} 秒（需容纳 SOP 检查窗口 window_before_s + window_after_s + 1），"
                f"当前为 {ev.ring_buffer_s:g}",
            )
        )

    vlm = cfg.vlm
    if vlm.review_enabled and vlm.provider == "openai_compatible":
        oc = vlm.openai_compatible
        if not oc.base_url:
            issues.append(
                ConfigIssue(
                    "vlm.openai_compatible.base_url",
                    "复核开启且 provider 为 openai_compatible 时必填（演示可改用 provider: mock）",
                )
            )
        if not oc.model:
            issues.append(
                ConfigIssue(
                    "vlm.openai_compatible.model",
                    "复核开启且 provider 为 openai_compatible 时必填（演示可改用 provider: mock）",
                )
            )
        if not env.get(oc.api_key_env):
            issues.append(
                ConfigIssue(
                    "vlm.openai_compatible.api_key_env",
                    f"环境变量 {oc.api_key_env} 未设置或为空（演示可改用 provider: mock 或关闭 review_enabled）",
                )
            )
    return issues


def parse_config(data: Any, env: Mapping[str, str] | None = None) -> AppConfig:
    """校验已解析的配置字典（字段级 + 跨字段），返回 AppConfig。

    ``env`` 用于检查密钥环境变量，默认读取 ``os.environ``。
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError([ConfigIssue("(root)", "配置顶层必须是映射（mapping）")])

    try:
        cfg = AppConfig.model_validate(data)
    except ValidationError as exc:
        issues = [
            ConfigIssue(_format_loc(tuple(err["loc"])), str(err["msg"])) for err in exc.errors()
        ]
        raise ConfigError(issues) from exc

    issues = _cross_checks(cfg, os.environ if env is None else env)
    if issues:
        raise ConfigError(issues)
    return cfg


def load_config(path: str | Path, env: Mapping[str, str] | None = None) -> AppConfig:
    """读取 YAML 配置文件并校验，非法时抛出 ConfigError。"""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError([ConfigIssue("(file)", f"配置文件不存在: {config_path}")])
    try:
        with config_path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError([ConfigIssue("(yaml)", f"YAML 解析失败: {exc}")]) from exc
    return parse_config(data, env)
