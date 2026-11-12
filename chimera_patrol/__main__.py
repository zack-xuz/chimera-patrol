"""单进程入口：``python -m chimera_patrol [--demo] [--config PATH]``（任务 17.2；需求 10.1、12.1）。

启动顺序：

1. 读取并校验配置。非法配置在此处退出，错误信息包含字段路径。
2. 初始化日志与 SQLite 仓储。
3. uvicorn 在本进程的事件循环中运行 FastAPI 应用。应用启动时：
   加载 YOLO 权重，启动 PatrolRuntime（采集、检测、复核、SOP）。
   应用关闭时：停止所有线程，取消后台任务，关闭数据库。

``--demo`` 使用 ``config.demo.yaml``（VLM 提供方为 mock，无需密钥）。
``--config`` 指定其他配置文件，优先级高于 ``--demo``。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

from chimera_patrol.api.app import create_app
from chimera_patrol.api.context import ApiNotifier, AppContext
from chimera_patrol.api.ws import WsHub
from chimera_patrol.config import ConfigError, load_config
from chimera_patrol.env import load_project_env
from chimera_patrol.logging_setup import setup_logging
from chimera_patrol.runtime import PatrolRuntime
from chimera_patrol.sop.loader import SopConfigError, load_sops
from chimera_patrol.store.repository import EventRepository

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DEFAULT_CONFIG = ROOT / "config.yaml"
DEMO_CONFIG = ROOT / "config.demo.yaml"

log = logging.getLogger("chimera_patrol")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="chimera_patrol", description="Chimera Patrol 多路视频智能巡检")
    parser.add_argument("--demo", action="store_true", help="使用 config.demo.yaml 启动演示模式")
    parser.add_argument("--config", type=Path, default=None, help="配置文件路径（优先于 --demo）")
    return parser.parse_args(argv)


def _resolve_config(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config
    return DEMO_CONFIG if args.demo else DEFAULT_CONFIG


def run(argv: list[str] | None = None) -> int:
    load_project_env()
    args = _parse_args(argv)
    config_path = _resolve_config(args)
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        load_sops(cfg.sop.file)
    except SopConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    setup_logging(
        level=cfg.logging.level,
        log_dir=cfg.logging.dir,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
    )
    log.info("starting config=%s demo=%s", config_path, args.demo)

    cfg.database.path.parent.mkdir(parents=True, exist_ok=True)
    repo = EventRepository(cfg.database.path)
    hub = WsHub(
        heartbeat_interval_s=cfg.ws.heartbeat_interval_s,
        vlm_review_enabled=cfg.vlm.review_enabled,
    )
    ctx = AppContext(config=cfg, repo=repo, hub=hub)
    notifier = ApiNotifier(ctx)
    runtime: PatrolRuntime | None = None

    async def on_startup() -> None:
        nonlocal runtime
        from chimera_patrol.detect.detector import Detector

        detector = Detector(cfg.detector)
        runtime = PatrolRuntime(
            cfg,
            repo,
            detector,
            asyncio.get_running_loop(),
            notifier=notifier,
            control=ctx.control,
        )
        runtime.start()
        ctx.cameras = list(runtime.cameras)
        ctx.frame_slots = runtime.frame_slots
        ctx.sop = runtime.sop_checker
        ctx.rereview = runtime.rereview

    async def on_shutdown() -> None:
        if runtime is not None:
            await runtime.ashutdown()
        repo.close()

    app = create_app(ctx, web_dir=WEB_DIR, on_startup=on_startup, on_shutdown=on_shutdown)
    uvicorn.run(
        app,
        host=cfg.app.host,
        port=cfg.app.port,
        log_config=None,
        lifespan="on",
    )
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
