---
inclusion: always
---

# Chimera Patrol 项目结构与代码约定

依据：`.kiro/specs/chimera-patrol-inspection/requirements.md` 与 `design.md`。与本文件冲突时以 spec 为准。

## 1. 技术边界

- 单进程单体：Python 3.11+、FastAPI、SQLite（WAL）。
- 前端为原生 HTML/CSS/JS，由 FastAPI `StaticFiles` 提供。不引入前端构建链（无 npm 工程、无打包器），不做 SPA。
- 视频画面走 MJPEG（`multipart/x-mixed-replace`），事件推送走 WebSocket。
- 不使用消息队列、微服务或前端框架。

## 2. 目录结构

```
chimera-patrol/
  config.yaml            带中文注释的示例配置（随仓库发布）
  config.demo.yaml       演示配置（--demo 使用）
  sops.yaml              SOP 定义，含示例「入场作业准备」
  samples/videos/        样例视频（含真事件、含典型误报）
  scripts/               工具脚本（如 generate_samples.py）
  web/                   Dashboard 静态文件（index.html、event.html、css/、js/）
  chimera_patrol/
    __main__.py          入口：python -m chimera_patrol [--demo] [--config PATH]
    config.py            pydantic 配置模型与跨字段校验
    pipeline.py          EventPipeline
    runtime.py           按配置装配各组件与每路工作线程
    ingest/              Video_Ingest（discover.py、capture.py）
    detect/              Detector、跟踪器、DetectWorker、FrameResult
    rules/               geometry、RuleEngine 及各规则子模块
    evidence/            Evidence_Capturer、RingBuffer、关键帧与片段
    vlm/                 VLM_Adapter 协议、OpenAI 兼容实现、MockAdapter、响应规范化
    review/              VLM_Reviewer、Event_State_Machine、启动恢复
    sop/                 SOP 加载、采样时刻、结果归一化、SopChecker
    store/               schema.sql、EventRepository、SOP 仓储、SQL 统计
    api/                 FastAPI 应用、REST 路由、WsHub、MJPEG、统计与导出
    mcp/vlm_server.py    VLM_Adapter 的 MCP 服务（FastMCP，stdio）
  tests/
    property/            Hypothesis 属性测试
    integration/         线程、WebSocket、故障隔离等集成测试
    smoke/               交付物完整性与启动冒烟测试
  .kiro/
    specs/chimera-patrol-inspection/   requirements.md、design.md、tasks.md
    steering/                          本目录
    hooks/                             Agent hook
    settings/mcp.json                  工作区 MCP 登记
```

## 3. 组件命名

文档中的组件名（需求术语表）与代码落点对应如下。文档与注释使用组件名，代码标识符使用 PascalCase 类名，已在设计中给出的类名优先使用。

| 组件名 | 代码落点 | 主要标识符 |
|---|---|---|
| Video_Ingest | `ingest/discover.py`、`ingest/capture.py` | `discover_cameras`、CaptureThread |
| Detector | `detect/detector.py` | `Detector` |
| Rule_Engine | `rules/engine.py` | `RuleEngine` |
| Evidence_Capturer | `evidence/` | `RingBuffer`、`select_frame`、关键帧与片段保存 |
| VLM_Adapter | `vlm/protocol.py` | `VLMAdapter`（Protocol）；实现 `OpenAICompatibleAdapter`、`MockAdapter` |
| VLM_Reviewer | `review/reviewer.py` | `VlmReviewer` |
| Event_State_Machine | `review/state_machine.py`、`review/decision.py` | `can_transition`、`decide_status` |
| SOP_Checker | `sop/checker.py` | `SopChecker` |
| Dashboard | `web/` | `index.html`、`event.html` |
| Stats_Panel | `web/js/stats.js`、`api/stats.py` | `compute_stats` |
| Config_File | `config.yaml`、`chimera_patrol/config.py` | 配置模型 `AppConfig` |

- 事件类型取值固定为 `intrusion`、`loitering`、`smoke_fire`、`no_helmet`、`sop_violation`。
- 事件状态取值固定为 `pending_review`、`confirmed`、`dismissed`、`needs_human`、`review_failed`。
- 术语“检查失败待重试”是 SOP 检查的整体状态，代码取值为 `check_failed_retry`。

## 4. 异步与线程边界

| 执行体 | 数量 | 职责 |
|---|---|---|
| asyncio 主循环（uvicorn） | 1 | REST、WebSocket、MJPEG、EventPipeline、VLM_Reviewer、SopChecker 调度 |
| CaptureThread | 每路 Camera 1 个 | 读帧、按源 fps 节拍、写 FrameSlot 与 RingBuffer，按抽帧间隔投递到 DetectQueue |
| DetectWorker | 每路 Camera 1 个 | 从 DetectQueue 取帧，运行 Detector 与 RuleEngine |
| 线程池 | 共享 | JPEG 编码、视频片段编码 |

规则：

1. **线程到事件循环**：DetectWorker 通过 `asyncio.run_coroutine_threadsafe(pipeline.submit(candidate), loop)` 投递候选事件。投递失败只记录日志。
2. **采集与检测解耦**：`DetectQueue` 的 `maxsize=1`，满时丢弃旧帧。推理慢只降低处理帧率，不拖慢采集与画面显示。
3. **阻塞操作离开事件循环**：数据库、磁盘、JPEG 与视频编码通过 `asyncio.to_thread` 或线程池执行，不在事件循环中直接调用。
4. **SQLite**：单连接写入，写入经写锁串行化；读取使用独立连接；均经 `asyncio.to_thread` 调用。连接初始化启用 WAL、外键与 `busy_timeout`。
5. **YOLO 共享**：person、烟火、安全帽三份权重各加载一次。`predict` 由 `threading.Lock` 串行化。
6. **跟踪器隔离**：每路 Camera 持有独立的 ByteTrack 实例，不共享跟踪器状态，避免跨路 track ID 串号。
7. **VLM 调用**：经 `asyncio.Semaphore(vlm.max_concurrency)` 限流，并以 `asyncio.wait_for(..., vlm.timeout_s)` 限时。调用不在检测线程中进行。
8. **MJPEG**：读取 FrameSlot 的最新帧，编码在线程池完成，画面输出不等待检测结果。
9. **时间基准**：规则时长取已处理帧时间戳之差，不依赖 `sleep` 与墙钟。时间戳为采集时刻的 UTC 毫秒。

## 5. 代码约定

- **语言与类型**：Python 3.11+。公开函数与数据结构带类型注解。
- **配置**：配置模型使用 pydantic，唯一来源为 `config.yaml`（SOP 定义为 `sops.yaml`）。
  - 阈值、超时、帧数、冷却时间、ROI 数量与顶点数限制、开关、采样参数等一律从配置读取。业务代码中不写死任何阈值。
  - 字段默认值只在 `config.py` 的配置模型中定义一次。
  - 非法配置使启动失败，错误信息包含字段路径。
- **内部数据结构**：`CandidateEvent`、`VLMVerdict`、`FrameResult` 等使用 dataclass。API 与配置边界使用 pydantic。
- **时间字段**：数据库与 API 时间字段统一为 UTC 毫秒整数，字段名以 `_ms` 结尾。
- **纯逻辑与 I/O 分离**：几何、规则判定、决策、结果归一化、统计口径、采样时刻均为纯函数或纯状态对象，不直接访问 I/O。
- **VLM 访问**：业务代码只通过 `VLMAdapter` 协议调用 VLM。更换提供方时只新增或替换适配实现。
- **密钥与日志**：API Key 只从环境变量读取，不写入配置、日志或响应。日志不记录图片内容与密钥，记录事件生成、VLM 请求与响应摘要、状态变更。
- **证据路径**：数据库只存相对 `evidence.base_dir` 的路径。对外通过 `/media/{相对路径}` 提供，路径须经规范化且不得越出 `base_dir`。
- **前端**：原生 HTML、CSS、JavaScript。不新增 `package.json`、打包配置或框架依赖。

## 6. 测试约定

- 框架：pytest。属性测试库：Hypothesis，不自行实现属性测试框架。
- 目录：
  - `tests/property/`：每个属性一个测试文件，如 `test_geometry.py`、`test_cooldown.py`。
  - `tests/integration/`：线程并发、WebSocket 推送、慢 VLM、故障隔离。
  - `tests/smoke/`：交付物完整性与 `--demo` 启动冒烟。
- 属性测试：
  - 每个属性对应一个属性测试，属性编号与 `design.md` 的 Correctness Properties 一致。
  - `@settings(max_examples=100)` 起步，不少于 100 次。
  - 注释标注 `# Feature: chimera-patrol-inspection, Property {编号}: {属性标题}`，并标注 `**Validates: Requirements x.y**`。
  - 生成器约束在合法输入空间内。
- 测试替身与资源：
  - 规则引擎测试直接喂入构造的 `FrameResult` 序列，不加载 YOLO 权重。
  - VLM 相关测试使用 `MockAdapter` 或注入故障的假适配器。
  - 数据库测试使用临时 SQLite 文件。
  - API 测试使用 FastAPI `TestClient`，覆盖 WebSocket 与 MJPEG 首分片。
- 不用属性测试的范围：Dashboard 渲染与布局、YOLO 与 ByteTrack 库本身、外部 VLM API 行为、视频编码与 RTSP 连接、性能指标。这些分别由示例测试、集成测试、冒烟测试或手动验证覆盖。

## 7. 范围外

以下内容不在本期范围内，不得在代码中引入：200 路以上并发、用户权限与多租户、短信与电话告警、模型训练与微调、移动端适配、复杂 SOP 场景库（时序型、持续状态型、空间关系型、读数记录型）。
