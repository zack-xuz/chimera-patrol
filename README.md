# Chimera Patrol 多路视频智能巡检

面向工业和能源场站的多路视频巡检工具。CV 管线（YOLO 检测、跟踪、ROI 规则）产生报警，VLM 对每条报警做二次研判，巡检台把两层判定融合展示，并支持 SOP 合规检查与统计导出。

技术栈：Python 3.11+、FastAPI、SQLite（WAL）。前端为原生 HTML、CSS、JavaScript，视频走 MJPEG，事件走 WebSocket。没有前端构建链。

## 依赖与安装

依赖清单见 `requirements.txt`，全部版本已精确锁定。

推荐使用 conda 环境（Windows 与 Linux 通用）：

```bash
conda create -n chimera-patrol python=3.11 -y
conda activate chimera-patrol
pip install -r requirements.txt
```

主要依赖：ultralytics（YOLO）、trackers（ByteTrack）、supervision、opencv-python、fastapi、uvicorn、pydantic、pyyaml、httpx、imageio-ffmpeg（片段编码，自带 ffmpeg）、mcp（FastMCP，VLM 的 MCP 服务）。测试使用 pytest 与 hypothesis。

## 权重获取

检测用的烟火与安全帽权重不进入 git（`.gitignore` 已排除 `weights/` 与 `*.pt`），需要在本地获取一次。

```bash
python scripts/fetch_weights.py
```

脚本下载以下两个文件到 `weights/`，并校验 SHA-256。已存在且校验通过的文件会跳过。校验失败时删除文件并报错退出。

| 本地文件 | 用途 | 来源 | 许可证 | 类别（config 中的配置值） |
|---|---|---|---|---|
| `weights/fire_smoke.pt` | 烟火检测 | [andreasN78/fire-smoke-detection-yolov8](https://github.com/andreasN78/fire-smoke-detection-yolov8)，文件 `weights/best.pt`，YOLOv8s | 基于 Ultralytics YOLOv8，AGPL-3.0；仓库代码为 MIT；数据集部分为 CC BY 4.0 | `Fire`、`smoke`（模型另有一个 `default` 类别，规则中忽略） |
| `weights/helmet.pt` | 安全帽检测 | [iaac-maai/day4-example-ppe-detector](https://huggingface.co/iaac-maai/day4-example-ppe-detector)，文件 `model.pt`，YOLO26s | AGPL-3.0；训练数据集 CC BY 4.0 | 佩戴 `Hardhat`，未佩戴 `NO-Hardhat` |

- 两个权重都是 AGPL-3.0 许可。本仓库不分发权重，只提供获取脚本。在仓库内部运行不涉及分发；如需对外分发权重或基于其提供网络服务，请先确认 AGPL-3.0 的义务。
- 候选但未采用：`melihuzunoglu/ppe-detection`（AGPL-3.0，类别 helmet、no-helmet，可直接加载）。`luminous0219/fire-and-smoke-detection-yolov8` 的权重需要 `dill` 才能加载，项目依赖中没有，因此未采用。
- 类别名与权重不一致时，修改 `config.yaml` 中 `detector.*.classes` 与 `*_classes` 的配置，不要改代码。检测器对类别名做区分大小写的精确匹配。

**person 权重 `yolov8n.pt`**：由 Ultralytics 在首次运行时自动从 GitHub 下载到仓库根目录（COCO 预训练，AGPL-3.0），无需手动处理。首次启动会多等待几秒。离线环境请提前把 `yolov8n.pt` 放到仓库根目录。

## 启动

Windows 与 Linux 使用同一条命令（在仓库根目录执行）：

```bash
python -m chimera_patrol --demo
```

- `--demo` 使用 `config.demo.yaml`。
- 不加 `--demo` 时使用 `config.yaml`。两者的 VLM 都是商汤日日新 TokenPlan（OpenAI 兼容接口），启动前必须设置环境变量 `VLM_API_KEY`，否则配置校验失败并提示字段路径。密钥只从环境变量读取，不写入配置文件、日志或响应。
  - PowerShell（当前会话）：`$env:VLM_API_KEY = "<你的密钥>"`
  - PowerShell（永久，新开终端生效）：`setx VLM_API_KEY "<你的密钥>"`
  - Linux / macOS：`export VLM_API_KEY="<你的密钥>"`
  - 也可以把密钥写入项目根目录的 `.env` 文件：复制 `.env.example` 为 `.env`，填写 `VLM_API_KEY=` 后的值。程序启动时自动读取，已在系统或 shell 中设置的同名变量优先，不会被 `.env` 覆盖。`.env` 已被 `.gitignore` 排除。
- VLM 地址为 `https://token.sensenova.cn/v1`，模型为 `sensenova-6.8-flash`，超时 20 秒（见 `vlm.openai_compatible.*` 与 `vlm.timeout_s`）。
- `--config PATH` 指定其他配置文件，优先于 `--demo`。

启动后访问 <http://127.0.0.1:8000/> 打开巡检台。

## 演示路径

演示使用 `samples/videos/` 中的两段真实素材（`intrusion_real.mp4` 与 `false_positive.mp4`），`--demo` 会把两路视频都接入。

## 样例视频

| 文件 | 内容 | 时长 | 来源 | 许可 |
|---|---|---|---|---|
| `samples/videos/intrusion_real.mp4` | 仓库通道中两名戴安全帽、穿反光背心的工人行走，用于区域闯入与滞留 | 14 秒（由 `samples/raw/10817415-hd_1920_1080_30fps.mp4` 缩放为 960×540 得到） | Pexels，视频「Workers with Safety Helmets in Warehouse」，https://www.pexels.com/video/workers-with-safety-helmets-in-warehouse-10817415/ | Pexels License，可免费用于商业用途，无需署名 |
| `samples/videos/false_positive.mp4` | 工厂烟囱排放蒸汽，傍晚光影变化，用于观察烟火模型的误报 | 30 秒（取原片第 15 秒起的 30 秒，缩放为 960×540） | 原片文件名 `13101854_3840_2160_30fps.mp4`，来源页面待确认（见下） | 待确认 |

- 原始素材放在 `samples/raw/`，该目录已被 `.gitignore` 排除（约 223 MB，其中 `13101854_3840_2160_30fps.mp4` 为 198 MB），不随仓库分发。在新环境中使用时，需要先把原始素材放回该目录，再运行 `python scripts/generate_samples.py` 重新剪辑。`samples/videos/` 中的剪辑结果已入库。
- 剪辑由 `scripts/generate_samples.py` 完成（使用 imageio-ffmpeg 自带的 ffmpeg）。
- 闯入视频的时长只有 14 秒，低于 20 秒的目标。演示时视频会循环播放（`video.loop: true`）。
- 误报视频的来源页面与许可尚未核实，发布前需要补充。

**演示路径**

1. 启动服务，打开巡检台，视频墙显示两路画面和在线状态。
2. 事件流中出现 CV 报警（当前样例中为烟火事件），每条带 CV 与 VLM 两层徽标。
3. 点击事件进入详情页，查看回放区（关键帧与片段）、CV 判定区和 VLM 判定区，也可以人工改判。
4. 统计面板显示 CV 总报警数、拦截数、确认数、待人工数与误报下降率，可按摄像头和类型分桶，并导出 JSON 或 CSV。
5. SOP 面板选择「入场作业准备」与摄像头，点击「触发检查」。检查结束后逐步骤亮灯，失败时可以点击「重新检查」。

**当前演示的已知限制（需要处理后才能完整演示）**

- 区域闯入已在真实素材上触发（见上文的 ROI 标定）。
- 烟火阈值：`config.demo.yaml` 中 `rules.smoke_fire.conf_threshold` 为 0.30，`config.yaml` 仍为 0.5。原因见下。
- 演示的 VLM 复核由 SenseNova 完成。没有设置 `VLM_API_KEY` 时，`--demo` 无法启动。

## 烟火阈值（演示配置）

`config.yaml` 的烟火检测置信度阈值为 0.5，这是需求规定的默认值。`config.demo.yaml` 把它调低为 0.30，只用于演示。

在 `false_positive.mp4`（工厂烟囱蒸汽）上，按运行时的抽帧（每 3 帧取 1 帧）和检测器封装统计，烟火模型的命中情况如下（`smoke_fire` 需要同一位置连续 3 个已处理帧命中）：

| 阈值 | 命中帧数（共 300 帧） | 最长连续命中段 | 能否触发 `smoke_fire` |
|---|---|---|---|
| 0.45 | 0 | 0 | 否 |
| 0.40 | 1 | 1 | 否 |
| 0.35 | 6 | 4 | 是 |
| 0.30 | 35 | 13 | 是 |
| 0.25 | 131 | 34 | 是 |

演示配置取 0.30，这段误报素材能产生 `smoke_fire` 事件，进入事件流后由 SenseNova 复核，用于演示误报拦截。0.30 也会放过更多弱信号，误报会增加。正式运行请使用 `config.yaml` 的 0.5，不要沿用演示配置的值。

## 配置

- `config.yaml`：带中文注释的完整示例配置，包括视频源、Camera 与 ROI、检测、规则、证据、VLM、SOP 与日志。修改后重启服务生效。
- `config.demo.yaml`：演示配置。
- `sops.yaml`：SOP 定义，内置示例「入场作业准备」（4 个步骤）。

ROI 为归一化坐标（0~1），只在配置中设置，巡检台只读显示。

## 检测暂停与配置展示

- 每路摄像头可以在视频墙上点击「暂停检测」。暂停只停止检测与报警，视频流继续播放，画面上显示“检测已暂停”。在途的 VLM 复核照常完成。恢复后，该路从干净状态重新开始检测。
- 暂停状态只存在内存中，不落库。进程重启后所有摄像头默认为运行状态。
- 接口：`POST /api/cameras/{camera_id}/pause`、`POST /api/cameras/{camera_id}/resume`；`GET /api/cameras` 每路包含 `paused`。
- 首页的“当前生效配置”为只读展示，包括每路的检测阈值、ROI、冷却时间，规则全局参数，以及 VLM 的提供方、模型、超时和复核开关。修改配置需编辑 `config.yaml` 并重启。
- 配置接口 `GET /api/config/effective` 对 API Key 脱敏：只返回“是否已设置”，不返回取值。

## VLM 复检

- 在事件详情页，待人工（`needs_human`）和复核失败（`review_failed`）的事件可以点击「VLM 复检」。复检会把事件迁移回复核中（`pending_review`），清空上一次的 VLM 结果，然后重新复核，结论仍按原来的决策规则迁移。
- 其他状态（已确认、已拦截、复核中）不能复检。复核开关关闭时也不能复检。
- 复检记录在状态日志中，原因为「复检请求」，操作人为请求体中的 `operator`。
- 接口：`POST /api/events/{event_id}/review`，请求体 `{ "operator": "..." }`，成功返回 202。

## 测试

```bash
python -m pytest
```

测试包括属性测试（`tests/property/`，Hypothesis）、集成测试（`tests/integration/`）与冒烟测试（`tests/smoke/`）。端到端测试 `tests/integration/test_e2e.py` 需要本地权重，缺失时自动跳过。

## 目录结构

```
config.yaml / config.demo.yaml   配置
sops.yaml                        SOP 定义
samples/videos/                  样例视频
scripts/                         generate_samples.py（样例生成）、fetch_weights.py（权重获取）
web/                             巡检台前端（index.html、event.html、css/、js/）
chimera_patrol/                  后端包（ingest、detect、rules、evidence、vlm、review、sop、store、api、mcp）
tests/                           测试
.kiro/                           spec、steering、hooks、MCP 登记
```

VLM 的 MCP 服务：`python -m chimera_patrol.mcp.vlm_server --config config.yaml`，已在 `.kiro/settings/mcp.json` 登记为 `chimera-vlm`。
