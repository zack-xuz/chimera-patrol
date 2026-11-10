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

- `--demo` 使用 `config.demo.yaml`（VLM 提供方为 mock，不需要密钥）。
- 不加 `--demo` 时使用 `config.yaml`。真实 VLM 需要设置环境变量（默认 `VLM_API_KEY`），密钥不写入配置文件。
- `--config PATH` 指定其他配置文件，优先于 `--demo`。

启动后访问 <http://127.0.0.1:8000/> 打开巡检台。

## 演示路径

演示使用 `samples/videos/` 中的两段样例视频（`intrusion_real.mp4` 与 `false_positive.mp4`），`--demo` 会把两路视频都接入。

1. 启动服务，打开巡检台，视频墙显示两路画面和在线状态。
2. 事件流中出现 CV 报警（当前样例中为烟火事件），每条带 CV 与 VLM 两层徽标。
3. 点击事件进入详情页，查看回放区（关键帧与片段）、CV 判定区和 VLM 判定区，也可以人工改判。
4. 统计面板显示 CV 总报警数、拦截数、确认数、待人工数与误报下降率，可按摄像头和类型分桶，并导出 JSON 或 CSV。
5. SOP 面板选择「入场作业准备」与摄像头，点击「触发检查」。检查结束后逐步骤亮灯，失败时可以点击「重新检查」。

**当前演示的已知限制（需要处理后才能完整演示）**

- 样例视频是合成视频。在 `intrusion_real.mp4` 的 300 帧中，yolov8n 检测不到人（conf 0.1 也没有），因此不会产生区域闯入事件。两段样例上烟火模型都会产生 `smoke_fire` 事件。
- `--demo` 的 VLM 为 mock，默认 verdict 为 `uncertain`，所有事件进入人工队列。因此演示中不会有 VLM 拦截误报的效果，误报下降率为 0。要演示拦截，可在 `config.demo.yaml` 的 `vlm.mock.by_camera` 中指定结论，或者换成真实 VLM。这种结论是模拟的，不反映画面内容。

## 配置

- `config.yaml`：带中文注释的完整示例配置，包括视频源、Camera 与 ROI、检测、规则、证据、VLM、SOP 与日志。修改后重启服务生效。
- `config.demo.yaml`：演示配置。
- `sops.yaml`：SOP 定义，内置示例「入场作业准备」（4 个步骤）。

ROI 为归一化坐标（0~1），只在配置中设置，巡检台只读显示。

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
