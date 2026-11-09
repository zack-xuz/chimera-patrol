# Chimera Patrol 事件数据契约

本文件是事件、SOP 检查、WebSocket 与错误格式的契约摘要，来源为 `.kiro/specs/chimera-patrol-inspection/design.md`。
修改契约时先改 design.md，再同步本文件。实现代码（后端、前端、测试）必须与本文件一致，不得新增或改名字段。

## 1. 时间字段约定

- 所有时间点统一为 **UTC 毫秒整数**，字段名以 `_ms` 结尾，例如 `start_ms`、`created_ms`、`ts_ms`。
- 数据库时间列同样为 UTC 毫秒整数（`INTEGER`）。
- 时长类配置使用 `_s` 后缀（秒），例如 `heartbeat_interval_s`、`dwell_threshold_s`。时长不是时间点，不使用 `_ms`。
- 查询参数 `from_ms`、`to_ms` 作用于 `created_ms`。

## 2. 事件类型与检测来源

| event_type | 允许的 detection_source | 产生方式 |
|---|---|---|
| `intrusion` | `yolo` | 参考点从 ROI 外进入 `intrusion` 区域 |
| `loitering` | `yolo` | 同一 track 在同一 `intrusion` 区域连续停留达到阈值 |
| `smoke_fire` | `yolo` | 烟火检测在同一位置连续若干已处理帧出现 |
| `no_helmet` | `yolo`、`vlm_fallback` | YOLO 关联失败持续达到阈值，或 VLM 兜底判定未佩戴 |
| `sop_violation` | `sop` | SOP 检查存在 `fail` 步骤 |

约束：`event_type = sop_violation` 当且仅当 `detection_source = sop`，且 `sop_run_id` 非空。其他类型的 `sop_run_id` 为空。

`detection_source` 取值：`yolo`、`vlm_fallback`、`sop`。状态机决策不读取 `detection_source`。

## 3. EventDTO

列表、详情与 WebSocket 共用同一结构。

```json
{
  "id": "evt_3f9a1c2b7d10",
  "camera_id": "intrusion_real",
  "camera_name": "东门通道",
  "event_type": "intrusion",
  "detection_source": "yolo",
  "status": "pending_review",
  "start_ms": 1790000000000,
  "end_ms": 1790000000000,
  "cv": {
    "confidence": 0.87,
    "track_id": 12,
    "roi_id": "gate_zone",
    "roi_type": "intrusion",
    "detail": { "label": "person", "box": [120, 80, 220, 400], "reference_point": "bottom_center" },
    "trajectory": [[1789999999000, 100, 80, 200, 400]]
  },
  "evidence": {
    "keyframe_annotated_url": "/media/intrusion_real/evt_3f9a1c2b7d10/keyframe_annotated.jpg",
    "keyframe_raw_url": "/media/intrusion_real/evt_3f9a1c2b7d10/keyframe_raw.jpg",
    "clip_url": null,
    "clip_status": "pending"
  },
  "vlm": null,
  "sop": null,
  "created_ms": 1790000000100,
  "updated_ms": 1790000000100
}
```

字段说明：

- `cv.trajectory` 仅在详情接口（`GET /api/events/{id}`）返回。列表与 WebSocket 消息中省略该字段。
- `cv.confidence`：`sop_violation` 为 `null`。
- `cv.roi_type`：`intrusion` 或 `work_zone`，无区域时为 `null`。
- `evidence.clip_status`：`pending`、`ready`、`failed`。`clip_url` 仅在 `ready` 时非空。
- `evidence` 中的 URL 以 `/media/` 开头，对应相对 `evidence.base_dir` 的路径。
- `vlm`：见第 4 节。
- `sop`：仅 `sop_violation` 事件非空，结构如下：

  ```json
  { "run_id": "sop_a1b2c3d4e5f6", "sop_id": "entry_prep",
    "failed_steps": [{ "step_id": "wear_gloves", "description": "佩戴手套", "evidence": "..." }] }
  ```

- 详情接口额外返回 `status_log`，每项为 `{ "from_status", "to_status", "actor_type", "actor", "reason", "created_ms" }`。

## 4. 事件状态与 VLM 复核块

### 4.1 状态枚举

`EventStatus = pending_review | confirmed | dismissed | needs_human | review_failed`

- `pending_review`：事件已创建，复核在途或排队中。
- `confirmed`：确认为真实事件。
- `dismissed`：判定为误报。事件仍保留在列表中。
- `needs_human`：进入人工队列，等待 Inspector 判定。
- `review_failed`：复核失败，进入人工队列。

人工队列计数 `pending_human = needs_human + review_failed`。

### 4.2 迁移规则

| 从 | 到 | 执行者 actor_type | 条件 / actor 取值 |
|---|---|---|---|
| `pending_review` | `confirmed` | `system` | VLM 判定 `true_positive` 且 confidence ≥ `vlm.auto_decision_threshold`（actor `vlm_reviewer`） |
| `pending_review` | `dismissed` | `system` | VLM 判定 `false_positive` 且 confidence ≥ 阈值（actor `vlm_reviewer`） |
| `pending_review` | `needs_human` | `system` | VLM 判定 `uncertain`，或 confidence < 阈值（actor `vlm_reviewer`）；复核关闭（actor `review_switch`）；SOP 事件（actor `sop_checker`） |
| `pending_review` | `review_failed` | `system` | 超时、适配器异常、响应非法（actor `vlm_reviewer`）；服务重启遗留（actor `startup_recovery`） |
| `needs_human` | `confirmed` | `inspector` | 改判 |
| `needs_human` | `dismissed` | `inspector` | 改判 |
| `review_failed` | `confirmed` | `inspector` | 改判 |
| `review_failed` | `dismissed` | `inspector` | 改判 |
| `confirmed` | `dismissed` | `inspector` | 改判 |
| `dismissed` | `confirmed` | `inspector` | 改判 |
| `confirmed` | `needs_human` | `inspector` | 改判 |
| `dismissed` | `needs_human` | `inspector` | 改判 |

规则：

- 系统迁移只能从 `pending_review` 发起。
- Inspector 改判的目标只能是 `confirmed`、`dismissed`、`needs_human`。目标不能是 `pending_review`。
- 当前状态为 `pending_review` 时不可改判，返回 409 `invalid_transition`（复核在途）。
- 目标与当前状态相同，返回 409 `invalid_transition`。
- 上表中未列出的迁移均不合法，例如 `review_failed → needs_human`。
- 每次迁移（系统或人工）都在 `event_status_log` 追加一行。`events.status` 始终等于该事件最后一条日志的 `to_status`。
- 事件创建时写入首条日志：`from_status = null`，`to_status = pending_review`，`actor_type = system`，`actor = cv_pipeline`（CV 流水线创建事件，修订契约：此枚举值为系统 actor 的一员）。
- `inspector` 的 `actor` 为请求体中的 `operator`（1~32 字符）。该字段仅用于留痕，无认证。

### 4.3 VLM 复核块（`vlm`）

未复核时 `vlm` 为 `null`。有三种形态：

复核成功：

```json
{ "verdict": "false_positive", "confidence": 0.91, "description": "...", "suggestion": "...",
  "reviewed_ms": 1790000003000, "error": null, "skipped": false }
```

复核关闭（`vlm.review_enabled = false`）：

```json
{ "verdict": null, "confidence": null, "description": null, "suggestion": null,
  "reviewed_ms": null, "error": null, "skipped": true }
```

复核失败：`error` 为失败原因字符串，`skipped` 为 `false`，其余字段按失败情况为空。

字段约束：

- `verdict` ∈ `true_positive | false_positive | uncertain`，失败或跳过时为 `null`。
- `confidence` 为 [0, 1] 内的数，失败或跳过时为 `null`。
- `description` 长度 1 至 `vlm.description_max_chars`（默认 50）。
- `suggestion` 缺失时为空字符串。
- `error` 取值：`timeout`、`response_invalid`、`service_restart`、`keyframe_missing`，以及 `vlm_error: <摘要>`（适配层异常）。
- `skipped = true` 与 `error` 非空不会同时出现。

VLM 视图徽标取值：`未复核`、`true_positive`、`false_positive`、`uncertain`、`复核失败`、`复核关闭`。

## 5. SOP 检查

### 5.1 运行状态枚举

`SopRunStatus = running | completed | check_failed_retry`

- `running`：检查进行中。同一 Camera 同时最多一个 `running`。
- `completed`：检查完成。存在 `fail` 步骤时会产生一条 `sop_violation` 事件。
- `check_failed_retry`：VLM 超时、适配器错误、响应非法或无可用帧。不产生事件，不写入任何 `fail` 步骤。Inspector 可通过 `retry_of` 重新触发。

### 5.2 SopRunDTO

```json
{
  "id": "sop_a1b2c3d4e5f6", "sop_id": "entry_prep", "sop_name": "入场作业准备",
  "camera_id": "intrusion_real",
  "status": "completed",
  "triggered_ms": 1790000100000, "window_start_ms": 1790000095000, "window_end_ms": 1790000105000,
  "sample_count": 8, "sample_times_ms": [1790000095000, 1790000096428],
  "failure_reason": null, "retry_of": null, "operator": "张三",
  "steps": [
    { "index": 0, "step_id": "wear_helmet", "description": "佩戴安全帽", "status": "pass", "evidence": "...", "approximate": false }
  ],
  "event_id": "evt_...", "created_ms": 1790000100000, "finished_ms": 1790000112000
}
```

- `steps[].status` ∈ `pass | fail | unknown`。
- `failure_reason` 取值：`vlm_timeout`、`vlm_error`、`response_invalid`、`no_frames`、`service_restart`。
- 当 `status = check_failed_retry` 时：`steps = []`，`failure_reason` 非空，`event_id = null`。
- 当 `status ≠ check_failed_retry` 时：`failure_reason = null`。
- `retry_of` 必须指向 `check_failed_retry` 的运行记录，否则返回 422。

## 6. WebSocket 协议

端点：`GET /ws/events`（无参数）。所有消息为 JSON 文本帧。

### 6.1 信封

```json
{ "type": "event.created", "seq": 128, "ts_ms": 1790000000200, "data": {} }
```

- `type`：消息类型，见下表。
- `seq`：进程内单调递增整数。
- `ts_ms`：服务端发送时刻，UTC 毫秒。
- `data`：消息负载。

### 6.2 服务端到客户端消息

| type | data | 触发时机 |
|---|---|---|
| `hello` | `{ "server_time_ms", "vlm_review_enabled", "heartbeat_interval_s" }` | 连接建立后立即发送 |
| `event.created` | EventDTO（无 `cv.trajectory`） | 事件落库后 |
| `event.updated` | EventDTO（无 `cv.trajectory`） | 状态变化、VLM 结果写入、片段就绪或失败、人工改判 |
| `camera.status` | `{ "camera_id", "status": "online" \| "offline", "status_changed_ms" }` | Camera 上线或离线 |
| `sop.run.updated` | SopRunDTO | SOP 检查状态变化 |
| `heartbeat` | `{}` | 每 `ws.heartbeat_interval_s` 秒（默认 15） |

### 6.3 客户端到服务端消息

- 仅支持 `{ "type": "ping" }`。服务端回复 `{ "type": "pong", ... }`。
- 其他消息被忽略。

### 6.4 客户端行为

- 断线重连后，客户端不依赖补发。应重新调用 `GET /api/events` 与 `GET /api/cameras` 对账。
- 客户端按 `id` 做 upsert。重复消息无副作用。

## 7. 错误格式

所有非 2xx 的 JSON 响应使用同一结构：

```json
{ "error": { "code": "not_found", "message": "event evt_xxx not found" } }
```

错误码与 HTTP 状态：

| code | HTTP 状态 | 典型场景 |
|---|---|---|
| `not_found` | 404 | 事件、Camera、SOP 运行记录不存在 |
| `validation_error` | 422 | 请求参数非法（pydantic 校验失败）；`retry_of` 指向非 `check_failed_retry` 的运行记录 |
| `invalid_transition` | 409 | 改判非法、`pending_review` 改判、目标与当前相同 |
| `camera_offline` | 409 | SOP 触发时 Camera 离线 |
| `run_in_progress` | 409 | SOP 触发时同一 Camera 已有 `running` |
| `sop_not_applicable` | 422 | SOP 的 `camera_ids` 不包含该 Camera |
| `internal_error` | 500 | 未预期的服务端错误 |

`validation_error` 与 `sop_not_applicable` 的 HTTP 状态都是 422，客户端应以 `error.code` 区分。

## 8. HTTP 端点关键约定

- 所有 JSON 响应 `Content-Type: application/json`。
- 列表接口返回 `{ "items": [...] }`，事件列表额外返回 `total`、`limit`、`offset`。
- 事件列表排序：`created_ms DESC, id DESC`。
- ID 格式：事件 `evt_` + 12 位十六进制；SOP 运行 `sop_` + 12 位十六进制。
- 改判接口：`POST /api/events/{event_id}/status`，请求体 `{ "to_status", "operator", "reason" }`（`reason` 可选，≤ 200 字符）。
- SOP 触发：`POST /api/sop/runs` 成功返回 202 与 `running` 状态的 SopRunDTO。

## 9. SQLite 表名

| 表名 | 用途 |
|---|---|
| `cameras` | Camera 状态与来源 |
| `events` | 事件主表，包含 CV、证据、VLM 复核与 SOP 关联字段 |
| `event_status_log` | 状态迁移日志，每次迁移追加一行 |
| `sop_runs` | SOP 检查运行记录 |
| `sop_step_results` | SOP 步骤结果，主键 `(run_id, step_id)` |

连接初始化：`PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;`。

约束摘要：

- `events` 的 `sop_run_id` 唯一（仅非空时）。
- `sop_runs` 对 `camera_id` 在 `status='running'` 时唯一。
- `events.clip_status = 'ready'` 时 `clip_path` 必须非空。
- 证据路径列（`keyframe_annotated_path`、`keyframe_raw_path`、`clip_path`）存储相对 `evidence.base_dir` 的路径。
