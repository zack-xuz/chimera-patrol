# Implementation Plan: Chimera Patrol 多路视频智能巡检

## Overview

按三个里程碑推进。M1（第 1 周）搭建脚手架、视频接入、检测跟踪、规则引擎与证据截取；M2（第 2 周）接入 VLM 适配层、复核、状态机、安全帽双路径与 MCP 登记；M3（第 3 周）完成 API、统计、SOP 检查、前端、演示模式与收尾验证。实现语言为 Python 3.11+，模块划分与接口以 design.md 为准。

约定：

- 测试类子任务（单元测试、属性测试、集成测试）默认标记为可选（`*`），但以下属性测试为必做、不带 `*`：任务 4 规则引擎的全部属性测试（几何、闯入滞留、烟火同位、安全帽关联、未戴帽计时、冷却去重），以及 M2 中复核决策与状态迁移（10.2、10.4）、VLM 响应规范化（8.2）的属性测试。属性测试使用 Hypothesis，每个属性不少于 100 次示例。
- 核心实现子任务不标记为可选。
- 涉及 Ultralytics、supervision、FastMCP、imageio-ffmpeg 的任务，第一步先按 requirements.txt 锁定版本的官方文档核对 API 再写代码。

## Tasks

### M1（第 1 周）：脚手架、视频接入、检测、规则与证据

- [x] 1. 仓库脚手架与工程资产
  - [x] 1.1 创建目录结构与依赖清单
    - 先按 requirements.txt 锁定版本的官方文档核对 API 再写代码（核对 Ultralytics、supervision、mcp（FastMCP）、imageio-ffmpeg 的当前稳定版本与 API）
    - 按设计目录结构创建 chimera_patrol/ 各子包（ingest、detect、rules、evidence、vlm、review、sop、store、api、mcp）、web/、tests/（含 property、integration、smoke 子目录）、samples/videos/、scripts/
    - 编写 requirements.txt，锁定 ultralytics、supervision、fastapi、uvicorn、pydantic、pyyaml、opencv-python、numpy、httpx、imageio-ffmpeg、mcp、pytest、hypothesis 的版本
    - 保留仓库根目录下的 .kiro 目录
    - _Requirements: 12.1, 12.2, 13.5_

  - [x] 1.2 实现 Config_File 模型与加载校验
    - 用 pydantic 在 chimera_patrol/config.py 中实现配置模型，字段、默认值与取值范围依据设计的 config.yaml 字段表，包括 vlm.auto_decision_threshold 与 rules.helmet.min_person_height_px
    - 实现跨字段校验：ROI 数量与顶点数上限、evidence.ring_buffer_s 下限、SOP 窗口下限、VLM 必填项与密钥环境变量检查；非法配置启动失败并给出字段路径
    - 编写带中文注释的 config.yaml 示例
    - _Requirements: 9.1, 9.2, 9.3, 2.3, 3.1, 3.11_
    - 设计参考：config.yaml 完整字段、跨字段校验

  - [ ]* 1.3 编写配置属性测试
    - **Property 3: 配置往返与校验**
    - **Validates: Requirements 2.3, 3.1, 3.11, 9.1**
    - 测试文件 tests/property/test_config.py，Hypothesis 示例数不少于 100

  - [x] 1.4 实现日志与可观测性
    - 按 logging 配置实现轮转日志，提供事件生成、VLM 请求与响应摘要、状态变更的日志记录函数
    - 日志不包含图片内容与 API Key
    - _Requirements: 12.6_

  - [x] 1.5 实现 SQLite 存储层
    - 在 chimera_patrol/store/schema.sql 中写入 cameras、events、event_status_log、sop_runs、sop_step_results 的表与索引，与设计 DDL 一致
    - 在 chimera_patrol/store/repository.py 中实现 EventRepository：连接初始化（WAL、外键、busy_timeout）、写锁、insert_event（同时写入首条状态日志）、get_event、list_events（过滤与分页）、update_evidence
    - _Requirements: 3.10, 12.5, 6.5_
    - 设计参考：Data Models、SQLite DDL

  - [ ]* 1.6 编写事件持久化往返属性测试
    - **Property 10: 事件持久化往返**
    - **Validates: Requirements 3.10, 12.5**
    - 测试文件 tests/property/test_repository_roundtrip.py，Hypothesis 示例数不少于 100

  - [ ]* 1.7 编写事件筛选属性测试
    - **Property 16: 事件筛选**
    - **Validates: Requirements 6.5, 7.6**
    - 测试文件 tests/property/test_list_events.py，Hypothesis 示例数不少于 100

  - [x] 1.8 编写 Steering 文档：项目结构与代码约定
    - 写入 .kiro/steering/project-structure.md，说明模块划分、组件命名（Video_Ingest、Rule_Engine 等）、异步与线程边界、测试目录约定
    - _Requirements: 13.2_

  - [x] 1.9 编写 Steering 文档：事件数据契约
    - 写入 .kiro/steering/event-contract.md，覆盖事件类型、EventDTO 字段、状态枚举、WebSocket 信封、错误格式与时间字段约定（*_ms）
    - _Requirements: 13.2_

  - [ ] 1.10 创建 Agent Hook：保存 Python 文件后运行冒烟测试
    - 使用 createHook 在 .kiro/hooks/ 下创建 PostFileSave 类型的 hook，matcher 匹配 .py 文件，命令运行 tests/smoke 下的冒烟测试
    - _Requirements: 13.3_

  - [x] 1.11 创建 Agent Hook：事件类型变更时同步契约文档
    - 使用 createHook 创建 agent 类型的 hook，matcher 匹配 chimera_patrol 下 rules 与 store 目录中的文件保存事件
    - 提示词要求：若新增或修改了事件类型，同步更新 .kiro/steering/event-contract.md 中的事件类型表与 schema
    - _Requirements: 13.3_

- [x] 2. 视频接入 Video_Ingest
  - [x] 2.1 实现 discover_cameras 与 Camera 数据结构
    - 在 chimera_patrol/ingest/discover.py 中扫描 video.source_dir 中允许扩展名的文件，按文件名排序取前 max_cameras 个
    - Camera ID 取文件名主干；按 cameras[] 覆盖 name、rtsp_url、rois；带 rtsp_url 时使用 RTSP 源替代文件源
    - 无对应文件但有 rtsp_url 的条目创建纯 RTSP Camera
    - _Requirements: 1.1, 1.3, 1.4_
    - 设计参考：视频接入（Video_Ingest）

  - [ ]* 2.2 编写 Camera 发现属性测试
    - **Property 1: Camera 发现与源解析**
    - **Validates: Requirements 1.1, 1.3, 1.4**
    - 测试文件 tests/property/test_discover_cameras.py，Hypothesis 示例数不少于 100

  - [x] 2.3 实现 CaptureThread、FrameSlot 与离线重连
    - 在 chimera_patrol/ingest/capture.py 中为每路 Camera 实现采集线程，按源 fps 节拍读帧，写入 FrameSlot 与 RingBuffer，并按抽帧间隔投递到 DetectQueue
    - 连续读帧失败达到 offline_after_failures 时置为 offline 并推送 camera.status，之后按 reconnect_interval_s 重试
    - 文件源读到末尾时循环回绕，并触发该路 tracker 与规则状态重置（假设 A4）
    - _Requirements: 1.1, 1.2, 1.3, 12.4_
    - 设计参考：线程与异步模型、Error Handling

  - [ ]* 2.4 编写坏文件隔离单元测试
    - 构造一路损坏视频与一路正常视频，验证损坏路被标记为离线，正常路持续输出帧
    - _Requirements: 1.2, 12.4_

- [x] 3. 检测与跟踪 Detector
  - [x] 3.1 实现 YOLO 权重共享加载与串行推理
    - 先按 requirements.txt 锁定版本的官方文档核对 Ultralytics 的模型加载与 predict API 再写代码
    - 在 chimera_patrol/detect/detector.py 中加载 person、smoke/fire、helmet 三份权重各一次，推理通过 threading.Lock 串行化
    - 类别名取配置值，烟火类别以所选模型实际支持为准
    - _Requirements: 2.1_
    - 设计参考：线程与异步模型要点 1

  - [x] 3.2 实现每路 ByteTrack 跟踪
    - 变更（2026）：supervision 0.31.5 已移除 sv.ByteTrack，改用 trackers==2.7.0 的 ByteTrackTracker，详见 design.md 调研结论
    - 先按 requirements.txt 锁定版本的官方文档核对 supervision 的 ByteTrack API 再写代码
    - 在 chimera_patrol/detect/tracker.py 中为每路 Camera 持有独立跟踪器实例，仅对 person 分配稳定 track ID
    - track 丢失保留时长取 detector.tracker.lost_buffer_s
    - _Requirements: 2.2_
    - 设计参考：调研结论（跟踪器状态不共享）

  - [x] 3.3 实现抽帧选择与 DetectWorker
    - 实现 select_processed（frame_index 对 N 取余为 0）与 FrameResult 数据结构
    - 在 chimera_patrol/detect/worker.py 中实现 DetectWorker，从 DetectQueue（maxsize=1，满时丢弃旧帧）取帧，调用 Detector 生成 FrameResult 并交给 Rule_Engine
    - 推理异常时记录日志并丢弃当前帧，继续处理
    - _Requirements: 2.3, 2.4_
    - 设计参考：线程与异步模型

  - [ ]* 3.4 编写抽帧选择属性测试
    - **Property 2: 抽帧选择**
    - **Validates: Requirements 2.3**
    - 测试文件 tests/property/test_frame_skip.py，Hypothesis 示例数不少于 100

  - [ ]* 3.5 编写合成视频跟踪与显示不阻塞集成测试
    - 先按 requirements.txt 锁定版本的官方文档核对 supervision ByteTrack API 再写代码
    - 使用合成视频验证 person 的 track ID 跨帧稳定
    - 注入慢推理检测器，验证采集线程与最新帧读取不被阻塞
    - _Requirements: 2.2, 2.4_

- [x] 4. 规则引擎 Rule_Engine（纯逻辑）
  - [x] 4.1 实现几何纯函数
    - 在 chimera_patrol/rules/geometry.py 中实现 ref_point（bottom_center 与 center 两种）、point_in_polygon（边界视为在内）、iou、ioh（helmet 面积为 0 时返回 0）、head_region
    - _Requirements: 3.2, 3.5, 3.6_
    - 设计参考：Rule_Engine 规则设计 · 几何基础

  - [x] 4.2 编写几何不变量属性测试
    - **Property 6: 头部区域几何不变量**
    - **Validates: Requirements 3.5, 3.6**
    - 测试文件 tests/property/test_geometry.py，Hypothesis 示例数不少于 100

  - [x] 4.3 实现区域闯入与人员滞留
    - 在 chimera_patrol/rules/intrusion.py 中实现 intrusion 状态：参考点从不在 ROI 内变为在 ROI 内时生成一条 intrusion 事件，离开需超过 roi_exit_grace_s 才确认
    - 实现 dwell 状态：同一 track 在 ROI 内连续停留达到 dwell_threshold_s 时生成一条 loitering 事件，离开后重置
    - 仅对 type 为 intrusion 的 ROI 生效
    - _Requirements: 3.1, 3.2, 3.3_
    - 设计参考：Rule_Engine 规则设计 · 区域闯入、人员滞留

  - [x] 4.4 编写闯入与滞留属性测试
    - **Property 4: 区域闯入与滞留**
    - **Validates: Requirements 3.2, 3.3**
    - 测试文件 tests/property/test_intrusion_dwell.py，Hypothesis 示例数不少于 100

  - [x] 4.5 实现 SmokeFireTracker
    - 在 chimera_patrol/rules/smoke_fire.py 中实现点位跟踪：过滤低于 smoke_fire.conf_threshold 的检测，同 label 且 IoU 不低于 same_spot_iou_threshold 的检测做一对一贪心匹配
    - 未匹配的点位删除；点位连续 consecutive_frames 帧命中时触发一次 smoke_fire
    - _Requirements: 3.4, 3.11_
    - 设计参考：决策 D4、D5

  - [x] 4.6 编写烟火连续同位属性测试
    - **Property 5: 烟火连续同位判定**
    - **Validates: Requirements 3.4, 3.11**
    - 测试文件 tests/property/test_smoke_fire.py，Hypothesis 示例数不少于 100

  - [x] 4.7 实现安全帽关联 associate_helmets
    - 在 chimera_patrol/rules/helmet_assoc.py 中实现：头部区域与 helmet 框按 ioh 或 iou 计算重叠（由 assoc_overlap_metric 决定）
    - 过滤低于 helmet.conf_threshold 的 helmet；重叠不低于 assoc_overlap_threshold 的候选按重叠度降序做一对一贪心分配
    - _Requirements: 3.5, 3.6, 3.11_
    - 设计参考：决策 D2、D5

  - [x] 4.8 编写安全帽关联属性测试
    - **Property 7: 安全帽关联**
    - **Validates: Requirements 3.6**
    - 测试文件 tests/property/test_helmet_association.py，Hypothesis 示例数不少于 100

  - [x] 4.9 实现 HelmetTimer
    - 在 chimera_patrol/rules/helmet_timer.py 中以 track 为单位记录违规起点，帧中关联到安全帽则清零
    - 连续违规达到 no_helmet_duration_s 时触发一次 no_helmet 事件；track 消失超过 track_gap_reset_s 或离开作用域超过宽限时清零
    - 高度小于 min_person_height_px 的 person 不参与计时
    - _Requirements: 3.5, 3.6_
    - 设计参考：决策 D3

  - [x] 4.10 编写未佩戴安全帽计时属性测试
    - **Property 8: 未佩戴安全帽连续计时**
    - **Validates: Requirements 3.5**
    - 测试文件 tests/property/test_helmet_timer.py，Hypothesis 示例数不少于 100

  - [x] 4.11 实现 CooldownTable 与轨迹记录
    - 在 chimera_patrol/rules/cooldown.py 中以 (event_type, target_key) 为键做冷却去重，cooldown_s 默认 30 秒
    - 烟火点位以框 IoU 判定是否为同一目标；person 类事件以 track 为目标键
    - 维护每个 track 的轨迹，事件输出时按 trajectory_max_points 下采样并保留首尾点
    - _Requirements: 3.9, 3.10_
    - 设计参考：Rule_Engine 规则设计 · 冷却去重、轨迹记录

  - [x] 4.12 编写冷却去重属性测试
    - **Property 9: 冷却去重**
    - **Validates: Requirements 3.9**
    - 测试文件 tests/property/test_cooldown.py，Hypothesis 示例数不少于 100

  - [x] 4.13 组装 RuleEngine.process
    - 在 chimera_patrol/rules/engine.py 中实现 RuleEngine.process(frame_result)，依次调用闯入、滞留、烟火、安全帽计时与冷却，输出 CandidateEvent
    - CandidateEvent 包含事件类型、起止时间、轨迹、检测置信度、关键帧叠加项与检测来源
    - M1 的安全帽作用域为全画面 person，work_zone 作用域在任务 11.1 接入
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.9, 3.10_
    - 设计参考：内部核心类型 CandidateEvent

- [x] 5. 证据截取 Evidence_Capturer
  - [x] 5.1 实现 RingBuffer、select_frame 与 snapshot
    - 在 chimera_patrol/evidence/ring_buffer.py 中为每路保存最近 ring_buffer_s 秒、最多 ring_fps 帧每秒的 JPEG 与时间戳，超时淘汰
    - select_frame 选取时间戳最接近事件中点的帧；snapshot 返回范围内按时间升序的帧
    - _Requirements: 4.1, 4.2_
    - 设计参考：证据截取（Evidence_Capturer）

  - [ ]* 5.2 编写环形缓存选帧与快照属性测试
    - **Property 11: 环形缓存选帧与快照**
    - **Validates: Requirements 4.1, 4.2**
    - 测试文件 tests/property/test_ring_buffer.py，Hypothesis 示例数不少于 100

  - [x] 5.3 实现关键帧保存
    - 在 chimera_patrol/evidence/keyframe.py 中保存标注版（绘制检测框、label、conf 与相关 ROI 多边形）与原图版两份 JPEG
    - 路径格式为 {base_dir}/{camera_id}/{event_id}/keyframe_annotated.jpg 与 keyframe_raw.jpg
    - _Requirements: 4.1_
    - 设计参考：证据截取

  - [x] 5.4 实现短视频片段编码
    - 先按 requirements.txt 锁定版本的官方文档核对 imageio-ffmpeg 的写出 API 再写代码
    - 在 chimera_patrol/evidence/clip.py 中快照 [end_ms - clip_pre_s, end_ms + clip_post_s] 的帧，以 H.264 yuv420p 写出 clip.mp4
    - 成功则 clip_status 置为 ready；失败则记录错误日志并置为 failed，不抛出到事件流
    - _Requirements: 4.2, 4.3_
    - 设计参考：假设 A8、证据截取

  - [ ]* 5.5 编写片段失败不阻塞单元测试
    - 注入编码失败，验证记录了错误日志、clip_status 为 failed，且事件仍能完成后续流转
    - _Requirements: 4.3_

- [x] 6. EventPipeline 与 M1 装配
  - [x] 6.1 实现 EventPipeline
    - 在 chimera_patrol/pipeline.py 中实现 submit(candidate)，顺序执行：保存关键帧、落库（pending_review）、记录日志、调度片段保存
    - VLM 复核与推送通过可注入回调接入，M1 默认为空操作，M2 与 M3 再接线
    - _Requirements: 3.10, 4.1, 4.2, 12.5, 12.6_
    - 设计参考：事件数据流

  - [x] 6.2 装配 M1 流水线
    - 先按 requirements.txt 锁定版本的官方文档核对 Ultralytics 的模型加载 API 再写代码（装配时加载权重）
    - 在 chimera_patrol/runtime.py 中按配置启动 Video_Ingest、Detector、RuleEngine、EventPipeline，每路 Camera 一组工作线程
    - 工作线程通过 run_coroutine_threadsafe 投递候选事件，单路异常不影响其他路
    - _Requirements: 1.1, 2.4, 9.2, 12.4_
    - 设计参考：线程与异步模型

  - [ ]* 6.3 编写 M1 流水线集成测试
    - 使用合成视频与假检测器，验证产生候选事件、落库、生成关键帧与片段
    - 注入单路故障时其余路继续输出事件
    - _Requirements: 3.2, 4.1, 12.4_

- [x] 7. Checkpoint - M1 完成
  - Ensure all tests pass, ask the user if questions arise.

### M2（第 2 周）：VLM、复核与状态机、安全帽双路径、MCP

- [x] 8. VLM 适配层 VLM_Adapter
  - [x] 8.1 实现 VLMVerdict 与 normalize_vlm_response
    - 在 chimera_patrol/vlm/normalize.py 中去除代码围栏后解析 JSON
    - 校验 verdict 枚举、confidence 位于 [0,1]、description 去空白后截断到 description_max_chars，空字符串视为非法；suggestion 缺失时为空串
    - 不合规时抛出 VLMResponseError
    - _Requirements: 5.2_

  - [x] 8.2 编写 VLM 响应规范化属性测试
    - **Property 12: VLM 响应规范化**
    - **Validates: Requirements 5.2**
    - 测试文件 tests/property/test_vlm_normalize.py，Hypothesis 示例数不少于 100

  - [x] 8.3 定义 VLMAdapter Protocol 与 MockAdapter
    - 在 chimera_patrol/vlm/protocol.py 中定义 review_event、check_sop、check_helmet_crop 三个异步接口及请求、结果数据类
    - 在 chimera_patrol/vlm/mock.py 中实现 MockAdapter，按 vlm.mock.default_verdict 与 by_camera 确定性返回
    - _Requirements: 5.3_
    - 设计参考：VLM 复核层 · VLM_Adapter 接口

  - [x] 8.4 实现 OpenAICompatibleAdapter
    - 在 chimera_patrol/vlm/openai_compat.py 中基于 httpx.AsyncClient 调用 OpenAI 兼容接口，图片以 base64 data URL 传递，要求只返回 JSON
    - 密钥从 vlm.openai_compatible.api_key_env 指定的环境变量读取；日志只记录请求与响应摘要
    - 实现 review_event 与 check_helmet_crop；max_retries 默认为 0
    - _Requirements: 5.2, 5.3, 12.6_

  - [ ]* 8.5 编写适配器合约测试
    - Mock 与基于 httpx MockTransport 的 OpenAI 兼容实现跑同一套用例
    - 覆盖合法响应、非法响应与 HTTP 错误
    - _Requirements: 5.3_

- [x] 9. MCP_Server 登记
  - [x] 9.1 实现 FastMCP VLM 服务
    - 先按 requirements.txt 锁定版本的官方文档核对 FastMCP（mcp Python SDK）的工具注册与 stdio 运行 API 再写代码
    - 在 chimera_patrol/mcp/vlm_server.py 中以 stdio 暴露 review_event、check_sop、check_helmet_crop 三个工具，内部复用 build_adapter(config)
    - _Requirements: 5.4, 13.4_
    - 设计参考：MCP 登记

  - [x] 9.2 登记工作区 MCP 配置
    - 创建 .kiro/settings/mcp.json，登记 chimera-vlm 服务，包括 command、args、env（引用 VLM_API_KEY）与 autoApprove 列表
    - _Requirements: 5.4, 13.4_

  - [ ]* 9.3 编写 MCP 配置校验测试
    - 校验 mcp.json 为合法 JSON，包含 chimera-vlm 项，且工具名与 vlm_server 注册的工具一致
    - _Requirements: 5.4, 13.4_

- [x] 10. VLM 复核与事件状态机
  - [x] 10.1 实现 decide_status
    - 在 chimera_patrol/review/decision.py 中按设计决策：false_positive 且 confidence 不低于 vlm.auto_decision_threshold 为 dismissed；true_positive 且不低于阈值为 confirmed；其余为 needs_human
    - 决策函数不读取 detection_source
    - _Requirements: 3.8, 6.2, 6.3, 6.4_

  - [x] 10.2 编写复核决策属性测试
    - **Property 14: 复核结果到状态的决策**
    - **Validates: Requirements 3.8, 6.2, 6.3, 6.4**
    - 测试文件 tests/property/test_decision.py，Hypothesis 示例数不少于 100

  - [x] 10.3 实现 can_transition 与 Event_State_Machine
    - 在 chimera_patrol/review/state_machine.py 中实现迁移合法性规则：系统迁移只能从 pending_review 发起；人工目标限于 confirmed、dismissed、needs_human；pending_review 事件不可改判；同态迁移被拒绝
    - 每次迁移写入 event_status_log 并同步 events.status
    - 实现 override(event_id, to_status, operator, reason)，记录改判人、时间、原状态与新状态；非法迁移返回冲突错误
    - operator 为 1 至 32 字符（假设 A9）
    - _Requirements: 6.1, 6.6_
    - 设计参考：事件状态机

  - [x] 10.4 编写状态迁移合法性与审计属性测试
    - **Property 15: 状态迁移合法性与审计**
    - **Validates: Requirements 6.1, 6.6**
    - 测试文件 tests/property/test_state_machine.py，Hypothesis 示例数不少于 100

  - [x] 10.5 实现 VLM_Reviewer
    - 在 chimera_patrol/review/reviewer.py 中异步复核：Semaphore 限制 vlm.max_concurrency，asyncio.wait_for 以 vlm.timeout_s 超时
    - 成功则按 decide_status 迁移；超时、适配器异常或响应非法则迁移为 review_failed 并记录 vlm_error；关键帧缺失直接置为 review_failed（keyframe_missing）
    - vlm.review_enabled 为 false 时不调用适配器，由系统将事件迁移为 needs_human 并标记 review_skipped
    - _Requirements: 5.1, 5.5, 5.6, 5.7_
    - 设计参考：VLM 复核层、Error Handling

  - [x]* 10.6 编写复核失败与复核关闭属性测试
    - **Property 13: 复核失败与复核关闭**
    - **Validates: Requirements 5.5, 5.6**
    - 测试文件 tests/property/test_review_failure.py，Hypothesis 示例数不少于 100

  - [x] 10.7 实现服务重启恢复
    - 在 chimera_patrol/review/recovery.py 中，启动时将遗留的 pending_review 事件迁移为 review_failed，原因记为 service_restart，状态日志 actor 为 startup_recovery
    - _Requirements: 5.5, 12.4_
    - 设计参考：假设 A10

  - [x] 10.8 将复核与状态机接入 EventPipeline
    - 事件落库后：复核开关开启则入队 VLM_Reviewer，关闭则直接迁移为 needs_human
    - 复核结果经状态机写入后触发推送回调位
    - _Requirements: 5.1, 5.6, 6.1_

  - [x]* 10.9 编写慢 VLM 与故障隔离集成测试
    - 注入慢速 VLM，验证检测主链路不被阻塞
    - 注入单路故障与 VLM 异常，验证其他路与功能继续运行
    - _Requirements: 5.7, 12.4_

- [x] 11. 安全帽双路径集成
  - [x] 11.1 接入 work_zone 作用域与权重路径
    - 在 chimera_patrol/rules/helmet_scope.py 中实现：该 Camera 配置了 work_zone 时仅参考点落在 work_zone 内的 person 参与安全帽判定，否则全画面参与
    - 权重路径检测结果接入 HelmetTimer 与 RuleEngine
    - _Requirements: 3.5, 3.6_
    - 设计参考：决策 D1、D2

  - [x] 11.2 实现 VLM 兜底路径
    - 在 chimera_patrol/rules/helmet_fallback.py 中，对作用域内每个 track 按 interval_s 低频裁剪头部区域（按 crop_padding_ratio 外扩）并调用 check_helmet_crop
    - 返回未佩戴且 confidence 不低于 min_confidence 时产生 no_helmet 事件，detection_source 为 vlm_fallback
    - 兜底失败仅记录日志，不产生事件
    - 开关 rules.helmet.vlm_fallback.enabled 默认开启；vlm.review_enabled 为 false 时禁用兜底
    - _Requirements: 3.7, 3.8, 5.6_

  - [x]* 11.3 编写兜底路径示例测试
    - 验证 Mock 返回未佩戴时产生 vlm_fallback 事件
    - 验证开关关闭或复核关闭时不调用适配器
    - 验证两路事件进入同一复核流程
    - _Requirements: 3.7, 3.8_

- [x] 12. Checkpoint - M2 完成
  - Ensure all tests pass, ask the user if questions arise.

### M3（第 3 周）：API、统计、SOP、前端、演示与收尾

- [ ] 13. FastAPI 应用与 REST API
  - [x] 13.1 实现应用工厂与事件、摄像头接口
    - 在 chimera_patrol/api/app.py 中实现应用工厂
    - 实现 /api/health、/api/cameras、/api/events（过滤与分页）、/api/events/{id}（含 trajectory 与 status_log）、POST /api/events/{id}/status
    - 统一错误格式与错误码
    - _Requirements: 6.5, 6.6, 7.1, 12.5_
    - 设计参考：HTTP API、EventDTO、CameraDTO

  - [x] 13.2 实现 MJPEG 流端点
    - 在 chimera_patrol/api/stream.py 中读取 FrameSlot 最新帧，支持 overlay 与 fps 参数
    - JPEG 编码放入线程池，以 multipart/x-mixed-replace 输出
    - _Requirements: 7.2, 2.4_

  - [x] 13.3 实现证据静态端点
    - 在 chimera_patrol/api/media.py 中实现 /media/{path}，规范化路径并限制在 evidence.base_dir 内，越界返回 404
    - _Requirements: 7.5_

  - [x] 13.4 实现 WsHub 与 /ws/events
    - 在 chimera_patrol/api/ws.py 中实现 hello、heartbeat、ping/pong 与递增 seq
    - 广播 event.created、event.updated、camera.status；单个客户端断开不影响其他客户端
    - _Requirements: 7.3, 12.4_
    - 设计参考：WebSocket 协议

  - [x] 13.5 接入推送与状态广播
    - 将 WsHub 注入 EventPipeline 与状态机
    - 在事件创建、状态变化、VLM 结果写入、片段就绪或失败时推送 EventDTO（不含 trajectory）
    - _Requirements: 7.3, 6.6_

  - [x]* 13.6 编写 API 与 WebSocket 测试
    - 使用 TestClient 覆盖事件列表过滤、详情、改判返回 409 与 422、MJPEG 响应头、WebSocket 推送与 hello 消息
    - _Requirements: 6.5, 6.6, 7.2, 7.3_

- [ ] 14. 统计面板与导出
  - [x] 14.1 实现 compute_stats 纯函数
    - 在 chimera_patrol/api/stats.py 中按设计口径计算 totals、by_camera、by_type
    - include_sop 默认 false；total_alarms 为 0 时误报下降率为 null；pending_human 为 needs_human 与 review_failed 之和
    - _Requirements: 8.1, 8.2_
    - 设计参考：StatsDTO、假设 A6、A13、A14

  - [x]* 14.2 编写统计口径属性测试
    - **Property 17: 统计口径**
    - **Validates: Requirements 8.1, 8.2**
    - 测试文件 tests/property/test_stats.py，Hypothesis 示例数不少于 100

  - [x] 14.3 实现 SQL 聚合与 /api/stats
    - 在 chimera_patrol/store/stats_sql.py 中用 GROUP BY 聚合，并与纯函数采用同一过滤口径
    - 在 chimera_patrol/api/routes_stats.py 中提供 /api/stats，支持 camera_id、from_ms、to_ms、include_sop 参数
    - _Requirements: 8.1, 8.2_

  - [ ]* 14.4 编写 SQL 与纯函数口径一致性测试
    - 对同一随机事件集合比对 SQL 聚合结果与 compute_stats 结果
    - _Requirements: 8.1, 8.2_

  - [x] 14.5 实现统计导出
    - 在 chimera_patrol/api/export.py 中实现 JSON 与 CSV 导出，CSV 表头与 dimension 列按设计
    - 在 routes_stats.py 中提供 /api/stats/export，附带 Content-Disposition 响应头
    - _Requirements: 8.3_

  - [ ]* 14.6 编写统计导出往返属性测试
    - **Property 18: 统计导出往返**
    - **Validates: Requirements 8.3**
    - 测试文件 tests/property/test_export_roundtrip.py，Hypothesis 示例数不少于 100

- [ ] 15. SOP_Checker
  - [x] 15.1 实现 sops.yaml 加载与示例 SOP
    - 在 chimera_patrol/sop/loader.py 中加载并校验 SOP 定义：id 全局唯一、步骤 id 在 SOP 内唯一、每份至少一步
    - 在仓库根目录编写 sops.yaml，内含「入场作业准备」SOP 的 4 个步骤
    - _Requirements: 11.1, 11.2_

  - [ ]* 15.2 编写 SOP 定义往返属性测试
    - **Property 19: SOP 定义往返与校验**
    - **Validates: Requirements 11.1**
    - 测试文件 tests/property/test_sop_loader.py，Hypothesis 示例数不少于 100

  - [x] 15.3 实现 sample_times 采样时刻
    - 在 chimera_patrol/sop/sampling.py 中按假设 A2 计算 n 个采样时刻
    - endpoints_inclusive 为真时首尾取到窗口两端；否则取 n 个等分子区间的中点
    - _Requirements: 11.3_

  - [ ]* 15.4 编写 SOP 采样时刻属性测试
    - **Property 20: SOP 采样时刻**
    - **Validates: Requirements 11.3**
    - 测试文件 tests/property/test_sop_sampling.py，Hypothesis 示例数不少于 100

  - [x] 15.5 实现 normalize_sop_result 并接入 check_sop
    - 在 chimera_patrol/sop/normalize.py 中补全缺失步骤为 unknown、忽略多余步骤、校验 status 枚举与 evidence 非空，结构非法时抛出错误
    - approximate 取自步骤的 requires_sequence
    - 在 MockAdapter 与 OpenAICompatibleAdapter 中实现 check_sop（多帧与步骤清单提示词）
    - _Requirements: 11.4, 11.8_

  - [x]* 15.6 编写 SOP 结果归一化属性测试
    - **Property 21: SOP 结果归一化与近似标注**
    - **Validates: Requirements 11.4, 11.8**
    - 测试文件 tests/property/test_sop_normalize.py，Hypothesis 示例数不少于 100

  - [x] 15.7 扩展仓储与恢复逻辑支持 SOP
    - 在 chimera_patrol/store/sop_repo.py 中实现 sop_runs 与 sop_step_results 的读写，并保证同一 Camera 只有一个 running 记录
    - 在 review/recovery.py 中增加启动恢复：遗留的 running 置为 check_failed_retry，原因 service_restart
    - _Requirements: 11.9, 11.10, 12.5_

  - [x] 15.8 实现 SopChecker 运行流程
    - 在 chimera_patrol/sop/checker.py 中实现触发校验（SOP 适用性、Camera 在线、无进行中检查）
    - 窗口结束后按 sample_times 从 RingBuffer 取帧并保存采样帧，调用 check_sop 并归一化
    - 成功且存在 fail 时创建 sop_violation 事件，关键帧取首个失败步骤对应的采样帧
    - 失败时记为 check_failed_retry，不产生事件、不写入任何步骤的 fail；retry_of 仅接受 check_failed_retry 的 run
    - _Requirements: 11.3, 11.6, 11.7, 11.9, 11.10_
    - 设计参考：SOP_Checker

  - [x]* 15.9 编写 SOP 检查结局属性测试
    - **Property 22: SOP 检查结局**
    - **Validates: Requirements 11.6, 11.9, 11.10**
    - 测试文件 tests/property/test_sop_outcome.py，Hypothesis 示例数不少于 100

  - [x] 15.10 实现 SOP API 与推送
    - 在 chimera_patrol/api/routes_sop.py 中实现 /api/sops、POST /api/sop/runs（返回 202、409、422）、GET /api/sop/runs 与 GET /api/sop/runs/{id}
    - 检查状态变化时推送 sop.run.updated
    - _Requirements: 11.3, 11.9, 7.3_
    - 设计参考：HTTP API、SopRunDTO

- [ ] 16. Dashboard 前端（原生 HTML、CSS、JavaScript）
  - [x] 16.1 实现页面骨架与 API、WebSocket 客户端
    - 在 web/ 下创建 index.html（含视频墙、事件流、筛选、Stats_Panel、SOP 面板的容器）、css/app.css、js/api.js、js/ws.js
    - ws.js 实现指数退避重连，重连后重新拉取事件与摄像头列表进行对账
    - _Requirements: 7.7, 7.3_

  - [x] 16.2 实现视频墙
    - 在 web/js/wall.js 中为每路 Camera 渲染指向 MJPEG 流的 img，显示在线或离线状态与最近事件角标，ROI 只读叠加
    - _Requirements: 7.1, 7.2, 1.2_

  - [x] 16.3 实现事件流与筛选
    - 在 web/js/events.js 中实现新事件置顶、按 id 进行 upsert 更新，显示类型、时间及 CV、VLM 双层徽标
    - 支持按摄像头、类型、状态筛选，dismissed 事件保持可见
    - _Requirements: 7.4, 7.6, 6.5_

  - [x] 16.4 实现事件详情页与人工改判
    - 在 web/event.html 与 web/js/detail.js 中实现回放区（关键帧与片段）、CV 判定区、VLM 判定区三栏并排
    - 提供改判操作并提交到状态接口
    - _Requirements: 7.5, 6.6, 4.1, 4.2_

  - [x] 16.5 实现 Stats_Panel 与导出
    - 在 web/js/stats.js 中展示 CV 总报警数、dismissed、confirmed、待人工数与误报下降率（null 显示为“—”）
    - 支持按摄像头与类型分桶，提供 JSON 与 CSV 导出按钮
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 16.6 实现 SOP 检查面板
    - 在 web/js/sop.js 中逐步骤亮灯（pass 绿、fail 红、unknown 灰）并显示依据
    - approximate 为真的步骤加“近似判定”标注；check_failed_retry 时显示失败原因与“重新检查”按钮
    - _Requirements: 11.5, 11.7, 11.9_

  - [ ]* 16.7 编写前端冒烟测试
    - 校验 web/ 下没有构建链配置文件，页面引用的脚本与样式文件均存在
    - 测试文件 tests/smoke/test_frontend.py
    - _Requirements: 7.7_

- [ ] 17. 一键演示模式
  - [x] 17.1 编写样例视频生成脚本与演示配置
    - 在 scripts/generate_samples.py 中生成至少 2 段样例视频：一段含区域闯入等真实事件，另一段含光影晃动或飞虫类典型误报
    - 输出到 samples/videos/，并编写 config.demo.yaml 指向这些样例
    - _Requirements: 10.2_

  - [x] 17.2 实现单进程入口与 --demo 模式
    - 先按 requirements.txt 锁定版本的官方文档核对 Ultralytics 的模型加载 API 再写代码（演示模式会加载权重）
    - 在 chimera_patrol/__main__.py 中实现 python -m chimera_patrol 入口，支持 --demo 与 --config 参数
    - 在单进程内启动 uvicorn、后端流水线、前端静态文件与样例数据
    - _Requirements: 10.1, 12.1_

  - [ ]* 17.3 编写演示启动冒烟测试
    - 先按 requirements.txt 锁定版本的官方文档核对 Ultralytics 的模型加载 API 再写代码
    - 验证 --help 可执行；使用 TestClient 验证 --demo 启动后 /api/health 返回 ok
    - 测试文件 tests/smoke/test_demo_startup.py
    - _Requirements: 10.1, 12.1_

- [ ] 18. README 与依赖说明
  - [x] 18.1 编写 README
    - 写明依赖清单与安装方式（指向 requirements.txt）
    - 写明 Windows 与 Linux 下的一条命令启动方式
    - 写明演示路径：启动、触发事件、VLM 拦截误报、在 Stats_Panel 查看误报下降率
    - _Requirements: 10.3, 12.1, 12.2_

- [ ] 19. 端到端自动化验证
  - [x] 19.1 编写端到端集成测试
    - 先按 requirements.txt 锁定版本的官方文档核对 Ultralytics 的模型加载 API 再写代码（若测试加载真实权重）
    - 用样例视频驱动流水线，验证事件生成、Mock 复核、API 查询与统计计数一致
    - 验证单路故障不影响其他功能
    - _Requirements: 1.1, 3.2, 5.1, 6.2, 8.1, 12.4_

  - [ ]* 19.2 编写端到端延迟断言测试
    - 使用假检测器与合成的 2 路 720p 输入，断言端到端延迟不超过 2 秒
    - 真实硬件上的基准测量不属于本任务范围
    - _Requirements: 12.3_

  - [x] 19.3 编写交付物完整性冒烟测试
    - 校验 .kiro/specs/chimera-patrol-inspection 下 requirements.md、design.md、tasks.md 存在
    - 校验 .kiro/steering 至少 2 份、.kiro/hooks 至少 2 个、.kiro/settings/mcp.json 登记了 chimera-vlm
    - 校验 config.yaml、sops.yaml 与样例视频存在
    - 测试文件 tests/smoke/test_deliverables.py
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [x] 20. Final Checkpoint - 全部测试通过
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- 标记 `*` 的子任务为可选测试任务，可在快速 MVP 中跳过；核心实现子任务不可跳过。
- 每个任务都引用了 requirements.md 中的具体需求条款，便于追溯。
- 属性测试直接对应 design.md 的 Correctness Properties 编号，并使用 Hypothesis 实现。
- 检查点任务用于在每个里程碑结束时运行测试套件，并与用户确认是否有问题。
- 所有任务仅涉及代码、配置与测试的编写。应用的运行仅限于测试环境中的自动化验证。

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4", "1.8", "1.9", "1.10", "1.11", "4.1"] },
    { "id": 2, "tasks": ["1.3", "1.5", "2.1", "3.1", "4.2", "4.3", "4.5", "4.7", "4.9", "4.11", "5.1", "5.4", "8.1", "10.1", "14.1", "15.1", "15.3", "17.1"] },
    { "id": 3, "tasks": ["1.6", "1.7", "2.2", "2.3", "3.2", "4.4", "4.6", "4.8", "4.10", "4.12", "4.13", "5.2", "5.3", "5.5", "8.2", "8.3", "10.2", "10.3", "14.2", "14.3", "15.2", "15.4"] },
    { "id": 4, "tasks": ["2.4", "3.3", "6.1", "8.4", "10.4", "10.5", "10.7", "11.1", "13.1", "14.4", "14.5"] },
    { "id": 5, "tasks": ["3.4", "3.5", "6.2", "8.5", "9.1", "10.6", "10.8", "11.2", "13.2", "13.3", "13.4", "14.6", "15.5", "15.7"] },
    { "id": 6, "tasks": ["6.3", "9.2", "10.9", "11.3", "13.5", "15.6", "15.8", "16.1"] },
    { "id": 7, "tasks": ["9.3", "13.6", "15.9", "15.10", "16.2", "16.3", "16.4", "16.5", "17.2", "19.3"] },
    { "id": 8, "tasks": ["16.6", "17.3", "18.1", "19.1", "19.2"] },
    { "id": 9, "tasks": ["16.7"] }
  ]
}
```
