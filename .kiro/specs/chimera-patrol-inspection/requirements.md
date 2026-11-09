# Requirements Document

## Introduction

Chimera Patrol 是面向工业/能源场站的多路视频智能巡检工具。传统 CV 管线负责实时检测与规则报警，VLM（视觉语言模型）对每条报警做二次研判（真/误报判定 + 中文事件描述），巡检台把两层判定融合展示，目标是降低误报并使事件可读、可追溯。

本期（MVP）范围：
- 2–4 路视频（本地视频文件循环播放模拟多路，预留 RTSP 配置项）
- 4 类事件：区域闯入、人员滞留、烟火、未佩戴安全帽
- VLM 复核、事件融合与状态机、巡检台 UI、统计面板
- 1 个示例 SOP 的合规检查
- 内置样例视频与一键演示模式

技术栈：Python 3.11+、FastAPI、SQLite；前端为原生 HTML/CSS/JS（视频走 MJPEG 流，事件走 WebSocket 推送），不引入前端构建链，不做 SPA；优先单进程一键启动。

本期不做（非目标）：
- 生产级并发与高可用（不做 200 路压测）
- 用户权限与多租户、告警短信与电话外呼
- 模型训练与微调（检测使用现成预训练权重，运行时 VLM 使用外部 API）
- 移动端适配
- 复杂 SOP 场景库（时序型、持续状态型、空间关系型、读数/记录型）仅登记为 Phase 2 路线图，本期不实现

## Glossary

- **Patrol_System**: Chimera Patrol 整体系统，包含后端服务与前端巡检台
- **Video_Ingest**: 视频接入模块，负责为每个视频源创建虚拟摄像头并读取帧
- **Camera**: 虚拟摄像头，对应一个视频文件源或一个 RTSP 源
- **Detector**: CV 检测与跟踪模块，使用预训练 YOLO 权重做目标检测并做多目标跟踪
- **Rule_Engine**: 规则判定模块，基于轨迹与 ROI 生成结构化事件
- **ROI**: 感兴趣区域，在画面上配置的多边形入侵区域
- **Event**: 结构化事件，类型为区域闯入、人员滞留、烟火、未佩戴安全帽或 SOP 不合规
- **Evidence_Capturer**: 证据截取模块，负责保存关键帧与短视频片段
- **Keyframe**: 事件中间时刻的帧图像，保存带检测框标注版与原图版两份
- **VLM_Adapter**: VLM 统一适配层，封装对外部 VLM API 的调用，运行时实现可替换
- **VLM_Reviewer**: VLM 复核模块，通过 VLM_Adapter 对事件做二次研判
- **VLM_Verdict**: VLM 返回的结构化结果，包含 verdict、confidence、description、suggestion
- **Event_State_Machine**: 事件状态机，状态为 pending_review、confirmed、dismissed、needs_human、review_failed
- **Dashboard**: 巡检台前端界面
- **Stats_Panel**: Dashboard 中的统计面板
- **Config_File**: 集中配置文件 `config.yaml`
- **SOP**: 标准作业程序，由名称、适用摄像头与步骤列表组成
- **SOP_File**: SOP 定义文件 `sops.yaml`
- **SOP_Checker**: SOP 合规检查模块
- **检查失败待重试**: SOP 检查的整体状态，表示因 VLM 调用超时或失败而未得到判定结果，可由 Inspector 重新触发；该状态不等同于任何步骤的 fail
- **Inspector**: 使用 Dashboard 的巡检员或安全管理员
- **False_Positive_Reduction_Rate**: 误报下降率，等于 dismissed 事件数除以 CV 总报警数
- **Demo_Mode**: 一键演示模式，使用内置样例视频启动全部组件
- **MCP_Server**: 以 Model Context Protocol 形态登记的 VLM_Adapter

## Requirements

### Requirement 1: 多路视频接入

**User Story:** As a 巡检系统运营者, I want 同时接入 2–4 路视频源, so that 可以模拟真实多摄像头场景。

#### Acceptance Criteria

1. WHEN 服务启动且指定了视频目录, THE Video_Ingest SHALL 为目录中每个视频文件创建一路 Camera 并循环播放
2. IF 某路 Camera 读取视频失败, THEN THE Dashboard SHALL 将该路标记为离线，且其他 Camera 的处理保持不变
3. THE Config_File SHALL 提供 RTSP URL 配置项，填入后该 Camera 使用 RTSP 源替代文件源
4. THE Video_Ingest SHALL 支持同时接入 2 至 4 路 Camera

### Requirement 2: CV 检测与跟踪

**User Story:** As a CV 管线, I want 对每路视频逐帧检测并跨帧跟踪目标, so that 规则判定拥有稳定的目标轨迹。

#### Acceptance Criteria

1. THE Detector SHALL 使用预训练 YOLO 模型检测 person 以及 smoke/fire 相关类别，类别范围以所选模型实际支持为准
2. THE Detector SHALL 对 person 目标做多目标跟踪并分配稳定的 track ID
3. THE Detector SHALL 支持通过 Config_File 配置抽帧间隔，表示为「每 N 帧检测 1 帧」，N 以帧数为单位，默认 N=3，N 的取值范围为 1 至 30 的整数且可配置
4. WHILE 检测帧率低于实时, THE Detector SHALL 按配置的抽帧间隔处理，且 Dashboard 的视频显示保持不被阻塞

### Requirement 3: 规则判定与事件生成

**User Story:** As a 巡检系统运营者, I want 基于轨迹与区域规则生成结构化事件, so that 后续复核与展示有统一的数据基础。

#### Acceptance Criteria

1. THE Rule_Engine SHALL 支持在画面上为每个 Camera 配置 1 至 10 个多边形 ROI，每个多边形包含 3 至 20 个顶点，上述数量限制为 Config_File 中可配置的默认值
2. WHEN person 的轨迹进入 ROI, THE Rule_Engine SHALL 生成「区域闯入」事件
3. WHEN person 在 ROI 内的停留时长超过可配置阈值（默认 10 秒）, THE Rule_Engine SHALL 生成「人员滞留」事件
4. WHEN Detector 在连续 3 个已处理帧中均检测到置信度不低于烟火置信度阈值的烟火目标, THE Rule_Engine SHALL 生成「烟火」事件，连续帧数 3 为 Config_File 中可配置的默认值
5. WHEN person 进入 ROI 后连续 5 秒未检测到与该 person 关联的安全帽, THE Rule_Engine SHALL 生成「未佩戴安全帽」事件，持续时长 5 秒为 Config_File 中可配置的默认值
6. THE Rule_Engine SHALL 优先使用公开 helmet 预训练 YOLO 权重检测 helmet 与 no-helmet 以产出「未佩戴安全帽」事件，且仅采纳置信度不低于未佩戴安全帽置信度阈值的检测结果
7. WHERE Config_File 中的兜底路径开关处于开启状态（默认开启）, THE Rule_Engine SHALL 对 person 头部区域裁剪图以低频抽帧方式送 VLM_Reviewer 判定，并据此产出「未佩戴安全帽」事件
8. THE Event_State_Machine SHALL 对权重路径与兜底路径产出的「未佩戴安全帽」事件使用同一复核与状态流转流程
9. WHEN 同一目标的同一类事件在冷却时间内再次触发, THE Rule_Engine SHALL 丢弃重复事件，冷却时间默认 30 秒
10. THE Rule_Engine SHALL 在每个事件中记录事件 ID、摄像头、类型、起止时间、轨迹、检测置信度、关键帧路径、短片段路径
11. THE Config_File SHALL 提供烟火检测置信度阈值与未佩戴安全帽检测置信度阈值两个配置项，取值范围均为 0 至 1，默认值均为 0.5

### Requirement 4: 证据截取

**User Story:** As a 复核层, I want 在报警时自动截取关键帧与短片段, so that VLM 研判与人工回看有可用证据。

#### Acceptance Criteria

1. WHEN 事件生成, THE Evidence_Capturer SHALL 保存事件中间时刻的 Keyframe，包含带检测框标注版与原图版各一份
2. WHEN 事件生成, THE Evidence_Capturer SHALL 保存事件前后共约 6 秒的短视频片段
3. IF 短视频片段保存失败, THEN THE Evidence_Capturer SHALL 记录错误日志，且事件继续进入后续流转

### Requirement 5: VLM 复核层

**User Story:** As a 巡检员, I want 每条报警被 VLM 二次研判, so that 可以过滤误报并获得一句话的事件描述。

#### Acceptance Criteria

1. WHEN 新事件生成且 VLM 复核开关为开启, THE VLM_Reviewer SHALL 将 Keyframe 与事件上下文（类型、区域、置信度）通过 VLM_Adapter 送 VLM 研判
2. THE VLM_Adapter SHALL 返回结构化 JSON，包含 verdict（true_positive、false_positive 或 uncertain）、confidence（0 至 1）、description（一句话中文事件描述，长度 1 至 50 个字符，长度上限为 Config_File 中可配置的默认值）、suggestion（处置建议）
3. THE VLM_Adapter SHALL 通过统一接口调用外部 VLM API，且更换 VLM 提供方时只需替换适配实现
4. THE Patrol_System SHALL 将 VLM_Adapter 以 MCP_Server 形态登记在工作区 MCP 配置中
5. IF VLM 调用超过 Config_File 中配置的 VLM 超时（默认 20 秒）或失败, THEN THE Event_State_Machine SHALL 将事件置为 review_failed 并放入人工队列，事件记录保持完整
6. THE Config_File SHALL 提供 VLM 复核开关，WHEN 该开关为关闭, THE VLM_Reviewer SHALL 跳过 VLM 复核并使事件直接进入 needs_human 状态，用于对比演示有无复核的误报量
7. THE VLM_Reviewer SHALL 以异步方式执行复核，检测主链路保持不被阻塞

### Requirement 6: 事件融合与状态机

**User Story:** As a 巡检员, I want 两层判定融合成可追溯的事件状态, so that 我能清楚了解每条事件的处理结果。

#### Acceptance Criteria

1. THE Event_State_Machine SHALL 支持状态 pending_review、confirmed、dismissed、needs_human、review_failed，新事件的初始状态为 pending_review
2. WHEN VLM_Verdict 的 verdict 为 false_positive 且 confidence 不低于 0.7, THE Event_State_Machine SHALL 将事件置为 dismissed
3. WHEN VLM_Verdict 的 verdict 为 true_positive 且 confidence 不低于 0.7, THE Event_State_Machine SHALL 将事件置为 confirmed
4. WHEN VLM_Verdict 的 verdict 为 uncertain，或 verdict 为 true_positive 或 false_positive 且 confidence 低于 0.7, THE Event_State_Machine SHALL 将事件置为 needs_human
5. WHILE 事件处于 dismissed 状态, THE Dashboard SHALL 保持该事件在事件列表中可查看
6. WHEN Inspector 在事件详情页手动改判, THE Event_State_Machine SHALL 更新事件状态并记录改判人、时间、原状态与新状态

### Requirement 7: 巡检台 UI

**User Story:** As a 巡检员, I want 在一个界面同时查看多路画面、事件流与事件详情, so that 我能高效完成巡检。

#### Acceptance Criteria

1. THE Dashboard SHALL 在首页展示多路视频墙，每路包含实时画面、该路状态与最近事件角标
2. THE Dashboard SHALL 通过 MJPEG 流展示实时画面
3. WHEN 新事件生成或事件状态变化, THE Patrol_System SHALL 通过 WebSocket 向 Dashboard 推送该事件
4. THE Dashboard SHALL 在首页展示实时事件流，新事件自动置于顶部，每条包含类型、时间与 CV、VLM 两层判定徽标
5. THE Dashboard SHALL 在事件详情页并排展示三个区域：回放区（关键帧与片段回放）、CV 判定区（类型、置信度、轨迹信息）、VLM 判定区（结论、置信度、描述、建议）
6. THE Dashboard SHALL 支持按摄像头、类型、状态筛选事件
7. THE Dashboard SHALL 使用原生 HTML、CSS、JavaScript 实现，不依赖前端构建链

### Requirement 8: 统计面板

**User Story:** As a 项目演示者, I want 用数字证明 VLM 复核的价值, so that 演示与文案有可引用的数据。

#### Acceptance Criteria

1. THE Stats_Panel SHALL 展示 CV 总报警数、VLM 拦截（dismissed）数、确认事件数、待人工数与 False_Positive_Reduction_Rate
2. THE Stats_Panel SHALL 支持按摄像头和事件类型分桶统计
3. WHEN Inspector 触发导出, THE Patrol_System SHALL 将统计数据导出为 JSON 或 CSV 文件

### Requirement 9: 配置管理

**User Story:** As a 开发者, I want 所有关键参数集中在单个配置文件, so that 调整行为时无需修改代码。

#### Acceptance Criteria

1. THE Config_File SHALL 集中包含视频源、ROI（含每 Camera 的 ROI 数量上限与每个 ROI 的顶点数范围）、滞留阈值、冷却时间、抽帧间隔（单位为帧，「每 N 帧检测 1 帧」，默认 N=3）、VLM 开关与 VLM 超时（默认 20 秒）、烟火与未佩戴安全帽检测置信度阈值（默认 0.5）、烟火事件连续帧数（默认 3）、未佩戴安全帽持续时长（默认 5 秒）、未佩戴安全帽 VLM 兜底路径开关（默认开启）、SOP 检查采样帧数（默认 8）与采样时间窗口（默认触发时刻前后各 5 秒）、VLM description 长度上限（默认 50 个字符）配置
2. WHEN Config_File 被修改后服务重启, THE Patrol_System SHALL 应用修改后的配置
3. THE Patrol_System SHALL 在仓库中提供带注释的示例配置文件

### Requirement 10: 一键演示模式

**User Story:** As a 项目演示者, I want 一条命令启动全部组件并带样例数据, so that 可以稳定复现 3 分钟 demo。

#### Acceptance Criteria

1. THE Patrol_System SHALL 提供一条命令，在单进程内启动后端、前端与样例数据
2. THE Patrol_System SHALL 在仓库中内置至少 2 段样例视频，其中一段含真实事件，另一段含典型误报（光影晃动、飞虫或雨雪纹理）
3. THE Patrol_System SHALL 在 README 中写明演示路径：启动、触发事件、VLM 拦截误报、在 Stats_Panel 查看误报下降率

### Requirement 11: SOP 合规检查

**User Story:** As a 安全管理员, I want 系统对照 SOP 步骤清单检查作业人员是否合规, so that 流程是否正确也纳入巡检。

#### Acceptance Criteria

1. THE SOP_Checker SHALL 从 SOP_File 加载 SOP 定义，每份 SOP 包含名称、适用摄像头与步骤列表，每个步骤包含自然语言描述与判定要点
2. THE Patrol_System SHALL 内置 1 个示例 SOP「入场作业准备」，包含佩戴安全帽、佩戴手套、穿反光背心、确认隔离栏就位共 4 个步骤
3. WHEN Inspector 对选定 Camera 手动触发 SOP 检查, THE SOP_Checker SHALL 在触发时刻前 5 秒至后 5 秒的时间窗口内均匀抽取 8 帧并送 VLM 对照步骤逐项判定，采样帧数与时间窗口均为 Config_File 中可配置的默认值
4. THE SOP_Checker SHALL 为每个步骤返回结构化结果，包含 status（pass、fail 或 unknown）与 evidence（一句话判定依据）
5. THE Dashboard SHALL 提供 SOP 检查面板，逐项以亮灯形式展示每个步骤的判定结果与依据
6. WHEN 任一步骤的 status 为 fail, THE SOP_Checker SHALL 生成一条「SOP 不合规」事件，包含失败步骤与 Keyframe，并进入事件流
7. THE Patrol_System SHALL 对「SOP 不合规」事件复用事件详情与统计功能
8. WHERE 步骤需要连续动作才能判定, THE SOP_Checker SHALL 以多帧采样近似判定，并在结果中标注该步骤为近似判定
9. IF SOP 检查中的 VLM 调用超过 Config_File 中配置的 VLM 超时或失败, THEN THE SOP_Checker SHALL 将整次检查标记为「检查失败待重试」，向 Inspector 展示失败原因，并允许 Inspector 重新触发该次检查
10. WHILE 一次 SOP 检查处于「检查失败待重试」状态, THE SOP_Checker SHALL 不生成「SOP 不合规」事件，且不将该失败记为任何步骤的 fail

### Requirement 12: 非功能需求

**User Story:** As a 开发者, I want 系统可运行、可靠、可观测, so that 演示稳定且问题可排查。

#### Acceptance Criteria

1. THE Patrol_System SHALL 在 Windows 和 Linux 的 Python 3.11 及以上环境中通过一条命令启动
2. THE Patrol_System SHALL 在 README 或依赖清单中写明全部依赖
3. WHEN 处理 2 路 720p 视频并按抽帧配置检测, THE Patrol_System SHALL 使端到端延迟不超过 2 秒
4. IF 单路 Camera 或单次 VLM 调用失败（含超过 Config_File 中配置的 VLM 超时）, THEN THE Patrol_System SHALL 继续运行其余 Camera 与功能
5. THE Patrol_System SHALL 将事件数据持久化到 SQLite，并支持回放历史事件
6. THE Patrol_System SHALL 对事件生成、VLM 请求与响应摘要、状态变更记录日志

### Requirement 13: Kiro 使用要求

**User Story:** As a 参赛者, I want 仓库满足 Kiro 评审的硬性要求, so that 项目具备参赛资格。

#### Acceptance Criteria

1. THE Patrol_System 仓库 SHALL 在 `.kiro/specs/` 中保留 requirements.md、design.md、tasks.md
2. THE Patrol_System 仓库 SHALL 在 `.kiro/steering/` 中提供至少 2 份 steering 文档，分别覆盖项目结构与代码约定、事件数据契约
3. THE Patrol_System 仓库 SHALL 在 `.kiro/hooks/` 中提供至少 2 个 agent hook
4. THE Patrol_System 仓库 SHALL 在工作区 MCP 配置中登记 VLM_Adapter 对应的 MCP_Server
5. THE Patrol_System 仓库 SHALL 在仓库根目录保留 `.kiro` 目录
