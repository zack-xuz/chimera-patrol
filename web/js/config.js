// 只读配置区：展示当前生效的关键配置（GET /api/config/effective）。
// 纯展示，没有保存按钮，不能修改。修改 config.yaml 后需重启服务生效。
import { api } from "/static/js/api.js";
import { el } from "/static/js/labels.js";

function kv(label, value) {
  const text = value === null || value === undefined || value === "" ? "—" : String(value);
  return el("div", { class: "kv" }, el("span", { class: "kv-label" }, label), el("span", { class: "kv-value" }, text));
}

function cameraRows(cameras) {
  if (cameras.length === 0) return [el("p", { class: "muted" }, "尚未接入任何视频源")];
  return [
    el(
      "table",
      { class: "cfg-table" },
      el(
        "thead",
        {},
        el("tr", {}, ...["摄像头", "检测", "人员置信度", "烟火置信度", "安全帽置信度", "冷却（秒）", "ROI"].map((h) => el("th", {}, h)))
      ),
      el(
        "tbody",
        {},
        ...cameras.map((cam) =>
          el(
            "tr",
            {},
            el("td", {}, cam.name),
            el("td", {}, cam.paused ? "已暂停" : "运行中"),
            el("td", {}, cam.thresholds.person_conf),
            el("td", {}, cam.thresholds.smoke_fire_conf),
            el("td", {}, cam.thresholds.helmet_conf),
            el("td", {}, cam.cooldown_s),
            el("td", {}, cam.rois.length === 0 ? "无" : cam.rois.map((r) => `${r.name}（${r.type === "work_zone" ? "作业区" : "闯入区"}）`).join("、"))
          )
        )
      )
    ),
  ];
}

export async function initConfigPanel() {
  const panel = document.querySelector("#config-panel");
  if (!panel) return;
  const body = panel.querySelector("#config-body");
  try {
    const cfg = await api.effectiveConfig();
    const rules = cfg.rules;
    const vlm = cfg.vlm;
    body.replaceChildren(
      el("h3", {}, "摄像头"),
      ...cameraRows(cfg.cameras),
      el("h3", {}, "规则（全局）"),
      el(
        "div",
        { class: "cfg-grid" },
        kv("人员参考点", rules.reference_point === "bottom_center" ? "脚点（底边中点）" : "框中心"),
        kv("ROI 容忍离开（秒）", rules.roi_exit_grace_s),
        kv("滞留阈值（秒）", rules.dwell_threshold_s),
        kv("冷却时间（秒）", rules.cooldown_s),
        kv("烟火连续帧数", rules.smoke_fire.consecutive_frames),
        kv("烟火同位 IoU", rules.smoke_fire.same_spot_iou_threshold),
        kv("未戴帽持续（秒）", rules.helmet.no_helmet_duration_s),
        kv("VLM 兜底", rules.helmet.vlm_fallback.enabled ? "开启" : "关闭")
      ),
      el("h3", {}, "检测器"),
      el(
        "div",
        { class: "cfg-grid" },
        kv("每 N 帧检测 1 帧（N）", cfg.detector.frame_skip_n),
        kv("推理尺寸", cfg.detector.imgsz),
        kv("设备", cfg.detector.device)
      ),
      el("h3", {}, "VLM 复核"),
      el(
        "div",
        { class: "cfg-grid" },
        kv("复核开关", vlm.review_enabled ? "开启" : "关闭"),
        kv("提供方", vlm.provider),
        kv("模型", vlm.model),
        kv("接口地址", vlm.base_url),
        kv("超时（秒）", vlm.timeout_s),
        kv("最大并发", vlm.max_concurrency),
        kv("API Key", vlm.api_key.configured ? `已设置（环境变量 ${vlm.api_key.env_var}）` : `未设置（环境变量 ${vlm.api_key.env_var}）`)
      ),
      el("p", { class: "muted" }, "只读展示。修改 config.yaml 后需重启服务生效。")
    );
  } catch (err) {
    body.replaceChildren(el("p", { class: "error" }, `配置加载失败：${err.message}`));
  }
}
