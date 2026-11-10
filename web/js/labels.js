// 共用的中文标签与格式化工具。取值与 event-contract.md 一致。

export const EVENT_TYPE_LABELS = {
  intrusion: "区域闯入",
  loitering: "人员滞留",
  smoke_fire: "烟火",
  no_helmet: "未戴安全帽",
  sop_violation: "SOP 不合规",
};

export const STATUS_LABELS = {
  pending_review: "复核中",
  confirmed: "已确认",
  dismissed: "已拦截（误报）",
  needs_human: "待人工",
  review_failed: "复核失败",
};

export const SOP_RUN_STATUS_LABELS = {
  running: "检查中",
  completed: "已完成",
  check_failed_retry: "检查失败待重试",
};

export const FAILURE_REASON_LABELS = {
  vlm_timeout: "VLM 超时",
  vlm_error: "VLM 调用错误",
  response_invalid: "VLM 响应非法",
  no_frames: "窗口内无可用帧",
  service_restart: "服务重启中断",
};

export const VLM_ERROR_LABELS = {
  timeout: "复核超时",
  response_invalid: "复核响应非法",
  service_restart: "服务重启中断",
  keyframe_missing: "关键帧缺失",
};

export const VERDICT_LABELS = {
  true_positive: "真实事件",
  false_positive: "误报",
  uncertain: "无法判定",
};

export const STEP_STATUS_LABELS = {
  pass: "通过",
  fail: "不通过",
  unknown: "未知",
};

export function formatTime(ms) {
  if (ms === null || ms === undefined) return "—";
  const d = new Date(ms);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export function formatRate(value) {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

// 返回 VLM 视图徽标文字（event-contract 4.3）。
export function vlmBadge(vlm) {
  if (!vlm) return { text: "未复核", kind: "muted" };
  if (vlm.skipped) return { text: "复核关闭", kind: "muted" };
  if (vlm.error) return { text: `复核失败：${VLM_ERROR_LABELS[vlm.error] || vlm.error}`, kind: "bad" };
  const label = VERDICT_LABELS[vlm.verdict] || vlm.verdict;
  // 误报为绿（拦截），真实事件为红（与已确认一致），不确定为黄（待人工）。
  const kind = vlm.verdict === "false_positive" ? "ok" : vlm.verdict === "true_positive" ? "confirm" : "warn";
  return { text: `${label}`, kind };
}

export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
