// 事件详情页（需求 7.5、6.6、4.1、4.2）：回放区、CV 判定区、VLM 判定区三栏并排，并提供人工改判。
import { api, ApiError } from "/static/js/api.js";
import {
  EVENT_TYPE_LABELS,
  STATUS_LABELS,
  VERDICT_LABELS,
  VLM_ERROR_LABELS,
  el,
  formatTime,
  vlmBadge,
} from "/static/js/labels.js";

const OVERRIDE_TARGETS = ["confirmed", "dismissed", "needs_human"];

function section(title, ...children) {
  return el("section", { class: "detail-col" }, el("h2", {}, title), ...children);
}

function kv(label, value) {
  return el("div", { class: "kv" }, el("span", { class: "kv-label" }, label), el("span", { class: "kv-value" }, value ?? "—"));
}

function replayColumn(event) {
  const ev = event.evidence;
  const children = [];
  if (ev.keyframe_annotated_url) {
    children.push(el("figure", {}, el("img", { src: ev.keyframe_annotated_url, alt: "标注关键帧" }), el("figcaption", {}, "标注关键帧")));
  }
  if (ev.keyframe_raw_url) {
    children.push(el("figure", {}, el("img", { src: ev.keyframe_raw_url, alt: "原始关键帧" }), el("figcaption", {}, "原始关键帧")));
  }
  if (ev.clip_url) {
    children.push(el("video", { src: ev.clip_url, controls: "", preload: "metadata" }));
  } else {
    const text = ev.clip_status === "failed" ? "片段保存失败" : "片段尚未就绪";
    children.push(el("p", { class: "muted" }, `${text}（${ev.clip_status}）`));
  }
  return section("回放区", ...children);
}

function cvColumn(event) {
  const cv = event.cv;
  const detail = cv.detail || {};
  const trajectory = cv.trajectory || [];
  return section(
    "CV 判定区",
    kv("类型", EVENT_TYPE_LABELS[event.event_type] || event.event_type),
    kv("检测来源", event.detection_source),
    kv("置信度", cv.confidence === null || cv.confidence === undefined ? "—" : cv.confidence.toFixed(3)),
    kv("跟踪 ID", cv.track_id),
    kv("区域", cv.roi_id ? `${cv.roi_id}（${cv.roi_type}）` : "—"),
    kv("开始", formatTime(event.start_ms)),
    kv("结束", formatTime(event.end_ms)),
    kv("轨迹点数", trajectory.length),
    el("pre", { class: "json" }, JSON.stringify(detail, null, 2)),
    event.sop
      ? el(
          "div",
          {},
          el("h3", {}, "不合规步骤"),
          el(
            "ul",
            {},
            ...event.sop.failed_steps.map((s) => el("li", {}, `${s.description}：${s.evidence}`))
          )
        )
      : null
  );
}

function vlmColumn(event) {
  const vlm = event.vlm;
  if (!vlm) {
    return section("VLM 判定区", el("p", { class: "muted" }, "尚未复核"));
  }
  if (vlm.skipped) {
    return section("VLM 判定区", el("p", { class: "muted" }, "复核已关闭，事件直接进入人工队列"));
  }
  if (vlm.error) {
    const reason = VLM_ERROR_LABELS[vlm.error] || vlm.error;
    return section("VLM 判定区", el("p", { class: "bad" }, `复核失败：${reason}`), el("p", { class: "muted" }, vlm.error));
  }
  return section(
    "VLM 判定区",
    kv("结论", VERDICT_LABELS[vlm.verdict] || vlm.verdict),
    kv("置信度", vlm.confidence === null ? "—" : vlm.confidence.toFixed(3)),
    kv("复核时间", formatTime(vlm.reviewed_ms)),
    el("p", {}, el("strong", {}, "描述："), vlm.description || "—"),
    el("p", {}, el("strong", {}, "建议："), vlm.suggestion || "—"),
    el("span", { class: `badge ${vlmBadge(vlm).kind}` }, vlmBadge(vlm).text)
  );
}

function statusColumn(event) {
  const rows = (event.status_log || []).map((entry) =>
    el(
      "tr",
      {},
      el("td", {}, formatTime(entry.created_ms)),
      el("td", {}, entry.from_status ? STATUS_LABELS[entry.from_status] || entry.from_status : "（创建）"),
      el("td", {}, STATUS_LABELS[entry.to_status] || entry.to_status),
      el("td", {}, `${entry.actor_type} / ${entry.actor}`),
      el("td", {}, entry.reason || "")
    )
  );
  return section(
    "状态与改判",
    kv("当前状态", STATUS_LABELS[event.status] || event.status),
    el(
      "table",
      { class: "log" },
      el("thead", {}, el("tr", {}, el("th", {}, "时间"), el("th", {}, "原状态"), el("th", {}, "新状态"), el("th", {}, "操作者"), el("th", {}, "原因"))),
      el("tbody", {}, ...rows)
    ),
    overrideForm(event)
  );
}

function overrideForm(event) {
  const message = el("p", { class: "error", hidden: "" });
  const operator = el("input", { type: "text", maxlength: "32", value: "巡检员", "aria-label": "改判人" });
  const reason = el("input", { type: "text", maxlength: "200", placeholder: "原因（可选）", "aria-label": "改判原因" });
  const target = el(
    "select",
    { "aria-label": "目标状态" },
    ...OVERRIDE_TARGETS.map((value) => el("option", { value }, STATUS_LABELS[value]))
  );
  const submit = el("button", { type: "button" }, "提交改判");

  if (event.status === "pending_review") {
    submit.disabled = true;
    return el("div", { class: "override" }, el("p", { class: "muted" }, "复核在途，暂不可改判"));
  }

  submit.addEventListener("click", async () => {
    message.hidden = true;
    try {
      const updated = await api.changeStatus(event.id, {
        to_status: target.value,
        operator: operator.value.trim(),
        reason: reason.value.trim() || null,
      });
      render(updated);
    } catch (err) {
      message.textContent = err instanceof ApiError ? `${err.message}（${err.code}）` : String(err);
      message.hidden = false;
    }
  });

  return el(
    "div",
    { class: "override" },
    el("h3", {}, "人工改判"),
    el("label", {}, "目标 ", target),
    el("label", {}, "改判人 ", operator),
    el("label", {}, "原因 ", reason),
    submit,
    message
  );
}

export function render(event) {
  document.querySelector("#event-id").textContent = event.id;
  document.querySelector("#detail").replaceChildren(
    replayColumn(event),
    cvColumn(event),
    vlmColumn(event),
    statusColumn(event)
  );
}

async function main() {
  const id = new URLSearchParams(location.search).get("id");
  if (!id) {
    document.querySelector("#detail").replaceChildren(el("p", { class: "error" }, "缺少事件 ID 参数 id"));
    return;
  }
  try {
    const event = await api.event(id);
    render(event);
  } catch (err) {
    const message = err instanceof ApiError ? `${err.message}（${err.code}）` : String(err);
    document.querySelector("#detail").replaceChildren(el("p", { class: "error" }, message));
  }
}

main();
