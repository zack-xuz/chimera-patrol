// Stats_Panel：总览卡片、按摄像头与类型分桶、JSON/CSV 导出（需求 8.1、8.2、8.3）。
// 误报下降率为 null（总报警数为 0）时显示为“—”（假设 A14）。
import { api } from "/static/js/api.js";
import { EVENT_TYPE_LABELS, el, formatRate } from "/static/js/labels.js";

function card(label, value, hint) {
  return el(
    "div",
    { class: "stat-card" },
    el("span", { class: "stat-label" }, label),
    el("strong", { class: "stat-value" }, value),
    hint ? el("span", { class: "stat-hint" }, hint) : null
  );
}

function bucketTable(title, rows, keyLabel, keyText) {
  const head = el(
    "tr",
    {},
    el("th", {}, keyLabel),
    el("th", {}, "CV 总报警"),
    el("th", {}, "已拦截"),
    el("th", {}, "已确认"),
    el("th", {}, "待人工"),
    el("th", {}, "误报下降率")
  );
  const body = rows.map((row) =>
    el(
      "tr",
      {},
      el("td", {}, keyText(row)),
      el("td", {}, String(row.total_alarms)),
      el("td", {}, String(row.dismissed)),
      el("td", {}, String(row.confirmed)),
      el("td", {}, String(row.pending_human)),
      el("td", {}, formatRate(row.false_positive_reduction_rate))
    )
  );
  if (body.length === 0) body.push(el("tr", {}, el("td", { colspan: 6, class: "muted" }, "暂无数据")));
  return el("section", { class: "bucket" }, el("h3", {}, title), el("table", {}, el("thead", {}, head), el("tbody", {}, ...body)));
}

export function currentStatsParams() {
  const params = {};
  const camera = document.querySelector("#stats-camera")?.value;
  if (camera) params.camera_id = camera;
  if (document.querySelector("#stats-include-sop")?.checked) params.include_sop = true;
  return params;
}

export async function refreshStats() {
  const panel = document.querySelector("#stats");
  if (!panel) return;
  const params = currentStatsParams();
  const data = await api.stats(params);
  const t = data.totals;
  panel.querySelector("#stats-cards").replaceChildren(
    card("CV 总报警数", String(t.total_alarms)),
    card("VLM 拦截（误报）", String(t.dismissed)),
    card("确认事件", String(t.confirmed)),
    card("待人工", String(t.pending_human), `待人工 ${t.needs_human} + 复核失败 ${t.review_failed}`),
    card("误报下降率", formatRate(t.false_positive_reduction_rate), "拦截数 ÷ 总报警数")
  );
  panel.querySelector("#stats-buckets").replaceChildren(
    bucketTable("按摄像头", data.by_camera, "摄像头", (row) => row.camera_id),
    bucketTable("按类型", data.by_type, "类型", (row) => EVENT_TYPE_LABELS[row.event_type] || row.event_type)
  );
}

export function bindStatsPanel() {
  const panel = document.querySelector("#stats");
  if (!panel) return;
  panel.querySelector("#stats-camera")?.addEventListener("change", refreshStats);
  panel.querySelector("#stats-include-sop")?.addEventListener("change", refreshStats);
  const exportParams = () => currentStatsParams();
  panel.querySelector("#export-json")?.addEventListener("click", () => {
    window.location.href = api.statsExportUrl("json", exportParams());
  });
  panel.querySelector("#export-csv")?.addEventListener("click", () => {
    window.location.href = api.statsExportUrl("csv", exportParams());
  });
}

export function fillStatsCameras(cameras) {
  const select = document.querySelector("#stats-camera");
  if (!select) return;
  const current = select.value;
  select.replaceChildren(
    el("option", { value: "" }, "全部摄像头"),
    ...cameras.map((c) => el("option", { value: c.id }, c.name))
  );
  select.value = current;
}
