// 事件流与筛选（需求 7.4、7.6、6.5）。
// - 新事件（event.created）置顶，event.updated 按 id upsert。
// - 每条显示类型、时间，以及 CV 与 VLM 两层徽标。
// - 筛选条件作用于前端列表；dismissed 事件不做隐藏，除非用户选择了其他状态。
import { api } from "/static/js/api.js";
import { EVENT_TYPE_LABELS, STATUS_LABELS, el, formatTime, vlmBadge } from "/static/js/labels.js";

const state = {
  items: new Map(), // id -> EventDTO（不含 trajectory）
  filters: { camera_id: "", event_type: "", status: "" },
  onChange: () => {},
};

export function setFilterOptions(cameras) {
  const select = document.querySelector("#filter-camera");
  if (!select) return;
  const current = select.value;
  select.replaceChildren(
    el("option", { value: "" }, "全部摄像头"),
    ...cameras.map((c) => el("option", { value: c.id }, c.name))
  );
  select.value = current;
}

function matches(event) {
  const f = state.filters;
  if (f.camera_id && event.camera_id !== f.camera_id) return false;
  if (f.event_type && event.event_type !== f.event_type) return false;
  if (f.status && event.status !== f.status) return false;
  return true;
}

function sorted() {
  return [...state.items.values()].sort((a, b) =>
    b.created_ms - a.created_ms || (a.id < b.id ? 1 : -1)
  );
}

function renderItem(event) {
  const cv = el(
    "span",
    { class: "badge" },
    `CV：${EVENT_TYPE_LABELS[event.event_type] || event.event_type}`
  );
  const vlm = vlmBadge(event.vlm);
  const vlmEl = el("span", { class: `badge ${vlm.kind}` }, `VLM：${vlm.text}`);
  const status = el("span", { class: "badge" }, STATUS_LABELS[event.status] || event.status);
  return el(
    "li",
    { class: "event-row", "data-event-id": event.id },
    el(
      "a",
      { href: `/static/event.html?id=${encodeURIComponent(event.id)}`, class: "event-link" },
      el("span", { class: "event-time" }, formatTime(event.created_ms)),
      el("span", { class: "event-cam" }, event.camera_name),
      el("span", { class: "event-badges" }, cv, vlmEl, status)
    )
  );
}

export function renderEvents() {
  const list = document.querySelector("#event-list");
  if (!list) return;
  const rows = sorted().filter(matches).map(renderItem);
  if (rows.length === 0) {
    list.replaceChildren(el("li", { class: "muted" }, "暂无符合条件的事件"));
  } else {
    list.replaceChildren(...rows);
  }
  const counter = document.querySelector("#event-count");
  if (counter) counter.textContent = `共 ${rows.length} 条`;
}

export function upsertEvent(event) {
  if (!event || !event.id) return;
  state.items.set(event.id, event);
  renderEvents();
}

export async function loadEvents() {
  const { items } = await api.events({ limit: 100, ...nonEmptyFilters() });
  state.items = new Map(items.map((e) => [e.id, e]));
  renderEvents();
}

function nonEmptyFilters() {
  const out = {};
  for (const [k, v] of Object.entries(state.filters)) {
    if (v) out[k] = v;
  }
  return out;
}

export function initEventFilters(onFilterChange) {
  const bind = (id, key) => {
    const node = document.querySelector(id);
    if (!node) return;
    node.addEventListener("change", () => {
      state.filters[key] = node.value;
      renderEvents();
      onFilterChange();
    });
  };
  bind("#filter-camera", "camera_id");
  bind("#filter-type", "event_type");
  bind("#filter-status", "status");
}

export function currentEvents() {
  return [...state.items.values()];
}

export function getStatusFilter() {
  return state.filters.status;
}

export function fillStatusFilter() {
  const select = document.querySelector("#filter-status");
  if (!select) return;
  select.replaceChildren(
    el("option", { value: "" }, "全部状态"),
    ...Object.entries(STATUS_LABELS).map(([value, label]) => el("option", { value }, label))
  );
}

export function fillTypeFilter() {
  const select = document.querySelector("#filter-type");
  if (!select) return;
  select.replaceChildren(
    el("option", { value: "" }, "全部类型"),
    ...Object.entries(EVENT_TYPE_LABELS).map(([value, label]) => el("option", { value }, label))
  );
}
