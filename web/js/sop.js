// SOP 检查面板（需求 11.5、11.7、11.9）。
// - 逐步骤亮灯：pass 绿、fail 红、unknown 灰，旁显判定依据。
// - approximate 为真的步骤标注“近似判定”。
// - check_failed_retry 时显示失败原因与“重新检查”按钮，重新检查通过 retry_of 发起。
import { api, ApiError } from "/static/js/api.js";
import {
  FAILURE_REASON_LABELS,
  SOP_RUN_STATUS_LABELS,
  STEP_STATUS_LABELS,
  el,
  formatTime,
} from "/static/js/labels.js";

const state = { selectedRunId: null, sops: [], cameras: [] };

function operatorName() {
  const value = document.querySelector("#sop-operator")?.value.trim();
  return value || "巡检员";
}

function renderRuns(items) {
  const list = document.querySelector("#sop-runs");
  if (!list) return;
  if (items.length === 0) {
    list.replaceChildren(el("li", { class: "muted" }, "暂无检查记录"));
    return;
  }
  list.replaceChildren(
    ...items.map((run) =>
      el(
        "li",
        {
          class: `run-row ${run.id === state.selectedRunId ? "is-selected" : ""}`,
          onclick: () => selectRun(run.id),
        },
        el("span", {}, formatTime(run.created_ms)),
        el("span", {}, run.sop_name),
        el("span", { class: `badge ${run.status === "completed" ? "ok" : run.status === "running" ? "muted" : "bad"}` },
          SOP_RUN_STATUS_LABELS[run.status] || run.status)
      )
    )
  );
}

function lamp(status) {
  return el("span", { class: `lamp lamp-${status}`, title: STEP_STATUS_LABELS[status] || status });
}

function renderDetail(run) {
  const box = document.querySelector("#sop-detail");
  if (!box) return;
  if (!run) {
    box.replaceChildren(el("p", { class: "muted" }, "选择一条检查记录查看结果"));
    return;
  }
  const header = el(
    "div",
    { class: "sop-head" },
    el("strong", {}, `${run.sop_name} · ${run.camera_id}`),
    el("span", { class: "badge" }, SOP_RUN_STATUS_LABELS[run.status] || run.status),
    el("span", { class: "muted" }, `操作人 ${run.operator}，采样 ${run.sample_count} 帧`)
  );
  const parts = [header];

  if (run.status === "check_failed_retry") {
    const reason = FAILURE_REASON_LABELS[run.failure_reason] || run.failure_reason;
    const retry = el(
      "button",
      {
        type: "button",
        onclick: async () => {
          try {
            const next = await api.triggerSop({
              sop_id: run.sop_id,
              camera_id: run.camera_id,
              operator: operatorName(),
              retry_of: run.id,
            });
            selectRun(next.id);
          } catch (err) {
            showError(err);
          }
        },
      },
      "重新检查"
    );
    parts.push(el("div", { class: "sop-failed" }, el("p", {}, `检查失败：${reason}。未生成任何判定结论。`), retry));
  }

  if (run.steps.length > 0) {
    parts.push(
      el(
        "ol",
        { class: "sop-steps" },
        ...run.steps.map((step) =>
          el(
            "li",
            { class: `step step-${step.status}` },
            lamp(step.status),
            el("span", { class: "step-name" }, step.description),
            el("span", { class: "step-status" }, STEP_STATUS_LABELS[step.status] || step.status),
            step.approximate ? el("span", { class: "badge muted" }, "近似判定") : null,
            el("p", { class: "step-evidence" }, step.evidence)
          )
        )
      )
    );
  }

  if (run.event_id) {
    parts.push(el("p", {}, el("a", { href: `/static/event.html?id=${encodeURIComponent(run.event_id)}` }, "查看不合规事件")));
  }
  box.replaceChildren(...parts);
}

function showError(err) {
  const box = document.querySelector("#sop-error");
  if (!box) return;
  box.textContent = err instanceof ApiError ? `${err.message}（${err.code}）` : String(err);
  box.hidden = false;
  setTimeout(() => (box.hidden = true), 5000);
}

export async function selectRun(runId) {
  state.selectedRunId = runId;
  try {
    const run = await api.sopRun(runId);
    renderDetail(run);
  } catch (err) {
    showError(err);
  }
  await refreshRuns();
}

export async function refreshRuns() {
  if (!document.querySelector("#sop-runs")) return;
  const { items } = await api.sopRuns({ limit: 20 });
  renderRuns(items);
  if (state.selectedRunId) {
    const current = items.find((r) => r.id === state.selectedRunId);
    if (current) renderDetail(current);
  }
}

// WebSocket 推送 sop.run.updated 时调用：列表刷新，选中的运行同步更新。
export function onSopRunUpdated(run) {
  if (!document.querySelector("#sop-runs")) return;
  if (run.id === state.selectedRunId) renderDetail(run);
  refreshRuns().catch(() => {});
}

export async function initSopPanel(cameras) {
  const panel = document.querySelector("#sop-panel");
  if (!panel) return;
  state.cameras = cameras;

  const [{ items: sops }] = await Promise.all([api.sops()]);
  state.sops = sops;
  const sopSelect = document.querySelector("#sop-select");
  sopSelect.replaceChildren(...sops.map((s) => el("option", { value: s.id }, s.name)));

  const camSelect = document.querySelector("#sop-camera");
  camSelect.replaceChildren(...cameras.map((c) => el("option", { value: c.id }, c.name)));

  document.querySelector("#sop-trigger")?.addEventListener("click", async () => {
    try {
      const run = await api.triggerSop({
        sop_id: sopSelect.value,
        camera_id: camSelect.value,
        operator: operatorName(),
      });
      await selectRun(run.id);
    } catch (err) {
      showError(err);
    }
  });

  renderDetail(null);
  await refreshRuns();
}
