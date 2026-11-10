// REST 客户端封装。所有请求返回解析后的 JSON；非 2xx 时抛出 ApiError（含 code 与 message）。

export class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function buildQuery(params) {
  const usable = Object.entries(params || {}).filter(
    ([, value]) => value !== undefined && value !== null && value !== ""
  );
  if (usable.length === 0) return "";
  return "?" + new URLSearchParams(usable.map(([k, v]) => [k, String(v)])).toString();
}

async function parseOrThrow(response) {
  let body = null;
  try {
    body = await response.json();
  } catch (_) {
    body = null;
  }
  if (!response.ok) {
    const err = (body && body.error) || {};
    throw new ApiError(response.status, err.code || "internal_error", err.message || response.statusText);
  }
  return body;
}

async function getJSON(path, params) {
  const response = await fetch(path + buildQuery(params), { headers: { Accept: "application/json" } });
  return parseOrThrow(response);
}

async function postJSON(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  return parseOrThrow(response);
}

export const api = {
  health: () => getJSON("/api/health"),
  cameras: () => getJSON("/api/cameras"),
  events: (params) => getJSON("/api/events", params),
  event: (id) => getJSON(`/api/events/${encodeURIComponent(id)}`),
  changeStatus: (id, payload) => postJSON(`/api/events/${encodeURIComponent(id)}/status`, payload),
  stats: (params) => getJSON("/api/stats", params),
  statsExportUrl: (format, params) => "/api/stats/export" + buildQuery({ ...(params || {}), format }),
  sops: () => getJSON("/api/sops"),
  sopRuns: (params) => getJSON("/api/sop/runs", params),
  sopRun: (id) => getJSON(`/api/sop/runs/${encodeURIComponent(id)}`),
  triggerSop: (payload) => postJSON("/api/sop/runs", payload),
};
