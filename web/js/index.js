// 首页入口：加载视频墙、事件流、统计面板与 SOP 面板，并把 WebSocket 消息分发给各组件。
import { api } from "/static/js/api.js";
import { connectWs } from "/static/js/ws.js";
import { fillStatusFilter, fillTypeFilter, initEventFilters, loadEvents, setFilterOptions, upsertEvent } from "/static/js/events.js";
import { bindStatsPanel, fillStatsCameras, refreshStats } from "/static/js/stats.js";
import { initSopPanel, onSopRunUpdated, refreshRuns } from "/static/js/sop.js";
import { renderWall, updateCameraStatus, updateLatestEvent } from "/static/js/wall.js";
import { initConfigPanel } from "/static/js/config.js";

const connState = document.querySelector("#ws-state");

// 顶栏时钟（本地时间，每秒刷新）。
function startClock() {
  const node = document.querySelector("#clock");
  if (!node) return;
  const tick = () => {
    node.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  };
  tick();
  setInterval(tick, 1000);
}

// 顶栏事件总数：读取 /api/events 的 total（不改变事件列表本身）。
async function refreshEventCounter() {
  const node = document.querySelector("#topbar-events");
  if (!node) return;
  try {
    const { total } = await api.events({ limit: 1 });
    const count = document.createElement("b");
    count.textContent = String(total);
    node.replaceChildren(document.createTextNode("事件 "), count);
  } catch (_) {
    // 顶栏计数失败不影响页面其他部分。
  }
}

function setConnState(status) {
  if (!connState) return;
  const text = { connecting: "连接中", online: "实时连接", offline: "连接断开，正在重连" }[status] || status;
  connState.textContent = text;
  connState.className = `badge ${status === "online" ? "ok" : status === "offline" ? "bad" : "muted"}`;
}

async function loadCameras() {
  const { items } = await api.cameras();
  renderWall(document.querySelector("#wall"), items);
  setFilterOptions(items);
  fillStatsCameras(items);
  return items;
}

async function resync() {
  // 断线期间的消息不会补发，重连后重新拉取列表对账。
  const cameras = await loadCameras();
  await loadEvents();
  await refreshStats();
  await refreshRuns();
  await refreshEventCounter();
  return cameras;
}

function handleMessage(msg) {
  switch (msg.type) {
    case "event.created":
    case "event.updated":
      upsertEvent(msg.data);
      if (msg.data.camera_id) updateLatestEvent(msg.data.camera_id, msg.data);
      refreshStats().catch(() => {});
      refreshEventCounter();
      break;
    case "camera.status":
      updateCameraStatus(msg.data.camera_id, msg.data.status);
      break;
    case "sop.run.updated":
      onSopRunUpdated(msg.data);
      break;
    default:
      break;
  }
}

async function main() {
  fillStatusFilter();
  fillTypeFilter();
  initEventFilters(() => loadEvents().catch(() => {}));
  bindStatsPanel();

  startClock();
  const cameras = await loadCameras();
  await loadEvents();
  await refreshStats();
  await refreshEventCounter();
  await initSopPanel(cameras);
  await initConfigPanel();

  connectWs({ onMessage: handleMessage, onStatus: setConnState, onResync: () => resync().catch(() => {}) });
}

main().catch((err) => {
  const wall = document.querySelector("#wall");
  if (wall) wall.textContent = `页面初始化失败：${err.message}`;
});
