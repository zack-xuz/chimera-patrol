// 视频墙：每路 Camera 一张卡片，画面为 MJPEG 流，显示在线状态与最近事件角标（需求 7.1、7.2、1.2）。
import { api } from "/static/js/api.js";
import { EVENT_TYPE_LABELS, STATUS_LABELS, el, formatTime } from "/static/js/labels.js";

const cards = new Map();

function badgeText(latest) {
  if (!latest) return "暂无事件";
  const type = EVENT_TYPE_LABELS[latest.event_type] || latest.event_type;
  const status = STATUS_LABELS[latest.status] || latest.status;
  return `${type} · ${status} · ${formatTime(latest.created_ms)}`;
}

function setStatus(card, status) {
  card.root.classList.toggle("is-offline", status !== "online");
  card.statusBadge.textContent = status === "online" ? "LIVE" : "离线";
  card.statusBadge.className = status === "online" ? "badge live" : "badge bad";
  if (status !== "online") {
    card.img.removeAttribute("src");
    card.overlayText.textContent = "画面离线，等待重连";
    card.overlayText.hidden = false;
  } else {
    card.overlayText.hidden = true;
    card.img.src = card.streamUrl;
  }
}

function setPaused(card, paused) {
  card.paused = paused;
  card.pausedBadge.hidden = !paused;
  card.toggleButton.textContent = paused ? "恢复检测" : "暂停检测";
  card.toggleButton.setAttribute("aria-pressed", paused ? "true" : "false");
}

function createCard(camera) {
  const img = el("img", { alt: `${camera.name} 实时画面`, class: "stream" });
  const overlayText = el("div", { class: "stream-off" }, "画面离线，等待重连");
  overlayText.hidden = true;
  // 检测暂停只影响报警，视频流继续播放，这里只在画面上叠加提示。
  const pausedBadge = el("div", { class: "paused-badge" }, "检测已暂停");
  pausedBadge.hidden = true;
  const toggleButton = el("button", { type: "button", class: "toggle-detect" }, "暂停检测");
  toggleButton.addEventListener("click", async () => {
    const card = cards.get(camera.id);
    if (!card) return;
    toggleButton.disabled = true;
    try {
      const action = card.paused ? api.resumeCamera : api.pauseCamera;
      const result = await action(camera.id);
      setPaused(card, result.paused);
    } catch (err) {
      card.latestBadge.textContent = `操作失败：${err.message}`;
    } finally {
      toggleButton.disabled = false;
    }
  });
  img.addEventListener("error", () => {
    if (cards.get(camera.id)?.statusBadge.textContent === "在线") {
      overlayText.textContent = "画面读取失败";
      overlayText.hidden = false;
    }
  });
  const statusBadge = el("span", { class: "badge" }, "—");
  const latestBadge = el("span", { class: "latest" }, badgeText(camera.latest_event));
  const root = el(
    "article",
    { class: "cam-card", "data-camera-id": camera.id },
    el("header", { class: "cam-head" }, el("strong", {}, camera.name), statusBadge),
    el("div", { class: "stream-box" }, img, overlayText, pausedBadge),
    el("footer", { class: "cam-foot" }, latestBadge, toggleButton)
  );
  const card = {
    root,
    img,
    overlayText,
    statusBadge,
    latestBadge,
    pausedBadge,
    toggleButton,
    paused: false,
    streamUrl: `${camera.stream_url}?overlay=1`,
  };
  return card;
}

export function renderWall(container, cameras) {
  const seen = new Set();
  for (const camera of cameras) {
    seen.add(camera.id);
    let card = cards.get(camera.id);
    if (!card) {
      card = createCard(camera);
      cards.set(camera.id, card);
      container.append(card.root);
    }
    card.latestBadge.textContent = badgeText(camera.latest_event);
    setStatus(card, camera.status);
    setPaused(card, Boolean(camera.paused));
  }
  for (const [id, card] of cards) {
    if (!seen.has(id)) {
      card.root.remove();
      cards.delete(id);
    }
  }
  if (cameras.length === 0) {
    container.replaceChildren(el("p", { class: "empty" }, "尚未接入任何视频源。在 config.yaml 或视频目录中添加视频后重启服务。"));
  }
}

export function updateCameraStatus(cameraId, status) {
  const card = cards.get(cameraId);
  if (card) setStatus(card, status);
}

export function updateLatestEvent(cameraId, event) {
  const card = cards.get(cameraId);
  if (!card) return;
  card.latestBadge.textContent = badgeText({
    event_type: event.event_type,
    status: event.status,
    created_ms: event.created_ms,
  });
}
