// 视频墙：每路 Camera 一张卡片，画面为 MJPEG 流，显示在线状态与最近事件角标（需求 7.1、7.2、1.2）。
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
  card.statusBadge.textContent = status === "online" ? "在线" : "离线";
  card.statusBadge.className = `badge ${status === "online" ? "ok" : "bad"}`;
  if (status !== "online") {
    card.img.removeAttribute("src");
    card.overlayText.textContent = "画面离线，等待重连";
    card.overlayText.hidden = false;
  } else {
    card.overlayText.hidden = true;
    card.img.src = card.streamUrl;
  }
}

function createCard(camera) {
  const img = el("img", { alt: `${camera.name} 实时画面`, class: "stream" });
  const overlayText = el("div", { class: "stream-off" }, "画面离线，等待重连");
  overlayText.hidden = true;
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
    el("div", { class: "stream-box" }, img, overlayText),
    el("footer", { class: "cam-foot" }, latestBadge)
  );
  const card = {
    root,
    img,
    overlayText,
    statusBadge,
    latestBadge,
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
  }
  for (const [id, card] of cards) {
    if (!seen.has(id)) {
      card.root.remove();
      cards.delete(id);
    }
  }
  if (cameras.length === 0) {
    container.replaceChildren(el("p", { class: "muted" }, "尚未接入任何视频源"));
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
