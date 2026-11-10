// WebSocket 客户端：指数退避重连。重连成功后调用 onResync，由页面重新拉取事件与摄像头列表对账。
// 服务端不补发断线期间的消息，客户端按 id 做 upsert，因此重复消息无副作用。

const MIN_DELAY_MS = 1000;
const MAX_DELAY_MS = 30000;
const PING_INTERVAL_MS = 20000;

export function connectWs({ onMessage, onStatus, onResync }) {
  let socket = null;
  let delay = MIN_DELAY_MS;
  let pingTimer = null;
  let hasConnectedBefore = false;

  function url() {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    return `${scheme}//${location.host}/ws/events`;
  }

  function open() {
    onStatus("connecting");
    socket = new WebSocket(url());

    socket.onopen = () => {
      delay = MIN_DELAY_MS;
      onStatus("online");
      if (hasConnectedBefore && onResync) onResync();
      hasConnectedBefore = true;
      pingTimer = setInterval(() => {
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "ping" }));
        }
      }, PING_INTERVAL_MS);
    };

    socket.onmessage = (ev) => {
      let msg = null;
      try {
        msg = JSON.parse(ev.data);
      } catch (_) {
        return;
      }
      onMessage(msg);
    };

    socket.onclose = () => {
      clearInterval(pingTimer);
      onStatus("offline");
      setTimeout(open, delay);
      delay = Math.min(delay * 2, MAX_DELAY_MS);
    };

    socket.onerror = () => {
      // 错误后浏览器会触发 onclose，重连在 onclose 中安排。
    };
  }

  open();
}
