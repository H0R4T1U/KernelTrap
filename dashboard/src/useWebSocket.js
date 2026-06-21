import { useEffect, useRef, useCallback } from "react";

const INITIAL_BACKOFF_MS = 2000;
const MAX_BACKOFF_MS = 30000;

export function useWebSocket(url, onMessage) {
  const wsRef = useRef(null);
  const timeoutRef = useRef(null);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      backoffRef.current = INITIAL_BACKOFF_MS;
    };

    ws.onmessage = (e) => {
      try {
        onMessageRef.current(JSON.parse(e.data));
      } catch (err) {
        console.warn("useWebSocket: dropping unparseable message", err);
      }
    };

    ws.onerror = (err) => {
      // Surface the error; onclose fires right after and drives the reconnect,
      // so we deliberately do NOT close() here (that would double the schedule).
      console.warn("useWebSocket: socket error", err);
    };

    ws.onclose = () => {
      // Reconnect with exponential backoff, capped at MAX_BACKOFF_MS.
      timeoutRef.current = setTimeout(connect, backoffRef.current);
      backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
    };
    // Note: we deliberately do NOT close the socket inside onerror —
    // the browser fires onclose automatically right after, and an extra
    // close() on top of that doubles the reconnect schedule.
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, [connect]);
}
