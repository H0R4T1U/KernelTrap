import { useEffect, useRef, useCallback } from "react";

export function useWebSocket(url, onMessage) {
  const wsRef = useRef(null);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      try {
        onMessageRef.current(JSON.parse(e.data));
      } catch {}
    };

    ws.onclose = () => {
      // Reconnect after 2 s
      setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);
}
