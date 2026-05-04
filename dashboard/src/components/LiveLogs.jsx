import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "../useWebSocket";
import { WS_URL as WS } from "../api";

const MAX_EVENTS = 500;

const SEV_CLASS = ["sev0", "sev1", "sev2"];
const SEV_LABEL = ["—", "LOW", "HIGH"];

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toTimeString().slice(0, 8);
}

export default function LiveLogs() {
  const [events, setEvents] = useState([]);
  const [filter, setFilter] = useState("all"); // "all" | "1" | "2"
  const [hostFilter, setHostFilter] = useState("all");
  const bottomRef = useRef(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useWebSocket(WS, (event) => {
    setEvents((prev) => {
      const next = [...prev, event];
      return next.length > MAX_EVENTS ? next.slice(next.length - MAX_EVENTS) : next;
    });
  });

  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [events, autoScroll]);

  const hosts = ["all", ...Array.from(new Set(events.map((e) => e.hostname)))];

  const visible = events.filter((e) => {
    if (filter !== "all" && String(e.severity) !== filter) return false;
    if (hostFilter !== "all" && e.hostname !== hostFilter) return false;
    return true;
  });

  return (
    <section className="panel panel-logs">
      <div className="panel-header">
        <h2 className="panel-title">Live Logs</h2>
        <div className="log-filters">
          <span className="dim">sev:</span>
          {["all", "0", "1", "2"].map((v) => (
            <button
              key={v}
              className={`filter-btn ${filter === v ? "active" : ""}`}
              onClick={() => setFilter(v)}
            >
              {v === "all" ? "all" : SEV_LABEL[parseInt(v)]}
            </button>
          ))}
          <span className="dim" style={{ marginLeft: 8 }}>host:</span>
          <select
            className="filter-select"
            value={hostFilter}
            onChange={(e) => setHostFilter(e.target.value)}
          >
            {hosts.map((h) => (
              <option key={h} value={h}>{h}</option>
            ))}
          </select>
          <button
            className={`filter-btn ${autoScroll ? "active" : ""}`}
            style={{ marginLeft: 8 }}
            onClick={() => setAutoScroll((v) => !v)}
          >
            auto-scroll
          </button>
        </div>
      </div>
      <div className="log-body">
        {visible.length === 0 && <p className="dim">Waiting for events…</p>}
        {visible.map((e, i) => (
          <div key={i} className={`log-row ${SEV_CLASS[e.severity] || ""}`}>
            <span className="log-time">{fmtTime(e.timestamp ?? Date.now() / 1000)}</span>
            <span className={`log-sev sev-badge-${e.severity}`}>{SEV_LABEL[e.severity] ?? "?"}</span>
            <span className="log-host dim">{e.hostname}</span>
            <span className="log-uid dim">uid:{e.userId}</span>
            <span className="log-event">{e.eventName}</span>
            <span className="log-proc dim">{e.processName}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </section>
  );
}
