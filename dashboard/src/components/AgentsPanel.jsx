import { useEffect, useState } from "react";
import { fetchAgents } from "../api";

function timeSince(ts) {
  if (!ts) return "never";
  const s = Math.floor(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  return `${Math.floor(s / 60)}m ago`;
}

export default function AgentsPanel() {
  const [agents, setAgents] = useState([]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await fetchAgents();
        if (!cancelled) setAgents(data.agents || []);
      } catch {}
    }
    load();
    const id = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return (
    <section className="panel">
      <h2 className="panel-title">Connected Agents</h2>
      {agents.length === 0 ? (
        <p className="dim">No agents connected</p>
      ) : (
        <ul className="agent-list">
          {agents.map((a) => (
            <li key={a.hostname} className="agent-item">
              <span className="agent-dot" />
              <span className="agent-name">{a.hostname}</span>
              <span className="agent-meta">{a.events_per_min} ev/min</span>
              <span className="agent-meta dim">{timeSince(a.last_seen)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
