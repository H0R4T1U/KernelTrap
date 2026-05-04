import { useEffect, useState } from "react";
import AgentsPanel from "./components/AgentsPanel";
import LiveLogs from "./components/LiveLogs";
import UsersTable from "./components/UsersTable";
import PivotHistory from "./components/PivotHistory";
import "./App.css";

const BASE = "";

export default function App() {
  const [healthy, setHealthy] = useState(null);
  const [agentCount, setAgentCount] = useState(0);

  useEffect(() => {
    async function check() {
      try {
        const r = await fetch(`${BASE}/health`);
        const d = await r.json();
        setHealthy(d.status === "ok");
      } catch {
        setHealthy(false);
      }
    }
    async function countAgents() {
      try {
        const r = await fetch(`${BASE}/agents`);
        const d = await r.json();
        setAgentCount((d.agents || []).length);
      } catch {}
    }
    check();
    countAgents();
    const id1 = setInterval(check, 10000);
    const id2 = setInterval(countAgents, 5000);
    return () => { clearInterval(id1); clearInterval(id2); };
  }, []);

  const statusDot = healthy === null ? "dot-unknown" : healthy ? "dot-ok" : "dot-err";
  const statusText = healthy === null ? "connecting…" : healthy ? "OK" : "OFFLINE";

  return (
    <div className="app">
      <header className="header">
        <span className="header-brand">KernelTrap</span>
        <span className="header-sep">|</span>
        <span className="header-meta">
          <span className={`status-dot ${statusDot}`} />
          system: {statusText}
        </span>
        <span className="header-sep">|</span>
        <span className="header-meta">{agentCount} agent{agentCount !== 1 ? "s" : ""}</span>
      </header>

      <main className="grid">
        <div className="col-left">
          <AgentsPanel />
        </div>
        <div className="col-right">
          <LiveLogs />
        </div>
        <div className="col-full">
          <UsersTable />
        </div>
        <div className="col-full">
          <PivotHistory />
        </div>
      </main>
    </div>
  );
}
