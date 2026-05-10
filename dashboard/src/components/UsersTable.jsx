import { useEffect, useState } from "react";
import { fetchUsers, triggerPivot } from "../api";

// Pivot trigger requires BOTH gates to pass:
//   - count >= severity2_threshold (absolute floor)
//   - severity2_rate >= min_severity2_rate (rate gate vs total events)

function CountGauge({ count, threshold }) {
  const pct = Math.min((count / threshold) * 100, 100);
  const cls = pct >= 100 ? "gauge-full" : pct >= 70 ? "gauge-warn" : "gauge-ok";
  return (
    <div className="gauge-wrap" title={`Count gate: ${count}/${threshold} severity-2 events (floor)`}>
      <span className="gauge-label">cnt</span>
      <div className="gauge-track">
        <div className={`gauge-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="gauge-num dim">{count}/{threshold}</span>
    </div>
  );
}

function RateGauge({ rate, minRate, total }) {
  const ratePct = (rate ?? 0) * 100;
  const minPct = (minRate ?? 0) * 100;
  const fillPct = Math.min((ratePct / Math.max(minPct, 0.01)) * 100, 100);
  const cls = fillPct >= 100 ? "gauge-full" : fillPct >= 70 ? "gauge-warn" : "gauge-ok";
  const ratio = total > 0 ? `${ratePct.toFixed(0)}%` : "—";
  return (
    <div
      className="gauge-wrap"
      title={`Rate gate: ${ratePct.toFixed(1)}% / ${minPct.toFixed(0)}% (${total} total events in window)`}
    >
      <span className="gauge-label">rate</span>
      <div className="gauge-track">
        <div className={`gauge-fill ${cls}`} style={{ width: `${fillPct}%` }} />
      </div>
      <span className="gauge-num dim">{ratio}/{minPct.toFixed(0)}%</span>
    </div>
  );
}

export default function UsersTable() {
  const [users, setUsers] = useState([]);
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await fetchUsers();
        if (!cancelled) setUsers(data.users || []);
      } catch {}
    }
    load();
    const id = setInterval(load, 3000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  function addToast(msg, ok) {
    const id = Date.now();
    setToasts((t) => [...t, { id, msg, ok }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3000);
  }

  async function handlePivot(hostname, userId) {
    try {
      await triggerPivot(hostname, userId);
      addToast(`Pivot sent → ${hostname} uid:${userId}`, true);
    } catch (e) {
      addToast(`Error: ${e.message}`, false);
    }
  }

  return (
    <section className="panel">
      <h2 className="panel-title">Users &amp; Window State</h2>
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.ok ? "toast-ok" : "toast-err"}`}>{t.msg}</div>
      ))}
      {users.length === 0 ? (
        <p className="dim">No users tracked yet</p>
      ) : (
        <div className="table-wrap">
          <table className="users-table">
            <thead>
              <tr>
                <th>Host</th>
                <th>UID</th>
                <th>Anomaly pressure (60 s window — both gates required)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => {
                const w = u.window || {};
                const count = w.severity2_count ?? 0;
                const total = w.total_count ?? 0;
                const rate = w.severity2_rate ?? 0;
                const threshold = w.severity2_threshold ?? 1;
                const minRate = w.min_severity2_rate ?? 0.30;
                const countOk = count >= threshold;
                const rateOk = rate >= minRate;
                const bothGates = countOk && rateOk;
                return (
                  <tr
                    key={`${u.hostname}-${u.user_id}`}
                    className={u.pivoted ? "row-pivoted" : (bothGates ? "row-armed" : "")}
                  >
                    <td>{u.hostname}</td>
                    <td>{u.user_id}</td>
                    <td>
                      <div className="gauge-stack">
                        <CountGauge count={count} threshold={threshold} />
                        <RateGauge rate={rate} minRate={minRate} total={total} />
                      </div>
                    </td>
                    <td>
                      {u.pivoted ? (
                        <span className="badge-pivoted">PIVOTED</span>
                      ) : (
                        <button
                          className="btn-pivot"
                          onClick={() => handlePivot(u.hostname, u.user_id)}
                        >
                          PIVOT
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
