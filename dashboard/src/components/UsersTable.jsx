import { useEffect, useState } from "react";
import { fetchUsers, triggerPivot } from "../api";

function Gauge({ count, threshold }) {
  const pct = Math.min((count / threshold) * 100, 100);
  const cls = pct >= 100 ? "gauge-full" : pct >= 70 ? "gauge-warn" : "gauge-ok";
  return (
    <div className="gauge-wrap" title={`${count}/${threshold} severity-2 events`}>
      <div className="gauge-track">
        <div className={`gauge-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="gauge-num dim">{count}/{threshold}</span>
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
                <th>Anomaly pressure (60 s window)</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={`${u.hostname}-${u.user_id}`} className={u.pivoted ? "row-pivoted" : ""}>
                  <td>{u.hostname}</td>
                  <td>{u.user_id}</td>
                  <td>
                    <Gauge
                      count={u.window.severity2_count ?? 0}
                      threshold={u.window.severity2_threshold ?? 1}
                    />
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
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
