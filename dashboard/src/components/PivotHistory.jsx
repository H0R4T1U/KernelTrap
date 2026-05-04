import { useEffect, useState } from "react";
import { fetchPivotHistory } from "../api";

function fmtTs(ts) {
  return new Date(ts * 1000).toLocaleTimeString();
}

const TRIGGER_CLASS = {
  manual: "trigger-manual",
  threshold: "trigger-threshold",
  H1: "trigger-h",
  H2: "trigger-h",
  H3: "trigger-h",
  H4: "trigger-h",
};

export default function PivotHistory() {
  const [pivots, setPivots] = useState([]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const data = await fetchPivotHistory();
        if (!cancelled) setPivots(data.pivots || []);
      } catch {}
    }
    load();
    const id = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return (
    <section className="panel">
      <h2 className="panel-title">Pivot History</h2>
      {pivots.length === 0 ? (
        <p className="dim">No pivots yet</p>
      ) : (
        <table className="history-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Host</th>
              <th>UID</th>
              <th>Trigger</th>
            </tr>
          </thead>
          <tbody>
            {pivots.map((p, i) => (
              <tr key={i}>
                <td className="dim">{fmtTs(p.timestamp)}</td>
                <td>{p.hostname}</td>
                <td>{p.user_id}</td>
                <td>
                  <span className={`trigger-badge ${TRIGGER_CLASS[p.trigger] || ""}`}>
                    {p.trigger}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
