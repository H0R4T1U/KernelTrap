const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchAgents() {
  const r = await fetch(`${BASE}/agents`);
  return r.json();
}

export async function fetchUsers() {
  const r = await fetch(`${BASE}/users`);
  return r.json();
}

export async function fetchPivotHistory() {
  const r = await fetch(`${BASE}/pivot-history`);
  return r.json();
}

export async function triggerPivot(hostname, userId) {
  const r = await fetch(`${BASE}/pivot/${encodeURIComponent(hostname)}/${userId}`, {
    method: "POST",
  });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.error || `HTTP ${r.status}`);
  }
  return r.json();
}

export const WS_URL = (import.meta.env.VITE_WS_URL || "ws://localhost:8000") + "/ws/logs";
