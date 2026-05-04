const BASE = "";

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

const wsProto = window.location.protocol === "https:" ? "wss" : "ws";
export const WS_URL = `${wsProto}://${window.location.host}/ws/logs`;
