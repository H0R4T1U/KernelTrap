// Empty BASE on purpose: FastAPI serves the dashboard at /dashboard/ but the
// API endpoints (/agents, /users, /pivot, /ws/logs, ...) live on the app root.
// Relative paths resolve against window.location.origin, hitting the API directly.
const BASE = "";

// Throw on non-2xx so callers' catch blocks fire instead of choking on an HTML
// error page parsed as JSON.
async function getJSON(path) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return r.json();
}

export async function fetchAgents() {
  return getJSON("/agents");
}

export async function fetchUsers() {
  return getJSON("/users");
}

export async function fetchPivotHistory() {
  return getJSON("/pivot-history");
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
