// This app is served same-origin alongside omnibioai-studio's web-ui
// (both under webstudio.omnibioai.org, via nginx-router.conf's
// /_svc/modelregistry route), so it can read the same JWT web-ui's own
// login flow already stores — no separate login/session concept needed
// here. Key name must match omnibioai-studio/src/ui/lib/session.js's
// TOKEN_KEY exactly.
const TOKEN_KEY = "omnibioai_access_token";

export function authHeader(): Record<string, string> {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}
