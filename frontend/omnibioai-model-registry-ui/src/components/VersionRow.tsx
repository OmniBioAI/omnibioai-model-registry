import { authHeader } from "../api/auth";

export default function VersionRow({ version }: { version: any }) {
  async function promote() {
    await fetch("/v1/promote", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader() },
      body: JSON.stringify({
        task: version.task,
        model_name: version.model_name,
        version: version.version,
        alias: "production",
        actor: "registry-ui",
      }),
    });
    alert("Promoted to production");
  }

  return (
    <div style={{ padding: 10, borderTop: "1px solid var(--border)" }}>
      <div>Version: <b>{version.version}</b></div>
      <div>Alias: {version.alias || "—"}</div>
      <button onClick={promote} style={{ marginTop: 6 }}>Promote → Production</button>
    </div>
  );
}
