export default function VersionRow({ version }: any) {
  async function promote() {
    await fetch("http://localhost:8095/v1/promote", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task: version.task,
        model_name: version.model_name,
        version: version.version,
        alias: "production",
      }),
    });

    alert("Promoted to production");
  }

  return (
    <div style={{ padding: 10, borderTop: "1px solid #eee" }}>
      <div>
        Version: <b>{version.version}</b>
      </div>

      <div>Alias: {version.alias || "-"}</div>

      <button onClick={promote}>Promote → Production</button>
    </div>
  );
}