import { useState } from "react";

const BASE_URL = "http://localhost:8095/v1";

export default function RegisterModelModal({
  onClose,
  onSuccess,
}: {
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [form, setForm] = useState({
    task: "",
    model_name: "",
    version: "",
    artifacts_dir: "",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  function update(field: string, value: string) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  async function submit() {
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(`${BASE_URL}/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          metadata: {},
          set_alias: "latest",
        }),
      });

      const data = await res.json();
      setResult(data);

      if (res.ok) {
        onSuccess();
      }
    } catch (e) {
      setResult({ error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.overlay}>
      <div style={styles.modal}>
        <h2>Register Model</h2>

        <input
          placeholder="Task"
          value={form.task}
          onChange={(e) => update("task", e.target.value)}
        />

        <input
          placeholder="Model Name"
          value={form.model_name}
          onChange={(e) => update("model_name", e.target.value)}
        />

        <input
          placeholder="Version"
          value={form.version}
          onChange={(e) => update("version", e.target.value)}
        />

        <input
          placeholder="Artifacts Dir (inside container)"
          value={form.artifacts_dir}
          onChange={(e) => update("artifacts_dir", e.target.value)}
        />

        <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
          <button onClick={submit} disabled={loading}>
            {loading ? "Registering..." : "Register"}
          </button>

          <button onClick={onClose}>Cancel</button>
        </div>

        {result && (
          <pre style={{ marginTop: 10, fontSize: 12 }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}

const styles: any = {
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0,0,0,0.4)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  modal: {
    background: "#fff",
    padding: 20,
    width: 400,
    display: "flex",
    flexDirection: "column",
    gap: 10,
  },
};