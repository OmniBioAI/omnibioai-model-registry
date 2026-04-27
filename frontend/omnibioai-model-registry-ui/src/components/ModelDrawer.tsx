import { useState } from "react";
import ModelLineageView from "./ModelLineageView";

const BASE_URL = "http://localhost:8095/v1";

type Model = {
  task: string;
  model_name: string;
  version: string;
  alias?: string;
  created_at?: string;
  [key: string]: any;
};

export default function ModelDrawer({
  model,
  onClose,
}: {
  model: Model;
  onClose: () => void;
}) {
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState<any>(null);
  const [showLineage, setShowLineage] = useState(false);

  async function callAPI(path: string, body?: any) {
    setLoading(true);
    setOutput(null);

    try {
      const res = await fetch(`${BASE_URL}${path}`, {
        method: body ? "POST" : "GET",
        headers: body ? { "Content-Type": "application/json" } : {},
        body: body ? JSON.stringify(body) : undefined,
      });

      const data = await res.json();
      setOutput(data);
    } catch (e) {
      setOutput({ error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  const ref = `${model.model_name}@${model.version}`;

  return (
    <div style={styles.overlay}>
      <div style={styles.drawer}>
        {/* HEADER */}
        <div style={styles.header}>
          <h2>Model Details</h2>
          <button onClick={onClose}>✕</button>
        </div>

        {/* BASIC INFO */}
        <div style={styles.section}>
          <div><b>Name:</b> {model.model_name}</div>
          <div><b>Task:</b> {model.task}</div>
          <div><b>Version:</b> {model.version}</div>
          <div><b>Alias:</b> {model.alias || "-"}</div>
          <div><b>Created:</b> {model.created_at || "-"}</div>
        </div>

        {/* ACTIONS */}
        <div style={styles.actions}>
          <button onClick={() =>
            callAPI(`/resolve?task=${model.task}&ref=${ref}`)
          }>
            Resolve
          </button>

          <button onClick={() =>
            callAPI(`/verify`, { task: model.task, ref })
          }>
            Verify
          </button>

          <button onClick={() =>
            callAPI(`/promote`, {
              task: model.task,
              model_name: model.model_name,
              alias: "latest",
              version: model.version,
            })
          }>
            Promote → latest
          </button>

          {/* NEW: LINEAGE VIEW */}
          <button onClick={() => setShowLineage(!showLineage)}>
            {showLineage ? "Hide Lineage" : "View Lineage"}
          </button>
        </div>

        {/* OUTPUT */}
        <div style={styles.output}>
          {loading && <div>Running...</div>}
          {output && (
            <pre style={{ fontSize: 12 }}>
              {JSON.stringify(output, null, 2)}
            </pre>
          )}
        </div>

        {/* LINEAGE VIEW (NEW MLflow-like feature) */}
        {showLineage && (
          <div style={{ marginTop: 20 }}>
            <ModelLineageView model={model} />
          </div>
        )}
      </div>
    </div>
  );
}

const styles: any = {
  overlay: {
    position: "fixed",
    right: 0,
    top: 0,
    width: "40%",
    height: "100%",
    background: "rgba(0,0,0,0.3)",
    display: "flex",
  },
  drawer: {
    background: "#fff",
    width: "100%",
    padding: 20,
    overflowY: "auto",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
  },
  section: {
    marginTop: 20,
    lineHeight: 1.6,
  },
  actions: {
    marginTop: 20,
    display: "flex",
    gap: 10,
    flexWrap: "wrap",
  },
  output: {
    marginTop: 20,
    background: "#f6f6f6",
    padding: 10,
  },
};