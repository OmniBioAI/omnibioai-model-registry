import { useEffect, useState } from "react";
import { fetchModels } from "./api/registry";
import ModelDrawer from "./components/ModelDrawer";
import RegisterModelModal from "./components/RegisterModelModal";
import ModelLineageView from "./components/ModelLineageView";

type Model = {
  task: string;
  model_name: string;
  version: string;
  alias?: string;
  created_at?: string;
  lineage?: any;
};

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Model | null>(null);
  const [showRegister, setShowRegister] = useState(false);

  // NEW: explorer selected model (group view)
  const [explorerModel, setExplorerModel] = useState<string | null>(null);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    try {
      const data = await fetchModels();
      const normalized = Array.isArray(data) ? data : data.models ?? [];
      setModels(normalized);
    } finally {
      setLoading(false);
    }
  }

  const filtered = models.filter((m) =>
    `${m.model_name ?? ""} ${m.task ?? ""} ${m.version ?? ""}`
      .toLowerCase()
      .includes(search.toLowerCase())
  );

  // NEW: group models by model_name (MLflow-style explorer)
  const grouped = filtered.reduce<Record<string, Model[]>>((acc, m) => {
    acc[m.model_name] = acc[m.model_name] || [];
    acc[m.model_name].push(m);
    return acc;
  }, {});

  return (
    <div style={{ padding: 20, fontFamily: "Arial" }}>
      <h1>OmniBioAI Model Registry (Explorer)</h1>

      {/* ACTION BAR */}
      <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
        <input
          placeholder="Search models..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ padding: 8, width: 300 }}
        />

        <button onClick={() => setShowRegister(true)}>➕ Register Model</button>
        <button onClick={load}>🔄 Refresh</button>
      </div>

      {/* LOADING */}
      {loading ? (
        <div>Loading...</div>
      ) : explorerModel ? (
        // =========================
        // EXPLORER VIEW (NEW STEP 13)
        // =========================
        <div>
          <button onClick={() => setExplorerModel(null)}>← Back</button>

          <h2>{explorerModel}</h2>

          {/* VERSION LIST */}
          <table border={1} cellPadding={10} width="100%">
            <thead>
              <tr>
                <th>Version</th>
                <th>Task</th>
                <th>Created</th>
              </tr>
            </thead>

            <tbody>
              {grouped[explorerModel]?.map((m) => (
                <tr key={m.version} onClick={() => setSelected(m)}>
                  <td>{m.version}</td>
                  <td>{m.task}</td>
                  <td>{m.created_at || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* LINEAGE GRAPH (TRUE MLflow STYLE) */}
          {grouped[explorerModel]?.[0] && (
            <ModelLineageView model={grouped[explorerModel][0]} />
          )}
        </div>
      ) : (
        // =========================
        // MAIN LIST VIEW
        // =========================
        <>
          <div style={{ marginBottom: 10 }}>
            Total: {models.length} | Filtered: {filtered.length}
          </div>

          <table border={1} cellPadding={10} width="100%">
            <thead>
              <tr>
                <th>Model</th>
                <th>Task</th>
                <th>Version</th>
                <th>Alias</th>
                <th>Created</th>
                <th>Open</th>
              </tr>
            </thead>

            <tbody>
              {Object.keys(grouped).map((name) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{grouped[name][0].task}</td>
                  <td>{grouped[name].length} versions</td>
                  <td>-</td>
                  <td>{grouped[name][0].created_at}</td>
                  <td>
                    <button onClick={() => setExplorerModel(name)}>
                      Explore
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {/* DRAWER */}
      {selected && (
        <ModelDrawer model={selected} onClose={() => setSelected(null)} />
      )}

      {/* REGISTER MODAL */}
      {showRegister && (
        <RegisterModelModal
          onClose={() => setShowRegister(false)}
          onSuccess={() => {
            setShowRegister(false);
            load();
          }}
        />
      )}
    </div>
  );
}