import { useEffect, useState, useMemo, type CSSProperties } from "react";
import { fetchModels, fetchAliases } from "./api/registry";
import ModelDrawer from "./components/ModelDrawer";
import RegisterModelModal from "./components/RegisterModelModal";
import ModelLineageView from "./components/ModelLineageView";
import ExperimentsView from "./components/ExperimentsView";
import HFModelsGallery from "./components/HFModelsGallery";

export type Model = {
  task: string;
  model_name: string;
  version: string;
  alias?: string;
  created_at?: string;
  lineage?: any;
  [key: string]: any;
};

type SortKey = "model_name" | "task" | "versions" | "created_at";
type SortDir = "asc" | "desc";

export function relativeTime(iso?: string): string {
  if (!iso) return "—";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  if (months < 12) return `${months}mo ago`;
  return `${Math.floor(months / 12)}y ago`;
}

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState<Model | null>(null);
  const [showRegister, setShowRegister] = useState(false);
  const [explorerModel, setExplorerModel] = useState<string | null>(null);
  const [activeTask, setActiveTask] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("model_name");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  // aliasMap: "task:model_name" → ["latest", "production", ...]
  const [aliasMap, setAliasMap] = useState<Record<string, string[]>>({});
  const [activeTab, setActiveTab] = useState<'models' | 'experiments' | 'hf-models'>('models');

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    setAliasMap({});
    try {
      const data = await fetchModels();
      const normalized: Model[] = Array.isArray(data) ? data : (data.models ?? []);
      setModels(normalized);

      const pairs = new Map<string, { task: string; model_name: string }>();
      for (const m of normalized) {
        const key = `${m.task}:${m.model_name}`;
        if (!pairs.has(key)) pairs.set(key, { task: m.task, model_name: m.model_name });
      }

      const aliasResults = await Promise.all(
        Array.from(pairs.entries()).map(async ([key, { task, model_name }]) => {
          try {
            const res = await fetchAliases(task, model_name);
            return { key, aliases: res.aliases.map((a) => a.alias) };
          } catch {
            return { key, aliases: [] as string[] };
          }
        })
      );

      const newMap: Record<string, string[]> = {};
      for (const { key, aliases } of aliasResults) {
        if (aliases.length > 0) newMap[key] = aliases;
      }
      setAliasMap(newMap);
    } finally {
      setLoading(false);
    }
  }

  // Group ALL models by model_name — unfiltered so explorer always shows full history
  const allGrouped = useMemo(() => {
    const g: Record<string, Model[]> = {};
    for (const m of models) {
      if (!g[m.model_name]) g[m.model_name] = [];
      g[m.model_name].push(m);
    }
    return g;
  }, [models]);

  // Sidebar: unique tasks with count of distinct model_names per task
  const taskCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const [, versions] of Object.entries(allGrouped)) {
      const task = versions[0].task;
      counts[task] = (counts[task] ?? 0) + 1;
    }
    return counts;
  }, [allGrouped]);

  // Filtered + sorted groups for the main table display
  const displayGroups = useMemo(() => {
    let groups = Object.entries(allGrouped);

    if (activeTask) {
      groups = groups.filter(([, versions]) => versions[0].task === activeTask);
    }

    if (search.trim()) {
      const q = search.toLowerCase();
      groups = groups.filter(
        ([name, versions]) =>
          name.toLowerCase().includes(q) ||
          versions[0].task.toLowerCase().includes(q) ||
          versions.some((v) => v.version.toLowerCase().includes(q))
      );
    }

    groups.sort(([nameA, versionsA], [nameB, versionsB]) => {
      let a: string | number = "";
      let b: string | number = "";
      if (sortKey === "model_name") { a = nameA; b = nameB; }
      else if (sortKey === "task") { a = versionsA[0].task; b = versionsB[0].task; }
      else if (sortKey === "versions") { a = versionsA.length; b = versionsB.length; }
      else if (sortKey === "created_at") {
        a = versionsA[0].created_at ?? "";
        b = versionsB[0].created_at ?? "";
      }
      if (a < b) return sortDir === "asc" ? -1 : 1;
      if (a > b) return sortDir === "asc" ? 1 : -1;
      return 0;
    });

    return groups;
  }, [allGrouped, activeTask, search, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("asc"); }
  }

  const totalModels = Object.keys(allGrouped).length;
  const totalVersions = models.length;
  const totalTasks = Object.keys(taskCounts).length;

  // Explorer: always from full ungrouped data, newest first
  const explorerVersions = explorerModel
    ? [...(allGrouped[explorerModel] ?? [])].sort((a, b) =>
        (b.created_at ?? "").localeCompare(a.created_at ?? "")
      )
    : [];

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
      {/* ── TOP HEADER ── */}
      <header style={hdrStyle}>
        <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: "-0.01em" }}>
          🧬 OmniBioAI ModelHub
        </span>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => setShowRegister(true)}
            style={{ background: "var(--accent)", color: "#fff", border: "none", fontWeight: 600 }}
          >
            + Register Model
          </button>
          <button onClick={load}>↺ Refresh</button>
        </div>
      </header>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* ── LEFT SIDEBAR ── */}
        <aside style={sidebarStyle}>
          <div style={sidebarTitleStyle}>Tasks</div>
          <SidebarItem
            label="All tasks"
            count={totalModels}
            active={activeTask === null}
            onClick={() => setActiveTask(null)}
          />
          {Object.entries(taskCounts)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([task, count]) => (
              <SidebarItem
                key={task}
                label={task}
                count={count}
                active={activeTask === task}
                onClick={() => setActiveTask(task)}
              />
            ))}
        </aside>

        {/* ── MAIN CONTENT ── flex-column so table can fill remaining height */}
        <main style={mainStyle}>
          {/* TAB BAR */}
          <div style={{ display: "flex", gap: 0, marginBottom: 20, borderBottom: "1px solid var(--border)", flexShrink: 0 }}>
            {(["models", "experiments", "hf-models"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  background: "transparent",
                  border: "none",
                  borderBottom: activeTab === tab ? "2px solid var(--accent)" : "2px solid transparent",
                  color: activeTab === tab ? "var(--accent)" : "var(--text-muted)",
                  fontWeight: activeTab === tab ? 700 : 400,
                  fontSize: 14,
                  padding: "6px 16px 10px",
                  cursor: "pointer",
                  borderRadius: 0,
                  textTransform: "capitalize",
                }}
              >
                {tab === "hf-models" ? "🤗 HF Models" : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {activeTab === "experiments" ? (
            <ExperimentsView
              selectedTask={activeTask}
              onOpenModel={(task, modelName) => {
                const m = models.find((x) => x.task === task && x.model_name === modelName);
                if (m) { setActiveTab("models"); setSelected(m); }
              }}
            />
          ) : activeTab === "hf-models" ? (
            <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
              <HFModelsGallery />
            </div>
          ) : loading ? (
            <div style={{ color: "var(--text-muted)", textAlign: "center", paddingTop: 60 }}>
              Loading…
            </div>
          ) : explorerModel ? (
            /* ── EXPLORER VIEW — scrollable container ── */
            <div style={{ flex: 1, minHeight: 0, overflowY: "auto" }}>
              <button onClick={() => setExplorerModel(null)} style={{ marginBottom: 16 }}>
                ← Back to all models
              </button>
              <h2 style={{ marginBottom: 4 }}>{explorerModel}</h2>
              <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 16 }}>
                Task: {explorerVersions[0]?.task} · {explorerVersions.length} version
                {explorerVersions.length !== 1 ? "s" : ""}
              </div>

              <div style={tableWrapperStyle}>
                <table style={tableStyle}>
                  <thead>
                    <tr style={stickyHeaderStyle}>
                      <th style={thStyle}>Version</th>
                      <th style={thStyle}>Task</th>
                      <th style={thStyle}>Created</th>
                      <th style={thStyle}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {explorerVersions.map((m) => (
                      <tr
                        key={m.version}
                        style={{ cursor: "pointer" }}
                        onClick={() => setSelected(m)}
                      >
                        <td style={{ ...tdStyle, whiteSpace: "nowrap" }}>
                          <code style={codeChip}>{m.version}</code>
                        </td>
                        <td style={tdStyle}>
                          <span style={taskBadgeStyle}>{m.task}</span>
                        </td>
                        <td style={tdStyle} title={m.created_at}>
                          {relativeTime(m.created_at)}
                        </td>
                        <td style={tdStyle}>
                          <button
                            style={{ fontSize: 12 }}
                            onClick={(e) => { e.stopPropagation(); setSelected(m); }}
                          >
                            Details
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {explorerVersions[0] && (
                <ModelLineageView model={explorerVersions[0]} />
              )}
            </div>
          ) : (
            /* ── MAIN LIST VIEW — fills remaining height ── */
            <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
              {/* STATS BAR */}
              <div style={statsGridStyle}>
                <StatCard label="Total Models" value={totalModels} />
                <StatCard label="Total Versions" value={totalVersions} />
                <StatCard label="Tasks" value={totalTasks} />
                <StatCard label="Showing" value={displayGroups.length} />
              </div>

              {/* SEARCH */}
              <div style={{ marginBottom: 16 }}>
                <input
                  placeholder="Search models, tasks, versions…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  style={{ width: 340 }}
                />
              </div>

              {/* TABLE WRAPPER — grows to fill remaining viewport height */}
              <div style={{ ...tableWrapperStyle, flex: 1, minHeight: 0 }}>
                <table style={tableStyle}>
                  <thead>
                    <tr style={stickyHeaderStyle}>
                      <SortableTh label="Model" sortKey="model_name" current={sortKey} dir={sortDir} onSort={toggleSort} />
                      <SortableTh label="Task" sortKey="task" current={sortKey} dir={sortDir} onSort={toggleSort} />
                      <SortableTh label="Versions" sortKey="versions" current={sortKey} dir={sortDir} onSort={toggleSort} />
                      <th style={thStyle}>Latest Version</th>
                      <th style={thStyle}>Alias</th>
                      <SortableTh label="Created" sortKey="created_at" current={sortKey} dir={sortDir} onSort={toggleSort} />
                      <th style={thStyle}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayGroups.map(([name, versions]) => {
                      const sorted = [...versions].sort((a, b) =>
                        (b.created_at ?? "").localeCompare(a.created_at ?? "")
                      );
                      const latest = sorted[0];
                      // Aliases come from the alias probe, not the API response
                      const aliases = aliasMap[`${latest.task}:${name}`] ?? [];

                      return (
                        <tr
                          key={name}
                          style={{ cursor: "pointer" }}
                          onClick={() => setSelected(latest)}
                        >
                          <td style={tdStyle}>
                            <b style={{ color: "var(--text)" }}>{name}</b>
                          </td>
                          <td style={tdStyle}>
                            <span style={taskBadgeStyle}>{latest.task}</span>
                          </td>
                          <td style={{ ...tdStyle, color: "var(--text-muted)" }}>
                            {versions.length}
                          </td>
                          <td style={{ ...tdStyle, whiteSpace: "nowrap" }}>
                            <code style={codeChip}>{latest.version}</code>
                          </td>
                          <td style={tdStyle}>
                            {aliases.length > 0 ? (
                              aliases.map((a) => (
                                <span key={a} style={aliasBadgeStyle(a)}>{a}</span>
                              ))
                            ) : (
                              <span style={{ color: "var(--text-muted)" }}>—</span>
                            )}
                          </td>
                          <td style={tdStyle} title={latest.created_at}>
                            {relativeTime(latest.created_at)}
                          </td>
                          <td style={tdStyle} onClick={(e) => e.stopPropagation()}>
                            <button
                              style={{ fontSize: 12 }}
                              onClick={() => setExplorerModel(name)}
                            >
                              Explore
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>

                {displayGroups.length === 0 && (
                  <div style={{ textAlign: "center", color: "var(--text-muted)", padding: 60, fontSize: 14 }}>
                    {search || activeTask
                      ? "No models match the current filter."
                      : "No models registered yet. Click + Register Model to add one."}
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>

      {selected && (
        <ModelDrawer
          model={selected}
          models={allGrouped[selected.model_name] ?? []}
          onClose={() => setSelected(null)}
          onRefresh={load}
        />
      )}

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

// ── Sub-components ────────────────────────────────────────────

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div style={statCardStyle}>
      <div style={{ fontSize: 30, fontWeight: 700, lineHeight: 1 }}>{value}</div>
      <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 4 }}>{label}</div>
    </div>
  );
}

function SidebarItem({
  label,
  count,
  active,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        width: "100%",
        padding: "7px 16px",
        border: "none",
        borderRadius: 0,
        background: active ? "var(--accent-bg)" : "transparent",
        color: active ? "var(--accent)" : "var(--text-muted)",
        fontSize: 13,
        textAlign: "left",
        cursor: "pointer",
        fontWeight: active ? 600 : 400,
      }}
    >
      <span
        style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
        title={label}
      >
        {label}
      </span>
      <span
        style={{
          background: "var(--surface-2)",
          borderRadius: 10,
          padding: "1px 7px",
          fontSize: 11,
          color: "var(--text-muted)",
          flexShrink: 0,
          marginLeft: 6,
        }}
      >
        {count}
      </span>
    </button>
  );
}

function SortableTh({
  label,
  sortKey,
  current,
  dir,
  onSort,
}: {
  label: string;
  sortKey: SortKey;
  current: SortKey;
  dir: SortDir;
  onSort: (k: SortKey) => void;
}) {
  const active = current === sortKey;
  return (
    <th
      style={{ ...thStyle, cursor: "pointer", userSelect: "none" }}
      onClick={() => onSort(sortKey)}
    >
      {label}{" "}
      {active ? (dir === "asc" ? "↑" : "↓") : <span style={{ opacity: 0.3 }}>↕</span>}
    </th>
  );
}

// ── Styles ────────────────────────────────────────────────────

const hdrStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "10px 24px",
  borderBottom: "1px solid var(--border)",
  background: "var(--surface)",
  flexShrink: 0,
};

const sidebarStyle: CSSProperties = {
  width: 220,
  background: "var(--surface)",
  borderRight: "1px solid var(--border)",
  padding: "16px 0",
  overflowY: "auto",
  flexShrink: 0,
};

const sidebarTitleStyle: CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  color: "var(--text-muted)",
  padding: "0 16px 8px",
};

// overflow: hidden + flex-column so children can use flex: 1 to fill remaining space
const mainStyle: CSSProperties = {
  flex: 1,
  overflow: "hidden",
  display: "flex",
  flexDirection: "column",
  padding: "20px 24px",
};

const statsGridStyle: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(4, 1fr)",
  gap: 12,
  marginBottom: 20,
  flexShrink: 0,
};

const statCardStyle: CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  padding: "16px 20px",
};

// The scroll container that wraps <table> — this element fills remaining height
const tableWrapperStyle: CSSProperties = {
  overflowY: "auto",
  border: "1px solid var(--border)",
  borderRadius: 8,
  background: "var(--surface)",
};

// table itself has no border/radius — those live on the wrapper
const tableStyle: CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
};

// thead row: sticky so it stays visible when the wrapper scrolls
const stickyHeaderStyle: CSSProperties = {
  position: "sticky",
  top: 0,
  zIndex: 1,
  background: "var(--surface-2)",
};

const thStyle: CSSProperties = {
  padding: "9px 14px",
  textAlign: "left",
  fontSize: 12,
  fontWeight: 600,
  color: "var(--text-muted)",
  borderBottom: "1px solid var(--border)",
  whiteSpace: "nowrap",
};

const tdStyle: CSSProperties = {
  padding: "10px 14px",
  fontSize: 13,
  borderBottom: "1px solid var(--border)",
  color: "var(--text)",
};

const codeChip: CSSProperties = {
  fontFamily: "ui-monospace, 'Cascadia Code', Consolas, 'Liberation Mono', monospace",
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  padding: "2px 7px",
  borderRadius: 4,
  fontSize: 12,
  color: "var(--text-muted)",
  whiteSpace: "nowrap",
  display: "inline-block",
};

const taskBadgeStyle: CSSProperties = {
  background: "var(--accent-bg)",
  color: "var(--accent)",
  padding: "2px 8px",
  borderRadius: 10,
  fontSize: 12,
  fontWeight: 500,
};

function aliasBadgeStyle(alias: string): CSSProperties {
  const colors: Record<string, { bg: string; fg: string }> = {
    latest:     { bg: "#3D3D3D", fg: "#9CA3AF" },
    staging:    { bg: "#1E3A5F", fg: "#60A5FA" },
    production: { bg: "#14532D", fg: "#4ADE80" },
    archived:   { bg: "#78350F", fg: "#FCD34D" },
  };
  const c = colors[alias] ?? { bg: "#3D3D3D", fg: "#9CA3AF" };
  return {
    background: c.bg,
    color: c.fg,
    fontSize: 11,
    padding: "2px 8px",
    borderRadius: 4,
    fontWeight: 500,
    marginRight: 4,
    display: "inline-block",
  };
}
