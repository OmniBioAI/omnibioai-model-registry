import { useState, type CSSProperties, type ReactNode } from "react";
import type { Model } from "../App";
import { relativeTime } from "../App";
import ModelLineageView from "./ModelLineageView";
import MetricsComparePanel from "./MetricsComparePanel";
import { setStage } from "../api/registry";

const BASE_URL = "/v1";

const ARTIFACT_FILES = [
  "model.pt",
  "model_genes.txt",
  "label_map.json",
  "model_meta.json",
  "metrics.json",
  "feature_schema.json",
  "sha256sums.txt",
];

export default function ModelDrawer({
  model,
  models,
  onClose,
  onRefresh,
}: {
  model: Model;
  models: Model[];
  onClose: () => void;
  onRefresh: () => void;
}) {
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState<{ label: string; status?: number; data?: any; error?: string } | null>(null);
  const [showLineage, setShowLineage] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);

  const sortedVersions = [...models].sort((a, b) =>
    (b.created_at ?? "").localeCompare(a.created_at ?? "")
  );
  const latestVersion = sortedVersions[0]?.version ?? model.version;

  async function callAPI(label: string, path: string, body?: unknown) {
    setLoading(true);
    setOutput(null);
    try {
      const res = await fetch(`${BASE_URL}${path}`, {
        method: body !== undefined ? "POST" : "GET",
        headers: body !== undefined ? { "Content-Type": "application/json" } : {},
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
      const data = await res.json();
      setOutput({ label, status: res.status, data });
    } catch (e) {
      setOutput({ label, error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  async function promote(version: string, alias: string) {
    await callAPI(`Promote → ${alias}`, `/promote`, {
      task: model.task,
      model_name: model.model_name,
      alias,
      version,
      actor: "registry-ui",
      reason: "promoted via ui",
    });
    onRefresh();
  }

  const isSuccess = output && !output.error && output.data?.ok !== false;
  const outputColor = output?.error || output?.data?.ok === false
    ? "var(--red)"
    : "var(--green)";

  return (
    <div style={overlayStyle}>
      {/* HEADER */}
      <div style={headerStyle}>
        <div>
          <div style={{ fontWeight: 700, fontSize: 15, color: "var(--text)" }}>
            {model.model_name}
          </div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
            {model.task}
          </div>
        </div>
        <button
          onClick={onClose}
          style={{ background: "transparent", border: "none", fontSize: 18, color: "var(--text-muted)", padding: "2px 6px" }}
        >
          ✕
        </button>
      </div>

      <div style={{ overflowY: "auto", flex: 1, padding: "0 20px 24px" }}>
        {/* ACTIONS */}
        <Section title="Actions">
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              disabled={loading}
              onClick={() =>
                callAPI(
                  "Resolve",
                  `/resolve?task=${encodeURIComponent(model.task)}&ref=${encodeURIComponent(
                    `${model.model_name}@${latestVersion}`
                  )}`
                )
              }
            >
              Resolve
            </button>
            <button
              disabled={loading}
              onClick={() =>
                callAPI("Verify", `/verify`, {
                  task: model.task,
                  ref: `${model.model_name}@${latestVersion}`,
                })
              }
            >
              Verify
            </button>
            <button
              onClick={() => setShowLineage((s) => !s)}
              style={showLineage ? { background: "var(--accent-bg)", color: "var(--accent)", borderColor: "var(--accent)" } : {}}
            >
              {showLineage ? "Hide Lineage" : "Lineage"}
            </button>
            <button
              onClick={() => setShowMetrics((s) => !s)}
              style={showMetrics ? { background: "var(--accent-bg)", color: "var(--accent)", borderColor: "var(--accent)" } : {}}
            >
              {showMetrics ? "Hide Metrics" : "Compare Metrics"}
            </button>
          </div>
        </Section>

        {/* API OUTPUT */}
        {(loading || output) && (
          <Section title={output ? `${output.label} — ${output.status ?? "error"}` : "Running…"}>
            <pre
              style={{
                background: "var(--surface-2)",
                border: `1px solid ${isSuccess ? "var(--border)" : "var(--red)"}`,
                borderRadius: 6,
                padding: 12,
                fontSize: 12,
                overflow: "auto",
                maxHeight: 200,
                color: loading ? "var(--text-muted)" : outputColor,
                lineHeight: 1.5,
              }}
            >
              {loading ? "…" : JSON.stringify(output, null, 2)}
            </pre>
          </Section>
        )}

        {/* VERSIONS */}
        <Section title={`Versions (${sortedVersions.length})`}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {sortedVersions.map((v) => (
              <div key={v.version} style={versionCardStyle}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
                  <div style={{ minWidth: 0 }}>
                    <code style={{ fontSize: 12, color: "var(--accent)", wordBreak: "break-all" }}>
                      {v.version}
                    </code>
                    {v.alias && (
                      <span style={drawerAliasBadge(v.alias)}>{v.alias}</span>
                    )}
                    <StageBadge stage={v.stage as string | undefined} />
                  </div>
                  <div style={{ display: "flex", gap: 6, flexShrink: 0, alignItems: "center" }}>
                    <select
                      value={(v.stage as string | undefined) ?? "none"}
                      style={{ fontSize: 11, padding: "2px 4px", background: "var(--surface)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: 4 }}
                      onChange={(e) => {
                        void setStage(model.task, model.model_name, v.version, e.target.value, "registry-ui").then(() => onRefresh());
                      }}
                    >
                      <option value="none">none</option>
                      <option value="staging">staging</option>
                      <option value="production">production</option>
                      <option value="archived">archived</option>
                    </select>
                    <button
                      style={{ fontSize: 11, padding: "3px 8px" }}
                      disabled={loading}
                      onClick={() => promote(v.version, "latest")}
                      title="Set as latest alias"
                    >
                      → latest
                    </button>
                    <button
                      style={{ fontSize: 11, padding: "3px 8px" }}
                      disabled={loading}
                      onClick={() => promote(v.version, "production")}
                      title="Set as production alias"
                    >
                      → production
                    </button>
                  </div>
                </div>
                {v.created_at && (
                  <div
                    style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 6 }}
                    title={v.created_at}
                  >
                    {relativeTime(v.created_at)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>

        {/* EXPECTED ARTIFACTS */}
        <Section title="Expected Artifacts">
          <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            {ARTIFACT_FILES.map((f) => (
              <div key={f} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13 }}>
                <span style={{ color: "var(--green)", fontSize: 9 }}>●</span>
                <code style={{ color: "var(--text-muted)", fontSize: 12 }}>{f}</code>
              </div>
            ))}
          </div>
        </Section>

        {/* LINEAGE */}
        {showLineage && (
          <Section title="Lineage Graph">
            <ModelLineageView model={model} />
          </Section>
        )}

        {/* METRICS COMPARISON */}
        {showMetrics && (
          <Section title="Metrics Comparison">
            <MetricsComparePanel
              task={model.task}
              model={model.model_name}
              versions={sortedVersions.map((v) => v.version)}
            />
          </Section>
        )}
      </div>
    </div>
  );
}

function StageBadge({ stage }: { stage?: string }) {
  if (!stage || stage === "none") return null;
  const colors: Record<string, { bg: string; fg: string }> = {
    staging:    { bg: "var(--teal-dim)",   fg: "var(--teal)" },
    production: { bg: "var(--blue-dim)",   fg: "var(--blue)" },
    archived:   { bg: "var(--surface-2)",  fg: "var(--text-muted)" },
  };
  const c = colors[stage] ?? { bg: "var(--surface-2)", fg: "var(--text-muted)" };
  return (
    <span
      style={{
        display: "inline-block",
        background: c.bg,
        color: c.fg,
        padding: "1px 7px",
        borderRadius: 10,
        fontSize: 11,
        marginLeft: 8,
        fontWeight: 500,
        textTransform: "capitalize",
      }}
    >
      {stage}
    </span>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div style={{ marginTop: 20 }}>
      <div
        style={{
          fontSize: 11,
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "var(--text-muted)",
          marginBottom: 8,
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

const overlayStyle: CSSProperties = {
  position: "fixed",
  right: 0,
  top: 0,
  width: "42%",
  minWidth: 380,
  height: "100%",
  background: "var(--surface)",
  borderLeft: "1px solid var(--border)",
  display: "flex",
  flexDirection: "column",
  zIndex: 100,
  boxShadow: "-4px 0 32px rgba(0,0,0,0.45)",
};

const headerStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  padding: "16px 20px",
  borderBottom: "1px solid var(--border)",
  flexShrink: 0,
};

const versionCardStyle: CSSProperties = {
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  padding: "10px 12px",
};

function drawerAliasBadge(alias: string): CSSProperties {
  const isProduction = alias === "production";
  return {
    display: "inline-block",
    background: isProduction ? "var(--blue-dim)" : "var(--teal-dim)",
    color: isProduction ? "var(--blue)" : "var(--teal)",
    padding: "1px 7px",
    borderRadius: 10,
    fontSize: 11,
    marginLeft: 8,
    fontWeight: 500,
  };
}
