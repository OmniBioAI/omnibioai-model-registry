import { useEffect, useState, type CSSProperties } from "react";
import { compareVersions, type CompareResponse } from "../api/registry";

export default function MetricsComparePanel({
  task,
  model,
  versions,
}: {
  task: string;
  model: string;
  versions: string[];
}) {
  const [data, setData] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // versions.join is stable across renders when contents haven't changed
  const versionKey = versions.join(",");

  useEffect(() => {
    if (versions.length < 2) {
      setError("At least 2 versions needed to compare.");
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setData(null);
    compareVersions(task, model, versions)
      .then((res) => {
        if (!cancelled) { setData(res); setLoading(false); }
      })
      .catch((e: unknown) => {
        if (!cancelled) { setError(String(e)); setLoading(false); }
      });
    return () => { cancelled = true; };
    // versionKey used instead of versions array to avoid re-fetching on identity change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [task, model, versionKey]);

  if (loading) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 8, padding: "8px 0" }}>
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            style={{ height: 24, background: "var(--surface-2)", borderRadius: 4, opacity: 0.6 }}
          />
        ))}
      </div>
    );
  }

  if (error) {
    return <div style={{ color: "var(--red)", fontSize: 13 }}>{error}</div>;
  }

  if (!data) return null;

  const versionData = data.versions;
  const orderedVersions = versions.filter((v) => v in versionData);

  // Union of all metric keys, sorted alphabetically
  const allKeys = Array.from(
    new Set(orderedVersions.flatMap((v) => Object.keys(versionData[v]?.metrics ?? {})))
  ).sort();

  function bestIndex(key: string): number {
    const isLoss = key.toLowerCase().includes("loss");
    let bestIdx = -1;
    let bestVal = isLoss ? Infinity : -Infinity;
    orderedVersions.forEach((v, i) => {
      const val = versionData[v]?.metrics?.[key];
      if (val == null) return;
      if (isLoss ? val < bestVal : val > bestVal) {
        bestVal = val;
        bestIdx = i;
      }
    });
    return bestIdx;
  }

  function truncate(s: string, n = 16): string {
    return s.length > n ? s.slice(0, n) + "…" : s;
  }

  if (allKeys.length === 0) {
    return (
      <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
        No metrics found for these versions.
      </div>
    );
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={thStyle}>Metric</th>
            {orderedVersions.map((v) => (
              <th key={v} style={thStyle} title={v}>
                {truncate(v)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {allKeys.map((key) => {
            const best = bestIndex(key);
            return (
              <tr key={key}>
                <td style={keyTdStyle}>{key}</td>
                {orderedVersions.map((v, i) => {
                  const val = versionData[v]?.metrics?.[key];
                  const isBest = i === best;
                  return (
                    <td
                      key={v}
                      style={{
                        ...valTdStyle,
                        color: isBest ? "var(--green)" : "var(--text)",
                        fontWeight: isBest ? 600 : 400,
                      }}
                    >
                      {val != null
                        ? typeof val === "number"
                          ? val.toPrecision(4)
                          : String(val)
                        : "—"}
                    </td>
                  );
                })}
              </tr>
            );
          })}
          {/* Footer: created_at per version */}
          <tr style={{ borderTop: "2px solid var(--border)" }}>
            <td style={keyTdStyle}>created</td>
            {orderedVersions.map((v) => {
              const createdAt = versionData[v]?.created_at;
              return (
                <td
                  key={v}
                  style={{ ...valTdStyle, color: "var(--text-muted)", fontSize: 11 }}
                >
                  {createdAt ? new Date(createdAt).toLocaleDateString() : "—"}
                </td>
              );
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

const tableStyle: CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: 12,
};

const thStyle: CSSProperties = {
  padding: "6px 12px",
  textAlign: "left",
  fontSize: 11,
  fontWeight: 600,
  color: "var(--text-muted)",
  borderBottom: "1px solid var(--border)",
  whiteSpace: "nowrap",
};

const keyTdStyle: CSSProperties = {
  padding: "6px 12px",
  fontSize: 12,
  color: "var(--text-muted)",
  borderBottom: "1px solid var(--border)",
  fontFamily: "ui-monospace, 'Cascadia Code', Consolas, monospace",
  whiteSpace: "nowrap",
};

const valTdStyle: CSSProperties = {
  padding: "6px 12px",
  fontSize: 12,
  borderBottom: "1px solid var(--border)",
  textAlign: "right",
};
