import { useEffect, useState, useMemo, type CSSProperties } from 'react';
import { fetchModels, fetchRuns, fetchRunDetail, fetchAliases, type RunSummary, type RunDetail } from '../api/registry';
import RunMetricChart from './RunMetricChart';

type Props = {
  selectedTask: string | null;
  onOpenModel?: (task: string, modelName: string) => void;
};

type LineageEntry = { task: string; model_name: string };

export default function ExperimentsView({ selectedTask, onOpenModel }: Props) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [details, setDetails] = useState<Record<string, RunDetail>>({});
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'running' | 'finished' | 'failed'>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lineageMap, setLineageMap] = useState<Record<string, LineageEntry>>({});
  const [registrationInfo, setRegistrationInfo] = useState<Record<string, string>>({});
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    load();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTask]);

  async function load() {
    setLoading(true);
    setError(null);
    setRuns([]);
    setDetails({});
    setExpandedId(null);
    try {
      const raw = await fetchModels();
      const modelList: Array<{ task: string; model_name: string; lineage?: { run_id?: string } }> =
        Array.isArray(raw) ? raw : (raw.models ?? []);

      // Build lineage reverse map: run_id → {task, model_name}
      const lMap: Record<string, LineageEntry> = {};
      for (const m of modelList) {
        const rid = m.lineage?.run_id;
        if (rid) lMap[rid] = { task: m.task, model_name: m.model_name };
      }
      setLineageMap(lMap);

      // Unique (task, model_name) pairs filtered by selectedTask
      const pairs = new Map<string, { task: string; model: string }>();
      for (const m of modelList) {
        if (selectedTask && m.task !== selectedTask) continue;
        const key = `${m.task}:${m.model_name}`;
        if (!pairs.has(key)) pairs.set(key, { task: m.task, model: m.model_name });
      }

      const results = await Promise.all(
        Array.from(pairs.values()).map(({ task, model }) =>
          fetchRuns(task, model).catch(() => [] as RunSummary[])
        )
      );
      setRuns(results.flat());
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function expandRow(run: RunSummary) {
    if (expandedId === run.run_id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(run.run_id);

    if (!details[run.run_id]) {
      try {
        const detail = await fetchRunDetail(run.task ?? '', run.model_name ?? '', run.run_id);
        setDetails((d) => ({ ...d, [run.run_id]: detail }));
      } catch {
        // fallback: show summary data only (no charts)
      }
    }

    // Lazy-fetch alias for "Registered as" chip
    const linked = lineageMap[run.run_id];
    if (linked && !registrationInfo[run.run_id]) {
      try {
        const aliasRes = await fetchAliases(linked.task, linked.model_name);
        const alias = aliasRes.aliases[0]?.alias ?? 'latest';
        setRegistrationInfo((r) => ({
          ...r,
          [run.run_id]: `${linked.task} / ${linked.model_name} @ ${alias}`,
        }));
      } catch {
        setRegistrationInfo((r) => ({
          ...r,
          [run.run_id]: `${linked.task} / ${linked.model_name}`,
        }));
      }
    }
  }

  function copyId(id: string) {
    navigator.clipboard.writeText(id).then(() => {
      setCopied(id);
      setTimeout(() => setCopied(null), 1500);
    });
  }

  const filtered = useMemo(() => {
    let list = runs;
    if (statusFilter !== 'all') list = list.filter((r) => r.status === statusFilter);
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        (r) =>
          r.run_id.toLowerCase().includes(q) ||
          (r.model_name ?? '').toLowerCase().includes(q) ||
          (r.task ?? '').toLowerCase().includes(q) ||
          Object.entries(r.params).some(
            ([k, v]) => k.toLowerCase().includes(q) || String(v).toLowerCase().includes(q)
          )
      );
    }
    return list;
  }, [runs, search, statusFilter]);

  return (
    <>
      <style>{`
        @keyframes omr-pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }
        .omr-pulse { animation: omr-pulse 1.5s ease-in-out infinite; }
        @keyframes omr-skeleton { 0%{opacity:0.4} 50%{opacity:0.7} 100%{opacity:0.4} }
        .omr-skeleton { animation: omr-skeleton 1.4s ease-in-out infinite; }
      `}</style>

      {/* Toolbar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 14, alignItems: 'center' }}>
        <input
          placeholder="Search runs, models, params…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ width: 280 }}
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as typeof statusFilter)}
          style={{ background: 'var(--surface)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: 4, padding: '4px 8px', fontSize: 13 }}
        >
          <option value="all">All statuses</option>
          <option value="running">Running</option>
          <option value="finished">Finished</option>
          <option value="failed">Failed</option>
        </select>
        <button onClick={load} style={{ marginLeft: 'auto' }}>↺ Refresh</button>
      </div>

      {/* Error */}
      {error && (
        <div style={{ color: 'var(--red)', fontSize: 13, marginBottom: 12, display: 'flex', gap: 10, alignItems: 'center' }}>
          <span>{error}</span>
          <button onClick={load} style={{ fontSize: 12 }}>Retry</button>
        </div>
      )}

      {/* Table */}
      <div style={{ overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 8, background: 'var(--surface)', flex: 1, minHeight: 0 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ position: 'sticky', top: 0, zIndex: 1, background: 'var(--surface-2)' }}>
              <th style={thS}>Run ID</th>
              <th style={thS}>Model</th>
              <th style={thS}>Status</th>
              <th style={thS}>Key Params</th>
              <th style={thS}>Metrics</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              [0, 1, 2].map((i) => (
                <tr key={i}>
                  {[0, 1, 2, 3, 4].map((j) => (
                    <td key={j} style={tdS}>
                      <div className="omr-skeleton" style={{ height: 16, borderRadius: 4, background: 'var(--surface-2)' }} />
                    </td>
                  ))}
                </tr>
              ))
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ textAlign: 'center', padding: 60, color: 'var(--text-muted)', fontSize: 13 }}>
                  {runs.length === 0
                    ? 'No runs recorded yet. Use RunLogger or PluginRunClient to log training runs from your plugins.'
                    : 'No runs match the current filter.'}
                </td>
              </tr>
            ) : (
              filtered.map((run) => {
                const isExpanded = expandedId === run.run_id;
                const detail = details[run.run_id];
                const paramEntries = Object.entries(run.params);
                const metricEntries = Object.entries(run.metrics_summary);

                return [
                  <tr
                    key={run.run_id}
                    onClick={() => expandRow(run)}
                    style={{ cursor: 'pointer', background: isExpanded ? 'var(--accent-bg)' : undefined }}
                  >
                    {/* Run ID */}
                    <td style={tdS}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <code style={monoChip}>{run.run_id.slice(0, 8)}</code>
                        <button
                          onClick={(e) => { e.stopPropagation(); copyId(run.run_id); }}
                          title="Copy full run ID"
                          style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 12, padding: '0 2px', lineHeight: 1 }}
                        >
                          {copied === run.run_id ? '✓' : '⎘'}
                        </button>
                      </div>
                    </td>

                    {/* Model */}
                    <td style={tdS}>
                      <button
                        onClick={(e) => { e.stopPropagation(); onOpenModel?.(run.task ?? '', run.model_name ?? ''); }}
                        style={{ background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid transparent', borderRadius: 10, padding: '2px 8px', fontSize: 12, fontWeight: 500, cursor: 'pointer' }}
                        title={`${run.task} / ${run.model_name}`}
                      >
                        {run.model_name ?? '—'}
                      </button>
                    </td>

                    {/* Status */}
                    <td style={tdS}>
                      <span style={statusBadge(run.status)}>
                        <span className={run.status === 'running' ? 'omr-pulse' : undefined} style={statusDot(run.status)} />
                        {run.status}
                      </span>
                    </td>

                    {/* Key Params */}
                    <td style={tdS}>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {paramEntries.slice(0, 2).map(([k, v]) => (
                          <span key={k} style={paramPill}>
                            {k}={String(v)}
                          </span>
                        ))}
                        {paramEntries.length > 2 && (
                          <span style={{ ...paramPill, color: 'var(--text-muted)' }}>
                            +{paramEntries.length - 2} more
                          </span>
                        )}
                        {paramEntries.length === 0 && <span style={{ color: 'var(--text-muted)' }}>—</span>}
                      </div>
                    </td>

                    {/* Metrics */}
                    <td style={tdS}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        {metricEntries.slice(0, 2).map(([k, v]) => (
                          <span key={k} style={{ fontSize: 12 }}>
                            <span style={{ color: 'var(--text-muted)' }}>{k}:</span>{' '}
                            <span style={{ color: '#4ADE80', fontWeight: 500 }}>{typeof v === 'number' ? v.toPrecision(4) : v}</span>
                          </span>
                        ))}
                        {metricEntries.length === 0 && <span style={{ color: 'var(--text-muted)' }}>—</span>}
                      </div>
                    </td>
                  </tr>,

                  // Expanded panel
                  isExpanded && (
                    <tr key={`${run.run_id}-exp`}>
                      <td colSpan={5} style={{ padding: 0, background: 'var(--surface-2)', borderBottom: '1px solid var(--border)' }}>
                        <div style={{ display: 'flex', padding: '16px 20px', gap: 24 }}>
                          {/* Left: params + tags + registration */}
                          <div style={{ width: '40%', minWidth: 0 }}>
                            <SectionLabel>Params</SectionLabel>
                            {paramEntries.length === 0 ? (
                              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>—</div>
                            ) : (
                              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                                <tbody>
                                  {paramEntries.map(([k, v]) => (
                                    <tr key={k}>
                                      <td style={kvKey}>{k}</td>
                                      <td style={kvVal}>{String(v)}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            )}

                            {Object.keys(run.tags).length > 0 && (
                              <>
                                <SectionLabel style={{ marginTop: 12 }}>Tags</SectionLabel>
                                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                                  <tbody>
                                    {Object.entries(run.tags).map(([k, v]) => (
                                      <tr key={k}>
                                        <td style={kvKey}>{k}</td>
                                        <td style={kvVal}>{v}</td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </>
                            )}

                            {registrationInfo[run.run_id] && (
                              <>
                                <SectionLabel style={{ marginTop: 12 }}>Registered as</SectionLabel>
                                <span style={{ background: '#0D3D3D', color: '#2DD4BF', fontSize: 12, padding: '3px 10px', borderRadius: 4, fontWeight: 500, display: 'inline-block' }}>
                                  {registrationInfo[run.run_id]}
                                </span>
                              </>
                            )}
                            {lineageMap[run.run_id] && !registrationInfo[run.run_id] && (
                              <>
                                <SectionLabel style={{ marginTop: 12 }}>Registered as</SectionLabel>
                                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading…</span>
                              </>
                            )}
                          </div>

                          {/* Right: metric charts */}
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <SectionLabel>Metric History</SectionLabel>
                            {!detail ? (
                              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading charts…</div>
                            ) : Object.keys(detail.metric_history).length === 0 ? (
                              <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No metric history available.</div>
                            ) : (
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
                                {Object.entries(detail.metric_history).map(([key, pts]) => (
                                  <div key={key} style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, padding: '10px 12px' }}>
                                    <RunMetricChart
                                      data={pts}
                                      label={key}
                                      width={180}
                                      height={60}
                                    />
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  ),
                ];
              })
            )}
          </tbody>
        </table>
      </div>
    </>
  );
}

function SectionLabel({ children, style }: { children: React.ReactNode; style?: CSSProperties }) {
  return (
    <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 6, ...style }}>
      {children}
    </div>
  );
}

function statusDot(status: string): CSSProperties {
  const colors: Record<string, string> = {
    running: '#FCD34D',
    finished: '#4ADE80',
    failed: '#F87171',
  };
  return {
    display: 'inline-block',
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: colors[status] ?? '#9CA3AF',
    flexShrink: 0,
  };
}

function statusBadge(status: string): CSSProperties {
  const map: Record<string, { background: string; color: string }> = {
    running:  { background: '#78350F', color: '#FCD34D' },
    finished: { background: '#14532D', color: '#4ADE80' },
    failed:   { background: '#4C1D1D', color: '#F87171' },
  };
  const c = map[status] ?? { background: '#3D3D3D', color: '#9CA3AF' };
  return {
    ...c,
    display: 'inline-flex',
    alignItems: 'center',
    gap: 5,
    fontSize: 11,
    padding: '2px 8px',
    borderRadius: 4,
    fontWeight: 500,
    whiteSpace: 'nowrap',
  };
}

const thS: CSSProperties = {
  padding: '9px 14px',
  textAlign: 'left',
  fontSize: 12,
  fontWeight: 600,
  color: 'var(--text-muted)',
  borderBottom: '1px solid var(--border)',
  whiteSpace: 'nowrap',
};

const tdS: CSSProperties = {
  padding: '10px 14px',
  fontSize: 13,
  borderBottom: '1px solid var(--border)',
  color: 'var(--text)',
  verticalAlign: 'middle',
};

const monoChip: CSSProperties = {
  fontFamily: "ui-monospace, 'Cascadia Code', Consolas, monospace",
  background: 'var(--surface-2)',
  border: '1px solid var(--border)',
  padding: '2px 6px',
  borderRadius: 4,
  fontSize: 11,
  color: 'var(--text-muted)',
};

const paramPill: CSSProperties = {
  background: 'var(--surface-2)',
  border: '1px solid var(--border)',
  borderRadius: 4,
  padding: '1px 6px',
  fontSize: 11,
  fontFamily: "ui-monospace, 'Cascadia Code', Consolas, monospace",
  color: 'var(--text)',
  whiteSpace: 'nowrap',
};

const kvKey: CSSProperties = {
  padding: '3px 8px 3px 0',
  color: 'var(--text-muted)',
  fontFamily: "ui-monospace, 'Cascadia Code', Consolas, monospace",
  whiteSpace: 'nowrap',
  verticalAlign: 'top',
};

const kvVal: CSSProperties = {
  padding: '3px 0',
  color: 'var(--text)',
  wordBreak: 'break-all',
};
