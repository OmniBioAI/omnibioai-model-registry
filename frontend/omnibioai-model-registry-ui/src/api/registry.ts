const BASE_URL = "/modelregistry/v1";

export async function fetchModels() {
  const res = await fetch(`${BASE_URL}/models`);

  if (!res.ok) {
    throw new Error(`Failed to fetch models: ${res.status}`);
  }

  return await res.json();
}

// ── Aliases ──────────────────────────────────────────────────────────────────

export type AliasEntry = {
  alias: string;
  version: string;
  updated_at?: string;
  actor?: string;
};

export type AliasesResponse = {
  ok: boolean;
  model_name: string;
  aliases: AliasEntry[];
};

export async function fetchAliases(task: string, model: string): Promise<AliasesResponse> {
  const res = await fetch(
    `${BASE_URL}/aliases?task=${encodeURIComponent(task)}&model=${encodeURIComponent(model)}`
  );
  if (!res.ok) throw new Error(`Failed to fetch aliases: ${res.status}`);
  return res.json() as Promise<AliasesResponse>;
}

// ── Metrics ───────────────────────────────────────────────────────────────────

export type MetricPoint = { value: number; step: number; ts_utc?: string };

export type MetricsResponse = {
  ok: boolean;
  version_metrics: Record<string, number>;
  run_history: Record<string, MetricPoint[]>;
};

export async function fetchMetrics(task: string, ref: string): Promise<MetricsResponse> {
  const res = await fetch(
    `${BASE_URL}/metrics?task=${encodeURIComponent(task)}&ref=${encodeURIComponent(ref)}`
  );
  if (!res.ok) throw new Error(`Failed to fetch metrics: ${res.status}`);
  return res.json() as Promise<MetricsResponse>;
}

// ── Compare ───────────────────────────────────────────────────────────────────

export type CompareVersionData = {
  metrics: Record<string, number>;
  created_at?: string;
  stage?: string;
  tags?: Record<string, string>;
};

export type CompareResponse = {
  ok: boolean;
  versions: Record<string, CompareVersionData>;
};

export async function compareVersions(
  task: string,
  model: string,
  versions: string[]
): Promise<CompareResponse> {
  const params = new URLSearchParams({ task, model });
  for (const v of versions) params.append("versions", v);
  const res = await fetch(`${BASE_URL}/compare?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to compare versions: ${res.status}`);
  return res.json() as Promise<CompareResponse>;
}

// ── Tags ──────────────────────────────────────────────────────────────────────

export async function setTag(
  task: string,
  model_name: string,
  version: string,
  key: string,
  value: string
): Promise<{ ok: boolean }> {
  const res = await fetch(`${BASE_URL}/tags`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, model_name, version, key, value }),
  });
  if (!res.ok) throw new Error(`Failed to set tag: ${res.status}`);
  return res.json() as Promise<{ ok: boolean }>;
}

// ── Stage ─────────────────────────────────────────────────────────────────────

export async function setStage(
  task: string,
  model_name: string,
  version: string,
  stage: string,
  actor?: string
): Promise<{ ok: boolean; stage: string; version: string }> {
  const res = await fetch(`${BASE_URL}/stage`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, model_name, version, stage, actor }),
  });
  if (!res.ok) throw new Error(`Failed to set stage: ${res.status}`);
  return res.json() as Promise<{ ok: boolean; stage: string; version: string }>;
}

// ── Artifacts ─────────────────────────────────────────────────────────────────

export type ArtifactEntry = { name: string; sha256?: string; size_bytes?: number };

export type ArtifactsResponse = {
  ok: boolean;
  version: string;
  files: ArtifactEntry[];
};

export async function fetchArtifacts(task: string, ref: string): Promise<ArtifactsResponse> {
  const res = await fetch(
    `${BASE_URL}/artifacts?task=${encodeURIComponent(task)}&ref=${encodeURIComponent(ref)}`
  );
  if (!res.ok) throw new Error(`Failed to fetch artifacts: ${res.status}`);
  return res.json() as Promise<ArtifactsResponse>;
}
