import { useState, type CSSProperties, type ReactNode } from "react";

const BASE_URL = "/v1";

function todayVersion(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}_001`;
}

type FormState = {
  task: string;
  model_name: string;
  version: string;
  artifacts_dir: string;
  set_alias: string;
  actor: string;
};

export default function RegisterModelModal({
  onClose,
  onSuccess,
}: {
  onClose: () => void;
  onSuccess: () => void;
}) {
  const [form, setForm] = useState<FormState>({
    task: "",
    model_name: "",
    version: todayVersion(),
    artifacts_dir: "",
    set_alias: "latest",
    actor: "registry-ui",
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ status: number; data: any } | { error: string } | null>(null);
  const [errors, setErrors] = useState<Partial<Record<keyof FormState, string>>>({});

  function update(field: keyof FormState, value: string) {
    setForm((f) => ({ ...f, [field]: value }));
    setErrors((e) => { const next = { ...e }; delete next[field]; return next; });
  }

  function validate(): boolean {
    const errs: Partial<Record<keyof FormState, string>> = {};
    if (!form.task.trim()) errs.task = "Required";
    if (!form.model_name.trim()) errs.model_name = "Required";
    if (!form.version.trim()) errs.version = "Required";
    if (!form.artifacts_dir.trim()) errs.artifacts_dir = "Required";
    setErrors(errs);
    return Object.keys(errs).length === 0;
  }

  async function submit() {
    if (!validate()) return;
    setLoading(true);
    setResult(null);
    try {
      const body: Record<string, unknown> = {
        task: form.task.trim(),
        model_name: form.model_name.trim(),
        version: form.version.trim(),
        artifacts_dir: form.artifacts_dir.trim(),
        metadata: {},
        actor: form.actor.trim() || "registry-ui",
      };
      if (form.set_alias.trim()) body.set_alias = form.set_alias.trim();

      const res = await fetch(`${BASE_URL}/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      setResult({ status: res.status, data });
      if (res.ok) {
        setTimeout(onSuccess, 1200);
      }
    } catch (e) {
      setResult({ error: String(e) });
    } finally {
      setLoading(false);
    }
  }

  const isSuccess = result && !("error" in result) && (result as any).data?.ok !== false;

  return (
    <div style={overlayStyle}>
      <div style={modalStyle}>
        {/* HEADER */}
        <div style={modalHeaderStyle}>
          <span style={{ fontWeight: 700, fontSize: 15 }}>Register Model</span>
          <button
            onClick={onClose}
            style={{ background: "transparent", border: "none", fontSize: 18, color: "var(--text-muted)", padding: "2px 6px" }}
          >
            ✕
          </button>
        </div>

        {/* BODY */}
        <div style={modalBodyStyle}>
          <Field label="Task *" error={errors.task}>
            <input
              placeholder="e.g. celltype_sc"
              value={form.task}
              onChange={(e) => update("task", e.target.value)}
              style={fieldInputStyle(!!errors.task)}
              autoFocus
            />
          </Field>

          <Field label="Model Name *" error={errors.model_name}>
            <input
              placeholder="e.g. human_pbmc"
              value={form.model_name}
              onChange={(e) => update("model_name", e.target.value)}
              style={fieldInputStyle(!!errors.model_name)}
            />
          </Field>

          <Field label="Version *" error={errors.version}>
            <input
              placeholder="YYYY-MM-DD_001"
              value={form.version}
              onChange={(e) => update("version", e.target.value)}
              style={fieldInputStyle(!!errors.version)}
            />
          </Field>

          <Field label="Artifacts Directory *" error={errors.artifacts_dir}>
            <input
              placeholder="/path/to/artifacts/"
              value={form.artifacts_dir}
              onChange={(e) => update("artifacts_dir", e.target.value)}
              style={fieldInputStyle(!!errors.artifacts_dir)}
            />
          </Field>

          <Field label="Set Alias (optional)" hint="Leave blank to skip alias assignment">
            <input
              placeholder="latest"
              value={form.set_alias}
              onChange={(e) => update("set_alias", e.target.value)}
              style={fieldInputStyle(false)}
            />
          </Field>

          <Field label="Actor">
            <input
              placeholder="registry-ui"
              value={form.actor}
              onChange={(e) => update("actor", e.target.value)}
              style={fieldInputStyle(false)}
            />
          </Field>

          {/* RESULT */}
          {result && (
            <pre
              style={{
                background: "var(--surface-2)",
                border: `1px solid ${isSuccess ? "var(--green)" : "var(--red)"}`,
                borderRadius: 6,
                padding: 10,
                fontSize: 12,
                overflow: "auto",
                maxHeight: 160,
                color: isSuccess ? "var(--green)" : "var(--red)",
                marginTop: 4,
              }}
            >
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>

        {/* FOOTER */}
        <div style={modalFooterStyle}>
          <button onClick={onClose}>Cancel</button>
          <button
            onClick={submit}
            disabled={loading}
            style={{ background: "var(--accent)", color: "#fff", border: "none", fontWeight: 600 }}
          >
            {loading ? "Registering…" : "Register"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({
  label,
  error,
  hint,
  children,
}: {
  label: string;
  error?: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ fontSize: 12, fontWeight: 600, color: "var(--text-muted)" }}>
        {label}
      </label>
      {children}
      {error && <div style={{ fontSize: 11, color: "var(--red)" }}>{error}</div>}
      {hint && !error && <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{hint}</div>}
    </div>
  );
}

function fieldInputStyle(hasError: boolean): CSSProperties {
  return {
    width: "100%",
    borderColor: hasError ? "var(--red)" : undefined,
  };
}

const overlayStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(15,23,42,0.35)",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  zIndex: 200,
};

const modalStyle: CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 10,
  width: 480,
  maxHeight: "90vh",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
  boxShadow: "0 12px 40px rgba(15,23,42,0.12)",
};

const modalHeaderStyle: CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "14px 20px",
  borderBottom: "1px solid var(--border)",
  flexShrink: 0,
};

const modalBodyStyle: CSSProperties = {
  padding: "20px",
  display: "flex",
  flexDirection: "column",
  gap: 14,
  overflowY: "auto",
};

const modalFooterStyle: CSSProperties = {
  display: "flex",
  justifyContent: "flex-end",
  gap: 8,
  padding: "12px 20px",
  borderTop: "1px solid var(--border)",
  flexShrink: 0,
};
