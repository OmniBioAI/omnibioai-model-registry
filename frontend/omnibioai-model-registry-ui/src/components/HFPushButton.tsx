import { useEffect, useRef, useState, type CSSProperties } from "react";

const BASE_URL = "/v1/hf";

type PushStatus = "idle" | "queued" | "running" | "success" | "error";

const LICENSES = ["apache-2.0", "mit", "cc-by-4.0"];

export default function HFPushButton({
  task,
  model_name,
  version,
  description,
}: {
  task: string;
  model_name: string;
  version: string;
  description?: string;
}) {
  const [open, setOpen] = useState(false);
  const [repoId, setRepoId] = useState(`omnibioai/${model_name}`);
  const [token, setToken] = useState("");
  const [desc, setDesc] = useState(description ?? "");
  const [license, setLicense] = useState("apache-2.0");
  const [isPrivate, setIsPrivate] = useState(true);
  const [hasDefaultToken, setHasDefaultToken] = useState(false);

  const [status, setStatus] = useState<PushStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    if (!open) return;
    fetch(`${BASE_URL}/settings`)
      .then((r) => r.json())
      .then((d) => setHasDefaultToken(!!d.has_default_token))
      .catch(() => setHasDefaultToken(false));
  }, [open]);

  useEffect(() => {
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, []);

  function reset() {
    setStatus("idle");
    setError(null);
    setUrl(null);
  }

  function close() {
    if (pollRef.current !== null) window.clearInterval(pollRef.current);
    setOpen(false);
    reset();
  }

  async function push() {
    if (!repoId.trim()) {
      setError("Repo name is required");
      return;
    }
    if (!token.trim() && !hasDefaultToken) {
      setError("HuggingFace token is required");
      return;
    }
    setStatus("queued");
    setError(null);

    try {
      const res = await fetch(`${BASE_URL}/push`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task,
          model_name,
          version,
          repo_id: repoId.trim(),
          token: token.trim() || undefined,
          description: desc,
          license,
          private: isPrivate,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        setStatus("error");
        setError(data?.detail ?? `Push failed: ${res.status}`);
        return;
      }

      const jobId = data.job_id as string;
      pollRef.current = window.setInterval(async () => {
        try {
          const sRes = await fetch(`${BASE_URL}/push/status/${jobId}`);
          const sData = await sRes.json();
          if (sData.status === "success") {
            setStatus("success");
            setUrl(sData.url);
            if (pollRef.current !== null) window.clearInterval(pollRef.current);
          } else if (sData.status === "error") {
            setStatus("error");
            setError(sData.error ?? "Push failed");
            if (pollRef.current !== null) window.clearInterval(pollRef.current);
          } else {
            setStatus(sData.status);
          }
        } catch {
          // transient poll failure — keep polling
        }
      }, 2000);
    } catch (e) {
      setStatus("error");
      setError(String(e));
    }
  }

  const busy = status === "queued" || status === "running";

  return (
    <>
      <button
        style={{ fontSize: 11, padding: "3px 8px" }}
        onClick={() => setOpen(true)}
        title="Push this version to HuggingFace Hub"
      >
        🤗 Push to HF
      </button>

      {open && (
        <div style={overlayStyle}>
          <div style={modalStyle}>
            <div style={modalHeaderStyle}>
              <span style={{ fontWeight: 700, fontSize: 15 }}>🤗 Push to HuggingFace</span>
              <button
                onClick={close}
                style={{ background: "transparent", border: "none", fontSize: 18, color: "var(--text-muted)", padding: "2px 6px" }}
              >
                ✕
              </button>
            </div>

            <div style={modalBodyStyle}>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                {task}/{model_name}@{version}
              </div>

              {status === "success" ? (
                <div style={{ textAlign: "center", padding: "12px 0" }}>
                  <div style={{ color: "var(--green)", fontWeight: 600, marginBottom: 8 }}>
                    Pushed successfully
                  </div>
                  <a href={url ?? "#"} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent)", fontSize: 13 }}>
                    {url} ↗
                  </a>
                </div>
              ) : (
                <>
                  <Field label="Repo name">
                    <input
                      value={repoId}
                      onChange={(e) => setRepoId(e.target.value)}
                      placeholder="namespace/model-name"
                      style={{ width: "100%" }}
                      disabled={busy}
                    />
                  </Field>

                  <Field
                    label="HuggingFace token"
                    hint={hasDefaultToken ? "Leave blank to use the server-configured token" : "Required — no server-side default is configured"}
                  >
                    <input
                      type="password"
                      value={token}
                      onChange={(e) => setToken(e.target.value)}
                      placeholder="hf_xxxxxxxxxxxx"
                      style={{ width: "100%" }}
                      disabled={busy}
                    />
                  </Field>

                  <Field label="Description">
                    <textarea
                      value={desc}
                      onChange={(e) => setDesc(e.target.value)}
                      rows={2}
                      style={{ width: "100%", resize: "vertical" }}
                      disabled={busy}
                    />
                  </Field>

                  <div style={{ display: "flex", gap: 12 }}>
                    <Field label="License">
                      <select value={license} onChange={(e) => setLicense(e.target.value)} disabled={busy}>
                        {LICENSES.map((l) => (
                          <option key={l} value={l}>{l}</option>
                        ))}
                      </select>
                    </Field>
                    <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-muted)", marginTop: 18 }}>
                      <input
                        type="checkbox"
                        checked={isPrivate}
                        onChange={(e) => setIsPrivate(e.target.checked)}
                        disabled={busy}
                      />
                      Private
                    </label>
                  </div>

                  {status !== "idle" && (
                    <div style={{ fontSize: 12, color: status === "error" ? "var(--red)" : "var(--text-muted)" }}>
                      {status === "queued" && "Queued…"}
                      {status === "running" && "Uploading to HuggingFace…"}
                      {status === "error" && (error ?? "Push failed")}
                    </div>
                  )}
                </>
              )}
            </div>

            <div style={modalFooterStyle}>
              <button onClick={close}>{status === "success" ? "Close" : "Cancel"}</button>
              {status !== "success" && (
                <button
                  onClick={push}
                  disabled={busy}
                  style={{ background: "var(--accent)", color: "#fff", border: "none", fontWeight: 600 }}
                >
                  {busy ? "Pushing…" : "Push"}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ fontSize: 12, fontWeight: 600, color: "var(--text-muted)" }}>{label}</label>
      {children}
      {hint && <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{hint}</div>}
    </div>
  );
}

const overlayStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.65)",
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  zIndex: 300,
};

const modalStyle: CSSProperties = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 10,
  width: 440,
  maxHeight: "90vh",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
  boxShadow: "0 12px 48px rgba(0,0,0,0.55)",
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
