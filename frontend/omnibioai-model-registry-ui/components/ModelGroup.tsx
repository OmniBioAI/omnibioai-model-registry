import { useState } from "react";
import VersionRow from "./VersionRow";

export default function ModelGroup({ group }: any) {
  const [open, setOpen] = useState(false);

  const latest = group.versions.find((v: any) => v.alias === "latest");

  return (
    <div style={{ border: "1px solid #ddd", marginBottom: 10 }}>
      <div
        style={{ padding: 10, cursor: "pointer", background: "#f5f5f5" }}
        onClick={() => setOpen(!open)}
      >
        <strong>{group.model_name}</strong> ({group.task}){" "}
        {latest?.version && <span>— latest: {latest.version}</span>}
      </div>

      {open && (
        <div>
          {group.versions.map((v: any, i: number) => (
            <VersionRow key={i} version={v} />
          ))}
        </div>
      )}
    </div>
  );
}