import { useEffect, useState } from "react";
import { fetchModels } from "../api/registry";
import { groupModels } from "../utils/groupModels";
import ModelGroup from "../components/ModelGroup";

export default function Dashboard() {
  const [groups, setGroups] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    setLoading(true);
    const data = await fetchModels();
    const grouped = groupModels(data);
    setGroups(grouped);
    setLoading(false);
  }

  if (loading) return <div>Loading registry console...</div>;

  return (
    <div style={{ padding: 20 }}>
      <h1>Model Registry Console</h1>

      {groups.map((g, i) => (
        <ModelGroup key={i} group={g} />
      ))}
    </div>
  );
}