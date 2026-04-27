export function groupModels(models: any[]) {
  const grouped: Record<string, any> = {};

  for (const m of models) {
    const key = `${m.task}:${m.model_name}`;

    if (!grouped[key]) {
      grouped[key] = {
        task: m.task,
        model_name: m.model_name,
        versions: [],
      };
    }

    grouped[key].versions.push(m);
  }

  return Object.values(grouped);
}