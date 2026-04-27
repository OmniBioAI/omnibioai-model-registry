const BASE_URL = "http://localhost:8095/v1";

export async function fetchModels() {
  const res = await fetch(`${BASE_URL}/models`);

  if (!res.ok) {
    throw new Error(`Failed to fetch models: ${res.status}`);
  }

  return await res.json();
}