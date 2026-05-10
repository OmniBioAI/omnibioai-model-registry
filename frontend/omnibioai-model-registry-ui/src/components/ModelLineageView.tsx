import { useMemo } from "react";
import ReactFlow, { Background, Controls } from "reactflow";
import type { Node, Edge } from "reactflow";

import "reactflow/dist/style.css";
import dagre from "dagre";

type Model = {
  model_name: string;
  version: string;
  task: string;
  alias?: string;
  created_at?: string;
  lineage?: {
    dataset?: string;
    run_id?: string;
    metrics?: Record<string, number>;
    parent_version?: string | null;
  };
};

function layout(nodes: Node[], edges: Edge[]) {
  // IMPORTANT FIX: new graph per render (no shared mutation)
  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({ rankdir: "TB" });

  nodes.forEach((n) => {
    dagreGraph.setNode(n.id, { width: 180, height: 70 });
  });

  edges.forEach((e) => {
    dagreGraph.setEdge(e.source, e.target);
  });

  dagre.layout(dagreGraph);

  return nodes.map((node) => {
    const pos = dagreGraph.node(node.id);

    return {
      ...node,
      position: {
        x: pos.x,
        y: pos.y,
      },
    };
  });
}

export default function ModelLineageView({ model }: { model: Model }) {
  const graph = useMemo(() => buildGraph(model), [
    model.model_name,
    model.version,
  ]);

  const layoutNodes = useMemo(
    () => layout(graph.nodes, graph.edges),
    [graph.nodes, graph.edges]
  );

  return (
    <div style={{ height: 500, border: "1px solid #ddd", marginTop: 10 }}>
      <ReactFlow nodes={layoutNodes} edges={graph.edges} fitView>
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}

function buildGraph(model: Model) {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  const modelId = model.model_name;
  const versionId = `${model.model_name}:${model.version}`;

  // -------------------------
  // Model node
  // -------------------------
  nodes.push({
    id: modelId,
    data: { label: `📦 ${model.model_name}` },
    position: { x: 0, y: 0 },
    style: {
      background: "#2563eb",
      color: "#fff",
      padding: 10,
      borderRadius: 8,
    },
  });

  // -------------------------
  // Version node
  // -------------------------
  nodes.push({
    id: versionId,
    data: {
      label: `Version: ${model.version}\nTask: ${model.task}\nRun: ${
        model.lineage?.run_id ?? "N/A"
      }`,
    },
    position: { x: 0, y: 0 },
    style: {
      background: "#f3f4f6",
      padding: 10,
      borderRadius: 8,
      whiteSpace: "pre-wrap",
      fontSize: 12,
    },
  });

  edges.push({
    id: "e-model-version",
    source: modelId,
    target: versionId,
  });

  // -------------------------
  // Dataset node
  // -------------------------
  if (model.lineage?.dataset) {
    const datasetId = `dataset:${model.lineage.dataset}`;

    nodes.push({
      id: datasetId,
      data: { label: `🧬 ${model.lineage.dataset}` },
      position: { x: 0, y: 0 },
      style: {
        background: "#ecfeff",
        padding: 10,
        borderRadius: 8,
      },
    });

    edges.push({
      id: "e-dataset-model",
      source: datasetId,
      target: versionId,
    });
  }

  // -------------------------
  // Parent lineage
  // -------------------------
  if (model.lineage?.parent_version) {
    const parentId = `${model.model_name}:${model.lineage.parent_version}`;

    nodes.push({
      id: parentId,
      data: {
        label: `⬅ Parent\n${model.lineage.parent_version}`,
      },
      position: { x: 0, y: 0 },
      style: {
        background: "#fef3c7",
        padding: 10,
        borderRadius: 8,
        fontSize: 12,
      },
    });

    edges.push({
      id: "e-parent",
      source: parentId,
      target: versionId,
    });
  }

  return { nodes, edges };
}