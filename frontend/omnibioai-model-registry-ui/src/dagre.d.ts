declare module "dagre" {
  namespace graphlib {
    class Graph {
      setDefaultEdgeLabel(fn: () => object): void;
      setGraph(opts: { rankdir?: string }): void;
      setNode(id: string, opts: { width: number; height: number }): void;
      setEdge(src: string, tgt: string): void;
      node(id: string): { x: number; y: number };
    }
  }
  function layout(graph: graphlib.Graph): void;
}
