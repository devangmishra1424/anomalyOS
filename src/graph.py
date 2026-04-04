# src/graph.py
# Loads the NetworkX knowledge graph and exposes 2-hop traversal
# Graph built in notebook 04, stored as node-link JSON on HF Dataset
# Loaded once at FastAPI startup, kept in memory

import os
import json
import networkx as nx


DATA_DIR = os.environ.get("DATA_DIR", "data")


class KnowledgeGraph:
    """
    Wraps the NetworkX DiGraph.
    Provides 2-hop context retrieval for the RAG orchestrator.
    """

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.graph = None

    def load(self):
        path = os.path.join(self.data_dir, "knowledge_graph.json")
        if not os.path.exists(path):
            print(f"Warning: Knowledge graph not found at {path}, using empty graph")
            self.graph = nx.DiGraph()
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load knowledge graph JSON: {e}")
            self.graph = nx.DiGraph()
            return

        # Try standard NetworkX node_link_graph format (expects 'links' or 'edges')
        try:
            self.graph = nx.node_link_graph(data, directed=True)
        except (KeyError, TypeError):
            # If that fails, try converting 'edges' key to 'links'
            if "edges" in data:
                data["links"] = data.pop("edges")
                try:
                    self.graph = nx.node_link_graph(data, directed=True)
                except Exception as e:
                    print(f"Warning: Failed to load graph with edges→links conversion: {e}")
                    self.graph = nx.DiGraph()
            else:
                # Last resort: create empty graph
                print(f"Warning: Knowledge graph format not recognized, using empty graph")
                self.graph = nx.DiGraph()

        print(f"Knowledge graph loaded: "
              f"{self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def get_context(self, category: str, defect_type: str) -> dict:
        """
        2-hop traversal from a defect node.
        Returns: root causes, remediations, co-occurring defects.

        Path: defect → [caused_by] → root_cause
                     → [remediated_by] → remediation
              defect → [co_occurs_with] → related_defect
        """
        if self.graph is None:
            return {"root_causes": [], "remediations": [], "co_occurs": []}

        defect_key = f"defect_{category}_{defect_type}"

        # Try exact match first, then fallback to category-level
        if defect_key not in self.graph:
            # Try to find any defect node for this category
            candidates = [
                n for n in self.graph.nodes
                if n.startswith(f"defect_{category}_")
            ]
            if not candidates:
                return {"root_causes": [], "remediations": [], "co_occurs": []}
            defect_key = candidates[0]

        root_causes = []
        remediations = []
        co_occurs = []

        for nb1 in self.graph.successors(defect_key):
            edge1 = self.graph[defect_key][nb1].get("edge_type", "")
            node1_data = self.graph.nodes[nb1]

            if edge1 == "caused_by":
                rc = node1_data.get("name", nb1.replace("root_cause_", ""))
                root_causes.append(rc)

                # Second hop: root_cause → remediation
                for nb2 in self.graph.successors(nb1):
                    edge2 = self.graph[nb1][nb2].get("edge_type", "")
                    if edge2 == "remediated_by":
                        node2_data = self.graph.nodes[nb2]
                        rem = node2_data.get("name",
                                             nb2.replace("remediation_", ""))
                        remediations.append(rem)

            elif edge1 == "co_occurs_with":
                co_key = nb1.replace("defect_", "")
                co_occurs.append(co_key)

        return {
            "defect_key": defect_key,
            "root_causes": list(set(root_causes)),
            "remediations": list(set(remediations)),
            "co_occurs": co_occurs
        }

    def get_all_defect_nodes(self) -> list:
        """Returns all defect nodes — used by Knowledge Base Explorer."""
        if self.graph is None:
            return []
        return [
            {
                "node_id": n,
                **self.graph.nodes[n]
            }
            for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "defect_instance"
        ]

    def get_status(self) -> dict:
        if self.graph is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges()
        }


# Global instance
knowledge_graph = KnowledgeGraph()