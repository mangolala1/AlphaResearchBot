"""NetworkX research graph — nodes are alphas, edges are parent→child."""

from __future__ import annotations

import networkx as nx

from core.memory import ExperimentStore
from core.types import ExperimentRecord


class ResearchGraph:
    """Maintains a directed graph of alpha experiments."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_experiment(self, record: ExperimentRecord) -> None:
        """Add a single experiment as a node, with an edge from its parent if present."""
        metrics = record.get("metrics", {})
        self._graph.add_node(
            record["alpha_id"],
            alpha_id=record["alpha_id"],
            hypothesis=record.get("hypothesis", ""),
            formula=record.get("formula", ""),
            mutation=record.get("mutation", ""),
            verdict=record.get("verdict", ""),
            failure_reason=record.get("failure_reason") or "N/A",
            reflection=record.get("reflection", ""),
            Sharpe=metrics.get("Sharpe", 0.0),
            ICIR=metrics.get("ICIR", 0.0),
            deflated_sharpe=metrics.get("deflated_sharpe", 0.0),
        )
        parent_id = record.get("parent_id")
        if parent_id and parent_id in self._graph:
            self._graph.add_edge(parent_id, record["alpha_id"])

    def build_from_store(self, store: ExperimentStore) -> None:
        """Load all experiments from the store and build the graph."""
        for record in store.load_all():
            self.add_experiment(record)

    def get_graph(self) -> nx.DiGraph:
        return self._graph
