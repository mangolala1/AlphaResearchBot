"""PyVis HTML export for the research graph."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
from pyvis.network import Network

_VERDICT_COLORS = {
    "promising": "#2ecc71",
    "failed": "#e74c3c",
    "inconclusive": "#f39c12",
}
_DEFAULT_COLOR = "#95a5a6"


def export_graph_html(
    graph: nx.DiGraph,
    output_path: str = "reports/research_graph.html",
) -> None:
    """Export the research graph to an interactive HTML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="750px", width="100%", directed=True, bgcolor="#1a1a2e", font_color="white")
    net.set_options("""
    {
      "nodes": {"borderWidth": 2, "size": 30, "font": {"size": 14}},
      "edges": {"arrows": {"to": {"enabled": true}}, "color": {"color": "#aaaaaa"}},
      "physics": {"stabilization": {"iterations": 200}}
    }
    """)

    for node_id, attrs in graph.nodes(data=True):
        verdict = attrs.get("verdict", "")
        color = _VERDICT_COLORS.get(verdict, _DEFAULT_COLOR)
        tooltip = _build_tooltip(attrs)
        net.add_node(
            node_id,
            label=node_id,
            title=tooltip,
            color=color,
        )

    for src, dst in graph.edges():
        net.add_edge(src, dst)

    net.save_graph(output_path)


def _build_tooltip(attrs: dict) -> str:
    reflection = attrs.get("reflection", "")
    # Truncate long reflections for readability in tooltip
    if len(reflection) > 300:
        reflection = reflection[:297] + "..."

    return (
        f"<b>ID:</b> {attrs.get('alpha_id', '')}<br>"
        f"<b>Hypothesis:</b> {attrs.get('hypothesis', '')}<br>"
        f"<b>Formula:</b> {attrs.get('formula', '')}<br>"
        f"<b>Sharpe:</b> {attrs.get('Sharpe', 0.0):.3f} &nbsp;|&nbsp; "
        f"<b>ICIR:</b> {attrs.get('ICIR', 0.0):.3f} &nbsp;|&nbsp; "
        f"<b>Deflated Sharpe:</b> {attrs.get('deflated_sharpe', 0.0):.3f}<br>"
        f"<b>Verdict:</b> {attrs.get('verdict', '')}<br>"
        f"<b>Failure:</b> {attrs.get('failure_reason', 'N/A')}<br>"
        f"<b>Reflection:</b> {reflection}"
    )
