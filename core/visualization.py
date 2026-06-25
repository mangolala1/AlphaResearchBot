"""Research graph export — custom standalone HTML with vis-network.js."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

_VERDICT_COLORS = {
    "promising": "#2ecc71",
    "revise":    "#f1c40f",
    "failed":    "#e74c3c",
}
_DEFAULT_COLOR = "#95a5a6"

# Sentinel strings replaced into the HTML template at export time.
_NODES_MARKER  = '"__NODES_DATA__"'
_EDGES_MARKER  = '"__EDGES_DATA__"'
_COLORS_MARKER = '"__VERDICT_COLORS__"'
_DFLT_MARKER   = "__DEFAULT_COLOR__"


def export_graph_html(
    graph: nx.DiGraph,
    output_path: str = "reports/research_graph.html",
) -> None:
    """Export the research graph to a self-contained interactive HTML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    nodes_data = []
    for node_id, attrs in graph.nodes(data=True):
        nodes_data.append({
            "id":            node_id,
            "alpha_id":      attrs.get("alpha_id", node_id),
            "hypothesis":    attrs.get("hypothesis", ""),
            "formula":       attrs.get("formula", ""),
            "mutation":      attrs.get("mutation", ""),
            "verdict":       attrs.get("verdict", ""),
            "Sharpe":        float(attrs.get("Sharpe", 0.0)),
            "ICIR":          float(attrs.get("ICIR", 0.0)),
            "IC_mean":       float(attrs.get("IC_mean", 0.0)),
            "monotonicity":  float(attrs.get("monotonicity", 0.0)),
            "turnover":      float(attrs.get("turnover", 0.0)),
            "max_drawdown":  float(attrs.get("max_drawdown", 0.0)),
            "deflated_sharpe": float(attrs.get("deflated_sharpe", 0.0)),
            "failure_reason": attrs.get("failure_reason") or "",
            "failure_category": attrs.get("failure_category") or "",
            "reflection":    attrs.get("reflection", ""),
        })

    node_attr_map = dict(graph.nodes(data=True))
    edges_data = [
        {
            "from":     src,
            "to":       dst,
            "mutation": node_attr_map.get(dst, {}).get("mutation", ""),
        }
        for src, dst in graph.edges()
    ]

    html = (
        _HTML_TEMPLATE
        .replace(_NODES_MARKER,  json.dumps(nodes_data))
        .replace(_EDGES_MARKER,  json.dumps(edges_data))
        .replace(_COLORS_MARKER, json.dumps(_VERDICT_COLORS))
        .replace(_DFLT_MARKER,   _DEFAULT_COLOR)
    )
    Path(output_path).write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML template — placeholders replaced by export_graph_html()
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Alpha Research Graph</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Courier New', Courier, monospace;
    height: 100vh;
    overflow: hidden;
  }

  #layout {
    display: flex;
    height: 100vh;
  }

  /* ── Left: graph canvas ─────────────────────────── */
  #graph-pane {
    flex: 0 0 70%;
    position: relative;
    border-right: 1px solid #2a2a4a;
  }

  #mynetwork {
    width: 100%;
    height: 100%;
  }

  /* ── Right: detail panel ────────────────────────── */
  #detail-pane {
    flex: 0 0 30%;
    display: flex;
    flex-direction: column;
    background: #0f0f1e;
    overflow: hidden;
  }

  #detail-header {
    flex-shrink: 0;
    padding: 12px 16px 10px;
    background: #13132a;
    border-bottom: 1px solid #2a2a4a;
  }

  .hdr-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .hdr-id {
    font-size: 15px;
    font-weight: bold;
    color: #fff;
    letter-spacing: 0.5px;
  }

  .verdict-badge {
    font-size: 11px;
    font-weight: bold;
    padding: 2px 9px;
    border-radius: 10px;
    color: #111;
  }

  #detail-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px 20px;
  }

  /* scrollbar */
  #detail-body::-webkit-scrollbar { width: 5px; }
  #detail-body::-webkit-scrollbar-track { background: #0f0f1e; }
  #detail-body::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }

  .placeholder {
    color: #3a3a5a;
    text-align: center;
    margin-top: 70px;
    font-size: 13px;
    line-height: 2;
  }

  /* section headers */
  .sec {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #444466;
    margin-top: 16px;
    margin-bottom: 5px;
    padding-bottom: 3px;
    border-bottom: 1px solid #1e1e38;
  }

  /* hypothesis / mutation / reflection text */
  .prose {
    font-size: 11px;
    color: #9090b0;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* formula code block */
  .formula-block {
    display: block;
    background: #131328;
    border: 1px solid #2a2a50;
    border-radius: 4px;
    padding: 7px 10px;
    font-size: 12px;
    color: #7ec8e3;
    white-space: pre-wrap;
    word-break: break-all;
    margin-top: 2px;
  }

  /* metric rows */
  .mrow {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 2px 0;
    font-size: 12px;
  }

  .mname { color: #666688; }

  .mval       { color: #ccccdd; font-weight: bold; }
  .mval.warn  { color: #f1c40f; }
  .mval.bad   { color: #e74c3c; }

  .warn-icon { color: #f1c40f; font-size: 10px; margin-left: 3px; }

  .failure-text {
    font-size: 11px;
    color: #e74c3c;
    line-height: 1.5;
    word-break: break-word;
  }
</style>
</head>
<body>
<div id="layout">

  <div id="graph-pane">
    <div id="mynetwork"></div>
  </div>

  <div id="detail-pane">
    <div id="detail-header">
      <div class="hdr-row">
        <span class="hdr-id" id="hdr-id">Alpha Research Graph</span>
        <span class="verdict-badge" id="hdr-badge" style="background:#2a2a4a;color:#888"></span>
      </div>
    </div>
    <div id="detail-body">
      <div class="placeholder">&#8592; click a node<br>to view details</div>
    </div>
  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js"></script>
<script>
// ── Data injected at export time ─────────────────────────────────────────────
const NODES_DATA       = "__NODES_DATA__";
const EDGES_DATA       = "__EDGES_DATA__";
const VERDICT_COLORS   = "__VERDICT_COLORS__";
const DEFAULT_COLOR    = "__DEFAULT_COLOR__";

// ── Tier thresholds (mirror decision.py) ─────────────────────────────────────
const T = {
  ic_mean_soft:   0.02,
  icir_soft:      0.30,
  mono_soft:      0.30,
  sharpe_hard:    0.0,
  sharpe_soft:    0.50,
  turnover_max:   0.70,
  drawdown_hard: -0.40,
  drawdown_soft: -0.25,
};

// ── Helpers ───────────────────────────────────────────────────────────────────
function nodeColor(verdict) {
  return VERDICT_COLORS[verdict] || DEFAULT_COLOR;
}

function esc(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function warnClass(val, soft, hard) {
  if (hard !== undefined && val <= hard) return 'bad';
  if (val <= soft) return 'warn';
  return '';
}

function highClass(val, max) {
  if (val >= max) return 'warn';
  return '';
}

function metricRow(label, valStr, cls) {
  const icon = cls ? '<span class="warn-icon">&#9888;</span>' : '';
  return '<div class="mrow">'
    + '<span class="mname">' + label + '</span>'
    + '<span class="mval ' + cls + '">' + valStr + icon + '</span>'
    + '</div>';
}

// ── Build vis-network nodes ───────────────────────────────────────────────────
const visNodes = new vis.DataSet(NODES_DATA.map(n => {
  const bg = nodeColor(n.verdict);
  return {
    id:    n.id,
    label: n.alpha_id + '\\nS:' + n.Sharpe.toFixed(2) + '  IC:' + n.ICIR.toFixed(2),
    shape: 'box',
    color: {
      background: bg,
      border: bg,
      highlight: { background: bg, border: '#ffffff' },
    },
    font: { color: '#111111', size: 12, face: 'Courier New', bold: { size: 13, color: '#000000' } },
    borderWidth: 3,
    borderWidthSelected: 4,
    shapeProperties: { borderDashes: n.verdict === 'failed' ? [5, 3] : false },
    margin: { top: 8, right: 10, bottom: 8, left: 10 },
  };
}));

// ── Build vis-network edges ───────────────────────────────────────────────────
const visEdges = new vis.DataSet(EDGES_DATA.map((e, i) => ({
  id:    i,
  from:  e.from,
  to:    e.to,
  label: e.mutation ? e.mutation.substring(0, 35) : '',
  font:  { size: 10, color: '#888888', align: 'middle', strokeWidth: 0 },
  color: { color: '#666688', highlight: '#9999cc' },
  arrows: 'to',
  smooth: { type: 'curvedCW', roundness: 0.08 },
})));

// ── Initialise network ────────────────────────────────────────────────────────
const container = document.getElementById('mynetwork');
const network = new vis.Network(
  container,
  { nodes: visNodes, edges: visEdges },
  {
    physics: { enabled: false },
    layout: {
      hierarchical: {
        enabled: true,
        direction: 'UD',
        sortMethod: 'directed',
        levelSeparation: 130,
        nodeSpacing: 190,
        treeSpacing: 220,
      },
    },
    interaction: { hover: false, tooltipDelay: 9999 },
  }
);

// ── Click handlers ────────────────────────────────────────────────────────────
network.on('selectNode', params => {
  const n = NODES_DATA.find(x => x.id === params.nodes[0]);
  if (n) renderDetail(n);
});
network.on('deselectNode', resetDetail);

function resetDetail() {
  document.getElementById('hdr-id').textContent    = 'Alpha Research Graph';
  document.getElementById('hdr-badge').textContent = '';
  document.getElementById('hdr-badge').style.background = '#2a2a4a';
  document.getElementById('hdr-badge').style.color      = '#888';
  document.getElementById('detail-body').innerHTML =
    '<div class="placeholder">&#8592; click a node<br>to view details</div>';
}

function renderDetail(n) {
  // Header
  document.getElementById('hdr-id').textContent = n.alpha_id;
  const badge = document.getElementById('hdr-badge');
  badge.textContent       = n.verdict.toUpperCase();
  badge.style.background  = nodeColor(n.verdict);
  badge.style.color       = '#111';

  // Warn classes
  const wIC   = warnClass(n.IC_mean,      T.ic_mean_soft);
  const wICIR = warnClass(n.ICIR,         T.icir_soft);
  const wMono = warnClass(n.monotonicity,  T.mono_soft);
  const wSharpe   = warnClass(n.Sharpe,       T.sharpe_soft, T.sharpe_hard);
  const wTurnover = highClass(n.turnover,     T.turnover_max);
  const wDD       = warnClass(n.max_drawdown, T.drawdown_soft, T.drawdown_hard);

  let html = '';

  if (n.hypothesis) {
    html += '<div class="sec">Hypothesis</div>'
          + '<div class="prose">' + esc(n.hypothesis) + '</div>';
  }

  if (n.formula) {
    html += '<div class="sec">Formula</div>'
          + '<code class="formula-block">' + esc(n.formula) + '</code>';
  }

  html += '<div class="sec">Tier 1 &mdash; Predictive Power</div>';
  html += metricRow('IC_mean',      n.IC_mean.toFixed(4),       wIC);
  html += metricRow('ICIR',         n.ICIR.toFixed(4),          wICIR);
  html += metricRow('Monotonicity', n.monotonicity.toFixed(4),  wMono);

  html += '<div class="sec">Tier 2 &mdash; Implementation</div>';
  html += metricRow('Sharpe',       n.Sharpe.toFixed(4),        wSharpe);
  html += metricRow('Turnover',     n.turnover.toFixed(4),      wTurnover);
  html += metricRow('Max Drawdown', n.max_drawdown.toFixed(4),  wDD);

  html += '<div class="sec">Tier 3 &mdash; Diagnostics</div>';
  html += metricRow('Deflated Sharpe', n.deflated_sharpe.toFixed(4), '');

  if (n.failure_reason) {
    html += '<div class="sec">Failure Reason</div>'
          + '<div class="failure-text">' + esc(n.failure_reason) + '</div>';
  }

  if (n.mutation) {
    html += '<div class="sec">Mutation</div>'
          + '<div class="prose">' + esc(n.mutation) + '</div>';
  }

  if (n.reflection) {
    html += '<div class="sec">Reflection</div>'
          + '<div class="prose">' + esc(n.reflection) + '</div>';
  }

  document.getElementById('detail-body').innerHTML = html;
}
</script>
</body>
</html>
"""
