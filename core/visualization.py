"""Research graph export — custom standalone HTML with vis-network.js."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

_VERDICT_COLORS = {
    "promising": "#2ecc71",
    "revise":    "#f39c12",
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
            "original_id":   attrs.get("original_id", node_id),
            "batch_id":      attrs.get("batch_id") or "",
            "ring":          int(attrs.get("ring", 0)),
            "timestamp":     attrs.get("timestamp", ""),
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
            "from":      src,
            "to":        dst,
            "mutation":  node_attr_map.get(dst, {}).get("mutation", ""),
            "edge_type": edata.get("type", "mutation"),
        }
        for src, dst, edata in graph.edges(data=True)
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #f5f6fa;
    color: #1a1a2e;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  /* ── Summary bar ────────────────────────────────── */
  #summary-bar {
    flex-shrink: 0;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 20px;
    background: #ffffff;
    border-bottom: 2px solid #dde1e7;
    flex-wrap: wrap;
  }

  #summary-bar .bar-title {
    font-size: 15px;
    font-weight: bold;
    color: #1a1a2e;
    letter-spacing: 0.5px;
    margin-right: 8px;
    white-space: nowrap;
  }

  .stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: bold;
    white-space: nowrap;
  }

  .stat-chip .chip-label {
    font-weight: normal;
    opacity: 0.75;
    font-size: 12px;
  }

  .chip-total     { background: #e8eaf6; color: #3949ab; }
  .chip-promising { background: #e8f5e9; color: #2e7d32; }
  .chip-revise    { background: #fff8e1; color: #f57f17; }
  .chip-failed    { background: #fce4e4; color: #c62828; }
  .chip-sharpe    { background: #e3f2fd; color: #1565c0; }
  .chip-ic        { background: #f3e5f5; color: #6a1b9a; }

  .bar-divider {
    width: 1px;
    height: 24px;
    background: #dde1e7;
    flex-shrink: 0;
  }

  /* ── Main layout ────────────────────────────────── */
  #layout {
    display: flex;
    flex: 1;
    min-height: 0;
  }

  /* ── Left: graph canvas ─────────────────────────── */
  #graph-pane {
    flex: 1;
    position: relative;
    min-width: 0;
  }

  #mynetwork {
    width: 100%;
    height: 100%;
    background: #fafbfc;
  }

  /* ── Resize handle ──────────────────────────────── */
  #resize-handle {
    flex: 0 0 5px;
    width: 5px;
    cursor: col-resize;
    background: transparent;
    position: relative;
    z-index: 10;
    display: none;
  }
  #resize-handle::after {
    content: '';
    position: absolute;
    inset: 0;
    background: #dde1e7;
    opacity: 0;
    transition: opacity 0.15s;
  }
  #resize-handle:hover::after,
  #resize-handle.dragging::after {
    opacity: 1;
  }
  #detail-pane.open ~ #resize-handle,
  #detail-pane.open + #resize-handle {
    display: block;
  }

  /* ── Right: detail panel (starts closed) ────────── */
  #detail-pane {
    flex: 0 0 auto;
    width: 0;
    overflow: hidden;
    transition: width 0.28s ease, border-color 0.28s ease;
    border-left: 2px solid transparent;
  }

  #detail-pane.open {
    width: 360px;
    border-left-color: #dde1e7;
  }

  #detail-pane.resizing {
    transition: none;
  }

  /* inner wrapper keeps content at full width during the slide */
  #detail-inner {
    width: 100%;
    min-width: 260px;
    height: 100%;
    display: flex;
    flex-direction: column;
    background: #ffffff;
  }

  #detail-header {
    flex-shrink: 0;
    padding: 14px 18px 12px;
    background: #f0f2f5;
    border-bottom: 1px solid #dde1e7;
  }

  .hdr-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .hdr-id {
    font-size: 17px;
    font-weight: bold;
    color: #1a1a2e;
    letter-spacing: 0.5px;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .verdict-badge {
    font-size: 12px;
    font-weight: bold;
    padding: 3px 11px;
    border-radius: 10px;
    color: #fff;
    flex-shrink: 0;
  }

  #close-btn {
    flex-shrink: 0;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 19px;
    color: #7a8499;
    line-height: 1;
    padding: 0 2px;
    transition: color 0.15s;
  }
  #close-btn:hover { color: #1a1a2e; }

  #detail-body {
    flex: 1;
    overflow-y: auto;
    padding: 14px 18px 24px;
  }

  /* scrollbar */
  #detail-body::-webkit-scrollbar { width: 6px; }
  #detail-body::-webkit-scrollbar-track { background: #f5f6fa; }
  #detail-body::-webkit-scrollbar-thumb { background: #c8cdd6; border-radius: 3px; }

  .placeholder {
    color: #b0b8c8;
    text-align: center;
    margin-top: 70px;
    font-size: 14px;
    line-height: 2.2;
  }

  /* section headers */
  .sec {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #7a8499;
    margin-top: 18px;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #e4e7ed;
  }

  /* hypothesis / mutation / reflection text */
  .prose {
    font-size: 13px;
    color: #3a4254;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* formula code block */
  .formula-block {
    display: block;
    background: #f0f4ff;
    border: 1px solid #c5d0e8;
    border-radius: 5px;
    padding: 8px 12px;
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: 13px;
    color: #1a56db;
    white-space: pre-wrap;
    word-break: break-all;
    margin-top: 3px;
  }

  /* metric rows */
  .mrow {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 3px 0;
    font-size: 13px;
  }

  .mname { color: #7a8499; }

  .mval       { color: #1a1a2e; font-weight: bold; }
  .mval.warn  { color: #d97706; }
  .mval.bad   { color: #dc2626; }

  .warn-icon { color: #d97706; font-size: 11px; margin-left: 4px; }

  .failure-text {
    font-size: 13px;
    color: #dc2626;
    line-height: 1.6;
    word-break: break-word;
  }
</style>
</head>
<body>

<div id="summary-bar">
  <span class="bar-title">Alpha Research</span>
  <div class="bar-divider"></div>
  <span class="stat-chip chip-total">
    <span class="chip-label">Total</span>
    <span id="stat-total">—</span>
  </span>
  <span class="stat-chip chip-promising">
    <span class="chip-label">Promising</span>
    <span id="stat-promising">—</span>
  </span>
  <span class="stat-chip chip-revise">
    <span class="chip-label">Revise</span>
    <span id="stat-revise">—</span>
  </span>
  <span class="stat-chip chip-failed">
    <span class="chip-label">Failed</span>
    <span id="stat-failed">—</span>
  </span>
  <div class="bar-divider"></div>
  <span class="stat-chip chip-sharpe">
    <span class="chip-label">Best Sharpe</span>
    <span id="stat-sharpe">—</span>
  </span>
  <span class="stat-chip chip-ic">
    <span class="chip-label">Best IC</span>
    <span id="stat-ic">—</span>
  </span>
</div>

<div id="layout">

  <div id="graph-pane">
    <div id="mynetwork"></div>
  </div>

  <div id="resize-handle"></div>

  <div id="detail-pane">
    <div id="detail-inner">
      <div id="detail-header">
        <div class="hdr-row">
          <span class="hdr-id" id="hdr-id">—</span>
          <span class="verdict-badge" id="hdr-badge" style="background:#dde1e7;color:#7a8499"></span>
          <button id="close-btn" title="Close panel">&#x2715;</button>
        </div>
      </div>
      <div id="detail-body"></div>
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

// ── Populate summary bar ──────────────────────────────────────────────────────
(function() {
  const total     = NODES_DATA.length;
  const promising = NODES_DATA.filter(n => n.verdict === 'promising').length;
  const revise    = NODES_DATA.filter(n => n.verdict === 'revise').length;
  const failed    = NODES_DATA.filter(n => n.verdict === 'failed').length;

  const sharpes = NODES_DATA.map(n => n.Sharpe).filter(v => isFinite(v));
  const ics     = NODES_DATA.map(n => n.IC_mean).filter(v => isFinite(v));
  const bestSharpe = sharpes.length ? Math.max(...sharpes) : null;
  const bestIC     = ics.length     ? Math.max(...ics)     : null;

  document.getElementById('stat-total').textContent     = total;
  document.getElementById('stat-promising').textContent = promising;
  document.getElementById('stat-revise').textContent    = revise;
  document.getElementById('stat-failed').textContent    = failed;
  document.getElementById('stat-sharpe').textContent    = bestSharpe !== null ? bestSharpe.toFixed(3) : '—';
  document.getElementById('stat-ic').textContent        = bestIC     !== null ? bestIC.toFixed(4)     : '—';
})();

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

// ── Seed ring positions before building nodes ─────────────────────────────────
(function() {
  const ringCounts = {};
  NODES_DATA.forEach(n => { ringCounts[n.ring] = (ringCounts[n.ring] || 0) + 1; });
  const ringIdx = {};
  NODES_DATA.forEach(n => {
    if (ringIdx[n.ring] === undefined) ringIdx[n.ring] = 0;
    const angle  = (2 * Math.PI * ringIdx[n.ring]++) / ringCounts[n.ring];
    const radius = 120 + n.ring * 220;
    n.x = radius * Math.cos(angle);
    n.y = radius * Math.sin(angle);
  });
})();

// ── Build vis-network nodes ───────────────────────────────────────────────────
const visNodes = new vis.DataSet(NODES_DATA.map(n => {
  const bg = nodeColor(n.verdict);
  return {
    id:    n.id,
    label: 'S:' + n.Sharpe.toFixed(2) + '\\nIC:' + n.IC_mean.toFixed(3),
    x:     n.x,
    y:     n.y,
    shape: 'circle',
    size:  44,
    color: {
      background: bg,
      border: bg,
      highlight: { background: bg, border: '#333333' },
    },
    font: { color: '#ffffff', size: 12, face: 'Inter', multi: false },
    borderWidth: 3,
    borderWidthSelected: 4,
    shapeProperties: { borderDashes: n.verdict === 'failed' ? [5, 3] : false },
  };
}));

// ── Build vis-network edges ───────────────────────────────────────────────────
const visEdges = new vis.DataSet(EDGES_DATA.map((e, i) => ({
  id:    i,
  from:  e.from,
  to:    e.to,
  label: e.mutation ? e.mutation.substring(0, 35) : '',
  font:  { size: 12, color: '#5a6480', align: 'middle', strokeWidth: 2, strokeColor: '#ffffff' },
  color: { color: '#e67e22', highlight: '#d35400' },
  width: 2.5,
  arrows: 'to',
  smooth: { type: 'curvedCW', roundness: 0.12 },
})));

// ── Initialise network ────────────────────────────────────────────────────────
const container = document.getElementById('mynetwork');
const network = new vis.Network(
  container,
  { nodes: visNodes, edges: visEdges },
  {
    physics: {
      enabled: true,
      repulsion: {
        nodeDistance: 180,
        springLength: 200,
        springConstant: 0.04,
        damping: 0.15,
      },
      solver: 'repulsion',
      stabilization: { iterations: 200, updateInterval: 25 },
    },
    layout: { improvedLayout: false },
    interaction: { hover: false, tooltipDelay: 9999 },
  }
);

// ── Ring layer boundaries ─────────────────────────────────────────────────────
(function() {
  const maxRing = Math.max(...NODES_DATA.map(n => n.ring));
  if (maxRing === 0) return;

  network.on('beforeDrawing', function(ctx) {
    ctx.save();
    ctx.setLineDash([14, 7]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(120, 140, 200, 0.30)';
    for (let ring = 0; ring <= maxRing; ring++) {
      const r = 120 + ring * 220 + 60;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.restore();
  });
})();

// ── Panel open / close ────────────────────────────────────────────────────────
const detailPane    = document.getElementById('detail-pane');
const resizeHandle  = document.getElementById('resize-handle');

function openPanel() {
  if (detailPane.classList.contains('open')) return;
  detailPane.classList.add('open');
  resizeHandle.style.display = 'block';
  detailPane.addEventListener('transitionend', function h(e) {
    if (e.propertyName !== 'width') return;
    network.redraw();
    detailPane.removeEventListener('transitionend', h);
  });
}

function closePanel() {
  if (!detailPane.classList.contains('open')) return;
  detailPane.classList.remove('open');
  resizeHandle.style.display = 'none';
  detailPane.addEventListener('transitionend', function h(e) {
    if (e.propertyName !== 'width') return;
    network.fit({ animation: { duration: 350, easingFunction: 'easeInOutQuad' } });
    detailPane.removeEventListener('transitionend', h);
  });
}

// ── Resize handle drag ────────────────────────────────────────────────────────
(function() {
  let dragging = false;
  let startX   = 0;
  let startW   = 0;

  resizeHandle.addEventListener('mousedown', function(e) {
    if (!detailPane.classList.contains('open')) return;
    dragging = true;
    startX   = e.clientX;
    startW   = detailPane.offsetWidth;
    detailPane.classList.add('resizing');
    resizeHandle.classList.add('dragging');
    document.body.style.cursor    = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', function(e) {
    if (!dragging) return;
    const delta = startX - e.clientX;
    const newW  = Math.min(700, Math.max(220, startW + delta));
    detailPane.style.width = newW + 'px';
    document.getElementById('detail-inner').style.minWidth = newW + 'px';
  });

  document.addEventListener('mouseup', function() {
    if (!dragging) return;
    dragging = false;
    detailPane.classList.remove('resizing');
    resizeHandle.classList.remove('dragging');
    document.body.style.cursor     = '';
    document.body.style.userSelect = '';
    network.redraw();
  });
})();

function clearDetail() {
  document.getElementById('hdr-id').textContent    = '—';
  document.getElementById('hdr-badge').textContent = '';
  document.getElementById('hdr-badge').style.background = '#dde1e7';
  document.getElementById('hdr-badge').style.color      = '#7a8499';
  document.getElementById('detail-body').innerHTML = '';
}

document.getElementById('close-btn').addEventListener('click', function() {
  network.unselectAll();
  clearDetail();
  closePanel();
});

// ── Click handlers ────────────────────────────────────────────────────────────
network.on('selectNode', params => {
  const n = NODES_DATA.find(x => x.id === params.nodes[0]);
  if (n) { openPanel(); renderDetail(n); }
});
network.on('deselectNode', function() { clearDetail(); closePanel(); });

function renderDetail(n) {
  // Header
  document.getElementById('hdr-id').textContent = n.alpha_id;
  const badge = document.getElementById('hdr-badge');
  badge.textContent       = n.verdict.toUpperCase();
  badge.style.background  = nodeColor(n.verdict);
  badge.style.color       = '#fff';

  // Warn classes
  const wIC   = warnClass(n.IC_mean,      T.ic_mean_soft);
  const wICIR = warnClass(n.ICIR,         T.icir_soft);
  const wMono = warnClass(n.monotonicity,  T.mono_soft);
  const wSharpe   = warnClass(n.Sharpe,       T.sharpe_soft, T.sharpe_hard);
  const wTurnover = highClass(n.turnover,     T.turnover_max);
  const wDD       = warnClass(n.max_drawdown, T.drawdown_soft, T.drawdown_hard);

  let html = '';

  html += '<div class="sec">ID / Batch</div>'
        + '<div class="prose">' + esc(n.original_id)
        + (n.batch_id ? ' &middot; ' + esc(n.batch_id) : '')
        + '</div>';

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
