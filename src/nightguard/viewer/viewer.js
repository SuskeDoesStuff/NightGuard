'use strict';
/*
 * NightGuard trace viewer. PROJECT.md 9.
 *
 * Consumes a JSONL trace: one header record, one record per sim tick, one footer. No server, no
 * build step, no dependencies.
 *
 * v0.3 subset. Out of scope until v1.2, because all three need `env/`: the belief-versus-truth
 * overlay, the policy probability panel, and the V(s) sparkline. Traces carry `belief` and
 * `policy` as null today, so every read of them is guarded.
 *
 * Two robustness requirements from 9.4 that are easy to get wrong:
 *   - a `--stride` subsampled trace must not break the timeline, so seeking is by array index and
 *     the tick is read from the record rather than assumed to equal the index;
 *   - a trace without a `policy` block must render, not throw.
 */

const NODE_LAYOUT = {
  // PROJECT.md 9.1's lane geometry. Column 1: the shared origin. Column 2: west pool (top),
  // COVE (middle), east chain (bottom). Column 3: the two corners. Column 4: the office.
  0:  { x: 60,  y: 120, label: 'STAGE' },
  1:  { x: 60,  y: 190, label: 'COMMONS' },
  3:  { x: 200, y: 40,  label: 'W_BACKSTAGE' },
  4:  { x: 280, y: 40,  label: 'W_CLOSET' },
  5:  { x: 200, y: 90,  label: 'W_HALL' },
  6:  { x: 430, y: 65,  label: 'W_CORNER' },
  2:  { x: 240, y: 170, label: 'COVE' },
  7:  { x: 175, y: 275, label: 'E_RESTROOMS' },
  8:  { x: 250, y: 275, label: 'E_KITCHEN' },
  9:  { x: 325, y: 275, label: 'E_HALL' },
  10: { x: 430, y: 275, label: 'E_CORNER' },
  11: { x: 600, y: 170, label: 'OFFICE' },
};

const EDGES = [
  [0, 1], [1, 5], [1, 7], [5, 6], [7, 8], [8, 9], [9, 10], [6, 11], [10, 11], [2, 11],
];

const ENTITY_STYLE = {
  warden:   { short: 'W', cls: 'east' },
  drifter:  { short: 'D', cls: 'west' },
  prowler:  { short: 'P', cls: 'east' },
  sprinter: { short: 'S', cls: 'cove' },
};

const state = {
  header: null,
  ticks: [],
  footer: null,
  index: 0,
  playing: false,
  timer: null,
};

const $ = (id) => document.getElementById(id);

/* --- parsing ------------------------------------------------------------------------------- */

function parseTrace(text) {
  const header = null;
  const out = { header: null, ticks: [], footer: null };
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    let record;
    try {
      record = JSON.parse(trimmed);
    } catch (err) {
      throw new Error(`malformed JSONL line: ${trimmed.slice(0, 80)}`);
    }
    if (record.type === 'header') out.header = record;
    else if (record.type === 'tick') out.ticks.push(record);
    else if (record.type === 'footer') out.footer = record;
  }
  if (!out.header) throw new Error('trace has no header record');
  if (!out.ticks.length) throw new Error('trace has no tick records');
  return out;
}

function nodeName(id) {
  if (id === null || id === undefined) return '—';
  const fromHeader = state.header && state.header.topology
    ? state.header.topology.find((n) => n.id === id)
    : null;
  if (fromHeader) return fromHeader.name;
  return NODE_LAYOUT[id] ? NODE_LAYOUT[id].label : `node ${id}`;
}

/* --- map ----------------------------------------------------------------------------------- */

function buildMap() {
  const svg = $('map');
  svg.innerHTML = '';
  const ns = 'http://www.w3.org/2000/svg';

  for (const [a, b] of EDGES) {
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', NODE_LAYOUT[a].x);
    line.setAttribute('y1', NODE_LAYOUT[a].y);
    line.setAttribute('x2', NODE_LAYOUT[b].x);
    line.setAttribute('y2', NODE_LAYOUT[b].y);
    // COVE reaches the office by a straight run between the two corners: SPRINTER bypasses the map.
    line.setAttribute('class', a === 2 ? 'edge bypass' : 'edge');
    svg.appendChild(line);
  }

  for (const [id, spec] of Object.entries(NODE_LAYOUT)) {
    const group = document.createElementNS(ns, 'g');
    group.setAttribute('class', 'node');
    group.dataset.node = id;

    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('x', spec.x - 46);
    rect.setAttribute('y', spec.y - 15);
    rect.setAttribute('rx', 5);
    rect.setAttribute('width', 92);
    rect.setAttribute('height', 30);
    group.appendChild(rect);

    const text = document.createElementNS(ns, 'text');
    text.setAttribute('x', spec.x);
    text.setAttribute('y', spec.y + 4);
    text.textContent = spec.label;
    group.appendChild(text);

    const tokens = document.createElementNS(ns, 'text');
    tokens.setAttribute('x', spec.x);
    tokens.setAttribute('y', spec.y + 27);
    tokens.setAttribute('class', 'tokens');
    tokens.dataset.tokens = id;
    group.appendChild(tokens);

    svg.appendChild(group);
  }
}

function paintMap(record) {
  const occupants = {};
  for (const [name, style] of Object.entries(ENTITY_STYLE)) {
    if (name === 'sprinter') continue;
    const node = record.entities[name] ? record.entities[name].node : null;
    if (node === null || node === undefined) continue;
    (occupants[node] = occupants[node] || []).push(style.short);
  }
  // SPRINTER has no position; show it at COVE with its stage, per 9.1's wireframe.
  const sprinter = record.entities.sprinter;
  if (sprinter) {
    (occupants[2] = occupants[2] || []).push(`S${sprinter.stage}${sprinter.armed ? '!' : ''}`);
  }

  for (const id of Object.keys(NODE_LAYOUT)) {
    const el = document.querySelector(`[data-tokens="${id}"]`);
    if (el) el.textContent = occupants[id] ? `(${occupants[id].join(' ')})` : '';
    const group = document.querySelector(`.node[data-node="${id}"]`);
    if (!group) continue;
    const selected = record.monitor.up && record.monitor.cam === Number(id);
    group.classList.toggle('selected', Boolean(selected));
  }
}

/* --- panels -------------------------------------------------------------------------------- */

function rows(target, pairs) {
  const el = $(target);
  el.innerHTML = '';
  for (const [key, value, cls] of pairs) {
    const dt = document.createElement('dt');
    dt.textContent = key;
    const dd = document.createElement('dd');
    dd.textContent = value;
    if (cls) dd.className = cls;
    el.appendChild(dt);
    el.appendChild(dd);
  }
}

function doorLabel(closed, jammed) {
  if (jammed) return 'JAMMED';
  return closed ? 'CLOSED' : 'open';
}

function clockFor(record) {
  // 12AM to 6AM over the night; the header hour field is authoritative.
  const hour = record.hour === 0 ? 12 : record.hour;
  return `${String(hour).padStart(2, '0')}:00`;
}

function paintPanels(record) {
  const jam = record.jams || [false, false];
  rows('office', [
    ['Left door', doorLabel(record.doors[0], jam[0]), record.doors[0] || jam[0] ? 'on' : ''],
    ['Right door', doorLabel(record.doors[1], jam[1]), record.doors[1] || jam[1] ? 'on' : ''],
    ['Lights', `${record.lights[0] ? 'L' : '-'} ${record.lights[1] ? 'R' : '-'}`, ''],
    ['Monitor', record.monitor.up ? `up — ${nodeName(record.monitor.cam)}` : 'down',
      record.monitor.up ? 'on' : ''],
  ]);

  const audio = record.audio || {};
  const cues = Object.keys(audio).filter((k) => audio[k]);
  rows('entities', [
    ['WARDEN', `${nodeName(record.entities.warden.node)}` +
      (record.entities.warden.countdown_ticks !== null
        ? ` — countdown ${record.entities.warden.countdown_ticks}` : '')],
    ['DRIFTER', nodeName(record.entities.drifter.node)],
    ['PROWLER', nodeName(record.entities.prowler.node)],
    ['SPRINTER', `stage ${record.entities.sprinter.stage}` +
      (record.entities.sprinter.armed ? ' — ARMED' : '') +
      ` — ${record.entities.sprinter.bangs} bangs`],
    ['Audio', cues.length ? cues.join(', ') : '—', cues.length ? 'on' : ''],
  ]);

  const metrics = record.metrics || {};
  const pairs = [
    ['Blackout', record.blackout ? record.blackout : '—',
      record.blackout ? 'alert' : ''],
    ['Cam duty', metrics.cam_duty !== null && metrics.cam_duty !== undefined
      ? metrics.cam_duty.toFixed(2) : '—'],
    ['Event', record.event || '—', record.event ? 'on' : ''],
    ['Seed', state.header.seed === null ? '—' : state.header.seed],
  ];
  if (state.footer) {
    pairs.push(['Cause', state.footer.cause || '—', 'cause']);
    pairs.push(['Ended at tick', state.footer.terminated_at]);
  }
  // belief / policy are null until v1.0; say so rather than rendering an empty panel.
  if (record.belief === null || record.policy === null) {
    pairs.push(['Belief / policy', 'not in trace (needs env/, v1.0)']);
  }
  rows('diagnostics', pairs);
}

/* --- transport ----------------------------------------------------------------------------- */

function seek(index) {
  if (!state.ticks.length) return;
  state.index = Math.max(0, Math.min(state.ticks.length - 1, index));
  const record = state.ticks[state.index];

  $('night').textContent = state.header.night;
  $('clock').textContent = clockFor(record);
  $('step').textContent = `${state.index + 1} / ${state.ticks.length}`;
  const pct = Math.max(0, Math.min(100, record.power));
  $('power').textContent = `${record.power.toFixed(1)}%`;
  const fill = $('power-fill');
  fill.style.width = `${pct}%`;
  fill.className = pct > 50 ? 'high' : pct > 20 ? 'mid' : 'low';
  $('tick').textContent = `t ${record.t}`;
  $('scrub').value = String(state.index);

  paintMap(record);
  paintPanels(record);
}

function buildMarkers() {
  const holder = $('markers');
  holder.innerHTML = '';
  const total = state.ticks.length - 1 || 1;
  state.ticks.forEach((record, index) => {
    if (!record.event) return;
    const mark = document.createElement('span');
    mark.className = 'marker';
    mark.style.left = `${(index / total) * 100}%`;
    mark.title = `${record.event} @ t${record.t}`;
    mark.addEventListener('click', () => seek(index));
    holder.appendChild(mark);
  });
}

function play(on) {
  state.playing = on;
  $('play').innerHTML = on ? '&#10073;&#10073;' : '&#9654;';
  if (state.timer) clearInterval(state.timer);
  if (!on) return;
  state.timer = setInterval(() => {
    if (state.index >= state.ticks.length - 1) return play(false);
    seek(state.index + 1);
  }, 16);
}

function load(text) {
  const parsed = parseTrace(text);
  state.header = parsed.header;
  state.ticks = parsed.ticks;
  state.footer = parsed.footer;
  state.index = 0;
  $('scrub').max = String(state.ticks.length - 1);
  buildMap();
  buildMarkers();
  seek(0);
  const cause = state.footer ? state.footer.cause : 'no footer';
  $('status').textContent =
    `${state.ticks.length} tick records — ${cause}` +
    (state.ticks.length > 1 && state.ticks[1].t - state.ticks[0].t > 1
      ? ` — stride ${state.ticks[1].t - state.ticks[0].t}` : '');
}

document.addEventListener('DOMContentLoaded', () => {
  buildMap();

  $('file').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        load(String(reader.result));
      } catch (err) {
        $('status').textContent = `Could not load trace: ${err.message}`;
      }
    };
    reader.readAsText(file);
  });

  $('scrub').addEventListener('input', (event) => seek(Number(event.target.value)));
  $('play').addEventListener('click', () => play(!state.playing));
  $('back').addEventListener('click', () => seek(state.index - 1));
  $('fwd').addEventListener('click', () => seek(state.index + 1));

  document.addEventListener('keydown', (event) => {
    const jump = event.shiftKey ? 10 : 1;
    if (event.code === 'Space') { event.preventDefault(); play(!state.playing); }
    else if (event.code === 'ArrowLeft') seek(state.index - jump);
    else if (event.code === 'ArrowRight') seek(state.index + jump);
  });

  // Convenience: if trace.jsonl sits next to this file and the page is served, load it.
  fetch('trace.jsonl')
    .then((response) => (response.ok ? response.text() : Promise.reject(new Error('no trace'))))
    .then(load)
    .catch(() => {});
});
