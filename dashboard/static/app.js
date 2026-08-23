/* Friday dashboard front end.
   Everything arrives over SSE; nothing here polls except the light /api/state
   heartbeat that keeps the header dot and the active mood honest. */
'use strict';

const $ = (id) => document.getElementById(id);
const log = $('log'), dot = $('dot');

const TYPES = ['session_init', 'thinking', 'tool_call', 'tool_result',
               'text', 'done'];
const LABEL = {
  session_init: 'Session Init', thinking: 'Thinking', tool_call: 'Tool Call',
  tool_result: 'Tool Result', text: 'Text', done: 'Done',
};

let hidden = new Set(JSON.parse(localStorage.getItem('friday.hidden') || '[]'));
let autoscroll = true;          // suspended while the pointer is over the log
let activeMood = null;
let connected = false;          // the dot tracks THIS, nothing else
let camState = {};              // last known availability, to spot hot-plugs

/* ---------- helpers ---------- */

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const clock = (ts) => new Date((ts || Date.now() / 1000) * 1000)
  .toLocaleTimeString('en-GB', { hour12: false });

/* Minimal markdown: **bold**, `code`, and - bullet lists. Deliberately not a
   full parser -- everything is escaped first, so no HTML can come through. */
function md(src) {
  const lines = String(src ?? '').split('\n');
  let out = '', list = false;
  for (const raw of lines) {
    const line = esc(raw);
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    if (bullet) {
      if (!list) { out += '<ul>'; list = true; }
      out += `<li>${inline(bullet[1])}</li>`;
    } else {
      if (list) { out += '</ul>'; list = false; }
      out += line.trim() ? `<div>${inline(line)}</div>` : '<div>&nbsp;</div>';
    }
  }
  if (list) out += '</ul>';
  return out;
}
const inline = (s) => s
  .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
  .replace(/`(.+?)`/g, '<code>$1</code>');

/* Long JSON folds away so one huge tool result cannot bury the stream. */
function jsonBlock(value) {
  const text = typeof value === 'string'
    ? value : JSON.stringify(value, null, 2);
  const big = text.split('\n').length > 12 || text.length > 700;
  const pre = `<pre class="json">${esc(text)}</pre>`;
  if (!big) return pre;
  return `<details class="fold"><summary>${text.split('\n').length} lines — expand</summary>${pre}</details>`;
}

/* ---------- cards ---------- */

function card(ev) {
  const el = document.createElement('div');
  el.className = `card c-${ev.type}`;
  el.dataset.type = ev.type;
  if (hidden.has(ev.type)) el.style.display = 'none';

  const head = `<div class="cardhead">
      <span class="pill">${LABEL[ev.type] || ev.type}</span>
      ${ev.type === 'tool_call' || ev.type === 'tool_result'
        ? `<span class="toolname">${esc(ev.tool || '')}</span>` : ''}
      <span class="time">${clock(ev.ts)}</span>
    </div>`;

  let body = '';
  if (ev.type === 'session_init') {
    const servers = (ev.servers || []).map((s) =>
      `<span class="server ${s.status === 'connected' ? '' : 'down'}">
         <span class="d"></span>${esc(s.name)} · ${esc(s.status)}</span>`).join('');
    body = `<div class="model">${esc(ev.model || 'unknown model')}</div>
            ${servers ? `<div class="servers">${servers}</div>` : ''}`;
  } else if (ev.type === 'thinking') {
    body = `<div class="body">${esc(ev.text || '')}</div>`;
  } else if (ev.type === 'tool_call') {
    body = jsonBlock(ev.args ?? {});
  } else if (ev.type === 'tool_result') {
    const refs = (ev.refs || []).map((r) =>
      `<div class="ref">[tool ref: ${esc(r)}]</div>`).join('');
    body = jsonBlock(ev.result ?? '') + refs;
  } else if (ev.type === 'text') {
    body = `<div class="body md">${md(ev.text)}</div>`;
  } else if (ev.type === 'done') {
    const cost = (ev.cost_usd ?? 0).toFixed(4);
    const secs = (ev.elapsed_s ?? 0).toFixed(1);
    const turns = ev.turns ?? 0;
    body = `<div class="body md">${md(ev.text)}</div>
            <div class="footer">cost <b>$${cost}</b> &nbsp;&nbsp; time <b>${secs}s</b>
             &nbsp;&nbsp; turns <b>${turns} turn${turns === 1 ? '' : 's'}</b></div>`;
  }
  el.innerHTML = head + body;
  return el;
}

function append(ev) {
  const first = log.querySelector('.empty');
  if (first) first.remove();
  log.appendChild(card(ev));
  while (log.children.length > 400) log.removeChild(log.firstChild);
  if (autoscroll) log.scrollTop = log.scrollHeight;
}

/* ---------- filters ---------- */

function buildFilters() {
  const bar = $('filters');
  bar.innerHTML = '';
  for (const t of TYPES) {
    const b = document.createElement('button');
    b.className = 'filter' + (hidden.has(t) ? '' : ' on');
    b.textContent = LABEL[t];
    b.onclick = () => {
      hidden.has(t) ? hidden.delete(t) : hidden.add(t);
      localStorage.setItem('friday.hidden', JSON.stringify([...hidden]));
      b.classList.toggle('on');
      for (const c of log.querySelectorAll('.card'))
        c.style.display = hidden.has(c.dataset.type) ? 'none' : '';
    };
    bar.appendChild(b);
  }
}

/* ---------- moods ---------- */

async function buildMoods() {
  let data;
  try { data = await (await fetch('/api/emotions')).json(); }
  catch { return; }
  const grid = $('moods');
  grid.innerHTML = '';
  const names = [...data.emotions, data.yawn].filter(Boolean);
  for (const name of names) {
    const b = document.createElement('button');
    b.className = 'mood' + (name === data.yawn ? ' yawn' : '');
    b.textContent = name;
    b.dataset.mood = name;
    b.onclick = async () => {
      b.disabled = true;
      try {
        const r = await fetch('/api/mood', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        if (r.ok) { const j = await r.json(); setActive(j.mood); }
      } finally { b.disabled = false; }
    };
    grid.appendChild(b);
  }
}

function setActive(mood) {
  if (mood === activeMood) return;
  activeMood = mood;
  for (const b of document.querySelectorAll('.mood'))
    b.classList.toggle('active', b.dataset.mood === mood);
}

/* ---------- cameras ---------- */

function mountCam(which, boxId, available) {
  camState[which] = available;
  const box = $(boxId);
  box.innerHTML = '';
  if (!available) {
    box.innerHTML = `<div class="camoff">${which.toUpperCase()} CAM<br>not detected</div>`;
    return;
  }
  const img = new Image();
  img.alt = `${which} camera`;
  // cache-buster so a reconnect does not replay a dead stream
  img.src = `/cam/${which}?t=${Date.now()}`;
  img.onerror = () => {
    box.innerHTML = `<div class="camoff">${which.toUpperCase()} CAM<br>stream unavailable</div>`;
  };
  box.appendChild(img);
}

/* ---------- calibration ---------- */

async function loadCalibration() {
  let data;
  try { data = await (await fetch('/api/calibration')).json(); }
  catch { return; }
  for (const [k, v] of Object.entries(data)) {
    const input = $(k);
    if (input) input.value = v;
  }
}

function wireCalibration() {
  for (const btn of document.querySelectorAll('[data-save]')) {
    btn.onclick = async () => {
      const key = btn.dataset.save;
      const note = $('calnote');
      btn.disabled = true;
      try {
        const r = await fetch('/api/calibration', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ [key]: Number($(key).value) }),
        });
        note.textContent = r.ok
          ? `Saved ${key} at ${new Date().toLocaleTimeString()}`
          : `Could not save ${key}: ${await r.text()}`;
      } catch (e) { note.textContent = `Could not save ${key}: ${e}`; }
      finally { btn.disabled = false; }
    };
  }
}

/* ---------- stream ---------- */

function connect() {
  const es = new EventSource('/api/stream');
  es.onopen = () => { connected = true; paintDot(); };
  es.onerror = () => { connected = false; paintDot(); };
  es.onmessage = (m) => {
    try { append(JSON.parse(m.data)); } catch { /* keepalive */ }
  };
}

function paintDot(faceRunning) {
  dot.className = 'dot ' + (connected ? 'on' : 'off');
  if (faceRunning !== undefined) {
    dot.title = connected
      ? (faceRunning ? 'connected \u00b7 face running'
                     : 'connected \u00b7 face process not running')
      : 'not connected';
  }
}

async function heartbeat() {
  let s;
  try { s = await (await fetch('/api/state')).json(); }
  catch { connected = false; paintDot(); return; }
  setActive(s.mood);
  paintDot(s.face_running);
  // Cameras are hot-pluggable, so remount when one appears or disappears
  // rather than making the page reload to notice.
  const cams = s.cameras || {};
  for (const which of ['up', 'down']) {
    const now = !!cams[which]?.available;
    if (now !== camState[which]) mountCam(which, 'box-' + which, now);
  }
}

/* ---------- boot ---------- */

(async function init() {
  buildFilters();
  await buildMoods();
  await loadCalibration();
  wireCalibration();

  let state = {};
  try { state = await (await fetch('/api/state')).json(); } catch {}
  const cams = state.cameras || {};
  mountCam('up', 'box-up', cams.up?.available);
  mountCam('down', 'box-down', cams.down?.available);
  setActive(state.mood);

  connect();
  setInterval(heartbeat, 4000);

  // Pause auto-scroll while reading; resume on leave or when scrolled to end.
  log.addEventListener('mouseenter', () => { autoscroll = false; });
  log.addEventListener('mouseleave', () => { autoscroll = true; });
  log.addEventListener('scroll', () => {
    const atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    if (atEnd) autoscroll = true;
  });

  $('btn-clear').onclick = async () => {
    await fetch('/api/clear', { method: 'POST' });
    log.innerHTML = '<p class="empty">Cleared.</p>';
  };

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  }
})();
