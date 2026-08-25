/* Friday dashboard front end.
   Everything arrives over SSE; nothing here polls except the light /api/state
   heartbeat that keeps the header, the active mood and camera availability
   honest. */
'use strict';

const $ = (id) => document.getElementById(id);
const log = $('log'), dot = $('dot'), statustext = $('statustext');

/* --- opening sequence ------------------------------------------------
   Six seconds: neutral face, wink, welcome. The CSS runs the whole thing;
   this only seeds the confetti and makes sure the overlay is REMOVED
   afterwards -- a full-screen element left lying around would swallow
   every tap on the dashboard underneath it. */
(function splash() {
  const el = $('splash');
  if (!el) return;

  const field = el.querySelector('.confetti');
  if (field) {
    const colours = ['#FF6A00', '#ffb066', '#ffffff', '#7cc4ff', '#ffd84d'];
    let html = '';
    for (let i = 0; i < 18; i++) {
      const left = Math.round(Math.random() * 96);
      const delay = (Math.random() * 0.5).toFixed(2);
      const colour = colours[i % colours.length];
      html += `<i style="left:${left}%;background:${colour};` +
              `animation-delay:${4.35 + Number(delay)}s"></i>`;
    }
    field.innerHTML = html;
  }

  const done = () => el.classList.add('gone');
  const timer = setTimeout(done, 6000);
  // Tapping skips it. Six seconds is pleasant once and tedious on the
  // fifth reload of the evening.
  el.addEventListener('click', () => { clearTimeout(timer); done(); });
})();

const TYPES = ['session_init', 'thinking', 'tool_call', 'tool_result',
               'text', 'done'];
const LABEL = {
  session_init: 'Session Init', thinking: 'Thinking', tool_call: 'Tool Call',
  tool_result: 'Tool Result', text: 'Text', done: 'Done',
};

const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; }
              catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

let hidden = new Set(store.get('friday.hidden', []));
let autoscroll = true;          // suspended while the pointer is over the log
let activeMood = null;
let connected = false;          // the dot tracks THIS, nothing else
/* Camera state comes from the Pi, not from this browser: the on/off switch
   is a real switch on the robot, so every phone looking at the dashboard
   sees the same thing. `present` is whether the device is on the USB bus;
   `enabled` is the switch. */
let cam = { up: { present: false, enabled: true },
            down: { present: false, enabled: true } };

/* ---------- helpers ---------- */

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const clock = (ts) => new Date((ts || Date.now() / 1000) * 1000)
  .toLocaleTimeString('en-GB', { hour12: false });

let toastTimer = null;
function toast(msg, bad) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast show' + (bad ? ' bad' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = 'toast'; }, 2600);
}

/* Minimal markdown: **bold**, `code`, and - bullet lists. Deliberately not a
   full parser -- everything is escaped first, so no HTML can come through. */
function md(src) {
  const lines = esc(src).split('\n');
  let out = '', list = false;
  for (const line of lines) {
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
  if (autoscroll) log.scrollTop = log.scrollHeight;
}

/* ---------- filters ---------- */

function buildFilters() {
  const bar = $('filters');
  bar.innerHTML = '';
  for (const t of TYPES) {
    const b = document.createElement('button');
    b.textContent = LABEL[t];
    b.className = hidden.has(t) ? 'off' : '';
    b.onclick = () => {
      hidden.has(t) ? hidden.delete(t) : hidden.add(t);
      store.set('friday.hidden', [...hidden]);
      b.className = hidden.has(t) ? 'off' : '';
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
    b.title = name === data.yawn
      ? 'Plays the 4s yawn once, then returns to the current mood'
      : `Show ${name} on the face`;
    b.onclick = async () => {
      b.disabled = true;
      try {
        const r = await fetch('/api/mood', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });
        if (r.ok) { const j = await r.json(); setActive(j.mood); toast(name); }
        else toast(await r.text(), true);
      } catch (e) { toast(String(e), true); }
      finally { b.disabled = false; }
    };
    grid.appendChild(b);
  }
}

/* ---------- the 7" panel ----------
   The dashboard does not draw on the panel and does not keep its own copy of
   the schedule. Everything here POSTs to /api/screen, which writes the same
   state files the voice path writes -- so an alarm set from the phone is the
   same row Friday can cancel by voice, and vice versa. */

const TIMER_STEPS = [1, 5, 10, 20];
let screenApp = null;

function buildScreen() {
  const apps = $('screenapps');
  if (!apps) return;

  // The face is the resting state, so it is the first choice, not a
  // "cancel" hidden somewhere else.
  const buttons = [['face', 'Face'], ['clock', 'Clock']];
  apps.innerHTML = '';
  for (const [name, text] of buttons) {
    const b = document.createElement('button');
    b.className = 'seg';
    b.dataset.app = name;
    b.textContent = text;
    b.onclick = () => screenPost(name === 'face'
      ? { action: 'dismiss' } : { action: 'launch', app: name });
    apps.appendChild(b);
  }

  const seg = $('timerseg');
  seg.innerHTML = '';
  for (const mins of TIMER_STEPS) {
    const b = document.createElement('button');
    b.className = 'seg';
    b.textContent = mins + 'm';
    b.onclick = () => screenPost({ action: 'timer',
                                   duration: mins + ' minutes' });
    seg.appendChild(b);
  }

  $('setalarm').onclick = () => {
    const time = $('alarmtime').value;
    if (!time) { toast('Pick a time first', true); return; }
    screenPost({ action: 'alarm', time, label: $('alarmlabel').value });
  };
}

async function screenPost(body) {
  try {
    const r = await fetch('/api/screen', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    const j = await r.json();
    toast(j.result);
    $('alarmlabel').value = '';
    paintScreen(j);
  } catch (e) { toast(String(e), true); }
}

function paintScreen(state) {
  if (!state) return;
  screenApp = state.app || null;
  $('screennow').textContent = screenApp || 'face';
  for (const b of document.querySelectorAll('#screenapps .seg'))
    b.classList.toggle('on', b.dataset.app === (screenApp || 'face'));

  const list = $('pending');
  const items = state.pending || [];
  if (!items.length) {
    list.innerHTML = '<li class="none">Nothing set</li>';
    return;
  }
  list.innerHTML = '';
  for (const item of items) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="kind">${esc(item.kind)}</span>` +
      `<span class="when">${esc(item.kind === 'timer'
        ? countdown(item.in_s) : item.when)}</span>` +
      `<span class="lbl">${esc(item.label || '')}</span>`;
    const kill = document.createElement('button');
    kill.className = 'kill';
    kill.title = 'Cancel';
    kill.textContent = '×';
    kill.onclick = () => screenPost({ action: 'cancel', id: item.id });
    li.appendChild(kill);
    list.appendChild(li);
  }
}

const countdown = (secs) => {
  const total = Math.max(0, Math.round(secs));
  const m = Math.floor(total / 60), s = total % 60;
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}:${String(s).padStart(2, '0')}`;
};

/* ---------- drive ----------
   Nothing here decides whether a move is safe. Every button POSTs to
   /api/motor, which calls the friday-robot MCP server, which runs the
   clearance check and the duration cap. A refusal comes back as a normal
   200 with an honest sentence in it -- that is a working robot, not an
   error, so it is shown as a note rather than as a failure. */

const DRIVE_STEPS = [0.3, 0.6, 1.0];
let driveStep = 0.6;
let driveBusy = false;

function buildDrive() {
  const seg = $('stepseg');
  if (!seg) return;
  seg.innerHTML = '';
  for (const secs of DRIVE_STEPS) {
    const b = document.createElement('button');
    b.className = 'seg' + (secs === driveStep ? ' on' : '');
    b.textContent = secs + 's';
    b.onclick = () => {
      driveStep = secs;
      for (const o of seg.querySelectorAll('.seg'))
        o.classList.toggle('on', o === b);
    };
    seg.appendChild(b);
  }

  for (const b of document.querySelectorAll('.dbtn'))
    b.onclick = () => drive(b.dataset.dir, b);

  const speed = $('drivespeed');
  if (speed) {
    speed.oninput = () => { $('speedval').textContent = speed.value + '%'; };
    speed.onchange = async () => {
      try {
        const r = await fetch('/api/motor/speed', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ percent: Number(speed.value) }),
        });
        const j = await r.json();
        note(j.result, !r.ok);
      } catch (e) { note(String(e), true); }
    };
  }
}

function note(text, bad) {
  const el = $('drivenote');
  if (el) { el.textContent = text; el.classList.toggle('bad', !!bad); }
  $('drivestate').textContent = bad ? 'error' : 'ok';
}

async function drive(direction, button) {
  // STOP must never be blocked by an in-flight move -- that is the one
  // button whose whole purpose is to interrupt.
  if (driveBusy && direction !== 'stop') return;
  if (direction !== 'stop') driveBusy = true;
  button && button.classList.add('firing');
  try {
    const r = await fetch('/api/motor', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ direction, seconds: driveStep }),
    });
    const j = await r.json();
    note(j.result, !r.ok);
  } catch (e) {
    note(String(e), true);
  } finally {
    driveBusy = false;
    button && button.classList.remove('firing');
  }
}

function setActive(mood) {
  if (mood === activeMood) return;
  activeMood = mood;
  $('moodnow').textContent = mood || '—';
  for (const b of document.querySelectorAll('.mood'))
    b.classList.toggle('active', b.dataset.mood === mood);
}

/* ---------- cameras ----------
   Switching a camera off POSTs to the Pi, which kills that camera's ffmpeg
   and releases the V4L2 handle. So the switch really frees the CPU -- it is
   not just hiding a picture in this one browser. */

function renderCam(which) {
  const box = $('box-' + which);
  const { present, enabled } = cam[which];
  box.innerHTML = '';

  if (!enabled) {
    box.innerHTML = `<div class="camoff"><b>Off</b>
      <span class="why">Switched off. No stream, and no CPU used.</span></div>`;
    return;
  }
  if (!present) {
    box.innerHTML = `<div class="camoff"><b>No signal</b>
      <span class="why">Not on the USB bus. Check the cable — it comes back
      on its own.</span></div>`;
    return;
  }
  const img = new Image();
  img.alt = `${which} camera`;
  img.src = `/cam/${which}?t=${Date.now()}`;   // cache-buster per (re)mount
  img.onerror = () => {
    box.innerHTML = `<div class="camoff"><b>Stream lost</b>
      <span class="why">The device went away mid-stream.</span></div>`;
  };
  box.appendChild(img);
  const live = document.createElement('span');
  live.className = 'live';
  live.textContent = 'live';
  box.appendChild(live);
}

function paintToggle(which) {
  const tg = $('tg-' + which);
  const on = cam[which].enabled;
  tg.setAttribute('aria-checked', on ? 'true' : 'false');
  tg.title = on ? `Turn the ${which} camera off`
                : `Turn the ${which} camera on`;
}

/* Apply a /api/state or /api/camera cameras block. The Pi is the authority on
   both halves: `present` is whether the device is on the USB bus, `enabled` is
   the switch -- keeping them apart is what lets the box say "switched off"
   rather than "no signal". Redraw only what actually changed, so a running
   stream is never torn down for no reason. */
function applyCams(cams) {
  for (const which of ['up', 'down']) {
    const s = cams[which];
    if (!s) continue;
    const before = cam[which];
    const now = { present: !!s.present, enabled: !!s.enabled };
    if (now.present !== before.present || now.enabled !== before.enabled) {
      cam[which] = now;
      paintToggle(which);
      renderCam(which);
    }
  }
}

function wireCams() {
  for (const which of ['up', 'down']) {
    paintToggle(which);
    $('tg-' + which).onclick = async () => {
      const tg = $('tg-' + which);
      const want = !cam[which].enabled;
      tg.disabled = true;
      try {
        const r = await fetch(`/api/camera/${which}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ on: want }),
        });
        if (!r.ok) { toast(await r.text(), true); return; }
        const j = await r.json();
        applyCams(j.cameras);
        toast(`${which} cam ${j.on ? 'on' : 'off'}`);
      } catch (e) { toast(String(e), true); }
      finally { tg.disabled = false; }
    };
  }
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
      btn.disabled = true;
      try {
        const r = await fetch('/api/calibration', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ [key]: Number($(key).value) }),
        });
        r.ok ? toast(`Saved ${key}`) : toast(await r.text(), true);
      } catch (e) { toast(String(e), true); }
      finally { btn.disabled = false; }
    };
  }
}

/* ---------- stream ---------- */

function paintDot(faceRunning) {
  dot.className = 'dot ' + (connected ? 'on' : 'off');
  statustext.textContent = connected ? 'live' : 'offline';
  if (faceRunning !== undefined) {
    $('status').title = connected
      ? (faceRunning ? 'connected · face running'
                     : 'connected · face process not running')
      : 'not connected';
  }
}

function connect() {
  const es = new EventSource('/api/stream');
  es.onopen = () => { connected = true; paintDot(); };
  es.onerror = () => { connected = false; paintDot(); };
  es.onmessage = (m) => {
    try { append(JSON.parse(m.data)); } catch { /* keepalive */ }
  };
}

async function heartbeat() {
  let s;
  try { s = await (await fetch('/api/state')).json(); }
  catch { connected = false; paintDot(); return; }
  setActive(s.mood);
  paintDot(s.face_running);
  applyCams(s.cameras || {});     // picks up hot-plugs and other viewers
  paintRobot(s.robot_online);
  paintScreen(s.screen);          // rides the same beat; no extra request
}

/* friday-mcp is a separate service from this one, so the dashboard opens and
   works whether or not the robot is running. Say which half is down rather
   than letting the drive buttons fail silently. */
function paintRobot(online) {
  const badge = $('robotoff');
  if (badge) badge.hidden = online !== false;
}

/* ---------- panes (phone) ---------- */

function showPane(id) {
  for (const p of ['pane-logs', 'pane-robot'])
    $(p).classList.toggle('show', p === id);
  for (const t of document.querySelectorAll('.tab'))
    t.classList.toggle('active', t.dataset.pane === id);
  store.set('friday.pane', id);
}

/* ---------- boot ---------- */

(async function init() {
  buildFilters();
  wireCams();
  await buildMoods();
  buildScreen();
  buildDrive();
  await loadCalibration();
  wireCalibration();

  let state = {};
  try { state = await (await fetch('/api/state')).json(); } catch {}
  paintRobot(state.robot_online);
  const cams = state.cameras || {};
  for (const which of ['up', 'down']) {
    cam[which] = { present: !!cams[which]?.present,
                   enabled: cams[which]?.enabled !== false };
    paintToggle(which);
    renderCam(which);
  }
  setActive(state.mood);
  paintScreen(state.screen);

  connect();
  setInterval(heartbeat, 4000);

  showPane(store.get('friday.pane', 'pane-logs'));
  for (const t of document.querySelectorAll('.tab'))
    t.onclick = () => showPane(t.dataset.pane);

  // Pause auto-scroll while reading; resume on leave or when scrolled to end.
  log.addEventListener('mouseenter', () => { autoscroll = false; });
  log.addEventListener('mouseleave', () => { autoscroll = true; });
  log.addEventListener('scroll', () => {
    const atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    if (atEnd) autoscroll = true;
  });

  $('btn-clear').onclick = async () => {
    try { await fetch('/api/clear', { method: 'POST' }); } catch {}
    log.innerHTML = `<div class="empty"><span class="pulse"></span>
      <p>Cleared</p><small>New events will appear here.</small></div>`;
    toast('Log cleared');
  };

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  }
})();
