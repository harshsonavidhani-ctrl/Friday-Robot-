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

const TYPES = ['heard', 'session_init', 'thinking', 'tool_call',
               'tool_result', 'text', 'done'];
const LABEL = {
  heard: 'Heard', session_init: 'Session Init', thinking: 'Thinking',
  tool_call: 'Tool Call', tool_result: 'Tool Result', text: 'Text',
  done: 'Done',
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
  } else if (ev.type === 'heard') {
    /* His words, not hers. Quoted, because a transcript is a claim about
       what was said and whisper is often wrong about it. */
    body = `<div class="body">“${esc(ev.text || '')}”</div>`
      + (ev.source && ev.source !== 'voice'
          ? `<div class="footer">${esc(ev.source)}</div>` : '');
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

/* Titles for the apps the Pi reports in /api/state.screen.apps. The LIST is
   the robot's, not the dashboard's -- adding a file to apps/ has to make the
   button appear here without anyone editing this file, which is the whole
   point of the apps/ directory. This map only supplies nicer labels, and an
   app missing from it still gets a button. */
const APP_TITLES = { clock: 'Clock', timer: 'Timer',
                     alarm: 'Alarm', weather: 'Weather' };

/* Apps that are a VIEW of something scheduled rather than something you can
   just put on the screen. Launching "timer" with no timer running shows an
   empty timer, which reads as a bug -- so they are offered only when there is
   one to look at. */
const SCHEDULED_APPS = new Set(['timer', 'alarm']);

let screenApp = null;
let screenApps = [];          // what the Pi says it has, verbatim
let screenKinds = new Set();  // kinds currently on the schedule

function buildScreen() {
  if (!$('screenapps')) return;

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

  $('settimer').onclick = () => {
    const mins = Number($('timermins').value);
    if (!mins || mins < 1) { toast('How many minutes?', true); return; }
    screenPost({ action: 'timer', duration: mins + ' minutes',
                 label: $('timerlabel').value });
  };

  $('dismissring').onclick = () => screenPost({ action: 'dismiss' });
}

/* --- explore mode ----------------------------------------------------
   Starts rl/modes/confident_explore for a chosen number of minutes. The mode's
   own settings are NOT editable from here on purpose -- they live in
   rl/modes/confident_explore.env, which is the single record of the run this
   was frozen from.

   ONLY THE FACE GOES DOWN DURING A RUN, since 2026-09-08 -- this dashboard
   stays up, because the stop button is in it. The cameras are the price: the
   Pi hands both /dev/video nodes to the robot for the duration, so the camera
   panel goes dark while she is out and comes back when she parks. That is the
   mode working, not a crash.

   THE BADGE MUST NOT SAY "OUT" FOR A ROBOT THAT IS NOT MOVING. Starting a run
   and exploring are ~45 s and several failure modes apart: every failure so
   far happened before the control loop, and a run can reach the loop and still
   sit reflex-stopped against a wall for its whole length. Both look identical
   from a panel that only knows whether systemd took the job, which is what
   "it says friday is turning on exploration but doesn't explore" was. */
const EXPLORE_STEPS = [5, 10, 20];
let exploreState = {};

/* Whether the Thinking box is ticked. A missing element reads as OFF, so a
   cached older page can never start a mode it has no control for. */
function thinkOn() {
  const box = $('explorethink');
  return !!(box && box.checked);
}

function buildExplore() {
  if (!$('exploreseg')) return;

  const seg = $('exploreseg');
  seg.innerHTML = '';
  for (const mins of EXPLORE_STEPS) {
    const b = document.createElement('button');
    b.className = 'seg';
    b.textContent = mins + 'm';
    b.onclick = () => explorePost({ action: 'start', minutes: mins, thinking: thinkOn() });
    seg.appendChild(b);
  }

  $('explorego').onclick = () => {
    const mins = Number($('exploremins').value);
    if (!mins || mins < 1) { toast('How many minutes?', true); return; }
    explorePost({ action: 'start', minutes: mins, thinking: thinkOn() });
  };

  $('explorestop').onclick = () => explorePost({ action: 'stop' });
}

async function explorePost(body) {
  try {
    const r = await fetch('/api/explore', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    const j = await r.json();
    toast(j.result);
    $('exploremins').value = '';
    paintExplore(j);
  } catch (e) { toast(String(e), true); }
}

/* ---------- map (read-only: dashboard/map_view.py) ---------- */


/* File 14 step 11 — Navigation panel. READ-ONLY: this only ever GETs.
   The plan is drawn in the SURVEY map's frame, which is a different map from
   the picture in the Map panel, so it is drawn separately and labelled. */
async function refreshNav() {
  if (!$('navnow')) return;
  let n;
  try { n = await (await fetch('/api/nav')).json(); } catch { return; }
  if (!n || !n.ok) { $('navnow').textContent = 'unavailable'; return; }

  const svg = $('navplan');
  const f = n.frame || {};
  if (!f.ok) {
    svg.innerHTML = '';
    $('navframe').textContent = f.why || 'No survey map yet.';
  } else {
    const parts = [];
    for (const m of n.landmarks) {
      if (m.u == null || !m.on_map) continue;
      const cx = (m.u * 100).toFixed(2), cy = (m.v * 100).toFixed(2);
      const rr = Math.max(1.5, (m.radius_m / Math.max(f.size_m[0], 0.1)) * 100);
      const cls = m.is_home ? ' home' : '';
      parts.push(`<circle class="ring${cls}" cx="${cx}" cy="${cy}" r="${rr.toFixed(2)}"/>`);
      parts.push(`<circle class="lm${cls}" cx="${cx}" cy="${cy}" r="1.6"/>`);
      parts.push(`<text x="${(+cx + 2.4).toFixed(2)}" y="${(+cy + 1.6).toFixed(2)}">${esc(m.name)}</text>`);
    }
    svg.innerHTML = parts.join('') ||
      '<text x="6" y="52">nothing named on this map yet</text>';
    $('navframe').textContent =
      `survey map ${f.size_m[0]} × ${f.size_m[1]} m · known ${f.known_m2} m² · ` +
      `${f.sweeps} sweeps · blue home, orange landmark · ` +
      `this is state/map.npz, not the Map panel's picture`;
  }

  $('navplaces').innerHTML = n.landmarks.length
    ? n.landmarks.map(m => `<span class="${m.is_home ? 'home' : ''}">${esc(m.name)}` +
        (m.from_home != null && !m.is_home ? ` · ${m.from_home} m from home` : '') +
        (m.visits ? ` · ${m.visits}×` : '') +
        `<button class="navgo" data-go="${esc(m.name)}">go</button></span>`).join('')
    : '<span>no places named yet</span>';
  for (const b of $('navplaces').querySelectorAll('.navgo')) {
    b.onclick = () => navSend({ action: 'go', landmark: b.dataset.go },
                              `sending her to ${b.dataset.go}…`);
  }
  if ($('navroundtrip')) $('navroundtrip').checked = !!n.round_trip;
  const pending = n.waiting_to_come_back;
  $('navcomeback').hidden = !pending;
  if (pending) $('navcomeback').textContent =
    `Come back now (noted ${pending.x.toFixed(1)}, ${pending.y.toFixed(1)})`;

  const last = n.last_return;
  $('navrth').textContent = last
    ? `Last return home: ${last.arrived ? 'arrived' : 'did not arrive'} ` +
      `${ago(last.at)} · ${last.seconds}s · ${last.attempts} attempt(s)` +
      (last.arrived ? '' : ` · ${last.why || ''}`)
    : 'No return home yet.';

  // Step 17: stopped for help. It only clears when he says "try again" (or
  // gives her a new order), so it is shown until then.
  const fs = n.failsafe;
  $('navfailsafe').hidden = !fs;
  if (fs) $('navfailsafe').textContent =
    `Stopped, waiting for help (${String(fs.mode).replace(/_/g, ' ')}` +
    (fs.label ? ` → ${fs.label}` : '') + `, ${ago(fs.at)}): ${fs.why || ''} ` +
    `· say “try again” when she is clear`;

  // Step 16: exploration that ran out of places and came back early.
  const early = n.early_return;
  $('navearly').hidden = !early;
  if (early) $('navearly').textContent = early.reason
    ? `Last early return: ${early.reason} ${ago(early.at)}` +
      (early.floor_m2 != null ? ` · ${early.floor_m2} m² mapped on the trip` : '')
    : `Last early return: frontiers ran out ${ago(early.at)} ` +
      `(${early.reads} readings in a row)`;

  // Everything navigation knows, saved permanently WITH the map it belongs to.
  const b = n.bundle || {};
  $('navbundle').textContent = b.saved
    ? `Saved with the map ${ago(b.at)}: ${b.places.length} place(s)` +
      (b.places.length ? ` (${b.places.join(', ')})` : '') +
      ` · ${b.trips_logged} trip(s) · ${b.history} earlier version(s) kept` +
      (b.same_map_as_live ? '' : ' · ⚠ the live map has changed since')
    : 'Not yet saved with the map.';

  // Stuck and blocked are different problems, so the log names which.
  const trouble = a => {
    const bits = [];
    if (a.stucks) bits.push(`stuck ×${a.stucks}` +
      (a.recoveries ? `, ${a.recoveries_freed || 0}/${a.recoveries} freed` : ''));
    if (a.blocks_seen) bits.push(`blocked ×${a.blocks_seen}`);
    if (a.retried) bits.push('retry');
    return bits.length ? ` · ${bits.join(' · ')}` : '';
  };
  const rows = n.attempts || [];
  $('navlog').innerHTML = rows.length
    ? rows.map(a => `<div><span>${esc(a.destination)}</span>` +
        `<span class="${a.arrived ? 'ok' : 'bad'}">` +
        `${a.arrived ? 'arrived' : esc(a.state)} · ${a.seconds}s · ${ago(a.at)}` +
        `${esc(trouble(a))}</span></div>`).join('')
    : '<div><span>no navigation attempts logged yet</span><span></span></div>';

  const live = rows[0];
  $('navnow').textContent = fs ? 'failsafe'
    : live ? (live.arrived ? 'arrived' : live.state) : 'idle';
}

/* One POST for every navigation action. It goes to the MCP server, never to
   the motors, so a tap gets the same clearance interrupt a spoken command
   does — see api_nav_post. Buttons are disabled while she drives so a second
   tap cannot queue a second errand. */
async function navSend(body, saying) {
  const note = $('navsaid');
  if (note) note.textContent = saying || 'asking her…';
  for (const b of document.querySelectorAll('#panel-nav button, #navroundtrip'))
    b.disabled = true;
  try {
    const r = await fetch('/api/nav', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const out = await r.json();
    if (note) note.textContent = out.ok
      ? String(out.result || 'done').split('\n')[0].slice(0, 160)
      : (out.why || 'she did not answer');
  } catch (e) {
    if (note) note.textContent = 'the dashboard could not reach her';
  } finally {
    for (const b of document.querySelectorAll('#panel-nav button, #navroundtrip'))
      b.disabled = false;
    refreshNav();
  }
}

function ago(when) {
  if (!when) return '';
  const s = Math.max(0, Math.round(Date.now() / 1000 - when));
  if (s < 60) return s + 's ago';
  if (s < 3600) return Math.round(s / 60) + 'm ago';
  if (s < 86400) return Math.round(s / 3600) + 'h ago';
  return Math.round(s / 86400) + 'd ago';
}

let mapVersion = null;
let mapLive = false;
async function refreshMap() {
  if (!$('mapimg')) return;
  let m;
  try { m = await (await fetch('/api/map')).json(); } catch { return; }
  if (!m.ok) { $('mapnow').textContent = 'no map'; return; }
  mapGeom = m.origin ? m : mapGeom;
  setMapLive(!!m.live);
  if (m.version !== mapVersion) {          // re-fetch the picture only when it changed
    mapVersion = m.version;
    $('mapimg').src = '/api/map.png?v=' + m.version;
  }
  $('mapnow').textContent = m.live ? '● LIVE' :
    (m.running ? 'last saved · run in progress' : 'saved ' + m.saved);
  $('mapnow').classList.toggle('maplive', !!m.live);
  $('mapnote').textContent =
    `${m.size_m[0]} × ${m.size_m[1]} m · known ${m.known_m2} m² · driven ${m.driven_m2} m² · ` +
    `${m.sweeps} sweeps · white free, black obstacle, orange driven, red her position, grid 1 m` +
    (m.live ? ' · updating as she drives' : '') +
    (m.running && !m.live ? ' · updates when this run ends' : '');
}

// Like a GPS while she is out: once a second, and only then. Idle, the heartbeat
// looks every 4 s, which is also how a run starting is noticed.
let mapLiveTimer = null, mapGpsTimer = null, mapGeom = null;
let gpsBusy = false, gpsTrail = [];
function setMapLive(on) {
  if (on === mapLive) return;
  mapLive = on;
  clearInterval(mapLiveTimer);
  clearInterval(mapGpsTimer);
  mapLiveTimer = on ? setInterval(refreshMap, 1000) : null;
  mapGpsTimer = on ? setInterval(moveMarker, 250) : null;
  gpsTrail = [];
  const g = $('mapgps');
  if (g) g.hidden = !on;
}

// HER POSITION IS NOT IN THE PICTURE WHILE SHE IS OUT. A picture takes seconds to
// reach a phone; this is a few bytes, four times a second, and the marker glides
// between them (the CSS transition), which is what makes it read as a GPS.
async function moveMarker() {
  if (gpsBusy || !mapGeom) return;
  gpsBusy = true;
  try {
    const p = await (await fetch('/api/map/pose')).json();
    if (!p.live) return;
    const x = (p.x - mapGeom.origin[0]) / mapGeom.extent_m[0] * 100;
    let y = (p.y - mapGeom.origin[1]) / mapGeom.extent_m[1] * 100;
    if (!mapGeom.mirrored) y = 100 - y;
    const mark = $('mapmark');
    mark.style.left = x + '%';
    mark.style.top = y + '%';
    // In the flipped picture +y runs down, so her heading is a plain clockwise angle.
    mark.style.transform = `translate(-50%,-50%) rotate(${mapGeom.mirrored ? p.theta : -p.theta}rad)`;
    gpsTrail.push(x.toFixed(2) + ',' + y.toFixed(2));
    if (gpsTrail.length > 2400) gpsTrail.shift();
    $('maptrail').setAttribute('points', gpsTrail.join(' '));
  } catch (e) {
    /* the next tick tries again */
  } finally {
    gpsBusy = false;
  }
}

// ---- Maps: every navigation map she has (read-only) ----------------------
let mapsKey = null;
async function refreshMaps() {
  if (!$('mapslist')) return;
  let r;
  try { r = await (await fetch('/api/maps')).json(); } catch { return; }
  if (!r.ok) { $('mapsnow').textContent = 'unavailable'; return; }
  const key = r.maps.map(m => m.id + ':' + m.version).join('|');
  $('mapsnow').textContent = r.maps.length + ' maps';
  if (key === mapsKey) return;             // redraw only when something changed
  mapsKey = key;
  const list = $('mapslist');
  list.textContent = '';
  let group = null;
  for (const m of r.maps) {
    if (m.group !== group) {
      group = m.group;
      const h = document.createElement('h3');
      h.className = 'mapsgroup';
      h.textContent = group;
      list.appendChild(h);
    }
    const card = document.createElement('button');
    card.className = 'mapcard';
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.alt = m.label;
    img.src = '/api/maps/png?id=' + encodeURIComponent(m.id) + '&v=' + m.version;
    const cap = document.createElement('span');
    cap.textContent = `${m.label} · ${ago(m.saved_at)}` +
      (m.walls_m2 !== undefined ? ` · walls ${m.walls_m2} m² · free ${m.free_m2} m²` : '');
    card.append(img, cap);
    card.onclick = () => {
      $('mapsbig').src = img.src;
      $('mapsbig').hidden = false;
      $('mapsbignote').textContent = `${m.group}: ${cap.textContent}`;
    };
    list.appendChild(card);
  }
}

async function refreshExplore() {
  try { paintExplore(await (await fetch('/api/explore')).json()); }
  catch { /* the dashboard is stopped during a run; the panel keeps its text */ }
}

function paintExplore(state) {
  if (!state || !$('explorenow')) return;
  exploreState = state;
  const running = !!state.running;

  const prog = state.progress || {};
  const badge = $('explorenow');
  const forMins = state.minutes ? ` · ${state.minutes}m` : '';
  // THE BADGE SAYS WHICH MODE SHE IS OUT IN, because from the outside a
  // thinking run and a plain one look identical -- she drives either way, and
  // the only visible difference is where she chooses to go.
  const brain = state.thinking ? ' · thinking' : '';
  badge.textContent = !running ? 'idle'
    : prog.phase === 'driving' ? `driving${forMins}${brain}`
    : `starting${forMins}${brain}`;
  badge.classList.toggle('urgent', running);

  $('explorestop').hidden = !running;
  // No second way to start a run while one is going: the ports are exclusive
  // and the second run would abort on them anyway.
  for (const b of $('exploreseg').children) b.disabled = running;
  $('explorego').disabled = running;
  $('exploremins').disabled = running;
  if ($('explorethink')) $('explorethink').disabled = running;

  const last = state.last || {};
  let wrong = false;                // red note: she is out and going nowhere,
                                    // or the last run did not finish
  if (running && prog.phase === 'driving') {
    // "moving" is her ACTUAL commanded output, not what the policy asked for.
    // A robot that reflex-stops every tick is asking to drive and going
    // nowhere, and that is the case worth naming out loud.
    const ticks = prog.ticks ? `${prog.ticks} ticks` : 'driving';
    const held = prog.held_pct;
    // Stuck is not only "commanded nothing". A robot pivoting on the spot
    // against a wall is commanding plenty and exploring none of it, which is
    // what 550 reflex-stopped ticks looked like on 2026-09-08.
    const stuck = prog.moving === false || held >= 80;
    wrong = stuck;
    $('explorenote').textContent = stuck
      ? `Out but going nowhere — ${ticks}, ` +
        (held != null ? `stopped on ${held}% of the last few. ` : '') +
        'Something is right in front of her. Move her clear and she picks up again.'
      : `Out and driving — ${ticks}. The face is down until she parks.`;
  } else if (running) {
    $('explorenote').textContent =
      'Starting — policy, voice, the map and relocalisation take about 45 s ' +
      'before the wheels do anything. The face is down until she parks.';
  } else if (last.state) {
    const when = last.ended ? new Date(last.ended).toLocaleTimeString(
      'en-GB', { hour12: false, hour: '2-digit', minute: '2-digit' }) : '';
    // 134 is the known teardown SIGABRT, which happens after everything is
    // written. The mode script already classifies it; don't paint it as red.
    const ok = last.state === 'completed'
            || last.state === 'completed_with_teardown_abort';
    // A RUN THAT DIED IN ITS FIRST MINUTE MUST NOT LOOK LIKE ONE THAT FINISHED.
    // The badge drops back to "idle" either way, so without this the three
    // 55-second deaths of 2026-09-08 18:38-18:41 were indistinguishable from a
    // completed 20-minute run unless you went and read explore_mode.json.
    wrong = !ok;
    $('explorenote').textContent =
      `Last run ${when}: ${last.state.replace(/_/g, ' ')}` +
      (ok ? '.' : ` (exit ${last.exit_code}) — she stopped early, ` +
                  'nothing was explored. Press again.');
  } else {
    $('explorenote').textContent =
      'Runs the saved confident_explore mode. Pick a length, or type one.';
  }
  $('explorenote').classList.toggle('bad', wrong);

  const map = state.map || {};
  $('exploremap').textContent = map.explored_m2
    ? `Map vaulted: ${map.explored_m2} m² — kept permanently, never deleted.`
    : '';
}

/* The app row is rebuilt from the robot's own list, so it has to survive that
   list changing under it. */
function paintApps() {
  const row = $('screenapps');
  if (!row) return;

  // The face is the resting state, so it is the first choice, not a "cancel"
  // hidden somewhere else.
  const wanted = [['face', 'Face']];
  for (const name of screenApps) {
    if (SCHEDULED_APPS.has(name) && !screenKinds.has(name)) continue;
    wanted.push([name, APP_TITLES[name] || name]);
  }

  const key = wanted.map((w) => w[0]).join(',');
  if (row.dataset.key !== key) {
    row.dataset.key = key;
    row.innerHTML = '';
    for (const [name, text] of wanted) {
      const b = document.createElement('button');
      b.className = 'seg';
      b.dataset.app = name;
      b.textContent = text;
      b.onclick = () => screenPost(name === 'face'
        ? { action: 'dismiss' } : { action: 'launch', app: name });
      row.appendChild(b);
    }
  }
  for (const b of row.children)
    b.classList.toggle('on', b.dataset.app === (screenApp || 'face'));
}

/* ---------- what she has learned (file 09 step 4) ----------
   She reflects on her own and writes it down; this reads it back. The two
   buttons under a flag are the ONLY way a learned preference is deleted --
   reflection FLAGS, Harsh decides. Nothing here is spoken. */

let growthNow = null;           // the last counts the heartbeat brought

async function reflectPost(body) {
  try {
    const r = await fetch('/api/reflection', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    const j = await r.json();
    toast(j.result);
    growthNow = j.growth || growthNow;
    paintReflection(j.reflection);
  } catch (e) { toast(String(e), true); }
}

function paintReflection(state) {
  if (!$('reflecttext') || !state) return;
  const note = state.note;
  $('reflectwhen').textContent = note ? (note.date || 'saved') : 'never';
  /* The growth line first: it is true even before she has ever reflected,
     and it is the one a person actually wants to see (file 09 step 6). */
  const g = growthNow || {};
  $('reflecttext').textContent = [
    g.summary || '',
    note ? (note.went_well || note.summary || '') : '',
    note ? (note.to_adjust || '') : '',
    (note && note.themes && note.themes.length)
      ? 'themes: ' + note.themes.join(' · ') : '',
    note ? '' : 'She has not reflected yet.',
  ].filter(Boolean).join('\n\n');

  /* What he has actually corrected her into (step 1), newest first. Read
     only: a preference is removed by ruling on a FLAG, never from a list. */
  const prefs = $('reflectprefs');
  if (prefs) {
    prefs.innerHTML = '';
    for (const row of (state.preferences || [])) {
      const line = document.createElement('p');
      line.className = 'calnote';
      line.textContent = `${row.date} — ${row.text}`;
      prefs.appendChild(line);
    }
    if (!(state.preferences || []).length) {
      const line = document.createElement('p');
      line.className = 'calnote';
      line.textContent = 'Nothing corrected yet — tell her when she gets '
        + 'something wrong and it lands here.';
      prefs.appendChild(line);
    }
  }

  const box = $('reflectflags');
  box.innerHTML = '';
  for (const flag of (state.flags || [])) {
    const row = document.createElement('div');
    row.className = 'calrow';
    const label = document.createElement('label');
    label.textContent = flag.text;
    const field = document.createElement('div');
    field.className = 'calfield';
    const why = document.createElement('span');
    why.className = 'calnote';
    why.textContent = `${flag.kind}: ${flag.why}`;
    const keep = document.createElement('button');
    keep.textContent = 'Keep';
    keep.onclick = () => reflectPost({ action: 'keep', id: flag.id });
    const drop = document.createElement('button');
    drop.textContent = 'Remove';
    drop.onclick = () => reflectPost({ action: 'remove', id: flag.id });
    field.append(keep, drop);
    row.append(label, why, field);
    box.appendChild(row);
  }
}

/* ⚠ NO FETCH OF ITS OWN. Step 7: "Reuse the dashboard's existing
   data-fetching pattern instead of adding a separate one." /api/state carries
   `growth` and `reflection` now, and heartbeat() paints this panel the same
   way it paints Screen. This only wires the button. */
function buildGrowth() {
  if (!$('reflectnow')) return;
  $('reflectnow').onclick = () => reflectPost({ action: 'run' });
}

/* ---------- study + diary (file 12) ----------
   Both ride the /api/state heartbeat like every other panel. The study
   numbers are FSRS's own (study.py); the diary entries are the sentences
   she wrote in her reflections, not a re-wording of them. */

let studyCard = null;           // the card on screen, with its answer
let studyRevealed = false;      // ...and whether he has asked to see it

async function studyPost(body) {
  try {
    const r = await fetch('/api/study', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    const j = await r.json();
    toast(j.result);
    studyRevealed = false;      // the next card starts face down
    paintStudy(j.study);
  } catch (e) { toast(String(e), true); }
}

function buildStudy() {
  if (!$('studyshow')) return;
  $('studyshow').onclick = () => { studyRevealed = true; paintStudy(null); };
  for (const b of $('studygrades').querySelectorAll('button')) {
    b.onclick = () => {
      if (!studyCard) return;
      studyPost({ action: 'review', card_id: studyCard.id,
                  rating: b.dataset.rating });
    };
  }
  $('studyadd').onclick = () => {
    const q = $('studynewq').value.trim(), a = $('studynewa').value.trim();
    if (!q || !a) { toast('A card needs a question and an answer', true); return; }
    studyPost({ action: 'add', question: q, answer: a,
                topic: $('studynewt').value.trim() });
    $('studynewq').value = ''; $('studynewa').value = ''; 
  };
  $('studyload').onclick = () => {
    const name = $('studyfile').value;
    if (!name) { toast('Nothing new in notes/', true); return; }
    studyPost({ action: 'load', name });
  };
}

/* `s` null means "repaint what we already have" — used by Show answer, which
   changes nothing on the robot and must not cost a round trip. */
let studyLast = null;
function paintStudy(s) {
  if (!$('studyline')) return;
  s = s || studyLast;
  if (!s) return;
  studyLast = s;
  const st = s.stats || {};
  const due = st.due_now || 0;
  $('studydue').textContent = due ? `${due} due` : 'nothing due';
  const bits = [];
  if (st.cards) bits.push(`${st.cards} card${st.cards === 1 ? '' : 's'}`);
  if (st.sources) bits.push(`${st.sources} source${st.sources === 1 ? '' : 's'}`);
  if (st.reviews) bits.push(`${st.reviews} review${st.reviews === 1 ? '' : 's'}`);
  if (s.streak) bits.push(`${s.streak}-day streak`);
  if (st.retention !== null && st.retention !== undefined)
    bits.push(`${Math.round(st.retention * 100)}% recalled`);
  $('studyline').textContent = bits.length
    ? bits.join(' · ') : 'No notes loaded yet.';

  /* The session */
  const card = s.card;
  studyCard = card;
  $('studydueline').textContent = card
    ? (card.is_due ? `due now · ${card.topic}`
                   : `nothing due — next in ${card.due_in} · ${card.topic}`)
    : 'no cards yet';
  $('studyq').textContent = card ? card.question : 'No cards yet.';
  const answer = $('studya');
  answer.textContent = card ? card.answer : '';
  answer.hidden = !(card && studyRevealed);
  $('studyshow').hidden = !card || studyRevealed;
  $('studygrades').hidden = !(card && studyRevealed);

  const picker = $('studyfile');
  if (picker) {
    const names = s.loadable || [];
    picker.innerHTML = '';
    for (const name of names) {
      const opt = document.createElement('option');
      opt.value = name; opt.textContent = name;
      picker.appendChild(opt);
    }
    if (!names.length) {
      const opt = document.createElement('option');
      opt.value = ''; opt.textContent = 'nothing new in notes/';
      picker.appendChild(opt);
    }
  }

  const box = $('studyhistory');
  box.innerHTML = '';
  for (const r of (s.recent || [])) {
    const dot = document.createElement('span');
    dot.className = 'qdot ' + (r.right ? 'right' : 'wrong');
    dot.title = `${r.rating} — ${r.topic || 'no topic'}: ${r.question}`;
    box.appendChild(dot);
  }
}

function paintDiary(entries) {
  const box = $('diary');
  if (!box) return;
  const rows = (entries || []).slice().reverse();      // newest first
  $('diarycount').textContent =
    `${rows.length} ${rows.length === 1 ? 'entry' : 'entries'}`;
  box.innerHTML = '';
  if (!rows.length) {
    const empty = document.createElement('p');
    empty.className = 'calnote';
    empty.textContent = 'Nothing written yet. She reflects when a '
      + 'conversation ends, when she has been idle a while, and once late.';
    box.appendChild(empty);
    return;
  }
  for (const e of rows) {
    const entry = document.createElement('div');
    entry.className = 'diaryentry';
    const when = document.createElement('div');
    when.className = 'diaryday';
    when.textContent = e.date + (e.reason ? ` — ${e.reason}` : '');
    entry.appendChild(when);
    for (const line of [e.went_well, e.to_adjust]) {
      if (!line) continue;
      const para = document.createElement('p');
      para.className = 'diarytext';
      para.textContent = line;
      entry.appendChild(para);
    }
    if (e.themes && e.themes.length) {
      const themes = document.createElement('p');
      themes.className = 'calnote';
      themes.textContent = e.themes.join(' · ');
      entry.appendChild(themes);
    }
    box.appendChild(entry);
  }
}

/* ---------- memory lane (file 12 step 3) ----------
   Names her memory mentions, and what she remembers about each. All of it
   arrives on the heartbeat; picking a name is a local repaint, not a
   request. */
let lanePages = [];
let laneOpen = null;

function paintLane(pages) {
  if (!$('lanenames')) return;
  lanePages = pages || lanePages || [];
  $('lanecount').textContent = lanePages.length
    ? `${lanePages.length} ${lanePages.length === 1 ? 'person' : 'people'}`
    : 'nobody yet';
  if (laneOpen && !lanePages.some((p) => p.name === laneOpen)) laneOpen = null;
  if (!laneOpen && lanePages.length) laneOpen = lanePages[0].name;

  const bar = $('lanenames');
  bar.innerHTML = '';
  for (const page of lanePages) {
    const b = document.createElement('button');
    b.className = 'seg' + (page.name === laneOpen ? '' : ' off');
    b.textContent = `${page.name} · ${page.count}`;
    b.onclick = () => { laneOpen = page.name; paintLane(null); };
    bar.appendChild(b);
  }

  const box = $('lanefacts');
  box.innerHTML = '';
  const page = lanePages.find((p) => p.name === laneOpen);
  if (!page) {
    const empty = document.createElement('p');
    empty.className = 'calnote';
    empty.textContent = 'No names have come up in conversation yet.';
    box.appendChild(empty);
    return;
  }
  const head = document.createElement('div');
  head.className = 'diaryday';
  head.textContent = `${page.name} — last mentioned ${page.last_said}`;
  box.appendChild(head);
  for (const fact of page.facts) {
    const line = document.createElement('p');
    line.className = 'diarytext lanefact';
    const words = document.createElement('span');
    words.textContent = fact.text;
    line.appendChild(words);
    if (fact.id != null) {
      // FILE 17: exactly two controls. The bin only ASKS; the server deletes
      // only when the confirmation's token comes back for this same row.
      const lock = document.createElement('button');
      lock.className = 'lanebtn' + (fact.protected ? ' on' : '');
      lock.title = fact.protected ? 'Protected — never flagged as stale' : 'Protect';
      lock.textContent = fact.protected ? '🔒' : '🔓';
      lock.onclick = () => memoryPost({ action: 'protect', id: fact.id, on: !fact.protected });
      const bin = document.createElement('button');
      bin.className = 'lanebtn';
      bin.title = 'Forget this';
      bin.textContent = '🗑';
      bin.onclick = () => forgetFact(fact);
      line.appendChild(lock);
      line.appendChild(bin);
    }
    box.appendChild(line);
  }
}

async function memoryPost(body) {
  try {
    const r = await fetch('/api/memory', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { toast(await r.text(), true); return null; }
    const j = await r.json();
    if (j.result) toast(j.result);
    if (body.action !== 'delete_request') heartbeat();
    return j;
  } catch (e) { toast(String(e), true); return null; }
}

async function forgetFact(fact) {
  const ask = await memoryPost({ action: 'delete_request', id: fact.id });
  if (!ask) return;
  const warn = ask.protected ? '\n\nThis one is protected.' : '';
  if (!window.confirm(`Forget this?\n\n“${ask.text}”${warn}`)) {
    toast('Kept');
    return;
  }
  await memoryPost({ action: 'delete_confirm', id: fact.id, token: ask.token });
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
    $('timerlabel').value = '';
    $('timermins').value = '';
    paintScreen(j);
  } catch (e) { toast(String(e), true); }
}

function paintScreen(state) {
  if (!state) return;
  screenApp = state.app || null;
  screenApps = state.apps || [];
  const items = state.pending || [];
  screenKinds = new Set(items.map((i) => i.kind));
  $('screennow').textContent = state.urgent ? 'ringing' : (screenApp || 'face');
  $('screennow').classList.toggle('urgent', !!state.urgent);

  // An alarm going off is the one thing on this panel that is time-critical,
  // so it gets its own row rather than being another badge to notice.
  const ring = $('ringing');
  if (ring) {
    ring.hidden = !state.urgent;
    if (state.urgent) {
      const live = items.find((i) => i.kind === screenApp) || items[0];
      $('ringtext').textContent = (live && live.label)
        ? `${screenApp || 'Alarm'} — ${live.label}` : 'Alarm ringing';
    }
  }

  paintApps();

  const list = $('pending');
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

/* ---------- environments (file 19) ----------
   Everything read-only except the one naming box per waiting area, which
   POSTs {pending_id, name} to /api/landmarks and nothing else. */

async function nameArea(id, input) {
  const name = input.value.trim();
  if (!name) { toast('Type a name first', true); return; }
  try {
    const r = await fetch('/api/landmarks', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pending_id: id, name }),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    const j = await r.json();
    toast(j.result);
    paintEnvs(j);
  } catch (e) { toast(String(e), true); }
}

function paintEnvs(d) {
  if (!d || !$('envlist') || d.error) return;
  const waiting = d.pending || [];
  const badge = $('envnow');
  badge.textContent = waiting.length
    ? `${waiting.length} to name` : (d.active || '—');
  badge.classList.toggle('live', waiting.length > 0);
  const st = d.status || {};
  $('envstatus').textContent = st.state === 'no_match'
    ? `⚠ ${st.ask || 'At start-up she did not recognise this place.'}`
    : st.state === 'matched' ? `In ${d.active} — recognised at start-up (${st.why || ''}).`
    : st.state === 'no_map' ? `In ${d.active} — no map yet here.`
    : `In ${d.active || '—'}${st.why ? ' — ' + st.why : ''}.`;

  const list = $('envlist');
  list.innerHTML = '';
  for (const e of d.environments || []) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="kind">${e.active ? 'here' : ''}</span>` +
      `<span class="lbl">${esc(e.name)} · ${e.landmarks} place${e.landmarks === 1 ? '' : 's'}</span>`;
    list.appendChild(li);
  }

  $('envpendhead').hidden = !waiting.length;
  const pend = $('envpending');
  pend.innerHTML = '';
  for (const p of waiting) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="kind">${esc(p.area_m2)} m²</span>`;
    const input = document.createElement('input');
    input.type = 'text'; input.maxLength = 40; input.placeholder = 'Name this space';
    input.onkeydown = (ev) => { if (ev.key === 'Enter') nameArea(p.id, input); };
    const save = document.createElement('button');
    save.className = 'kill'; save.title = 'Save name'; save.textContent = '✓';
    save.onclick = () => nameArea(p.id, input);
    li.appendChild(input); li.appendChild(save);
    pend.appendChild(li);
  }

  const obst = Object.entries(d.obstacles || {});
  $('envobst').textContent = obst.length
    ? 'Obstacles noted: ' + obst.map(([place, o]) =>
        `${place} ${o.count}${o.last && o.last.class !== 'unclassified' ? ' (' + o.last.class + ')' : ''}`).join(' · ')
    : '';

  const runs = ((d.summaries || {}).runs) || [];
  const rl = $('envruns');
  rl.innerHTML = runs.length ? '' : '<li class="none">No runs summarised yet</li>';
  for (const r of runs.slice(0, 6)) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="kind">${esc(ago(r.ended))}</span><span class="lbl">${esc(r.line || '')}</span>`;
    rl.appendChild(li);
  }
  const older = (d.summaries || {}).older;
  if (older && older.runs) {
    const li = document.createElement('li');
    li.className = 'none';
    li.textContent = `…and ${older.runs} older runs, ${older.minutes} min, ${older.area_m2} m² in all`;
    rl.appendChild(li);
  }
}

/* ---------- the exploration mission (file 21 step 11) ----------
   Read-only except the one naming box per area, which POSTs exactly
   {area_id, name} to /api/mission. Everything is drawn in the one frame the
   server chose (the mission's own while it runs, else the home anchor's). */

async function nameMissionArea(id, input) {
  const name = input.value.trim();
  if (!name) { toast('Type a name first', true); return; }
  try {
    const r = await fetch('/api/mission', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ area_id: id, name }),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    const j = await r.json();
    toast(`Saved as ${(j.area || {}).display_name || name}`);
    refreshMission();
  } catch (e) { toast(String(e), true); }
}

function missionMap(d) {
  const pts = [];
  for (const a of d.areas || []) for (const q of a.polygon) pts.push(q);
  for (const k of ['home', 'start', 'pose']) if (d[k]) pts.push([d[k].x, d[k].y]);
  for (const q of d.trail || []) pts.push(q);
  if (!pts.length) return '';
  const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
  const x0 = Math.min(...xs) - 0.5, y1 = Math.max(...ys) + 0.5;
  const span = Math.max(Math.max(...xs) + 0.5 - x0, y1 - (Math.min(...ys) - 0.5), 1);
  const u = (x) => ((x - x0) / span * 100).toFixed(2);
  const v = (y) => ((y1 - y) / span * 100).toFixed(2);
  const out = [];
  for (const a of d.areas || []) {
    if (a.polygon.length < 3) continue;
    const cls = a.status === 'named' ? '' : ' ' + a.status;
    out.push(`<polygon class="area${cls}" points="${a.polygon.map(q => u(q[0]) + ',' + v(q[1])).join(' ')}"/>`);
    if (a.center) out.push(`<text x="${u(a.center[0])}" y="${v(a.center[1])}">${esc(a.display_name)}</text>`);
  }
  if ((d.trail || []).length > 1)
    out.push(`<polyline class="trail" points="${d.trail.map(q => u(q[0]) + ',' + v(q[1])).join(' ')}"/>`);
  for (const f of d.frontiers || []) out.push(`<circle class="edge" cx="${u(f.x)}" cy="${v(f.y)}" r="1.2"/>`);
  if (d.home) out.push(`<circle class="lm home" cx="${u(d.home.x)}" cy="${v(d.home.y)}" r="2"/>`);
  if (d.start) out.push(`<circle class="start" cx="${u(d.start.x)}" cy="${v(d.start.y)}" r="1.6"/>`);
  if (d.pose) out.push(`<circle class="her" cx="${u(d.pose.x)}" cy="${v(d.pose.y)}" r="1.6"/>`);
  return out.join('');
}

async function refreshMission() {
  if (!$('missionnow')) return;
  let d;
  try { d = await (await fetch('/api/mission')).json(); } catch { return; }
  const badge = $('missionnow');
  const on = !!(d && d.running);
  badge.textContent = on ? String(d.state || '').toLowerCase().replace(/_/g, ' ') : 'off';
  badge.classList.toggle('live', on && d.state !== 'IDLE');
  if (!on) {
    $('missionline').textContent = d && d.enabled
      ? `The mission is not running${d.why ? ' — ' + d.why : ''}.`
      : 'The mission service is off (EXPLORATION_MISSION_SERVICE).';
    for (const id of ['missionmap', 'missionkey', 'missionqhead', 'missionareashead',
                      'missionhisthead']) $(id).hidden = true;
    for (const id of ['missionqueue', 'missionareas', 'missionhist']) $(id).innerHTML = '';
    $('missionladder').textContent = '';
    return;
  }
  const s = d.snapshot || {};
  const bits = [];
  if (s.mission_id) {
    bits.push(`out ${d.elapsed_s != null ? Math.round(d.elapsed_s / 60) + ' min' : ''}`);
    bits.push(`${d.mapped_m2 ?? 0} m² mapped`, `${s.distance_m ?? 0} m driven`);
    if (s.frontiers_left != null) bits.push(`${s.frontiers_left} edges left`);
  } else {
    bits.push(s.placed === 'home' ? 'at her home spot' : 'not placed at home');
    if (s.interrupted) bits.push('a mission is waiting for “continue” or “go back”');
  }
  const cur = s.curiosity || {};
  if (cur.mode && cur.mode !== 'off') bits.push(`curiosity ${cur.mode}${(cur.last || {}).why ? ' (' + cur.last.why + ')' : ''}`);
  $('missionline').textContent = bits.join(' · ');

  const svg = $('missionmap');
  const pic = missionMap(d);
  svg.hidden = !pic; $('missionkey').hidden = !pic;
  svg.innerHTML = pic;

  const q = d.naming_queue || [];
  $('missionqhead').hidden = !q.length;
  const ql = $('missionqueue');
  ql.innerHTML = '';
  for (const a of q) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="kind">${esc(a.status)}</span><span class="lbl">${esc(a.display_name)} · ${esc(a.area_m2)} m²</span>`;
    const input = document.createElement('input');
    input.type = 'text'; input.maxLength = 40; input.placeholder = 'Name it';
    input.onkeydown = (ev) => { if (ev.key === 'Enter') nameMissionArea(a.id, input); };
    const save = document.createElement('button');
    save.className = 'kill'; save.title = 'Save name'; save.textContent = '✓';
    save.onclick = () => nameMissionArea(a.id, input);
    li.appendChild(input); li.appendChild(save);
    ql.appendChild(li);
  }

  const named = (d.areas || []).filter(a => a.status === 'named');
  $('missionareashead').hidden = !named.length;
  $('missionareas').innerHTML = named.map(a =>
    `<li><span class="kind">${esc(a.area_m2)} m²</span><span class="lbl">${esc(a.display_name)}</span></li>`).join('') +
    (d.areas_elsewhere ? `<li class="none">…and ${d.areas_elsewhere} saved in another frame</li>` : '');

  const lad = d.ladder;
  $('missionladder').textContent = lad
    ? `Way back: rung ${lad.rung ?? '—'} · ${lad.phase || ''} · cycle ${lad.cycle ?? 0}` +
      (lad.in_help ? ' · asking for help' : '')
    : (s.verified === false ? 'Back at the start, but the view did not match it.' : '');

  const hist = d.missions || [];
  $('missionhisthead').hidden = !hist.length;
  $('missionhist').innerHTML = hist.map(m =>
    `<li><span class="kind">${esc(m.status)}</span><span class="lbl">` +
    `${esc(ago(m.started_wall))} · ${esc(m.stop_reason || '')}` +
    `${m.explored_m2 != null ? ' · ' + esc(m.explored_m2) + ' m²' : ''}` +
    `${m.home_error_m != null ? ' · ended ' + esc(m.home_error_m) + ' m from start' : ''}` +
    `${m.rth_attempts ? ' · ' + esc(m.rth_attempts) + ' return tries' : ''}` +
    `${m.by === 'curiosity' ? ' · her own idea' : ''}</span></li>`).join('');
}

/* ---------- reminders, to-do, routines, calendar (file 16) ----------
   Rides the 4 s heartbeat like the other panels. Times are typed the way
   they are said ("5 pm", "tomorrow morning", "in 20 minutes"); the server
   parses them with the same code her voice uses. */

async function dayPost(body) {
  try {
    const r = await fetch('/api/reminders', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    const j = await r.json();
    toast(j.result);
    paintDay(j);
  } catch (e) { toast(String(e), true); }
}

function dayItem(item, withTime) {
  const li = document.createElement('li');
  const when = withTime && item.due_at
    ? new Date(item.due_at * 1000).toLocaleString([], {
        weekday: 'short', hour: '2-digit', minute: '2-digit' })
    : 'to-do';
  li.innerHTML = `<span class="kind">${esc(when)}</span>` +
    `<span class="lbl">${esc(item.text)}</span>`;
  const done = document.createElement('button');
  done.className = 'kill';
  done.title = 'Done';
  done.textContent = '✓';
  done.onclick = () => dayPost({ action: 'done', id: item.id });
  li.appendChild(done);
  return li;
}

function paintDay(d) {
  if (!d || !$('remlist') || d.error) return;
  const rem = d.reminders || [], todo = d.todos || [];
  $('daynow').textContent = `${rem.length} reminder${rem.length === 1 ? '' : 's'}` +
    ` · ${todo.length} to-do`;
  const remList = $('remlist');
  remList.innerHTML = rem.length ? '' : '<li class="none">No reminders set</li>';
  for (const item of rem) remList.appendChild(dayItem(item, true));
  const todoList = $('todolist');
  todoList.innerHTML = todo.length ? '' : '<li class="none">To-do list is empty</li>';
  for (const item of todo) todoList.appendChild(dayItem(item, false));

  const cal = d.calendar || {};
  $('calnote').textContent = cal.events && cal.events.length
    ? 'Today on your calendar:' : 'Nothing on your calendar today.';
  const calList = $('callist');
  calList.innerHTML = '';
  for (const e of (cal.events || [])) {
    const li = document.createElement('li');
    const when = e.all_day ? 'all day' : new Date(e.start_ts * 1000)
      .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    li.innerHTML = `<span class="kind">${esc(when)}</span><span class="lbl">${esc(e.title)}</span>`;
    if (e.local && e.id != null) {
      const kill = document.createElement('button');
      kill.className = 'kill';
      kill.title = 'Remove';
      kill.textContent = '×';
      kill.onclick = () => dayPost({ action: 'event_delete', id: e.id });
      li.appendChild(kill);
    }
    calList.appendChild(li);
  }
  const routines = (d.routines || []).map((r) => `“${r.name}” → ${r.steps.join(', ')}`);
  $('routinenote').textContent = (routines.length ? `Routines: ${routines.join(' · ')}` : '') +
    (d.briefing_done_today ? ' · today\'s briefing: said' : ' · today\'s briefing: not yet');
}

function wireDay() {
  if (!$('remadd')) return;
  const add = () => {
    const text = $('remtext').value.trim();
    if (!text) { toast('What should I remind you of?', true); return; }
    dayPost({ action: 'add', text, when: $('remwhen').value.trim() });
    $('remtext').value = ''; $('remwhen').value = '';
  };
  $('remadd').onclick = add;
  const addEvent = () => {
    const text = $('evtext').value.trim(), when = $('evwhen').value.trim();
    if (!text || !when) { toast('An event needs a name and a time', true); return; }
    dayPost({ action: 'event', text, when });
    $('evtext').value = ''; $('evwhen').value = '';
  };
  $('evadd').onclick = addEvent;
  $('evwhen').onkeydown = (e) => { if (e.key === 'Enter') addEvent(); };
  $('remtext').onkeydown = (e) => { if (e.key === 'Enter') add(); };
}

/* ---------- laptop (2026-10-21) ----------
   Her play goes through the same MCP tools as her voice. Status is polled
   here, not in the heartbeat: it asks the laptop over Wi-Fi, which can take
   seconds, and only while this tab is visible. The picture is fetched only
   while "Show the game" is on. */

let laptopPicOn = false, laptopPicTimer = null, laptopTimer = null;

function fill(sel, names, keepFirst) {
  const want = names.join('\n');
  if (sel.dataset.names === want) return;
  sel.dataset.names = want;
  const first = keepFirst ? sel.firstElementChild : null;
  sel.textContent = '';
  if (first) sel.appendChild(first);
  for (const n of names) {
    const o = document.createElement('option');
    o.value = n; o.textContent = n;
    sel.appendChild(o);
  }
}

function paintLaptop(d) {
  const badge = $('laptopnow');
  if (!badge || !d) return;
  const play = d.play || {};
  if (!d.configured) { badge.textContent = 'not set up'; return; }
  badge.textContent = play.running ? 'she is playing' : (d.on ? 'on' : 'off');
  badge.classList.toggle('on', !!play.running);
  const f = d.front || {};
  $('laptopfront').textContent = !d.on
    ? 'The laptop is off, asleep, or its helper is not running.'
    : `In front: ${f.title || f.app || 'nothing'}` +
      (d.game ? ` (your game: ${d.game})` : '') +
      (f.idle_s != null ? ` · untouched ${Math.round(f.idle_s)} s` : '') +
      (d.version && d.version < 8 ? ' · helper ' + d.version + ': copy the new one to play fast games' : '');
  fill($('laptopgame'), d.games || [], true);
  fill($('laptopapps'), d.apps || [], false);
  $('laptopplaying').textContent = play.game
    ? (play.running
        ? `Playing ${play.game}: ${play.moves || 0} moves` + (play.last ? ` · last: ${play.last}` : '')
        : `Last game: ${play.game}, ${play.moves || 0} moves${play.why ? ' · ' + play.why : ''}`)
    : '';
}

async function refreshLaptop() {
  if (document.hidden || !$('panel-laptop')) return;
  try {
    const r = await fetch('/api/laptop');
    if (r.ok) paintLaptop(await r.json());
  } catch {}
}

async function laptopPost(body, okMsg) {
  try {
    const r = await fetch('/api/laptop', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    const d = await r.json();
    toast(d.ok ? (d.result || okMsg) : d.why, !d.ok);
    setTimeout(refreshLaptop, 1500);
  } catch (e) { toast(String(e), true); }
}

async function laptopPicture() {
  if (!laptopPicOn || document.hidden) return;
  const img = $('laptoppic'), note = $('laptoppicnote');
  try {
    const r = await fetch('/api/laptop/screen.png', { cache: 'no-store' });
    if (r.ok) {
      const url = URL.createObjectURL(await r.blob());
      const old = img.src;
      img.src = url; img.hidden = false; note.hidden = true;
      if (old.startsWith('blob:')) URL.revokeObjectURL(old);
    } else {
      const d = await r.json().catch(() => ({}));
      note.textContent = d.why || 'No picture'; note.hidden = false;
    }
  } catch (e) { note.textContent = String(e); note.hidden = false; }
}

function wireLaptop() {
  if (!$('panel-laptop')) return;
  $('laptopplay').onclick = () => laptopPost({ action: 'play', game: $('laptopgame').value }, 'She is playing');
  $('laptopstop').onclick = () => laptopPost({ action: 'stop' }, 'Your turn');
  const open = () => {
    const name = $('laptopapp').value.trim();
    if (!name) { toast('Which app?', true); return; }
    laptopPost({ action: 'open', name }, 'Opening ' + name);
    $('laptopapp').value = '';
  };
  $('laptopopen').onclick = open;
  $('laptopapp').onkeydown = (e) => { if (e.key === 'Enter') open(); };
  $('laptoppicbtn').onclick = () => {
    laptopPicOn = !laptopPicOn;
    $('laptoppicbtn').classList.toggle('on', laptopPicOn);
    $('laptoppicbtn').textContent = laptopPicOn ? 'Hide the game' : 'Show the game';
    clearInterval(laptopPicTimer);
    if (laptopPicOn) { laptopPicture(); laptopPicTimer = setInterval(laptopPicture, 2500); }
    else { $('laptoppic').hidden = true; $('laptoppicnote').hidden = true; }
  };
  refreshLaptop();
  laptopTimer = setInterval(refreshLaptop, 5000);
}

/* ---------- describe (file 15) ----------
   The button writes the same switch as "keep describing what you see". The
   heartbeat repaints it, so a voice "stop describing" or the timeout shows
   up here within a beat without anything to reload. */

let narrateOn = false;

async function narratePost(on) {
  const btn = $('narratego');
  btn.disabled = true;
  try {
    const r = await fetch('/api/narration', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ on }),
    });
    if (!r.ok) { toast(await r.text(), true); return; }
    paintNarration(await r.json());
    toast(on ? 'She will describe what changes' : 'Describing stopped');
  } catch (e) { toast(String(e), true); }
  finally { btn.disabled = false; }
}

function paintNarration(n) {
  const btn = $('narratego');
  if (!n || !btn) return;
  narrateOn = !!n.on;
  btn.classList.toggle('on', narrateOn);
  btn.textContent = narrateOn ? 'Stop describing' : 'Keep describing what you see';
  btn.onclick = () => narratePost(!narrateOn);
  const badge = $('narratenow');
  badge.textContent = narrateOn ? `on · ${countdown(n.left_s || 0)} left` : 'off';
  badge.classList.toggle('live', narrateOn);
  const counts = narrateOn && n.looks
    ? ` · ${n.looks} look${n.looks === 1 ? '' : 's'}, ${n.no_change || 0} with nothing new`
    : '';
  $('narratestatus').textContent = narrateOn
    ? `${n.status || 'watching'}${counts}`
    : `Off. She says what changes in front of her camera until you stop it, `
      + `or for ${Math.round((n.timeout_s || 900) / 60)} minutes.`
      + (n.status && n.status !== 'off' ? ` (${n.status})` : '');
  const said = $('narratesaid');
  said.hidden = !n.last_said;
  if (n.last_said) said.textContent = `Last said ${ago(n.last_said_at)}: “${n.last_said}”`;
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

  for (const b of document.querySelectorAll('.dbtn')) {
    if (b.dataset.dir === 'stop') b.onclick = () => drive('stop', b);
    else holdable(b);
  }

  const speed = $('drivespeed');
  if (speed) {
    speed.oninput = () => { $('speedval').textContent = speed.value + '%'; };
    // Show the speed she is really at, not the page's own idea of it.
    fetch('/api/calibration').then((r) => r.json()).then((c) => {
      if (c.speed_percent) {
        speed.value = c.speed_percent;
        $('speedval').textContent = c.speed_percent + '%';
      }
    }).catch(() => {});
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

/* HOLD TO DRIVE. A tap is one step. Keep the thumb down and she keeps going,
   step after step, each one still guarded by the server; lift it and she stops.
   If the phone loses the Pi mid-hold the step that is running simply ends. */
function holdable(button) {
  let held = false, started = 0;
  const release = () => {
    if (!held) return;
    held = false;
    // A tap plays its one step out; a hold ends the moment the thumb lifts.
    if (performance.now() - started > driveStep * 1000) drive('stop');
  };
  button.onpointerdown = async (e) => {
    e.preventDefault();
    if (held) return;
    try { button.setPointerCapture(e.pointerId); } catch (err) { /* it still works */ }
    held = true;
    started = performance.now();
    do {
      await drive(button.dataset.dir, button);
      await new Promise((r) => setTimeout(r, 15));
    } while (held);
  };
  button.onpointerup = button.onpointercancel = release;
  button.oncontextmenu = (e) => e.preventDefault();
  // A keyboard's Enter arrives as a click with no pointer behind it.
  button.onclick = (e) => { if (e.detail === 0) drive(button.dataset.dir, button); };
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

/* The UI the server is serving. If it moves while this page is open, the
   page in his hand is the OLD one — see server.ui_version(). Reload once,
   and only once, so a fix he was told about actually reaches him. */
let uiVersion = null;
function checkUi(stamp) {
  if (!stamp) return;
  if (uiVersion === null) { uiVersion = stamp; return; }
  if (uiVersion !== stamp) { uiVersion = stamp; location.reload(); }
}

async function heartbeat() {
  let s;
  try { s = await (await fetch('/api/state')).json(); }
  catch { connected = false; paintDot(); return; }
  checkUi(s.ui);
  setActive(s.mood);
  paintDot(s.face_running);
  applyCams(s.cameras || {});     // picks up hot-plugs and other viewers
  paintRobot(s.robot_online);
  paintScreen(s.screen);          // rides the same beat; no extra request
  paintNarration(s.narration);    // ...and so does Describe (file 15)
  paintDay(s.day);                // ...and Reminders (file 16)
  paintEnvs(s.environments);      // ...and Environments (file 19)
  growthNow = s.growth || growthNow;
  paintReflection(s.reflection); // ...and so does Brain Growth (step 7)
  paintStudy(s.study);           // ...and Study and Diary (file 12)
  paintDiary((s.reflection || {}).log);
  paintLane(s.people);           // ...and Memory Lane
  refreshExplore();               // cheap, and the panel must not go stale
  refreshMap();                   // one stat() server-side until the map changes
  refreshMaps();
  refreshNav();                   // landmarks + the attempt log, one read
  refreshMission();               // the exploration mission (file 21)
  if ($('navroundtrip')) $('navroundtrip').onchange = (e) =>
    navSend({ action: 'round_trip', on: e.target.checked },
            e.target.checked ? 'she will come back afterwards…'
                             : 'she will stay where she is sent…');
  if ($('navcomeback')) $('navcomeback').onclick = () =>
    navSend({ action: 'come_back' }, 'bringing her back…');
}

/* friday-mcp is a separate service from this one, so the dashboard opens and
   works whether or not the robot is running. Say which half is down rather
   than letting the drive buttons fail silently.

   ⚠ TWO BEATS BEFORE IT SAYS OFF. Harsh, 2026-09-20: "its showing robot off
   but robot is on". robot_online() is a bare TCP connect, and friday-mcp
   takes ~20 s to start accepting them again after a restart — so every
   restart flashed ROBOT OFF at whoever was holding the phone, and a single
   dropped poll did the same. The badge now needs two failures in a row, and
   clears on the first success. */
let robotMisses = 0;
function paintRobot(online) {
  const badge = $('robotoff');
  if (!badge) return;
  if (online === false) robotMisses += 1; else robotMisses = 0;
  badge.hidden = robotMisses < 2;
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
  paintNarration(state.narration);
  wireLaptop();
  wireDay();
  paintDay(state.day);
  paintEnvs(state.environments);

  buildExplore();
  await refreshExplore();
  refreshMap();
  refreshMaps();
  refreshNav();
  refreshMission();
  buildGrowth();
  buildStudy();
  growthNow = state.growth || null;
  paintReflection(state.reflection);
  paintStudy(state.study);
  paintDiary((state.reflection || {}).log);
  paintLane(state.people);

  connect();
  setInterval(heartbeat, 4000);

  /* A PHONE THAT WAS ASLEEP SHOWS OLD STATE. Mobile browsers freeze timers
     in a backgrounded tab, so the first thing he sees on unlocking is
     whatever was true when he locked it — up to four seconds stale at best,
     and minutes stale in practice. Beat once, immediately, on coming back. */
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) { robotMisses = 0; heartbeat(); }
  });
  window.addEventListener('focus', () => heartbeat());

  // A manifest shortcut lands on /?pane=... -- honour it, then fall back to
  // whichever pane was last open. Anything else is ignored rather than trusted.
  const asked = new URLSearchParams(location.search).get('pane');
  const panes = ['pane-logs', 'pane-robot'];
  showPane(panes.includes(asked) ? asked : store.get('friday.pane', 'pane-logs'));
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

  installable();
})();

/* --- installing it as an app -----------------------------------------
   A phone offers "Install app" only when it TRUSTS the page. When it does
   not, it offers a home-screen shortcut instead -- which opens in a browser
   tab -- and the only place the reason is written is the service worker's
   registration error, which this used to swallow with catch(() => {}). So
   the reason is shown on the page now, with the one tap that fixes it, and
   reported to the Pi (logs/dashboard_clients.log), where "why won't it
   install" can be answered without the phone in hand. */
function installable() {
  const standalone = matchMedia('(display-mode: standalone)').matches ||
                     navigator.standalone === true;

  const report = (state) => fetch('/api/client', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(Object.assign({
      href: location.href, secure: window.isSecureContext, standalone,
    }, state)),
  }).then((r) => r.json()).catch(() => null);

  const help = (text, link, label) => {
    if ($('installbar')) return;
    const bar = document.createElement('div');
    bar.id = 'installbar';
    bar.className = 'installbar';
    bar.setAttribute('role', 'status');
    const say = document.createElement('span');
    say.textContent = text;
    bar.appendChild(say);
    if (link) {
      const go = document.createElement('a');
      go.className = 'go';
      go.href = link;
      go.textContent = label;
      bar.appendChild(go);
    }
    const close = document.createElement('button');
    close.className = 'x';
    close.title = 'Dismiss';
    close.textContent = '×';
    close.onclick = () => bar.remove();
    bar.appendChild(close);
    document.body.appendChild(bar);
  };

  // Our own Install button, shown only once the browser says it can install.
  const btn = document.createElement('button');
  btn.className = 'ghost';
  btn.id = 'btn-install';
  btn.title = 'Install Friday as an app';
  btn.textContent = 'Install';
  btn.hidden = true;
  const clear = $('btn-clear');
  clear.parentNode.insertBefore(btn, clear);

  let offer = null;
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    offer = e;
    btn.hidden = false;
    report({ sw: 'ok', prompt: true });
  });
  btn.onclick = async () => {
    if (!offer) return;
    offer.prompt();
    try { await offer.userChoice; } catch (e) {}
    offer = null;
    btn.hidden = true;
  };
  window.addEventListener('appinstalled', () => {
    btn.hidden = true;
    toast('Installed. Open Friday from your home screen.');
    report({ installed: true });
  });

  if (!('serviceWorker' in navigator)) {
    // Plain HTTP, or a browser with no service workers at all.
    report({ sw: 'unavailable' }).then((a) => {
      if (!standalone) help('This page is not secure, so it can only be a ' +
        'shortcut. Open the secure address to install the app.',
        a && a.secure_url, 'Open secure page');
    });
    return;
  }
  navigator.serviceWorker.register('/sw.js')
    .then(() => report({ sw: 'ok' }))
    .catch((err) => {
      report({ sw: 'failed', error: String((err && err.message) || err) })
        .then((a) => help('Your phone does not trust Friday\'s certificate, ' +
          'so it can only add a shortcut. Install the certificate, then ' +
          'reload this page.', a && a.ca_url, 'Get certificate'));
    });
}
