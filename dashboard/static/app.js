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
        (m.visits ? ` · ${m.visits}×` : '') + '</span>').join('')
    : '<span>no places named yet</span>';

  const last = n.last_return;
  $('navrth').textContent = last
    ? `Last return home: ${last.arrived ? 'arrived' : 'did not arrive'} ` +
      `${ago(last.at)} · ${last.seconds}s · ${last.attempts} attempt(s)` +
      (last.arrived ? '' : ` · ${last.why || ''}`)
    : 'No return home yet.';

  const rows = n.attempts || [];
  $('navlog').innerHTML = rows.length
    ? rows.map(a => `<div><span>${esc(a.destination)}</span>` +
        `<span class="${a.arrived ? 'ok' : 'bad'}">` +
        `${a.arrived ? 'arrived' : esc(a.state)} · ${a.seconds}s · ${ago(a.at)}</span></div>`).join('')
    : '<div><span>no navigation attempts logged yet</span><span></span></div>';

  const live = rows[0];
  $('navnow').textContent = live ? (live.arrived ? 'arrived' : live.state) : 'idle';
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
async function refreshMap() {
  if (!$('mapimg')) return;
  let m;
  try { m = await (await fetch('/api/map')).json(); } catch { return; }
  if (!m.ok) { $('mapnow').textContent = 'no map'; return; }
  if (m.version !== mapVersion) {          // re-fetch the picture only when it changed
    mapVersion = m.version;
    $('mapimg').src = '/api/map.png?v=' + m.version;
  }
  $('mapnow').textContent = m.running ? 'last saved · run in progress' : 'saved ' + m.saved;
  $('mapnote').textContent =
    `${m.size_m[0]} × ${m.size_m[1]} m · known ${m.known_m2} m² · driven ${m.driven_m2} m² · ` +
    `${m.sweeps} sweeps · white free, black obstacle, orange driven, red her position, grid 1 m` +
    (m.running ? ' · updates when this run ends' : '');
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
    line.className = 'diarytext';
    line.textContent = fact.text;
    box.appendChild(line);
  }
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
  growthNow = s.growth || growthNow;
  paintReflection(s.reflection); // ...and so does Brain Growth (step 7)
  paintStudy(s.study);           // ...and Study and Diary (file 12)
  paintDiary((s.reflection || {}).log);
  paintLane(s.people);           // ...and Memory Lane
  refreshExplore();               // cheap, and the panel must not go stale
  refreshMap();                   // one stat() server-side until the map changes
  refreshNav();                   // landmarks + the attempt log, one read
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

  buildExplore();
  await refreshExplore();
  refreshMap();
  refreshNav();
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
