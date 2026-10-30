/* Friday dashboard service worker.
 *
 * Deliberately minimal. The dashboard's whole point is live data, so nothing
 * dynamic is ever served from cache -- the SSE stream, the camera feeds and
 * every /api/ call go straight to the network. The cache exists only so the
 * app shell opens instantly and so an installed PWA still shows its own UI
 * (with a disconnected status dot) when the Pi is off.
 */
const CACHE = "friday-shell-v6";   // bump to force a clean activate after a UI change
const SHELL = [
  "/",
  "/static/style.css",
  "/static/app.js",
  "/static/icon-192.png",
  "/static/icon-512.png",
  "/static/icon-maskable-192.png",
  "/static/icon-maskable-512.png",
  "/static/apple-touch-icon-180.png",
  "/static/manifest.webmanifest",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  // Never cache live endpoints.
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/cam/")) {
    return;
  }
  // Network first, fall back to the cached shell when the Pi is unreachable.
  e.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy));
        }
        return res;
      })
      .catch(() => caches.match(req).then((hit) => hit || caches.match("/")))
  );
});
