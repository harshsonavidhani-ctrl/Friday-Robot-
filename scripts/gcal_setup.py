#!/usr/bin/env python3
"""ONE-TIME Google Calendar sign-in for Friday (File 16 step 4). Run it on a
LAPTOP, not the Pi -- it opens your browser for Google's consent screen.

    python3 gcal_setup.py client_secret.json

Needs only Python 3 (standard library; nothing to pip install). It writes
google_token.json next to where you ran it. Copy that ONE file to the Pi:

    scp google_token.json harsh@Harsh.local:~/friday/secrets/google_token.json

and then on the Pi:

    chmod 600 ~/friday/secrets/google_token.json

BEFORE RUNNING (once, in a browser, ~5 minutes):
  1. https://console.cloud.google.com/ -> create a project (any name, "Friday").
  2. APIs & Services -> Library -> "Google Calendar API" -> Enable.
  3. APIs & Services -> OAuth consent screen -> External -> fill the app name
     and your email -> add YOURSELF under "Test users".
  4. APIs & Services -> Credentials -> Create credentials -> OAuth client ID ->
     Application type "Desktop app" -> Create -> Download JSON. That file is
     the client_secret.json this script takes.

WHAT IT ASKS FOR: calendar.events -- read and create events on your
calendars, nothing else (no Gmail, no Drive). The token never leaves your
laptop except to the Pi's secrets/ folder, which git ignores.
"""

import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser

SCOPE = "https://www.googleapis.com/auth/calendar.events"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    with open(sys.argv[1]) as handle:
        raw = json.load(handle)
    client = raw.get("installed") or raw.get("web") or raw
    client_id, client_secret = client["client_id"], client.get("client_secret", "")
    auth_uri = client.get("auth_uri", "https://accounts.google.com/o/oauth2/auth")
    token_uri = client.get("token_uri", "https://oauth2.googleapis.com/token")

    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    got: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):                            # noqa: N802
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            got.update({k: v[0] for k, v in query.items()})
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            ok = "code" in got and got.get("state") == state
            self.wfile.write((
                "<h2>Friday is connected to your calendar. You can close this tab.</h2>"
                if ok else "<h2>Something went wrong — see the terminal.</h2>"
            ).encode())

        def log_message(self, *args):                # quiet
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    redirect = f"http://127.0.0.1:{server.server_address[1]}"
    url = auth_uri + "?" + urllib.parse.urlencode({
        "client_id": client_id, "redirect_uri": redirect,
        "response_type": "code", "scope": SCOPE, "access_type": "offline",
        "prompt": "consent", "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256"})
    threading.Thread(target=server.handle_request, daemon=True).start()
    print("Opening your browser for Google's consent screen...\n"
          "If it does not open, paste this into a browser on THIS computer:\n\n"
          f"{url}\n")
    webbrowser.open(url)

    deadline = time.time() + 300
    while "code" not in got and "error" not in got and time.time() < deadline:
        time.sleep(0.3)
    if got.get("error") or "code" not in got or got.get("state") != state:
        print(f"Sign-in did not finish: {got.get('error', 'timed out')}")
        return 1

    body = urllib.parse.urlencode({
        "code": got["code"], "client_id": client_id,
        "client_secret": client_secret, "redirect_uri": redirect,
        "grant_type": "authorization_code", "code_verifier": verifier}).encode()
    with urllib.request.urlopen(urllib.request.Request(token_uri, data=body)) as r:
        token = json.loads(r.read().decode())
    if "refresh_token" not in token:
        print("Google did not return a refresh token. Remove Friday's access at "
              "https://myaccount.google.com/permissions and run this again.")
        return 1

    out = {"client_id": client_id, "client_secret": client_secret,
           "refresh_token": token["refresh_token"],
           "access_token": token.get("access_token", ""),
           "expires_at": time.time() + float(token.get("expires_in", 3600)),
           "token_uri": token_uri, "scope": SCOPE}
    with open("google_token.json", "w") as handle:
        json.dump(out, handle)
    os.chmod("google_token.json", 0o600)
    print("Wrote google_token.json. Now copy it to the Pi:\n\n"
          "  scp google_token.json harsh@Harsh.local:~/friday/secrets/google_token.json\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
