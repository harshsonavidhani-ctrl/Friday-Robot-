#!/usr/bin/env bash
# Generate TLS material for Friday's dashboard.
#
# A plain self-signed certificate always shows "Not secure", because nothing
# vouches for it. So this makes TWO things:
#
#   1. a small local CA  (certs/friday-ca.crt / .key)  -- install the .crt on
#      your phone ONCE, and it is trusted from then on
#   2. the dashboard's own certificate, signed by that CA
#
# After installing the CA, the padlock is closed and the PWA installs cleanly.
# Regenerating the server cert later does NOT need a second phone install --
# the CA stays the same, so only re-run with --new-ca if you really mean it.
#
# Everything written lives inside Friday's project folder. No system trust
# store, no system config, no network settings are touched.
#
# Usage:  ./scripts/make_dashboard_cert.sh [--new-ca] [extra-host-or-ip ...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/certs"
CA_CRT="$OUT/friday-ca.crt"
CA_KEY="$OUT/friday-ca.key"
CRT="$OUT/dashboard.crt"
KEY="$OUT/dashboard.key"
CA_DAYS=3650
DAYS=825          # iOS/macOS reject server certs valid for longer than this

NEW_CA=0
EXTRA=()
for arg in "$@"; do
  if [ "$arg" = "--new-ca" ]; then NEW_CA=1; else EXTRA+=("$arg"); fi
done

mkdir -p "$OUT"

# --- 1. the CA -------------------------------------------------------
if [ ! -f "$CA_CRT" ] || [ ! -f "$CA_KEY" ] || [ "$NEW_CA" = 1 ]; then
  echo "creating a new local CA"
  openssl req -x509 -newkey rsa:2048 -nodes -sha256 \
    -keyout "$CA_KEY" -out "$CA_CRT" -days "$CA_DAYS" \
    -subj "/CN=Friday Local CA/O=Friday" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" 2>/dev/null
  chmod 600 "$CA_KEY"; chmod 644 "$CA_CRT"
else
  echo "reusing the existing CA (pass --new-ca to replace it)"
fi

# --- 2. every address the Pi answers on ------------------------------
# All of them must be in the SAN list; browsers have ignored CN for years.
NAMES=("$(hostname)" "$(hostname).local" "localhost" "friday" "friday.local")
IPS=("127.0.0.1")
while read -r ip; do
  [ -n "$ip" ] && IPS+=("$ip")
done < <(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.' || true)

for extra in ${EXTRA+"${EXTRA[@]}"}; do
  if [[ "$extra" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    IPS+=("$extra")
  else
    NAMES+=("$extra")
  fi
done

SAN=""
i=1; for n in "${NAMES[@]}"; do SAN+="DNS.$i:$n,"; i=$((i+1)); done
i=1; for a in "${IPS[@]}";   do SAN+="IP.$i:$a,";  i=$((i+1)); done
SAN="${SAN%,}"
echo "SAN: $SAN"

# --- 3. the server certificate, signed by the CA ---------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/ext.cnf" <<EXT
subjectAltName=$SAN
basicConstraints=CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
EXT

openssl req -newkey rsa:2048 -nodes -sha256 \
  -keyout "$KEY" -out "$TMP/csr.pem" \
  -subj "/CN=friday-dashboard/O=Friday" 2>/dev/null

openssl x509 -req -in "$TMP/csr.pem" -sha256 -days "$DAYS" \
  -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
  -extfile "$TMP/ext.cnf" -out "$CRT" 2>/dev/null

chmod 600 "$KEY"; chmod 644 "$CRT"

IP="$(hostname -I | awk '{print $1}')"
cat <<MSG

wrote $CA_CRT   <- install this on the phone, once
wrote $CRT
wrote $KEY

To get rid of "Not secure" on your phone:

  Android
    1. open  http://$IP:8080/ca.crt   (plain http, no warning)
    2. Settings > Security > More settings > Encryption & credentials
       > Install a certificate > CA certificate > pick the downloaded file
    3. open  https://$IP:8443/  -- padlock closed, then Add to Home Screen

  iPhone / iPad
    1. open  http://$IP:8080/ca.crt  in Safari, allow the profile download
    2. Settings > Profile Downloaded > Install
    3. Settings > General > About > Certificate Trust Settings
       > turn ON "Friday Local CA"      <- this step is required
    4. open  https://$IP:8443/  -- padlock closed, then Add to Home Screen
MSG
