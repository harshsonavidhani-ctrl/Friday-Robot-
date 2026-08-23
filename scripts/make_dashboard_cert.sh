#!/usr/bin/env bash
# Generate a self-signed TLS certificate for Friday's dashboard.
#
# A PWA will only install (and only gets a service worker) from a secure
# context, so the dashboard needs HTTPS even on the LAN. Everything this
# script writes lives inside Friday's own project folder -- it does not
# touch any system trust store, any system config, or the network.
#
# Usage:  ./scripts/make_dashboard_cert.sh [extra-hostname-or-ip ...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/certs"
CRT="$OUT/dashboard.crt"
KEY="$OUT/dashboard.key"
DAYS=3650

mkdir -p "$OUT"

# Every address the Pi can be reached on has to be in the SAN list, or the
# browser rejects the cert outright (CN alone has not been honoured for years).
NAMES=("$(hostname)" "$(hostname).local" "localhost")
IPS=("127.0.0.1")
while read -r ip; do
  [ -n "$ip" ] && IPS+=("$ip")
done < <(hostname -I | tr ' ' '\n' | grep -E '^[0-9]+\.' || true)

for extra in "$@"; do
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

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout "$KEY" -out "$CRT" -days "$DAYS" -sha256 \
  -subj "/CN=friday-dashboard" \
  -addext "subjectAltName=$SAN" \
  -addext "basicConstraints=CA:FALSE" \
  -addext "keyUsage=digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth" 2>/dev/null

chmod 600 "$KEY"
chmod 644 "$CRT"

echo "wrote $CRT"
echo "wrote $KEY"
echo
echo "On the phone: open https://$(hostname -I | awk '{print $1}'):8443/ once,"
echo "accept the warning, then use Add to Home Screen. To skip the warning"
echo "entirely, install $CRT on the phone as a trusted CA."
