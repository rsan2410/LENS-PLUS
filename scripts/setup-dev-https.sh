#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="$ROOT_DIR/web/certs"
KEY_FILE="$CERT_DIR/dev-key.pem"
CERT_FILE="$CERT_DIR/dev-cert.pem"
ENV_FILE="$ROOT_DIR/.env"

if ! command -v mkcert >/dev/null 2>&1; then
  printf "mkcert is required but not installed.\n"
  printf "Install on macOS with: brew install mkcert\n"
  exit 1
fi

detect_ip() {
  local ip=""
  ip="$(ipconfig getifaddr en0 2>/dev/null || true)"
  if [ -z "$ip" ]; then
    ip="$(ipconfig getifaddr en1 2>/dev/null || true)"
  fi
  printf "%s" "$ip"
}

LAN_IP="${1:-$(detect_ip)}"
if [ -z "$LAN_IP" ]; then
  printf "Could not auto-detect LAN IP.\n"
  printf "Run again with your LAN IP: scripts/setup-dev-https.sh 192.168.1.42\n"
  exit 1
fi

mkdir -p "$CERT_DIR"

printf "Installing local CA (mkcert -install)...\n"
mkcert -install

printf "Generating cert for localhost + %s...\n" "$LAN_IP"
mkcert \
  -key-file "$KEY_FILE" \
  -cert-file "$CERT_FILE" \
  localhost 127.0.0.1 "$LAN_IP"

python3 - "$ENV_FILE" <<'PY'
from pathlib import Path
import sys

env_path = Path(sys.argv[1])
existing = {}

if env_path.exists():
    for line in env_path.read_text().splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        existing[key.strip()] = value.strip()

existing["DEV_HTTPS"] = "true"
existing["DEV_HTTPS_KEY_FILE"] = "/app/certs/dev-key.pem"
existing["DEV_HTTPS_CERT_FILE"] = "/app/certs/dev-cert.pem"
existing["VITE_SIGNALING_BASE_URL"] = "/api"
existing.setdefault("VITE_API_PROXY_TARGET", "http://api:8000")

ordered_keys = [
    "VITE_SIGNALING_BASE_URL",
    "VITE_API_PROXY_TARGET",
    "DEV_HTTPS",
    "DEV_HTTPS_KEY_FILE",
    "DEV_HTTPS_CERT_FILE",
]

lines = [f"{key}={existing[key]}" for key in ordered_keys]

for key in sorted(k for k in existing.keys() if k not in set(ordered_keys)):
    lines.append(f"{key}={existing[key]}")

env_path.write_text("\n".join(lines) + "\n")
PY

printf "\nHTTPS dev setup complete.\n"
printf "- Cert: %s\n" "$CERT_FILE"
printf "- Key:  %s\n" "$KEY_FILE"
printf "- LAN URL: https://%s:5173\n" "$LAN_IP"
printf "\nNext steps:\n"
printf "1) docker compose down\n"
printf "2) docker compose up --build\n"
printf "3) Confirm Vite logs show https:// URLs\n"
