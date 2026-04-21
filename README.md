# LENS-PLUS

Local Environmental Navigation Support +

LENS-PLUS is a WebRTC prototype that streams video from a phone or desktop browser to a FastAPI backend. The backend returns mock guidance events over a WebRTC data channel and exposes debug endpoints to verify frame intake.

## What is implemented

- `web/` (Vite + TypeScript)
  - Camera source (`getUserMedia`) for phone testing
  - Video file source for desktop dev testing
  - WebRTC connect/disconnect flow
  - Data-channel event log and optional TTS
  - Overlay canvas scaffold for detection boxes
- `api/` (FastAPI + aiortc)
  - `POST /webrtc/offer`
  - `POST /webrtc/ice`
  - `GET /health`
  - `GET /debug/sessions`
  - `GET /debug/sessions/history`
  - `GET /debug/sessions/{session_id}/latest.jpg`
  - Per-session frame dump artifacts in `api/app/session_artifacts/`
  - Mock inference events streamed over WebRTC data channel `results`

## Repository layout

- `web/` frontend app
- `api/` backend signaling service
- `api/README.md` backend details + model integration guide
- `scripts/setup-dev-https.sh` local HTTPS helper for phone camera testing
- `docker-compose.yml` shared dev setup

## Prerequisites

- Docker Desktop (for Docker workflow)
- Python 3.11+ and `pip` (for local backend workflow)
- Node.js 18+ and npm (for local frontend workflow)
- `mkcert` (optional, but recommended for real phone camera testing)

## Environment file

Copy the example env file before running either workflow:

```bash
cp .env.example .env
```

Default `.env.example`:

```bash
VITE_SIGNALING_BASE_URL=http://localhost:8000
SNAPSHOT_INTERVAL_SECONDS=0.1
SNAPSHOT_JPEG_QUALITY=92
ANALYSIS_TARGET_FPS=5
```

`ANALYSIS_TARGET_FPS` controls server-side processing cadence and is clamped to `1..30`.

Optional backend env var:

- `SESSION_ARTIFACTS_DIR` (default: `api/app/session_artifacts` in local dev and `/app/app/session_artifacts` in Docker)
  - Directory where per-session processed frame dumps and manifest files are written.

For HTTPS + Docker phone testing, use `/api` for signaling and set `VITE_API_PROXY_TARGET=http://api:8000` (the helper script below configures this automatically).

For better detection quality from backend snapshots, keep:

- `SNAPSHOT_INTERVAL_SECONDS` around `0.1` (about 10 FPS snapshots)
- `SNAPSHOT_JPEG_QUALITY` around `90-95`

## Quick start (Docker)

1. Start services:

```bash
docker compose up --build
```

2. Open web app:

```text
http://localhost:5173
```

3. Verify API health:

```text
http://localhost:8000/health
```

## Quick start (Local, no Docker)

### Backend

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd web
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## LAN phone testing

- Connect phone and dev machine to the same Wi-Fi.
- Camera access on mobile browsers generally requires HTTPS (or localhost).
- If you open `http://<your-dev-machine-ip>:5173`, some browsers may block `getUserMedia` and show `navigator.mediaDevices` as undefined.
- Allow camera access when prompted.
- In the UI, select `Phone Camera`, click `Start Source`, then `Connect`.

If camera is blocked, use `Video File (dev)` mode or enable HTTPS as shown below.

### Local HTTPS with mkcert (recommended)

Fast path (auto setup):

```bash
scripts/setup-dev-https.sh
```

If IP auto-detect fails:

```bash
scripts/setup-dev-https.sh 192.168.1.42
```

The script:

- installs/trusts mkcert local CA
- creates `web/certs/dev-cert.pem` and `web/certs/dev-key.pem`
- updates `.env` with Docker-compatible HTTPS vars
- sets signaling to `/api` to avoid HTTPS mixed-content errors

Then restart services:

```bash
docker compose down
docker compose up --build
```

Open from phone:

```text
https://<your-dev-machine-ip>:5173
```

If your phone still shows trust warnings, install/trust the mkcert local CA on the phone.

### Confirm HTTPS is enabled

- Vite startup output should show `https://` URLs.
- If it still shows `http://`, `DEV_HTTPS` vars were not loaded.
- With HTTPS enabled, signaling should go to `/api/...` (Vite proxy), not `http://localhost:8000/...`.

## Signaling API contract

### `POST /webrtc/offer`

Request:

```json
{
  "sdp": "...",
  "type": "offer",
  "session_id": "optional"
}
```

Response:

```json
{
  "sdp": "...",
  "type": "answer",
  "session_id": "uuid"
}
```

### `POST /webrtc/ice`

Request:

```json
{
  "session_id": "uuid",
  "candidate": "candidate:...",
  "sdpMid": "0",
  "sdpMLineIndex": 0
}
```

Response:

```json
{
  "ok": true
}
```

## Visual stream proof (backend)

1. Start a stream and connect from the web UI.
2. Copy the session id from the UI log line:

```text
Connected signaling session <session_id>
```

Or use the built-in `Debug mode` toggle in the web UI and select a session.

3. Verify backend frame intake:

```text
https://<your-dev-machine-ip>:5173/api/debug/sessions
```

You should see `total_frames` increasing and `has_snapshot: true`.

4. Open live snapshot:

```text
https://<your-dev-machine-ip>:5173/api/debug/sessions/<session_id>/latest.jpg
```

Refresh to view updated snapshots from the incoming stream.

5. View persisted session dump history:

```text
https://<your-dev-machine-ip>:5173/api/debug/sessions/history
```

Each session artifact stores all processed frames (`frame-*.jpg`) plus `session.json` metadata.
The artifact path is ignored by git and excluded from API Docker build context.

## Clear session frame artifacts

Use the cleanup helper to remove all stored session dumps and recreate an empty artifact directory:

```bash
scripts/clean-session-artifacts.sh
```

You can also pass a custom artifact path:

```bash
scripts/clean-session-artifacts.sh /tmp/lens-plus-artifacts
```
