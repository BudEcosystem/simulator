#!/usr/bin/env bash
#
# BudSimulator — fully automated, portable start script.
#
#   ./start.sh                  # start backend (FastAPI :8000) + frontend (React :3000)
#   ./start.sh --backend-only   # start only the backend API
#   ./start.sh --backend-port 8000 --frontend-port 3000
#
# Health-check based (no fixed sleeps), frees the ports first, and cleanly shuts BOTH down on Ctrl+C.
# No hardcoded paths — resolves everything from this script's location. Run ./setup.sh first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_ONLY=0
RUNDIR="$ROOT/.run"; mkdir -p "$RUNDIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend-only) BACKEND_ONLY=1; shift ;;
    --backend-port) BACKEND_PORT="$2"; shift 2 ;;
    --frontend-port) FRONTEND_PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m  ✗\033[0m %s\n' "$*" >&2; }

[[ -x "$VENV/bin/python" ]] || { err "venv missing — run ./setup.sh first"; exit 1; }
PYBIN="$VENV/bin/python"

# Portable "free this TCP port" using psutil (a backend dependency) — no lsof/fuser needed.
free_port() {
  "$PYBIN" - "$1" <<'PYKILL' 2>/dev/null || true
import sys
try:
    import psutil
except Exception:
    sys.exit(0)
port = int(sys.argv[1])
for c in psutil.net_connections(kind="inet"):
    if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
        try: psutil.Process(c.pid).terminate()
        except Exception: pass
PYKILL
}

BACKEND_PID=""; FRONTEND_PID=""
cleanup() {
  echo
  say "Shutting down..."
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  free_port "$BACKEND_PORT"; [[ "$BACKEND_ONLY" == 0 ]] && free_port "$FRONTEND_PORT"
  ok "stopped"
}
trap cleanup INT TERM EXIT

# ---- backend ----------------------------------------------------------------
say "Freeing port $BACKEND_PORT and starting backend (FastAPI)"
free_port "$BACKEND_PORT"
( cd "$ROOT/BudSimulator" && BACKEND_PORT="$BACKEND_PORT" "$PYBIN" run_api.py ) > "$RUNDIR/backend.log" 2>&1 &
BACKEND_PID=$!
ok "backend launched (pid $BACKEND_PID), logging to $RUNDIR/backend.log"

say "Waiting for backend health on http://localhost:$BACKEND_PORT ..."
HEALTHY=0
for _ in $(seq 1 60); do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    err "backend exited early — see $RUNDIR/backend.log"; tail -n 20 "$RUNDIR/backend.log" >&2; exit 1
  fi
  if "$PYBIN" - "$BACKEND_PORT" <<'PYHC' 2>/dev/null
import sys, urllib.request
port = sys.argv[1]
for path in ("/api/health", "/docs"):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=2) as r:
            if r.status < 500: sys.exit(0)
    except Exception:
        pass
sys.exit(1)
PYHC
  then HEALTHY=1; break; fi
  sleep 1
done
[[ "$HEALTHY" == 1 ]] && ok "backend is UP  →  http://localhost:$BACKEND_PORT/docs" \
  || { err "backend did not become healthy in 60s — see $RUNDIR/backend.log"; exit 1; }

# ---- frontend ---------------------------------------------------------------
if [[ "$BACKEND_ONLY" == 0 ]]; then
  if command -v npm >/dev/null 2>&1 && [[ -f "$ROOT/BudSimulator/frontend/package.json" ]]; then
    if [[ ! -d "$ROOT/BudSimulator/frontend/node_modules" ]]; then
      say "frontend deps missing — running npm install"
      ( cd "$ROOT/BudSimulator/frontend" && npm install --no-fund --no-audit )
    fi
    say "Freeing port $FRONTEND_PORT and starting frontend (React)"
    free_port "$FRONTEND_PORT"
    ( cd "$ROOT/BudSimulator/frontend" && PORT="$FRONTEND_PORT" BROWSER=none npm start ) > "$RUNDIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    ok "frontend launched (pid $FRONTEND_PID)  →  http://localhost:$FRONTEND_PORT  (logging to $RUNDIR/frontend.log)"
  else
    err "npm/frontend not found — starting backend only"
  fi
fi

echo
say "BudSimulator is running. Press Ctrl+C to stop."
say "  Backend  : http://localhost:$BACKEND_PORT/docs"
[[ "$BACKEND_ONLY" == 0 && -n "$FRONTEND_PID" ]] && say "  Frontend : http://localhost:$FRONTEND_PORT"
# Wait on the foreground process(es); cleanup runs on exit/Ctrl+C.
wait "$BACKEND_PID" ${FRONTEND_PID:+"$FRONTEND_PID"}
