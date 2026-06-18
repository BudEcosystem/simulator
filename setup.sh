#!/usr/bin/env bash
#
# BudSimulator — fully automated, portable setup.
#
#   ./setup.sh                 # full setup: venv + core engine + backend deps + frontend + DB
#   ./setup.sh --no-frontend   # skip the React frontend (npm) install
#   ./setup.sh --python python3.12   # use a specific python interpreter
#
# Idempotent: safe to re-run. No hardcoded paths — resolves the repo root from this script's location.
set -euo pipefail

# ---- resolve repo root (this script lives at the repo root) -----------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PYTHON_BIN=""
DO_FRONTEND=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-frontend) DO_FRONTEND=0; shift ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ! \033[0m%s\n' "$*"; }

# ---- pick a Python interpreter (>=3.9) --------------------------------------
if [[ -z "$PYTHON_BIN" ]]; then
  for c in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PYTHON_BIN="$c"; break; fi
  done
fi
[[ -n "$PYTHON_BIN" ]] || { echo "No python3 found. Install Python >=3.9."; exit 1; }
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] >= (3,9), sys.version' \
  || { echo "Python >=3.9 required (found $("$PYTHON_BIN" -V))."; exit 1; }
say "Using Python: $("$PYTHON_BIN" -V) ($PYTHON_BIN)"

# ---- virtual environment ----------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
  say "Creating virtualenv at $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
  ok "venv created"
else
  ok "venv already exists at $VENV"
fi
PIP="$VENV/bin/python -m pip"
# Upgrade pip only; leave setuptools/wheel as-is. torch pins setuptools<82, so an unbounded
# `--upgrade setuptools` introduces a resolver conflict — pin it defensively if it must be touched.
$PIP install --quiet --upgrade pip "setuptools<82" wheel
ok "pip toolchain up to date"

# ---- core engine (editable) — the piece setup.py historically forgot --------
say "Installing core engine: llm-memory-calculator (editable)"
$PIP install --quiet -e "$ROOT/llm-memory-calculator"
ok "llm-memory-calculator installed (editable)"

# ---- backend dependencies + BudSimulator ------------------------------------
say "Installing BudSimulator backend requirements"
$PIP install --quiet -r "$ROOT/BudSimulator/requirements.txt"
ok "backend requirements installed"
if [[ -f "$ROOT/BudSimulator/pyproject.toml" || -f "$ROOT/BudSimulator/setup.py" ]]; then
  $PIP install --quiet -e "$ROOT/BudSimulator" 2>/dev/null \
    && ok "BudSimulator installed (editable)" \
    || warn "BudSimulator editable install skipped (runs fine from source via run_api.py)"
fi

# ---- sanity import check (fail loudly if a dep is missing) ------------------
say "Verifying core imports"
"$VENV/bin/python" - <<'PYCHK'
import importlib.util, sys
mods = ["llm_memory_calculator", "fastapi", "uvicorn", "pydantic", "numpy", "pandas",
        "scipy", "yaml", "sqlalchemy", "transformers", "huggingface_hub", "pymoo"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print("  MISSING after install:", ", ".join(missing)); sys.exit(1)
print("  all core imports OK")
PYCHK
ok "imports verified"

# ---- database ---------------------------------------------------------------
say "Initializing database"
if [[ -f "$ROOT/BudSimulator/data/prepopulated.db" ]]; then
  ok "prepopulated.db present"
fi
if [[ -f "$ROOT/BudSimulator/scripts/setup_database.py" ]]; then
  ( cd "$ROOT/BudSimulator" && "$VENV/bin/python" scripts/setup_database.py ) >/dev/null 2>&1 \
    && ok "database initialized" || warn "setup_database.py reported issues (prepopulated.db still usable)"
fi

# ---- frontend ---------------------------------------------------------------
if [[ "$DO_FRONTEND" == "1" ]]; then
  if command -v npm >/dev/null 2>&1 && [[ -f "$ROOT/BudSimulator/frontend/package.json" ]]; then
    say "Installing frontend dependencies (npm)"
    ( cd "$ROOT/BudSimulator/frontend" && npm install --no-fund --no-audit ) \
      && ok "frontend deps installed" || warn "npm install failed — backend still usable"
  else
    warn "npm or frontend/package.json not found — skipping frontend (backend-only setup)"
  fi
else
  warn "frontend install skipped (--no-frontend)"
fi

echo
say "Setup complete. Start everything with:  ./start.sh"
