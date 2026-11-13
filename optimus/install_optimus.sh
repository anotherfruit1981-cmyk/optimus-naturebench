#!/usr/bin/env bash
# Install the OPTIMUS harness into a NatureBench checkout.
#   bash install_optimus.sh [path-to-NatureBench]     (default: ~/NatureBench)
# Safe to run more than once.
set -euo pipefail

NB="${1:-$HOME/NatureBench}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$NB/agent/adapter.py" ]; then
  echo "NatureBench not found at $NB (expected $NB/agent/adapter.py)"; exit 1
fi

cp "$SRC/agent/optimus_adapter.py" "$NB/agent/optimus_adapter.py"
rm -rf "$NB/agent/optimus"
cp -r "$SRC/agent/optimus" "$NB/agent/optimus"
chmod +x "$NB/agent/optimus/bin/"* "$NB/agent/optimus/optimus_loop.py"

if ! grep -q "optimus_adapter" "$NB/agent/__init__.py"; then
  printf '\nfrom . import optimus_adapter  # noqa: F401,E402  (registers the OPTIMUS adapter on import)\n' >> "$NB/agent/__init__.py"
fi

# Let the judge recognise OpenRouter's "openai/gpt-5.5" model id (no-op if already patched).
sed -i 's|model.lower().startswith("gpt-")|model.lower().split("/")[-1].startswith("gpt-")|' "$NB/judge_core/clients.py"

PY="python3"
[ -x "$HOME/miniconda3/envs/naturebench/bin/python" ] && PY="$HOME/miniconda3/envs/naturebench/bin/python"
cd "$NB"
"$PY" - <<'EOF'
import agent.cli_adapters  # noqa: F401
from agent.adapter import REGISTRY
assert REGISTRY.has("optimus"), REGISTRY.names()
print("OPTIMUS installed. Registered agents:", ", ".join(REGISTRY.names()))
EOF
grep -q 'split("/")\[-1\].startswith("gpt-")' judge_core/clients.py && echo "Judge patch: OK"
