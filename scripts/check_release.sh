#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

echo "==> Compile Python sources"
python3 -m compileall -q paths.py api_key_util.py scripts experiments tests vllm_specsteer

echo "==> Run static regression tests"
python3 -m unittest discover -s tests -v

echo "==> Check shell syntax"
while IFS= read -r -d '' script; do
  bash -n "${script}"
done < <(git ls-files -z '*.sh')

if python3 -c 'import yaml' >/dev/null 2>&1; then
  echo "==> Parse YAML metadata"
  python3 - <<'PY'
from pathlib import Path
import yaml

for name in ["CITATION.cff", "configs/paper.yaml", "conf.example.yaml"]:
    with Path(name).open() as handle:
        yaml.safe_load(handle)
    print(f"ok: {name}")
PY
else
  echo "==> PyYAML unavailable; YAML parsing skipped"
fi

echo "Release static checks passed."
