#!/usr/bin/env bash
set -euo pipefail

# All operations before publication are performed in the disposable Actions checkout.
CURRENT="$(python - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as file:
    print(tomllib.load(file)["project"]["version"])
PY
)"
PERIOD="$(date -u +%Y).$(date -u +%-m)"
CURRENT_PERIOD="$(python - "$CURRENT" <<'PY'
import sys
year, month, _ = sys.argv[1].split(".")
print(f"{int(year)}.{int(month)}")
PY
)"
if [[ "$CURRENT_PERIOD" == "$PERIOD" ]]; then
  PART=patch
else
  PART=release
fi
echo "Bumping $CURRENT with part $PART (UTC period: $PERIOD)"
uv run --no-sync bump-my-version bump "$PART" --no-commit --no-tag

NEW_VERSION="$(python - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as file:
    print(tomllib.load(file)["project"]["version"])
PY
)"
echo "Proposed version: $NEW_VERSION"
export NEW_VERSION

python - <<'PY'
import os
import re
from datetime import datetime, timezone

value = os.environ["NEW_VERSION"]
assert re.fullmatch(r"[1-9]\d{3}\.(?:[1-9]|1[0-2])\.\d+", value), value
year, month, _ = map(int, value.split("."))
now = datetime.now(timezone.utc)
assert (year, month) == (now.year, now.month), value
PY
if git show-ref --verify --quiet "refs/tags/v$NEW_VERSION" ||
   git ls-remote --exit-code --tags origin "refs/tags/v$NEW_VERSION" > /dev/null; then
  echo "::error::Release tag v$NEW_VERSION already exists"
  exit 1
fi

uv lock
uv lock --check
uv sync --locked --no-default-groups --group dev
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv build

uv venv "$RUNNER_TEMP/cpax-release-install" --python 3.12
uv pip install --python "$RUNNER_TEMP/cpax-release-install/bin/python" dist/*.whl
"$RUNNER_TEMP/cpax-release-install/bin/python" - <<'PY'
import os
from importlib.metadata import version
import cpax

assert cpax.__version__ == version("CPax")
assert cpax.__version__ == os.environ["NEW_VERSION"]
PY

git diff --check
git diff --stat
git diff -- pyproject.toml uv.lock
if [[ -n "$(git status --porcelain --untracked-files=no -- . ':!pyproject.toml' ':!uv.lock')" ]]; then
  echo "::error::Unexpected tracked-file changes"
  exit 1
fi

{
  echo "## CPax release preview"
  echo "New version: $NEW_VERSION"
  echo "Tag: v$NEW_VERSION"
  echo "Dry run: $DRY_RUN"
  echo "All package and test checks passed."
} >> "$GITHUB_STEP_SUMMARY"
