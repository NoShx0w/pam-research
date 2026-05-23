#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-main}"

echo "## Branch"
git branch --show-current

echo
echo "## Commits"
git log --oneline "$BASE"..HEAD

echo
echo "## Changed files"
git diff --name-only "$BASE"..HEAD

echo
echo "## Diff stat"
git diff --stat "$BASE"..HEAD

echo
echo "Please draft branch name, commit message, and PR description from this."
