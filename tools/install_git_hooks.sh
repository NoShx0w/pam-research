#!/usr/bin/env bash

set -euo pipefail

mkdir -p .git/hooks
cp tools/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push

echo "Installed Git hooks."
