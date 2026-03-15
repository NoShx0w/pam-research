#!/usr/bin/env bash

set -e

echo "Running PAM repository verification..."

python tools/repo_check.py
echo ""

python tools/repo_check_with_scaffolding.py

echo ""
echo "Verification complete."
