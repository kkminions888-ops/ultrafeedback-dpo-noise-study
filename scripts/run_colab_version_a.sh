#!/usr/bin/env bash
set -euo pipefail

python main.py batch --plan configs/version_a.yaml --results-root results --data-root data
