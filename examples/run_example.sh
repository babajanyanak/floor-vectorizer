#!/usr/bin/env bash
# Example: run the CLI against sample inputs.
set -euo pipefail

python vectorize_floor.py \
  --plan ./input/floor_2_plan.png \
  --overlay ./input/floor_2_overlay.png \
  --mapping ./input/floor_2_lots.csv \
  --floor-id floor_2 \
  --out-dir ./output/floor_2 \
  --debug \
  --log-level INFO
