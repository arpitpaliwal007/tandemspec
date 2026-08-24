#!/bin/bash
cd "$(dirname "$0")/.."
echo "=== E2 ==="; python3 experiments/e2_companion_repair.py > results/e2.log 2>&1; echo "E2 exit $?"
echo "=== E4 ==="; python3 experiments/e4_quant_mismatch.py > results/e4.log 2>&1; echo "E4 exit $?"
echo "=== E3 ==="; python3 experiments/e3_serving_model.py > results/e3.log 2>&1; echo "E3 exit $?"
echo "=== ALL PILOT STAGES DONE ==="
