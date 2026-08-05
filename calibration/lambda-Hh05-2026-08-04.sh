#!/bin/bash
# lambda calibration at H_h = 0.5 (Thomas's ask, 2026-08-04 night), plus a
# far-bounds control at 0.45.
#
# Why the control: the production constant lambda(H_h=0.45) = 0.28927 was
# fitted under the PRE-2026-08-03 bounded procedure. Every run below uses the
# far-bounds procedure (b86a539), so lambda(0.5) from job 2 is not comparable
# to 0.28927 as-is -- part of any difference would be the procedure, not H_h.
# Jobs 3/4 refit 0.45 under the same procedure so the H_h effect is isolated.
# (A 0.45 far-bounds refit is independently on the open list: "lambda refit
# (production config)".)
#
# Strictly serial (the 07-29 OOM lesson). Simulations are CPU -- simulate()
# defaults to device='cpu' and the calibration does not override it; only the
# scaleinvariance Haar analysis touches CUDA.
#
# Nothing here overwrites existing files:
#   - round1_Hh0.5.npz / round1_Hh0.45.npz do not exist yet
#   - the 07-30 bounded round2_Hh0.45_iter{1..4}.npz are preserved by
#     --start-iter 11 on job 4 (writes iter11+, figures likewise)
set -x
cd "$(dirname "$0")/.."
PY=.venv/bin/python
CAL=calibration
LOG=$CAL

echo "=== [1/4] round1 H_h=0.5 (lambda=1 baseline) ($(date +%H:%M)) ==="
$PY -u $CAL/calibrate_lambda.py round1 --hurst-horizontal 0.5 \
    2>&1 | tee $LOG/round1_Hh0.5.log

echo "=== [2/4] iterate H_h=0.5 to fixed point ($(date +%H:%M)) ==="
$PY -u $CAL/calibrate_lambda.py iterate --hurst-horizontal 0.5 \
    --tol 0.02 --max-iters 4 \
    2>&1 | tee $LOG/iterate_Hh0.5.log

echo "=== [3/4] CONTROL round1 H_h=0.45 far-bounds ($(date +%H:%M)) ==="
$PY -u $CAL/calibrate_lambda.py round1 --hurst-horizontal 0.45 \
    2>&1 | tee $LOG/round1_Hh0.45_farbounds.log

echo "=== [4/4] CONTROL iterate H_h=0.45 far-bounds ($(date +%H:%M)) ==="
$PY -u $CAL/calibrate_lambda.py iterate --hurst-horizontal 0.45 \
    --tol 0.02 --max-iters 4 --start-iter 11 \
    2>&1 | tee $LOG/iterate_Hh0.45_farbounds.log

echo "=== QUEUE DONE ($(date +%H:%M)) ==="
