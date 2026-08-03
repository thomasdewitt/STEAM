set -e
cd ~/code-and-data/interp-tests/align-false/turbulon-model
export TMPDIR=$PWD/tmp PYTHONPATH=$PWD
PY=~/code-and-data/turbulon-analysis/.venv/bin/python
H=tests/heavy/interpolation_compensation.py

echo "=== P1 canonical 384km (y=2..32), 3 seeds ==="
date
$PY $H TAG=can384 K_TARGET=12000 DOMAIN=384000 OUTER=96000 SPHEROSCALE=10 \
    DX_LIST=6000,3000,1500,750,375 SEEDS=0,1,2 | grep -v '^Step'

echo "=== P2 canonical 96km deep (y=2..256), 1 seed ==="
date
$PY $H TAG=can96deep K_TARGET=12000 DOMAIN=96000 OUTER=96000 SPHEROSCALE=10 \
    DX_LIST=6000,3000,1500,750,375,187.5,93.75,46.875 SEEDS=0 | grep -v '^Step'

echo "=== P3 isotropic (y=2..64), 3 seeds ==="
date
$PY $H TAG=iso SPHEROSCALE=1e7 K_TARGET=1500 DOMAIN=12000 OUTER=12000 \
    HEIGHT=16000 PROFILE_DZ=100 BAND=4000,12000 \
    DX_LIST=750,375,187.5,93.75,46.875,23.4375 SEEDS=0,1,2 | grep -v '^Step'
date
echo ALLDONE
