#!/bin/zsh
# Unattended v1.2 chain: finish calibration -> TOML -> fleet -> merge ->
# annual probability -> maps. Every stage is resumable and gated on the
# artifact the next stage needs, so this can be killed and relaunched at any
# point without redoing finished work.
set -u
# Interpreter: set FIRESCAPE_PYTHON to your env's python, or activate it first.
PY="${FIRESCAPE_PYTHON:-python3}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOG=/tmp/v12_overnight.log
TOML=$REPO/firescape/data/calibration/statewide_v1_2.toml
DATA="${FIRESCAPE_DATA:-$HOME/Library/CloudStorage/Box-Box/SWMresearch/PostFireDebrisFlows/PreFireAssessment}"
PROD="$DATA/products/prefire/statewide_v1_2"

say() { print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

say "=== v1.2 chain starting ==="

# 1. calibration: exit 42 means more fires remain. MARTIN needs the full
#    raster path (~20 min); everything else re-solves from cache in ~2 s.
for i in {1..12}; do
  FIRESCAPE_CAL_BUDGET=3600 $PY $REPO/scripts/calibrate/p7_calib_driver.py >> $LOG 2>&1
  rc=$?
  say "calib pass $i rc=$rc"
  [ $rc -eq 0 ] && break
  [ $rc -ne 42 ] && { say "calib FAILED rc=$rc"; exit 1; }
done

# 2. regional medians -> statewide_v1_2.toml
if [ ! -f $TOML ]; then
  $PY $REPO/scripts/calibrate/p7_calib_summary.py >> $LOG 2>&1 || { say "summary FAILED"; exit 1; }
fi
[ -f $TOML ] || { say "no $TOML"; exit 1; }
say "calibration TOML ready"
grep -E '^\[|pdsim|break' $TOML | head -30 | tee -a $LOG

# 3. fleet: 591 HU10 units, two lanes, budgeted passes
for i in {1..60}; do
  $REPO/scripts/surface/sw_fleet_v12.sh >> $LOG 2>&1
  rc=$?
  n=$(ls $PROD/*_meta.json 2>/dev/null | wc -l | tr -d ' ')
  say "fleet pass $i rc=$rc units=$n"
  [ $rc -eq 0 ] && break
  [ $rc -ne 42 ] && { say "fleet FAILED rc=$rc"; exit 1; }
done

# 4-6. merge, annual probability, maps
$PY $REPO/scripts/surface/sw_merge_v12.py >> $LOG 2>&1 || { say "merge FAILED"; exit 1; }
say "merged"
$PY $REPO/scripts/surface/p7_annual_probability.py statewide_v1_2 >> $LOG 2>&1 \
  || { say "annualprob FAILED"; exit 1; }
say "annual probability added"
$PY $REPO/scripts/figures/sw_maps_v12.py >> $LOG 2>&1 || say "hazard map FAILED"
$PY $REPO/scripts/figures/p7_annual_probability_map.py statewide_v1_2 >> $LOG 2>&1 \
  || say "annualprob map FAILED"

say "=== v1.2 chain COMPLETE ==="
tail -40 $LOG
