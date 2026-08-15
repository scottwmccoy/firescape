#!/bin/zsh
# One time-budgeted pass over all three staging lanes, in parallel.
# Exit 42 if any lane still has pending work (relaunch me), 0 when all done.
set -u
PY=/opt/anaconda3/envs/FireMan/bin/python
SP="$(cd "$(dirname "$0")" && pwd)"
export FIRESCAPE_STAGE_BUDGET=470
export FIRESCAPE_LFPS_EMAIL=scottmccoy@unr.edu

$PY "$SP/sw_stage_dem.py"   > "$SP/lane_dem.log"   2>&1 &
p1=$!
$PY "$SP/sw_stage_evt.py"   > "$SP/lane_evt.log"   2>&1 &
p2=$!
$PY "$SP/sw_stage_soils.py" > "$SP/lane_soils.log" 2>&1 &
p3=$!
wait $p1; r1=$?
wait $p2; r2=$?
wait $p3; r3=$?
echo "--- lane tails ---"
for f in lane_dem lane_evt lane_soils; do
  echo "[$f rc]"; tail -2 "$SP/$f.log"
done
echo "rcs: dem=$r1 evt=$r2 soils=$r3"
if [ $r1 -eq 42 ] || [ $r2 -eq 42 ] || [ $r3 -eq 42 ]; then exit 42; fi
if [ $r1 -ne 0 ] || [ $r2 -ne 0 ] || [ $r3 -ne 0 ]; then exit 1; fi
exit 0
