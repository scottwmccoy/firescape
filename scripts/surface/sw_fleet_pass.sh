#!/bin/zsh
# One time-budgeted statewide fleet pass: both lanes in parallel.
set -u
# Interpreter: set FIRESCAPE_PYTHON to your env's python, or activate it first.
PY="${FIRESCAPE_PYTHON:-python3}"
SP="$(cd "$(dirname "$0")" && pwd)"
export FIRESCAPE_M4_BUDGET=520 FIRESCAPE_NLANES=2

FIRESCAPE_LANE=0 $PY "$SP/sw_driver.py" > "$SP/fleet_lane0.log" 2>&1 &
p0=$!
FIRESCAPE_LANE=1 $PY "$SP/sw_driver.py" > "$SP/fleet_lane1.log" 2>&1 &
p1=$!
wait $p0; r0=$?
wait $p1; r1=$?
echo "--- lane summaries ---"
tail -2 "$SP/fleet_lane0.log"
tail -2 "$SP/fleet_lane1.log"
echo "rcs: lane0=$r0 lane1=$r1"
if [ $r0 -eq 42 ] || [ $r1 -eq 42 ]; then exit 42; fi
if [ $r0 -ne 0 ] || [ $r1 -ne 0 ]; then exit 1; fi
exit 0
