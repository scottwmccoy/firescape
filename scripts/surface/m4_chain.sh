#!/bin/zsh
# Wait for the running M4 fleet, resume until every HU10 is done, then merge + map.
set -u
PY=/opt/anaconda3/envs/FireMan/bin/python
SP="/private/tmp/claude-501/-Users-scottmccoy-Library-CloudStorage-Box-Box-SWMresearch-PostFireDebrisFlows-PreFireAssessment/7eafe9f9-5a07-4337-8ea7-7267ec1788f2/scratchpad"
OUT="/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch/PostFireDebrisFlows/PreFireAssessment/products/prefire/pilot_v1"
cd "$SP" || exit 1

count() { ls "$OUT" 2>/dev/null | grep -c 'meta.json'; }

# 1) let the in-flight fleet finish
while pgrep -f "m4_driver.py" > /dev/null; do sleep 15; done
echo "in-flight fleet exited; units done: $(count)/35"

# 2) resume until complete or no further progress
prev=-1
for attempt in 1 2 3 4 5 6 7 8; do
  now=$(count)
  if [ "$now" -ge 35 ]; then echo "all 35 units complete"; break; fi
  if [ "$now" -eq "$prev" ]; then
    echo "no progress on attempt $attempt (stuck at $now) - stopping resume loop"
    break
  fi
  prev=$now
  echo "resume attempt $attempt (at $now/35)"
  FIRESCAPE_M4_BUDGET=3300 $PY m4_driver.py
done

echo "=== FINAL UNIT COUNT: $(count)/35 ==="
echo "=== MERGE + ANNUAL PROBABILITY ==="
$PY m5_merge.py || exit 1
echo "=== MAPS ==="
$PY m4_maps.py || exit 1
echo "=== CHAIN COMPLETE ==="
