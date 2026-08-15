#!/bin/zsh
# Run one staging lane to completion (resumable driver, exit 42 = keep going).
set -u
PY=/opt/anaconda3/envs/FireMan/bin/python
SP="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$1"
export FIRESCAPE_STAGE_BUDGET=520
export FIRESCAPE_LFPS_EMAIL=scottmccoy@unr.edu
for attempt in $(seq 1 40); do
  $PY "$SP/$SCRIPT"
  rc=$?
  if [ $rc -ne 42 ]; then
    echo "=== $SCRIPT finished (rc=$rc) after $attempt passes ==="
    exit $rc
  fi
done
echo "=== $SCRIPT hit pass limit ==="
exit 1
