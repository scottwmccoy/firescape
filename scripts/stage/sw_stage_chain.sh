#!/bin/zsh
# Run one staging lane to completion (resumable driver, exit 42 = keep going).
set -u
PY=/opt/anaconda3/envs/FireMan/bin/python
SP="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$1"
export FIRESCAPE_STAGE_BUDGET=520
# Delivery address for LFPS/MTBS notices: set FIRESCAPE_EMAIL in your own
# environment before running; there is no default.
if [ -z "${FIRESCAPE_EMAIL:-}" ] && [ -z "${FIRESCAPE_LFPS_EMAIL:-}" ]; then
  echo "set FIRESCAPE_EMAIL (or FIRESCAPE_LFPS_EMAIL) to your address first" >&2
  exit 2
fi
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
