"""Place the statewide MTBS bundle order (fires missing from raw/mtbs/fires)."""
import json
from datetime import datetime, timezone

import pandas as pd

from firescape import landfire, mtbs, paths

EMAIL = landfire.delivery_email()      # $FIRESCAPE_EMAIL; no default on purpose
SET = (paths.package_data("calibration", "fire_sets", "statewide_v1.csv"))

df = pd.read_csv(SET)
need = sorted(df.loc[df["include"] & ~df["have_bundle"], "event_id"])
print(f"ordering {len(need)} bundles -> {EMAIL}")

resp = mtbs.order_bundles(need, EMAIL)
print("server response:", resp)

rec = {
    "ordered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "email": EMAIL,
    "n_fires": len(need),
    "event_ids": need,
    "products": list(mtbs.BUNDLE_PRODUCTS_DEFAULT),
    "projection": "Albers",
    "response": resp,
    "fire_set": "statewide_v1.csv",
}
out = paths.raw_dir("mtbs") / "bundle_order_statewide_20260813.json"
out.write_text(json.dumps(rec, indent=1))
print("order record ->", out)
