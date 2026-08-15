"""Stage 3DEP 1/3-arcsecond tiles from the USGS S3 bucket (resumable)."""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import requests

from firescape import paths

T0 = time.time()
BUDGET = float(os.environ.get("FIRESCAPE_STAGE_BUDGET", 480))
TILE_DIR = paths.cache_root() / "3dep_tiles"     # local: public re-fetchable tiles
TILE_DIR.mkdir(parents=True, exist_ok=True)

tiles = json.loads((paths.interim_dir("statewide") / "dem_tiles.json").read_text())
done, pending, failed = [], [], []
for t in tiles:
    name = f"USGS_13_n{t['n']}w{t['w']:03d}.tif"
    dest = TILE_DIR / name
    if dest.exists() and dest.stat().st_size > 1e6:
        done.append(name)
        continue
    if time.time() - T0 > BUDGET:
        pending.append(name)
        continue
    t1 = time.time()
    try:
        paths.download(t["url"], dest, timeout=900)
        print(f"{name}: {dest.stat().st_size/1e6:.0f} MB in {time.time()-t1:.0f}s", flush=True)
        done.append(name)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        print(f"{name}: HTTP {code} (tile may not exist - ocean/UT gap?)", flush=True)
        failed.append(name)
    except Exception as e:
        print(f"{name}: {type(e).__name__}", flush=True)
        failed.append(name)

manifest = paths.raw_dir("statewide") / "dem_tiles_manifest.json"
manifest.write_text(json.dumps({"tile_dir": str(TILE_DIR), "done": done,
                                "failed": failed, "pending": pending}, indent=1))
print(f"\nDEM tiles: {len(done)} done, {len(pending)} pending, {len(failed)} failed, "
      f"{time.time()-T0:.0f}s", flush=True)
sys.exit(42 if pending else 0)
