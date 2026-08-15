"""Stage annual burn probability P(F) -- Wildfire Risk to Communities, 2nd ed.

P(R>T) answers "how often does the triggering storm arrive?" but is conditional
on the basin having burned. The pre-fire annual probability of a postfire
debris flow is P(F) x P(R>T), and P(F) is the annual burn probability from
FSim, distributed in the Wildfire Risk to Communities data release.

    Dillon, G.K., Scott, J.H., Jaffe, M.R., et al. (2023). Spatial datasets of
    wildfire risk for populated areas in the United States, 2nd Edition.
    Forest Service Research Data Archive. doi:10.2737/RDS-2020-0016-2

Notes that matter for how far to trust this layer:

* FSim runs at **270 m**; the distributed raster is that surface upsampled to
  LANDFIRE's 30 m. There is no real 30 m information in it, and burn
  probability was deliberately spread into non-burnable developed pixels.
* It represents landscape conditions **as of the end of 2020** -- fuels, not
  weather years since. It is a climatological rate, not a forecast.
* The ArcGIS image services of the same data are explicitly visualization-only
  ("pixel values have been altered"), so this stages the archive raster.

The state bundle is 8.6 GB and carries eight rasters; only BP is kept on Box.
The zip stages to the local cache and is deleted after extraction.
"""
import hashlib
import shutil
import subprocess
import sys
import zipfile

from firescape import paths

URL = "https://usfs-public.box.com/shared/static/qdhjh8u6m91nytawj1r737dbghhnj1ft.zip"
SHA256 = "f065e2f8bd667e82e3d6c31e2d06150a5c98c56c7363e937fb797e5dc207110f"
DOI = "10.2737/RDS-2020-0016-2"
DEST = paths.raw_dir("wrc")
STAGE = paths.cache_root() / "downloads"
STAGE.mkdir(parents=True, exist_ok=True)
ZIP = STAGE / "RDS-2020-0016-2_Nevada.zip"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


bp = sorted(DEST.glob("BP_*.tif"))
if bp:
    print(f"already staged: {bp[0]} ({bp[0].stat().st_size/1e6:.0f} MB)")
    sys.exit(0)

if not ZIP.exists() or sha256(ZIP) != SHA256:
    print(f"downloading {URL} -> {ZIP}", flush=True)
    # curl with resume: an 8.6 GB transfer should survive a dropped connection
    rc = subprocess.call(["curl", "-L", "-C", "-", "--retry", "5",
                          "--retry-delay", "10", "-o", str(ZIP), URL])
    if rc != 0:
        sys.exit(f"curl failed ({rc}) -- rerun to resume")

print("verifying checksum...", flush=True)
got = sha256(ZIP)
if got != SHA256:
    sys.exit(f"checksum mismatch\n  expected {SHA256}\n  got      {got}")
print("checksum OK", flush=True)

DEST.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    members = [n for n in z.namelist()
               if n.rsplit("/", 1)[-1].upper().startswith("BP_")
               and n.lower().endswith(".tif")]
    if not members:
        sys.exit(f"no BP raster in the bundle; members: {z.namelist()[:20]}")
    for m in members:
        out = DEST / m.rsplit("/", 1)[-1]
        print(f"extracting {m} -> {out}", flush=True)
        with z.open(m) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
        paths.write_provenance(
            out, url=URL, doi=DOI,
            note=("Wildfire Risk to Communities 2nd ed, Nevada bundle; annual "
                  "burn probability from FSim at 270 m upsampled to 30 m; "
                  "landscape conditions as of end of 2020. Only the BP layer "
                  "is retained -- the 8.6 GB source bundle held eight rasters "
                  f"and staged through {ZIP.name} (sha256 {SHA256})."))

ZIP.unlink()
print(f"staged {[p.name for p in sorted(DEST.glob('BP_*.tif'))]}; removed the zip")
