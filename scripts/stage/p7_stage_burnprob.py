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

#: One bundle per state. Nevada alone leaves 13.6% of basins without P(F) --
#: HU10 units straddle the state line, and the rasters are clipped to it, so
#: the neighbours are needed for the borderland basins (CA 20,910 basins,
#: ID 11,071, UT 10,613, OR 6,980, AZ 6,107). Only the BP layer of each is
#: kept; the bundles carry eight.
BUNDLES = {
    "NV": ("https://usfs-public.box.com/shared/static/qdhjh8u6m91nytawj1r737dbghhnj1ft.zip",
           "f065e2f8bd667e82e3d6c31e2d06150a5c98c56c7363e937fb797e5dc207110f"),
    "CA": ("https://usfs-public.box.com/shared/static/et4mghz8sq0kxsk2ag6fpey5uec4f8fn.zip", None),
    "OR": ("https://usfs-public.box.com/shared/static/bqotzr1k53et252mf6k82gwz1904kbhl.zip", None),
    "ID": ("https://usfs-public.box.com/shared/static/jgmjss1rgjc7pnguc7480544ktvk5ild.zip", None),
    "UT": ("https://usfs-public.box.com/shared/static/vgd5pm9bggnyu5muj9l86vx74srah1s5.zip", None),
    "AZ": ("https://usfs-public.box.com/shared/static/er1j9w81buzpi6spu2ode6krj57frxts.zip", None),
}
DOI = "10.2737/RDS-2020-0016-2"
DEST = paths.raw_dir("wrc")
STAGE = paths.cache_root() / "downloads"
STAGE.mkdir(parents=True, exist_ok=True)
DEST.mkdir(parents=True, exist_ok=True)


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stage(state, url, expect):
    """Download one state bundle, keep only its BP raster, drop the zip."""
    out = DEST / f"BP_{state}.tif"
    if out.exists():
        print(f"{state}: already staged ({out.stat().st_size/1e6:.0f} MB)", flush=True)
        return out
    zpath = STAGE / f"RDS-2020-0016-2_{state}.zip"
    if not zpath.exists() or (expect and sha256(zpath) != expect):
        print(f"{state}: downloading {url}", flush=True)
        rc = subprocess.call(["curl", "-L", "-C", "-", "--retry", "5",
                              "--retry-delay", "10", "-sS", "-o", str(zpath), url])
        if rc != 0:
            print(f"{state}: curl failed ({rc}) -- rerun to resume", flush=True)
            return None
    if expect:
        got = sha256(zpath)
        if got != expect:
            print(f"{state}: CHECKSUM MISMATCH\n  expected {expect}\n  got {got}",
                  flush=True)
            return None
        print(f"{state}: checksum OK", flush=True)

    with zipfile.ZipFile(zpath) as z:
        members = [n for n in z.namelist()
                   if n.rsplit("/", 1)[-1].upper().startswith("BP_")
                   and n.lower().endswith(".tif")]
        if not members:
            print(f"{state}: no BP raster; members {z.namelist()[:8]}", flush=True)
            return None
        m = members[0]
        with z.open(m) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 22)
    paths.write_provenance(
        out, url=url, doi=DOI,
        note=("Wildfire Risk to Communities 2nd ed; annual burn probability from "
              "FSim at 270 m upsampled to 30 m, landscape conditions as of end of "
              f"2020. Extracted {m} from the {state} bundle; the other seven "
              "layers were discarded and the zip deleted."))
    zpath.unlink()
    print(f"{state}: staged {out.name} ({out.stat().st_size/1e6:.0f} MB)", flush=True)
    return out


want = [s.upper() for s in sys.argv[1:]] or list(BUNDLES)
missing = [s for s in want if s not in BUNDLES]
if missing:
    sys.exit(f"no bundle URL for {missing}; have {sorted(BUNDLES)}")

staged, failed = [], []
for state in want:
    url, expect = BUNDLES[state]
    got = stage(state, url, expect)
    (staged if got else failed).append(state)

print(f"\nstaged {staged}" + (f"; FAILED {failed} -- rerun" if failed else ""))
sys.exit(1 if failed else 0)
