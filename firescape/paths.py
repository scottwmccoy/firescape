"""Path policy: where data, caches, and products live.

All heavy data lives on Box under ``$FIRESCAPE_DATA`` (default: the
PreFireAssessment project folder at Box Drive's standard sync location under
the current user's home). Sibling project folders come from
``research_root()``; files shipped in the package from ``package_data()``. Caches and download staging are LOCAL
(``$FIRESCAPE_CACHE``, default ``~/.cache/firescape``) because Box sync
performs poorly with sqlite files and many small writes. Downloads stream to
local staging and are moved into Box atomically so partially-written files
never appear under ``raw/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ScienceBase (and some other USGS hosts) return 403 to non-browser agents.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# The shared Box folder, at the location Box Drive syncs it to on every
# member's Mac. Override with $FIRESCAPE_DATA (a different sync location, a
# Linux box, CI).
_BOX_DEFAULT = (Path.home() / "Library" / "CloudStorage" / "Box-Box" / "SWMresearch"
                / "PostFireDebrisFlows" / "PreFireAssessment")

_SHA256_MAX_BYTES = 200 * 1024 * 1024  # skip hashing very large files


def data_root() -> Path:
    """The project data folder (``raw/``, ``interim/``, ``products/``, ``figures/``).

    ``$FIRESCAPE_DATA`` if set, else the Box folder at its standard sync
    location. When neither exists nothing is created: a stray directory tree
    under ``~/Library/CloudStorage`` on a machine without Box is worse than a
    clear error.
    """
    env = os.environ.get("FIRESCAPE_DATA")
    if env:
        return Path(env)
    if not _BOX_DEFAULT.is_dir():
        raise FileNotFoundError(
            f"firescape data folder not found at {_BOX_DEFAULT}. Sync the "
            "PreFireAssessment Box folder, or set $FIRESCAPE_DATA to where it lives.")
    return _BOX_DEFAULT


def research_root() -> Path:
    """The folder ABOVE the project data folder: ``SWMresearch/PostFireDebrisFlows``,
    which holds the sibling per-fire and per-storm project folders some
    scripts read (``2026_Bug_Stalion/storms/...``, ``2024_BearFire``,
    ``Volume_debrisFlows/...``). ``$FIRESCAPE_RESEARCH`` overrides; the
    default is the parent of :func:`data_root`.
    """
    env = os.environ.get("FIRESCAPE_RESEARCH")
    return Path(env) if env else data_root().parent


def package_data(*parts: str) -> Path:
    """A file shipped inside the package: ``firescape/data/<parts...>``
    (calibration TOMLs and fire sets, region GeoJSONs, the Staley 2018
    tables). Resolved from the installed package, never from a checkout path.
    """
    return Path(__file__).resolve().parent.joinpath("data", *parts)


def python_executable() -> str:
    """Interpreter for subprocess fan-out: ``$FIRESCAPE_PYTHON`` if set, else
    the one running this code."""
    import sys

    return os.environ.get("FIRESCAPE_PYTHON") or sys.executable


def cache_root() -> Path:
    root = Path(os.environ.get("FIRESCAPE_CACHE", Path.home() / ".cache" / "firescape"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sub(root: Path, *parts: str) -> Path:
    p = root.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def raw_dir(*parts: str) -> Path:
    return _sub(data_root() / "raw", *parts)


def interim_dir(*parts: str) -> Path:
    return _sub(data_root() / "interim", *parts)


def products_dir(*parts: str) -> Path:
    return _sub(data_root() / "products", *parts)


def figures_dir(*parts: str) -> Path:
    return _sub(data_root() / "figures", *parts)


def atomic_into(src: Path, dest: Path) -> Path:
    """Move a fully-written local file into place (Box-safe)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest


def download(url: str, dest: Path, *, timeout: float = 300.0, headers: dict | None = None) -> Path:
    """Stream ``url`` to local staging, then move atomically to ``dest``."""
    import requests

    dest = Path(dest)
    hdrs = {"User-Agent": BROWSER_UA}
    if headers:
        hdrs.update(headers)
    with tempfile.NamedTemporaryFile(dir=cache_root(), delete=False, suffix=".part") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=hdrs) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1 << 20):
                    tmp.write(chunk)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return atomic_into(tmp_path, dest)


def sha256(path: Path) -> str | None:
    path = Path(path)
    if path.stat().st_size > _SHA256_MAX_BYTES:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(dest: Path, *, url: str | None = None, doi: str | None = None,
                     note: str | None = None, **extra) -> Path:
    """Write ``<dest>.provenance.json`` beside a raw-data file."""
    dest = Path(dest)
    meta = {
        "file": dest.name,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "size_bytes": dest.stat().st_size if dest.exists() else None,
        "sha256": sha256(dest) if dest.exists() else None,
    }
    if url:
        meta["url"] = url
    if doi:
        meta["doi"] = doi
    if note:
        meta["note"] = note
    meta.update(extra)
    out = dest.with_name(dest.name + ".provenance.json")
    out.write_text(json.dumps(meta, indent=2) + "\n")
    return out
