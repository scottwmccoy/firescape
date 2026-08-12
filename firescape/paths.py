"""Path policy: where data, caches, and products live.

All heavy data lives on Box under ``$FIRESCAPE_DATA`` (default: the
PreFireAssessment project folder). Caches and download staging are LOCAL
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

_BOX_DEFAULT = Path(
    "/Users/scottmccoy/Library/CloudStorage/Box-Box/SWMresearch"
    "/PostFireDebrisFlows/PreFireAssessment"
)

_SHA256_MAX_BYTES = 200 * 1024 * 1024  # skip hashing very large files


def data_root() -> Path:
    return Path(os.environ.get("FIRESCAPE_DATA", _BOX_DEFAULT))


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
