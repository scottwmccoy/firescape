"""Imagery-source adapters: everything sensor-specific in one place.

The verification pipeline is sensor-agnostic from "reflectance-ish bands on
a common grid" onward -- epoch statistics (:mod:`epochs`), change math
(:mod:`change`), corridor/fan conditioning (:mod:`corridor`, :mod:`fans`)
never ask what satellite they are looking at. What differs per source is
exactly: how scenes are discovered on disk, band order and which four map
to blue/green/red/nir, DN -> physical scaling, the validity mask, and where
sun geometry lives. One adapter class per source owns those five things;
new sensors (Pleiades, SkySat...) are a new ~60-line adapter, never a new
pipeline.

Planet scenes keep their existing path (:mod:`epochs` reads the staged
SR+udm2 pairs directly). This module adds Maxar/NextView strip deliveries
(WorldView-1/2/3, GeoEye-1) as found in the Dolan archive: per-strip
``*-M2AS-*`` orthorectified Standard products with IMD metadata.

Maxar scaling note: DNs are converted to **top-of-atmosphere spectral
radiance** with the IMD's per-band absCalFactor/effectiveBandwidth. We stop
at radiance (no Esun/earth-sun-distance reflectance step): the change
pipeline normalizes per epoch and regresses illumination on the DEM anyway,
and skipping the constants table removes a transcription risk. Radiance is
consistent within a sensor -- do not mix sensors within an epoch.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: WV-2/WV-3 8-band order -> the pipeline's 4-band contract.
MAXAR_8B = {"blue": 2, "green": 3, "red": 5, "nir": 7}
#: GeoEye-1 / 4-band products (B, G, R, N).
MAXAR_4B = {"blue": 1, "green": 2, "red": 3, "nir": 4}


def _imd_val(txt: str, key: str):
    m = re.search(rf"{key}\s*=\s*([^;]+);", txt)
    return m.group(1).strip().strip('"') if m else None


def parse_imd(imd_path: Path) -> dict:
    """The IMD fields the pipeline needs: sun, bands, per-band calibration."""
    txt = Path(imd_path).read_text(errors="replace")
    bands = []
    for m in re.finditer(r"BEGIN_GROUP = (BAND_\w+)(.*?)END_GROUP", txt,
                         re.S):
        grp = m.group(2)
        bands.append({
            "name": m.group(1),
            "abscal": float(_imd_val(grp, "absCalFactor") or "nan"),
            "bandwidth": float(_imd_val(grp, "effectiveBandwidth") or "nan"),
        })
    return {
        "sat": _imd_val(txt, "satId"),
        "sun_el": float(_imd_val(txt, "meanSunEl") or "nan"),
        "sun_az": float(_imd_val(txt, "meanSunAz") or "nan"),
        "cloud": float(_imd_val(txt, "cloudCover") or "nan"),
        "gsd": float(_imd_val(txt, "meanProductGSD")
                     or _imd_val(txt, "productGSD") or "nan"),
        "bands": bands,
    }


class MaxarStrips:
    """One epoch = the ``*-M2AS-*`` strips under a date directory.

    ``read_into(bands_out, count)`` mosaics every strip into the supplied
    band arrays (first-valid-wins), so an epoch composite is built without
    stacking: strips of one date are a spatial mosaic, not a temporal
    stack. Deliberate consequence -- Maxar epochs have no temporal MAD;
    the z scale must come from a spatial robust estimate downstream.
    """

    def __init__(self, date_dir: Path, *, product: str = "M2AS"):
        self.dir = Path(date_dir)
        self.tifs = sorted(self.dir.glob(f"*/*-{product}-*/*.TIF"))
        if not self.tifs:
            raise FileNotFoundError(f"no {product} strips under {date_dir}")

    def meta(self) -> list[dict]:
        return [parse_imd(t.with_suffix(".IMD")) for t in self.tifs]

    def sun(self) -> tuple[float, float]:
        m = self.meta()
        return (float(np.mean([x["sun_az"] for x in m])),
                float(np.mean([x["sun_el"] for x in m])))

    def read_into(self, ref: dict, band_map: dict | None = None):
        """Mosaic all strips onto ``ref``; returns dict of masked radiance.

        Each strip is warped once (bilinear) into a per-band accumulator;
        pixels already filled by an earlier strip are kept (strips of one
        pass are radiometrically consistent, so seam choice is cosmetic).
        """
        import rasterio
        from rasterio.warp import Resampling, reproject

        shape = (ref["height"], ref["width"])
        acc = None
        for tif in self.tifs:
            imd = parse_imd(tif.with_suffix(".IMD"))
            bm = band_map or (MAXAR_8B if len(imd["bands"]) == 8 else MAXAR_4B)
            if acc is None:
                acc = {k: np.full(shape, np.nan, np.float32) for k in bm}
            with rasterio.open(tif) as src:
                for name, idx in bm.items():
                    cal = imd["bands"][idx - 1]
                    scale = cal["abscal"] / cal["bandwidth"]
                    dst = np.full(shape, np.nan, np.float32)
                    reproject(rasterio.band(src, idx), dst,
                              dst_transform=ref["transform"],
                              dst_crs=ref["crs"],
                              resampling=Resampling.bilinear,
                              src_nodata=0, dst_nodata=np.nan)
                    a = acc[name]
                    fill = np.isnan(a) & np.isfinite(dst)
                    a[fill] = dst[fill] * scale
        return {k: np.ma.masked_invalid(v) for k, v in acc.items()}
