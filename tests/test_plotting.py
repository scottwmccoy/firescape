"""Map house style: the choices that were made against printed output.

Offline -- the network-backed half of plotting.py (fetch_context) is exercised
only through its caching contract, with pre-placed GeoJSON standing in for a
download.
"""

import math

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from firescape import paths, plotting  # noqa: E402

NV_EXTENT = (-120.2, -113.9, 34.9, 42.1)      # (w, e, s, n)


@pytest.fixture()
def ax():
    fig, a = plt.subplots()
    yield a
    plt.close(fig)


# --- the display grid ------------------------------------------------------

def test_grid_snaps_and_covers_requested_bounds():
    bounds = (-120.13, 34.97, -113.94, 42.03)   # (w, s, e, n)
    tr, (rows, cols), (w, e, s, n) = plotting.grid(bounds, res=0.01)

    assert w <= bounds[0] and s <= bounds[1]     # never clips the domain
    assert e >= bounds[2] and n >= bounds[3]
    assert math.isclose(w / 0.01, round(w / 0.01), abs_tol=1e-6)   # snapped
    assert math.isclose(tr.a, 0.01) and math.isclose(tr.e, -0.01)
    assert (tr.c, tr.f) == (w, n)                # origin is north-west
    assert math.isclose(w + cols * tr.a, e)
    assert math.isclose(n + rows * tr.e, s)


def test_default_display_grid_is_finer_than_a_print_pixel():
    """~150 m at 0.0015 deg, against ~300 m per pixel for a statewide panel at
    300 dpi. Coarser than this and the hillshade prints mushy."""
    metres = plotting.RES_DEG * 111_320 * math.cos(math.radians(39))
    assert 100 < metres < 200


# --- axes ------------------------------------------------------------------

def test_style_axes_labels_and_aspect(ax):
    plotting.style_axes(ax, NV_EXTENT)

    assert ax.get_xlim() == (NV_EXTENT[0], NV_EXTENT[1])
    assert ax.get_ylim() == (NV_EXTENT[2], NV_EXTENT[3])
    # 1/cos(lat) makes Nevada's meridian borders vertical and shapes locally true
    mid = math.radians((NV_EXTENT[2] + NV_EXTENT[3]) / 2)
    assert math.isclose(ax.get_aspect(), 1.0 / math.cos(mid), rel_tol=1e-9)

    assert [t.get_text() for t in ax.get_xticklabels()][:2] == ["120°W", "119°W"]
    assert [t.get_text() for t in ax.get_yticklabels()][:2] == ["35°N", "36°N"]


def test_style_axes_draws_no_graticule(ax):
    """White lines over a hillshade read as terrain edges, not as a grid."""
    plotting.style_axes(ax, NV_EXTENT)
    assert not any(line.get_visible()
                   for line in ax.get_xgridlines() + ax.get_ygridlines())


def test_style_axes_honours_a_finer_tick_step(ax):
    """Hindcast-scale extents need sub-degree ticks."""
    plotting.style_axes(ax, (-120.0, -119.5, 39.2, 39.6), step=0.25)
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "120°W", "119.75°W", "119.5°W"]


# --- rasterizing a data layer ---------------------------------------------

def test_burn_rasterizes_values_and_leaves_nan_elsewhere():
    tr, shape, _ = plotting.grid((-120.0, 39.0, -119.0, 40.0), res=0.1)
    gdf = gpd.GeoDataFrame(
        {"v": [3.0, np.nan]},
        geometry=[box(-120.0, 39.5, -119.5, 40.0), box(-119.5, 39.0, -119.0, 39.5)],
        crs="EPSG:4326")

    arr = plotting.burn(gdf, "v", tr, shape)

    assert arr.shape == tuple(shape)
    assert np.nanmax(arr) == 3.0
    assert np.isnan(arr[-1, -1])          # the NaN-valued feature is skipped
    assert np.isnan(arr).any() and np.isfinite(arr).any()


# --- perimeter styling -----------------------------------------------------

def test_fire_styles_are_distinct_and_cased():
    cal = plotting.fire_style("calibration")
    cur = plotting.fire_style("current")
    assert cal["color"] == "white"            # survives magma/viridis beneath
    assert cal["color"] != cur["color"]
    assert cal["path_effects"] and cur["path_effects"]


def test_unknown_fire_style_names_the_alternatives():
    with pytest.raises(KeyError, match="calibration"):
        plotting.fire_style("pink")


# --- context ---------------------------------------------------------------

def test_fetch_context_reads_caches_without_network(tmp_path):
    """Every layer cached -> no download, no stormscape import."""
    cache = tmp_path / "ctx"
    cache.mkdir()
    for name, geom in (
        ("counties", box(-120, 39, -119, 40)),
        ("rivers", LineString([(-120, 39), (-119, 40)])),
        ("lakes", box(-119.6, 39.4, -119.5, 39.5)),
        ("roads", LineString([(-120, 39.5), (-119, 39.5)])),
        ("places", Point(-119.8, 39.5)),
    ):
        gpd.GeoDataFrame({"name": ["x"], "kind": ["primary"]},
                         geometry=[geom], crs="EPSG:4326").to_file(
            cache / f"context_{name}.geojson", driver="GeoJSON")

    ctx = plotting.fetch_context((-120.0, 39.0, -119.0, 40.0), cache_dir=cache)

    assert set(ctx) == {"counties", "rivers", "lakes", "roads", "places"}
    assert all(len(g) == 1 for g in ctx.values())


def test_draw_context_plots_every_layer(ax, tmp_path):
    cache = tmp_path / "ctx"
    cache.mkdir()
    for name, geom in (
        ("counties", box(-120, 39, -119, 40)),
        ("rivers", LineString([(-120, 39), (-119, 40)])),
        ("lakes", box(-119.6, 39.4, -119.5, 39.5)),
        ("roads", LineString([(-120, 39.5), (-119, 39.5)])),
        ("places", Point(-119.8, 39.5)),
    ):
        gpd.GeoDataFrame({"name": ["Reno"], "kind": ["primary"]},
                         geometry=[geom], crs="EPSG:4326").to_file(
            cache / f"context_{name}.geojson", driver="GeoJSON")
    ctx = plotting.fetch_context((-120.0, 39.0, -119.0, 40.0), cache_dir=cache)

    plotting.draw_context(ax, ctx)

    assert ax.collections and ax.texts       # geometry drawn, labels written
    assert any("Reno" == t.get_text() for t in ax.texts)


def test_draw_context_tolerates_empty_layers(tmp_path, monkeypatch):
    """An arid window has no lake and no named perennial stream.

    Plotting an empty GeoDataFrame raises "aspect must be finite and positive"
    rather than drawing nothing, so an empty layer used to take the whole
    figure down -- which is the normal case for the Elko and NNSS windows,
    not an exceptional one.
    """
    monkeypatch.setattr(paths, "raw_dir", lambda *a: tmp_path / "raw")
    cache = tmp_path / "ctx"
    cache.mkdir()
    fig, ax = plt.subplots()

    empty = gpd.GeoDataFrame({"name": [], "kind": []},
                             geometry=[], crs="EPSG:4326")
    for name in ("counties", "rivers", "lakes", "roads", "places"):
        empty.to_file(cache / f"context_{name}.geojson", driver="GeoJSON")
    ctx = plotting.fetch_context((-116.5, 36.5, -115.5, 37.5), cache_dir=cache)

    plotting.draw_context(ax, ctx)           # must not raise
    plt.close(fig)


# --- output ----------------------------------------------------------------

def test_save_writes_pdf_only_by_default(tmp_path, monkeypatch):
    fig, a = plt.subplots()
    a.plot([0, 1], [0, 1])
    written = plotting.save(fig, "unit_test_fig")
    plt.close(fig)

    assert [p.suffix for p in written] == [".pdf"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    assert all(p.parent == paths.figures_dir() for p in written)


def test_save_still_writes_png_on_request(tmp_path, monkeypatch):
    """A raster is opt-in, not gone -- some downstream uses need one."""
    fig, a = plt.subplots()
    a.plot([0, 1], [0, 1])
    written = plotting.save(fig, "unit_test_fig_both", formats=("png", "pdf"))
    plt.close(fig)

    assert [p.suffix for p in written] == [".png", ".pdf"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_layer_alpha_ceiling_is_documented():
    """Terrain must read through every data layer."""
    assert plotting.MAX_LAYER_ALPHA <= 0.6
