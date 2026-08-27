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


# --- shading the display grid ----------------------------------------------

def test_statewide_grid_keeps_the_module_shading_resolution():
    """A coarse domain is already shaded finer than it is drawn, so the rule
    must not disturb it — statewide figures re-render byte-identical."""
    from stormscape import relief

    tr, shape, _ = plotting.grid(
        (-120.13, 34.97, -113.94, 42.03), res=plotting.RES_DEG)
    assert plotting.grid_res_m(tr, shape) > relief.SHADE_RES_M
    assert plotting.shade_resolution(tr, shape) == relief.SHADE_RES_M


def test_zoom_grid_is_shaded_finer_than_its_own_pixels():
    """relief.py rule 3. A zoom panel at ~43 m/px used to be shaded at the
    statewide 50 m — coarser than it is drawn — and arrived pre-blurred."""
    from stormscape import relief

    tr, shape, _ = plotting.grid((-119.75, 39.2, -119.5, 39.45), res=0.0005)
    res_m = plotting.grid_res_m(tr, shape)
    shade = plotting.shade_resolution(tr, shape)

    assert res_m < relief.SHADE_RES_M           # the case that was broken
    assert shade < res_m
    assert shade == pytest.approx(res_m / 3.0, rel=1e-6)


def test_shading_never_asks_for_finer_than_the_source_dem():
    """3DEP is 1/3 arcsecond; asking below ~10 m buys time, not detail."""
    tr, shape, _ = plotting.grid((-119.72, 39.50, -119.70, 39.52), res=0.00002)
    assert plotting.shade_resolution(tr, shape) == 10.0


def test_hillshade_cache_key_separates_two_windows_of_one_shape():
    """The old key was resolution plus shading only, so a second window with
    the same shape silently rendered the first window's terrain."""
    a_tr, a_shape, _ = plotting.grid((-119.75, 39.2, -119.5, 39.45), res=0.001)
    b_tr, b_shape, _ = plotting.grid((-117.75, 41.2, -117.5, 41.45), res=0.001)

    assert a_shape == b_shape and a_tr.a == b_tr.a
    assert (plotting._hillshade_cache(a_tr, a_shape, 30.0)
            != plotting._hillshade_cache(b_tr, b_shape, 30.0))


# --- panel windows ---------------------------------------------------------

def test_square_window_is_square_on_screen_not_in_degrees():
    w, s, e, n = plotting.square_window((-119.7, 39.3, -119.6, 39.35))
    cos_lat = math.cos(math.radians((s + n) / 2))

    assert (n - s) == pytest.approx((e - w) * cos_lat, rel=1e-9)
    assert (n - s) < (e - w)                      # taller per degree up here
    # the screen-space edges match, which is what removes the letterbox
    assert (e - w) == pytest.approx((n - s) / cos_lat, rel=1e-9)


def test_square_window_covers_the_requested_bounds_with_pad():
    req = (-119.7, 39.3, -119.6, 39.35)
    w, s, e, n = plotting.square_window(req, pad=0.02)
    tol = 1e-9
    assert w <= req[0] - 0.02 + tol and e >= req[2] + 0.02 - tol
    assert s <= req[1] - 0.02 + tol and n >= req[3] + 0.02 - tol


def test_tick_step_keeps_a_readable_number_of_ticks():
    for span in (0.05, 0.09, 0.1, 0.3, 1.2, 4.0, 7.5):
        assert span / plotting.tick_step(span) <= 6


def test_tick_step_still_labels_a_district_window():
    """A ~10 km window spans <0.1 deg; a 0.05 floor left it with one tick."""
    assert 2 <= 0.09 / plotting.tick_step(0.09) <= 6


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


# --- bottom notes ----------------------------------------------------------

def test_wrap_text_respects_the_measured_width():
    """A caption bounded to the panels must not exceed them; the wrap is
    measured, because a character count is wrong by 2x between 'MMM' and
    'iii'."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
    fig.canvas.draw()
    long = ("A site counts as near-channel when it lies within 30-55 m of a "
            "modelled channel, half a corridor 9-60 m wide that widens with "
            "contributing area, plus 25 m for registration. ") * 2
    wrapped = plotting.wrap_text(fig, long, 4.0, fontsize=6.5)

    assert "\n" in wrapped
    for line in wrapped.split("\n"):
        probe = fig.text(0, 0, line, fontsize=6.5)
        fig.canvas.draw()
        w = (probe.get_window_extent()
             .transformed(fig.dpi_scale_trans.inverted()).width)
        probe.remove()
        assert w <= 4.0 * 1.05        # a whole word may overhang slightly
    plt.close(fig)


def test_panel_span_ignores_the_colourbar():
    """Colourbars are axes too; a note bounded to 'the figure' must not be
    stretched by the thin bar beside the map."""
    fig = plt.figure(figsize=(8, 5), dpi=100)
    fig.add_axes([0.10, 0.20, 0.60, 0.70])          # the panel
    fig.add_axes([0.86, 0.20, 0.02, 0.70])          # a colourbar
    x0, x1 = plotting.panel_span(fig)

    assert x0 == pytest.approx(0.8, abs=1e-6)       # 0.10 * 8 in
    assert x1 == pytest.approx(5.6, abs=1e-6)       # 0.70 * 8 in
    plt.close(fig)


def test_footnote_sits_inside_the_panel_span():
    fig = plt.figure(figsize=(8, 5), dpi=100)
    fig.add_axes([0.10, 0.20, 0.60, 0.70])
    t = plotting.footnote(fig, "a caption " * 40)
    fig.canvas.draw()
    bb = (t.get_window_extent()
          .transformed(fig.dpi_scale_trans.inverted()))

    assert bb.x0 >= 0.8 - 0.2 and bb.x1 <= 5.6 + 0.2
    assert "\n" in t.get_text()
    plt.close(fig)


# --- key placement ---------------------------------------------------------

def test_clear_corner_keeps_the_default_when_no_corner_is_covered():
    # A blob in the middle blocks nothing: the key stays where the sheets
    # have always put it rather than drifting to a marginally emptier corner.
    extent = (-120.0, -119.0, 39.0, 40.0)
    per = gpd.GeoSeries([box(-119.6, 39.4, -119.4, 39.6)])

    key = plotting.clear_corner(extent, per, size=(0.30, 0.20))

    assert key["loc"] == "lower left"
    assert key["xy"] == (0.015, 0.015)
    assert max(key["overlap"].values()) == 0.0


def test_clear_corner_moves_off_a_perimeter_in_the_default_corner():
    extent = (-120.0, -119.0, 39.0, 40.0)
    per = gpd.GeoSeries([box(-120.0, 39.0, -119.6, 39.4)])   # lower-left lobe

    key = plotting.clear_corner(extent, per, size=(0.30, 0.20))

    assert key["loc"] != "lower left"
    assert key["overlap"]["lower left"] == pytest.approx(1.0)
    assert key["overlap"][key["loc"]] == 0.0
    assert key["xy"][1] > 0.5 or key["xy"][0] > 0.5          # not lower-left


def test_clear_corner_ignores_a_clip_smaller_than_the_tolerance():
    # The perimeter just clips the key's own corner -> not worth moving for.
    extent = (-120.0, -119.0, 39.0, 40.0)
    per = gpd.GeoSeries([box(-120.0, 39.0, -119.98, 39.02)])

    key = plotting.clear_corner(extent, per, size=(0.30, 0.20))

    assert 0.0 < key["overlap"]["lower left"] < 0.01
    assert key["loc"] == "lower left"


def test_clear_corner_gives_a_line_network_width_to_overlap_with():
    # Arealess geometry would intersect every corner at zero and always pick
    # the default; the buffer is what makes a stream network usable here.
    extent = (-120.0, -119.0, 39.0, 40.0)
    net = gpd.GeoSeries([LineString([(-119.99, 39.01), (-119.9, 39.1)])])

    key = plotting.clear_corner(extent, net, size=(0.30, 0.20))

    assert key["overlap"]["lower left"] > 0.0
    assert key["loc"] != "lower left"


def test_corner_xy_places_a_box_inside_every_corner():
    for loc in plotting.KEY_CORNERS:
        x, y = plotting.corner_xy(loc, (0.30, 0.20), pad=0.02)
        assert 0.0 <= x <= 1.0 - 0.30 and 0.0 <= y <= 1.0 - 0.20
        assert (x == pytest.approx(0.02)) == loc.endswith("left")
        assert (y == pytest.approx(0.02)) == loc.startswith("lower")


def test_caption_reserves_space_below_the_axes():
    # subplots, not add_axes: the reserve works through subplots_adjust, which
    # an absolutely-positioned axes ignores.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.plot([0, 1], [0, 1])
    art = plotting.caption(fig, "a caption " * 40, label="Figure 1.")
    fig.canvas.draw()
    bb = art.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    assert fig.subplotpars.bottom > 0.11               # axes moved up
    assert bb.y1 < ax.get_tightbbox().transformed(
        fig.dpi_scale_trans.inverted()).y0            # and cleared the labels
    assert art.get_text().startswith(r"$\mathbf{Figure\;1.}$")
    plt.close(fig)
