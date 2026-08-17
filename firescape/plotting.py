"""Map house style for firescape figures.

Every statewide figure is the same object: a hillshade backdrop, one data
layer draped over it, reference context, and geographic axes. The choices
below were made against printed output and are deliberate:

* **Geographic display grid, not the working CRS.** Nevada's north-south
  borders are meridians; drawn in EPSG:5070 they lean, which reads as a
  projection error to anyone who knows the state. Figures are built on a plain
  EPSG:4326 grid with degree-labelled ticks and a ``1/cos(lat)`` aspect, so the
  borders stand vertical. Shading still happens on a metric grid inside
  :mod:`firescape.relief`, so illumination stays geometrically true.
* **Data layers at alpha <= 0.6.** Terrain context is how a reader judges
  whether a hazard pattern is plausible -- range fronts, canyon mouths, fan
  apexes. Earlier figures used 0.8 and buried the relief.
* **No graticule.** Thin white lines over a hillshade read as terrain edges,
  not as a grid. The degree ticks carry the georeference.
* **Colourblind-safe ramps only** (Okabe-Ito categorical, viridis/magma
  continuous). Fire perimeters use white with a black casing, which survives
  every ramp underneath -- the earlier pink clashed with magma.
* **PDF only.** These figures are mostly one big rasterized data layer plus
  vector furniture, and a PDF stores that layer once at its own resolution
  while keeping linework, labels and boundaries as vectors. The PNG of the
  same figure was consistently two to three times the size for strictly less
  -- 25.9 MB against 6.7 MB for the Las Vegas corridor -- so it is no longer
  written by default. ``save(..., formats=("png", "pdf"))`` still gets both
  when something downstream needs a raster.
"""

from __future__ import annotations

import math

import numpy as np

from firescape import paths

#: Display grid spacing in degrees (~150 m). Finer than a 300 dpi print pixel
#: at statewide extent (~300 m), so the figure stays crisp when zoomed.
RES_DEG = 0.0015

#: Padding around the domain, degrees.
PAD_DEG = 0.05

NATURAL_EARTH_S3 = "https://naturalearth.s3.amazonaws.com/10m_physical"
COUNTY_URL = ("https://www2.census.gov/geo/tiger/GENZ2023/shp/"
              "cb_2023_us_county_500k.zip")

#: Places labelled on statewide maps. A whitelist, because GNIS returns far too
#: many points to label legibly at this scale.
MAJOR_CITIES = frozenset({
    "Reno", "Sparks", "Carson City", "Las Vegas", "Henderson", "Elko", "Ely",
    "Winnemucca", "Tonopah", "Fallon", "Pahrump", "Hawthorne", "Eureka",
    "Battle Mountain", "Wells", "Caliente", "Mesquite", "West Wendover",
    "Pioche", "Austin", "Gerlach",
})

#: Maximum alpha for a data layer drawn over the hillshade.
MAX_LAYER_ALPHA = 0.6


def _fire_styles():
    """Perimeter styles. Built lazily so importing this module stays cheap."""
    import matplotlib.patheffects as pe

    casing = lambda w: [pe.withStroke(linewidth=w, foreground="black")]  # noqa: E731
    return {
        # calibration fires: white survives magma/viridis underneath
        "calibration": dict(color="white", linewidth=0.9, path_effects=casing(1.7)),
        # actively burning fires: Okabe-Ito sky blue
        "current": dict(color="#56B4E9", linewidth=1.4, path_effects=casing(2.2)),
        # historic perimeters: recessive, they are context not subject
        "historic": dict(color="#333333", linewidth=0.4, alpha=0.85),
    }


def fire_style(kind: str) -> dict:
    """Keyword arguments for plotting a fire perimeter boundary.

    ``kind`` is one of ``calibration``, ``current`` or ``historic``.
    """
    styles = _fire_styles()
    try:
        return styles[kind]
    except KeyError:
        raise KeyError(f"unknown fire style {kind!r}; "
                       f"have {sorted(styles)}") from None


def domain_bounds4326(hu_path=None):
    """Padded lon/lat bounds of the statewide modelling domain."""
    import geopandas as gpd

    hu_path = hu_path or (paths.interim_dir("statewide") / "nv_hu10.geojson")
    w, s, e, n = gpd.read_file(hu_path).total_bounds
    return (w - PAD_DEG, s - PAD_DEG, e + PAD_DEG, n + PAD_DEG)


def grid(bounds4326, res: float = RES_DEG):
    """Display grid snapped to ``res``.

    Returns ``(transform, (rows, cols), (west, east, south, north))`` -- note
    the extent tuple is in matplotlib's ``imshow`` order, not rasterio's.
    """
    w, s, e, n = bounds4326
    w, s = math.floor(w / res) * res, math.floor(s / res) * res
    e, n = math.ceil(e / res) * res, math.ceil(n / res) * res
    width, height = int(round((e - w) / res)), int(round((n - s) / res))
    from rasterio.transform import from_origin
    return from_origin(w, n, res, res), (height, width), (w, e, s, n)


def grid_res_m(transform, shape) -> float:
    """Ground size of one display pixel, in metres, for an EPSG:4326 grid."""
    lat = transform.f + transform.e * shape[0] / 2.0
    return abs(transform.a) * 111_320 * math.cos(math.radians(lat))


def shade_resolution(transform, shape) -> float:
    """Metric resolution to shade a display grid at.

    :mod:`firescape.relief` rule 3 is *shade finer than the display grid, then
    average down* — landforms smaller than a display pixel are what give a
    hillshade its texture. :data:`relief.SHADE_RES_M` is a single statewide
    constant, so a zoomed panel used to be shaded **coarser** than its own
    pixels (50 m under a 15 m grid) and arrived pre-blurred; the district and
    urban-zoom sheets each worked around it by calling
    :func:`relief.shaded_relief` directly with a hand-picked value.

    The module constant is kept wherever it already puts two shaded cells in a
    display pixel — which is every statewide figure, so those re-render
    unchanged — and only a grid it cannot satisfy is refined, to a third of the
    display pixel. The floor is the 10 m 3DEP grid the tiles are sampled at;
    asking for finer buys time, not detail.
    """
    from firescape import relief

    res_m = grid_res_m(transform, shape)
    if relief.SHADE_RES_M <= res_m / 2.0:
        return relief.SHADE_RES_M
    return max(10.0, res_m / 3.0)


def _hillshade_cache(transform, shape, shade_res_m):
    """Cache path for one shaded grid.

    Keyed on the grid **origin** as well as its resolution: two windows can
    easily share a resolution and a shape, and the older resolution-only key
    let the second one silently render the first one's terrain.
    """
    import hashlib

    stamp = (f"{transform.c:.6f},{transform.f:.6f},{transform.a:.8f},"
             f"{shape[0]}x{shape[1]},{shade_res_m:.2f}")
    key = hashlib.sha1(stamp.encode()).hexdigest()[:10]
    return (paths.interim_dir("statewide")
            / f"hs4326_{abs(transform.a):g}_shade{shade_res_m:g}_{key}.npz")


def hillshade(transform, shape, *, shade_res_m: float = None, tile_dir=None):
    """Cached hillshade on a display grid; see :mod:`firescape.relief`.

    ``shade_res_m`` defaults to :func:`shade_resolution` for the grid, which is
    what keeps a zoomed panel from being shaded coarser than it is drawn.
    """
    from firescape import relief

    if shade_res_m is None:
        shade_res_m = shade_resolution(transform, shape)
    tile_dir = tile_dir or (paths.cache_root() / "3dep_tiles")
    cache = _hillshade_cache(transform, shape, shade_res_m)
    if cache.exists():
        cached = np.load(cache)["hs"]
        if cached.shape == tuple(shape):
            return cached
    hs = relief.shaded_relief(sorted(tile_dir.glob("USGS_13_*.tif")),
                              transform, shape, crs="EPSG:4326",
                              shade_res_m=shade_res_m)
    np.savez_compressed(cache, hs=hs)
    return hs


def square_window(bounds4326, *, pad: float = 0.0):
    """Expand ``(w, s, e, n)`` to a window that is square *on screen*.

    :func:`style_axes` gives an axis a ``1/cos(lat)`` aspect, so a window
    square in degrees draws ~1.3x taller than wide at Nevada latitudes and
    letterboxes inside its subplot box. Squaring in screen units is what
    removes that white space, and it is the only reason a panel's degree span
    is not square.
    """
    w, s, e, n = bounds4326
    w, s, e, n = w - pad, s - pad, e + pad, n + pad
    cx, cy = (w + e) / 2.0, (s + n) / 2.0
    cos_lat = math.cos(math.radians(cy))
    span = max(e - w, (n - s) / cos_lat)
    return (cx - span / 2, cy - span * cos_lat / 2,
            cx + span / 2, cy + span * cos_lat / 2)


def tick_step(span_deg: float) -> float:
    """Degree tick interval giving at most ~6 ticks across ``span_deg``."""
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0):
        if span_deg / step <= 6:
            return step
    return 5.0


def burn(gdf, column: str, transform, shape):
    """Rasterize a column of a lon/lat GeoDataFrame onto the display grid.

    Non-finite values are skipped, so the result is NaN wherever no feature
    carried a usable value -- ready for ``np.ma.masked_invalid``.
    """
    from rasterio.features import rasterize

    shapes = ((g, v) for g, v in zip(gdf.geometry, gdf[column])
              if np.isfinite(v))
    return rasterize(shapes, out_shape=tuple(shape), transform=transform,
                     fill=np.nan, dtype="float32")


def _static(name: str, url: str):
    """Download a static context archive to raw/boundaries once, with a
    provenance sidecar."""
    dest = paths.raw_dir("boundaries") / name
    if not dest.exists():
        paths.download(url, dest)
        paths.write_provenance(dest, url=url, note="map context layer")
    return dest


def fetch_context(bounds4326=None, *, cache_dir=None):
    """Reference layers for statewide maps, as lon/lat GeoDataFrames.

    Returns a dict with ``counties``, ``lakes``, ``rivers``, ``roads`` and
    ``places``. Each layer is clipped to the domain and cached as GeoJSON, so
    only the first call touches the network. Roads and places come from
    stormscape's refdata (TIGER, GNIS); counties from Census TIGER; rivers and
    lakes from Natural Earth.
    """
    import geopandas as gpd
    from shapely.geometry import box

    inter = cache_dir or paths.interim_dir("statewide")
    b = tuple(bounds4326) if bounds4326 is not None else domain_bounds4326()
    clip = box(*b)
    out = {}

    def _cached(fname, build):
        f = inter / fname
        if not f.exists():
            build().to_file(f, driver="GeoJSON")
        return gpd.read_file(f)

    def _counties():
        z = _static("cb_2023_us_county_500k.zip", COUNTY_URL)
        g = gpd.read_file("zip://" + str(z))
        return g[g.intersects(clip)][["NAME", "STUSPS", "geometry"]]

    def _natural_earth(stem):
        def build():
            z = _static(f"{stem}.zip", f"{NATURAL_EARTH_S3}/{stem}.zip")
            g = gpd.read_file("zip://" + str(z))
            return g[g.intersects(clip)][["name", "geometry"]]
        return build

    out["counties"] = _cached("context_counties.geojson", _counties)
    out["rivers"] = _cached("context_rivers.geojson",
                            _natural_earth("ne_10m_rivers_lake_centerlines"))
    out["lakes"] = _cached("context_lakes.geojson",
                           _natural_earth("ne_10m_lakes"))

    def _roads():
        from stormscape import refdata
        return refdata.roads(b)

    def _places():
        from stormscape import refdata
        g = refdata.places(b)
        g = g[g["name"].isin(MAJOR_CITIES)].copy()
        g["geometry"] = g.geometry.representative_point()
        # The whitelist names Nevada cities, but GNIS carries same-named places
        # in neighbouring states (a "Caliente" in Kern County, CA once got
        # labelled), so clip to the state outright.
        nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson"
                           ).to_crs("EPSG:4326").union_all()
        return g[g.geometry.within(nv)].drop_duplicates("name")

    out["roads"] = _cached("context_roads.geojson", _roads)
    out["places"] = _cached("context_places.geojson", _places)
    return out


def draw_context(ax, ctx, *, label_cities: bool = True,
                 label_rivers: bool = True, extent=None):
    """Draw reference context onto an axis, in reading order back to front.

    Pass ``extent`` (the ``(w, e, s, n)`` from :func:`grid`) to get
    collision-aware labels: names are placed through
    :class:`stormscape.plot.Labeller`, which moves a label that would land on
    another one or run off the frame, and connects it with a leader once it
    has moved far enough to be ambiguous. That placement is computed in
    display coordinates, so the axis limits have to be final first — which is
    why the extent is passed here rather than left to a later
    :func:`style_axes` call. Without it the labels still avoid each other, but
    against whatever limits the axis has at the time.
    """
    import matplotlib.patheffects as pe

    # Every layer is checked for emptiness first. Plotting an empty
    # GeoDataFrame raises "aspect must be finite and positive" rather than
    # drawing nothing, and empty layers are normal, not exceptional: the Elko
    # window has no Natural Earth lake in it and the Nevada National Security
    # Site has no named perennial stream at all.
    if len(ctx["counties"]):
        ctx["counties"].boundary.plot(ax=ax, color="0.35", linewidth=0.35,
                                      alpha=0.7, zorder=3)
    if len(ctx["lakes"]):
        ctx["lakes"].plot(ax=ax, color="#a6cee3", alpha=0.8, zorder=3.1)
    if len(ctx["rivers"]):
        ctx["rivers"].plot(ax=ax, color="#2c7fb8", linewidth=0.7, alpha=0.9,
                           zorder=3.2)

    roads = ctx["roads"]
    if "kind" in roads.columns:
        # Local roads only appear where a figure asked for them (TIGER's local
        # layer is a separate query and ~200x the feature count). Drawn first
        # and faintest, so the highways still read as highways over them.
        local = roads[roads["kind"] == "local"]
        if len(local):
            local.plot(ax=ax, color="0.3", linewidth=0.15, alpha=0.55,
                       zorder=3.25)
        secondary = roads[roads["kind"] == "secondary"]
        if len(secondary):
            secondary.plot(ax=ax, color="0.25", linewidth=0.3, alpha=0.55,
                           zorder=3.3)
        primary = roads[roads["kind"] == "primary"]
        if len(primary):
            primary.plot(ax=ax, color="0.1", linewidth=0.7, zorder=3.4)

    places = ctx["places"]
    ax.scatter(places.geometry.x, places.geometry.y, s=9, color="black",
               edgecolor="white", linewidth=0.5, zorder=9)

    if extent is not None:
        style_axes(ax, extent)          # placement needs the final frame
    from stormscape.plot import Labeller, interior_point, shorten

    lab = Labeller(ax)
    lab.block_many(places.geometry.x, places.geometry.y, radius_px=4.5)
    if label_cities:
        for _, row in places.iterrows():
            lab.label(row.geometry.x, row.geometry.y, shorten(row["name"]),
                      fontsize=5.5, color="black", zorder=9.1, halo=1.6)
    if label_rivers:
        w, e = ax.get_xlim()
        s, n = ax.get_ylim()
        for _, row in ctx["rivers"].iterrows():
            name = row.get("name")
            if not name:
                continue
            pt = interior_point(row.geometry, (w, s, e, n))
            if pt is None:
                continue
            lab.label(pt[0], pt[1], shorten(str(name)), fontsize=5,
                      style="italic", color="#1b4f72", zorder=9.1, halo=1.4)


def style_axes(ax, extent, *, step: float = 1.0):
    """Degree-labelled WGS84 axes with a latitude-true aspect.

    ``extent`` is ``(west, east, south, north)``, matching :func:`grid`. The
    aspect ratio ``1/cos(lat)`` makes a degree of longitude the right width
    relative to a degree of latitude, so shapes are locally true and Nevada's
    meridian borders stand vertical.
    """
    w, e, s, n = extent
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1.0 / math.cos(math.radians((s + n) / 2)))
    xt = np.arange(math.ceil(w / step) * step, math.floor(e / step) * step + 1e-9, step)
    yt = np.arange(math.ceil(s / step) * step, math.floor(n / step) * step + 1e-9, step)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    fmt = "{:.0f}" if step >= 1 else "{:g}"
    ax.set_xticklabels([f"{fmt.format(abs(x))}°{'W' if x < 0 else 'E'}"
                        for x in xt], fontsize=7)
    ax.set_yticklabels([f"{fmt.format(abs(y))}°{'S' if y < 0 else 'N'}"
                        for y in yt], fontsize=7)
    # No graticule: white lines over a hillshade read as terrain edges rather
    # than as a grid. Degree ticks alone carry the georeference.
    ax.grid(False)
    ax.tick_params(length=2.5)


def save(fig, stem: str, *, dpi: int = 300, formats=("pdf",)):
    """Write a figure to ``figures/`` in each format. Returns the paths.

    PDF alone by default -- see the module docstring. ``dpi`` applies to
    raster formats only; a PDF carries its rasterized layers at their own
    resolution and its linework as vectors.
    """
    written = []
    for ext in formats:
        out = paths.figures_dir() / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", **({"dpi": dpi} if ext != "pdf" else {}))
        print("wrote", out)
        written.append(out)
    return written
