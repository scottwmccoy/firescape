"""Shared spec and furniture for the regional-zoom sheets.

Two figures render the same set of windows -- :mod:`p10_urban_zooms` the
annualized triptych, :mod:`p12_urban_zoom_forecast` the forecast-style pair --
and they have to agree on where each window is, which places and watercourses
get named, and how those labels look. Everything both need lives here, so
moving the Reno bounds or adding a region moves both maps at once.

The two urban corridors came first. The four that follow were chosen from a
scan of modelled hazard within ~20 km of every Nevada town outside them, which
is worth knowing because the two rankings it produces disagree: conditional
hazard peaks in White Pine County (32% of McGill's basins go moderate) while
the annual rate peaks on the Idaho line (1 in 51 yr at Jackpot), where burn
probability is highest but the terrain is gentler.

Not a package module: these are figure decisions for one product, not house
style. The house style (hillshade, alpha ceiling, degree axes, colour ramps)
is :mod:`firescape.plotting`.
"""
import math

import geopandas as gpd

from firescape import paths, plotting as mc

#: One entry per corridor. ``bounds`` is (west, south, east, north) in degrees;
#: ``step`` the graticule tick spacing; ``res`` the display-grid spacing.
#: ``rivers``/``lakes`` are the only watercourses named -- at corridor scale
#: every named creek and wash is clutter, and the trunk rivers are what the
#: corridor is organised around.
ZOOMS = {
    "reno_carson": dict(
        bounds=(-120.36, 38.58, -119.15, 40.42), step=0.25, res=0.0005,
        rivers=("Truckee River", "Carson River"),
        lakes=("Lake Tahoe", "Pyramid Lake"),
        label="Reno–Carson corridor",
        blurb="north Pyramid Lake through Reno–Sparks to Carson City, "
              "Minden and Gardnerville, west to the Sierra crest above "
              "Lake Tahoe and south past Topaz Lake"),
    "las_vegas": dict(
        bounds=(-116.35, 34.90, -113.90, 36.95), step=0.5, res=0.0006,
        rivers=("Colorado River", "Virgin River", "Muddy River"),
        lakes=("Lake Mead", "Lake Mohave"),
        label="Las Vegas / Clark County",
        blurb="Clark County with the Spring Mountains north of Pahrump, "
              "to the southern and eastern state lines"),
    # The I-80 / Humboldt River corridor, split in two. One window from
    # Lovelock to Wells would be 4 degrees wide -- statewide scale again, and
    # the point of a zoom is the drainage above a particular town.
    "elko_corridor": dict(
        bounds=(-116.45, 40.10, -114.45, 41.55), step=0.5, res=0.0005,
        rivers=("Humboldt River", "Marys River"),
        lakes=(),
        label="Carlin–Elko–Wells corridor",
        blurb="the I-80 and Humboldt River corridor through Elko County, "
              "with the Ruby Mountains and Lamoille Canyon south of Elko "
              "and Spring Creek"),
    "winnemucca_battle_mtn": dict(
        bounds=(-118.30, 40.15, -116.35, 41.60), step=0.5, res=0.0005,
        rivers=("Humboldt River", "Reese River", "Quinn River"),
        lakes=("Rye Patch Reservoir",),
        label="Winnemucca – Battle Mountain",
        blurb="the western I-80 corridor from Imlay to Beowawe, with the "
              "Sonoma, Osgood and Shoshone ranges and Paradise Valley"),
    "nnss": dict(
        bounds=(-117.00, 36.30, -115.30, 37.60), step=0.25, res=0.0005,
        rivers=("Amargosa River",),
        lakes=(),
        label="Nevada National Security Site",
        blurb="Pahute and Rainier mesas, Yucca and Frenchman flats and "
              "Mercury, west to Beatty and the Amargosa Desert, south to "
              "Indian Springs"),
    # Chosen from a scan of hazard within ~20 km of every Nevada town outside
    # the windows above; see the three notes below for what each one is for.
    #
    # The highest conditional hazard of any populated place in the state:
    # 32% of McGill's basins and 25% of Ely's go moderate if they burn,
    # against 11% in the Reno window.
    "ely_white_pine": dict(
        bounds=(-115.60, 38.60, -113.95, 40.10), step=0.25, res=0.0005,
        rivers=("Steptoe Creek", "Duck Creek", "Snake Creek", "Baker Creek",
                "Lehman Creek", "White River"),
        lakes=(),
        label="Ely – White Pine County",
        blurb="the Egan and Schell Creek ranges either side of Steptoe "
              "Valley at Ely, Ruth and McGill, east to the Snake Range and "
              "Great Basin National Park"),
    # Moderate hazard, but the strongest infrastructure case in Nevada:
    # Rainbow Canyon carries the Union Pacific mainline and US-93 through a
    # slot with documented flash-flood history.
    "lincoln_caliente": dict(
        bounds=(-115.45, 36.90, -113.95, 38.20), step=0.25, res=0.0005,
        rivers=("Meadow Valley Wash", "White River", "Muddy River"),
        lakes=(),
        label="Caliente – Lincoln County",
        blurb="Meadow Valley Wash and Rainbow Canyon through Caliente, "
              "Panaca and Pioche, with the Delamar and Clover mountains and "
              "the US-93 corridor south to Alamo"),
    # The shortest annual return intervals in the state -- 51 to 88 years,
    # against ~1,000 in the Reno corridor -- because burn probability peaks
    # along the Idaho line. Sparsely populated: this window is where the
    # annual rate lives, not where the people are.
    "north_elko": dict(
        bounds=(-116.35, 40.95, -114.40, 42.05), step=0.25, res=0.0005,
        rivers=("Owyhee River", "Bruneau River", "Jarbidge River",
                "Salmon Falls Creek", "Marys River"),
        lakes=("Wild Horse Reservoir",),
        label="Northern Elko County",
        blurb="the Idaho border country -- Jarbidge, Mountain City, Owyhee "
              "and Jackpot -- with the Independence, Bull Run and Jarbidge "
              "mountains"),
}

#: Places to label even though GNIS does not call them incorporated. Minden,
#: Gardnerville and Carson City are exactly the towns these corridors are
#: about, and an "incorporated only" filter drops all three.
NOTABLE = {
    "Carson City", "Minden", "Gardnerville", "Gardnerville Ranchos",
    "Incline Village", "Dayton", "Virginia City", "Stateline", "Verdi",
    "Boulder City", "Pahrump", "Mesquite", "Indian Springs", "Searchlight",
    "Laughlin", "Overton", "Logandale", "Bunkerville", "Blue Diamond",
    "Mount Charleston", "Sandy Valley", "Primm", "Moapa Valley", "Jean",
    # I-80 east: Elko, Carlin and Wells are incorporated, but Spring Creek
    # (the largest settlement in the window) and Lamoille, which sits at the
    # mouth of Lamoille Canyon, are not.
    "Spring Creek", "Lamoille", "Ryndon", "Osino", "Jiggs", "Deeth",
    "Halleck", "Ruby Valley", "Tuscarora",
    # I-80 west: only Winnemucca is incorporated -- Battle Mountain, the
    # other town this window is about, is not.
    "Battle Mountain", "Golconda", "Valmy", "Paradise Valley", "Orovada",
    "Imlay", "Mill City", "Crescent Valley", "Beowawe", "Unionville",
    # NNSS and the Amargosa Desert: nothing here is incorporated at all.
    "Mercury", "Beatty", "Amargosa Valley", "Johnnie", "Crystal",
    "Cactus Springs", "Rhyolite", "Death Valley Junction", "Furnace Creek",
    # White Pine: only Ely is incorporated, and McGill -- which carries the
    # highest conditional hazard of any settlement in the state -- is not.
    "McGill", "Ruth", "East Ely", "Lund", "Preston", "Cherry Creek",
    "Baker", "Garrison", "Osceola", "Steptoe", "Majors Place",
    # Lincoln: Caliente is the only incorporated place; Panaca and Pioche,
    # the other two towns on Meadow Valley Wash, are not.
    "Panaca", "Pioche", "Alamo", "Ursine", "Elgin", "Hiko", "Ash Springs",
    "Crystal Springs", "Rose Valley", "Caselton", "Delamar",
    # Northern Elko: nothing here is incorporated except Wells, which the
    # Carlin-Elko-Wells window already covers.
    "Jarbidge", "Mountain City", "Owyhee", "Jackpot", "Contact", "Charleston",
    "Rowland", "Murphy Hot Springs", "North Fork", "Deeth",
}
_PREFIX = ("City of ", "Town of ", "Village of ", "Township of ")


def places_for(bounds):
    """Labelled places in the window.

    The statewide whitelist is tuned for a statewide sheet; at corridor scale
    it drops the towns the map is about. GNIS prefixes its civil names ("City
    of Reno"), which is noise on a figure, so those are stripped.
    """
    from stormscape import refdata

    g = refdata.places(bounds).to_crs("EPSG:4326").copy()
    g["geometry"] = g.geometry.representative_point()
    nm = g["name"].astype(str)
    for p in _PREFIX:
        nm = nm.str.removeprefix(p)
    g["name"] = nm.str.strip()
    inc = (g["kind"].astype(str).str.contains("incorporated", case=False, na=False)
           if "kind" in g.columns else False)
    g = g[inc | g["name"].isin(NOTABLE)]
    g = g[~g["name"].str.contains("Township|County", case=False, na=False)]
    return g.drop_duplicates("name")


def context(key):
    """Reference layers plus label anchors for one corridor.

    Returns a dict with the :func:`firescape.plotting.fetch_context` layers
    (``rivers`` replaced by the NHD trunk rivers), one label point per river
    and per named lake, the place points, and the state outline.
    """
    z = ZOOMS[key]
    bounds = z["bounds"]
    inter = paths.interim_dir("zooms", key)
    ctx = mc.fetch_context(bounds, cache_dir=inter)

    # Natural Earth 10 m carries two rivers in the Reno window and does not
    # include the Truckee at all -- at corridor scale the watercourses have to
    # come from NHD, which is what the map is actually about.
    #
    # The query is slow (~9 min over the Las Vegas window) and sits at 0% CPU
    # the whole time, blocked on the network. It looks exactly like a hang; it
    # is not. It caches, so it is a one-time cost per window.
    #
    # Cache it UNFILTERED and apply the whitelist on read. The query is the
    # expensive half and the whitelist is the half that gets revised, so
    # binding the two together made every change of mind cost another query.
    # The older pre-filtered cache is still honoured where it exists --
    # filtering it again by the same names is a no-op.
    legacy, cache = (inter / "context_major_rivers.geojson",
                     inter / "context_named_streams.geojson")
    if cache.exists():
        rv = gpd.read_file(cache)
    elif legacy.exists():
        rv = gpd.read_file(legacy)
    else:
        from stormscape import refdata
        rv = refdata.streams(bounds, named_only=True).to_crs("EPSG:4326")
        rv.to_file(cache, driver="GeoJSON")
    want = z["rivers"]
    ctx["rivers"] = rv[rv["name"].astype(str).apply(
        lambda n: any(w.lower() in n.lower() for w in want))].reset_index(drop=True)

    # EVERY reach is kept for drawing. Deduplicating by name here -- the
    # obvious way to get one label per river -- silently throws away the rest
    # of the river: NHD splits the Truckee into dozens of reaches, so what got
    # drawn was a 5 km fragment. Labels are deduplicated separately, below.
    rv_all = ctx["rivers"]
    rv_all = rv_all[rv_all.geometry.notna()]
    _len = rv_all.to_crs("EPSG:5070").length
    rv_lab = (rv_all.assign(_len=_len.values)
                    .sort_values("_len", ascending=False)
                    .drop_duplicates("name").copy())
    rv_lab["geometry"] = rv_lab.geometry.interpolate(0.5, normalized=True)

    lk = ctx["lakes"]
    lk_lab = (lk[lk["name"].astype(str).isin(z["lakes"])]
              if "name" in lk else lk.iloc[:0])

    nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson"
                       ).to_crs("EPSG:4326")
    return dict(ctx=ctx, rivers=rv_all, river_labels=rv_lab, lake_labels=lk_lab,
                places=places_for(bounds), state=nv)


def decorate(ax, C, extent, *, step):
    """Draw the context and every label onto one panel, front to back.

    Identical across both sheet formats on purpose: the corridor figures are
    read side by side, and a place name that moves or changes weight between
    them reads as a difference in the data.
    """
    import matplotlib.patheffects as pe

    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    mc.draw_context(ax, C["ctx"], label_cities=False, label_rivers=False)
    C["state"].boundary.plot(ax=ax, color="black", linewidth=1.6, zorder=8)

    for _, r in C["river_labels"].iterrows():
        ax.annotate(str(r["name"]), (r.geometry.x, r.geometry.y), fontsize=6.5,
                    style="italic", color="#1b4f72", ha="center", zorder=9.1,
                    path_effects=halo)
    for _, r in C["lake_labels"].iterrows():
        c = r.geometry.representative_point()
        ax.annotate(str(r["name"]), (c.x, c.y), fontsize=7, style="italic",
                    color="#1b4f72", ha="center", zorder=9.1, path_effects=halo)

    pl = C["places"]
    if len(pl):
        ax.scatter(pl.geometry.x, pl.geometry.y, s=10, color="black",
                   edgecolor="white", linewidth=0.5, zorder=9)
        for _, r in pl.iterrows():
            # a WHITE halo: the calibration-perimeter style is a black stroke,
            # which around black text just reads as bold
            ax.annotate(str(r["name"]), (r.geometry.x, r.geometry.y),
                        xytext=(3, 2), textcoords="offset points", fontsize=6,
                        zorder=9.1, color="black", path_effects=halo)
    mc.style_axes(ax, extent, step=step)


def suptitle(fig, lines, *, fontsize=13, gap_in=0.16):
    """Wrap a suptitle to the figure width and hang it off the panels.

    **Call this after** ``tight_layout``. Two separate things go wrong if a
    corridor sheet is titled the ordinary way:

    * ``bbox_inches="tight"`` grows the saved canvas to whatever the widest
      artist needs, so an over-long single-line subtitle is not clipped -- it
      silently re-centres the maps inside its own width. The first two-panel
      sheet had a subtitle that wanted 26 inches in a 12-inch figure. Hence
      the wrap, measured rather than guessed at a character count.
    * Map panels are aspect-locked, so as soon as the width budget binds --
      which it does the moment a colourbar and its label eat into the slot --
      ``tight_layout`` leaves the unusable height as a band of white between
      the title and the maps. A tight bbox cannot crop that: it is interior.
      Anchoring the title to the top of the panels instead of to the top of
      the canvas removes it, and the leftover space above the title is
      exterior, so the tight bbox does crop it.
    """
    import textwrap

    fig_w, fig_h = fig.get_size_inches()
    probe = fig.text(0, 0, lines[0], fontsize=fontsize)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    per_char = (probe.get_window_extent()
                     .transformed(fig.dpi_scale_trans.inverted()).width
                / max(len(lines[0]), 1))
    probe.remove()

    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, max(24, int(fig_w * 0.97 / per_char)))
                       or [""])

    inv = fig.transFigure.inverted()
    top = max(inv.transform(ax.get_tightbbox(renderer).p1)[1]
              for ax in fig.axes)
    return fig.suptitle("\n".join(wrapped), y=top + gap_in / fig_h,
                        va="bottom", fontsize=fontsize)


#: Below this, the KF gap fill is a footnote; above it, it belongs on the map.
KF_NOTE_THRESHOLD = 0.05


def kf_note(fraction, noun="basins"):
    """A disclosure line for windows where SSURGO never mapped the soil.

    Returns ``None`` when the fill is negligible. The K-factor gap is not
    uniform across Nevada -- it is 0.0% over Elko County and 52% over the
    Nevada National Security Site, where the survey was never completed, and
    17.6% around Las Vegas. Those segments take the statewide median KF, which
    is what the published basin layer does and is therefore the consistent
    choice; the failure would be shipping a sheet that does not say so, since
    a figure travels away from its summary JSON.
    """
    if not fraction or fraction < KF_NOTE_THRESHOLD:
        return None
    return (f"soil erodibility unmapped by SSURGO for {fraction:.0%} of "
            f"{noun} in this window — those take the statewide median K "
            f"and their likelihood is correspondingly less certain")


def figure_size(bounds, panels, *, panel_h=9.4, panel_w_max=10.0,
                cbar=0.95, edge=0.5, title_h=2.1):
    """Figure size that leaves no slack around ``panels`` map panels.

    Panels are aspect-locked to ``1/cos(lat)`` by :func:`plotting.style_axes`,
    so a fixed figure width leaves a different amount of white space per
    window -- tall-narrow Reno floated in a sea of margin. Size the figure to
    the window instead. ``cbar`` is the room a colourbar and its labels take.

    ``panel_w_max`` bounds the damage from a wide, short window: at a fixed
    panel height, the northern Elko window (aspect 1.3) would want a 40-inch
    sheet. Panels wider than the cap trade height for it instead. No window
    already drawn reaches the cap, so it changes nothing retroactively.
    """
    lon_span, lat_span = bounds[2] - bounds[0], bounds[3] - bounds[1]
    ratio = lon_span * math.cos(math.radians((bounds[1] + bounds[3]) / 2)) / lat_span
    if panel_h * ratio > panel_w_max:
        panel_h = panel_w_max / ratio
    return panels * (panel_h * ratio + cbar) + edge, panel_h + title_h
