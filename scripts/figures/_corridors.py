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

from stormscape.plot import Labeller, shorten

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
    # A zoom inside reno_carson, an order of magnitude tighter than any other
    # window here (14 km, against 100+ km for the corridors). The 2026-06-19
    # storm put debris flows off the Virginia Range front onto the fans above
    # Hidden Valley, and they reached the Truckee at Vista -- so this is the
    # one window where a modelled channel can be laid beside a flow that
    # actually happened. Extent matches the stormscape rainfall zoom for the
    # event, so the hazard sheets and the I15 sheets overlay.
    #
    # `res` is a tenth of the corridor windows': at 0.0005 deg this window
    # would render 280 px wide (43 m/px) and the individual range-front
    # drainages -- the entire point of it -- would be two pixels across.
    # 0.0001 deg is 8.6 m/px at this latitude, just off the 10 m 3DEP floor.
    "hidden_valley": dict(
        bounds=(-119.77, 39.46, -119.63, 39.58), step=0.05, res=0.0001,
        rivers=("Truckee River", "Steamboat Creek", "Boynton Slough",
                "Long Valley Creek", "Red Ravine Creek", "Alum Creek"),
        lakes=(),
        label="Hidden Valley – Vista",
        blurb="the Virginia Range front above east Reno and the Truckee "
              "River reach at Vista, site of the 19 June 2026 unburned "
              "debris flows"),
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
    # The only window that draws local roads and the federal land boundaries.
    # Both are specific to what this sheet is for: the access network *is* the
    # exposure here -- there is no town to speak of -- and a hazard map of the
    # Test Site is hard to read without knowing where the Site ends.
    "nnss": dict(
        bounds=(-117.00, 36.30, -115.30, 37.60), step=0.25, res=0.0005,
        rivers=("Amargosa River",),
        lakes=(),
        local_roads=True,
        agencies=("DOE", "DOD"),
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
        # Great Basin NP: the Snake Range canyons above Baker are steep, and
        # park roads and campgrounds sit in them. Clips to 312 km2, matching
        # the park's published area.
        agencies=("NPS",),
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
        # 190 km of Union Pacific mainline, most of it in Rainbow Canyon --
        # the reason this window was chosen over a purely hazard-ranked one.
        rail=True,
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
    # Added 2026-09-03 for the same reason as hidden_valley: 19 of the 55
    # points in the statewide debris-flow inventory (nv_debrisflow_inventory,
    # a hand-digitized Google Earth record, not modelled output) sit on this
    # one range front, from two separate storms 9 months apart (27 Sep 2023,
    # 13 Jun 2024) -- more observed flows than any other window in this file
    # covers, and none of the existing ones reach it. Closed basin: no named
    # river drains it.
    "railroad_valley": dict(
        bounds=(-115.85, 38.00, -115.32, 38.49), step=0.1, res=0.0004,
        rivers=(), lakes=(),
        label="Railroad Valley range front",
        blurb="the Quinn Canyon / Golden Gate range front above Railroad "
              "Valley, Nye County -- Adaven, Nyala and Crows Nest are the "
              "nearest named places"),
}

#: Places to label even though GNIS does not call them incorporated. Minden,
#: Gardnerville and Carson City are exactly the towns these corridors are
#: about, and an "incorporated only" filter drops all three.
NOTABLE = {
    "Carson City", "Minden", "Gardnerville", "Gardnerville Ranchos",
    "Incline Village", "Dayton", "Virginia City", "Stateline", "Verdi",
    # East Reno, for the Hidden Valley window: Reno and Sparks are
    # incorporated and come through on their own, but the neighbourhoods the
    # 2026-06-19 flows actually reached are GNIS populated places and would
    # otherwise go unlabelled on the sheet they are the subject of.
    "Hidden Valley", "Vista", "Glendale", "Lockwood", "Sun Valley",
    "Spanish Springs", "Mustang", "Donovan Mill",
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
    # Railroad Valley: nothing here is incorporated, or close to it -- the
    # nearest named GNIS place to the debris-flow cluster is 14 km away.
    "Adaven", "Nyala", "Crows Nest", "Currant", "Lockes",
}
_PREFIX = ("City of ", "Town of ", "Village of ", "Township of ")


def places_for(bounds):
    """Labelled places in the window.

    The statewide whitelist is tuned for a statewide sheet; at corridor scale
    it drops the towns the map is about. GNIS prefixes its civil names ("City
    of Reno"), which is noise on a figure, so those are stripped.
    """
    from stormscape import refdata

    g = refdata.places(bounds)
    if not len(g) or "name" not in g.columns:
        # GNIS answers with non-JSON often enough to matter: it went down
        # mid-batch once and took the rest of a sheet run with it. A window
        # with no place names is a worse figure, not a failed one.
        print("  no place names available for this window", flush=True)
        return gpd.GeoDataFrame({"name": [], "kind": []},
                                geometry=gpd.GeoSeries([], crs=4326), crs=4326)
    g = g.to_crs("EPSG:4326").copy()
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

    # Local roads, where the window asks for them. TIGER's local layer is a
    # separate query and about 200x the feature count (8,196 against 37 over
    # the Test Site), so it is opt-in and cached separately from the
    # primary/secondary roads every other window uses.
    if z.get("local_roads"):
        rd_path = inter / "context_roads_local.geojson"
        if rd_path.exists():
            ctx["roads"] = gpd.read_file(rd_path)
        else:
            from stormscape import refdata
            rd = refdata.roads(bounds, local=True).to_crs("EPSG:4326")
            rd.to_file(rd_path, driver="GeoJSON")
            ctx["roads"] = rd

    # Railroads, where the window asks. TIGER layer 9, which roads() does not
    # fetch -- rail is not a road tier. Only 27 features over the Caliente
    # window, but they are the Union Pacific mainline up Rainbow Canyon, which
    # is the exposure that window is about.
    rail = None
    if z.get("rail"):
        rail_path = inter / "context_rail.geojson"
        if rail_path.exists():
            rail = gpd.read_file(rail_path)
        else:
            from stormscape import refdata
            rail = refdata._query(refdata.TIGER_TRANS, 9, bounds,
                                  out_fields="NAME").to_crs("EPSG:4326")
            rail.to_file(rail_path, driver="GeoJSON")

    nv = gpd.read_file(paths.raw_dir("boundaries") / "nv_state.geojson"
                       ).to_crs("EPSG:4326")
    return dict(ctx=ctx, rivers=rv_all, river_labels=rv_lab, lake_labels=lk_lab,
                places=places_for(bounds), state=nv, rail=rail,
                agencies=federal_lands(bounds, z.get("agencies", ()), inter))


#: BLM's national Surface Management Agency service. The Nevada National
#: Security Site is administered by DOE, not DoD, so it is absent from the
#: Census military-installation file -- that carries Nellis Air Force Range
#: and Creech AFB but not the Site. This service has it: the DOE polygon
#: clips to 3,515 km2 in the window, against a published Site area of ~3,520.
SMA_URL = ("https://gis.blm.gov/arcgis/rest/services/lands/"
           "BLM_Natl_SMA_Cached_without_PriUnk/MapServer/1/query")

#: How each agency's boundary is drawn. The subject of the sheet gets a
#: saturated cyan, chosen because it is the one strong hue absent from BOTH
#: panels: plasma_r runs yellow-salmon-purple-blue and the hazard classes are
#: green/orange/red. Okabe-Ito reddish purple was tried first and disappeared
#: into plasma's salmon midtones, which is most of this window. Its neighbour
#: gets neutral dashes. Both take a white casing to survive either ramp.
AGENCY_STYLE = {
    "DOE": dict(label="Nevada National Security Site (DOE)",
                color="#00A0B0", linewidth=1.9, linestyle="-", casing=3.6),
    "DOD": dict(label="Nellis Air Force Range / Creech AFB (DoD)",
                color="#333333", linewidth=1.1, linestyle=(0, (5, 3)),
                casing=2.6),
    "NPS": dict(label="Great Basin National Park (NPS)",
                color="#00A0B0", linewidth=1.9, linestyle="-", casing=3.6),
}


def federal_lands(bounds, agencies, inter):
    """Surface-management polygons for the window, one per agency code.

    Simplified server-side to ~70 m, which is finer than the display grid and
    keeps each polygon under 50 kB. Returns ``{}`` when a window asks for
    none, which is every window but the Test Site.
    """
    import json
    import urllib.parse
    import urllib.request

    out = {}
    for dept in agencies:
        path = inter / f"context_sma_{dept}.geojson"
        if not path.exists():
            q = urllib.parse.urlencode({
                "geometry": ",".join(str(b) for b in bounds),
                "geometryType": "esriGeometryEnvelope",
                "inSR": 4326, "outSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
                # Agency, not department: NPS lives under DOI alongside BLM,
                # so a department filter would return every acre of BLM land
                # in the window instead of the park. DOE and DOD carry the
                # same code in both fields, so this selects them unchanged.
                "where": f"ADMIN_AGENCY_CODE='{dept}'",
                "outFields": "ADMIN_DEPT_CODE,ADMIN_AGENCY_CODE",
                "maxAllowableOffset": 0.0008,
                "returnGeometry": "true", "f": "geojson"})
            with urllib.request.urlopen(f"{SMA_URL}?{q}", timeout=180) as r:
                payload = json.load(r)
            if not payload.get("features"):
                print(f"  no {dept} land in the window", flush=True)
                continue
            path.write_text(json.dumps(payload))
        # Deliberately NOT clipped to the window. The service returns whole
        # features -- verified by re-querying with an envelope twice the size
        # and getting identical bounds and area -- so clipping here would add
        # the window edge to the polygon and draw it as if it were an agency
        # boundary. The axes limits do the clipping instead, which only ever
        # hides a real edge rather than inventing one.
        g = gpd.read_file(path)
        if len(g):
            out[dept] = g
    return out


def decorate(ax, C, extent, *, step, extra_handles=()):
    """Draw the context and every label onto one panel, front to back.

    Identical across both sheet formats on purpose: the corridor figures are
    read side by side, and a place name that moves or changes weight between
    them reads as a difference in the data.
    """
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    halo = [pe.withStroke(linewidth=1.6, foreground="white")]
    mc.draw_context(ax, C["ctx"], label_cities=False, label_rivers=False)
    C["state"].boundary.plot(ax=ax, color="black", linewidth=1.6, zorder=8)

    # Federal land boundaries, where a window asks for them. Never clipped to
    # the window in code -- see federal_lands -- so the axes limits hide the
    # parts outside rather than drawing the frame as a boundary.
    # One legend per axis: a second ax.legend() call would replace this one,
    # so anything a sheet wants listed comes in through extra_handles.
    handles = list(extra_handles)
    rail = C.get("rail")
    if rail is not None and len(rail):
        # The conventional rail symbol, in two passes: a solid line with a
        # dashed white overlay for the ties. Drawn ABOVE the hazard layer --
        # unlike roads, which stay beneath it. The whole point of putting rail
        # on the Caliente sheet is to see the mainline against the flagged
        # tributaries it runs past, and 190 km of thin line hides nothing.
        rail.plot(ax=ax, color="#111111", linewidth=1.5, zorder=8.3)
        rail.plot(ax=ax, color="white", linewidth=0.85, zorder=8.31,
                  linestyle=(0, (1.4, 2.4)))
        name = str(rail["NAME"].dropna().iloc[0]) if "NAME" in rail else "railroad"
        handles.append(Line2D([0], [0], color="#111111", lw=1.5,
                              label=f"{name} mainline"))

    for dept, g in (C.get("agencies") or {}).items():
        s = AGENCY_STYLE[dept]
        g.boundary.plot(ax=ax, color=s["color"], linewidth=s["linewidth"],
                        linestyle=s["linestyle"], zorder=8.5,
                        path_effects=[pe.withStroke(linewidth=s["casing"],
                                                    foreground="white")])
        handles.append(Line2D([0], [0], color=s["color"], lw=s["linewidth"],
                              linestyle=s["linestyle"], label=s["label"]))
    if handles:
        ax.legend(handles=handles, loc="lower left", fontsize=7,
                  framealpha=0.9, borderpad=0.5).set_zorder(9.5)

    # Labels last, and through one Labeller for the whole panel: the corridor
    # windows put a town on the river it sits on, so water names and place
    # names compete for the same few pixels (Dayton and the Carson River were
    # printing on top of each other). Limits first -- placement is computed in
    # display coordinates.
    mc.style_axes(ax, extent, step=step)
    ax.figure.canvas.draw()
    lab = Labeller(ax)

    pl = C["places"]
    if len(pl):
        ax.scatter(pl.geometry.x, pl.geometry.y, s=10, color="black",
                   edgecolor="white", linewidth=0.5, zorder=9)
        lab.block_many(pl.geometry.x, pl.geometry.y, radius_px=4.0)

    for _, r in C["river_labels"].iterrows():
        lab.label(r.geometry.x, r.geometry.y, shorten(str(r["name"])),
                  fontsize=6.5, style="italic", color="#1b4f72", zorder=9.1,
                  halo=1.6)
    for _, r in C["lake_labels"].iterrows():
        c = r.geometry.representative_point()
        lab.label(c.x, c.y, shorten(str(r["name"])), fontsize=7,
                  style="italic", color="#1b4f72", zorder=9.1, halo=1.6)
    for _, r in pl.iterrows():
        # a WHITE halo: the calibration-perimeter style is a black stroke,
        # which around black text just reads as bold
        lab.label(r.geometry.x, r.geometry.y, shorten(str(r["name"])),
                  fontsize=6, color="black", zorder=9.1, halo=1.6)


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


def unsupported_note(n, km, total):
    """A disclosure line for channel the model has no real input over.

    Staley's M1 reads terrain, fire and soil. Where no upslope cell reaches
    23 degrees the terrain term is exactly zero, and where SSURGO never mapped
    the soil the K-factor is a statewide median rather than a measurement --
    so on ground that is both, the likelihood rests on a filled constant and a
    simulated severity, with the soil term carrying ~91% of it. That is 15% of
    the Test Site window and ~0% of Elko, which is why the gate is written
    against the data rather than against a region.
    """
    if not n or n / max(total, 1) < 0.01:
        return None
    return (f"{n:,} segments ({n / total:.0%}, {km:,.0f} km) drawn grey: no "
            f"mapped soil and no slope ≥23° upslope, so M1 has no measured "
            f"input there — the channel is real, the number would not be")


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
