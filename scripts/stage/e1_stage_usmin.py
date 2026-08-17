"""Stage USMIN mine features for Nevada -> raw/usmin/nv_mines.gpkg.

Pulls the storm-relevant groups (waste + openings, both geometry layers) from
the public USMIN topographic-symbol service via stormscape.mine_features and
stages them immutably with a provenance sidecar. The waste polygons — dumps
and tailings digitised as mapped extents — are the asset layer for the AML
exposure product.

Caveat carried into the sidecar: USMIN is digitised from published topo
sheets. It is historical (a feature may be reclaimed or misregistered) and it
is not the state's operational AML hazard inventory (token-gated; see
stormscape.mines.SOURCES["ndom"]).
"""
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from stormscape import mine_features
from stormscape.mines import SOURCES

from firescape import paths

DEST = paths.raw_dir("usmin") / "nv_mines.gpkg"

if DEST.exists():
    print(f"already staged (raw/ is immutable): {DEST}")
    sys.exit(0)

nv = paths.raw_dir("boundaries") / "nv_state.geojson"
print("querying USMIN over the Nevada boundary ...", flush=True)
gdf = mine_features(str(nv), kinds=("waste", "openings"), geometry="both",
                    pad_deg=0.01)
print(f"{len(gdf)} features")
print(gdf.groupby(["group", "geom_kind"]).size().to_string())

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td) / DEST.name
    gdf.to_file(tmp, layer="mines", driver="GPKG")
    paths.atomic_into(tmp, DEST)

paths.write_provenance(
    DEST, url=SOURCES["usmin"]["service"],
    note="USGS USMIN prospect- and mine-related features (topo-sheet "
         "symbols), NV + 0.01 deg pad; kinds=waste+openings, both geometry "
         "layers. Historical record, not an operational AML hazard "
         "inventory.",
    n_features=int(len(gdf)),
    counts={f"{g}/{k}": int(n) for (g, k), n in
            gdf.groupby(["group", "geom_kind"]).size().items()})
print(f"staged {DEST}")
