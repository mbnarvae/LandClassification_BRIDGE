"""EXP003 smoke test: Antisana + Zhurucay → HLS 2023 → frozen Prithvi-EO-2.0 → k-means vs MAATE maps.

Steps (each skips work already on disk):
  refs     clip MAATE ecosystems, land cover (all years), Ramsar wetlands and soil points to each site
  hls      dry/wet HLS median composites for one year (needs Earthdata login in ~/.netrc)
  embed    Prithvi patch embeddings (16×16 px = 480 m) for the whole rectangle
  analyze  rasterise references, k-means on embeddings and on a spectral baseline, scores and figures

python scripts/exp003_ec_sites_smoke.py all
python scripts/exp003_ec_sites_smoke.py analyze
"""

from __future__ import annotations

import argparse
import glob
import json
import unicodedata
from pathlib import Path

import numpy as np

from bridge_lc.config import REPO_ROOT, load_yaml

CFG_PATH = REPO_ROOT / "experiments/exp003_ec_sites_smoke/experiment.yaml"
FILL = -9999
LEVEL1_VOCAB = {
    "BOSQUE",
    "CUERPO DE AGUA",
    "OTRAS TIERRAS",
    "TIERRA AGROPECUARIA",
    "VEGETACION ARBUSTIVA Y HERBACEA",
    "ZONA ANTROPICA",
    "SIN INFORMACION",
}


def norm(s) -> str:
    """Upper-case, accent-free label (MAATE files mix encodings and accents across years)."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    if "Ã" in s or "Â" in s:  # UTF-8 stored as Latin-1 in several MAATE DBFs (e.g. "HÃMEDO")
        try:
            s = s.encode("latin-1").decode("utf-8")
        except UnicodeError:
            pass
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.upper().split())


def site_dirs(cfg: dict, site_id: str) -> Path:
    d = REPO_ROOT / cfg["output_dir"] / site_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def site_bbox(cfg: dict, site: dict) -> tuple[float, float, float, float]:
    import geopandas as gpd

    g = gpd.read_file(Path(cfg["data_root"]).expanduser() / site["shape"]).to_crs(4326)
    return tuple(float(v) for v in g.total_bounds)


# --------------------------------------------------------------------------- refs
def land_cover_levels(gdf):
    """Find the level-1 and level-2 class fields (their names change between MAATE editions)."""
    cats = {}
    for c in gdf.columns:
        if c == "geometry" or gdf[c].dtype.kind not in "OU" and str(gdf[c].dtype) != "str":
            continue
        vals = {norm(v) for v in gdf[c].dropna().unique()}
        if 3 <= len(vals) <= 40:
            cats[c] = vals
    lvl1 = next((c for c, v in cats.items() if v <= LEVEL1_VOCAB), None)
    rest = {c: v for c, v in cats.items() if c != lvl1 and not v <= {"NATURAL", "ARTIFICIAL"} | LEVEL1_VOCAB}
    lvl2 = max(rest, key=lambda c: len(rest[c])) if rest else None
    return lvl1, lvl2


def step_refs(cfg: dict) -> None:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    root = Path(cfg["data_root"]).expanduser()
    refs = cfg["references"]
    for site in cfg["sites"]:
        out = site_dirs(cfg, site["id"]) / "references.gpkg"
        if out.exists():
            print(f"[{site['id']}] references exist, skipping")
            continue
        bbox = site_bbox(cfg, site)
        aoi = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326).to_crs(cfg["crs"])
        print(f"[{site['id']}] bbox {np.round(bbox, 4).tolist()}")

        def clip(path: Path, aoi=aoi):
            g = gpd.read_file(path, bbox=aoi)  # MAATE files are UTF-8 (.cst)
            return gpd.clip(g.to_crs(cfg["crs"]), aoi) if len(g) else g

        aoi.assign(site=site["id"]).to_file(out, layer="aoi")
        eco = clip(root / refs["ecosystems"])
        eco["label"] = eco["ecosistema"].map(norm)
        eco.to_file(out, layer="ecosystems")
        print(f"  ecosystems: {eco['label'].nunique()} classes, {len(eco)} polygons")

        ram = clip(root / refs["ramsar"])
        ram.to_file(out, layer="ramsar") if len(ram) else None
        print(f"  ramsar: {len(ram)} polygons {list(ram.get('nombre', []))}")

        for year in refs["land_cover_years"]:
            path = Path(glob.glob(str(root / refs["land_cover_glob"].format(year=year)))[0])
            lc = clip(path)
            l1, l2 = land_cover_levels(lc)
            lc["level1"] = lc[l1].map(norm) if l1 else ""
            lc["level2"] = lc[l2].map(norm) if l2 else ""
            lc.to_file(out, layer=f"land_cover_{year}")
            print(f"  land cover {year}: fields {l1}/{l2}, {lc['level2'].nunique()} level-2 classes")

        soil = pd.read_csv(root / refs["soil_points"])
        pts = gpd.GeoDataFrame(soil, geometry=gpd.points_from_xy(soil.lon_wgs84, soil.lat_wgs84), crs=4326)
        pts = pts.to_crs(cfg["crs"])
        pts = pts[pts.within(aoi.geometry[0])]
        if len(pts):
            pts["cover"] = pts["Cobertura"].map(norm)
            pts["in_ramsar"] = pts.within(ram.union_all()) if len(ram) else False
            pts.to_file(out, layer="soil_points")
        print(f"  soil points: {len(pts)}")


# --------------------------------------------------------------------------- hls
def _retry(fn, tries: int = 4, wait: float = 15.0):
    import time

    for a in range(tries):
        try:
            return fn()
        except Exception as e:  # LP DAAC / CMR fail transiently (empty reads, HTML instead of a COG)
            if a == tries - 1:
                raise
            print(f"    retry {a + 1}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(wait * (a + 1))


def season_scenes(bbox, start: str, end: str, max_cloud: float) -> list:
    from pystac_client import Client

    from bridge_lc.data.stac import CMR_LPCLOUD

    def query():
        client = Client.open(CMR_LPCLOUD)
        return [
            it
            for col in ("HLSS30_2.0", "HLSL30_2.0")
            for it in client.search(collections=[col], bbox=bbox, datetime=f"{start}/{end}").items()
            if it.properties.get("eo:cloud_cover", 0) <= max_cloud
        ]

    return _retry(query)


def cache_scene(item, gbox, bands: list[str], path: Path) -> None:
    """Cloud-masked 6-band int16 scene on the site grid, one file per granule (resumable)."""
    import dask
    import rioxarray  # noqa: F401
    from odc.stac import load

    from bridge_lc.data.composites import HLS_BANDS, HLS_FILL, fmask_valid

    product = "hls_s30" if ".S30." in item.id else "hls_l30"
    mapping = HLS_BANDS[product]

    def read():
        with dask.config.set(scheduler="synchronous"):  # one request at a time: no cookie-jar races
            ds = (
                load(
                    [item],
                    bands=[mapping[b] for b in bands] + ["Fmask"],
                    geobox=gbox,
                    nodata=HLS_FILL,
                    chunks={},
                )
                .isel(time=0)
                .load()
            )
        return ds

    ds = _retry(read)
    valid = fmask_valid(ds["Fmask"].values) & (ds["Fmask"].values != 255)
    arr = ds[[mapping[b] for b in bands]].to_array("band")
    arr = arr.where(valid & (arr != HLS_FILL), HLS_FILL).astype("int16")
    tmp = path.with_suffix(".tmp.tif")
    arr.rio.write_crs(gbox.crs.to_wkt()).rio.write_nodata(HLS_FILL).rio.to_raster(tmp, compress="deflate")
    tmp.rename(path)


def median_from_cache(paths: list[Path], rows: int = 256):
    """Per-pixel median over cached scenes (block by block to keep memory low) + clear-obs count."""
    import rasterio

    with rasterio.open(paths[0]) as src:
        C, H, W = src.count, src.height, src.width
    med = np.full((C, H, W), FILL, "int16")
    nclear = np.zeros((H, W), "int16")
    srcs = [rasterio.open(p) for p in paths]
    try:
        for r0 in range(0, H, rows):
            win = rasterio.windows.Window(0, r0, W, min(rows, H - r0))
            block = np.stack([s.read(window=win) for s in srcs]).astype("float32")  # (T, C, h, w)
            block[block == FILL] = np.nan
            nclear[r0 : r0 + win.height] = np.isfinite(block[:, 0]).sum(0)
            with np.errstate(all="ignore"):
                m = np.nanmedian(block, axis=0)
            med[:, r0 : r0 + win.height] = np.where(np.isfinite(m), np.round(m), FILL).astype("int16")
    finally:
        for s in srcs:
            s.close()
    return med, nclear


def step_hls(cfg: dict) -> None:
    import rasterio
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import BoundingBox

    from bridge_lc.data.composites import configure_earthdata_gdal

    configure_earthdata_gdal()
    im = cfg["imagery"]
    for site in cfg["sites"]:
        d = site_dirs(cfg, site["id"])
        img = d / f"hls_{im['year']}.tif"
        if img.exists():
            print(f"[{site['id']}] {img.name} exists, skipping")
            continue
        bbox = site_bbox(cfg, site)
        gbox = GeoBox.from_bbox(
            BoundingBox(*bbox, crs="EPSG:4326").to_crs(cfg["crs"]), resolution=im["resolution_m"], tight=False
        )
        frames, nclear = [], []
        for name, (s, e) in im["seasons"].items():
            y0 = im["year"] - 1 if s > e else im["year"]
            start, end = f"{y0}-{s}", f"{im['year']}-{e}"
            items = season_scenes(bbox, start, end, im["max_scene_cloud_pct"])
            cache = d / "hls_scenes" / name
            cache.mkdir(parents=True, exist_ok=True)
            print(f"[{site['id']}] {name}: {start} → {end}, {len(items)} granules", flush=True)
            for n, it in enumerate(items, 1):
                path = cache / f"{it.id}.tif"
                if path.exists():
                    continue
                try:
                    cache_scene(it, gbox, im["bands"], path)
                except Exception as e:
                    print(f"    skip {it.id}: {type(e).__name__}", flush=True)
                if n % 10 == 0:
                    print(f"    {n}/{len(items)}", flush=True)
            med, n = median_from_cache(sorted(cache.glob("HLS*.tif")))
            med[:, n < im["min_clear_obs"]] = FILL
            frames.append(med)
            nclear.append(n)
        prof = {
            "driver": "GTiff",
            "height": gbox.shape[0],
            "width": gbox.shape[1],
            "crs": cfg["crs"],
            "transform": gbox.transform,
            "compress": "deflate",
            "tiled": True,
        }
        names = [f"{s}_{b}" for s in im["seasons"] for b in im["bands"]]
        with rasterio.open(img, "w", **prof, count=len(names), dtype="int16", nodata=FILL) as dst:
            dst.write(np.concatenate(frames))
            dst.descriptions = tuple(names)
        nclear_path = d / f"nclear_{im['year']}.tif"
        with rasterio.open(nclear_path, "w", **prof, count=len(nclear), dtype="int16") as dst:
            dst.write(np.stack(nclear))
            dst.descriptions = tuple(im["seasons"])


# --------------------------------------------------------------------------- embed
def step_embed(cfg: dict) -> None:
    import rasterio
    import torch
    from terratorch.registry import BACKBONE_REGISTRY

    from bridge_lc.models.prithvi import PRITHVI_V2_MEAN, PRITHVI_V2_STD

    em, im = cfg["embedding"], cfg["imagery"]
    T, P, tile = len(im["seasons"]), 16, em["tile"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = None
    for site in cfg["sites"]:
        d = site_dirs(cfg, site["id"])
        out = d / "prithvi_embeddings.npz"
        if out.exists():
            print(f"[{site['id']}] embeddings exist, skipping")
            continue
        if model is None:
            model = (
                BACKBONE_REGISTRY.build(
                    em["backbone"], pretrained=True, num_frames=T, bands=im["bands"], coords_encoding=[]
                )
                .eval()
                .to(dev)
            )
        with rasterio.open(d / f"hls_{im['year']}.tif") as src:
            x = src.read().astype("float32")
        C, H, W = len(im["bands"]), x.shape[1], x.shape[2]
        x = x.reshape(T, C, H, W).transpose(1, 0, 2, 3)  # (C, T, H, W)
        valid = (x != FILL).all(axis=(0, 1))
        mean = np.array(PRITHVI_V2_MEAN, "float32")[:, None, None, None]
        std = np.array(PRITHVI_V2_STD, "float32")[:, None, None, None]
        x = np.where(x == FILL, mean, x)
        x = (x - mean) / std
        Hp, Wp = int(np.ceil(H / tile)) * tile, int(np.ceil(W / tile)) * tile
        x = np.pad(x, ((0, 0), (0, 0), (0, Hp - H), (0, Wp - W)))
        valid = np.pad(valid, ((0, Hp - H), (0, Wp - W)))
        gh, gw = Hp // P, Wp // P
        feats = {ly: np.zeros((gh, gw, 1024), "float16") for ly in em["layers"]}
        tiles = [(i, j) for i in range(0, Hp, tile) for j in range(0, Wp, tile)]
        print(f"[{site['id']}] {H}×{W} px → {len(tiles)} tiles on {dev}", flush=True)
        for b in range(0, len(tiles), 8):
            batch = tiles[b : b + 8]
            xb = torch.from_numpy(np.stack([x[:, :, i : i + tile, j : j + tile] for i, j in batch])).to(dev)
            with torch.no_grad():
                outs = model(xb)
            for ly in em["layers"]:
                f = outs[ly][:, 1:, :].reshape(len(batch), T, tile // P, tile // P, -1).mean(1)
                f = f.float().cpu().numpy().astype("float16")
                for k, (i, j) in enumerate(batch):
                    feats[ly][i // P : (i + tile) // P, j // P : (j + tile) // P] = f[k]
        vfrac = valid.reshape(gh, P, gw, P).mean(axis=(1, 3))
        np.savez(out, valid_frac=vfrac, H=H, W=W, **{f"layer{ly}": v for ly, v in feats.items()})


# --------------------------------------------------------------------------- analyze
def rasterize(gdf, column: str, shape, transform, codes: dict) -> np.ndarray:
    from rasterio.features import rasterize as rio_rasterize

    shapes = ((g, codes[v]) for g, v in zip(gdf.geometry, gdf[column], strict=True) if v and g is not None)
    return rio_rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")


def patch_labels(lab: np.ndarray, gh: int, gw: int, P: int = 16):
    """Majority class and its share per 16×16 patch (0 = no reference)."""
    full = np.zeros((gh * P, gw * P), lab.dtype)
    full[: lab.shape[0], : lab.shape[1]] = lab
    blocks = full.reshape(gh, P, gw, P).transpose(0, 2, 1, 3).reshape(gh * gw, P * P)
    maj = np.zeros(gh * gw, "int32")
    pur = np.zeros(gh * gw, "float32")
    for n, row in enumerate(blocks):
        row = row[row > 0]
        if row.size:
            v, c = np.unique(row, return_counts=True)
            maj[n], pur[n] = v[c.argmax()], c.max() / (P * P)
    return maj, pur


def score(clusters: np.ndarray, ref: np.ndarray, min_n: int, f1_thr: float) -> dict:
    """Map each cluster to its majority class; an ecosystem is recognised if its F1 ≥ f1_thr."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    classes, counts = np.unique(ref, return_counts=True)
    scored = classes[counts >= min_n]
    keep = np.isin(ref, scored)
    c, r = clusters[keep], ref[keep]
    cmap = {k: np.bincount(r[c == k]).argmax() for k in np.unique(c)}
    pred = np.array([cmap[k] for k in c])
    per_class = {}
    for e in scored:
        tp = np.sum((pred == e) & (r == e))
        prec = tp / max(np.sum(pred == e), 1)
        rec = tp / max(np.sum(r == e), 1)
        per_class[int(e)] = {"n": int(np.sum(r == e)), "f1": float(2 * prec * rec / max(prec + rec, 1e-9))}
    return {
        "n_scored": int(len(scored)),
        "n_recognised": int(sum(v["f1"] >= f1_thr for v in per_class.values())),
        "nmi": float(normalized_mutual_info_score(r, c)),
        "ari": float(adjusted_rand_score(r, c)),
        "overall_acc": float(np.mean(pred == r)),
        "per_class": per_class,
    }


def step_analyze(cfg: dict) -> None:
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cl, em, im = cfg["clustering"], cfg["embedding"], cfg["imagery"]
    out_root = REPO_ROOT / cfg["output_dir"]
    lc_year = cfg["references"]["compare_land_cover_year"]
    T, C, P = len(im["seasons"]), len(im["bands"]), 16

    # 1. reference rasters on the HLS grid + per-patch tables
    refsets = {"ecosystem": ("ecosystems", "label"), "land_cover": (f"land_cover_{lc_year}", "level2")}
    names = {k: set() for k in refsets}
    for s in cfg["sites"]:
        g = site_dirs(cfg, s["id"]) / "references.gpkg"
        for k, (layer, col) in refsets.items():
            names[k] |= set(gpd.read_file(g, layer=layer)[col].dropna()) - {""}
    codes = {k: {n: i + 1 for i, n in enumerate(sorted(v))} for k, v in names.items()}

    tables = []
    for s in cfg["sites"]:
        d = site_dirs(cfg, s["id"])
        with rasterio.open(d / f"hls_{im['year']}.tif") as src:
            prof, img = src.profile, src.read()
        emb = np.load(d / "prithvi_embeddings.npz")
        gh, gw = emb["valid_frac"].shape
        df = pd.DataFrame(
            {
                "site": s["id"],
                "row": np.repeat(np.arange(gh), gw),
                "col": np.tile(np.arange(gw), gh),
                "valid_frac": emb["valid_frac"].ravel(),
            }
        )
        lc_years = {}
        for k, (layer, col) in refsets.items():
            lab = rasterize(
                gpd.read_file(d / "references.gpkg", layer=layer),
                col,
                img.shape[1:],
                prof["transform"],
                codes[k],
            )
            df[k], df[f"{k}_purity"] = patch_labels(lab, gh, gw)
            lc_years[k] = lab
        # all land-cover years as a multi-band raster (for later multi-year work)
        lc_all = {}
        for y in cfg["references"]["land_cover_years"]:
            gdf = gpd.read_file(d / "references.gpkg", layer=f"land_cover_{y}")
            vals = sorted(set(gdf["level2"]) - {""})
            vcodes = {v: i + 1 for i, v in enumerate(vals)}
            lc_all[y] = (rasterize(gdf, "level2", img.shape[1:], prof["transform"], vcodes), vals)
        p = prof | {"count": len(lc_all), "dtype": "uint8", "nodata": 0}
        with rasterio.open(d / "maate_land_cover_level2_allyears.tif", "w", **p) as dst:
            for b, (y, (arr, vals)) in enumerate(lc_all.items(), 1):
                dst.write(arr.astype("uint8"), b)
                dst.set_band_description(b, f"{y}: " + "; ".join(f"{i + 1}={v}" for i, v in enumerate(vals)))
        p = prof | {"count": 1, "dtype": "uint8", "nodata": 0}
        with rasterio.open(d / "maate_ecosystems.tif", "w", **p) as dst:
            dst.write(lc_years["ecosystem"].astype("uint8"), 1)
        # spectral baseline: patch means of the 12 bands + NDVI per season
        x = img.astype("float32")
        x[x == FILL] = np.nan
        Hp, Wp = gh * P, gw * P
        xp = np.full((x.shape[0], Hp, Wp), np.nan, "float32")
        xp[:, : x.shape[1], : x.shape[2]] = x
        means = np.nanmean(xp.reshape(x.shape[0], gh, P, gw, P), axis=(2, 4)).reshape(x.shape[0], -1).T
        for t in range(T):
            red, nir = means[:, t * C + 2], means[:, t * C + 3]
            means = np.column_stack([means, (nir - red) / (nir + red + 1e-6)])
        df[[f"spec{i}" for i in range(means.shape[1])]] = means
        for ly in em["layers"]:
            df[f"emb{ly}"] = list(emb[f"layer{ly}"].reshape(gh * gw, -1))
        tables.append(df)
    df = pd.concat(tables, ignore_index=True)
    df = df[(df.valid_frac >= em["min_valid_frac"])].reset_index(drop=True)
    df = df[np.isfinite(df.filter(like="spec").to_numpy()).all(1)].reset_index(drop=True)
    print(f"{len(df)} valid patches ({df.groupby('site').size().to_dict()})")

    # 2. clustering + scores
    feats = {f"prithvi_l{ly}": np.stack(df[f"emb{ly}"].to_numpy()).astype("float32") for ly in em["layers"]}
    feats["spectral"] = df.filter(like="spec").to_numpy()
    rows, best = [], {}
    for fname, X in feats.items():
        Z = StandardScaler().fit_transform(X)
        if Z.shape[1] > cl["pca_dims"]:
            Z = PCA(cl["pca_dims"], random_state=cl["seed"]).fit_transform(Z)
        for k in cl["k"]:
            lab = KMeans(k, n_init=4, random_state=cl["seed"]).fit_predict(Z)
            df[f"{fname}_k{k}"] = lab
            for ref in refsets:
                pure = (df[f"{ref}_purity"] >= cl["min_purity"]).to_numpy()
                for site in ["all"] + [s["id"] for s in cfg["sites"]]:
                    m = pure & ((df.site == site).to_numpy() if site != "all" else True)
                    sc = score(
                        lab[m], df[ref].to_numpy()[m], cl["min_patches_per_class"], cl["recognised_f1"]
                    )
                    rows.append(
                        {
                            "features": fname,
                            "k": k,
                            "reference": ref,
                            "site": site,
                            **{a: b for a, b in sc.items() if a != "per_class"},
                        }
                    )
                    if site == "all":
                        key = (fname, ref)
                        if key not in best or sc["n_recognised"] > best[key]["n_recognised"]:
                            best[key] = sc | {"k": k}
    res = pd.DataFrame(rows)
    res.to_csv(out_root / "scores.csv", index=False)
    inv = {k: {i: n for n, i in v.items()} for k, v in codes.items()}
    detail = {
        f"{f}|{r}": v | {"per_class": {inv[r][c]: pc for c, pc in v["per_class"].items()}}
        for (f, r), v in best.items()
    }
    (out_root / "best_per_class.json").write_text(json.dumps(detail, indent=1, ensure_ascii=False))
    df.drop(columns=[c for c in df.columns if c.startswith("emb")]).to_parquet(out_root / "patches.parquet")
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(res[res.site == "all"].drop(columns="site").round(3).to_string(index=False))

    # 3. wetlands: soil points vs clusters (Antisana)
    wet_rows = []
    for s in cfg["sites"]:
        d = site_dirs(cfg, s["id"])
        try:
            pts = gpd.read_file(d / "references.gpkg", layer="soil_points")
        except Exception:
            continue
        with rasterio.open(d / f"hls_{im['year']}.tif") as src:
            rc = [src.index(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y, strict=True)]
        pts["row"], pts["col"] = [r // P for r, _ in rc], [c // P for _, c in rc]
        best_k = best[("prithvi_l23", "ecosystem")]["k"]
        sub = df[df.site == s["id"]].set_index(["row", "col"])
        cols = [
            f"prithvi_l{em['layers'][0]}_k{best_k}",
            f"prithvi_l23_k{best_k}",
            f"spectral_k{best_k}",
            "ecosystem",
        ]
        for col in cols:
            pts[col] = [sub[col].get((r, c), -1) for r, c in zip(pts.row, pts.col, strict=True)]
        pts["ecosystem"] = pts["ecosystem"].map(lambda v: inv["ecosystem"].get(v, "none"))
        wet_rows.append(pts.drop(columns="geometry"))
    if wet_rows:
        wet = pd.concat(wet_rows)
        wet.to_csv(out_root / "soil_points_clusters.csv", index=False)
        print("\nSoil points: cover × in_ramsar\n", pd.crosstab(wet["cover"], wet["in_ramsar"]))
        ccol = [c for c in wet.columns if c.startswith("prithvi_l23_k")][0]
        print(f"\nSoil points: cover × {ccol}\n", pd.crosstab(wet["cover"], wet[ccol]))
        print(
            "\nSoil points: cover × MAATE ecosystem (patch majority)\n",
            pd.crosstab(wet["cover"], wet["ecosystem"]),
        )

    # 4. figures: RGB, MAATE ecosystems, best Prithvi clusters, spectral clusters
    import matplotlib.pyplot as plt

    fdir = out_root / "figures"
    fdir.mkdir(exist_ok=True)
    k = best[("prithvi_l23", "ecosystem")]["k"]
    for s in cfg["sites"]:
        d = site_dirs(cfg, s["id"])
        with rasterio.open(d / f"hls_{im['year']}.tif") as src:
            rgb = src.read([3, 2, 1]).astype("float32")
        rgb[rgb == FILL] = np.nan
        rgb = np.clip(rgb.transpose(1, 2, 0) / 1500, 0, 1)
        sub = df[df.site == s["id"]]
        gh, gw = np.load(d / "prithvi_embeddings.npz")["valid_frac"].shape

        def grid(col, sub=sub, gh=gh, gw=gw):
            g = np.full((gh, gw), np.nan)
            g[sub.row, sub.col] = sub[col]
            return g

        with rasterio.open(d / "maate_ecosystems.tif") as src:
            eco = src.read(1).astype("float32")
        eco[eco == 0] = np.nan
        fig, ax = plt.subplots(1, 4, figsize=(22, 6))
        ax[0].imshow(rgb), ax[0].set_title(f"HLS {im['year']} dry RGB")
        ax[1].imshow(eco, cmap="tab20", interpolation="nearest"), ax[1].set_title("MAATE ecosystems (30 m)")
        ax[2].imshow(grid(f"prithvi_l23_k{k}"), cmap="tab20", interpolation="nearest")
        ax[2].set_title(f"Prithvi last layer, k={k} (480 m)")
        ax[3].imshow(grid(f"spectral_k{k}"), cmap="tab20", interpolation="nearest")
        ax[3].set_title(f"Spectral baseline, k={k} (480 m)")
        for a in ax:
            a.set_axis_off()
        fig.suptitle(s["name"])
        fig.tight_layout()
        fig.savefig(fdir / f"{s['id']}_clusters.png", dpi=110)
        plt.close(fig)
    print(f"\nOutputs in {out_root}")


# --------------------------------------------------------------------------- figures
# Validated categorical palette (dataviz reference, light mode), fixed order; neutral for "no reference".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL, INK, MUTED = "#d9d8d4", "#0b0b0b", "#52514e"
FEATURE_STYLE = {
    "prithvi_l11": ("Prithvi layer 11", PALETTE[0]),
    "prithvi_l23": ("Prithvi layer 23", PALETTE[6]),
    "spectral": ("Spectral baseline", PALETTE[1]),
}
ECO_GROUPS = [  # first match wins; physiognomy / belt, ignoring region words (norte/sur, oriental/occidental)
    ("Páramo grassland", lambda n: n == "HERBAZAL DEL PARAMO"),
    ("Wet / subnival páramo", lambda n: "HUMEDO" in n or "INUNDABLE" in n or "SUBNIVAL" in n),
    ("Páramo & montane shrub", lambda n: n.startswith("ARBUSTAL SIEMPREVERDE")),
    ("Upper montane forest", lambda n: "MONTANO ALTO" in n or n == "BOSQUE SIEMPREVERDE DEL PARAMO"),
    ("Montane & lower forest", lambda n: n.startswith("BOSQUE SIEMPREVERDE")),
    ("Dry inter-Andean valleys", lambda n: "VALLES" in n),
]
LC_GROUPS = [
    ("Native forest", {"BOSQUE NATIVO"}),
    ("Páramo", {"PARAMO"}),
    ("Shrub & herbaceous", {"VEGETACION ARBUSTIVA Y HERBACEA"}),
    ("Agriculture & pasture", {"MOSAICO AGROPECUARIO", "PASTIZAL"}),
    ("Forest plantation", {"PLANTACION FORESTAL"}),
    ("Urban & infrastructure", {"AREA POBLADA", "INFRAESTRUCTURA"}),
    (
        "Bare, glacier, water",
        {"AREA SIN COBERTURA VEGETAL", "GLACIAR", "CUERPO DE AGUA NATURAL", "CUERPO DE AGUA ARTIFICIAL"},
    ),
]


def ref_names(cfg: dict) -> dict:
    """Code → name per reference, same coding as step_analyze (sorted names over both sites)."""
    import geopandas as gpd

    lc_year = cfg["references"]["compare_land_cover_year"]
    refsets = {"ecosystem": ("ecosystems", "label"), "land_cover": (f"land_cover_{lc_year}", "level2")}
    names = {k: set() for k in refsets}
    for s in cfg["sites"]:
        g = site_dirs(cfg, s["id"]) / "references.gpkg"
        for k, (layer, col) in refsets.items():
            names[k] |= set(gpd.read_file(g, layer=layer)[col].dropna()) - {""}
    return {k: {i + 1: n for i, n in enumerate(sorted(v))} for k, v in names.items()}


def group_code(name: str, ref: str) -> int:
    if ref == "ecosystem":
        return next((i + 1 for i, (_, f) in enumerate(ECO_GROUPS) if f(name)), 0)
    return next((i + 1 for i, (_, members) in enumerate(LC_GROUPS) if name in members), 0)


def map_clusters(clusters: np.ndarray, ref: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Give every cluster the majority reference group of its pure patches (``fit``)."""
    out = np.zeros_like(ref)
    for c in np.unique(clusters):
        m = (clusters == c) & fit & (ref > 0)
        out[clusters == c] = np.bincount(ref[m]).argmax() if m.any() else 0
    return out


def step_figures(cfg: dict) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import rasterio
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import normalized_mutual_info_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    cl, em = cfg["clustering"], cfg["embedding"]
    out_root = REPO_ROOT / cfg["output_dir"]
    fdir = out_root / "figures"
    fdir.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    df = pd.read_parquet(out_root / "patches.parquet")
    names = ref_names(cfg)
    groups = {"ecosystem": [g for g, _ in ECO_GROUPS], "land_cover": [g for g, _ in LC_GROUPS]}
    for ref in ("ecosystem", "land_cover"):
        df[f"{ref}_grp"] = [group_code(names[ref].get(c, ""), ref) for c in df[ref]]
    pure = {
        r: ((df[f"{r}_purity"] >= cl["min_purity"]) & (df[f"{r}_grp"] > 0)).to_numpy()
        for r in ("ecosystem", "land_cover")
    }

    # features (same as step_analyze)
    feats = {"spectral": df.filter(like="spec").to_numpy()}
    for ly in em["layers"]:
        parts = []
        for s in cfg["sites"]:
            e = np.load(site_dirs(cfg, s["id"]) / "prithvi_embeddings.npz")[f"layer{ly}"]
            sub = df[df.site == s["id"]]
            parts.append(e[sub.row.to_numpy(), sub.col.to_numpy()].astype("float32"))
        feats[f"prithvi_l{ly}"] = np.concatenate(parts)
    Z = {}
    for f, X in feats.items():
        z = StandardScaler().fit_transform(X)
        Z[f] = (
            PCA(cl["pca_dims"], random_state=cl["seed"]).fit_transform(z)
            if z.shape[1] > cl["pca_dims"]
            else z
        )

    # 1. elbow / silhouette / agreement vs k
    ks = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30]
    rng = np.random.default_rng(cl["seed"])
    sil_idx = rng.choice(len(df), min(6000, len(df)), replace=False)
    rows, labels = [], {}
    for f, z in Z.items():
        for k in ks:
            km = KMeans(k, n_init=3, random_state=cl["seed"]).fit(z)
            labels[(f, k)] = km.labels_
            row = {
                "features": f,
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(z[sil_idx], km.labels_[sil_idx]),
            }
            for r in ("ecosystem", "land_cover"):
                row[f"nmi_{r}"] = normalized_mutual_info_score(
                    df[f"{r}_grp"].to_numpy()[pure[r]], km.labels_[pure[r]]
                )
            rows.append(row)
            print(f"  {f} k={k} sil={row['silhouette']:.3f}", flush=True)
    kdf = pd.DataFrame(rows)
    kdf.to_csv(out_root / "k_sweep.csv", index=False)

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    panels = [
        ("inertia", "Inertia (share of k = 2)", "Elbow: within-cluster variance"),
        ("silhouette", "Silhouette (6,000-patch sample)", "Cluster separation"),
        ("nmi_ecosystem", "NMI vs MAATE ecosystem groups", "Agreement: ecosystem groups (6)"),
        ("nmi_land_cover", "NMI vs MAATE land cover 2022", "Agreement: land-cover groups (7)"),
    ]
    for a, (col, ylab, title) in zip(ax, panels, strict=True):
        for f, (lab, color) in FEATURE_STYLE.items():
            d = kdf[kdf.features == f]
            y = d[col] / d[col].iloc[0] if col == "inertia" else d[col]
            a.plot(d.k, y, color=color, lw=2, marker="o", ms=4, label=lab)
        a.set_xlabel("k (clusters)")
        a.set_ylabel(ylab)
        a.set_title(title, loc="left", fontsize=11, color=INK)
        a.grid(axis="y", color="#ecebe8", lw=0.8)
    ax[0].legend(frameon=False)
    fig.suptitle(
        "EXP003 · k-means on 480 m patches, Antisana + Zhurucay (HLS 2023, dry + wet)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(fdir / "k_sweep.png", dpi=130)
    plt.close(fig)

    # 2. per-group F1 at a common k
    k_map = 16
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    f1_rows = []
    for a, ref in zip(ax, ("ecosystem", "land_cover"), strict=True):
        r, m = df[f"{ref}_grp"].to_numpy(), pure[ref]
        gnames = groups[ref]
        y = np.arange(len(gnames))
        for j, f in enumerate(("prithvi_l23", "spectral")):
            pred = map_clusters(labels[(f, k_map)], r, m)
            f1 = []
            for g in range(1, len(gnames) + 1):
                tp = np.sum((pred[m] == g) & (r[m] == g))
                prec, rec = tp / max(np.sum(pred[m] == g), 1), tp / max(np.sum(r[m] == g), 1)
                f1.append(2 * prec * rec / max(prec + rec, 1e-9))
                f1_rows.append(
                    {
                        "reference": ref,
                        "group": gnames[g - 1],
                        "features": f,
                        "k": k_map,
                        "f1": f1[-1],
                        "n_patches": int(np.sum(r[m] == g)),
                    }
                )
            lab, color = FEATURE_STYLE[f]
            a.barh(y + (j - 0.5) * 0.38, f1, height=0.36, color=color, label=lab)
        a.axvline(0.5, color=MUTED, lw=1, ls="--")
        a.text(0.51, -0.62, "recognised ≥ 0.5", color=MUTED, fontsize=9)
        counts = [int(np.sum(r[m] == g)) for g in range(1, len(gnames) + 1)]
        a.set_yticks(y, [f"{g}  (n={c:,})" for g, c in zip(gnames, counts, strict=True)])
        a.invert_yaxis()
        a.set_xlim(0, 1)
        a.set_xlabel(f"F1 (clusters mapped to majority group, k = {k_map})")
        a.set_title(
            "MAATE ecosystems, grouped" if ref == "ecosystem" else "MAATE land cover 2022, grouped",
            loc="left",
            fontsize=11,
            color=INK,
        )
    ax[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(fdir / "f1_by_group.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(f1_rows).to_csv(out_root / "f1_by_group.csv", index=False)

    # 3. comparison maps per site and reference
    for ref in ("ecosystem", "land_cover"):
        gnames = groups[ref]
        cmap = ListedColormap([NEUTRAL] + PALETTE[: len(gnames)])
        agree_cols = [
            "#3d3d3a",
            PALETTE[7],
            "#8a6d3b",
            "#c9c8c3",
        ]  # both, only Prithvi, only spectral, neither
        agree_cmap = ListedColormap(["#ffffff", *agree_cols])
        r_all, m_all = df[f"{ref}_grp"].to_numpy(), pure[ref]
        mapped = {f: map_clusters(labels[(f, k_map)], r_all, m_all) for f in ("prithvi_l23", "spectral")}
        for s in cfg["sites"]:
            d = site_dirs(cfg, s["id"])
            sel = (df.site == s["id"]).to_numpy()
            gh, gw = np.load(d / "prithvi_embeddings.npz")["valid_frac"].shape
            # 30 m reference, grouped
            if ref == "ecosystem":
                with rasterio.open(d / "maate_ecosystems.tif") as src:
                    raw = src.read(1)
                lut = np.array([0] + [group_code(names[ref][c], ref) for c in range(1, max(names[ref]) + 1)])
                ref30 = lut[raw]
            else:
                ref30 = None

            def grid(v, sel=sel, gh=gh, gw=gw):
                g = np.full((gh, gw), np.nan)
                g[df.row[sel], df.col[sel]] = v[sel]
                return g

            agree = np.where(
                r_all == 0,
                np.nan,
                np.select(
                    [
                        (mapped["prithvi_l23"] == r_all) & (mapped["spectral"] == r_all),
                        (mapped["prithvi_l23"] == r_all),
                        (mapped["spectral"] == r_all),
                    ],
                    [1, 2, 3],
                    4,
                ),
            )
            refgrid = grid(r_all.astype(float))
            H30, W30 = ref30.shape if ref30 is not None else (gh * 16, gw * 16)
            ext = (0, gw * 16, gh * 16, 0)
            fig, ax = plt.subplots(1, 4, figsize=(20, 5.6))
            show_ref = ref30 if ref30 is not None else None
            if show_ref is not None:
                ax[0].imshow(show_ref, cmap=cmap, vmin=-0.5, vmax=len(gnames) + 0.5, interpolation="nearest")
                ax[0].set_title("MAATE ecosystems, grouped (30 m)", loc="left", fontsize=11)
            else:
                ax[0].imshow(
                    refgrid, cmap=cmap, vmin=-0.5, vmax=len(gnames) + 0.5, interpolation="nearest", extent=ext
                )
                ax[0].set_title(
                    f"MAATE land cover {cfg['references']['compare_land_cover_year']}, "
                    "patch majority (480 m)",
                    loc="left",
                    fontsize=11,
                )
            for a, f in zip(ax[1:3], ("prithvi_l23", "spectral"), strict=True):
                a.imshow(
                    grid(np.where(r_all > 0, mapped[f], 0).astype(float)),
                    cmap=cmap,
                    vmin=-0.5,
                    vmax=len(gnames) + 0.5,
                    interpolation="nearest",
                    extent=ext,
                )
                a.set_title(
                    f"{FEATURE_STYLE[f][0]}, k = {k_map} → matched group (480 m)", loc="left", fontsize=11
                )
            ax[3].imshow(
                grid(agree.astype(float)),
                cmap=agree_cmap,
                vmin=-0.5,
                vmax=4.5,
                interpolation="nearest",
                extent=ext,
            )
            ax[3].set_title("Agreement with MAATE (480 m)", loc="left", fontsize=11)
            for a in ax:
                a.set_xlim(0, W30), a.set_ylim(H30, 0)
                a.set_xticks([]), a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
            share = {
                n: np.mean(agree[sel & ~np.isnan(agree)] == v)
                for v, n in [(1, "both"), (2, "prithvi"), (3, "spectral"), (4, "neither")]
            }
            leg1 = [Patch(color=PALETTE[i], label=g) for i, g in enumerate(gnames)] + [
                Patch(color=NEUTRAL, label="Not mapped / intervened" if ref == "ecosystem" else "No class")
            ]
            leg2 = [
                Patch(color=agree_cols[0], label=f"Both agree ({share['both']:.0%})"),
                Patch(color=agree_cols[1], label=f"Only Prithvi ({share['prithvi']:.0%})"),
                Patch(color=agree_cols[2], label=f"Only spectral ({share['spectral']:.0%})"),
                Patch(color=agree_cols[3], label=f"Neither ({share['neither']:.0%})"),
            ]
            fig.legend(handles=leg1, loc="lower left", ncol=4, frameon=False, bbox_to_anchor=(0.01, 0.0))
            fig.legend(handles=leg2, loc="lower right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.0))
            fig.suptitle(
                f"{s['name']} · MAATE {'ecosystems' if ref == 'ecosystem' else 'land cover'} vs "
                "unsupervised clusters (white = cloud gap or no data)",
                x=0.01,
                ha="left",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0.1, 1, 1))
            fig.savefig(fdir / f"{s['id']}_compare_{ref}.png", dpi=120)
            plt.close(fig)
    print(f"Figures in {fdir}")


STEPS = {
    "refs": step_refs,
    "hls": step_hls,
    "embed": step_embed,
    "analyze": step_analyze,
    "figures": step_figures,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=[*STEPS, "all"])
    args = ap.parse_args()
    cfg = load_yaml(CFG_PATH)
    for name, fn in STEPS.items():
        if args.step in (name, "all"):
            print(f"=== {name}", flush=True)
            fn(cfg)


if __name__ == "__main__":
    main()
