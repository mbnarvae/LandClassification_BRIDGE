"""EXP001 dataset builder: basins → HLS seasonal composites → labels → TerraTorch chips + splits.

    python scripts/exp001_build_dataset.py                       # all steps, all basins
    python scripts/exp001_build_dataset.py --step composites     # only download/composite
    python scripts/exp001_build_dataset.py --basins b01 b02      # subset

Output (``output_dir`` in experiment.yaml):
    stacks/<basin>_image.tif     (T*C bands, frame-major: dry[6 bands], wet[6 bands]; reflectance ×10000)
    stacks/<basin>_nclear.tif    (clear observations per season)
    stacks/<basin>_labels.tif    (BRIDGE legend ids, 0 = nodata; outside basin = 0)
    chips/<basin>_rRRRR_cCCCC_img.tif / _mask.tif
    splits/{train,val,test}.txt  and splits/lobo/<basin>/{train,val,test}.txt
    chips_index.csv              (chip, basin, block, split, valid_frac, class histogram)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bridge_lc.config import REPO_ROOT, load_yaml
from bridge_lc.data.basins import load_basins
from bridge_lc.data.chips import iter_chip_windows

EXP_DIR = REPO_ROOT / "experiments/exp001_prithvi_8basins"
FILL = -9999


def season_dates(season: list[str], year: int) -> tuple[str, str]:
    """('MM-DD','MM-DD') → ISO dates; a window that wraps the new year starts in year-1."""
    start, end = season
    y0 = year - 1 if start > end else year
    return f"{y0}-{start}", f"{year}-{end}"


# --------------------------------------------------------------------------- step 1: composites
def build_composites(cfg: dict, basin, out_dir: Path) -> None:
    import rioxarray  # noqa: F401  (registers .rio)
    import xarray as xr

    from bridge_lc.data.composites import seasonal_composite

    img_path = out_dir / f"{basin.id}_image.tif"
    if img_path.exists():
        print(f"[{basin.id}] composites exist, skipping")
        return
    im = cfg["imagery"]
    frames, nclear = [], []
    for name, season in im["seasons"].items():
        start, end = season_dates(season, im["year"])
        print(f"[{basin.id}] {name}: {start} → {end}")
        comp, n = seasonal_composite(
            basin.bbox,
            basin.utm_crs,
            start,
            end,
            resolution=im["resolution_m"],
            max_cloud_pct=im["max_scene_cloud_pct"],
            bands=im["bands"],
        )
        comp = comp.where(n >= im["min_clear_obs"])
        frames.append(comp.compute())
        nclear.append(n.compute())

    stack = xr.concat(frames, dim="band")  # frame-major band order
    stack = stack.rio.write_crs(basin.utm_crs).rio.clip([basin.geometry], crs="EPSG:4326", drop=False)
    stack = stack.fillna(FILL).astype("int16").rio.write_nodata(FILL)
    names = [f"{s}_{b}" for s in im["seasons"] for b in im["bands"]]
    stack = stack.assign_coords(band=np.arange(1, len(names) + 1)).assign_attrs(long_name=names)
    stack.rio.to_raster(img_path, compress="deflate", tiled=True)
    xr.concat(nclear, dim="band").rio.write_crs(basin.utm_crs).rio.to_raster(
        out_dir / f"{basin.id}_nclear.tif", compress="deflate"
    )


# --------------------------------------------------------------------------- step 2: labels
def build_labels(cfg: dict, basin, out_dir: Path) -> None:
    """Reproject the reference label raster onto the basin image grid (nearest), mask outside basin."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import Resampling, reproject
    from shapely.geometry import mapping
    from shapely.ops import transform as shp_transform

    lab_cfg = cfg["labels"]
    if not lab_cfg.get("path"):
        raise SystemExit("labels.path is not set in experiment.yaml — choose a label source first.")
    from bridge_lc.data.labels import load_mapping, remap
    from bridge_lc.legend import load_legend

    mapping_ = None
    if lab_cfg.get("source") and lab_cfg["source"] != "bridge":
        mapping_ = load_mapping(lab_cfg["source"], load_legend())

    with rasterio.open(out_dir / f"{basin.id}_image.tif") as ref:
        profile, shape, transform, crs = ref.profile, ref.shape, ref.transform, ref.crs
    dst = np.zeros(shape, dtype=np.int32)
    with rasterio.open(REPO_ROOT / lab_cfg["path"]) as src:
        reproject(
            rasterio.band(src, 1),
            dst,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
            dst_nodata=0,
        )
    if mapping_ is not None:
        dst = remap(dst, mapping_)

    import pyproj

    to_utm = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    outside = geometry_mask([mapping(shp_transform(to_utm, basin.geometry))], shape, transform)
    dst[outside] = 0
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_dir / f"{basin.id}_labels.tif", "w", **profile) as out:
        out.write(dst.astype(np.uint8), 1)


# --------------------------------------------------------------------------- step 3: chips
def build_chips(cfg: dict, basin, stacks: Path, chips_dir: Path) -> list[dict]:
    import rasterio
    from rasterio.windows import Window

    c = cfg["chips"]
    rows = []
    with (
        rasterio.open(stacks / f"{basin.id}_image.tif") as img,
        rasterio.open(stacks / f"{basin.id}_labels.tif") as lab,
    ):
        labels = lab.read(1)
        image_valid = (img.read(1) != FILL) & (img.read(img.count) != FILL)
        for r, col in iter_chip_windows(*labels.shape, c["size"], c["stride"]):
            y = labels[r : r + c["size"], col : col + c["size"]]
            ok = image_valid[r : r + c["size"], col : col + c["size"]]
            y = np.where(ok, y, 0)
            valid = float((y != 0).mean())
            if valid < c["min_valid_frac"]:
                continue
            win = Window(col, r, c["size"], c["size"])
            name = f"{basin.id}_r{r:05d}_c{col:05d}"
            prof = img.profile | {
                "width": c["size"],
                "height": c["size"],
                "transform": img.window_transform(win),
                "compress": "deflate",
                "tiled": False,
            }
            with rasterio.open(chips_dir / f"{name}_img.tif", "w", **prof) as out:
                out.write(img.read(window=win))
            with rasterio.open(
                chips_dir / f"{name}_mask.tif", "w", **(prof | {"count": 1, "dtype": "uint8", "nodata": 0})
            ) as out:
                out.write(y.astype(np.uint8), 1)
            x0, y0 = img.window_transform(win) * (0, 0)
            hist = np.bincount(y.ravel(), minlength=256)
            rows.append(
                {
                    "chip": name,
                    "basin": basin.id,
                    "x": x0,
                    "y": y0,
                    "valid_frac": valid,
                    **{f"n_{k}": int(v) for k, v in enumerate(hist) if v and k},
                }
            )
    print(f"[{basin.id}] {len(rows)} chips")
    return rows


# --------------------------------------------------------------------------- step 4: splits
def assign_splits(index, cfg: dict):
    """Spatial blocks *within each basin* → train/val/test, so every basin is represented."""
    from bridge_lc.evaluation.spatial_cv import block_ids

    s = cfg["split"]
    rng = np.random.default_rng(s["seed"])
    index["block"] = -1
    index["split"] = ""
    for _, g in index.groupby("basin"):
        blocks = block_ids(g["x"].to_numpy(), g["y"].to_numpy(), s["block_size_m"])
        index.loc[g.index, "block"] = blocks
        uniq = rng.permutation(np.unique(blocks))
        n_val = max(1, round(len(uniq) * s["fractions"]["val"]))
        n_test = max(1, round(len(uniq) * s["fractions"]["test"]))
        if len(uniq) < 3:
            print(f"WARNING: basin {g['basin'].iloc[0]} has only {len(uniq)} blocks → all train")
            n_val = n_test = 0
        split_of = {b: "val" for b in uniq[:n_val]} | {b: "test" for b in uniq[n_val : n_val + n_test]}
        index.loc[g.index, "split"] = [split_of.get(b, "train") for b in blocks]
    return index


def write_splits(index, out: Path, lobo: bool) -> None:
    sp = out / "splits"
    sp.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        (sp / f"{name}.txt").write_text("\n".join(index.loc[index.split == name, "chip"]) + "\n")
    if lobo:
        for b in index.basin.unique():
            d = sp / "lobo" / b
            d.mkdir(parents=True, exist_ok=True)
            held = index.basin == b
            # Held-out basin = test; validation = the in-basin val blocks of the other basins.
            (d / "test.txt").write_text("\n".join(index.loc[held, "chip"]) + "\n")
            (d / "val.txt").write_text("\n".join(index.loc[~held & (index.split == "val"), "chip"]) + "\n")
            (d / "train.txt").write_text("\n".join(index.loc[~held & (index.split != "val"), "chip"]) + "\n")


def class_stats(index, out: Path) -> None:
    """Pixel counts per class/split + inverse-sqrt-frequency weights for the loss."""
    cols = sorted((c for c in index.columns if c.startswith("n_")), key=lambda c: int(c[2:]))
    counts = index.groupby("split")[cols].sum().T.fillna(0).astype(int)
    counts.to_csv(out / "class_counts.csv")
    train = counts.get("train", counts.sum(1)).astype(float)
    from bridge_lc.legend import load_legend

    n_classes = load_legend().num_classes + 1  # must match the model's num_classes, even if a class is absent
    w = np.zeros(n_classes)
    for c, n in train.items():
        w[int(c[2:])] = 1 / np.sqrt(n) if n else 0.0
    w = w / w[w > 0].mean()
    missing = [k for k in range(1, n_classes) if w[k] == 0]
    if missing:
        print(f"NOTE: classes {missing} have no training pixels (weight 0)")
    (out / "class_weights.txt").write_text(", ".join(f"{v:.4f}" for v in w) + "\n")
    print(counts)
    print("class weights (index 0 = nodata, ignored):", np.round(w, 3).tolist())


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(EXP_DIR / "experiment.yaml"))
    ap.add_argument("--step", choices=["all", "composites", "labels", "chips"], default="all")
    ap.add_argument("--basins", nargs="*", help="subset of basin ids")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    b = cfg["basins"]
    basins = load_basins(b["path"], b["id_field"], b["buffer_m"], b["expected"])
    if args.basins:
        basins = [x for x in basins if x.id in args.basins]
    out = REPO_ROOT / cfg["output_dir"]
    stacks, chips_dir = out / "stacks", out / "chips"
    stacks.mkdir(parents=True, exist_ok=True)
    chips_dir.mkdir(parents=True, exist_ok=True)

    if args.step in ("all", "composites"):
        for basin in basins:
            build_composites(cfg, basin, stacks)
    if args.step in ("all", "labels"):
        for basin in basins:
            build_labels(cfg, basin, stacks)
    if args.step in ("all", "chips"):
        rows = [r for basin in basins for r in build_chips(cfg, basin, stacks, chips_dir)]
        index = assign_splits(pd.DataFrame(rows), cfg)
        index.to_csv(out / "chips_index.csv", index=False)
        write_splits(index, out, cfg["split"]["lobo"])
        class_stats(index, out)


if __name__ == "__main__":
    main()
