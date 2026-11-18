"""EXP004: the EXP003 smoke test (Antisana + Zhurucay, HLS 2023 dry + wet, frozen encoder → k-means vs MAATE)
repeated for several foundation models. Built to run on a Colab GPU; works on CPU / Apple MPS too.

Inputs are the EXP003 per-site folders (``<inputs>/<site>/hls_2023.tif`` and ``references.gpkg``), so the
imagery, the reference rasters, the valid patches and the scoring are identical to EXP003.

Steps (each skips work already on disk, so a dropped Colab session resumes where it stopped):
  embed    patch embeddings (16 px = 480 m) per model and site → <out>/<site>/emb_<model>.npz
  analyze  k-means on every model layer and on the spectral baseline, EXP003 scores → scores.csv, summary.csv
  figures  model comparison, per-group F1 and per-site maps → <out>/figures/

python scripts/exp004_multi_fm_smoke.py all
python scripts/exp004_multi_fm_smoke.py embed --models prithvi_eo2_300_tl,dofa_base
python scripts/exp004_multi_fm_smoke.py all --inputs /content/drive/MyDrive/BRIDGE_colab/exp003_inputs \
    --out /content/drive/MyDrive/BRIDGE_colab/exp004
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from bridge_lc.config import REPO_ROOT, load_yaml

CFG_PATH = REPO_ROOT / "experiments/exp004_ec_sites_multi_fm/experiment.yaml"


def _load_exp003():
    spec = importlib.util.spec_from_file_location("exp003", REPO_ROOT / "scripts/exp003_ec_sites_smoke.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E3 = _load_exp003()  # scoring, rasterisation, class groups and palette are shared with EXP003
FILL = E3.FILL
REFSETS = ("ecosystem", "land_cover")


def load_cfg(inputs: str | None = None, out: str | None = None) -> dict:
    cfg = load_yaml(CFG_PATH)
    base = load_yaml(cfg["base"])
    merged = base | cfg
    merged["embedding"] = base["embedding"] | cfg["embedding"]
    merged["inputs_dir"] = Path(inputs).expanduser() if inputs else REPO_ROOT / cfg["inputs_dir"]
    merged["output_dir"] = Path(out).expanduser() if out else REPO_ROOT / cfg["output_dir"]
    return merged


def in_dir(cfg: dict, site: dict) -> Path:
    return cfg["inputs_dir"] / site["id"]


def out_dir(cfg: dict, site: dict | None = None) -> Path:
    d = cfg["output_dir"] / site["id"] if site else cfg["output_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def padded_grid(H: int, W: int, tile: int, P: int = 16) -> tuple[int, int, int, int]:
    """Tile-padded image size and the matching patch grid (same padding as EXP003)."""
    Hp, Wp = int(np.ceil(H / tile)) * tile, int(np.ceil(W / tile)) * tile
    return Hp, Wp, Hp // P, Wp // P


# --------------------------------------------------------------------------- embed
def step_embed(cfg: dict, models: list[str]) -> None:
    import pandas as pd
    import rasterio
    import torch

    from bridge_lc.models.embeddings import PATCH, PatchEmbedder

    em, im = cfg["embedding"], cfg["imagery"]
    T, C, tile, bs = len(im["seasons"]), len(im["bands"]), em["tile"], em["batch_size"]
    dev = pick_device()
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    timing = []
    for name in models:
        todo = [s for s in cfg["sites"] if not (out_dir(cfg, s) / f"emb_{name}.npz").exists()]
        if not todo:
            print(f"[{name}] embeddings exist for all sites, skipping", flush=True)
            continue
        t0 = time.time()
        model = PatchEmbedder(name, n_frames=T, bands=im["bands"], device=dev)
        print(f"[{name}] loaded in {time.time() - t0:.0f} s on {dev}, layers {model.layers}", flush=True)
        for site in todo:
            t1 = time.time()
            with rasterio.open(in_dir(cfg, site) / f"hls_{im['year']}.tif") as src:
                x = src.read().astype("float32")
            H, W = x.shape[1:]
            x = x.reshape(T, C, H, W).transpose(1, 0, 2, 3)  # (C, T, H, W)
            valid = (x != FILL).all(axis=(0, 1))
            x = model.standardise(x, FILL)  # cloud gaps → band mean, as in EXP003
            Hp, Wp, gh, gw = padded_grid(H, W, tile, PATCH)
            x = np.pad(x, ((0, 0), (0, 0), (0, Hp - H), (0, Wp - W)))
            valid = np.pad(valid, ((0, Hp - H), (0, Wp - W)))
            tiles = [(i, j) for i in range(0, Hp, tile) for j in range(0, Wp, tile)]
            feats: dict = {}
            for b in range(0, len(tiles), bs):
                batch = tiles[b : b + bs]
                xb = torch.from_numpy(np.stack([x[:, :, i : i + tile, j : j + tile] for i, j in batch]))
                outs = model(xb.to(dev), amp=em["amp"])
                for ly, f in outs.items():
                    f = f.cpu().numpy().astype("float16")
                    if ly not in feats:
                        feats[ly] = np.zeros((gh, gw, f.shape[-1]), "float16")
                    for k, (i, j) in enumerate(batch):
                        feats[ly][i // PATCH : (i + tile) // PATCH, j // PATCH : (j + tile) // PATCH] = f[k]
            vfrac = valid.reshape(gh, PATCH, gw, PATCH).mean(axis=(1, 3))
            path = out_dir(cfg, site) / f"emb_{name}.npz"
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, valid_frac=vfrac, H=H, W=W, **{f"layer{ly}": v for ly, v in feats.items()})
            tmp.rename(path)
            dt = time.time() - t1
            dims = {ly: v.shape[-1] for ly, v in feats.items()}
            print(
                f"[{name}] {site['id']}: {H}×{W} px, {len(tiles)} tiles, dims {dims}, {dt:.0f} s", flush=True
            )
            timing.append({"model": name, "site": site["id"], "tiles": len(tiles), "seconds": round(dt, 1)})
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    if timing:
        tpath = out_dir(cfg) / "embed_timing.csv"
        old = pd.read_csv(tpath) if tpath.exists() else None
        new = pd.DataFrame(timing).assign(device=dev)
        (pd.concat([old, new]) if old is not None else new).to_csv(tpath, index=False)


# --------------------------------------------------------------------------- analyze
def ref_codes(cfg: dict) -> dict:
    """Name → code per reference, sorted over both sites (same coding as EXP003)."""
    import geopandas as gpd

    lc_year = cfg["references"]["compare_land_cover_year"]
    layers = {"ecosystem": ("ecosystems", "label"), "land_cover": (f"land_cover_{lc_year}", "level2")}
    names = {k: set() for k in layers}
    for s in cfg["sites"]:
        for k, (layer, col) in layers.items():
            g = gpd.read_file(in_dir(cfg, s) / "references.gpkg", layer=layer)
            names[k] |= set(g[col].dropna()) - {""}
    return {k: {n: i + 1 for i, n in enumerate(sorted(v))} for k, v in names.items()}, layers


def patch_table(cfg: dict):
    """One row per 480 m patch: site, grid cell, valid share, MAATE majority + purity, spectral features."""
    import geopandas as gpd
    import pandas as pd
    import rasterio

    im, em = cfg["imagery"], cfg["embedding"]
    T, C, P = len(im["seasons"]), len(im["bands"]), 16
    codes, layers = ref_codes(cfg)
    tables = []
    for s in cfg["sites"]:
        d = in_dir(cfg, s)
        with rasterio.open(d / f"hls_{im['year']}.tif") as src:
            prof, img = src.profile, src.read()
        H, W = img.shape[1:]
        Hp, Wp, gh, gw = padded_grid(H, W, em["tile"], P)
        valid = np.pad((img != FILL).all(0), ((0, Hp - H), (0, Wp - W)))
        df = pd.DataFrame(
            {
                "site": s["id"],
                "row": np.repeat(np.arange(gh), gw),
                "col": np.tile(np.arange(gw), gh),
                "valid_frac": valid.reshape(gh, P, gw, P).mean(axis=(1, 3)).ravel(),
            }
        )
        for k, (layer, col) in layers.items():
            gdf = gpd.read_file(d / "references.gpkg", layer=layer)
            lab = E3.rasterize(gdf, col, (H, W), prof["transform"], codes[k])
            df[k], df[f"{k}_purity"] = E3.patch_labels(lab, gh, gw)
        x = img.astype("float32")
        x[x == FILL] = np.nan
        xp = np.full((x.shape[0], Hp, Wp), np.nan, "float32")
        xp[:, :H, :W] = x
        with np.errstate(all="ignore"):
            means = np.nanmean(xp.reshape(x.shape[0], gh, P, gw, P), axis=(2, 4)).reshape(x.shape[0], -1).T
        for t in range(T):
            red, nir = means[:, t * C + 2], means[:, t * C + 3]
            means = np.column_stack([means, (nir - red) / (nir + red + 1e-6)])
        df[[f"spec{i}" for i in range(means.shape[1])]] = means
        tables.append(df)
    df = pd.concat(tables, ignore_index=True)
    df = df[df.valid_frac >= em["min_valid_frac"]].reset_index(drop=True)
    df = df[np.isfinite(df.filter(like="spec").to_numpy()).all(1)].reset_index(drop=True)
    return df, codes


def embedded_models(cfg: dict) -> list[str]:
    """Models with embeddings for every site, in config order (then any others found)."""
    found = None
    for s in cfg["sites"]:
        names = {p.stem[4:] for p in out_dir(cfg, s).glob("emb_*.npz") if not p.stem.endswith(".tmp")}
        found = names if found is None else found & names
    order = cfg["models"] + cfg.get("optional_models", [])
    return [m for m in order if m in found] + sorted(found - set(order))


def feature_sets(cfg: dict, df) -> dict:
    """``{feature name: (model, depth, matrix)}`` for the spectral baseline and every embedded model layer."""
    sets = {"spectral": ("spectral", "baseline", df.filter(like="spec").to_numpy().astype("float32"))}
    for m in embedded_models(cfg):
        per_site = {s["id"]: np.load(out_dir(cfg, s) / f"emb_{m}.npz") for s in cfg["sites"]}
        lys = sorted(int(k[5:]) for k in next(iter(per_site.values())).files if k.startswith("layer"))
        for depth, ly in zip(("mid", "last"), lys, strict=False):
            parts = []
            for s in cfg["sites"]:
                sub = df[df.site == s["id"]]
                parts.append(per_site[s["id"]][f"layer{ly}"][sub.row.to_numpy(), sub.col.to_numpy()])
            sets[f"{m}_l{ly}"] = (m, depth, np.concatenate(parts).astype("float32"))
    return sets


def reduce(X: np.ndarray, dims: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)
    return PCA(dims, random_state=seed).fit_transform(Z) if Z.shape[1] > dims else Z


def step_analyze(cfg: dict) -> None:
    import pandas as pd
    from sklearn.cluster import KMeans

    cl = cfg["clustering"]
    out = out_dir(cfg)
    df, codes = patch_table(cfg)
    print(f"{len(df)} valid patches ({df.groupby('site').size().to_dict()})", flush=True)
    sets = feature_sets(cfg, df)
    print(f"feature sets: {list(sets)}", flush=True)
    rows, best = [], {}
    for fname, (model, depth, X) in sets.items():
        Z = reduce(X, cl["pca_dims"], cl["seed"])
        for k in cl["k"]:
            lab = KMeans(k, n_init=4, random_state=cl["seed"]).fit_predict(Z)
            df[f"{fname}_k{k}"] = lab
            for ref in REFSETS:
                pure = (df[f"{ref}_purity"] >= cl["min_purity"]).to_numpy()
                for site in ["all"] + [s["id"] for s in cfg["sites"]]:
                    m = pure & ((df.site == site).to_numpy() if site != "all" else True)
                    sc = E3.score(
                        lab[m], df[ref].to_numpy()[m], cl["min_patches_per_class"], cl["recognised_f1"]
                    )
                    rows.append(
                        {
                            "features": fname,
                            "model": model,
                            "depth": depth,
                            "dim": X.shape[1],
                            "k": k,
                            "reference": ref,
                            "site": site,
                            **{a: b for a, b in sc.items() if a != "per_class"},
                        }
                    )
                    if site == "all" and (
                        (fname, ref) not in best or sc["n_recognised"] > best[(fname, ref)]["n_recognised"]
                    ):
                        best[(fname, ref)] = sc | {"k": k}
        print(f"  {fname}: done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "scores.csv", index=False)
    inv = {k: {i: n for n, i in v.items()} for k, v in codes.items()}
    detail = {
        f"{f}|{r}": v | {"per_class": {inv[r][c]: pc for c, pc in v["per_class"].items()}}
        for (f, r), v in best.items()
    }
    (out / "best_per_class.json").write_text(json.dumps(detail, indent=1, ensure_ascii=False))
    df.to_parquet(out / "patches.parquet")
    summary = summarise(res)
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 180, "display.max_rows", 200):
        print(summary.to_string(index=False))
    print(f"\nOutputs in {out}")


def summarise(res):
    """Best over k per feature set and reference (both sites pooled)."""
    a = res[res.site == "all"]
    idx = a.sort_values(["n_recognised", "nmi"], ascending=False).groupby(["features", "reference"]).head(1)
    s = idx[["model", "depth", "features", "dim", "reference", "k", "n_recognised", "n_scored", "nmi"]]
    s = s.rename(columns={"k": "best_k", "nmi": "nmi_at_best_k"})
    s = s.merge(
        a.groupby(["features", "reference"]).nmi.max().rename("max_nmi").reset_index(),
        on=["features", "reference"],
    )
    return s.sort_values(["reference", "n_recognised", "max_nmi"], ascending=[True, False, False]).round(3)


# --------------------------------------------------------------------------- figures
def step_figures(cfg: dict) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import rasterio
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cl, em = cfg["clustering"], cfg["embedding"]
    out = out_dir(cfg)
    fdir = out / "figures"
    fdir.mkdir(exist_ok=True)
    PAL, NEUTRAL, INK, MUTED = E3.PALETTE, E3.NEUTRAL, E3.INK, E3.MUTED
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
    res = pd.read_csv(out / "scores.csv")
    df = pd.read_parquet(out / "patches.parquet")
    models = [m for m in dict.fromkeys(res.model) if m != "spectral"]
    depth_style = {"mid": ("Mid layer", PAL[0]), "last": ("Last layer", PAL[6])}
    a = res[res.site == "all"]

    # 1. model comparison: best over k, both sites pooled
    panels = [
        ("ecosystem", "n_recognised", "MAATE ecosystems recognised (F1 ≥ 0.5)"),
        ("land_cover", "n_recognised", "MAATE land-cover classes recognised"),
        ("ecosystem", "nmi", "NMI vs MAATE ecosystems"),
        ("land_cover", "nmi", "NMI vs MAATE land cover 2022"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 0.45 * len(models) + 2.2), sharey=True)
    y = np.arange(len(models))
    for ax, (ref, col, title) in zip(axes, panels, strict=True):
        r = a[a.reference == ref]
        for j, (depth, (lab, color)) in enumerate(depth_style.items()):
            vals = [r[(r.model == m) & (r.depth == depth)][col].max() for m in models]
            ax.scatter(vals, y + (j - 0.5) * 0.22, s=46, color=color, label=lab, zorder=3, ec="white", lw=1)
        base = r[r.model == "spectral"][col].max()
        ax.axvline(base, color=MUTED, ls="--", lw=1.2, zorder=2)
        ax.grid(axis="x", color="#ecebe8", lw=0.8)
        n_sc = int(r.n_scored.max())
        ax.set_title(title + (f" (of {n_sc})" if col == "n_recognised" else ""), loc="left", fontsize=10.5)
        ax.set_xlabel("best over k ∈ " + ", ".join(map(str, cl["k"])))
        if col == "n_recognised":
            ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].set_yticks(y, models)
    axes[0].invert_yaxis()
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=7, label=lab) for lab, c in depth_style.values()]
    handles.append(Line2D([], [], color=MUTED, ls="--", label="Spectral baseline (12 bands + 2 NDVI)"))
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle(
        "EXP004 · frozen encoders → k-means on 480 m patches, Antisana + Zhurucay (HLS 2023, dry + wet)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(fdir / "model_comparison.png", dpi=130)
    plt.close(fig)

    # 2. per-group F1 at k_map (last layer of each model + spectral) as a heatmap
    k_map = em["k_map"]
    codes, _ = ref_codes(cfg)
    names = {r: {i: n for n, i in v.items()} for r, v in codes.items()}
    groups = {"ecosystem": [g for g, _ in E3.ECO_GROUPS], "land_cover": [g for g, _ in E3.LC_GROUPS]}
    for ref in REFSETS:
        df[f"{ref}_grp"] = [E3.group_code(names[ref].get(c, ""), ref) for c in df[ref]]
    pure = {r: ((df[f"{r}_purity"] >= cl["min_purity"]) & (df[f"{r}_grp"] > 0)).to_numpy() for r in REFSETS}
    last = {m: res[(res.model == m) & (res.depth == "last")].features.iloc[0] for m in models}
    rows_f = {**last, "spectral": "spectral"}
    f1_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.42 * len(rows_f) + 2.6))
    for ax, ref in zip(axes, REFSETS, strict=True):
        r, m = df[f"{ref}_grp"].to_numpy(), pure[ref]
        mat = np.zeros((len(rows_f), len(groups[ref])))
        for i, fname in enumerate(rows_f.values()):
            pred = E3.map_clusters(df[f"{fname}_k{k_map}"].to_numpy(), r, m)
            for g in range(1, len(groups[ref]) + 1):
                tp = np.sum((pred[m] == g) & (r[m] == g))
                prec, rec = tp / max(np.sum(pred[m] == g), 1), tp / max(np.sum(r[m] == g), 1)
                mat[i, g - 1] = 2 * prec * rec / max(prec + rec, 1e-9)
                f1_rows.append(
                    {"reference": ref, "group": groups[ref][g - 1], "features": fname, "f1": mat[i, g - 1]}
                )
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.6 else INK
            )
        counts = [int(np.sum(r[m] == g)) for g in range(1, len(groups[ref]) + 1)]
        ax.set_xticks(
            range(len(groups[ref])),
            [f"{g}\n(n={c:,})" for g, c in zip(groups[ref], counts, strict=True)],
            rotation=30,
            ha="right",
            fontsize=8.5,
        )
        ax.set_yticks(range(len(rows_f)), list(rows_f))
        ax.set_title(
            ("MAATE ecosystems" if ref == "ecosystem" else "MAATE land cover 2022")
            + f", grouped · F1 at k = {k_map}",
            loc="left",
            fontsize=11,
        )
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(fdir / "f1_by_group.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(f1_rows).to_csv(out / "f1_by_group.csv", index=False)

    # 3. per-site maps: MAATE groups vs each model's last layer (k_map clusters → matched group)
    for ref in REFSETS:
        gnames = groups[ref]
        cmap = ListedColormap([NEUTRAL] + PAL[: len(gnames)])
        r_all, m_all = df[f"{ref}_grp"].to_numpy(), pure[ref]
        mapped = {
            lab: E3.map_clusters(df[f"{f}_k{k_map}"].to_numpy(), r_all, m_all) for lab, f in rows_f.items()
        }
        for s in cfg["sites"]:
            sel = (df.site == s["id"]).to_numpy()
            with rasterio.open(in_dir(cfg, s) / f"hls_{cfg['imagery']['year']}.tif") as src:
                H, W = src.height, src.width
            _, _, gh, gw = padded_grid(H, W, em["tile"])

            def grid(v, sel=sel, gh=gh, gw=gw):
                g = np.full((gh, gw), np.nan)
                g[df.row[sel], df.col[sel]] = v[sel]
                return g

            panels = [("MAATE (patch majority)", r_all.astype(float))]
            panels += [(lab, np.where(r_all > 0, v, 0).astype(float)) for lab, v in mapped.items()]
            ncol = 4
            nrow = int(np.ceil(len(panels) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow + 0.8))
            for ax in axes.ravel():
                ax.set_axis_off()
            for ax, (lab, v) in zip(axes.ravel(), panels, strict=False):
                ax.imshow(grid(v), cmap=cmap, vmin=-0.5, vmax=len(gnames) + 0.5, interpolation="nearest")
                acc = (
                    ""
                    if lab.startswith("MAATE")
                    else f"  ·  agrees {np.mean(v[sel & (r_all > 0)] == r_all[sel & (r_all > 0)]):.0%}"
                )
                ax.set_title(lab + acc, loc="left", fontsize=10)
            leg = [Patch(color=PAL[i], label=g) for i, g in enumerate(gnames)]
            leg.append(Patch(color=NEUTRAL, label="No reference class"))
            fig.legend(handles=leg, loc="lower left", ncol=4, frameon=False, bbox_to_anchor=(0.01, 0.0))
            fig.suptitle(
                f"{s['name']} · MAATE {'ecosystems' if ref == 'ecosystem' else 'land cover'} vs k = {k_map} "
                "clusters matched to groups (480 m; white = cloud gap)",
                x=0.01,
                ha="left",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            fig.savefig(fdir / f"{s['id']}_maps_{ref}.png", dpi=110)
            plt.close(fig)
    print(f"Figures in {fdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["embed", "analyze", "figures", "all"])
    ap.add_argument("--models", help="comma-separated EMBED_SPECS names; 'all' adds the optional models")
    ap.add_argument("--inputs", help="folder with the EXP003 site folders (default: data/processed/exp003)")
    ap.add_argument("--out", help="output folder (default: data/processed/exp004)")
    ap.add_argument("--batch-size", type=int, help="override embedding.batch_size")
    ap.add_argument("--amp", action="store_true", help="float16 autocast on CUDA")
    args = ap.parse_args()
    cfg = load_cfg(args.inputs, args.out)
    if args.batch_size:
        cfg["embedding"]["batch_size"] = args.batch_size
    if args.amp:
        cfg["embedding"]["amp"] = True
    if args.models == "all":
        models = cfg["models"] + cfg.get("optional_models", [])
    elif args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = cfg["models"]
    if args.step in ("embed", "all"):
        print("=== embed", flush=True)
        step_embed(cfg, models)
    if args.step in ("analyze", "all"):
        print("=== analyze", flush=True)
        step_analyze(cfg)
    if args.step in ("figures", "all"):
        print("=== figures", flush=True)
        step_figures(cfg)


if __name__ == "__main__":
    main()
