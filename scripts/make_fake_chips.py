"""Write synthetic chips in the exact EXP001 layout, to smoke-test the TerraTorch configs without real data.

python scripts/make_fake_chips.py --out data/processed/fake_exp001 --n 24
experiments/exp001_prithvi_8basins/run_exp001.sh smoke b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    import rasterio
    from rasterio.transform import from_origin

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/fake_exp001")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--classes", type=int, default=18)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "chips").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = []
    for i in range(args.n):
        name = f"fake_r{i:05d}_c00000"
        y = rng.integers(0, args.classes + 1, (args.size // 16, args.size // 16)).repeat(16, 0).repeat(16, 1)
        x = rng.normal(1500, 600, (6 * args.frames, args.size, args.size)) + 40 * y[None]
        x[:, :10, :10] = -9999  # some nodata like a cloud gap
        prof = {
            "driver": "GTiff",
            "width": args.size,
            "height": args.size,
            "crs": "EPSG:32717",
            "transform": from_origin(700000 + i * 6720, 9900000, 30, 30),
        }
        with rasterio.open(
            out / "chips" / f"{name}_img.tif", "w", count=x.shape[0], dtype="int16", nodata=-9999, **prof
        ) as f:
            f.write(x.astype(np.int16))
        with rasterio.open(
            out / "chips" / f"{name}_mask.tif", "w", count=1, dtype="uint8", nodata=0, **prof
        ) as f:
            f.write(y.astype(np.uint8), 1)
        names.append(name)
    k = args.n // 4
    for split, sel in (("train", names[: 2 * k]), ("val", names[2 * k : 3 * k]), ("test", names[3 * k :])):
        (out / "splits" / f"{split}.txt").write_text("\n".join(sel) + "\n")
    print(f"wrote {args.n} fake chips to {out}")


if __name__ == "__main__":
    main()
