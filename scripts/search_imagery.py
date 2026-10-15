"""List available imagery for a test-bed site (requires the ``geo`` extra).

python scripts/search_imagery.py paramo_ec --product sentinel2 --start 2023-06-01 --end 2023-09-30
"""

from __future__ import annotations

import argparse

from bridge_lc.data.aoi import load_site
from bridge_lc.data.stac import COLLECTIONS, search


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("site")
    ap.add_argument("--product", default="sentinel2", choices=sorted(COLLECTIONS))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max-cloud", type=float, default=40)
    args = ap.parse_args()

    items = search(load_site(args.site), args.product, args.start, args.end, args.max_cloud)
    print(f"{len(items)} items")
    for it in items[:50]:
        print(it.id, it.datetime, it.properties.get("eo:cloud_cover"))


if __name__ == "__main__":
    main()
