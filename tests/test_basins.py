import pytest

from bridge_lc.data.basins import utm_epsg


def test_utm_epsg_andes():
    assert utm_epsg(-78.5, -0.2) == "EPSG:32717"  # Quito area
    assert utm_epsg(-72.0, -13.5) == "EPSG:32719"  # Cusco area
    assert utm_epsg(-74.0, 4.6) == "EPSG:32618"  # Bogotá (north of equator)


def test_load_basins_roundtrip(tmp_path):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    from bridge_lc.data.basins import load_basins

    gdf = gpd.GeoDataFrame(
        {"name": ["a", "b"]}, geometry=[box(-78, -1, -77.5, -0.5), box(-72, -14, -71.5, -13.5)], crs=4326
    )
    gdf.to_file(tmp_path / "b.shp")
    basins = load_basins(tmp_path / "b.shp", "name", buffer_m=500, expected=2)
    assert [b.id for b in basins] == ["a", "b"]
    assert basins[0].geometry.area > box(-78, -1, -77.5, -0.5).area  # buffered
    with pytest.raises(ValueError):
        load_basins(tmp_path / "b.shp", "name", expected=8)
