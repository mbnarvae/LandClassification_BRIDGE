import numpy as np
import pytest

from bridge_lc.data.aoi import list_sites
from bridge_lc.data.labels import available_sources, load_mapping, remap
from bridge_lc.legend import IGBP20, load_legend
from bridge_lc.noahmp.crosswalk import mptable_stub, to_noahmp


@pytest.fixture(scope="module")
def legend():
    return load_legend()


def test_legend_loads_and_validates(legend):
    assert legend.num_classes > 0
    assert legend.nodata not in legend.ids
    assert all(c.igbp in IGBP20 for c in legend.classes)


def test_site_targets_exist_in_legend(legend):
    names = {c.name for c in legend.classes}
    for site in list_sites():
        assert set(site.target_ecosystems) <= names, site.id
        min_lon, min_lat, max_lon, max_lat = site.bbox
        assert min_lon < max_lon and min_lat < max_lat


def test_crosswalk_igbp_and_ext(legend):
    ids = np.array([[0, legend.by_name("paramo_grassland").id], [legend.by_name("water").id, 0]])
    igbp = to_noahmp(ids, legend, "igbp")
    ext = to_noahmp(ids, legend, "noahmp_ext")
    assert igbp.tolist() == [[0, 10], [17, 0]]
    assert ext[0, 1] > 20 and ext[1, 0] == 17


def test_mptable_stub_lists_extended(legend):
    text = mptable_stub(legend)
    for c in legend.extended_classes:
        assert c.name in text


def test_label_mappings(legend):
    for source in available_sources():
        mapping = load_mapping(source, legend)
        assert set(mapping.values()) <= set(legend.ids)
    out = remap(np.array([5, 7, 9]), {5: 1, 9: 3})
    assert out.tolist() == [1, 0, 3]
