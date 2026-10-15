import numpy as np

from bridge_lc.data.chips import make_chips
from bridge_lc.evaluation import confusion_matrix, macro_f1, mean_iou
from bridge_lc.evaluation.spatial_cv import iter_folds, spatial_folds
from bridge_lc.features.climate import aridity_index, seasonality_index
from bridge_lc.features.indices import compute_indices, ndvi
from bridge_lc.features.terrain import slope_aspect, tpi


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 1, 2, 2, 3])
    cm = confusion_matrix(y, y, num_classes=3)
    assert macro_f1(cm) == 1.0 and mean_iou(cm) == 1.0


def test_nodata_is_ignored():
    y = np.array([0, 0, 1, 2])
    p = np.array([2, 1, 1, 2])  # wrong predictions only on nodata pixels
    assert macro_f1(confusion_matrix(y, p, 2)) == 1.0


def test_spatial_folds_keep_blocks_together():
    x = np.repeat(np.arange(10) * 1000.0, 10)
    y = np.tile(np.arange(10) * 1000.0, 10)
    folds = spatial_folds(x, y, block_size=5000, n_folds=2)
    blocks = (x // 5000).astype(int) * 10 + (y // 5000).astype(int)
    for b in np.unique(blocks):
        assert len(np.unique(folds[blocks == b])) == 1
    assert sum(1 for _ in iter_folds(folds)) == 2


def test_indices():
    nir, red = np.array([0.5]), np.array([0.1])
    assert np.isclose(ndvi(nir, red), 0.4 / 0.6, atol=1e-5).all()
    img = np.random.default_rng(0).random((6, 4, 4)).astype(np.float32)
    bands = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    assert compute_indices(img, bands, ["ndvi", "ndmi", "ndsi"]).shape == (3, 4, 4)


def test_terrain_on_plane():
    # Plane rising 1 m per 1 m eastward → 45° slope, facing west (aspect 270°).
    dem = np.tile(np.arange(20, dtype=float), (20, 1))
    slope, aspect = slope_aspect(dem, res=1.0)
    assert np.allclose(slope, 45.0, atol=1e-4)
    assert np.allclose(np.degrees(aspect[5:-5, 5:-5]), 270.0, atol=1e-4)
    assert np.allclose(tpi(dem)[5:-5, 5:-5], 0.0, atol=1e-9)


def test_climate():
    p = np.full((12, 2, 2), 100.0)
    pet = np.full((12, 2, 2), 50.0)
    assert np.allclose(aridity_index(p, pet), 2.0)
    assert np.allclose(seasonality_index(p), 0.0)


def test_make_chips():
    img = np.zeros((3, 100, 100), np.float32)
    lab = np.ones((100, 100), int)
    lab[:50] = 0
    x, y, o = make_chips(img, lab, size=50, min_valid_frac=0.5)
    assert x.shape == (2, 3, 50, 50) and (y == 1).all() and o.shape == (2, 2)
