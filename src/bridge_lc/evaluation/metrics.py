"""Segmentation / classification metrics that ignore nodata labels."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int, ignore_index: int | None = 0) -> np.ndarray:
    """Confusion matrix of shape (num_classes+1, num_classes+1) indexed by raw label id.

    Row = truth, column = prediction. Pixels with ``y_true == ignore_index`` are skipped.
    """
    t = np.asarray(y_true).ravel().astype(np.int64)
    p = np.asarray(y_pred).ravel().astype(np.int64)
    if ignore_index is not None:
        keep = t != ignore_index
        t, p = t[keep], p[keep]
    n = num_classes + 1
    return np.bincount(t * n + p, minlength=n * n).reshape(n, n)


def _per_class(cm: np.ndarray, ignore_index: int | None = 0):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    present = cm.sum(1) > 0  # only score classes that appear in the truth
    if ignore_index is not None:
        present[ignore_index] = False
    return tp, fp, fn, present


def macro_f1(cm: np.ndarray, ignore_index: int | None = 0) -> float:
    tp, fp, fn, present = _per_class(cm, ignore_index)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    return float(f1[present].mean()) if present.any() else 0.0


def mean_iou(cm: np.ndarray, ignore_index: int | None = 0) -> float:
    tp, fp, fn, present = _per_class(cm, ignore_index)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    return float(iou[present].mean()) if present.any() else 0.0


def overall_accuracy(cm: np.ndarray) -> float:
    """Fraction correct; nodata truth pixels are already excluded by ``confusion_matrix``."""
    total = cm.sum()
    return float(np.trace(cm) / total) if total else 0.0


def classification_report(cm: np.ndarray, names: dict[int, str] | None = None, ignore_index=0):
    """Per-class precision/recall/F1/IoU as a list of dicts (classes present in the truth)."""
    tp, fp, fn, present = _per_class(cm, ignore_index)
    rows = []
    for k in np.flatnonzero(present):
        prec = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        rec = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        iou = tp[k] / (tp[k] + fp[k] + fn[k])
        rows.append(
            {
                "id": int(k),
                "name": (names or {}).get(int(k), str(k)),
                "support": int(cm[k].sum()),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "iou": iou,
            }
        )
    return rows
