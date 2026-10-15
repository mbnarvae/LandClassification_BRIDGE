"""Classical pixel/object baselines (spectral + terrain + climate features)."""

from __future__ import annotations

from bridge_lc.models.registry import register


@register("random_forest")
def random_forest(num_classes: int, params: dict | None = None, **_):
    from sklearn.ensemble import RandomForestClassifier

    defaults = {"n_estimators": 500, "class_weight": "balanced_subsample", "n_jobs": -1}
    return RandomForestClassifier(**(defaults | (params or {})))


@register("xgboost")
def xgboost(num_classes: int, params: dict | None = None, **_):
    from xgboost import XGBClassifier

    defaults = {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.05, "tree_method": "hist"}
    return XGBClassifier(num_class=num_classes, **(defaults | (params or {})))
