from __future__ import annotations

from typing import Any

from sklearn.metrics import rmse

def test_scores_metrics(y_true, y_prob) -> dict[str, float]:
    metrics = {"rmse": float(rmse(y_true, y_prob))}
    return metrics

def metric_score(metric_fn: Any, y_true, y_pred):
    return metric_fn(y_true, y_pred)
