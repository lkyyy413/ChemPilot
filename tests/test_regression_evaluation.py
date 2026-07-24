import numpy as np
import pytest

from chempilot.evaluation.regression import (
    prediction_table,
    regression_metrics,
)


def test_regression_metrics_known_values():
    y_true = np.array([-3.0, -2.0, -1.0])
    y_pred = np.array([-2.5, -2.0, -1.5])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["n_samples"] == 3
    assert metrics["mae"] == pytest.approx(1.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(
        np.sqrt(1.0 / 6.0)
    )
    assert metrics["r2"] == pytest.approx(0.75)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_mean_prediction_has_undefined_spearman():
    y_true = np.array([-3.0, -2.0, -1.0])
    y_pred = np.array([-2.0, -2.0, -2.0])

    metrics = regression_metrics(y_true, y_pred)

    assert np.isnan(metrics["spearman"])


def test_prediction_table_sorted_by_error():
    table = prediction_table(
        sample_ids=np.array(["A", "B"]),
        smiles=np.array(["CCO", "CC"]),
        y_true=np.array([-3.0, -1.0]),
        y_pred=np.array([-2.9, -3.0]),
        in_druglike_scope=np.array([True, False]),
    )

    assert table.iloc[0]["sample_id"] == "B"
    assert table.iloc[0]["absolute_error"] == pytest.approx(
        2.0
    )