import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis import (  # noqa: E402
    aggregate_to_endpoint,
    bh_fdr,
    comparison_table,
    paired_comparison,
    pareto_summary,
)


def _fake_results(seed=0):
    """8 endpoints x 2 treatments x 5 folds x 3 seeds. satisficing has lower
    violation but higher MAE than exact_only (the expected trade-off)."""
    rng = np.random.default_rng(seed)
    rows = []
    for ep in [f"EP{i}" for i in range(8)]:
        base_mae = rng.uniform(0.4, 0.8)
        base_viol = rng.uniform(0.5, 0.7)
        for fold in range(5):
            for sd in range(3):
                for trt, eps in [("exact_only", 0.0), ("satisficing", 0.05)]:
                    noise = rng.normal(0, 0.02)
                    if trt == "exact_only":
                        mae = base_mae + noise; viol = base_viol + noise
                    else:
                        mae = base_mae + 0.08 + noise; viol = base_viol - 0.25 + noise
                    rows.append(dict(endpoint=ep, split_kind="scaffold", fold=fold,
                                     seed=sd, treatment=trt, eps=eps,
                                     acc_mae=mae, viol_violation_rate=viol,
                                     cov90=0.9, exceed_ece=0.05))
    df = pd.DataFrame(rows)
    from analysis import _treatment_label
    df["treatment_label"] = df.apply(_treatment_label, axis=1)
    return df


def test_aggregate_to_endpoint_reduces_replicates():
    df = _fake_results()
    agg = aggregate_to_endpoint(df)
    # 8 endpoints x 1 split x 2 treatments = 16 rows, each averaging 15 cells
    assert len(agg) == 16
    assert (agg["n_cells"] == 15).all()


def test_bh_fdr_basic():
    rej, q = bh_fdr([0.001, 0.02, 0.5, 0.9])
    assert rej[0] and not rej[3]
    assert (q >= 0).all() and (q <= 1).all()
    # monotone
    assert np.all(np.diff(q[np.argsort([0.001, 0.02, 0.5, 0.9])]) >= -1e-9)


def test_paired_comparison_detects_tradeoff():
    df = _fake_results()
    agg = aggregate_to_endpoint(df)
    # satisficing@0.05 vs exact_only: violation should be significantly LOWER
    r = paired_comparison(agg, "viol_violation_rate", "satisficing@0.05", "exact_only",
                          split_kind="scaffold")
    assert r["n_endpoints"] == 8
    assert r["median_delta"] < 0          # satisficing has lower violation
    assert r["p_value"] < 0.05            # consistently, across endpoints
    # and MAE should be significantly HIGHER
    r2 = paired_comparison(agg, "acc_mae", "satisficing@0.05", "exact_only", "scaffold")
    assert r2["median_delta"] > 0
    assert r2["p_value"] < 0.05


def test_comparison_table_has_fdr():
    df = _fake_results()
    agg = aggregate_to_endpoint(df)
    tbl = comparison_table(agg, "viol_violation_rate",
                           ["satisficing@0.05", "exact_only"], "exact_only", "scaffold")
    assert "fdr_qvalue" in tbl.columns
    assert len(tbl) == 1  # only satisficing vs baseline


def test_pareto_summary_shape():
    df = _fake_results()
    agg = aggregate_to_endpoint(df)
    ps = pareto_summary(agg, "scaffold")
    assert set(["endpoint", "treatment_label", "viol_violation_rate", "acc_mae"]).issubset(ps.columns)
    assert len(ps) > 0


def test_no_pseudoreplication_n_is_endpoints():
    # The paired test's n must equal the number of ENDPOINTS (8), never the
    # number of cells (240). This guards the pseudoreplication fix.
    df = _fake_results()
    agg = aggregate_to_endpoint(df)
    r = paired_comparison(agg, "acc_mae", "satisficing@0.05", "exact_only", "scaffold")
    assert r["n_endpoints"] == 8, "inference unit must be endpoint, not cell"
