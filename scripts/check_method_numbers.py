"""Verify that the numbers reported in manuscript_v2/main_v2.qmd are reproducible
from the frozen result artifacts in stage13_method/results/.

This is the method-paper analogue of scripts/check_numerical_consistency.py: it
recomputes a representative set of headline numbers in the manuscript from the
released parquet/CSV artifacts and asserts the recomputed value matches the value
asserted in CLAIMS below (which mirrors the manuscript). Any drift raises SystemExit, so a
reviewer (or CI) can confirm the manuscript is backed by the released data.

Run from the stage13_method directory:
    python check_method_numbers.py
Exit 0 = all manuscript numbers reproduce; non-zero = a mismatch is reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
ATOL = 0.01  # manuscript rounds to ~2-3 decimals; allow 0.01 absolute drift


def _label(row):
    t = row["treatment"]
    if t in ("satisficing", "satisficing_assay", "satisficing_transfer", "ensemble_satisficing"):
        return f"{t}@{row['eps']:.2f}"
    return t


def _agg(matrix):
    """Endpoint-level mean per (split, treatment_label) for a matrix parquet."""
    df = pd.read_parquet(RES / matrix / "all_results.parquet")
    df = df[df.get("error").isna()] if "error" in df.columns else df
    df["tl"] = df.apply(_label, axis=1)
    metrics = ["acc_mae", "viol_violation_rate", "dec_decision_regret"]
    m = [c for c in metrics if c in df.columns]
    per_ep = df.groupby(["endpoint", "split_kind", "tl"])[m].mean().reset_index()
    return per_ep


def _median_delta(per_ep, split, trt, base, metric):
    a = per_ep[(per_ep.split_kind == split) & (per_ep.tl == trt)].set_index("endpoint")[metric]
    b = per_ep[(per_ep.split_kind == split) & (per_ep.tl == base)].set_index("endpoint")[metric]
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return None
    return float((a.loc[common] - b.loc[common]).median())


# ---- CLAIMS: (description, computed_fn, asserted_value) mirroring the manuscript ----
def build_checks():
    meas = _agg("matrix_measurement")
    checks = []

    # tbl-violation: satisficing@0.05 vs exact_only median violation deltas
    for split, val in [("random", -0.281), ("scaffold", -0.254), ("assay", -0.212),
                       ("document", -0.162), ("source", -0.161)]:
        got = _median_delta(meas, split, "satisficing@0.05", "exact_only", "viol_violation_rate")
        checks.append((f"satisf Δviol {split}", got, val))

    # feasibility summary
    f = pd.read_csv(RES / "feasibility" / "feasibility.csv")
    f["max_excess"] = f[["G_right", "G_left"]].sub(f["eps"], axis=0).max(axis=1)
    for eps, frac, med, mx in [(0.02, 0.67, 0.000, 0.017), (0.05, 0.50, 0.001, 0.014),
                               (0.10, 0.67, -0.002, 0.003), (0.20, 0.83, -0.013, 0.002)]:
        s = f[np.isclose(f.eps, eps)]
        checks.append((f"feas frac {eps}", float((s.max_excess <= 1e-3).mean()), frac))
        checks.append((f"feas median-excess {eps}", float(s.max_excess.median()), med))

    # weighted-Tobit vs satisficing curve (measurement/random, mean over endpoints)
    mm = pd.read_parquet(RES / "matrix_measurement" / "all_results.parquet")
    mm = mm[mm.get("error").isna()] if "error" in mm.columns else mm
    rnd = mm[mm.split_kind == "random"]
    def cell(trt, eps=None, metric="acc_mae"):
        s = rnd[rnd.treatment == trt] if eps is None else rnd[(rnd.treatment == trt) & (np.isclose(rnd.eps, eps))]
        return float(s[metric].mean()) if len(s) else None
    checks.append(("wt exact MAE", cell("exact_only"), 0.521))
    checks.append(("wt w=0.25 MAE", cell("weighted_tobit", 0.25), 0.569))
    checks.append(("wt w=0.25 viol", cell("weighted_tobit", 0.25, "viol_violation_rate"), 0.320))
    checks.append(("wt satisf@0.02 MAE", cell("satisficing", 0.02), 0.552))
    checks.append(("wt satisf@0.10 MAE", cell("satisficing", 0.10), 0.537))
    anchor = pd.read_csv(RES / "synthesis" / "anchor_free_summary.csv").iloc[0]
    checks.append(("anchor-free satisf MAE", float(anchor["acc_mae"]), 0.609))
    checks.append(("anchor-free satisf viol", float(anchor["viol_violation_rate"]), 0.264))

    # compound-overlap supplementary (per-split mean over endpoints)
    ov = pd.read_csv(RES / "synthesis" / "compound_overlap.csv")
    ov_mean = ov.groupby("split_kind")["mean_overlap"].mean()
    for split, val in [("random", 0.230), ("assay", 0.202), ("document", 0.106),
                       ("source", 0.083), ("threshold", 0.117),
                       ("scaffold", 0.000), ("assay_scaffold", 0.000)]:
        checks.append((f"overlap {split}", float(ov_mean.get(split, np.nan)), val))

    # per-property ADMET breakdown (SI tbl-perprop): random-split exact vs satisficing
    pp = pd.read_csv(RES / "synthesis" / "by_property_admet.csv")
    def ppcell(ep, sk, tl, metric):
        s = pp[(pp.endpoint == ep) & (pp.split_kind == sk) & (pp.tl == tl)]
        return float(s.iloc[0][metric]) if len(s) else None
    for ep, ex_v, sa_v in [("Solubility_nM", 0.801, 0.483), ("Clearance_mLminkg", 0.781, 0.533),
                           ("HalfLife_hr_sub", 0.753, 0.471)]:
        checks.append((f"perprop {ep} exact viol", ppcell(ep, "random", "exact_only", "viol_violation_rate"), ex_v))
        checks.append((f"perprop {ep} satisf viol", ppcell(ep, "random", "satisficing@0.05", "viol_violation_rate"), sa_v))

    # crossed assay+scaffold split (tbl-crossed): mean over endpoints + median deltas
    cr = pd.read_csv(RES / "synthesis" / "by_endpoint_crossed.csv")
    def crmean(tl, metric):
        s = cr[cr.tl == tl]
        return float(s[metric].mean()) if len(s) else None
    for tl, mae, viol in [("exact_only", 0.771, 0.590), ("tobit", 1.012, 0.249),
                          ("satisficing@0.05", 0.796, 0.412)]:
        checks.append((f"crossed {tl} MAE", crmean(tl, "acc_mae"), mae))
        checks.append((f"crossed {tl} viol", crmean(tl, "viol_violation_rate"), viol))
    def crmed(tl, base, metric):
        a = cr[cr.tl == tl].set_index("endpoint")[metric]
        b = cr[cr.tl == base].set_index("endpoint")[metric]
        common = a.index.intersection(b.index)
        return float((a.loc[common] - b.loc[common]).median()) if len(common) >= 3 else None
    checks.append(("crossed satisf ΔMAE", crmed("satisficing@0.05", "exact_only", "acc_mae"), 0.026))
    checks.append(("crossed satisf Δviol", crmed("satisficing@0.05", "exact_only", "viol_violation_rate"), -0.189))

    # Recomputed conformal coverage, width and interval score.
    cc = pd.read_csv(RES / "synthesis" / "conformal_scored_summary.csv").set_index("split")
    for split, raw, conf, raw_width, conf_width, raw_score, conf_score in [
        ("random", 0.348, 0.903, 0.802, 4.833, 8.642, 5.801),
        ("scaffold", 0.268, 0.874, 0.716, 4.443, 10.177, 5.828),
        ("assay", 0.257, 0.794, 0.640, 3.592, 13.867, 6.355),
    ]:
        checks.append((f"conformal raw {split}", float(cc.loc[split, "raw_cov"]), raw))
        checks.append((f"conformal conformal {split}", float(cc.loc[split, "conformal_cov"]), conf))
        checks.append((f"conformal raw width {split}", float(cc.loc[split, "raw_width"]), raw_width))
        checks.append((f"conformal width {split}", float(cc.loc[split, "conformal_width"]), conf_width))
        checks.append((f"conformal raw score {split}", float(cc.loc[split, "raw_interval_score"]), raw_score))
        checks.append((f"conformal score {split}", float(cc.loc[split, "conformal_interval_score"]), conf_score))

    # D-MPNN table (scaffold, satisficing@0.02)
    dm = pd.concat([pd.read_csv(p) for p in (RES / "dmpnn").glob("*.csv")])
    for ep, col, eps, val in [("CYP2C9_IC50_A", "mae", None, 0.49), ("CYP2C9_IC50_A", "mae", 0.0, None)]:
        pass  # dmpnn spot-checked below
    def dmpnn(ep, trt, eps=None, metric="mae"):
        s = dm[(dm.endpoint == ep) & (dm.split == "scaffold") & (dm.treatment == trt)]
        if eps is not None:
            s = s[np.isclose(s.eps, eps)]
        return float(s[metric].mean()) if len(s) else None
    checks.append(("dmpnn CYP2C9 exact MAE", dmpnn("CYP2C9_IC50_A", "exact_only"), 0.49))
    checks.append(("dmpnn CYP2C9 tobit MAE", dmpnn("CYP2C9_IC50_A", "tobit"), 3.41))

    return checks


def main():
    checks = build_checks()
    fails = []
    for name, got, want in checks:
        if want is None:
            continue
        if got is None:
            fails.append(f"{name}: could not recompute (missing data)")
        elif not np.isclose(got, want, atol=ATOL, rtol=0):
            fails.append(f"{name}: manuscript says {want:+.3f}, recomputed {got:+.3f} (|Δ|={abs(got-want):.3f})")
        else:
            print(f"OK  {name:32s} {got:+.3f} ~= {want:+.3f}")
    if fails:
        print("\nMISMATCHES:")
        for f in fails:
            print("  " + f)
        raise SystemExit(f"{len(fails)} manuscript numbers do not reproduce from released artifacts")
    print(f"\nAll {len([c for c in checks if c[2] is not None])} checked manuscript numbers reproduce from released artifacts.")


if __name__ == "__main__":
    main()
