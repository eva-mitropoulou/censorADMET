"""Top-level synthesis of ALL result sources into the paper's headline findings
and the §18 predefined success-criteria scorecard.

Consumes (whatever exists):
  results/matrix_endpoint/all_results.parquet
  results/matrix_measurement/all_results.parquet
  results/hidden_truth/*.csv
  results/conformal/*.csv
  results/dmpnn/*.csv

Emits results/synthesis/:
  - success_criteria.json     : §18 pass/fail on prespecified criteria
  - headline_findings.md      : prose-ready summary with numbers + p-values
  - by_split_tradeoff.csv      : accuracy/violation by split-kind x treatment
  - assay_aware_gain.csv       : satisficing_assay vs satisficing (the novelty)

Honest: every claim carries its n (endpoints), and criteria not evaluable from
available results are marked 'not_evaluable' rather than assumed pass.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

import analysis as A


def _safe_load(p):
    try:
        return A.load_results(p) if p.endswith(".parquet") else pd.read_csv(p)
    except Exception as e:
        print(f"[synth] could not load {p}: {e}", flush=True)
        return None


def _concat_csvs(pattern):
    frames = []
    for f in glob.glob(pattern):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--outdir", default="results/synthesis")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    criteria = {}
    findings = []

    # ---------- endpoint + measurement + broadened-ADMET matrices ----------
    for tag in ["endpoint", "measurement", "admet"]:
        pq = root / f"matrix_{tag}" / "all_results.parquet"
        if not pq.exists():
            continue
        df = A.load_results(str(pq))
        agg = A.aggregate_to_endpoint(df)
        splits = sorted(agg["split_kind"].unique())
        # by-split trade-off table
        g = (agg.groupby(["split_kind", "treatment_label"])
             [["acc_mae", "viol_violation_rate", "dec_false_safe_rate", "cov90"]]
             .mean().round(3).reset_index())
        g.to_csv(out / f"by_split_tradeoff_{tag}.csv", index=False)

        # CRITERION 1: censor-aware (tobit / satisficing) reduces violation vs exact_only,
        # consistently across endpoints (paired, FDR).
        for sk in splits:
            for cand in ["tobit", "satisficing@0.05"]:
                r = A.paired_comparison(agg, "viol_violation_rate", cand, "exact_only", sk)
                if "p_value" in r:
                    findings.append({"matrix": tag, **r})
        # CRITERION: satisficing gives a MONOTONE frontier (violation increases with eps)
        for sk in splits:
            d = agg[agg.split_kind == sk]
            sat = d[d.treatment_label.str.startswith("satisficing@")]
            if len(sat):
                piv = sat.groupby("treatment_label")["viol_violation_rate"].mean()
                order = sorted(piv.index, key=lambda s: float(s.split("@")[1]))
                vals = [piv[o] for o in order]
                # The threshold split holds out an EXACT-potency band, so its test
                # set has (essentially) no censored rows and the violation rate is
                # undefined (NaN). Report as not_evaluable rather than a false FAIL;
                # for this split the Pareto frontier is assessed on MAE instead.
                if not all(np.isfinite(v) for v in vals):
                    mae_piv = sat.groupby("treatment_label")["acc_mae"].mean()
                    mae_vals = [round(float(mae_piv[o]), 3) for o in order]
                    criteria[f"pareto_monotone_{tag}_{sk}"] = {
                        "pass": None, "status": "not_evaluable_no_censored_test_rows",
                        "eps_order": order, "mae_by_eps": mae_vals}
                else:
                    mono = all(vals[i] <= vals[i + 1] + 0.03 for i in range(len(vals) - 1))
                    criteria[f"pareto_monotone_{tag}_{sk}"] = {
                        "pass": bool(mono), "eps_order": order, "violations": [round(v, 3) for v in vals]}

        # ---------- assay-aware + transfer gains (measurement and broadened ADMET) ----------
        if tag in ("measurement", "admet") and "satisficing_assay@0.05" in set(agg.treatment_label):
            gains = []
            comparisons = [("satisficing_assay@0.05", "satisficing@0.05", "assay_vs_satisf")]
            if "satisficing_transfer@0.05" in set(agg.treatment_label):
                comparisons += [
                    ("satisficing_transfer@0.05", "satisficing@0.05", "transfer_vs_satisf"),
                    ("satisficing_transfer@0.05", "satisficing_assay@0.05", "transfer_vs_assay"),
                    ("satisficing_transfer@0.05", "exact_only", "transfer_vs_exact"),
                ]
            for a, b, label in comparisons:
                for sk in splits:
                    for metric in ["acc_mae", "viol_violation_rate", "dec_false_safe_rate"]:
                        r = A.paired_comparison(agg, metric, a, b, sk)
                        if "p_value" in r:
                            gains.append({"comparison": label, "metric": metric, **r})
            gdf = pd.DataFrame(gains)
            gdf.to_csv(out / f"assay_transfer_gain_{tag}.csv", index=False)
            # guard: with too few endpoints (e.g. broadened ADMET set) paired tests
            # may yield no rows, so the df lacks the 'comparison' column.
            if "comparison" not in gdf.columns:
                continue
            # CRITERION: assay-aware helps most under ASSAY shift
            am = gdf[(gdf.comparison == "assay_vs_satisf") & (gdf.split_kind == "assay") & (gdf.metric == "acc_mae")]
            if len(am):
                row = am.iloc[0]
                criteria[f"assay_aware_improves_mae_under_assay_shift_{tag}"] = {
                    "pass": bool(row["median_delta"] < 0 and row["p_value"] < 0.1),
                    "median_delta_mae": round(float(row["median_delta"]), 4),
                    "p_value": round(float(row["p_value"]), 4), "n_endpoints": int(row["n_endpoints"])}
            # CRITERION: transfer improves MAE under SOURCE shift (the old weak spot)
            tm = gdf[(gdf.comparison == "transfer_vs_exact") & (gdf.split_kind == "source") & (gdf.metric == "acc_mae")]
            if len(tm):
                row = tm.iloc[0]
                criteria[f"transfer_improves_mae_under_source_shift_{tag}"] = {
                    "pass": bool(row["median_delta"] < 0 and row["p_value"] < 0.15),
                    "median_delta_mae": round(float(row["median_delta"]), 4),
                    "p_value": round(float(row["p_value"]), 4), "n_endpoints": int(row["n_endpoints"])}

    # ---------- hidden-truth recovery ----------
    ht = _concat_csvs(str(root / "hidden_truth" / "*.csv"))
    if ht is not None and len(ht):
        hg = ht.groupby(["mode", "treatment"])["hidden_mae"].agg(["mean", "std", "count"]).round(3)
        hg.to_csv(out / "hidden_truth_recovery.csv")
        # CRITERION: censor-aware recovers hidden truth better than exact_only (natural mode)
        nat = ht[ht["mode"] == "natural"]
        piv = nat.pivot_table(index="endpoint", columns="treatment", values="hidden_mae")
        if {"exact_only", "tobit"}.issubset(piv.columns):
            pair = piv[["exact_only", "tobit"]].dropna()
            if len(pair) >= 3:
                from scipy.stats import wilcoxon
                try:
                    _, p = wilcoxon(pair["tobit"], pair["exact_only"])
                except ValueError:
                    p = 1.0
                criteria["censor_aware_recovers_hidden_truth"] = {
                    "pass": bool(pair["tobit"].median() < pair["exact_only"].median()),
                    "median_hidden_mae_tobit": round(float(pair["tobit"].median()), 3),
                    "median_hidden_mae_exact": round(float(pair["exact_only"].median()), 3),
                    "p_value": round(float(p), 4), "n_endpoints": int(len(pair))}

    # ---------- conformal calibration ----------
    cf = _concat_csvs(str(root / "conformal" / "*.csv"))
    if cf is not None and len(cf):
        cg = cf.groupby("split")[["raw_cov", "conformal_cov", "mondrian_cov"]].mean().round(3)
        cg.to_csv(out / "conformal_coverage.csv")
        # CRITERION: conformal restores ~0.9 coverage where raw fails
        overall = cf[["raw_cov", "conformal_cov", "mondrian_cov"]].mean()
        criteria["conformal_restores_coverage"] = {
            "pass": bool(abs(overall["conformal_cov"] - 0.9) < abs(overall["raw_cov"] - 0.9)),
            "raw": round(float(overall["raw_cov"]), 3),
            "conformal": round(float(overall["conformal_cov"]), 3),
            "mondrian": round(float(overall["mondrian_cov"]), 3)}

    # ---------- dmpnn backbone agreement ----------
    dm = _concat_csvs(str(root / "dmpnn" / "*.csv"))
    if dm is not None and len(dm):
        dg = dm.groupby("treatment")[["mae", "violation"]].mean().round(3)
        dg.to_csv(out / "dmpnn_summary.csv")
        # CRITERION: D-MPNN shows the same direction (exact_only higher violation than satisficing)
        piv = dm.groupby("treatment")["violation"].mean()
        if "exact_only" in piv.index:
            sat = piv[[i for i in piv.index if str(i).startswith("satisficing")]].mean()
            criteria["dmpnn_same_direction"] = {
                "pass": bool(piv["exact_only"] > sat),
                "exact_only_violation": round(float(piv["exact_only"]), 3),
                "satisficing_violation": round(float(sat), 3)}

    # ---------- write ----------
    (out / "success_criteria.json").write_text(json.dumps(criteria, indent=2, default=float))
    fdf = pd.DataFrame(findings)
    if len(fdf):
        fdf.to_csv(out / "paired_comparisons.csv", index=False)

    # prose-ready markdown
    lines = ["# CensorADMET 2.0 — synthesized headline findings\n"]
    lines.append("## §18 Prespecified success criteria\n")
    for k, v in criteria.items():
        p = v.get("pass")
        status = "N/A" if p is None else ("PASS" if p else "FAIL")
        lines.append(f"- **[{status}] {k}** — {json.dumps({x: y for x, y in v.items() if x != 'pass'})}")
    lines.append("\n## Key paired comparisons (inference unit = endpoint)\n")
    for f in findings:
        if f.get("p_value") is not None:
            lines.append(f"- {f['matrix']} / {f.get('split_kind')}: {f['a']} vs {f['b']} on "
                         f"{f['metric']}: median Δ={f['median_delta']:.3f}, p={f['p_value']:.4f} "
                         f"(n={f['n_endpoints']} endpoints)")
    (out / "headline_findings.md").write_text("\n".join(lines))
    print(f"[synth] wrote {len(criteria)} criteria + findings to {out}", flush=True)
    print("\n".join(lines[:40]), flush=True)


if __name__ == "__main__":
    main()
