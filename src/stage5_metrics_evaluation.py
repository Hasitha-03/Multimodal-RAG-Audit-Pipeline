"""
Stage 5 — Metrics & Evaluation

Loads the Stage 4 scope matrix (JSON preferred, CSV supported) and computes
the pipeline's evaluation metrics against ground truth:

  1. Status Classification Accuracy — per-class Precision/Recall/F1 for
     INCLUDED / EXCLUDED / NOT_MENTIONED, plus a macro-average, computed at
     all three scoring tiers.
  2. Cost Match Rate — fraction of rows where the contractor's extracted
     cost matches ground truth cost, restricted to rows where gt_cost is
     non-null (lump-sum/null ground-truth rows are skipped entirely from
     this metric, per user decision — they carry no signal either way).
  3. Citation Grounding Success Rate — valid (grounded) citations / total
     extracted citations, a diagnostic tracked across ALL items regardless
     of tier (this is about citation mechanics, not final-answer scoring).
  4. Human Review Flag Frequency — % of items with needs_human_review=True.

Scoring tiers (confirmed with user, matches the framework designed earlier):
  - Autonomous: metrics computed ONLY on rows where needs_human_review is
    False. Answers "how trustworthy is the model when it acts alone?"
  - Overall (Conservative): metrics computed on ALL rows, with flagged rows
    counted as WRONG by default. Answers "what's the true end-to-end
    automation accuracy, unassisted?"
  - Human Review Rate: not an accuracy metric — the % of rows flagged,
    i.e. how much manual review this pipeline is actually saving.

Cost matching rule: a cost "matches" if it's within COST_MATCH_TOLERANCE
(a small relative tolerance, since dollar figures can differ by rounding
even when both sides agree) OR both sides are exactly equal. Rows where
gt_cost is present but the contractor's cost_value is null count as a
non-match (a real miss, not skipped) — only gt_cost being null causes a
row to be skipped from this metric entirely.

Usage:
    python stage5_metrics_evaluation.py data/scope_matrix.json [--out report.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from sklearn.metrics import precision_recall_fscore_support


STATUS_CLASSES = ["INCLUDED", "EXCLUDED", "NOT_MENTIONED"]

# Relative tolerance for cost matching (e.g. 0.01 = within 1%). A small
# absolute floor is also applied so tiny dollar amounts aren't unfairly
# strict (e.g. $0.001 tolerance on a $10 item would be absurd).
COST_MATCH_RELATIVE_TOLERANCE = 0.01
COST_MATCH_ABSOLUTE_FLOOR = 1.00  # dollars


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int  # number of ground-truth rows actually in this class (this tier's subset)


@dataclass
class StatusMetrics:
    per_class: dict[str, PerClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    accuracy: float
    num_rows: int


@dataclass
class TierMetrics:
    tier_name: str  # "autonomous" | "overall_conservative"
    status_metrics: StatusMetrics
    cost_match_rate: Optional[float]  # None if zero eligible rows
    cost_match_num_eligible: int  # rows where gt_cost is non-null
    cost_match_num_matched: int


@dataclass
class EvaluationReport:
    contractor_name: str
    tiers: dict[str, TierMetrics]  # keyed by tier_name
    citation_grounding_success_rate: Optional[float]
    citation_grounding_num_checked: int
    citation_grounding_num_valid: int
    human_review_flag_frequency: float
    num_total_rows: int
    num_flagged_rows: int


# ----------------------------------------------------------------------------
# Loading the scope matrix
# ----------------------------------------------------------------------------

def load_scope_matrix_json(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["line_items"]


def load_scope_matrix_csv(path: Path, contractor_names: list[str]) -> list[dict]:
    """
    Reconstruct the nested per-line-item structure from the flattened wide
    CSV, for the given contractor column-group names. Requires knowing
    contractor_names in advance since the CSV's column names encode them
    and there's no other way to recover which columns belong to which
    contractor from the flat header alone without this list.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for csv_row in reader:
            row = {
                "line_item_id": csv_row["line_item_id"],
                "description": csv_row["description"],
                "section": csv_row["section"],
                "ground_truth": {
                    "status": csv_row["gt_status"] or None,
                    "cost": float(csv_row["gt_cost"]) if csv_row["gt_cost"] else None,
                    "notes": csv_row["gt_notes"] or None,
                },
                "contractors": {},
            }
            for name in contractor_names:
                status = csv_row.get(f"{name}_status") or None
                if status is None:
                    row["contractors"][name] = None
                    continue
                cost_str = csv_row.get(f"{name}_cost")
                needs_review_str = csv_row.get(f"{name}_needs_review")
                row["contractors"][name] = {
                    "status": status,
                    "cost_value": float(cost_str) if cost_str else None,
                    "confidence": csv_row.get(f"{name}_confidence") or None,
                    "citation_page_number": csv_row.get(f"{name}_citation_page") or None,
                    "citation_verbatim": csv_row.get(f"{name}_citation_text") or None,
                    "grounding_score": None,  # not present in the CSV export — JSON is the
                                              # authoritative source for grounding_score; CSV
                                              # loading is a best-effort fallback, not the
                                              # primary path (see module docstring).
                    "needs_human_review": (needs_review_str or "").strip().lower() == "true",
                    "flag_reason": csv_row.get(f"{name}_flag_reason") or None,
                }
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Cost matching
# ----------------------------------------------------------------------------

def costs_match(gt_cost: float, predicted_cost: Optional[float]) -> bool:
    if predicted_cost is None:
        return False
    tolerance = max(COST_MATCH_ABSOLUTE_FLOOR, gt_cost * COST_MATCH_RELATIVE_TOLERANCE)
    return abs(gt_cost - predicted_cost) <= tolerance


# ----------------------------------------------------------------------------
# Status classification metrics
# ----------------------------------------------------------------------------

def compute_status_metrics(gt_statuses: list[str], predicted_statuses: list[str]) -> StatusMetrics:
    """
    Computes per-class + macro-averaged precision/recall/F1 and overall
    accuracy. Uses sklearn's precision_recall_fscore_support with a fixed
    `labels=STATUS_CLASSES` so a class with zero support in this tier's
    subset still appears in the report (with 0.0 metrics) rather than
    silently vanishing — a missing class is itself informative (e.g. "this
    contractor's bid never had any EXCLUDED ground-truth items to test
    against").
    """
    if not gt_statuses:
        empty_per_class = {
            cls: PerClassMetrics(precision=0.0, recall=0.0, f1=0.0, support=0) for cls in STATUS_CLASSES
        }
        return StatusMetrics(
            per_class=empty_per_class, macro_precision=0.0, macro_recall=0.0,
            macro_f1=0.0, accuracy=0.0, num_rows=0,
        )

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        gt_statuses, predicted_statuses, labels=STATUS_CLASSES, average=None, zero_division=0
    )

    per_class = {
        cls: PerClassMetrics(precision=float(p), recall=float(r), f1=float(f), support=int(s))
        for cls, p, r, f, s in zip(STATUS_CLASSES, precisions, recalls, f1s, supports)
    }

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gt_statuses, predicted_statuses, labels=STATUS_CLASSES, average="macro", zero_division=0
    )

    accuracy = sum(1 for gt, pred in zip(gt_statuses, predicted_statuses) if gt == pred) / len(gt_statuses)

    return StatusMetrics(
        per_class=per_class,
        macro_precision=float(macro_p),
        macro_recall=float(macro_r),
        macro_f1=float(macro_f1),
        accuracy=accuracy,
        num_rows=len(gt_statuses),
    )


# ----------------------------------------------------------------------------
# Main per-contractor evaluation
# ----------------------------------------------------------------------------

def evaluate_contractor(contractor_name: str, matrix: list[dict]) -> EvaluationReport:
    """
    Computes the full evaluation report for one contractor across the
    3-tier framework, restricted to rows where this contractor actually
    has a result (a row where the contractor has no result at all — e.g.
    that line item was never processed — is excluded from every metric
    here rather than counted as a wrong/flagged guess, since "no attempt
    was made" is a distinct condition from "an attempt was made and
    failed"; the run-completeness gap itself is worth surfacing separately
    if it's large, which num_total_rows vs. the ground-truth item count
    lets a caller check).
    """
    autonomous_gt, autonomous_pred = [], []
    overall_gt, overall_pred = [], []

    autonomous_cost_gt, autonomous_cost_pred = [], []
    overall_cost_gt, overall_cost_pred = [], []

    citation_scores: list[float] = []  # grounding_score for every row with a citation expected+checked
    num_flagged = 0
    num_total = 0

    for row in matrix:
        c = row["contractors"].get(contractor_name)
        if c is None:
            continue  # no attempt was made on this line item — excluded, not penalized

        num_total += 1
        gt_status = row["ground_truth"]["status"]
        gt_cost = row["ground_truth"]["cost"]

        is_flagged = bool(c.get("needs_human_review"))
        if is_flagged:
            num_flagged += 1

        pred_status = c.get("status")

        # Overall (conservative) tier: every row counts. A flagged row's
        # prediction is deliberately treated as WRONG — substitute a
        # sentinel that can never match gt_status, rather than skipping it
        # or trusting a flagged (unverified) prediction.
        overall_gt.append(gt_status)
        overall_pred.append(pred_status if not is_flagged else "__FLAGGED_UNVERIFIED__")

        # Autonomous tier: flagged rows are excluded entirely (this tier
        # measures trustworthiness only when the model acted without
        # needing review, so a flagged row simply isn't part of this
        # population — it's neither a hit nor a miss here).
        if not is_flagged:
            autonomous_gt.append(gt_status)
            autonomous_pred.append(pred_status)

        # Cost matching: only rows where gt_cost is non-null (per user
        # decision). Applied at both tiers using the same flagged/unflagged
        # split as status.
        if gt_cost is not None:
            pred_cost = c.get("cost_value")
            overall_cost_gt.append(gt_cost)
            overall_cost_pred.append(pred_cost if not is_flagged else None)  # flagged = treated as non-match
            if not is_flagged:
                autonomous_cost_gt.append(gt_cost)
                autonomous_cost_pred.append(pred_cost)

        # Citation grounding: diagnostic across ALL items where a citation
        # was actually expected+checked (i.e. grounding_score is not None —
        # NOT_MENTIONED rows never had a citation to check, so they simply
        # don't contribute a score either way).
        score = c.get("grounding_score")
        if score is not None:
            citation_scores.append(score)

    def build_tier(tier_name: str, gt_list: list[str], pred_list: list[str], cost_gt: list[float], cost_pred: list) -> TierMetrics:
        status_metrics = compute_status_metrics(gt_list, pred_list)
        num_eligible = len(cost_gt)
        num_matched = sum(1 for gt, pred in zip(cost_gt, cost_pred) if costs_match(gt, pred))
        cost_match_rate = (num_matched / num_eligible) if num_eligible > 0 else None
        return TierMetrics(
            tier_name=tier_name,
            status_metrics=status_metrics,
            cost_match_rate=cost_match_rate,
            cost_match_num_eligible=num_eligible,
            cost_match_num_matched=num_matched,
        )

    tiers = {
        "autonomous": build_tier("autonomous", autonomous_gt, autonomous_pred, autonomous_cost_gt, autonomous_cost_pred),
        "overall_conservative": build_tier("overall_conservative", overall_gt, overall_pred, overall_cost_gt, overall_cost_pred),
    }

    grounding_rate = (sum(1 for s in citation_scores if s >= 0) and
                      sum(1 for s in citation_scores if _is_valid_grounding_score(s)) / len(citation_scores)) if citation_scores else None

    return EvaluationReport(
        contractor_name=contractor_name,
        tiers=tiers,
        citation_grounding_success_rate=grounding_rate,
        citation_grounding_num_checked=len(citation_scores),
        citation_grounding_num_valid=sum(1 for s in citation_scores if _is_valid_grounding_score(s)),
        human_review_flag_frequency=(num_flagged / num_total) if num_total > 0 else 0.0,
        num_total_rows=num_total,
        num_flagged_rows=num_flagged,
    )


def _is_valid_grounding_score(score: float, threshold: float = 88.0) -> bool:
    """
    A citation counts as "valid" for the grounding success-rate diagnostic
    if its score met the Stage 3.5 threshold. Re-deriving this from the
    stored numeric score (rather than only trusting needs_human_review)
    means this diagnostic reflects citation mechanics specifically, not
    conflated with OTHER reasons a row might be flagged in the future.
    Threshold matches Stage 3.5's DEFAULT_GROUNDING_THRESHOLD; pass a
    different value here if Stage 3.5 was run with a non-default threshold.
    """
    return score >= threshold


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def report_to_dict(report: EvaluationReport) -> dict:
    d = asdict(report)
    return d


def print_human_readable_report(report: EvaluationReport) -> None:
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT — {report.contractor_name}")
    print(f"{'='*70}")
    print(f"Total rows evaluated: {report.num_total_rows}")
    print(f"Human Review Flag Frequency: {report.human_review_flag_frequency:.1%} ({report.num_flagged_rows} rows)")

    if report.citation_grounding_success_rate is not None:
        print(
            f"Citation Grounding Success Rate: {report.citation_grounding_success_rate:.1%} "
            f"({report.citation_grounding_num_valid}/{report.citation_grounding_num_checked})"
        )
    else:
        print("Citation Grounding Success Rate: N/A (no citations were checked)")

    for tier_name in ["autonomous", "overall_conservative"]:
        tier = report.tiers[tier_name]
        sm = tier.status_metrics
        print(f"\n--- Tier: {tier_name} ({sm.num_rows} rows) ---")
        print(f"  Status Accuracy: {sm.accuracy:.1%}")
        print(f"  Macro Precision/Recall/F1: {sm.macro_precision:.3f} / {sm.macro_recall:.3f} / {sm.macro_f1:.3f}")
        for cls in STATUS_CLASSES:
            pc = sm.per_class[cls]
            print(f"    {cls:15s} P={pc.precision:.3f} R={pc.recall:.3f} F1={pc.f1:.3f} (support={pc.support})")
        if tier.cost_match_rate is not None:
            print(f"  Cost Match Rate: {tier.cost_match_rate:.1%} ({tier.cost_match_num_matched}/{tier.cost_match_num_eligible} eligible rows)")
        else:
            print("  Cost Match Rate: N/A (no rows with non-null ground-truth cost)")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Compute evaluation metrics from the scope matrix")
    parser.add_argument("scope_matrix_path", type=Path, help="Path to scope_matrix.json or .csv")
    parser.add_argument(
        "--contractors",
        nargs="*",
        default=None,
        help="Contractor names to evaluate (required if loading a .csv; auto-detected from .json)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write the full JSON report")
    args = parser.parse_args()

    if not args.scope_matrix_path.exists():
        print(f"ERROR: file not found: {args.scope_matrix_path}", file=sys.stderr)
        sys.exit(1)

    suffix = args.scope_matrix_path.suffix.lower()
    if suffix == ".json":
        matrix = load_scope_matrix_json(args.scope_matrix_path)
        contractor_names = args.contractors or sorted(matrix[0]["contractors"].keys()) if matrix else []
    elif suffix == ".csv":
        if not args.contractors:
            print("ERROR: --contractors is required when loading a .csv scope matrix.", file=sys.stderr)
            sys.exit(1)
        matrix = load_scope_matrix_csv(args.scope_matrix_path, args.contractors)
        contractor_names = args.contractors
    else:
        print(f"ERROR: unsupported file type '{suffix}'. Expected .json or .csv.", file=sys.stderr)
        sys.exit(1)

    all_reports = []
    for name in contractor_names:
        report = evaluate_contractor(name, matrix)
        print_human_readable_report(report)
        all_reports.append(report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump([report_to_dict(r) for r in all_reports], f, indent=2)
        print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
