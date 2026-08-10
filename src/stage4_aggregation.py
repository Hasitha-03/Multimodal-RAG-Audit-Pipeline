"""
Stage 4 — Aggregation & Final Scope Matrix Export

Combines Stage 0 ground truth with every contractor's Stage 3.5
GroundingResult records into one comparative "scope matrix": one row per
scope line item, with ground-truth columns first and then a repeating
column-group per contractor (status/cost/confidence/citation/review flag),
so a human can scan across a row and see how every contractor answered the
same line item, side by side against the truth.

Design (confirmed with user):
- Wide format: one row per line_item_id, one column-group per contractor —
  not long/tidy — because the primary use case here is a human visually
  comparing contractors against each other and against ground truth on one
  screen/sheet, not programmatic pivoting (Stage 5 will consume the JSON
  form directly for that).
- Ground truth IS included in this matrix (gt_status, gt_cost, gt_notes)
  for human eyeballing. This is presentation only — Stage 5's actual
  accuracy scoring is a separate, more careful computation and does not
  read from this matrix; it reads Stage 0 and Stage 3.5 output directly.
- Output both a JSON (nested, one object per line item with a
  `contractors` dict keyed by contractor name — easy for later stages or
  scripts to consume without re-parsing a flattened CSV) and a CSV (wide,
  flattened, for opening directly in Excel/Sheets).

Line-item matching: contractor results are joined to ground truth rows by
`line_item_id`. If a contractor's results reference a line_item_id that
doesn't exist in the ground truth (e.g. a Stage 0 parsing difference), that
row is skipped from the matrix with a warning printed — silently dropping
mismatched rows would hide a real data-quality problem.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.stage3_5_grounding_verification import GroundingResult


def load_ground_truth(ground_truth_path: Path) -> dict[str, dict]:
    """Load a Stage 0 ground-truth JSON file, indexed by line_item_id."""
    with open(ground_truth_path) as f:
        data = json.load(f)
    return {item["line_item_id"]: item for item in data["line_items"]}


def load_contractor_results(results_path: Path) -> list[GroundingResult]:
    """
    Load a JSON file of Stage 3.5 GroundingResult records for one
    contractor (expected shape: a list of dicts matching GroundingResult's
    fields, e.g. as produced by json.dump([asdict(r) for r in results], f)
    in whatever orchestration script runs Stages 2/3/3.5 per contractor).
    """
    with open(results_path) as f:
        raw = json.load(f)
    return [GroundingResult(**r) for r in raw]


def build_scope_matrix(
    ground_truth: dict[str, dict],
    contractor_results: dict[str, list[GroundingResult]],
) -> list[dict]:
    """
    Build the nested (JSON-ready) scope matrix: one dict per line item, in
    ground-truth order, each with a `contractors` sub-dict keyed by
    contractor name holding that contractor's result for this line item (or
    None if that contractor has no result for this line item at all —
    distinct from a result that exists but is NOT_MENTIONED).
    """
    contractor_names = sorted(contractor_results.keys())

    # Index each contractor's results by line_item_id for fast lookup.
    indexed: dict[str, dict[str, GroundingResult]] = {}
    for contractor_name, results in contractor_results.items():
        by_id = {}
        for r in results:
            if r.line_item_id in by_id:
                print(
                    f"WARNING: contractor '{contractor_name}' has more than one result for "
                    f"line_item_id '{r.line_item_id}' — keeping the last one seen.",
                    file=sys.stderr,
                )
            by_id[r.line_item_id] = r
        indexed[contractor_name] = by_id

    # Flag (but don't silently drop) any contractor result referencing a
    # line_item_id absent from ground truth — a real data-quality signal.
    gt_ids = set(ground_truth.keys())
    for contractor_name, by_id in indexed.items():
        unknown_ids = set(by_id.keys()) - gt_ids
        for uid in sorted(unknown_ids):
            print(
                f"WARNING: contractor '{contractor_name}' has a result for line_item_id "
                f"'{uid}' which does not exist in ground truth — this row will be excluded "
                f"from the matrix. Check for a Stage 0 parsing mismatch.",
                file=sys.stderr,
            )

    matrix = []
    for line_item_id, gt in ground_truth.items():
        row = {
            "line_item_id": line_item_id,
            "description": gt.get("description"),
            "section": gt.get("section"),
            "ground_truth": {
                "status": gt.get("ground_truth_status"),
                "cost": gt.get("ground_truth_cost"),
                "notes": gt.get("ground_truth_notes"),
            },
            "contractors": {},
        }
        for contractor_name in contractor_names:
            result = indexed[contractor_name].get(line_item_id)
            row["contractors"][contractor_name] = _grounding_result_to_dict(result)
        matrix.append(row)

    return matrix


def _grounding_result_to_dict(result: Optional[GroundingResult]) -> Optional[dict]:
    if result is None:
        return None
    return {
        "status": result.status,
        "cost_value": result.cost_value,
        "cost_type": result.cost_type,
        "comments": result.comments,
        "confidence": result.confidence,
        "citation_page_number": result.citation_page_number,
        "citation_verbatim": result.citation_verbatim,
        "citation_section_label": result.citation_section_label,
        "grounding_score": result.grounding_score,
        "retrieval_tier": result.retrieval_tier,
        "candidate_pages_sent": result.candidate_pages_sent,
        "needs_human_review": result.needs_human_review,
        "flag_reason": result.flag_reason,
    }


def write_matrix_json(matrix: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"num_line_items": len(matrix), "line_items": matrix}, f, indent=2)


def write_matrix_csv(matrix: list[dict], contractor_names: list[str], out_path: Path) -> None:
    """
    Flatten the nested matrix into the wide CSV layout:
    line_item_id | description | section | gt_status | gt_cost | gt_notes |
    {contractor}_status | {contractor}_cost | {contractor}_confidence |
    {contractor}_citation_page | {contractor}_citation_text |
    {contractor}_needs_review | {contractor}_flag_reason
    (repeated per contractor, in sorted contractor-name order)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["line_item_id", "description", "section", "gt_status", "gt_cost", "gt_notes"]
    for name in contractor_names:
        fieldnames += [
            f"{name}_status",
            f"{name}_cost",
            f"{name}_confidence",
            f"{name}_citation_page",
            f"{name}_citation_text",
            f"{name}_needs_review",
            f"{name}_flag_reason",
        ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in matrix:
            flat = {
                "line_item_id": row["line_item_id"],
                "description": row["description"],
                "section": row["section"],
                "gt_status": row["ground_truth"]["status"],
                "gt_cost": row["ground_truth"]["cost"],
                "gt_notes": row["ground_truth"]["notes"],
            }
            for name in contractor_names:
                c = row["contractors"].get(name)
                if c is None:
                    flat[f"{name}_status"] = None
                    flat[f"{name}_cost"] = None
                    flat[f"{name}_confidence"] = None
                    flat[f"{name}_citation_page"] = None
                    flat[f"{name}_citation_text"] = None
                    flat[f"{name}_needs_review"] = None
                    flat[f"{name}_flag_reason"] = None
                else:
                    flat[f"{name}_status"] = c["status"]
                    flat[f"{name}_cost"] = c["cost_value"]
                    flat[f"{name}_confidence"] = c["confidence"]
                    flat[f"{name}_citation_page"] = c["citation_page_number"]
                    flat[f"{name}_citation_text"] = c["citation_verbatim"]
                    flat[f"{name}_needs_review"] = c["needs_human_review"]
                    flat[f"{name}_flag_reason"] = c["flag_reason"]
            writer.writerow(flat)


# ----------------------------------------------------------------------------
# Standalone CLI
# ----------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: Aggregate results into a scope matrix")
    parser.add_argument("ground_truth_path", type=Path, help="Path to Stage 0 ground-truth JSON")
    parser.add_argument(
        "contractor_results",
        nargs="+",
        help="One or more 'name=path.json' pairs, e.g. AAPL_10K=results/AAPL_10K.json",
    )
    parser.add_argument("--out-json", type=Path, default=Path("data/scope_matrix.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/scope_matrix.csv"))
    args = parser.parse_args()

    contractor_paths = {}
    for pair in args.contractor_results:
        if "=" not in pair:
            print(f"ERROR: expected 'name=path.json', got '{pair}'", file=sys.stderr)
            sys.exit(1)
        name, path_str = pair.split("=", 1)
        contractor_paths[name] = Path(path_str)

    ground_truth = load_ground_truth(args.ground_truth_path)
    contractor_results = {
        name: load_contractor_results(path) for name, path in contractor_paths.items()
    }

    matrix = build_scope_matrix(ground_truth, contractor_results)
    write_matrix_json(matrix, args.out_json)
    write_matrix_csv(matrix, sorted(contractor_paths.keys()), args.out_csv)

    print(f"Wrote {len(matrix)} line items to {args.out_json} and {args.out_csv}")


if __name__ == "__main__":
    main()
