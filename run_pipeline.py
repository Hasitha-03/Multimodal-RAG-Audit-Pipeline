"""
run_pipeline.py — Pipeline Batch Runner

Automates Stage 2 (semantic retrieval) -> Stage 3 (VLM call, with Tier 1/2
retry) -> Stage 3.5 (citation grounding verification) across every scope
line item, for one or more contractors, and saves each contractor's
GroundingResult list to disk as JSON — ready for Stage 4 to aggregate.

Usage:
    python run_pipeline.py \
        data/scope_matrix.json \
        AAPL_10K="data/contractors/AAPL_10K/manifest.json" \
        --out-dir data/results \
        --provider gemini \
        --model gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.stage0_scope_ingestion import build_ground_truth
from src.stage2_semantic_retrieval import ContractorPageIndex
from src.stage3_vlm_orchestrator import VLMOrchestrator, VLMProvider, OpenRouterProvider, GeminiProvider
from src.stage3_5_grounding_verification import apply_grounding_check, build_page_texts_map, GroundingResult


def load_ground_truth_line_items(ground_truth_path: Path) -> list[dict]:
    """Load Stage 0's ground-truth JSON and return its line_items list."""
    with open(ground_truth_path) as f:
        data = json.load(f)
    return data["line_items"]


def build_provider(provider_name: str, model: Optional[str] = None) -> VLMProvider:
    """Build and return the selected VLMProvider instance."""
    if provider_name == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Set it with: export OPENROUTER_API_KEY='sk-or-...'"
            )
        kwargs = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        return OpenRouterProvider(**kwargs)

    elif provider_name == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set it with: export GEMINI_API_KEY='AIzaSy...'"
            )
        kwargs = {"api_key": api_key}
        if model:
            kwargs["model"] = model
        return GeminiProvider(**kwargs)

    else:
        raise ValueError(f"Unknown provider '{provider_name}'. Expected 'openrouter' or 'gemini'.")


def process_contractor(
    contractor_name: str,
    manifest_path: Path,
    line_items: list[dict],
    orchestrator: VLMOrchestrator,
    grounding_threshold: float,
) -> list[GroundingResult]:
    """
    Run Stage 2 -> 3 -> 3.5 for every line item against one contractor.
    """
    contractor_index = ContractorPageIndex.from_manifest(manifest_path)
    images_root = manifest_path.parent.parent
    page_texts = build_page_texts_map(manifest_path)

    results: list[GroundingResult] = []
    total = len(line_items)

    for i, item in enumerate(line_items, start=1):
        line_item_id = item["line_item_id"]
        line_item_text = item["description"]

        print(f"[{contractor_name}] ({i}/{total}) item {line_item_id}: {line_item_text[:60]!r}...", file=sys.stderr)

        try:
            stage3_result = orchestrator.process_line_item(
                line_item_id=line_item_id,
                line_item_text=line_item_text,
                contractor_index=contractor_index,
                images_root=images_root,
            )
        except Exception as e:
            print(f"  UNEXPECTED ERROR on item {line_item_id}: {type(e).__name__}: {e}", file=sys.stderr)
            results.append(
                GroundingResult(
                    line_item_id=line_item_id,
                    contractor_name=contractor_name,
                    status=None, cost_value=None, cost_type=None, comments=None, confidence=None,
                    citation_page_number=None, citation_verbatim=None, citation_section_label=None,
                    grounding_score=None, retrieval_tier="tier1", candidate_pages_sent=[],
                    needs_human_review=True, flag_reason="UNEXPECTED_RUNNER_ERROR",
                )
            )
            continue

        grounding_result = apply_grounding_check(stage3_result, page_texts, threshold=grounding_threshold)
        results.append(grounding_result)

        status_str = grounding_result.status or "FAILED"
        review_str = " [NEEDS REVIEW]" if grounding_result.needs_human_review else ""
        print(f"  -> {status_str}{review_str}", file=sys.stderr)

    return results


def save_contractor_results(contractor_name: str, results: list[GroundingResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{contractor_name}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run full SEC audit verification pipeline batch.")
    parser.add_argument("ground_truth_path", type=Path, help="Path to Stage 0 ground-truth JSON")
    parser.add_argument(
        "contractors",
        nargs="+",
        help="One or more 'name=path/to/manifest.json' pairs",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--provider", choices=["openrouter", "gemini"], default="gemini")
    parser.add_argument("--model", type=str, default=None, help="Model name override (e.g., gemini-2.0-flash)")
    parser.add_argument("--tier1-k", type=int, default=5)
    parser.add_argument("--grounding-threshold", type=float, default=88.0)
    args = parser.parse_args()

    if not args.ground_truth_path.exists():
        print(f"ERROR: ground truth file not found: {args.ground_truth_path}", file=sys.stderr)
        sys.exit(1)

    contractor_manifest_paths: dict[str, Path] = {}
    for pair in args.contractors:
        if "=" not in pair:
            print(f"ERROR: expected 'name=path.json', got '{pair}'", file=sys.stderr)
            sys.exit(1)
        name, path_str = pair.split("=", 1)
        contractor_manifest_paths[name] = Path(path_str)

    try:
        provider = build_provider(args.provider, model=args.model)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    orchestrator = VLMOrchestrator(provider=provider, tier1_k=args.tier1_k)
    line_items = load_ground_truth_line_items(args.ground_truth_path)
    print(f"Loaded {len(line_items)} ground-truth line items from {args.ground_truth_path}", file=sys.stderr)

    succeeded: list[str] = []
    skipped: list[tuple[str, str]] = []

    for contractor_name, manifest_path in contractor_manifest_paths.items():
        print(f"\n=== {contractor_name} ===", file=sys.stderr)

        if not manifest_path.exists():
            reason = f"manifest not found: {manifest_path}"
            print(f"SKIPPING {contractor_name}: {reason}", file=sys.stderr)
            skipped.append((contractor_name, reason))
            continue

        try:
            t0 = time.time()
            results = process_contractor(
                contractor_name, manifest_path, line_items, orchestrator, args.grounding_threshold
            )
            elapsed = time.time() - t0
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            print(f"SKIPPING {contractor_name} due to error: {reason}", file=sys.stderr)
            skipped.append((contractor_name, reason))
            continue

        out_path = save_contractor_results(contractor_name, results, args.out_dir)
        num_flagged = sum(1 for r in results if r.needs_human_review)
        print(
            f"[{contractor_name}] done in {elapsed:.1f}s: {len(results)} line items processed, "
            f"{num_flagged} flagged for human review. Saved to {out_path}",
            file=sys.stderr,
        )
        succeeded.append(contractor_name)

    print("\n=== Run summary ===", file=sys.stderr)
    print(f"Succeeded: {succeeded}", file=sys.stderr)
    if skipped:
        print(f"Skipped ({len(skipped)}):", file=sys.stderr)
        for name, reason in skipped:
            print(f"  - {name}: {reason}", file=sys.stderr)

    if not succeeded:
        print("ERROR: no contractors completed successfully.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
