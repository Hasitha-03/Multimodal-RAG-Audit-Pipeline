"""
generate_matrix.py — Out-of-process Stage 2->3->3.5->4 batch runner

Same rationale as prepare_run.py: keeps torch/sentence-transformers/VLM
provider SDKs out of the Streamlit server process. Streamlit subprocesses
this; it prints progress lines to stdout and a final MATRIX_OK / MATRIX_ERROR.

Usage:
    python generate_matrix.py \
        --run-dir data/runs/run_20260810_120000 \
        --provider openrouter --model openai/gpt-4o-mini
    (API key is read from OPENROUTER_API_KEY / GEMINI_API_KEY env vars,
    which the parent Streamlit process should set on the subprocess env.)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2/3/3.5/4 full matrix, run out-of-process from Streamlit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=["openrouter", "gemini"], default="openrouter")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tier1-k", type=int, default=3)
    parser.add_argument("--grounding-threshold", type=float, default=88.0)
    args = parser.parse_args()

    run_dir: Path = args.run_dir

    try:
        from src.stage2_semantic_retrieval import ContractorPageIndex
        from src.stage3_vlm_orchestrator import VLMOrchestrator, OpenRouterProvider, GeminiProvider
        from src.stage3_5_grounding_verification import apply_grounding_check, build_page_texts_map, GroundingResult
        from src.stage4_aggregation import build_scope_matrix, write_matrix_json, write_matrix_csv
        from run_pipeline import build_provider  # reuses existing provider-construction logic

        gt_path = run_dir / "scope_ground_truth.json"
        with open(gt_path) as f:
            gt_data = json.load(f)
        line_items = gt_data["line_items"]
        ground_truth_indexed = {item["line_item_id"]: item for item in line_items}

        contractors_dir = run_dir / "contractors"
        contractor_names = (
            sorted(p.name for p in contractors_dir.iterdir() if p.is_dir())
            if contractors_dir.exists() else []
        )
        if not contractor_names:
            log("MATRIX_ERROR: no prepared contractors found in this run.")
            return 1

        provider = build_provider(args.provider, model=args.model)
        orchestrator = VLMOrchestrator(provider=provider, tier1_k=args.tier1_k)

        results_dir = run_dir / "results"
        contractor_results: dict[str, list[GroundingResult]] = {}

        for contractor_name in contractor_names:
            manifest_path = contractors_dir / contractor_name / "manifest.json"
            images_root = manifest_path.parent.parent
            contractor_index = ContractorPageIndex.from_manifest(manifest_path)
            page_texts = build_page_texts_map(manifest_path)

            log(f"'{contractor_name}': processing {len(line_items)} line items "
                f"({contractor_index.total_pages} pages)...")

            stage3_results = orchestrator.process_line_items_batch(
                [{"line_item_id": i["line_item_id"], "description": i["description"]} for i in line_items],
                contractor_index,
                images_root,
            )
            results = [apply_grounding_check(r, page_texts, threshold=args.grounding_threshold) for r in stage3_results]

            contractor_results[contractor_name] = results
            results_dir.mkdir(parents=True, exist_ok=True)
            with open(results_dir / f"{contractor_name}.json", "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)

            num_flagged = sum(1 for r in results if r.needs_human_review)
            log(f"  -> '{contractor_name}' done: {num_flagged}/{len(results)} flagged for review.")

        log("Stage 4: Building final audit matrix...")
        matrix = build_scope_matrix(ground_truth_indexed, contractor_results)
        write_matrix_json(matrix, run_dir / "scope_matrix.json")
        write_matrix_csv(matrix, contractor_names, run_dir / "scope_matrix.csv")

        log("MATRIX_OK")
        return 0

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"MATRIX_ERROR: {type(e).__name__}: {e}")
        log(tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())
