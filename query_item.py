"""
query_item.py — Out-of-process single (line_item, contractor) query

Same rationale as prepare_run.py / generate_matrix.py. Prints exactly one
line of JSON to stdout on success (the GroundingResult as a dict) and
nothing else on stdout — all progress/logging goes to stderr, so the
Streamlit caller can do `json.loads(stdout)` directly. Exit code 0 = ok,
1 = error (error message printed to stderr).

Usage:
    python query_item.py \
        --run-dir data/runs/run_20260810_120000 \
        --line-item-id AUDIT-001 \
        --line-item-text "Extract total net revenue..." \
        --contractor AAPL_10K \
        --provider openrouter --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Single line-item query, run out-of-process from Streamlit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--line-item-id", type=str, required=True)
    parser.add_argument("--line-item-text", type=str, required=True)
    parser.add_argument("--contractor", type=str, required=True)
    parser.add_argument("--provider", choices=["openrouter", "gemini"], default="openrouter")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tier1-k", type=int, default=3)
    args = parser.parse_args()

    try:
        from src.stage2_semantic_retrieval import ContractorPageIndex
        from src.stage3_vlm_orchestrator import VLMOrchestrator
        from src.stage3_5_grounding_verification import apply_grounding_check, build_page_texts_map
        from run_pipeline import build_provider

        manifest_path = args.run_dir / "contractors" / args.contractor / "manifest.json"
        images_root = manifest_path.parent.parent

        provider = build_provider(args.provider, model=args.model)
        orchestrator = VLMOrchestrator(provider=provider, tier1_k=args.tier1_k)

        contractor_index = ContractorPageIndex.from_manifest(manifest_path)
        page_texts = build_page_texts_map(manifest_path)

        stage3_result = orchestrator.process_line_item(
            line_item_id=args.line_item_id,
            line_item_text=args.line_item_text,
            contractor_index=contractor_index,
            images_root=images_root,
        )
        grounding_result = apply_grounding_check(stage3_result, page_texts)

        # Exactly one line of JSON on stdout — nothing else may go to stdout.
        print(json.dumps(asdict(grounding_result)))
        return 0

    except Exception as e:
        import traceback
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
