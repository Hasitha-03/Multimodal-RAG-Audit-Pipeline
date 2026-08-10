import os
from pathlib import Path

from src.stage3_vlm_orchestrator import VLMOrchestrator, OpenRouterProvider
from src.stage2_semantic_retrieval import ContractorPageIndex


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is not set. "
            "Set it with: export OPENROUTER_API_KEY='sk-or-...'"
        )

    provider = OpenRouterProvider(api_key=api_key, model="openrouter/free")
    orchestrator = VLMOrchestrator(provider=provider, tier1_k=5)

    # Adjust this path to point at a real Stage 1 manifest.json for one of
    # your contractor PDFs (produced by stage1_pdf_ingestion.py).
    manifest_path = Path("data/contractors/AAPL_10K/manifest.json")
    images_root = manifest_path.parent.parent

    contractor_index = ContractorPageIndex.from_manifest(manifest_path)

    line_item_text = "Extract total net revenue for fiscal year 2023"
    result = orchestrator.process_line_item(
        line_item_id="test-1",
        line_item_text=line_item_text,
        contractor_index=contractor_index,
        images_root=images_root,
    )

    print(f"Status: {result.vlm_output['status'] if result.vlm_output else 'N/A'}")
    print(f"Tier used: {result.retrieval_tier}")
    print(f"Pages sent: {result.candidate_pages_sent}")
    print(f"OK: {result.ok} | Needs human review: {result.needs_human_review}")
    if result.vlm_output:
        print(f"Citation: {result.vlm_output.get('citation')}")
        print(f"Comments: {result.vlm_output.get('comments')}")


if __name__ == "__main__":
    main()
