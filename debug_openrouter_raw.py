"""
debug_openrouter_raw.py — One-off diagnostic script.

Calls OpenRouter directly for ONE line item against ONE contractor manifest,
and prints the raw, unparsed response text/object so we can see exactly what
the model is returning before Stage 3's JSON parser touches it.

Usage:
    export OPENROUTER_API_KEY='sk-or-...'
    python3 debug_openrouter_raw.py data/runs/<run_id>/contractors/<contractor>/manifest.json "line item text here"

If you don't remember the exact manifest path, run this first to find it:
    find data/runs -name manifest.json
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.stage2_semantic_retrieval import ContractorPageIndex
from src.stage3_vlm_orchestrator import OpenRouterProvider


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 debug_openrouter_raw.py <manifest.json> <line_item_text>")
        sys.exit(1)

    manifest_path = Path(sys.argv[1])
    line_item_text = sys.argv[2]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY first.")
        sys.exit(1)

    print(f"Loading manifest: {manifest_path}")
    contractor_index = ContractorPageIndex.from_manifest(manifest_path)
    images_root = manifest_path.parent.parent

    print(f"Getting Tier 1 candidate pages for: {line_item_text!r}")
    pages = contractor_index.get_tier1_candidates(line_item_text, k=5)
    print(f"Pages selected: {[p.page_number for p in pages]}")

    provider = OpenRouterProvider(api_key=api_key)
    messages = provider._build_messages(line_item_text, pages, images_root)

    print("\n--- Calling OpenRouter directly (bypassing retry/parse logic) ---\n")
    response = provider._client.chat.completions.create(
        model=provider._model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    print("=== FULL RAW RESPONSE OBJECT ===")
    print(response)
    print()
    print("=== response.choices[0].finish_reason ===")
    print(response.choices[0].finish_reason)
    print()
    print("=== response.choices[0].message.content (this is what gets parsed) ===")
    content = response.choices[0].message.content
    print(repr(content))
    print()
    print("=== Length of content ===")
    print(len(content) if content else 0)


if __name__ == "__main__":
    main()
