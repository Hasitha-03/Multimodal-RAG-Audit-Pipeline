"""
prepare_run.py — Out-of-process Stage 0 -> 1 -> 2 prep runner

Exists so the Streamlit app never imports torch / sentence-transformers /
pdf2image directly in its own process. Streamlit launches this as a
subprocess instead; all heavy C-extension work (PyTorch, HF tokenizers,
poppler bindings) happens in a separate OS process with its own memory
space, so anything that crashes here can't take the Streamlit server down
with it.

Prints one line of progress per step to stdout (unbuffered), which the
Streamlit app reads and displays live. On success, prints a final line
"PREPARE_OK" and exits 0. On failure, prints "PREPARE_ERROR: <message>" to
stdout and exits 1 — the caller can rely on the exit code rather than
parsing stderr.

Usage:
    python prepare_run.py \
        --run-dir data/runs/run_20260810_120000 \
        --scope-sheet path/to/checklist.json_or.pdf \
        --contractor name1=path1.pdf --contractor name2=path2.pdf
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0/1/2 prep, run out-of-process from Streamlit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope-sheet", type=Path, required=True)
    parser.add_argument(
        "--contractor", action="append", default=[], dest="contractors",
        help="'name=path/to/file.pdf', repeatable",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    scope_sheet_path: Path = args.scope_sheet

    contractor_pdf_paths: dict[str, Path] = {}
    for pair in args.contractors:
        if "=" not in pair:
            log(f"PREPARE_ERROR: bad --contractor value '{pair}', expected name=path")
            return 1
        name, path_str = pair.split("=", 1)
        contractor_pdf_paths[name] = Path(path_str)

    try:
        # Imports are deliberately deferred until after arg parsing so a
        # bad invocation fails fast without paying PyTorch's import cost.
        from src.stage0_scope_ingestion import build_ground_truth
        from src.stage1_pdf_ingestion import process_contractor_pdf
        from src.stage2_semantic_retrieval import ContractorPageIndex

        log("Stage 0: Parsing master audit checklist...")
        ground_truth_out = run_dir / "scope_ground_truth.json"

        if scope_sheet_path.suffix.lower() == ".json":
            ground_truth_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(scope_sheet_path, ground_truth_out)
            with open(ground_truth_out) as f:
                records_data = json.load(f)
            num_items = len(records_data.get("line_items", []))
            log(f"  -> {num_items} ground-truth audit checklist items loaded.")
        else:
            records, warnings = build_ground_truth(scope_sheet_path)
            for w in warnings:
                log(f"  [warning] {w}")
            ground_truth_out.parent.mkdir(parents=True, exist_ok=True)
            with open(ground_truth_out, "w") as f:
                json.dump(
                    {
                        "source_file": str(scope_sheet_path),
                        "num_line_items": len(records),
                        "warnings": warnings,
                        "line_items": [asdict(r) for r in records],
                    },
                    f,
                    indent=2,
                )
            log(f"  -> {len(records)} ground-truth line items extracted.")

        contractors_dir = run_dir / "contractors"

        for contractor_name, pdf_path in contractor_pdf_paths.items():
            log(f"Stage 1: Rendering + extracting text for '{contractor_name}'...")
            renamed_pdf_path = pdf_path.parent / f"{contractor_name}.pdf"
            if pdf_path != renamed_pdf_path:
                shutil.copy(pdf_path, renamed_pdf_path)
            process_contractor_pdf(renamed_pdf_path, contractors_dir, dpi=200)

            log(f"Stage 2: Building semantic index for '{contractor_name}'...")
            manifest_path = contractors_dir / contractor_name / "manifest.json"
            ContractorPageIndex.from_manifest(manifest_path)
            log(f"  -> '{contractor_name}' ready for querying.")

        log("PREPARE_OK")
        return 0

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"PREPARE_ERROR: {type(e).__name__}: {e}")
        log(tb)
        return 1


if __name__ == "__main__":
    sys.exit(main())
