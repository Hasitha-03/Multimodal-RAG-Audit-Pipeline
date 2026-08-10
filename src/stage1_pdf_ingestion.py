"""
Stage 1 — Contractor PDF Image Rendering & Text Extraction

For each contractor bid PDF, this script:
  1. Renders every page to a high-resolution image (for the VLM in Stage 3).
  2. Extracts per-page text: tries the PDF's native text layer first
     (fast, cheap), and falls back to OCR only for pages where native
     extraction yields too little text (i.e. likely a scanned image page).

Output per contractor: a directory of page images + a single JSON manifest
containing per-page text and metadata, ready for Stage 2 (retrieval) to
index and Stage 3 (VLM) to consume.

Usage:
    python stage1_pdf_ingestion.py <path_to_contractor_pdf> [--dpi 200] [--out-dir DIR]

Design notes:
- "Native text first, OCR fallback per-page" means each PAGE is judged
  independently — a 40-page bid where only 3 pages are scanned images will
  only OCR those 3 pages, not the whole document.
- The fallback trigger is a minimum-character-count heuristic on the native
  extraction. This is a simple, inspectable rule rather than a black box;
  the threshold is a constant at the top of the file so it's easy to tune
  once you see real documents.
- Document-level fallback: if pdfplumber cannot open/read the file at all,
  or if its page count disagrees with pdf2image's (a sign the file is
  malformed or not a genuine PDF despite its extension — e.g. some
  "sample" files in this project turned out to be zip archives with a
  misleading .pdf extension), this script does NOT raise. It falls back to
  treating every page as needing OCR, using the images pdf2image already
  rendered successfully. This is recorded in the manifest as
  `forced_full_document_ocr: true` so it's visible downstream, not silent.
- Mac note: pdf2image/pdftoppm requires poppler. On macOS install via
  `brew install poppler` if `pdftoppm`/`pdftotext` are not already on PATH.
  Tesseract OCR requires `brew install tesseract`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pdfplumber
from pdf2image import convert_from_path

try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


# Minimum characters of native-extracted text before we consider a page
# "has real text" rather than "probably a scanned image needing OCR".
# Tune this once you've looked at a handful of real pages from this project.
MIN_NATIVE_TEXT_CHARS = 40

DEFAULT_DPI = 200  # balances legibility for the VLM against image size/cost


@dataclass
class PageRecord:
    page_number: int  # 1-indexed
    image_path: str
    text: str
    text_source: str  # "native" | "ocr" | "none"
    char_count: int


def extract_native_text_per_page(pdf_path: Path) -> Optional[list[str]]:
    """
    Returns per-page native text, or None if pdfplumber could not open/read
    the file at all (corrupt/malformed PDF, wrong file type despite the
    .pdf extension, etc.) — signaling the caller to fall back to
    OCR-everything using the images pdf2image already rendered.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return [page.extract_text() or "" for page in pdf.pages]
    except Exception as e:
        print(f"WARNING: pdfplumber failed to open/read '{pdf_path.name}': {type(e).__name__}: {e}", file=sys.stderr)
        return None


def render_pages_to_images(pdf_path: Path, out_dir: Path, dpi: int) -> list[Path]:
    images = convert_from_path(str(pdf_path), dpi=dpi)
    paths = []
    for i, img in enumerate(images, start=1):
        img_path = out_dir / f"page_{i:03d}.png"
        img.save(img_path, "PNG")
        paths.append(img_path)
    return paths


def ocr_page(image_path: Path) -> str:
    if not _OCR_AVAILABLE:
        raise RuntimeError(
            "pytesseract/Pillow not available — install with: "
            "pip install pytesseract pillow --break-system-packages "
            "(and ensure tesseract binary is installed: brew install tesseract)"
        )
    img = Image.open(image_path)
    return pytesseract.image_to_string(img) or ""


def process_contractor_pdf(pdf_path: Path, out_root: Path, dpi: int = DEFAULT_DPI) -> dict:
    contractor_name = pdf_path.stem
    out_dir = out_root / contractor_name
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{contractor_name}] Rendering pages to images (dpi={dpi})...", file=sys.stderr)
    image_paths = render_pages_to_images(pdf_path, images_dir, dpi)
    num_pages = len(image_paths)

    print(f"[{contractor_name}] Extracting native text ({num_pages} pages)...", file=sys.stderr)
    native_texts = extract_native_text_per_page(pdf_path)

    # Fallback trigger #1: pdfplumber couldn't open the file at all.
    # Fallback trigger #2: pdfplumber opened it but disagrees with pdf2image
    # on page count (a real signal of a malformed/atypical PDF — the two
    # libraries parse the file structure differently, so a mismatch means at
    # least one of them is misreading it, and native text is not trustworthy
    # to align page-by-page against the images).
    # In either case: don't raise. Fall back to treating every page as if it
    # needed OCR, using the images pdf2image already rendered successfully.
    force_full_ocr = False
    if native_texts is None:
        print(
            f"[{contractor_name}] Falling back to full-document OCR: pdfplumber could not "
            f"read this file at all, but pdf2image successfully rendered {num_pages} page image(s).",
            file=sys.stderr,
        )
        force_full_ocr = True
    elif len(native_texts) != num_pages:
        print(
            f"[{contractor_name}] Falling back to full-document OCR: page count mismatch "
            f"(pdf2image saw {num_pages}, pdfplumber saw {len(native_texts)}). Native text "
            f"cannot be reliably aligned to page images, so every page will be OCR'd instead.",
            file=sys.stderr,
        )
        force_full_ocr = True

    if force_full_ocr:
        native_texts = [""] * num_pages  # treat as if every page had no native text

    records: list[PageRecord] = []
    ocr_count = 0

    for i, (img_path, native_text) in enumerate(zip(image_paths, native_texts), start=1):
        native_text = native_text.strip()

        if not force_full_ocr and len(native_text) >= MIN_NATIVE_TEXT_CHARS:
            records.append(
                PageRecord(
                    page_number=i,
                    image_path=str(img_path.relative_to(out_root)),
                    text=native_text,
                    text_source="native",
                    char_count=len(native_text),
                )
            )
            continue

        # Fallback: OCR this page (either because native text was too short,
        # or because we're in force_full_ocr mode for the whole document).
        reason = "forced full-document OCR" if force_full_ocr else f"native text too short ({len(native_text)} chars)"
        print(f"[{contractor_name}] Page {i}: {reason} — running OCR...", file=sys.stderr)
        try:
            ocr_text = ocr_page(img_path).strip()
        except RuntimeError as e:
            print(f"[{contractor_name}] Page {i}: OCR failed — {e}", file=sys.stderr)
            ocr_text = ""

        if ocr_text:
            ocr_count += 1
            records.append(
                PageRecord(
                    page_number=i,
                    image_path=str(img_path.relative_to(out_root)),
                    text=ocr_text,
                    text_source="ocr",
                    char_count=len(ocr_text),
                )
            )
        else:
            # Neither native nor OCR produced usable text. Keep the page
            # (the image still goes to the VLM) but flag it has no text
            # for Stage 2's retrieval index to skip/deprioritize.
            records.append(
                PageRecord(
                    page_number=i,
                    image_path=str(img_path.relative_to(out_root)),
                    text="",
                    text_source="none",
                    char_count=0,
                )
            )

    manifest = {
        "contractor_name": contractor_name,
        "source_pdf": str(pdf_path),
        "num_pages": num_pages,
        "dpi": dpi,
        "forced_full_document_ocr": force_full_ocr,
        "pages_ocr_fallback_count": ocr_count,
        "pages": [asdict(r) for r in records],
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    no_text_pages = [r.page_number for r in records if r.text_source == "none"]
    if no_text_pages:
        print(
            f"[{contractor_name}] WARNING: {len(no_text_pages)} page(s) have NO extractable text "
            f"(native or OCR): pages {no_text_pages}. These pages will still be sent to the VLM as "
            f"images but will not be findable via Stage 2's text-based retrieval.",
            file=sys.stderr,
        )

    print(f"[{contractor_name}] Done: {num_pages} pages, {ocr_count} required OCR fallback. Manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Contractor PDF ingestion")
    parser.add_argument("pdf_path", type=Path, help="Path to contractor bid PDF")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"Render DPI (default {DEFAULT_DPI})")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Root output directory (default: data/contractors/ relative to this script's parent)",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"ERROR: file not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_root = args.out_dir or (Path(__file__).resolve().parent.parent / "data" / "contractors")
    out_root.mkdir(parents=True, exist_ok=True)

    process_contractor_pdf(args.pdf_path, out_root, dpi=args.dpi)


if __name__ == "__main__":
    main()
