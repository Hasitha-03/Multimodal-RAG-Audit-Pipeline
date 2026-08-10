"""
Stage 0 — Master Scope Sheet Ingestion (PDF)

Master Scope Sheets are PDFs exported from Excel (Excel-exported "Project
Scope Sheet" templates observed in practice). Each PDF page contains one or
more visually-tabled sections (BASE BID, SCHEDULE/PRICING & BILLING
REQUIREMENTS, GENERAL ITEMS, PROJECT SPECIFIC SCOPE, MANDATORY CONTRACT
ALTERNATES, LABOR & EQUIPMENT RATES, MANPOWER, etc.), each with numbered
line items: an id, a description, and a status/cost value that is heavily
overloaded in practice (Included / Excluded / TAX EXEMPT / Y / percentages /
unit rates / blank).

This script parses those PDFs into a structured JSON ground-truth file: one
record per scorable line item. This ground truth is NEVER passed to the VLM
(Stage 3) — it is held back exclusively for downstream comparison in Stage 5.

Usage:
    python stage0_scope_ingestion.py <path_to_scope_sheet.pdf> [--out OUTPUT.json]

Why native text, not pdfplumber table extraction:
    Table-grid detection was tested against real scope sheets and found
    unreliable: some pages split one logical table into 2-4 separate
    pdfplumber Table objects (column boundaries detected inconsistently),
    and on at least one page a table lost its status/cost column entirely.
    Native per-page text, by contrast, reliably keeps a line item's number,
    description, and status/cost together as one contiguous text block (the
    row number and trailing status appear on the *same physical line* even
    though the description wraps across several lines above/below it). This
    script reconstructs line items directly from native text using the
    row-number pattern, rather than from extracted tables.

Design notes / decisions confirmed with the user:
- Rows under "MANDATORY CONTRACT ALTERNATES", "LABOR & EQUIPMENT RATES",
  "MANPOWER", and any row whose number is "#REF!" (a broken Excel cell
  reference baked into the PDF export) are EXCLUDED from the ground-truth
  set — only core scope items are scored.
- Status is normalized to only three buckets: INCLUDED / EXCLUDED /
  NOT_MENTIONED. Any other raw value (TAX EXEMPT, "Y", "100% MBE",
  "$120/HR", "5 DAYS/FL", etc.) is treated as INCLUDED, since in every
  observed case such values appear in rows that were affirmatively
  answered/confirmed rather than skipped, and the raw text is preserved
  verbatim in `ground_truth_notes` for a human reviewer to see.
- Each line item records which section/table (e.g. "GENERAL ITEMS",
  "PROJECT SPECIFIC SCOPE") it came from, since sections carry real meaning.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pdfplumber


# ----------------------------------------------------------------------------
# Section detection
# ----------------------------------------------------------------------------

# Section header text seen in real scope sheets (appears as the first cell
# of a table's header row, e.g. ['SCHEDULE, PRICING & BILLING REQUIREMENTS',
# None, 'COMMENTS /\nQTY', 'COST']). Matched case-insensitively against a
# whitespace-collapsed version of the cell text.
SECTION_HEADERS = [
    "SCHEDULE, PRICING & BILLING REQUIREMENTS",
    "GENERAL ITEMS",
    "PROJECT SPECIFIC SCOPE",
    "MANDATORY CONTRACT ALTERNATES",
    "LABOR & EQUIPMENT RATES",
    "MANPOWER",
    "BASE BID",
]

# Sections whose rows are excluded from the scorable ground-truth set
# entirely (per user decision: alternates/rates/manpower are not core scope
# items to be evaluated against contractor bids).
EXCLUDED_SECTIONS = {
    "MANDATORY CONTRACT ALTERNATES",
    "LABOR & EQUIPMENT RATES",
    "MANPOWER",
}


def _normalize_cell(cell) -> str:
    return re.sub(r"\s+", " ", str(cell or "").strip())


def _match_section_header(cell_text: str) -> Optional[str]:
    norm = _normalize_cell(cell_text).upper()
    for header in SECTION_HEADERS:
        if norm == header or norm.startswith(header):
            return header
    return None


# A line item's id cell is either digits or the literal "#REF!" (a broken
# Excel cell reference baked into the PDF export).
ID_CELL_RE = re.compile(r"^(\d+|#REF!)$")


# ----------------------------------------------------------------------------
# Table-based row extraction
# ----------------------------------------------------------------------------

def _is_item_row(row: list) -> bool:
    """True if this table row's first cell looks like a line-item id."""
    if not row:
        return False
    first = _normalize_cell(row[0])
    return bool(ID_CELL_RE.match(first))


def _pair_split_tables(tables: list[list[list]]) -> list[dict]:
    """
    Some pages render one logical table as two adjacent pdfplumber tables:
    a left one (id, description) and a right one (comments/qty, cost),
    because pdfplumber detects a column-boundary gap as a table split. This
    function detects that pattern — two consecutive tables with identical
    row counts, where the first table's rows are 2-col id+description and
    the second's are 2-col comments+cost — and merges them row-by-row.

    Returns a list of {"table": merged_4col_rows, "header": header_row_or_None}
    dicts, one per logical table found on the page (whether it was split
    or not).
    """
    logical_tables = []
    i = 0
    while i < len(tables):
        t = tables[i]
        if not t:
            i += 1
            continue

        # Does this table's first data-looking row have 2 columns and start
        # with an id? If there's a next table with the SAME row count, try
        # pairing them.
        if (
            i + 1 < len(tables)
            and tables[i + 1]
            and len(t) == len(tables[i + 1])
            and all(len(r) == 2 for r in t if r)
            and all(len(r) == 2 for r in tables[i + 1] if r)
        ):
            left, right = t, tables[i + 1]
            merged = []
            for lrow, rrow in zip(left, right):
                lrow = lrow or [None, None]
                rrow = rrow or [None, None]
                merged.append([lrow[0], lrow[1], rrow[0], rrow[1]])
            logical_tables.append({"table": merged, "header": None})
            i += 2
            continue

        logical_tables.append({"table": t, "header": None})
        i += 1

    return logical_tables


def extract_items_from_page(page, current_section: Optional[str]) -> tuple[list[dict], Optional[str]]:
    """
    Extract raw line-item dicts (id, description, raw_status, section) from
    one page's tables. Returns (items, updated_current_section).
    """
    items: list[dict] = []
    tables = page.extract_tables()
    if not tables:
        return items, current_section

    logical_tables = _pair_split_tables(tables)

    for lt in logical_tables:
        rows = lt["table"]
        for row in rows:
            if not row:
                continue

            # Section header rows: first cell matches a known header text.
            # These appear as the header row of a table, e.g.
            # ['GENERAL ITEMS', None, 'COMMENTS /\nQTY', 'COST'].
            section_match = _match_section_header(row[0]) if row[0] else None
            if section_match:
                current_section = section_match
                continue

            # Column-label header rows (e.g. ['COMMENTS / QTY', 'COST']
            # style leftovers, or a repeated header) — skip.
            first_cell = _normalize_cell(row[0])
            if first_cell.upper() in {"COMMENTS / QTY", "COST", "COMMENTS", "QTY"}:
                continue

            if not _is_item_row(row):
                continue  # not a numbered line item (blank row, stray text, etc.)

            line_item_id = first_cell
            description = _normalize_cell(row[1]) if len(row) > 1 else ""
            if not description:
                continue  # id with no description text is not a usable item

            # Status/cost can live in column 2 (comments/qty) and/or column
            # 3 (cost) depending on table width; concatenate whichever cells
            # are present and non-empty, preferring the cost column's text
            # when both are informative.
            trailing_cells = [
                _normalize_cell(c) for c in row[2:] if c is not None and _normalize_cell(c)
            ]
            raw_status = " ".join(trailing_cells) if trailing_cells else None

            items.append(
                {
                    "line_item_id": line_item_id,
                    "description": description,
                    "raw_status": raw_status,
                    "section": current_section,
                }
            )

    return items, current_section


# ----------------------------------------------------------------------------
# Status / cost normalization
# ----------------------------------------------------------------------------

# Matches "Excluded", "NOT INCLUDED", "NOT INCLUDED (BY OTHERS)", "... EXCLUDED"
# trailing text — real scope sheets use both phrasings interchangeably for
# the same meaning (this item is not this contractor's responsibility).
_STATUS_EXCLUDE_PATTERNS = re.compile(r"\b(excl(uded)?|not\s+includ(ed)?)\b", re.I)
_CURRENCY_RE = re.compile(r"\$\s?[\d,]+(?:\.\d+)?")


def normalize_status(raw_status: Optional[str]) -> str:
    """
    Map a raw status/cost cell string to INCLUDED / EXCLUDED / NOT_MENTIONED,
    per the confirmed rule: explicit "Excluded" / "Not Included" (in any
    phrasing/case, with or without trailing qualifiers like "(BY OTHERS)")
    maps to EXCLUDED; blank/dash/None maps to NOT_MENTIONED; every other
    non-empty value (Included, Y, TAX EXEMPT, percentages, unit rates, etc.)
    maps to INCLUDED, since in every observed real-world case such values
    appear on rows that were affirmatively answered.
    """
    if raw_status is None:
        return "NOT_MENTIONED"
    s = raw_status.strip()
    if not s or s in {"-", "—", "N/A", "n/a", "TBD"}:
        return "NOT_MENTIONED"
    if _STATUS_EXCLUDE_PATTERNS.search(s):
        return "EXCLUDED"
    return "INCLUDED"


def extract_cost(raw_status: Optional[str]) -> Optional[float]:
    """
    Pull a real dollar figure out of the raw status/cost text, if present
    (e.g. "$224,050.00" -> 224050.0). Returns None for everything else
    (Included, TAX EXEMPT, percentages, unit rates like "$120/HR" which are
    per-unit not line-item totals, blank, etc.) — never fabricated or
    allocated, matching the same "never allocate" rule used for VLM output.
    """
    if not raw_status:
        return None
    # Exclude unit-rate patterns like "$120/HR" or "$300/Flat" — these are
    # rates, not line-item totals, and would misrepresent cost if captured.
    if re.search(r"\$\s?[\d,.]+\s*/", raw_status):
        return None
    m = _CURRENCY_RE.search(raw_status)
    if not m:
        return None
    cleaned = re.sub(r"[^\d.]", "", m.group(0))
    try:
        return float(cleaned)
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Record structure
# ----------------------------------------------------------------------------

@dataclass
class ScopeLineItem:
    line_item_id: str
    description: str
    section: Optional[str]
    ground_truth_status: str
    ground_truth_cost: Optional[float]
    ground_truth_notes: Optional[str]  # raw status/cost text, verbatim


# ----------------------------------------------------------------------------
# Main ingestion logic
# ----------------------------------------------------------------------------

def build_ground_truth(pdf_path: Path) -> tuple[list[ScopeLineItem], list[str]]:
    """Returns (records, warnings)."""
    warnings: list[str] = []
    raw_items: list[dict] = []
    current_section: Optional[str] = None

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text_check = page.extract_text() or ""
            if not text_check.strip():
                warnings.append(
                    f"Page {page_num}: no extractable native text (possibly a scanned/image "
                    f"page) — line items on this page, if any, were NOT captured. This script "
                    f"does not OCR scope sheets; re-export or flag this page for manual entry."
                )
                continue
            page_items, current_section = extract_items_from_page(page, current_section)
            raw_items.extend(page_items)

    if not raw_items:
        return [], warnings + ["No line items were parsed from this document at all."]

    records: list[ScopeLineItem] = []
    seen_ids_in_section: dict[str, set] = {}

    for item in raw_items:
        line_item_id = item["line_item_id"]
        section = item["section"] or "UNSECTIONED"

        if line_item_id == "#REF!":
            continue  # excluded per user decision: broken Excel references
        if section in EXCLUDED_SECTIONS:
            continue  # excluded per user decision: alternates/rates/manpower

        description = item["description"]
        raw_status = item["raw_status"]

        status = normalize_status(raw_status)
        cost = extract_cost(raw_status)

        # Duplicate-id detection within a section is a real signal that
        # parsing merged something incorrectly (e.g. two adjacent items
        # collapsed) — flag rather than silently double-counting.
        seen = seen_ids_in_section.setdefault(section, set())
        if line_item_id in seen:
            warnings.append(
                f"Section '{section}': line item id '{line_item_id}' appears more than once "
                f"— possible parsing merge/split issue, review manually."
            )
        seen.add(line_item_id)

        records.append(
            ScopeLineItem(
                line_item_id=str(line_item_id),
                description=description,
                section=section,
                ground_truth_status=status,
                ground_truth_cost=cost,
                ground_truth_notes=raw_status,
            )
        )

    return records, warnings


def main():
    parser = argparse.ArgumentParser(description="Stage 0: Master scope sheet ingestion (PDF)")
    parser.add_argument("scope_sheet_path", type=Path, help="Path to master scope sheet PDF")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path (default: data/scope/<stem>_ground_truth.json)")
    args = parser.parse_args()

    if not args.scope_sheet_path.exists():
        print(f"ERROR: file not found: {args.scope_sheet_path}", file=sys.stderr)
        sys.exit(1)
    if args.scope_sheet_path.suffix.lower() != ".pdf":
        print(
            f"ERROR: expected a .pdf file, got '{args.scope_sheet_path.suffix}'. "
            f"This script parses PDF Master Scope Sheets only.",
            file=sys.stderr,
        )
        sys.exit(1)

    records, warnings = build_ground_truth(args.scope_sheet_path)

    if not records:
        print("ERROR: no usable line items were extracted. Check the PDF's structure against this script's assumptions.", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or (
        Path(__file__).resolve().parent.parent / "data" / "scope" / f"{args.scope_sheet_path.stem}_ground_truth.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source_file": str(args.scope_sheet_path),
        "num_line_items": len(records),
        "warnings": warnings,
        "line_items": [asdict(r) for r in records],
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(records)} ground-truth line items to {out_path}")
    if warnings:
        print(f"\n{len(warnings)} warning(s) — review before using as eval ground truth:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
