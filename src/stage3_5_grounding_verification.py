"""
Stage 3.5 — Citation Grounding Verification

Runs immediately after every Stage 3 VLM call, before results reach
aggregation (Stage 4). Verifies that the VLM's `verbatim_citation` for a
given line item actually exists on the page it claims to be from, using
fuzzy substring matching to tolerate minor OCR/whitespace/punctuation noise
without accepting fabricated or unrelated quotes.

Design (per user's confirmed spec, decided during Stage design phase):
- The check is a Python-side fuzzy substring match, NOT another LLM call —
  cheap, deterministic, and auditable.
- If the citation cannot be found on the claimed page (or is empty/missing
  when a status of INCLUDED/EXCLUDED requires one), the result is flagged
  `needs_human_review = True` with `flag_reason = "CITATION_GROUNDING_FAILURE"`.
  The VLM's original status/cost/comments are NOT discarded — they are kept
  alongside the flag so a human reviewer can see what the model claimed and
  check it against the source page themselves.
- NOT_MENTIONED results need no citation and are never flagged by this stage.
- A `grounding_score` (0-100 fuzzy match score) is recorded on every checked
  result — this feeds Stage 5's "Citation Grounding Success Rate" diagnostic
  metric (valid citations / total extracted citations).

Library choice: rapidfuzz, using fuzz.partial_ratio — the right primitive
because a citation is expected to be a SUBSTRING of a page's full text, not
a match against the whole page as one string (which is what a plain
Levenshtein ratio would compute, and would always score low regardless of
citation quality). Verified against realistic cases before writing this:
  - Exact real citation vs. its source page: partial_ratio ~100
  - Same citation with OCR-style noise (extra space, dropped punctuation):
    partial_ratio ~96
  - A legitimately paraphrased-but-real near-quote: partial_ratio ~94
  - A fabricated but topically plausible quote: partial_ratio ~70
  - A wholly unrelated fabricated quote: partial_ratio ~49
  This gives a clean separation band; DEFAULT_GROUNDING_THRESHOLD = 88 sits
  comfortably between the "real, even if slightly noisy" cluster (94-100)
  and the "fabricated" cluster (<=70), with room to retune once Stage 5 is
  run against real ground truth.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

from src.stage3_vlm_orchestrator import LineItemResult


DEFAULT_GROUNDING_THRESHOLD = 88.0


@dataclass
class GroundingResult:
    """Final, Stage-4-ready record for one (line_item, contractor) pair,
    combining the VLM's output with the grounding verification outcome."""
    line_item_id: str
    contractor_name: str
    status: Optional[str]
    cost_value: Optional[float]
    cost_type: Optional[str]
    comments: Optional[str]
    confidence: Optional[str]
    citation_page_number: Optional[int]
    citation_verbatim: Optional[str]
    citation_section_label: Optional[str]
    grounding_score: Optional[float]  # 0-100, None if no citation was expected/checked
    retrieval_tier: str
    candidate_pages_sent: list[int]
    needs_human_review: bool
    flag_reason: Optional[str]


def _normalize_text(text: str) -> str:
    """
    Collapse whitespace and normalize a few punctuation variants that OCR
    and PDF text extraction commonly render inconsistently (curly vs.
    straight quotes, en/em dashes vs. hyphens). This runs on BOTH the
    citation and the page text before fuzzy matching, so it never changes
    the comparison's fairness — it just removes noise that isn't a
    meaningful difference.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
    text = text.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash
    return text


def verify_citation(
    verbatim_citation: Optional[str],
    claimed_page_number: Optional[int],
    page_texts_by_number: dict[int, str],
    threshold: float = DEFAULT_GROUNDING_THRESHOLD,
) -> tuple[bool, Optional[float], Optional[str]]:
    """
    Check whether `verbatim_citation` actually appears (fuzzily) on the page
    numbered `claimed_page_number`, using the page text captured in Stage 1.

    Returns (is_grounded, score, reason_if_not_grounded):
      - is_grounded: True if the fuzzy match score meets `threshold`.
      - score: the best partial_ratio score found (0-100), or None if there
        was nothing to check (empty citation / page not found).
      - reason_if_not_grounded: short machine-readable reason string when
        is_grounded is False, else None.
    """
    if not verbatim_citation or not verbatim_citation.strip():
        return False, None, "EMPTY_CITATION"

    if claimed_page_number is None:
        return False, None, "MISSING_PAGE_NUMBER"

    page_text = page_texts_by_number.get(claimed_page_number)
    if page_text is None:
        return False, None, f"PAGE_NOT_FOUND(page={claimed_page_number})"

    if not page_text.strip():
        return False, 0.0, f"PAGE_HAS_NO_TEXT(page={claimed_page_number})"

    norm_citation = _normalize_text(verbatim_citation)
    norm_page = _normalize_text(page_text)

    score = fuzz.partial_ratio(norm_citation, norm_page)

    if score >= threshold:
        return True, score, None
    return False, score, f"SCORE_BELOW_THRESHOLD({score:.1f}<{threshold})"


def apply_grounding_check(
    result: LineItemResult,
    page_texts_by_number: dict[int, str],
    threshold: float = DEFAULT_GROUNDING_THRESHOLD,
) -> GroundingResult:
    """
    Take one Stage 3 LineItemResult and produce a Stage-4-ready
    GroundingResult, running the citation grounding check when applicable.

    Rules:
    - If Stage 3 already failed (result.ok == False, e.g. VLM output never
      parsed), pass the failure straight through — there's nothing to
      ground-check, and the existing flag_reason is preserved as-is.
    - If status == "NOT_MENTIONED": no citation is expected. Not flagged by
      this stage regardless of whether citation fields are null (they
      should be, per the system prompt, but this stage doesn't enforce
      that — a non-null citation on a NOT_MENTIONED item is unusual but not
      this check's concern).
    - If status is "INCLUDED" or "EXCLUDED": a citation IS expected. Run
      verify_citation(). If it fails, set needs_human_review=True and
      flag_reason="CITATION_GROUNDING_FAILURE" — but KEEP the VLM's
      status/cost/comments so a reviewer can see what was claimed.
    """
    if not result.ok or result.vlm_output is None:
        # Stage 3 itself failed (parse failure, API error, etc.) — pass the
        # failure through unchanged. Nothing here to ground-check.
        return GroundingResult(
            line_item_id=result.line_item_id,
            contractor_name=result.contractor_name,
            status=None,
            cost_value=None,
            cost_type=None,
            comments=None,
            confidence=None,
            citation_page_number=None,
            citation_verbatim=None,
            citation_section_label=None,
            grounding_score=None,
            retrieval_tier=result.retrieval_tier,
            candidate_pages_sent=result.candidate_pages_sent,
            needs_human_review=result.needs_human_review,
            flag_reason=result.flag_reason,
        )

    output = result.vlm_output
    status = output.get("status")
    citation = output.get("citation") or {}
    page_number = citation.get("page_number")
    verbatim = citation.get("verbatim_citation")
    section_label = citation.get("section_label")

    needs_human_review = False
    flag_reason = None
    grounding_score = None

    if status in ("INCLUDED", "EXCLUDED"):
        is_grounded, grounding_score, reason = verify_citation(
            verbatim, page_number, page_texts_by_number, threshold=threshold
        )
        if not is_grounded:
            needs_human_review = True
            flag_reason = "CITATION_GROUNDING_FAILURE"
            print(
                f"  [{result.contractor_name}] item {result.line_item_id}: "
                f"CITATION_GROUNDING_FAILURE ({reason}) — status '{status}' kept, flagged for review.",
                file=sys.stderr,
            )
    # status == "NOT_MENTIONED": no citation expected, nothing to check.

    return GroundingResult(
        line_item_id=result.line_item_id,
        contractor_name=result.contractor_name,
        status=status,
        cost_value=output.get("cost_value"),
        cost_type=output.get("cost_type"),
        comments=output.get("comments"),
        confidence=output.get("confidence"),
        citation_page_number=page_number,
        citation_verbatim=verbatim,
        citation_section_label=section_label,
        grounding_score=grounding_score,
        retrieval_tier=result.retrieval_tier,
        candidate_pages_sent=result.candidate_pages_sent,
        needs_human_review=needs_human_review,
        flag_reason=flag_reason,
    )


def build_page_texts_map(manifest_path: Path) -> dict[int, str]:
    """
    Load a Stage 1 manifest.json and build the {page_number: text} map that
    verify_citation()/apply_grounding_check() need. Uses whichever text
    Stage 1 captured for that page (native or OCR) — text_source == "none"
    pages simply won't be found in this map, which correctly causes any
    citation claiming that page to fail grounding (PAGE_NOT_FOUND or
    PAGE_HAS_NO_TEXT), since there's no text to have grounded it in.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    return {p["page_number"]: p["text"] for p in manifest["pages"]}


# ----------------------------------------------------------------------------
# Standalone smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3.5 standalone smoke test")
    parser.add_argument("manifest_path", type=Path, help="Path to a Stage 1 contractor manifest.json")
    parser.add_argument("page_number", type=int, help="Page number the citation claims to be from")
    parser.add_argument("citation_text", type=str, help="The verbatim_citation text to check")
    parser.add_argument("--threshold", type=float, default=DEFAULT_GROUNDING_THRESHOLD)
    args = parser.parse_args()

    page_texts = build_page_texts_map(args.manifest_path)
    is_grounded, score, reason = verify_citation(
        args.citation_text, args.page_number, page_texts, threshold=args.threshold
    )
    print(f"Grounded: {is_grounded}")
    print(f"Score: {score}")
    print(f"Reason (if not grounded): {reason}")
