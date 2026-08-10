"""
Stage 2 — Semantic Retrieval for Candidate Pages

For a given (line item, contractor) pair, this module selects which pages of
that contractor's bid should be sent to the VLM in Stage 3 — Tier 1 first
(a small, high-confidence set), and a broader Tier 2 set as a fallback if
Stage 3 reports NOT_MENTIONED on the Tier 1 pages.

Design (confirmed with user):
- Embedding model: local `sentence-transformers/all-mpnet-base-v2`.
  No API calls, no per-query cost, works offline. Chosen because contractor
  bids paraphrase scope items heavily (e.g. "vendor to supply own equipment" vs.
  "skip provided by others"), which needs real semantic similarity, not
  keyword matching.
- Indexing: no vector database. At this project's scale (a few dozen pages
  per contractor, embedded once per contractor and queried once per line
  item) an in-memory NumPy matrix + cosine similarity is simpler, faster to
  build, and has zero extra infrastructure — a vector DB would only start
  earning its complexity at a much larger scale than this pipeline operates
  at today.
- Chunking granularity: one embedding per PAGE, not sub-page chunks. Stage 3
  operates on whole page images and Stage 3.5 grounds citations against
  whole pages, so keeping retrieval aligned to the same unit avoids a
  mismatch between "what was searched" and "what the VLM/grounding check
  actually sees."
- Tier 1: fixed top-k (k=5 default, confirmed with user), not a similarity
  threshold — thresholds on cosine similarity are dataset/model-specific and
  we don't have calibration data yet; fixed-k is simpler and easy to retune
  once Stage 5 gives us real precision/recall numbers.
- Tier 2 (fallback): every page NOT already sent in Tier 1. For documents
  small enough that Tier 1 + Tier 2 covers everything, this just means "send
  the rest of the document."
- Pages with text_source == "none" (Stage 1 found no extractable text, even
  after OCR) are excluded from the embedding index — they can't be
  semantically searched, but are noted so the caller knows they exist and
  may want to include them defensively in Tier 2.

This module has no CLI entry point of its own (unlike Stage 0/1) — it's a
library used by Stage 3's orchestration loop, since retrieval must happen
per (line item, contractor) pair as part of that loop, not as a standalone
batch step. A small __main__ smoke test is included at the bottom for
standalone sanity-checking against one contractor manifest.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TIER1_K = 5


@dataclass
class PageCandidate:
    page_number: int
    image_path: str
    text: str
    text_source: str  # "native" | "ocr" | "none"
    similarity: Optional[float]  # None for pages excluded from embedding (text_source == "none")


class ContractorPageIndex:
    """
    Holds embeddings for every page of one contractor's bid, built once from
    that contractor's Stage 1 manifest, then queried once per line item.

    Usage:
        index = ContractorPageIndex.from_manifest(manifest_path)
        tier1 = index.get_tier1_candidates(line_item_text, k=5)
        tier2 = index.get_tier2_candidates(already_sent_page_numbers=[p.page_number for p in tier1])
    """

    def __init__(self, contractor_name: str, pages: list[dict], embedder: "SentenceTransformer"):
        self.contractor_name = contractor_name
        self.embedder = embedder

        # Separate pages with usable text from those without (text_source == "none").
        self._embeddable_pages = [p for p in pages if p.get("text_source") != "none"]
        self._unembeddable_pages = [p for p in pages if p.get("text_source") == "none"]
        self._all_pages_by_number = {p["page_number"]: p for p in pages}

        if self._embeddable_pages:
            texts = [p["text"] for p in self._embeddable_pages]
            self._page_embeddings = self._embed(texts)  # (num_pages, dim), L2-normalized
        else:
            self._page_embeddings = np.zeros((0, 0))

    @classmethod
    def from_manifest(cls, manifest_path: Path, embedder: Optional["SentenceTransformer"] = None) -> "ContractorPageIndex":
        with open(manifest_path) as f:
            manifest = json.load(f)
        if embedder is None:
            embedder = get_shared_embedder()
        return cls(manifest["contractor_name"], manifest["pages"], embedder)

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize so dot product == cosine similarity
            show_progress_bar=False,
        )
        return embeddings

    def _embed_query(self, text: str) -> np.ndarray:
        return self.embedder.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    def get_tier1_candidates(self, line_item_text: str, k: int = DEFAULT_TIER1_K) -> list[PageCandidate]:
        """
        Return the top-k pages by cosine similarity to the line item text.
        If the contractor doc has fewer than k embeddable pages, returns all
        of them (ranked). If there are zero embeddable pages, returns [].
        """
        if not self._embeddable_pages:
            return []

        query_emb = self._embed_query(line_item_text)
        # Embeddings are L2-normalized, so dot product == cosine similarity.
        similarities = self._page_embeddings @ query_emb  # (num_pages,)

        ranked_indices = np.argsort(-similarities)[:k]
        results = []
        for idx in ranked_indices:
            page = self._embeddable_pages[idx]
            results.append(
                PageCandidate(
                    page_number=page["page_number"],
                    image_path=page["image_path"],
                    text=page["text"],
                    text_source=page["text_source"],
                    similarity=float(similarities[idx]),
                )
            )
        return results

    def get_tier2_candidates(self, already_sent_page_numbers: list[int]) -> list[PageCandidate]:
        """
        Return every page NOT already sent in Tier 1 — the broader fallback
        set used when Stage 3 reports NOT_MENTIONED on the Tier 1 pages.
        Includes text_source == "none" pages too (defensively): even though
        they can't be semantically ranked, the VLM can still visually
        inspect them, and skipping them entirely on the retry would mean
        never showing the model a real page that happens to have no
        extractable text.
        """
        already_sent = set(already_sent_page_numbers)
        remaining = [
            page for num, page in sorted(self._all_pages_by_number.items())
            if num not in already_sent
        ]

        # Rank the remaining embeddable pages by similarity too, if we have
        # a way to (we don't have the query text here by design — Tier 2 is
        # called with just page numbers already sent). Since Tier 2 is a
        # "send everything else" fallback rather than a ranked retrieval,
        # we return remaining pages in page-number order rather than
        # re-ranking, keeping this method's contract simple and predictable.
        results = []
        for page in remaining:
            results.append(
                PageCandidate(
                    page_number=page["page_number"],
                    image_path=page["image_path"],
                    text=page["text"],
                    text_source=page["text_source"],
                    similarity=None,
                )
            )
        return results

    @property
    def total_pages(self) -> int:
        return len(self._all_pages_by_number)

    @property
    def unembeddable_page_numbers(self) -> list[int]:
        return sorted(p["page_number"] for p in self._unembeddable_pages)


# ----------------------------------------------------------------------------
# Shared embedder (loaded once, reused across contractors within one run)
# ----------------------------------------------------------------------------

_shared_embedder: Optional["SentenceTransformer"] = None


def get_shared_embedder() -> "SentenceTransformer":
    """
    Loads sentence-transformers/all-mpnet-base-v2 once per process and
    reuses it — loading the model is the slow part (a few seconds), so
    Stage 3's orchestration loop should call this once and pass the result
    into every ContractorPageIndex it builds, rather than letting each
    index load its own copy.
    """
    global _shared_embedder
    if not _ST_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is not installed. Install with:\n"
            "  pip install sentence-transformers --break-system-packages\n"
            "The first run will download the model (~420MB) from Hugging Face; "
            "an internet connection is required once, then it's cached locally."
        )
    if _shared_embedder is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (first call only)...", file=sys.stderr)
        _shared_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _shared_embedder


# ----------------------------------------------------------------------------
# Standalone smoke test
# ----------------------------------------------------------------------------

def _smoke_test(manifest_path: Path, line_item_text: str, k: int = DEFAULT_TIER1_K):
    """Quick standalone check: given a Stage 1 manifest and a line item
    description, print the Tier 1 candidate pages and their similarity
    scores. Useful for sanity-checking retrieval quality by hand before
    wiring this into Stage 3's full loop."""
    index = ContractorPageIndex.from_manifest(manifest_path)
    print(f"Contractor: {index.contractor_name} ({index.total_pages} total pages, "
          f"{len(index.unembeddable_page_numbers)} unembeddable: {index.unembeddable_page_numbers})")
    print(f"Query: {line_item_text!r}\n")

    tier1 = index.get_tier1_candidates(line_item_text, k=k)
    print(f"Tier 1 candidates (top {k}):")
    for c in tier1:
        preview = c.text[:80].replace("\n", " ")
        print(f"  page {c.page_number} (sim={c.similarity:.3f}, source={c.text_source}): {preview}...")

    tier2 = index.get_tier2_candidates(already_sent_page_numbers=[c.page_number for c in tier1])
    print(f"\nTier 2 fallback would add {len(tier2)} more page(s): "
          f"{[c.page_number for c in tier2]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2 standalone smoke test")
    parser.add_argument("manifest_path", type=Path, help="Path to a Stage 1 contractor manifest.json")
    parser.add_argument("line_item_text", type=str, help="Line item description to search for")
    parser.add_argument("--k", type=int, default=DEFAULT_TIER1_K)
    args = parser.parse_args()

    _smoke_test(args.manifest_path, args.line_item_text, k=args.k)
