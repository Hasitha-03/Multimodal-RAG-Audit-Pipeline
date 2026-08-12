# Architecture

This document covers the reasoning behind each major design decision. For setup and usage, see [README.md](./README.md).

## The core problem

Financial filings are not clean text. A 10-K mixes native PDF text with scanned pages, multi-column tables, and layout that carries meaning — a lease schedule's structure is part of what makes it readable. Extract that page as plain text and the table often comes out scrambled: rows out of order, numbers detached from their labels. A pipeline that only reads extracted text will silently fail on pages like this, with no signal that anything went wrong.

Reading the page correctly still isn't enough on its own. A model can answer confidently whether it's reading the right page or not, and a citation alone doesn't fix that — a model can generate a citation that looks correct and isn't. The system needs a separate step that checks the citation against the actual source text, independent of the model that generated it.

## Why a vision-language model

The direct fix for the layout problem is to stop treating the document as text and start treating it as an image. A VLM is given the actual page — tables, grid lines, alignment intact — the same way a human reviewer would look at it.

This ruled out two simpler alternatives:

- **A text-only LLM over extracted text.** Cheaper and simpler, but it inherits every extraction error. A table that jumbles on extraction jumbles the model's answer too, with no way to tell the two failures apart.
- **A general-purpose Hugging Face OCR/layout model** (LayoutLM-style architectures). Reasonable for high-volume, single-format pipelines, but not the right fit here — source documents vary in layout from filer to filer, and a general VLM generalizes across that variation without task-specific fine-tuning, at the cost of a per-call API price instead of a self-hosted model.

The tradeoff is real: VLM calls cost more per page than text extraction and are slower than a local model. That's why the pipeline doesn't send every page to the model for every query.

## Why retrieval, not full-document context

A 10-K runs 100+ pages. Sending the whole document to the VLM for every checklist item is a cost and latency problem — the price and time scale linearly with document size and checklist length, which breaks down fast in a batch setting with dozens of line items across multiple filings.

The fix is retrieval: every page is embedded once locally, and for each checklist item, only the top few pages by cosine similarity are sent to the VLM.

Two decisions worth naming:

- **Fixed top-k over a similarity threshold.** A threshold needs calibration data to set correctly. Fixed-k has a simpler failure mode and is easy to retune once real accuracy numbers exist.
- **A two-tier fallback, not a single retrieval pass.** If the top candidate pages come back "not mentioned," the system retries with a broader page set before concluding the item genuinely isn't in the document — a small extra cost, only paid when the first pass comes back empty, against a meaningfully lower false-negative rate.

## Why grounding is a separate, non-model step

The VLM's citation is not automatically trusted. A separate step checks whether the quoted text actually exists on the page it claims to be from, using fuzzy string matching (`rapidfuzz`) against the source text extracted in ingestion — not another model call.

Grounding checks whether the claimed sentence is genuinely on the claimed page. It's not a perfect character-for-character match — OCR and PDF extraction introduce small noise, like an extra space or a curly quote instead of a straight one — so the match is fuzzy: a score from 0 to 100 for how well the claimed sentence shows up in the real page text. A real quote, even with minor noise, scores in the 90s. A fabricated quote scores much lower.

**Why a text match instead of a second AI call:** the obvious alternative is asking a second model to verify the first one's citation. That doesn't fully solve the problem — the second model can make the same kind of mistake as the first, confidently confirming a citation that isn't real. A deterministic text match against the real page can't do that. It's checking whether the exact words are physically present in text extracted directly from the document, with no model in the loop to be fooled. It's also faster and free to run.

If the score is high enough, the citation is trusted. If it's low, the result isn't discarded — it's flagged for human review, with the model's original answer kept alongside the flag, so the reviewer can see exactly what was claimed.

## Known limitations

**Grounding can pass perfectly while the answer is still wrong.** A citation can be verbatim and genuine — a real quote from the real page — while the model still misread which figure to extract, for instance picking the wrong column out of a multi-year comparison table. Grounding only ever answers one question: does this quote exist where the model says it does. It says nothing about whether the model understood what the quote meant.

**Self-reported confidence isn't independently checked.** The model labels each answer High, Medium, or Low confidence, and nothing in the pipeline verifies whether that label is warranted. A "High confidence" wrong answer currently gets no more scrutiny than a "Low confidence" one.

**Ground truth in the sample checklist isn't calibrated per filer.** The expected figures in the sample checklist were written against one specific company's disclosure. Comparing a different company's real filing against that fixed number will show a mismatch that reflects the sample data's limits, not necessarily pipeline error.

## What's next

Two directions worth pursuing first, because they're direct extensions of the limitations above:

**Gate the VLM call on OCR confidence.** Right now every candidate page goes to the VLM regardless of how clean it is. The ingestion stage already knows which pages needed OCR and which didn't — that signal could decide whether the VLM is even called, rather than the VLM reading every page by default. Clean, typed pages get read cheaply; only genuinely ambiguous pages escalate.

**Let a model decide the retrieval strategy per query.** Retrieval currently runs one fixed strategy — embed, compare, take the top-k — for every checklist item. A model deciding, per query, whether semantic search alone is sufficient or whether it should be combined with keyword search would adapt the strategy instead of applying the same approach regardless of what's being asked.
