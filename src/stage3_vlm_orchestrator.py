"""
Stage 3 — VLM Prompting & Orchestration Loop
"""

from __future__ import annotations

import json
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.stage2_semantic_retrieval import ContractorPageIndex, PageCandidate


# ----------------------------------------------------------------------------
# System Prompt & Schemas
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an Enterprise Financial Auditor AI analyzing SEC 10-K corporate filings.
Examine the provided filing page image(s) to verify the financial audit item.

Return valid JSON with the following structure:
{
  "status": "INCLUDED" | "EXCLUDED" | "NOT_MENTIONED",
  "cost_value": "Extracted monetary amount or key metric value (or null)",
  "cost_type": "Lump Sum / Footnote / Metric",
  "comments": "Detailed audit observation or management notes",
  "confidence": "High" | "Medium" | "Low",
  "citation_page_number": integer,
  "citation_verbatim": "Exact verbatim substring from the document page image",
  "citation_section_label": "Table name or section header"
}
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "line_item": {"type": "string"},
        "status": {"type": "string", "enum": ["INCLUDED", "EXCLUDED", "NOT_MENTIONED"]},
        "cost_value": {"type": "string", "nullable": True},
        "cost_type": {
            "type": "string",
            "enum": ["Lump Sum", "Footnote", "Metric", "Not Stated"],
            "nullable": True,
        },
        "comments": {"type": "string"},
        "citation": {
            "type": "object",
            "properties": {
                "page_number": {"type": "integer", "nullable": True},
                "verbatim_citation": {"type": "string", "nullable": True},
                "section_label": {"type": "string", "nullable": True},
            },
            "required": ["page_number", "verbatim_citation", "section_label"],
        },
        "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
    },
    "required": ["line_item", "status", "cost_value", "cost_type", "comments", "citation", "confidence"],
}


# ----------------------------------------------------------------------------
# Provider Abstraction
# ----------------------------------------------------------------------------

@dataclass
class VLMCallResult:
    """Raw result of one VLM call, before pipeline metadata is attached."""
    parsed_json: Optional[dict]
    raw_text: str
    ok: bool
    error: Optional[str] = None


class VLMProvider(ABC):
    @abstractmethod
    def call(self, line_item_text: str, pages: list[PageCandidate], images_root: Path) -> VLMCallResult:
        raise NotImplementedError


class GeminiProvider(VLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", max_retries: int = 3, retry_delay_seconds: float = 5.0):
        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds

    def _build_contents(self, line_item_text: str, pages: list[PageCandidate], images_root: Path) -> list:
        types = self._types
        parts = [types.Part.from_text(text=f"SCOPE LINE ITEM:\n{line_item_text}\n\nPage images follow, labeled by page number.")]
        for page in pages:
            img_path = images_root / page.image_path
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            parts.append(types.Part.from_text(text=f"--- Page {page.page_number} ---"))
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/png"))
        return [types.Content(role="user", parts=parts)]

    def call(self, line_item_text: str, pages: list[PageCandidate], images_root: Path) -> VLMCallResult:
        types = self._types
        contents = self._build_contents(line_item_text, pages, images_root)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=OUTPUT_SCHEMA,
            temperature=0.0,
        )

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                raw_text = response.text or ""
                parsed = _parse_json_response(raw_text)
                if parsed is not None:
                    return VLMCallResult(parsed_json=parsed, raw_text=raw_text, ok=True)
                last_error = f"Response was not valid JSON matching schema (attempt {attempt}): {raw_text[:200]!r}"
                print(f"WARNING: {last_error}", file=sys.stderr)

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 35
                    print(f"WARNING: Gemini rate limit hit (429). Sleeping {wait_time}s before retry...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    print(f"WARNING: Gemini API call failed (attempt {attempt}/{self._max_retries}): {last_error}", file=sys.stderr)

            if attempt < self._max_retries and "429" not in str(last_error):
                time.sleep(self._retry_delay_seconds * attempt)

        return VLMCallResult(parsed_json=None, raw_text="", ok=False, error=last_error)


class OpenRouterProvider(VLMProvider):
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str = None, max_retries: int = 2, retry_delay_seconds: float = 2.0):
        from openai import OpenAI
        self._OpenAI = OpenAI
        self._client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        self._model = model or self.DEFAULT_MODEL
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds

    def verify_model(self) -> str:
        return self._model

    def _build_messages(self, line_item_text: str, pages: list[PageCandidate], images_root: Path) -> list[dict]:
        import base64

        schema_instructions = (
            "Respond with ONLY a single JSON object (no markdown fences, no commentary) "
            "matching exactly this structure:\n"
            f"{json.dumps(OUTPUT_SCHEMA, indent=2)}"
        )

        user_content = [
            {
                "type": "text",
                "text": (
                    f"SCOPE LINE ITEM:\n{line_item_text}\n\n"
                    f"Page images follow, labeled by page number.\n\n{schema_instructions}"
                ),
            }
        ]
        for page in pages:
            img_path = images_root / page.image_path
            with open(img_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")
            user_content.append({"type": "text", "text": f"--- Page {page.page_number} ---"})
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                }
            )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def call(self, line_item_text: str, pages: list[PageCandidate], images_root: Path) -> VLMCallResult:
        messages = self._build_messages(line_item_text, pages, images_root)

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw_text = response.choices[0].message.content or ""
                parsed = _parse_json_response(raw_text)
                if parsed is not None:
                    return VLMCallResult(parsed_json=parsed, raw_text=raw_text, ok=True)
                last_error = f"Response was not valid JSON (attempt {attempt}): {raw_text[:200]!r}"
                print(f"WARNING: {last_error}", file=sys.stderr)
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                print(f"WARNING: OpenRouter API call failed (attempt {attempt}/{self._max_retries}): {last_error}", file=sys.stderr)
                if "404" in str(e):
                    break

            if attempt < self._max_retries:
                time.sleep(self._retry_delay_seconds * attempt)

        return VLMCallResult(parsed_json=None, raw_text="", ok=False, error=last_error)


def _parse_json_response(raw_text: str) -> Optional[dict]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# ----------------------------------------------------------------------------
# Orchestrator Loop
# ----------------------------------------------------------------------------

@dataclass
class LineItemResult:
    line_item_id: str
    contractor_name: str
    vlm_output: Optional[dict]
    retrieval_tier: str
    candidate_pages_sent: list[int]
    ok: bool
    needs_human_review: bool
    flag_reason: Optional[str]


class VLMOrchestrator:
    def __init__(self, provider: VLMProvider, tier1_k: int = 5):
        self.provider = provider
        self.tier1_k = tier1_k

    def process_line_item(
        self,
        line_item_id: str,
        line_item_text: str,
        contractor_index: ContractorPageIndex,
        images_root: Path,
    ) -> LineItemResult:
        contractor_name = contractor_index.contractor_name

        tier1_pages = contractor_index.get_tier1_candidates(line_item_text, k=self.tier1_k)
        result = self._call_and_parse(line_item_text, tier1_pages, images_root)
        tier_used = "tier1"
        pages_sent = [p.page_number for p in tier1_pages]

        if result.ok and result.parsed_json is not None and result.parsed_json.get("status") == "NOT_MENTIONED":
            tier2_pages = contractor_index.get_tier2_candidates(already_sent_page_numbers=pages_sent)
            if tier2_pages:
                print(f"  [{contractor_name}] item {line_item_id}: NOT_MENTIONED on Tier 1, retrying with {len(tier2_pages)} Tier 2 page(s)...", file=sys.stderr)
                tier2_result = self._call_and_parse(line_item_text, tier2_pages, images_root)
                if tier2_result.ok and tier2_result.parsed_json is not None:
                    result = tier2_result
                    tier_used = "tier2"
                    pages_sent = [p.page_number for p in tier2_pages]

        if not result.ok or result.parsed_json is None:
            return LineItemResult(
                line_item_id=line_item_id,
                contractor_name=contractor_name,
                vlm_output=None,
                retrieval_tier=tier_used,
                candidate_pages_sent=pages_sent,
                ok=False,
                needs_human_review=True,
                flag_reason="VLM_OUTPUT_PARSE_FAILURE",
            )

        return LineItemResult(
            line_item_id=line_item_id,
            contractor_name=contractor_name,
            vlm_output=result.parsed_json,
            retrieval_tier=tier_used,
            candidate_pages_sent=pages_sent,
            ok=True,
            needs_human_review=False,
            flag_reason=None,
        )

    def process_line_items_batch(
        self,
        line_items: list[dict],
        contractor_index: ContractorPageIndex,
        images_root: Path,
    ) -> list[LineItemResult]:
        results = []
        for item in line_items:
            res = self.process_line_item(
                line_item_id=item["line_item_id"],
                line_item_text=item["description"],
                contractor_index=contractor_index,
                images_root=images_root,
            )
            results.append(res)
        return results

    def _call_and_parse(self, line_item_text: str, pages: list[PageCandidate], images_root: Path) -> VLMCallResult:
        if not pages:
            return VLMCallResult(
                parsed_json={
                    "line_item": line_item_text,
                    "status": "NOT_MENTIONED",
                    "cost_value": None,
                    "cost_type": None,
                    "comments": "No candidate pages were available to check for this document.",
                    "citation": {"page_number": None, "verbatim_citation": None, "section_label": None},
                    "confidence": "Low",
                },
                raw_text="",
                ok=True,
            )
        return self.provider.call(line_item_text, pages, images_root)
