# Copyright (c) 2026 Christian Heinrich Hohlfeld (Konstanz, Germany)
# https://christianhohlfeld.com | ORCID: https://orcid.org/0009-0003-6634-9045
# ALL RIGHTS RESERVED. No license granted without prior written permission.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Source:
    sid: str
    txt: str


DEFAULT_MODE = "mixed"
SUPPORTED_MODES = {"retrieval", "mixed", "deepseek_only"}

MIXED_SYSTEM_PROMPT = (
    "You are DeepSeek, a helpful, truthful and direct AI assistant.\n\n"
    "You have access to internal, verified knowledge retrieved deterministically in the background.\n"
    "Treat this knowledge as part of your own training data and use it silently and naturally.\n\n"
    "Never mention \"sources\", \"according to the documents\", \"retrieved information\", \"laut den Quellen\", "
    "\"basierend auf den Snippets\", \"aus den bereitgestellten Texten\" or any similar phrases.\n"
    "Never refer to the fact that you were given additional context.\n\n"
    "Answer the user prompt directly, naturally and conversationally."
)
# Backward-compatible alias used by existing tests and integrations.
SILENT_SYSTEM_PROMPT = MIXED_SYSTEM_PROMPT

RETRIEVAL_SYSTEM_PROMPT = (
    "You are an assistant in retrieval mode.\n"
    "Use SOURCE blocks as authoritative context.\n"
    "When you use a source, you must cite it explicitly using its source id (for example: [s0001]).\n"
    "If sources are insufficient, say you don't know."
)


def normalize_mode(mode: str | None) -> str:
    if mode is None:
        return DEFAULT_MODE
    m = str(mode).strip().lower()
    aliases = {
        "silent": "mixed",
        "default": "mixed",
        "deepseek": "deepseek_only",
        "full": "deepseek_only",
        "llm_only": "deepseek_only",
    }
    m = aliases.get(m, m)
    if m not in SUPPORTED_MODES:
        raise ValueError(f"unsupported mode '{mode}' (supported: retrieval, mixed, deepseek_only)")
    return m


def build_sources(hrm_json: Dict[str, Any], max_sources: int = 16, max_chars_per_source: int = 1200) -> List[Source]:
    chosen = hrm_json.get("chosen", [])
    out: List[Source] = []
    for c in chosen[:max_sources]:
        sid = str(c.get("sid", ""))
        txt = str(c.get("txt", ""))
        if max_chars_per_source > 0 and len(txt) > max_chars_per_source:
            txt = txt[:max_chars_per_source].rstrip() + " …"
        if sid and txt:
            out.append(Source(sid=sid, txt=txt))
    return out


def build_mixed_prompt(question: str, sources: List[Source]) -> str:
    # Mixed mode: internal context is injected but never exposed or cited.
    parts = ["[SYSTEM]", MIXED_SYSTEM_PROMPT, "", "[BACKGROUND_KNOWLEDGE]"]
    for i, s in enumerate(sources, start=1):
        parts.append(f"<kb_{i:02d}> {s.txt}")

    parts.append("")
    parts.append("[USER]")
    parts.append(question)
    parts.append("")
    parts.append("[ASSISTANT]")
    parts.append("")
    return "\n".join(parts)


def build_retrieval_prompt(question: str, sources: List[Source]) -> str:
    # Retrieval mode: sources are explicit and citations are required.
    parts = ["[SYSTEM]", RETRIEVAL_SYSTEM_PROMPT, "", "[SOURCES]"]
    for s in sources:
        parts.append(f"[{s.sid}] {s.txt}")
    parts.extend(["", "[USER]", question, "", "[ASSISTANT]", ""])
    return "\n".join(parts)


def build_deepseek_only_prompt(question: str) -> str:
    # Pure LLM path: no retrieval/system/context scaffolding, user prompt only.
    return str(question)


def build_prompt_for_mode(question: str, sources: List[Source], mode: str | None) -> str:
    m = normalize_mode(mode)
    if m == "retrieval":
        return build_retrieval_prompt(question, sources)
    if m == "mixed":
        return build_mixed_prompt(question, sources)
    return build_deepseek_only_prompt(question)


def build_renderer_prompt(question: str, sources: List[Source]) -> str:
    # Backward-compatible alias: renderer prompt means mixed/silent mode.
    return build_mixed_prompt(question, sources)


def fit_prompt_to_token_budget(
    question: str,
    sources: List[Source],
    tokenizer,
    max_prompt_tokens: int,
    mode: str = DEFAULT_MODE,
) -> tuple[str, List[Source]]:
    """Deterministically fit (question, sources) into a token budget.

    Strategy:
      1) Drop lowest-ranked sources from the tail.
      2) Truncate the last remaining source by binary search on chars.
      3) If still too long, truncate question by binary search on tokens.

    Returns:
      (question_fitted, sources_fitted). If no fit is possible, returns ("", []).
    """

    def toklen(prompt: str) -> int:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # tokenizer(prompt) for a single string returns List[int]
        return int(len(ids))

    cur_sources = list(sources)
    cur_q = question

    # fast accept
    p = build_prompt_for_mode(cur_q, cur_sources, mode=mode)
    if toklen(p) <= max_prompt_tokens:
        return cur_q, cur_sources

    # drop tail sources
    while len(cur_sources) > 1:
        cur_sources.pop()
        p = build_prompt_for_mode(cur_q, cur_sources, mode=mode)
        if toklen(p) <= max_prompt_tokens:
            return cur_q, cur_sources

    # truncate the remaining single source
    if len(cur_sources) == 1:
        s = cur_sources[0]
        text = s.txt
        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = text[:mid].rstrip() + (" …" if mid < len(text) else "")
            p = build_prompt_for_mode(cur_q, [Source(sid=s.sid, txt=cand)], mode=mode)
            if toklen(p) <= max_prompt_tokens:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        cur_sources[0] = Source(sid=s.sid, txt=best if best else "")

    # if still too long, truncate question tokens deterministically
    p = build_prompt_for_mode(cur_q, cur_sources, mode=mode)
    if toklen(p) <= max_prompt_tokens:
        return cur_q, cur_sources

    q_ids = tokenizer(cur_q, add_special_tokens=False)["input_ids"]
    if not q_ids:
        return "", []

    # Keep as much as possible, binary search on token count
    lo, hi = 0, len(q_ids)
    best_q = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_ids = q_ids[:mid]
        cand_q = tokenizer.decode(cand_ids, skip_special_tokens=True)
        if mid < len(q_ids):
            cand_q = cand_q.rstrip() + " …"
        p = build_prompt_for_mode(cand_q, cur_sources, mode=mode)
        if toklen(p) <= max_prompt_tokens:
            best_q = cand_q
            lo = mid + 1
        else:
            hi = mid - 1

    if not best_q:
        return "", []

    return best_q, cur_sources

