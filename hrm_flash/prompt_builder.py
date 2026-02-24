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


def build_sources(hrm_json: Dict[str, Any], max_sources: int = 8, max_chars_per_source: int = 1200) -> List[Source]:
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


def build_renderer_prompt(question: str, sources: List[Source]) -> str:
    # Deterministic, strict instruction.
    sys = (
        "You are a precise assistant. Answer the QUESTION using ONLY the SOURCES. "
        "If the SOURCES do not contain the answer, say you don't know. "
        "Cite sources in brackets like [0001#s0003].\n"
    )

    parts = ["[SYSTEM]", sys, "\n[SOURCES]"]
    for s in sources:
        parts.append(f"[{s.sid}] {s.txt}")

    parts.append("\n[QUESTION]\n" + question.strip())
    parts.append("\n[ANSWER]\n")
    return "\n".join(parts)


def fit_prompt_to_token_budget(
    question: str,
    sources: List[Source],
    tokenizer,
    max_prompt_tokens: int,
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
    p = build_renderer_prompt(cur_q, cur_sources)
    if toklen(p) <= max_prompt_tokens:
        return cur_q, cur_sources

    # drop tail sources
    while len(cur_sources) > 1:
        cur_sources.pop()
        p = build_renderer_prompt(cur_q, cur_sources)
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
            p = build_renderer_prompt(cur_q, [Source(sid=s.sid, txt=cand)])
            if toklen(p) <= max_prompt_tokens:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        cur_sources[0] = Source(sid=s.sid, txt=best if best else "")

    # if still too long, truncate question tokens deterministically
    p = build_renderer_prompt(cur_q, cur_sources)
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
        p = build_renderer_prompt(cand_q, cur_sources)
        if toklen(p) <= max_prompt_tokens:
            best_q = cand_q
            lo = mid + 1
        else:
            hi = mid - 1

    if not best_q:
        return "", []

    return best_q, cur_sources

