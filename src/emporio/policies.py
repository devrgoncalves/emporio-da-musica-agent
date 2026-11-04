"""Serves sections of the store policy manual by topic.

The manual is small (8 pages, 10 sections), so instead of a vector store we
split it by its own numbered headings and let the LLM pick a topic from a
fixed list. Deterministic, traceable in the logs, and easy to test.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal, get_args

from emporio.data_store import DATA_DIR

Topic = Literal[
    "about",
    "hours",
    "payment",
    "returns",
    "shipping",
    "promotions",
    "customer_service",
    "warranty",
    "privacy",
]

# topic -> section number(s) in data/policies.md
TOPIC_SECTIONS: dict[str, tuple[int, ...]] = {
    "about": (1,),
    "hours": (2,),
    "payment": (3,),
    "returns": (4,),
    "shipping": (5,),
    "promotions": (6,),
    "customer_service": (7, 10),
    "warranty": (8,),
    "privacy": (9,),
}

HEADING = re.compile(r"^## (\d+)\.", re.MULTILINE)


@lru_cache(maxsize=1)
def load_sections() -> dict[int, str]:
    text = (DATA_DIR / "policies.md").read_text(encoding="utf-8")
    matches = list(HEADING.finditer(text))
    sections = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[int(match.group(1))] = text[match.start() : end].strip()
    return sections


def get_policy(topic: str) -> str:
    if topic not in TOPIC_SECTIONS:
        valid = ", ".join(get_args(Topic))
        return f"Unknown topic '{topic}'. Valid topics: {valid}."
    sections = load_sections()
    return "\n\n".join(sections[n] for n in TOPIC_SECTIONS[topic])
