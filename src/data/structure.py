"""Structure extraction and canonical serialization for Option A plan embeddings.

Given a continuation text, we extract lightweight structural attributes and
produce a deterministic *canonical serialization* string.  That string is then
embedded by Sentence-T5 to yield z0_structure in R^768.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class StructureFeatures:
    """Bag of structural attributes extracted from a text span."""

    token_count: int = 0
    sentence_count: int = 0
    has_list: bool = False
    has_question: bool = False
    has_code_block: bool = False
    section_tags: list[str] = field(default_factory=list)
    length_bucket: str = "short"  # short / medium / long / very_long

    def to_label_dict(self) -> dict[str, int]:
        """Return integer labels suitable for classifier guidance training."""
        bucket_map = {"short": 0, "medium": 1, "long": 2, "very_long": 3}
        return {
            "length_bucket": bucket_map.get(self.length_bucket, 0),
            "has_list": int(self.has_list),
            "has_question": int(self.has_question),
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_PATTERN = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.\)])\s", re.MULTILINE)
_QUESTION_PATTERN = re.compile(r"\?")
_CODE_BLOCK = re.compile(r"```")
_SECTION_HEADER = re.compile(r"(?:^|\n)(#{1,4})\s+(.+)")


def extract_structure(text: str) -> StructureFeatures:
    """Extract structural attributes from *text*."""
    tokens = text.split()
    token_count = len(tokens)
    sentences = [s for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    sentence_count = max(len(sentences), 1)

    has_list = bool(_LIST_PATTERN.search(text))
    has_question = bool(_QUESTION_PATTERN.search(text))
    has_code_block = len(_CODE_BLOCK.findall(text)) >= 2

    section_tags: list[str] = []
    for m in _SECTION_HEADER.finditer(text):
        section_tags.append(m.group(2).strip()[:64])

    if token_count < 32:
        bucket = "short"
    elif token_count < 96:
        bucket = "medium"
    elif token_count < 256:
        bucket = "long"
    else:
        bucket = "very_long"

    return StructureFeatures(
        token_count=token_count,
        sentence_count=sentence_count,
        has_list=has_list,
        has_question=has_question,
        has_code_block=has_code_block,
        section_tags=section_tags,
        length_bucket=bucket,
    )


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------


def canonical_serialize(feats: StructureFeatures) -> str:
    """Deterministic string encoding of *feats*.

    The resulting string is fed to Sentence-T5 to produce the plan embedding.
    """
    parts = [
        f"length={feats.length_bucket}",
        f"tokens={feats.token_count}",
        f"sentences={feats.sentence_count}",
    ]
    if feats.has_list:
        parts.append("list=yes")
    if feats.has_question:
        parts.append("question=yes")
    if feats.has_code_block:
        parts.append("code=yes")
    if feats.section_tags:
        parts.append("sections=" + "|".join(feats.section_tags[:8]))
    return " ; ".join(parts)
