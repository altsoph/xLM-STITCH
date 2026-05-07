"""Token metadata normalization across models.

Provides a unified TokenMeta dataclass for each token in a sequence,
including family classification, special-token flags, and position info.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from transformers import PreTrainedTokenizerBase


class TokenFamily(Enum):
    SPECIAL = auto()
    PUNCTUATION = auto()
    FUNCTION = auto()
    CONTENT = auto()
    NAME_ENTITY = auto()
    MASK = auto()
    NUMBER = auto()
    UNKNOWN = auto()


FUNCTION_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "because",
    "but", "and", "or", "if", "while", "although", "that", "which", "who",
    "whom", "this", "these", "those", "i", "me", "my", "myself", "we",
    "our", "ours", "ourselves", "you", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what",
}

_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_NUMBER_RE = re.compile(r"^[\d.,]+$")
_UPPER_RE = re.compile(r"^[A-Z]")


@dataclass
class TokenMeta:
    """Metadata for a single token in a sequence."""
    token_id: int
    token_text: str
    position: int
    is_special: bool
    family: TokenFamily
    segment_id: int = 0
    is_mask: bool = False
    is_last_visible: bool = False
    is_delimiter: bool = False
    clean_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "token_text": self.token_text,
            "clean_text": self.clean_text,
            "position": self.position,
            "is_special": self.is_special,
            "family": self.family.name,
            "segment_id": self.segment_id,
            "is_mask": self.is_mask,
            "is_last_visible": self.is_last_visible,
            "is_delimiter": self.is_delimiter,
        }


def classify_token(
    token_text: str,
    token_id: int,
    special_ids: Set[int],
    mask_id: Optional[int],
) -> TokenFamily:
    if token_id in special_ids:
        return TokenFamily.SPECIAL
    if mask_id is not None and token_id == mask_id:
        return TokenFamily.MASK

    clean = token_text.replace("Ġ", "").replace("▁", "").replace("##", "").strip()
    if not clean:
        return TokenFamily.SPECIAL

    if _PUNCT_RE.match(clean):
        return TokenFamily.PUNCTUATION
    if _NUMBER_RE.match(clean):
        return TokenFamily.NUMBER
    if clean.lower() in FUNCTION_WORDS:
        return TokenFamily.FUNCTION
    if _UPPER_RE.match(clean) and len(clean) > 1:
        return TokenFamily.NAME_ENTITY
    return TokenFamily.CONTENT


def extract_token_metadata(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizerBase,
    segment_ids: Optional[List[int]] = None,
) -> List[TokenMeta]:
    """Build metadata for every token in a sequence."""
    special_ids = set(tokenizer.all_special_ids)
    mask_id = getattr(tokenizer, "mask_token_id", None)

    sep_ids = set()
    if hasattr(tokenizer, "sep_token_id") and tokenizer.sep_token_id is not None:
        sep_ids.add(tokenizer.sep_token_id)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    metas: List[TokenMeta] = []
    last_non_pad = len(input_ids) - 1
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        while last_non_pad > 0 and input_ids[last_non_pad] == pad_id:
            last_non_pad -= 1

    for pos, tid in enumerate(input_ids):
        tok_text = tokenizer.convert_ids_to_tokens(tid)
        clean = tok_text.replace("Ġ", "").replace("▁", "").replace("##", "").strip()
        family = classify_token(tok_text, tid, special_ids, mask_id)
        is_sp = tid in special_ids
        is_mask = mask_id is not None and tid == mask_id
        is_delim = tid in sep_ids or (eos_id is not None and tid == eos_id)
        is_last = pos == last_non_pad

        metas.append(TokenMeta(
            token_id=tid,
            token_text=tok_text,
            clean_text=clean,
            position=pos,
            is_special=is_sp,
            family=family,
            segment_id=(segment_ids[pos] if segment_ids else 0),
            is_mask=is_mask,
            is_last_visible=is_last,
            is_delimiter=is_delim,
        ))
    return metas
