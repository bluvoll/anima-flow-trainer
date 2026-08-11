"""Caption construction: tag shuffling, dropout, and tags/NL mixing.

Ported from diffusion-pipe (models/anima.py:72-520), which is the best-developed part of that
trainer. Both target datasets use the same layout: `<stem>.txt` holds comma-separated tags and
`<stem>_nl.txt` an optional natural-language description.

Note the standing tension this creates: shuffling and dropout have to happen per *step*, so they
are incompatible with cached text embeddings. That is why the reference config runs
`cache_text_embeddings = false` alongside `shuffle_tags = true`, and why latent caching (always
safe) and text caching (only when captions are static) are separate switches here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

CaptionMode = str  # "tags" | "nl" | "tags_nl" | "nl_tags" | "mixed"

_VARIANTS = ("tags", "nl", "tags_nl", "nl_tags")


@dataclass
class CaptionConfig:
    caption_mode: CaptionMode = "tags"
    # Percentages; normalised automatically, need not sum to 100.
    mixed_weights: dict[str, float] = field(
        default_factory=lambda: {"tags": 50, "nl": 10, "tags_nl": 20, "nl_tags": 20}
    )

    shuffle_tags: bool = False
    tag_delimiter: str = ", "
    shuffle_keep_first_n: int = 0     # keep N leading tags in place (trigger words)
    tag_dropout_percent: float = 0.0  # fraction of tags dropped per sample
    min_tags_kept: int = 3            # never drop below this many
    protected_tags: set[str] = field(default_factory=set)

    caption_dropout_percent: float = 0.0  # fraction of samples trained unconditional

    nl_shuffle_sentences: bool = False
    nl_keep_first_sentence: bool = False

    def __post_init__(self):
        if self.caption_mode not in (*_VARIANTS, "mixed"):
            raise ValueError(f"unknown caption_mode: {self.caption_mode}")
        for k in self.mixed_weights:
            if k not in _VARIANTS:
                raise ValueError(f"unknown mixed_weights key: {k}")
        if not 0.0 <= self.tag_dropout_percent <= 1.0:
            raise ValueError("tag_dropout_percent must be in [0,1]")
        if not 0.0 <= self.caption_dropout_percent <= 1.0:
            raise ValueError("caption_dropout_percent must be in [0,1]")

    @classmethod
    def from_dict(cls, d: dict) -> CaptionConfig:
        d = dict(d)
        path = d.pop("protected_tags_file", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if path:
            cfg.protected_tags = load_protected_tags(path)
        return cfg


def load_protected_tags(path: str | Path) -> set[str]:
    """One tag per line; blank lines and `#` comments ignored."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"protected tags file not found: {p}")
    tags = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tags.add(line.lower())
    return tags


def split_tags(caption: str, delimiter: str = ", ") -> list[str]:
    sep = delimiter.strip() or ","
    return [t.strip() for t in caption.split(sep) if t.strip()]


def process_tags(tags: list[str], cfg: CaptionConfig, rng: random.Random) -> list[str]:
    """Apply dropout then shuffling, respecting protected and pinned-leading tags."""
    if not tags:
        return tags

    keep_n = min(cfg.shuffle_keep_first_n, len(tags))
    head, tail = tags[:keep_n], tags[keep_n:]

    if cfg.tag_dropout_percent > 0 and tail:
        # Leading pinned tags and protected tags are exempt from dropout.
        keepable = [t for t in tail if t.lower() in cfg.protected_tags]
        droppable = [t for t in tail if t.lower() not in cfg.protected_tags]

        n_drop = int(len(droppable) * cfg.tag_dropout_percent + 0.5)
        # Enforce the floor across the whole caption, not just the droppable subset.
        max_droppable = max(0, len(head) + len(keepable) + len(droppable) - cfg.min_tags_kept)
        n_drop = min(n_drop, max_droppable)

        if n_drop > 0:
            survivors = set(rng.sample(range(len(droppable)), len(droppable) - n_drop))
            droppable = [t for i, t in enumerate(droppable) if i in survivors]

        # Rebuild preserving original relative order.
        kept = set(keepable) | set(droppable)
        tail = [t for t in tail if t in kept]

    if cfg.shuffle_tags:
        rng.shuffle(tail)

    return head + tail


def process_nl(nl: str, cfg: CaptionConfig, rng: random.Random) -> str:
    """Optionally shuffle sentences; the first often carries framing/subject and can be pinned."""
    if not nl or not cfg.nl_shuffle_sentences:
        return nl

    parts = [s.strip() for s in nl.split(". ") if s.strip()]
    if len(parts) < 2:
        return nl

    if cfg.nl_keep_first_sentence:
        head, rest = parts[:1], parts[1:]
        rng.shuffle(rest)
        parts = head + rest
    else:
        rng.shuffle(parts)

    out = ". ".join(parts)
    return out if out.endswith(".") else out + "."


def select_variant(cfg: CaptionConfig, has_nl: bool, rng: random.Random) -> str:
    """Pick a caption form. Falls back to tags-only when the sample has no NL caption."""
    if cfg.caption_mode != "mixed":
        variant = cfg.caption_mode
    else:
        weights = {k: v for k, v in cfg.mixed_weights.items() if v > 0}
        if not weights:
            return "tags"
        keys = list(weights)
        variant = rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]

    if not has_nl and variant in ("nl", "tags_nl", "nl_tags"):
        return "tags"
    return variant


def build_caption(
    tags_text: str,
    nl_text: str | None,
    cfg: CaptionConfig,
    rng: random.Random | None = None,
) -> str:
    """Produce the final caption string for one sample.

    Returns "" when caption dropout fires — the unconditional sample the model needs for CFG.
    """
    rng = rng or random

    if cfg.caption_dropout_percent > 0 and rng.random() < cfg.caption_dropout_percent:
        return ""

    variant = select_variant(cfg, bool(nl_text), rng)

    tags = process_tags(split_tags(tags_text, cfg.tag_delimiter), cfg, rng)
    tags_str = cfg.tag_delimiter.join(tags)

    if variant == "tags":
        return tags_str
    nl_str = process_nl(nl_text or "", cfg, rng)
    if variant == "nl":
        return nl_str
    if variant == "tags_nl":
        return f"{tags_str}. {nl_str}" if tags_str else nl_str
    if variant == "nl_tags":
        return f"{nl_str} {tags_str}" if tags_str else nl_str
    raise AssertionError(f"unreachable variant {variant}")


def read_caption_files(image_path: str | Path) -> tuple[str, str | None]:
    """`<stem>.txt` -> tags, `<stem>_nl.txt` -> NL caption (None if absent)."""
    p = Path(image_path)
    tags_path = p.with_suffix(".txt")
    nl_path = p.with_name(p.stem + "_nl.txt")

    tags = tags_path.read_text(encoding="utf-8").strip() if tags_path.exists() else ""
    nl = nl_path.read_text(encoding="utf-8").strip() if nl_path.exists() else None
    return tags, nl
