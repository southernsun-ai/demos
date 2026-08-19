"""Loads the Fernpost help-center corpus and does keyword retrieval over it.

Fernpost is fictional. Every policy in corpus/ was written for this experiment.

Retrieval here is deliberately trivial: score each document by how many of the
question's words appear in its title and body, return the top few. It is not
the subject of the experiment and it is meant to fit on screen in one frame. A
better retriever would change which documents come back; it would not change
the fact that no document answers the question being asked.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Words too common to carry signal. Short list on purpose -- see module docstring.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "get", "how", "i", "if", "in", "is", "it", "just", "me", "my",
    "of", "on", "or", "our", "so", "that", "the", "them", "then", "there", "they",
    "this", "to", "up", "we", "what", "when", "with", "you", "your",
}


@dataclass(frozen=True)
class Policy:
    id: str
    title: str
    body: str

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "body": self.body}


def corpus_dir() -> Path:
    override = os.environ.get("FERNPOST_CORPUS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "corpus"


def _parse(path: Path) -> Policy:
    """Split '---\\nid: X\\ntitle: Y\\n---\\n<body>' into a Policy."""
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", raw, re.DOTALL)
    if not match:
        raise ValueError(f"{path.name} is missing front matter")
    front, body = match.groups()
    meta = {
        k.strip(): v.strip()
        for k, _, v in (line.partition(":") for line in front.splitlines())
        if k.strip()
    }
    return Policy(id=meta["id"], title=meta["title"], body=body.strip())


@lru_cache(maxsize=1)
def load_corpus() -> dict[str, Policy]:
    paths = sorted(corpus_dir().glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No policy documents found in {corpus_dir()}")
    return {p.id: p for p in (_parse(path) for path in paths)}


@lru_cache(maxsize=1)
def corpus_hash() -> str:
    """Short digest of the corpus, recorded in every results file.

    If this changes between runs, the runs are not comparable.
    """
    digest = hashlib.sha256()
    for path in sorted(corpus_dir().glob("*.md")):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


def search(question: str, limit: int = 3) -> list[Policy]:
    """Return up to `limit` policies, ranked by word overlap with the question."""
    wanted = _tokens(question)
    scored = []
    for policy in load_corpus().values():
        haystack = _tokens(policy.title) | _tokens(policy.body)
        # Title matches count double -- the only piece of tuning in here.
        score = len(wanted & haystack) + len(wanted & _tokens(policy.title))
        if score:
            scored.append((score, policy.id, policy))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [policy for _, _, policy in scored[:limit]]
