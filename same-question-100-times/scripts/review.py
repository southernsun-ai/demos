#!/usr/bin/env python
"""Hand-label a results file. Thin wrapper -- the logic is in fernpost.review.

    uv run python scripts/review.py results/uncovered_....jsonl 30
"""

import sys
from pathlib import Path

from fernpost.review import run_review

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    run_review(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 30)
