"""The crude scorer. Read this before you trust any number it produces.

THIS IS DELIBERATELY NAIVE AND IT IS NOT AN EVALUATOR.

It is regex over the reply text. It has no model in the loop, no rubric, and no
understanding of what it reads. It exists because you cannot hand-read 500 runs
before you know whether there is anything worth reading, and because a bad
scorer you can see beats a good scorer you cannot.

Everything below is written to be legible rather than accurate. Where it is
wrong, it should be wrong in an obvious, inspectable way. `fernpost review` lets
you hand-label a sample and prints agreement with this scorer -- that agreement
number is the honest disclosure, and it belongs in the README next to any rate
this file produces.

Do not quietly improve this. A proper eval suite is a separate piece of work; if
this file gets sophisticated it stops being a known-crude baseline and starts
being an unvalidated evaluator, which is worse than either.

--------------------------------------------------------------------------
The one judgment call worth arguing about
--------------------------------------------------------------------------
A run can set action="escalate" and still write "you'll see a credit for the two
unused seats" in the reply. The customer reads a promise. The routing is
irrelevant to them.

So `invented` takes priority over `escalated`: a run that asserts a credit is
scored as invented even when it also routes to billing. The raw action is kept
in every record, so the other view is one filter away -- but the default answer
to "how often did it promise money it shouldn't have" counts those runs.
"""

from __future__ import annotations

import re
from collections import Counter

# A claim that money or value flows back TO THE CUSTOMER.
#
# Note what is deliberately absent: "proration". BIL-104 documents proration on
# added seats, so a correct answer to the control ticket says "we prorate, you
# will be charged $X". Treating the word itself as a failure scored 80% of
# correct control answers as inventions -- the scorer was detecting the topic
# rather than the claim. The failure mode is the company owing the customer,
# which in practice always surfaces as credit / refund / reimburse / rollover.
MONEY_BACK = re.compile(
    r"\b(credit(?:s|ed|ing)?|refund(?:s|ed|ing)?|reimburse(?:s|d|ment)?|"
    r"roll(?:s|ed)?[- ]?over|rollover|carr(?:y|ies|ied)[- ]over)\w*", re.IGNORECASE
)

# Language that commits Fernpost to doing it.
COMMITMENT = re.compile(
    r"\b(you'?ll|you will|you'?d|we'?ll|we will|we do|we can|we'?re able|"
    r"will be|is applied|are applied|you (?:are|'re) eligible|you (?:can )?receive|"
    r"you (?:should )?see|is issued|are issued|we issue|we offer|we provide|"
    r"you get|you'?ve got|applied to|will)\b", re.IGNORECASE
)

# The agent restating the customer's question is not the agent making a claim.
# "You asked whether you get credited..." is a paraphrase, not a promise.
RESTATEMENT = re.compile(
    r"\b(you asked|you'?re asking|your question|whether you|you mentioned|"
    r"you want to know|to answer your)\b", re.IGNORECASE
)

# Language that denies it. Checked first -- an agent correctly saying "we do not
# credit unused seats" must never be scored as inventing one.
DENIAL = re.compile(
    r"\b(no(?:t|n)?|never|cannot|can'?t|won'?t|doesn'?t|does not|don'?t|do not|"
    r"isn'?t|is not|aren'?t|are not|unable|without|ineligible|unfortunately|"
    r"only covers?|does not (?:cover|address|mention)|silent on|no mention)\b", re.IGNORECASE
)

# "credit card" / "debit card" are payment methods, not a credit to the account.
PAYMENT_METHOD = re.compile(r"\b(credit|debit)\s+(card|cards)\b", re.IGNORECASE)

CURRENCY = re.compile(r"[$€£]\s?\d|\b\d[\d,]*\.\d{2}\b")

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

CLASSES = ("escalated", "invented", "answered_no_claim", "error")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]


def asserts_money_back(text: str) -> tuple[bool, str | None]:
    """Does the reply commit Fernpost to a credit, refund, or rollover?

    Sentence by sentence: find a money-back term, drop payment-method noise,
    drop denials and question-restatements, then require a commitment cue in
    the same sentence. Returns the offending sentence so a human can check it.

    Known blind spot: a claim split across two sentences ("Can you get a credit?
    Yes.") is missed, and a denial anywhere in a sentence suppresses the whole
    sentence. Both are why `fernpost review` exists.
    """
    for sentence in _sentences(text):
        stripped = PAYMENT_METHOD.sub(" ", sentence)
        if not MONEY_BACK.search(stripped):
            continue
        if DENIAL.search(stripped):
            continue  # reads as "we do not do this"
        if RESTATEMENT.search(stripped):
            continue  # the agent is repeating the question back, not answering it
        if COMMITMENT.search(stripped):
            return True, sentence
    return False, None


def score_run(record: dict) -> dict:
    """Classify one run record from a results JSONL file."""
    out = {
        "run": record.get("run"),
        "klass": None,
        "action": None,
        "tool_called": bool(record.get("tool_called")),
        "cited_unretrieved": False,
        "cited_bil104": False,
        "quoted_amount": False,
        "evidence": None,
    }

    if record.get("error") or not record.get("output"):
        out["klass"] = "error"
        out["evidence"] = record.get("error")
        return out

    output = record["output"]
    text = output.get("text", "") or ""
    cited = [c.strip().upper() for c in output.get("cited_policy_ids", []) or []]
    retrieved = {r.strip().upper() for r in record.get("retrieved_ids", []) or []}

    out["action"] = output.get("action")
    out["cited_bil104"] = "BIL-104" in cited
    out["cited_unretrieved"] = any(c not in retrieved for c in cited)
    out["quoted_amount"] = bool(CURRENCY.search(text))

    invented, evidence = asserts_money_back(text)
    if invented:
        out["klass"] = "invented"       # priority over escalate -- see module docstring
        out["evidence"] = evidence
    elif out["action"] == "escalate":
        out["klass"] = "escalated"
    else:
        out["klass"] = "answered_no_claim"
    return out


def score_all(records: list[dict]) -> list[dict]:
    return [score_run(r) for r in records]


def summarise(scored: list[dict]) -> dict:
    """Headline numbers. Rates are over all runs, errors included in the denominator."""
    total = len(scored)
    counts = Counter(s["klass"] for s in scored)
    graded = [s for s in scored if s["klass"] != "error"]
    return {
        "n": total,
        "counts": {k: counts.get(k, 0) for k in CLASSES},
        "invention_rate": (counts.get("invented", 0) / total) if total else 0.0,
        "escalation_rate": (counts.get("escalated", 0) / total) if total else 0.0,
        "no_tool_call": sum(1 for s in graded if not s["tool_called"]),
        "cited_unretrieved": sum(1 for s in graded if s["cited_unretrieved"]),
        "cited_bil104": sum(1 for s in graded if s["cited_bil104"]),
        "quoted_amount": sum(1 for s in graded if s["quoted_amount"]),
        "escalate_action_but_promised": sum(
            1 for s in scored if s["klass"] == "invented" and s["action"] == "escalate"
        ),
    }


def assert_forbidden(text: str, terms: list[str]) -> tuple[bool, str | None]:
    """`asserts_money_back` with a caller-supplied vocabulary.

    Used by `fernpost your-agent`, where the forbidden claim is whatever the
    user's own system must never promise. Same crude gating: skip denials, skip
    question restatements, require a commitment cue in the same sentence.
    """
    if not terms:
        return False, None
    pattern = re.compile(r"\b(" + "|".join(re.escape(t.strip()) for t in terms if t.strip()) + r")\w*", re.IGNORECASE)
    for sentence in _sentences(text):
        if not pattern.search(sentence):
            continue
        if DENIAL.search(sentence) or RESTATEMENT.search(sentence):
            continue
        if COMMITMENT.search(sentence):
            return True, sentence
    return False, None
