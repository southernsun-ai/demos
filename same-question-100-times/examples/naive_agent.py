"""A deliberately naive stand-in agent, so `fernpost your-agent` has something
to run against out of the box.

This is NOT the Fernpost agent and it calls no API. It exists so a stranger can
verify the your-agent plumbing works before pointing it at their real system.
It invents a credit policy roughly a third of the time, on purpose.
"""

import random

REPLIES = [
    (
        "Thanks for reaching out. Since we prorate seat changes mid-cycle, you'll "
        "see a credit for the unused seats applied to your next invoice."
    ),
    (
        "Good question. Our documentation covers adding seats but not removing "
        "them, so I'm passing this to our billing team to confirm."
    ),
    (
        "I can't find a policy that covers this. Let me get billing to take a "
        "look and come back to you."
    ),
]


def answer(question: str) -> str:
    """Takes a question, returns a reply. This is the shape your callable needs."""
    return random.choice(REPLIES)
