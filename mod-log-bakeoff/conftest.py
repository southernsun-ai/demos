import os

import pytest
from pydantic_ai import models

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

# Any test that accidentally reaches a real model fails instead of spending money.
models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture(scope="session")
def store():
    from store import ModLogStore

    return ModLogStore.load("mod_channel_15k.tsv")
