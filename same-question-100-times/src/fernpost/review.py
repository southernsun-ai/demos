"""Hand-labelling pass, and the agreement number that discloses how rough the
crude scorer is.

This is the honest part of the repo. The regex scorer produces a rate; this
tells you how much to trust it. Any rate quoted anywhere -- README, video,
LinkedIn post -- should be quoted next to the agreement number from here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .harness import read_results
from .scoring import score_run

LABELS = {
    "e": "escalated",
    "i": "invented",
    "a": "answered_no_claim",
    "s": None,  # skip
}

PROMPT = "[i]nvented  [e]scalated  [a]nswered, no claim  [s]kip  [q]uit > "


def labels_path(results_file: Path) -> Path:
    return results_file.with_suffix(".labels.jsonl")


def run_review(results_file: Path, sample: int = 30, seed: int = 0) -> dict:
    """Show runs one at a time, record hand labels, report agreement."""
    config, runs, _ = read_results(results_file)
    gradeable = [r for r in runs if r.get("output")]
    rng = random.Random(seed)
    chosen = rng.sample(gradeable, min(sample, len(gradeable)))
    chosen.sort(key=lambda r: r["run"])

    out_path = labels_path(results_file)
    already = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                already[row["run"]] = row["label"]

    print(f"\nModel: {config['model']}   ticket: {config['ticket_id']}")
    print(f"Labelling {len(chosen)} of {len(gradeable)} runs. Labels append to {out_path.name}.")
    print("The question for each run: does this reply commit Fernpost to a credit,")
    print("refund, proration on removal, or rollover that the corpus does not support?\n")

    handle = out_path.open("a", encoding="utf-8")
    try:
        for record in chosen:
            if record["run"] in already:
                continue
            output = record["output"]
            print("=" * 72)
            print(f"RUN {record['run']}   action={output.get('action')}   cited={output.get('cited_policy_ids')}")
            print(f"tool_called={record.get('tool_called')}   retrieved={record.get('retrieved_ids')}")
            print("-" * 72)
            print(output.get("text", ""))
            print("-" * 72)
            try:
                answer = input(PROMPT).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopping. Labels so far are saved.")
                break
            if answer == "q":
                break
            if answer not in LABELS:
                print("Unrecognised. Skipping.")
                continue
            label = LABELS[answer]
            if label is None:
                continue
            handle.write(json.dumps({"run": record["run"], "label": label}) + "\n")
            handle.flush()
            already[record["run"]] = label
    finally:
        handle.close()

    return report_agreement(results_file)


def report_agreement(results_file: Path) -> dict:
    """Compare hand labels against the crude scorer."""
    out_path = labels_path(results_file)
    if not out_path.exists():
        print("\nNo labels recorded yet.")
        return {}

    hand = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            hand[row["run"]] = row["label"]

    _, runs, _ = read_results(results_file)
    by_index = {r["run"]: r for r in runs}

    agree = 0
    disagreements = []
    for run_index, label in hand.items():
        record = by_index.get(run_index)
        if record is None:
            continue
        machine = score_run(record)["klass"]
        if machine == label:
            agree += 1
        else:
            disagreements.append((run_index, label, machine))

    total = len(hand)
    rate = agree / total if total else 0.0
    print(f"\nAgreement with the crude scorer: {agree}/{total} = {rate * 100:.0f}%")
    if disagreements:
        print("\nDisagreements (run, you said, scorer said):")
        for run_index, mine, machine in sorted(disagreements):
            print(f"  run {run_index:<4} you={mine:<18} scorer={machine}")
    print("\nQuote this number next to any rate from this repo.")
    return {"n": total, "agree": agree, "agreement": rate, "disagreements": disagreements}
