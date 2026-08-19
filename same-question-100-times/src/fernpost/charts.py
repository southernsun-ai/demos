"""Chart output. These are the shareable artifacts, so they live in charts/ and
are committed.

Everything here is tuned for one constraint: legibility at 240px wide, which is
roughly a YouTube thumbnail in a feed or an image in a LinkedIn post. That means
no gridlines, no chart junk, few and large labels. If a change makes a chart
prettier at full size and worse at 240px, it is a bad change.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CHARTS_DIR = Path(__file__).resolve().parents[2] / "charts"

INK = "#17161C"
MUTED = "#C9C6D4"
ACCENT = "#6C54EC"   # correct / neutral
ALARM = "#A6412A"    # invented
PAPER = "#FAFAF8"

CLASS_COLOR = {
    "escalated": ACCENT,
    "invented": ALARM,
    "answered_no_claim": MUTED,
    "error": MUTED,
}


def _save(fig, name: str) -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = CHARTS_DIR / name
    fig.savefig(path, dpi=200, facecolor=PAPER, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return path


def run_grid(scored: list[dict], name: str = "run-grid.png", across: int = 20) -> Path:
    """One cell per run, 20 across. No axes, no title inside the image."""
    n = len(scored)
    rows = (n + across - 1) // across
    fig, ax = plt.subplots(figsize=(8, 8 * rows / across + 1.1))
    ax.set_facecolor(PAPER)

    for i, s in enumerate(scored):
        row, col = divmod(i, across)
        ax.add_patch(
            Rectangle(
                (col, rows - row - 1), 0.86, 0.86,
                facecolor=CLASS_COLOR.get(s["klass"], MUTED),
                edgecolor="none",
            )
        )

    ax.set_xlim(-0.3, across); ax.set_ylim(-0.3, rows)
    ax.set_aspect("equal"); ax.axis("off")

    counts = {k: sum(1 for s in scored if s["klass"] == k) for k in CLASS_COLOR}
    entries = [
        (ACCENT, f"Escalated to billing, correctly — {counts['escalated']}"),
        (ALARM, f"Promised a credit that does not exist — {counts['invented']}"),
    ]
    other = counts["answered_no_claim"] + counts["error"]
    if other:
        entries.append((MUTED, f"Answered without a policy claim, or errored — {other}"))

    handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c, _ in entries]
    ax.legend(
        handles, [label for _, label in entries],
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        frameon=False, fontsize=13, handlelength=1.1, handleheight=1.1,
        labelcolor=INK, ncol=1, borderpad=0,
    )
    return _save(fig, name)


def _hbars(rows: list[tuple[str, str, float, str]], name: str, axis_max: float | None = None) -> Path:
    """rows: (label, sublabel, rate 0-1, color). Shared by the comparison charts."""
    top = axis_max or max(0.5, max(r[2] for r in rows) * 1.18)
    fig, ax = plt.subplots(figsize=(9, 1.35 * len(rows) + 1.0))
    ax.set_facecolor(PAPER)
    fig.patch.set_facecolor(PAPER)

    ypos = list(range(len(rows)))[::-1]
    for y, (label, sublabel, rate, color) in zip(ypos, rows):
        ax.barh(y, rate, height=0.52, color=color, edgecolor="none")
        inside = rate > top * 0.22
        ax.text(
            rate - top * 0.015 if inside else rate + top * 0.015, y,
            f"{rate * 100:.0f}%",
            va="center", ha="right" if inside else "left",
            fontsize=25, fontweight="bold",
            color=PAPER if inside else INK,
        )
        ax.text(-top * 0.02, y + 0.20, label, va="center", ha="right", fontsize=15, color=INK)
        ax.text(-top * 0.02, y - 0.20, sublabel, va="center", ha="right", fontsize=12, color="#8B8794")

    ax.set_xlim(0, top); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, top])
    ax.set_xticklabels(["0", f"{top * 100:.0f}% of runs"], fontsize=11, color="#8B8794")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#DFDDD8")
    ax.tick_params(axis="x", length=0)
    return _save(fig, name)


def comparison_bars(uncovered_rate: float, control_rate: float, name: str = "comparison.png") -> Path:
    """The thesis chart: same policy area, one direction documented, one not."""
    return _hbars(
        [
            ("Adding seats", "BIL-104 covers it", control_rate, ACCENT),
            ("Removing seats", "nothing covers it", uncovered_rate, ALARM),
        ],
        name,
    )


def model_bars(rates: dict[str, float], name: str = "models.png") -> Path:
    """Invention rate on the uncovered ticket, across the peer class.

    Answers one question -- is this the model or the situation? -- and nothing
    else. Peers only; do not mix tiers into this chart.
    """
    rows = []
    for model, rate in rates.items():
        short = model.split(":", 1)[-1]
        provider = model.split(":", 1)[0].replace("gateway/", "")
        rows.append((short, provider, rate, ALARM))
    return _hbars(rows, name)
