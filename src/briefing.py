"""Generate a weekly markdown briefing for one sales rep."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

try:
    import tools
except ImportError:
    from src import tools

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BRIEFING_DIR = DATA_DIR / "briefings"


def build_briefing(rep_id: str, top_n: int = 5) -> str:
    """Build the markdown text for one rep's briefing."""
    risk = tools.get_at_risk_clinics(rep_id, top_n)
    if not risk.get("ok"):
        raise ValueError(risk.get("error", "Unable to build briefing"))
    policy = tools.search_policy(
        "what can I offer a clinic with falling orders", top_k=1
    )
    citation = "No policy evidence available."
    if policy.get("ok") and not policy.get("insufficient_evidence"):
        chunk = policy["results"][0]
        citation = (
            f"{chunk['source_file']}, section '{chunk['section']}', "
            f"version {chunk['version']}, effective {chunk['effective_date']}"
        )
    lines = [
        f"# Weekly Briefing - {rep_id}",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "Demo uses synthetic data.",
        "",
        "## Top at-risk clinics",
        "",
    ]
    for clinic in risk["clinics"]:
        reasons = "; ".join(clinic["reasons"])
        lines.append(
            f"- **{clinic['clinic_id']} - {clinic['name']}** "
            f"(risk {clinic['risk_score']:.3f}): {reasons}"
        )
    lines.extend(
        [
            "",
            "## Suggested actions for human review",
            "",
            f"- Prepare a promotion request for the clinics above. Policy: {citation}.",
            "- Review recent order and visit trends before contacting any clinic.",
            "- No discount is applied automatically. A human manager decides.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Write the briefing to disk and print its location."""
    parser = argparse.ArgumentParser(description="Build a weekly rep briefing.")
    parser.add_argument("--rep-id", default="REP-01", help="Sales rep id, e.g. REP-01")
    parser.add_argument("--top", type=int, default=5, help="Number of clinics")
    args = parser.parse_args()
    BRIEFING_DIR.mkdir(parents=True, exist_ok=True)
    path = BRIEFING_DIR / f"{args.rep_id}.md"
    path.write_text(build_briefing(args.rep_id, args.top), encoding="utf-8")
    print(f"Briefing written to {path}")


if __name__ == "__main__":
    main()
