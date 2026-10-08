"""Copy the ablation's key JSON into docs/notes/ablation/data/ and print the comparison table.

The state and critic files live in gitignored output directories, so the numbers the
README quotes are copied here as evidence. Reads only; writes only under data/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parent / "data"

PAPERS = [
    ("2013-06 - Deep Learning using Linear Support Vector Machines", "c5"),
    ("2023-10 - Multi Level Dense Layer Neural Network Model for Housing Price Prediction", "c3"),
    ("2003 - A Practical Guide to Support Vector Classification", "c1"),
    (
        "2017-08 - Fashion-MNIST - a Novel Image Dataset for Benchmarking "
        "Machine Learning Algorithms",
        "c17",
    ),
    ("2017-11 - Distilling a Neural Network Into a Soft Decision Tree", "c1"),
]


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def copy(src: Path, dest_name: str) -> str | None:
    if not src.exists():
        return None
    dest = DATA / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest.relative_to(ROOT))


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for paper, claim in PAPERS:
        short = paper.split(" - ")[0]

        off_state = load(ROOT / "orchestrator/output-ablation" / paper / "state.json")
        off_crit = load(ROOT / "critic/output-ablation" / f"{paper}.json")
        on_state = load(ROOT / "orchestrator/output" / paper / "state.json")
        on_crit = (on_state or {}).get("critic_output") or load(
            ROOT / "critic/output-ablation-loopon" / f"{paper}.json"
        )

        copy(
            ROOT / "orchestrator/output-ablation" / paper / "state.json",
            f"loop-off/{short}.state.json",
        )
        copy(ROOT / "critic/output-ablation" / f"{paper}.json", f"loop-off/{short}.critic.json")
        copy(
            ROOT / "critic/output-ablation-loopon" / f"{paper}.json",
            f"loop-on/{short}.critic.json",
        )
        copy(ROOT / "orchestrator/output" / paper / "state.json", f"loop-on/{short}.state.json")

        rows.append(
            {
                "paper": paper,
                "claim_id": claim,
                "loop_off": {
                    "execution_verdict": (off_state or {}).get("verdict"),
                    "retry_count": (off_state or {}).get("retry_count"),
                    "attempts": len((off_state or {}).get("attempts") or []),
                    "stage_reached": (
                        ((off_state or {}).get("attempts") or [{}])[-1].get("stage_reached")
                        if (off_state or {}).get("attempts")
                        else None
                    ),
                    "wall_clock_seconds": sum(
                        a.get("wall_clock_seconds") or 0
                        for a in ((off_state or {}).get("attempts") or [])
                    ),
                    "fidelity_verdict": (off_crit or {}).get("verdict"),
                    "reproduced": (off_crit or {}).get("reproduced"),
                    "claimed": (off_crit or {}).get("claimed"),
                    "tolerance": (off_crit or {}).get("tolerance"),
                    "run_values": (off_crit or {}).get("run_values"),
                    "exceeds_claim": (off_crit or {}).get("exceeds_claim"),
                    "evidence": (off_crit or {}).get("evidence"),
                },
                "loop_on": {
                    "execution_verdict": (on_state or {}).get("verdict"),
                    "retry_count": (on_state or {}).get("retry_count"),
                    "attempts": len((on_state or {}).get("attempts") or []),
                    "fidelity_verdict": (on_crit or {}).get("verdict"),
                    "reproduced": (on_crit or {}).get("reproduced"),
                    "claimed": (on_crit or {}).get("claimed"),
                    "tolerance": (on_crit or {}).get("tolerance"),
                    "run_values": (on_crit or {}).get("run_values"),
                    "exceeds_claim": (on_crit or {}).get("exceeds_claim"),
                    "evidence": (on_crit or {}).get("evidence"),
                },
            }
        )

    (DATA / "summary.json").write_text(json.dumps(rows, indent=2))
    for r in rows:
        off, on = r["loop_off"], r["loop_on"]
        print(f"{r['paper'][:38]:40s} claim={r['claim_id']}")
        print(
            f"   loop-off: exec={off['execution_verdict']} fid={off['fidelity_verdict']} "
            f"val={off['reproduced']} retries={off['retry_count']} "
            f"wall={off['wall_clock_seconds']:.0f}s"
        )
        print(
            f"   loop-on : exec={on['execution_verdict']} fid={on['fidelity_verdict']} "
            f"val={on['reproduced']} retries={on['retry_count']}"
        )


if __name__ == "__main__":
    main()
