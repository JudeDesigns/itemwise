#!/usr/bin/env python3
"""Pull the SWE-bench Verified submission results and build the score matrix.

Every submission to the SWE-bench leaderboard publishes, in the public
`swe-bench/experiments` repository, exactly which of the 500 Verified instances
it resolved. That is a per-model, per-item pass/fail matrix - the input item
analysis needs, and one of the very few published at this scale.

Writes matrix.json next to this file. A copy is committed, so you can run
analyse.py without running this first; re-run it to refresh from upstream.

    python3 fetch.py [--repo /path/to/an/existing/clone]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = "https://github.com/swe-bench/experiments.git"
SPLIT = "verified"
N_INSTANCES = 500


def clone(dest: Path) -> Path:
    """Sparse, blobless clone of just the results files. ~5 MB, not ~2 GB."""
    print(f"cloning {UPSTREAM} (sparse) -> {dest}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
         UPSTREAM, str(dest)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set", "--no-cone",
         f"/evaluation/{SPLIT}/*/results/results.json",
         f"/evaluation/{SPLIT}/*/results/resolved_by_repo.json",
         f"/evaluation/{SPLIT}/*/metadata.yaml"],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "checkout"], check=True)
    return dest


def build(repo: Path) -> dict:
    root = repo / "evaluation" / SPLIT
    subs = sorted(p.parent.parent.name for p in root.glob("*/results/results.json"))
    if not subs:
        sys.exit(f"no submissions found under {root}")
    print(f"{len(subs)} submissions")

    # Per-repo instance counts. A handful of early submissions report totals for
    # the full SWE-bench split rather than Verified, so take the mode and check
    # it sums to the 500 instances Verified is defined to have.
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for s in subs:
        f = root / s / "results" / "resolved_by_repo.json"
        if f.exists():
            for repo_name, v in json.loads(f.read_text()).items():
                counts[repo_name][v["total"]] += 1
    totals = {r: c.most_common(1)[0][0] for r, c in counts.items()}
    if sum(totals.values()) != N_INSTANCES:
        sys.exit(f"per-repo totals sum to {sum(totals.values())}, expected {N_INSTANCES}")

    resolved = {
        s: set(json.loads((root / s / "results" / "results.json").read_text())
               .get("resolved", []))
        for s in subs
    }
    union = sorted(set().union(*resolved.values()))
    print(f"{len(union)} distinct instances resolved by at least one submission")

    # Instances no submission ever solved never appear in any `resolved` list, but
    # they are part of the suite and are the purest dead items there are. The
    # per-repo denominators tell us how many are missing from each repository.
    prefix_to_repo = {r.split("/")[0]: r for r in totals}
    seen = collections.Counter(prefix_to_repo[i.split("__")[0]] for i in union)
    ghosts = []
    for repo_name, total in sorted(totals.items()):
        missing = total - seen.get(repo_name, 0)
        if missing < 0:
            sys.exit(f"{repo_name}: more resolved than exist")
        ghosts += [f"{repo_name.replace('/', '__')}__UNSOLVED-{k + 1:02d}"
                   for k in range(missing)]
    print(f"{len(ghosts)} instances solved by nobody")

    item_ids = union + ghosts
    if len(item_ids) != N_INSTANCES:
        sys.exit(f"built {len(item_ids)} items, expected {N_INSTANCES}")

    names = {}
    for s in subs:
        f = root / s / "metadata.yaml"
        label = s
        if f.exists():
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("name:"):
                    label = line.split(":", 1)[1].strip().strip("\"'")
                    break
        names[s] = label

    return {
        "source": UPSTREAM,
        "split": SPLIT,
        "commit": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip(),
        "item_ids": item_ids,
        "n_never_solved": len(ghosts),
        "names": names,
        # `no_generation` / `no_logs` count as unresolved, matching how the
        # leaderboard itself scores a submission.
        "scores": {s: [1 if i in resolved[s] else 0 for i in item_ids] for s in subs},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", help="use an existing clone instead of cloning")
    args = ap.parse_args()

    if args.repo:
        data = build(Path(args.repo))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            data = build(clone(Path(tmp) / "experiments"))

    out = HERE / "matrix.json"
    out.write_text(json.dumps(data, indent=1, sort_keys=True))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
