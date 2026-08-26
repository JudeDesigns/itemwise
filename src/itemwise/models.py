"""Data model: what an eval suite and a run over it look like."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class Item:
    """One eval case."""

    id: str
    prompt: str = ""
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        known = {"id", "prompt", "tags"}
        if "id" not in d:
            raise ValueError(f"item is missing required field 'id': {d!r}")
        return cls(
            id=str(d["id"]),
            prompt=str(d.get("prompt", "")),
            tags=tuple(d.get("tags", ()) or ()),
            meta={k: v for k, v in d.items() if k not in known},
        )


@dataclass
class Suite:
    """An ordered collection of eval items with unique ids."""

    items: list[Item]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        dupes: list[str] = []
        for it in self.items:
            if it.id in seen:
                dupes.append(it.id)
            seen.add(it.id)
        if dupes:
            raise ValueError(f"duplicate item ids: {sorted(set(dupes))}")

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)

    @property
    def ids(self) -> list[str]:
        return [i.id for i in self.items]

    @classmethod
    def from_dicts(cls, rows: Iterable[dict[str, Any]]) -> "Suite":
        return cls([Item.from_dict(r) for r in rows])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Suite":
        rows = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON - {exc}") from exc
        return cls.from_dicts(rows)


@dataclass
class RunResult:
    """Pass/fail outcomes for every (model, item) pair.

    ``scores`` maps a model name to that model's 0/1 score for each item, in
    the same order as ``suite.items``. This is deliberately the dumbest
    possible representation: whatever harness you already run - promptfoo,
    braintrust, a bare pytest loop, a notebook - can produce it in a few lines,
    and itemwise never needs to know how you called a model.
    """

    suite: Suite
    scores: dict[str, list[int]]

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("no models in scores")
        n = len(self.suite)
        for model, row in self.scores.items():
            if len(row) != n:
                raise ValueError(
                    f"model {model!r} has {len(row)} scores but the suite has {n} items"
                )
            bad = {s for s in row if s not in (0, 1)}
            if bad:
                raise ValueError(f"model {model!r} has non-binary scores: {sorted(bad)!r}")

    @property
    def models(self) -> list[str]:
        return list(self.scores)

    def item_scores(self, index: int) -> list[int]:
        """Every model's score on one item."""
        return [self.scores[m][index] for m in self.models]

    def item_matrix(self) -> list[list[int]]:
        """One row per item, one column per model."""
        return [self.item_scores(i) for i in range(len(self.suite))]

    def total_scores(self) -> list[float]:
        """Each model's total number of items passed, in ``models`` order."""
        return [float(sum(self.scores[m])) for m in self.models]

    @classmethod
    def from_records(cls, suite: Suite, records: Iterable[dict[str, Any]]) -> "RunResult":
        """Build from flat ``{"model": ..., "item_id": ..., "passed": ...}`` rows.

        Usually the most convenient entry point, because it is the shape most
        eval harnesses already emit.
        """
        index = {item_id: i for i, item_id in enumerate(suite.ids)}
        scores: dict[str, list[int | None]] = {}
        for rec in records:
            for key in ("model", "item_id", "passed"):
                if key not in rec:
                    raise ValueError(f"record missing {key!r}: {rec!r}")
            model, item_id = str(rec["model"]), str(rec["item_id"])
            if item_id not in index:
                raise ValueError(f"record refers to unknown item id {item_id!r}")
            scores.setdefault(model, [None] * len(suite))
            scores[model][index[item_id]] = 1 if rec["passed"] else 0

        complete: dict[str, list[int]] = {}
        for model, row in scores.items():
            missing = [suite.ids[i] for i, v in enumerate(row) if v is None]
            if missing:
                raise ValueError(
                    f"model {model!r} has no result for {len(missing)} item(s): "
                    f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
                )
            complete[model] = [int(v) for v in row]  # type: ignore[arg-type]
        return cls(suite=suite, scores=complete)
