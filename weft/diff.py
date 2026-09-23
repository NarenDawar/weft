from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from weft.storage import Storage


@dataclass(frozen=True)
class DiffResult:
    shared_prefix_length: int
    divergence_index: Optional[int]
    divergence_reason: Optional[str]
    run_a_length: int
    run_b_length: int


def diff_runs(storage: Storage, run_id_a: str, run_id_b: str) -> DiffResult:
    history_a = storage.resolve_full_history(run_id_a)
    history_b = storage.resolve_full_history(run_id_b)

    shortest = min(len(history_a), len(history_b))
    for i in range(shortest):
        entry_a = history_a[i]
        entry_b = history_b[i]
        if entry_a.step != entry_b.step:
            return DiffResult(
                shared_prefix_length=i,
                divergence_index=i,
                divergence_reason="decision",
                run_a_length=len(history_a),
                run_b_length=len(history_b),
            )
        if entry_a.observation != entry_b.observation:
            return DiffResult(
                shared_prefix_length=i,
                divergence_index=i,
                divergence_reason="observation",
                run_a_length=len(history_a),
                run_b_length=len(history_b),
            )

    divergence_index = shortest if len(history_a) != len(history_b) else None
    return DiffResult(
        shared_prefix_length=shortest,
        divergence_index=divergence_index,
        divergence_reason=None,
        run_a_length=len(history_a),
        run_b_length=len(history_b),
    )
