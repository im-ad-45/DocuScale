"""Reciprocal Rank Fusion."""
from collections import defaultdict
from typing import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Merge several best-first ID rankings into one.

    Each ID earns ``1 / (k + rank)`` from every list it appears in (rank starts
    at 1). Returns ``(id, score)`` pairs, highest score first.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
