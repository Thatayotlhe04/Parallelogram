"""Choose a training subset under a token budget and coverage minimums.

Every policy runs through the same two phases, so that a comparison between
policies is a comparison of their *ordering* and nothing else:

1. Coverage. For each minimum in the contract (so many records or tokens of
   a language, domain or skill), take that group's best candidates, by the
   policy's own ordering, until the minimum is met.
2. Fill. Spend what is left of the budget in the policy's order, skipping
   records that no longer fit.

Policies differ only in ordering: `random` is a seeded permutation,
`diversity` is greedy k-center on lexical features, and `less` ranks by
estimated utility with an optional redundancy penalty.

Nothing here stores a similarity for every pair of candidates. The adaptive
orderings keep one number per candidate (distance to the nearest pick, or
similarity to the most similar pick) and refresh it with one batched product
per pick.
"""
from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .config import CoverageConstraint, digest
from .errors import SelectionInfeasible

TEXT_FEATURE_DIM = 256
_TEXT_LIMIT = 2000


@dataclass(frozen=True)
class Item:
    id: str
    tokens: int
    attrs: dict[str, str] = field(default_factory=dict, compare=False)
    sha: str = ""          # content hash of the source line


@dataclass
class Selection:
    policy: str
    indices: list[int]
    reasons: dict[int, str]
    tokens: int
    budget: int
    coverage: list[dict]
    ids: list[str] = field(default_factory=list)
    content: list[str] = field(default_factory=list)

    @property
    def sha256(self) -> str:
        """Identity of the subset: the same records, by id and by content, in
        any order hash alike. An edited record under an unchanged id does not."""
        return digest(sorted(zip(self.ids, self.content or [""] * len(self.ids))))


class StaticOrder:
    """A fixed ranking of the pool, best first."""

    def __init__(self, name: str, ranking: Sequence[int]):
        self.name = name
        self.ranking = list(ranking)
        self.static = True

    def update(self, index: int) -> None:
        return None


class KCenterOrder:
    """Greedy k-center: always take the candidate farthest from every pick.

    Starts from the most central candidate so the result does not depend on a
    random draw. Holds one distance per candidate; each pick costs one
    features-times-vector product, done in blocks.
    """
    static = False
    name = "diversity"

    def __init__(self, np: Any, features: Any, block: int = 65536):
        self.np = np
        self.features = features
        self.block = block
        self.min_distance = np.full(features.shape[0], np.inf, dtype=np.float64)
        center = features.mean(axis=0)
        self._centrality = features @ center
        self._started = False

    def next(self, eligible: Any) -> int:
        np = self.np
        score = self.min_distance if self._started else self._centrality
        return int(np.argmax(np.where(eligible, score, -np.inf)))

    def update(self, index: int) -> None:
        np = self.np
        self._started = True
        pick = self.features[index]
        for start in range(0, self.features.shape[0], self.block):
            block = self.features[start:start + self.block]
            distance = 1.0 - block @ pick
            np.minimum(self.min_distance[start:start + self.block], distance,
                       out=self.min_distance[start:start + self.block])


class UtilityOrder:
    """Estimated utility, minus a penalty for resembling something already picked.

    `priority = utility - lambda * max_similarity_to_picks` (maximal marginal
    relevance). The penalty needs gradient features, which are gathered only
    for a shortlist of the highest-utility candidates; the rest of the pool
    is reached, by plain utility, only once the shortlist is used up or to
    satisfy a coverage minimum.
    """
    static = False
    name = "less"

    def __init__(self, np: Any, utility: Any, shortlist: Sequence[int], features: Any, lam: float):
        self.np = np
        self.utility = np.asarray(utility, dtype=np.float64)
        self.lam = lam
        self.features = features
        self.position = {int(i): p for p, i in enumerate(shortlist)}
        self.in_shortlist = np.zeros(len(self.utility), dtype=bool)
        self.in_shortlist[list(shortlist)] = True
        self.shortlist = np.asarray(list(shortlist), dtype=np.int64)
        self.max_similarity = np.zeros(len(shortlist), dtype=np.float64)

    def next(self, eligible: Any) -> int:
        np = self.np
        inside = eligible & self.in_shortlist
        if inside.any():
            priority = np.full(len(self.utility), -np.inf)
            priority[self.shortlist] = self.utility[self.shortlist] - self.lam * self.max_similarity
            return int(np.argmax(np.where(inside, priority, -np.inf)))
        return int(np.argmax(np.where(eligible, self.utility, -np.inf)))

    def update(self, index: int) -> None:
        position = self.position.get(int(index))
        if position is None:
            return
        similarity = self.features @ self.features[position]
        self.np.maximum(self.max_similarity, similarity, out=self.max_similarity)


def text_features(np: Any, texts: Iterable[str], dim: int = TEXT_FEATURE_DIM) -> Any:
    """Unit-length hashed character-trigram counts, one row per text.

    A deliberately model-free notion of similarity for the diversity
    baseline: it must not borrow anything from the model-aware scorer it is
    compared against. Only the first 2,000 characters of a record count.
    """
    rows = []
    for text in texts:
        row = np.zeros(dim, dtype=np.float32)
        text = " ".join(text.lower().split())[:_TEXT_LIMIT]
        for i in range(max(1, len(text) - 2)):
            row[zlib.crc32(text[i:i + 3].encode("utf-8")) % dim] += 1.0
        np.sqrt(row, out=row)
        norm = float(np.linalg.norm(row))
        rows.append(row / norm if norm else row)
    return np.stack(rows) if rows else np.zeros((0, dim), dtype=np.float32)


def random_order(n: int, seed: int) -> StaticOrder:
    ranking = list(range(n))
    random.Random(f"select:{seed}").shuffle(ranking)
    return StaticOrder("random", ranking)


def utility_order(np: Any, utility: Any) -> StaticOrder:
    """Highest estimated utility first; ties broken by pool position."""
    return StaticOrder("less", np.argsort(-np.asarray(utility, dtype=np.float64), kind="stable").tolist())


def shortlist(np: Any, utility: Any, tokens: Sequence[int], budget: int, pool_factor: float,
              available: Any) -> list[int]:
    """The highest-utility candidates the redundancy penalty will consider."""
    live = [i for i in np.argsort(-np.asarray(utility, dtype=np.float64), kind="stable").tolist() if available[i]]
    if not live:
        return []
    mean = sum(tokens[i] for i in live) / len(live)
    return live[:min(len(live), max(1, math.ceil(pool_factor * budget / max(mean, 1.0))))]


def select(np: Any, items: Sequence[Item], budget: int, constraints: Sequence[CoverageConstraint],
           order: Any, available: Any = None) -> Selection:
    """Run the two phases for one policy. See the module docstring.

    `available` masks candidates the policy may not use at all (for example
    ones below a retention floor). Raises `SelectionInfeasible` when a
    coverage minimum cannot be met from what is available within the budget.
    """
    n = len(items)
    tokens = np.asarray([item.tokens for item in items], dtype=np.int64)
    chosen = np.zeros(n, dtype=bool)
    usable = np.ones(n, dtype=bool) if available is None else np.asarray(available, dtype=bool).copy()
    picks: list[int] = []
    reasons: dict[int, str] = {}
    remaining = budget

    def take(index: int, reason: str) -> None:
        nonlocal remaining
        chosen[index] = True
        remaining -= int(tokens[index])
        picks.append(index)
        reasons[index] = reason
        order.update(index)

    def best(member: Any) -> int | None:
        eligible = member & ~chosen & usable & (tokens <= remaining)
        if not eligible.any():
            return None
        if order.static:
            for index in order.ranking:
                if eligible[index]:
                    return index
        return order.next(eligible)

    coverage: list[dict] = []
    for constraint in constraints:
        member = np.asarray([item.attrs.get(constraint.field) == constraint.value for item in items], dtype=bool)
        if not member.any():
            raise SelectionInfeasible(
                f"selection.coverage: no candidate has {constraint.field}={constraint.value!r}")
        while (int((chosen & member).sum()) < constraint.min_records
               or int(tokens[chosen & member].sum()) < constraint.min_tokens):
            index = best(member)
            if index is None:
                raise SelectionInfeasible(
                    f"selection.coverage: cannot reach the minimum for {constraint.field}="
                    f"{constraint.value!r} within the token budget ({int((chosen & member).sum())} records, "
                    f"{int(tokens[chosen & member].sum())} tokens selected; {remaining} tokens left)")
            take(index, f"coverage:{constraint.field}={constraint.value}")

    everyone = np.ones(n, dtype=bool)
    if order.static:
        # One pass: a record that does not fit now cannot fit later either,
        # because the remaining budget only shrinks.
        for index in order.ranking:
            if not chosen[index] and usable[index] and tokens[index] <= remaining:
                take(index, f"policy:{order.name}")
    else:
        while (index := best(everyone)) is not None:
            take(index, f"policy:{order.name}")

    for constraint in constraints:
        member = np.asarray([item.attrs.get(constraint.field) == constraint.value for item in items], dtype=bool)
        coverage.append({
            "field": constraint.field, "value": constraint.value,
            "min_records": constraint.min_records, "min_tokens": constraint.min_tokens,
            "selected_records": int((chosen & member).sum()),
            "selected_tokens": int(tokens[chosen & member].sum()),
            "pool_records": int(member.sum()), "pool_tokens": int(tokens[member].sum()),
        })
    if not picks:
        raise SelectionInfeasible(
            "selection: no available candidate fits the token budget "
            f"({int(usable.sum())} available, smallest is {int(tokens[usable].min()) if usable.any() else 'n/a'} "
            f"tokens, budget {budget})")
    return Selection(policy=order.name, indices=picks, reasons=reasons, tokens=budget - remaining,
                     budget=budget, coverage=coverage, ids=[items[i].id for i in picks],
                     content=[items[i].sha for i in picks])


def distribution(items: Sequence[Item], indices: Sequence[int], fields: Sequence[str]) -> dict:
    """Share of each field value in the pool and in the selection, by records
    and by tokens, so a report can show what selection did to coverage."""
    picked = set(indices)
    out: dict[str, dict] = {}
    pool_tokens = sum(item.tokens for item in items) or 1
    selected_tokens = sum(items[i].tokens for i in picked) or 1
    for name in fields:
        values: dict[str, dict] = {}
        for index, item in enumerate(items):
            value = item.attrs.get(name, "(missing)")
            row = values.setdefault(value, {"pool_records": 0, "pool_tokens": 0,
                                            "selected_records": 0, "selected_tokens": 0})
            row["pool_records"] += 1
            row["pool_tokens"] += item.tokens
            if index in picked:
                row["selected_records"] += 1
                row["selected_tokens"] += item.tokens
        for row in values.values():
            row["pool_token_share"] = round(row["pool_tokens"] / pool_tokens, 4)
            row["selected_token_share"] = round(row["selected_tokens"] / selected_tokens, 4)
        out[name] = dict(sorted(values.items()))
    return out
