"""Evaluation metrics and exact 双色球 baselines."""

from __future__ import annotations

import math
import random
from math import comb, log2
from typing import Iterable, List, Sequence, Tuple

from .schema import BLUE_MAX, RED_MAX, prize_tier


RED_NUMBERS = tuple(range(1, RED_MAX + 1))
BLUE_NUMBERS = tuple(range(1, BLUE_MAX + 1))
TOTAL_COMBINATIONS = comb(33, 6) * 16


def at_least_hits_probability(
    pool_size: int,
    min_hits: int,
    draw_size: int = 6,
    universe_size: int = RED_MAX,
) -> float:
    """Exact hypergeometric probability that a fixed pool covers >= min_hits."""

    if pool_size < 0 or pool_size > universe_size:
        raise ValueError("pool_size out of range")
    if min_hits < 0 or min_hits > draw_size:
        raise ValueError("min_hits out of range")
    total = comb(universe_size, draw_size)
    low = max(min_hits, draw_size - (universe_size - pool_size))
    high = min(draw_size, pool_size)
    successes = sum(
        comb(pool_size, hits) * comb(universe_size - pool_size, draw_size - hits)
        for hits in range(low, high + 1)
    )
    return successes / total


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def ndcg_at_k(ranking: Sequence[int], relevant: Iterable[int], k: int = 10) -> float:
    relevant_set = set(relevant)
    dcg = 0.0
    for index, number in enumerate(ranking[:k]):
        if number in relevant_set:
            dcg += 1.0 / log2(index + 2.0)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / log2(index + 2.0) for index in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def mean_actual_rank(ranking: Sequence[int], relevant: Iterable[int]) -> float:
    positions = {number: index + 1 for index, number in enumerate(ranking)}
    values = [positions[number] for number in relevant if number in positions]
    return sum(values) / len(values) if values else float(len(ranking) + 1)


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels)


def log_loss(probabilities: Sequence[float], labels: Sequence[int], eps: float = 1e-12) -> float:
    total = 0.0
    for probability, label in zip(probabilities, labels):
        p = min(1.0 - eps, max(eps, probability))
        total -= label * math.log(p) + (1 - label) * math.log(1.0 - p)
    return total / len(labels)


def paired_bootstrap_difference(
    model_values: Sequence[float],
    baseline_values: Sequence[float],
    n_boot: int = 2000,
    seed: int = 20261120,
) -> Tuple[float, float, float]:
    """Return point estimate and 95% paired-bootstrap interval."""

    if len(model_values) != len(baseline_values) or not model_values:
        raise ValueError("paired bootstrap requires equal, non-empty vectors")
    differences = [a - b for a, b in zip(model_values, baseline_values)]
    point = sum(differences) / len(differences)
    rng = random.Random(seed)
    samples: List[float] = []
    n = len(differences)
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += differences[rng.randrange(n)]
        samples.append(total / n)
    samples.sort()
    low = samples[int(0.025 * (n_boot - 1))]
    high = samples[int(0.975 * (n_boot - 1))]
    return (point, low, high)


def _any_prize_combinations() -> int:
    total = 0
    for red_hits in range(7):
        red_ways = comb(6, red_hits) * comb(27, 6 - red_hits)
        if prize_tier(red_hits, True) is not None:
            total += red_ways
        if prize_tier(red_hits, False) is not None:
            total += red_ways * 15
    return total


def theoretical_baselines() -> dict:
    top3_or_better = 16 + comb(6, 5) * comb(27, 1)
    return {
        "exact_jackpot_rate": 1.0 / TOTAL_COMBINATIONS,
        "second_prize_rate": 15.0 / TOTAL_COMBINATIONS,
        "third_prize_rate": float(comb(6, 5) * comb(27, 1)) / TOTAL_COMBINATIONS,
        "top3_prize_rate": top3_or_better / TOTAL_COMBINATIONS,
        "jackpot_one_in": float(TOTAL_COMBINATIONS),
        "top3_prize_one_in": TOTAL_COMBINATIONS / top3_or_better,
        "top10_red_coverage_rate": 1.0 - comb(23, 6) / comb(33, 6),
        "top10_five_red_rate": at_least_hits_probability(10, 5),
        "top10_top4_top3_candidate_rate": (
            at_least_hits_probability(10, 5) * 4.0 / BLUE_MAX
        ),
        "top6_red_coverage_rate": 1.0 - comb(27, 6) / comb(33, 6),
        "any_prize_rate": _any_prize_combinations() / TOTAL_COMBINATIONS,
        "blue_hit_rate": 1.0 / 16.0,
        "expected_red_hits": 6.0 * 6.0 / 33.0,
        "expected_top10_red_hits": 10.0 * 6.0 / 33.0,
        "total_combinations": TOTAL_COMBINATIONS,
    }
