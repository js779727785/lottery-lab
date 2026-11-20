"""Deterministic multi-ticket coverage without claiming predictive edge."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .metrics import BLUE_NUMBERS, RED_NUMBERS, TOTAL_COMBINATIONS, theoretical_baselines
from .models import Prediction
from .schema import TICKET_COST


RedSet = Tuple[int, ...]


@dataclass(frozen=True, order=True)
class Ticket:
    reds: RedSet
    blue: int

    def __post_init__(self) -> None:
        normalized = tuple(sorted(self.reds))
        if len(normalized) != 6 or len(set(normalized)) != 6:
            raise ValueError("a ticket must contain six distinct red balls")
        if any(number not in RED_NUMBERS for number in normalized):
            raise ValueError("red ball out of range")
        if self.blue not in BLUE_NUMBERS:
            raise ValueError("blue ball out of range")
        object.__setattr__(self, "reds", normalized)

    def as_dict(self) -> Dict[str, object]:
        return {"reds": list(self.reds), "blue": self.blue}


def _seed(target_issue: str, ticket_count: int) -> int:
    digest = hashlib.sha256(
        f"ssq-coverage-portfolio-v1:{target_issue}:{ticket_count}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def _model_score(reds: RedSet, rank_positions: Dict[int, int]) -> float:
    return sum(1.0 / (rank_positions[number] + 1.0) for number in reds)


def _candidate_red_sets(
    prediction: Prediction,
    target_issue: str,
    ticket_count: int,
) -> List[RedSet]:
    ranking = tuple(prediction.red_ranking)
    candidates: Set[RedSet] = {tuple(sorted(ranking[:6]))}

    for step in range(1, len(ranking)):
        if math.gcd(step, len(ranking)) != 1:
            continue
        for start in range(len(ranking)):
            candidates.add(
                tuple(
                    sorted(
                        ranking[(start + offset * step) % len(ranking)]
                        for offset in range(6)
                    )
                )
            )

    target_size = max(512, ticket_count * 64)
    rng = random.Random(_seed(target_issue, ticket_count))
    attempts = 0
    while len(candidates) < target_size and attempts < target_size * 20:
        candidates.add(tuple(sorted(rng.sample(ranking, 6))))
        attempts += 1
    return sorted(candidates)


def _red_four_plus_neighborhood(reds: RedSet) -> Set[RedSet]:
    outside = tuple(number for number in RED_NUMBERS if number not in reds)
    outcomes: Set[RedSet] = set()
    for hits in (4, 5, 6):
        for matching in combinations(reds, hits):
            for missing in combinations(outside, 6 - hits):
                outcomes.add(tuple(sorted((*matching, *missing))))
    return outcomes


def _overlap(left: RedSet, right: RedSet) -> int:
    return len(set(left) & set(right))


def generate_coverage_portfolio(
    prediction: Prediction,
    target_issue: str,
    ticket_count: int = 10,
) -> Tuple[Ticket, ...]:
    """Generate unique tickets with blue coverage and low red-neighborhood overlap."""

    if ticket_count < 1 or ticket_count > 128:
        raise ValueError("ticket_count must be between 1 and 128")

    candidates = _candidate_red_sets(prediction, target_issue, ticket_count)
    rank_positions = {
        number: index for index, number in enumerate(prediction.red_ranking)
    }
    selected: List[RedSet] = [tuple(sorted(prediction.red_ranking[:6]))]
    selected_set = set(selected)
    usage = {number: 0 for number in RED_NUMBERS}
    for number in selected[0]:
        usage[number] += 1
    covered = _red_four_plus_neighborhood(selected[0])

    while len(selected) < ticket_count:
        available = [candidate for candidate in candidates if candidate not in selected_set]
        if not available:
            raise RuntimeError("not enough unique red combinations")

        def heuristic(candidate: RedSet) -> Tuple[float, ...]:
            overlaps = [_overlap(candidate, existing) for existing in selected]
            return (
                float(max(overlaps)),
                float(sum(value * value for value in overlaps)),
                float(sum(usage[number] for number in candidate)),
                -_model_score(candidate, rank_positions),
            )

        shortlist = sorted(available, key=lambda item: (heuristic(item), item))[:24]
        best_candidate = None
        best_neighborhood: Set[RedSet] | None = None
        best_key = None
        for candidate in shortlist:
            neighborhood = _red_four_plus_neighborhood(candidate)
            overlaps = [_overlap(candidate, existing) for existing in selected]
            key = (
                len(neighborhood - covered),
                -max(overlaps),
                -sum(value * value for value in overlaps),
                -sum(usage[number] for number in candidate),
                _model_score(candidate, rank_positions),
                candidate,
            )
            if best_key is None or key > best_key:
                best_candidate = candidate
                best_neighborhood = neighborhood
                best_key = key

        assert best_candidate is not None and best_neighborhood is not None
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        covered.update(best_neighborhood)
        for number in best_candidate:
            usage[number] += 1

    blue_ranking = tuple(prediction.blue_ranking)
    return tuple(
        Ticket(reds=reds, blue=blue_ranking[index % len(blue_ranking)])
        for index, reds in enumerate(selected)
    )


def _top3_outcomes(tickets: Iterable[Ticket]) -> Set[Tuple[RedSet, int]]:
    outcomes: Set[Tuple[RedSet, int]] = set()
    for ticket in tickets:
        outside = tuple(number for number in RED_NUMBERS if number not in ticket.reds)
        for blue in BLUE_NUMBERS:
            outcomes.add((ticket.reds, blue))
        for matching in combinations(ticket.reds, 5):
            for missing in outside:
                outcomes.add((tuple(sorted((*matching, missing))), ticket.blue))
    return outcomes


def evaluate_portfolio(tickets: Sequence[Ticket]) -> Dict[str, object]:
    """Return exact union probabilities over the full 双色球 outcome space."""

    if not tickets:
        raise ValueError("portfolio must contain at least one ticket")

    unique_tickets = tuple(sorted(set(tickets)))
    unique_red_sets = {ticket.reds for ticket in unique_tickets}
    distinct_blues = {ticket.blue for ticket in unique_tickets}
    covered_red_outcomes: Set[RedSet] = set()
    for reds in unique_red_sets:
        covered_red_outcomes.update(_red_four_plus_neighborhood(reds))

    total_red_outcomes = comb(len(RED_NUMBERS), 6)
    red_coverage_count = len(covered_red_outcomes)
    blue_count = len(distinct_blues)
    any_prize_outcome_count = (
        red_coverage_count * len(BLUE_NUMBERS)
        + (total_red_outcomes - red_coverage_count) * blue_count
    )
    top3_outcome_count = len(_top3_outcomes(unique_tickets))
    jackpot_outcome_count = len(unique_tickets)

    baselines = theoretical_baselines()
    ticket_count = len(unique_tickets)
    random_any = 1.0 - (1.0 - baselines["any_prize_rate"]) ** ticket_count
    random_top3 = 1.0 - (1.0 - baselines["top3_prize_rate"]) ** ticket_count
    random_jackpot = 1.0 - (1.0 - baselines["exact_jackpot_rate"]) ** ticket_count

    return {
        "ticket_count": len(tickets),
        "unique_ticket_count": ticket_count,
        "distinct_blue_count": blue_count,
        "cost_yuan": len(tickets) * TICKET_COST,
        "guaranteed_any_prize": blue_count == len(BLUE_NUMBERS),
        "red_four_plus_coverage_count": red_coverage_count,
        "red_four_plus_coverage_rate": red_coverage_count / total_red_outcomes,
        "any_prize_outcome_count": any_prize_outcome_count,
        "any_prize_probability": any_prize_outcome_count / TOTAL_COMBINATIONS,
        "top3_outcome_count": top3_outcome_count,
        "top3_probability": top3_outcome_count / TOTAL_COMBINATIONS,
        "jackpot_outcome_count": jackpot_outcome_count,
        "jackpot_probability": jackpot_outcome_count / TOTAL_COMBINATIONS,
        "random_independent_baseline": {
            "any_prize_probability": random_any,
            "top3_probability": random_top3,
            "jackpot_probability": random_jackpot,
        },
        "absolute_lift_vs_random": {
            "any_prize_probability": any_prize_outcome_count / TOTAL_COMBINATIONS
            - random_any,
            "top3_probability": top3_outcome_count / TOTAL_COMBINATIONS - random_top3,
            "jackpot_probability": jackpot_outcome_count / TOTAL_COMBINATIONS
            - random_jackpot,
        },
    }


def build_portfolio_payload(
    prediction: Prediction,
    target_issue: str,
    ticket_count: int = 10,
) -> Dict[str, object]:
    tickets = generate_coverage_portfolio(prediction, target_issue, ticket_count)
    return {
        "label": f"{target_issue} · {ticket_count} 注覆盖组合",
        "strategy": (
            "保证票面唯一；前 16 注优先使用不同蓝球；红球组合以四中以上结果邻域增量最大化。"
        ),
        "tickets": [ticket.as_dict() for ticket in tickets],
        "metrics": evaluate_portfolio(tickets),
        "disclaimer": (
            "覆盖组合能提高同等注数下至少中一个奖的概率，但每张不同票的头奖概率仍为"
            " 1/17,721,088；提升来自增加不同组合覆盖，不是预测未来开奖号码。"
        ),
    }
