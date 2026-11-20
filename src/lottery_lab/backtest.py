"""Walk-forward evaluation with strict no-lookahead semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Sequence, Tuple

from .metrics import (
    BLUE_NUMBERS,
    RED_NUMBERS,
    brier_score,
    log_loss,
    mean_actual_rank,
    ndcg_at_k,
    theoretical_baselines,
    wilson_interval,
)
from .models import Model, Prediction
from .schema import Draw, TICKET_COST, prize_tier


@dataclass(frozen=True)
class CaseResult:
    issue: str
    draw_date: date
    actual_reds: Tuple[int, ...]
    actual_blue: int
    red_ranking: Tuple[int, ...]
    blue_ranking: Tuple[int, ...]
    top10_reds: Tuple[int, ...]
    top4_blues: Tuple[int, ...]
    top10_hits: int
    red_hits: int
    blue_hit: bool
    tier: int | None
    payout: int
    top10_coverage: bool
    any_prize: bool
    ndcg10: float
    mean_rank: float
    brier_red: float
    brier_blue: float
    logloss_red: float
    logloss_blue: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "issue": self.issue,
            "date": self.draw_date.isoformat(),
            "actual_reds": list(self.actual_reds),
            "actual_blue": self.actual_blue,
            "red_ranking": list(self.red_ranking),
            "blue_ranking": list(self.blue_ranking),
            "top10_reds": list(self.top10_reds),
            "top4_blues": list(self.top4_blues),
            "top10_hits": self.top10_hits,
            "red_hits": self.red_hits,
            "blue_hit": self.blue_hit,
            "tier": self.tier,
            "payout": self.payout,
            "top10_coverage": self.top10_coverage,
            "any_prize": self.any_prize,
            "ndcg10": self.ndcg10,
            "mean_rank": self.mean_rank,
            "brier_red": self.brier_red,
            "brier_blue": self.brier_blue,
            "logloss_red": self.logloss_red,
            "logloss_blue": self.logloss_blue,
        }


def evaluate_prediction(prediction: Prediction, draw: Draw) -> CaseResult:
    actual_reds = frozenset(draw.reds)
    top10_reds = tuple(prediction.red_ranking[:10])
    top4_blues = tuple(prediction.blue_ranking[:4])
    top10_hits = len(set(top10_reds) & actual_reds)
    single_reds = tuple(prediction.red_ranking[:6])
    red_hits = len(set(single_reds) & actual_reds)
    blue_hit = prediction.blue_ranking[0] == draw.blue
    tier = prize_tier(red_hits, blue_hit)
    payout = int((draw.prize_amounts or {}).get(tier, 0)) if tier is not None else 0

    red_labels = [1 if number in actual_reds else 0 for number in RED_NUMBERS]
    blue_labels = [1 if number == draw.blue else 0 for number in BLUE_NUMBERS]
    red_probabilities = [float(prediction.red_probs[number]) for number in RED_NUMBERS]
    blue_probabilities = [float(prediction.blue_probs[number]) for number in BLUE_NUMBERS]

    return CaseResult(
        issue=draw.issue,
        draw_date=draw.draw_date,
        actual_reds=tuple(draw.reds),
        actual_blue=draw.blue,
        red_ranking=tuple(prediction.red_ranking),
        blue_ranking=tuple(prediction.blue_ranking),
        top10_reds=top10_reds,
        top4_blues=top4_blues,
        top10_hits=top10_hits,
        red_hits=red_hits,
        blue_hit=blue_hit,
        tier=tier,
        payout=payout,
        top10_coverage=top10_hits > 0,
        any_prize=tier is not None,
        ndcg10=ndcg_at_k(prediction.red_ranking, actual_reds, 10),
        mean_rank=mean_actual_rank(prediction.red_ranking, actual_reds),
        brier_red=brier_score(red_probabilities, red_labels),
        brier_blue=brier_score(blue_probabilities, blue_labels),
        logloss_red=log_loss(red_probabilities, red_labels),
        logloss_blue=log_loss(blue_probabilities, blue_labels),
    )


def run_model(
    model: Model,
    draws: Sequence[Draw],
    eval_start: date,
    eval_end: date,
) -> Tuple[List[CaseResult], Model]:
    """Warm up on all prior draws, then evaluate only within the date window."""

    model.reset()
    cases: List[CaseResult] = []
    for draw in draws:
        if draw.draw_date < eval_start:
            model.observe(draw)
            continue
        if draw.draw_date > eval_end:
            break
        prediction = model.predict(draw.issue)
        cases.append(evaluate_prediction(prediction, draw))
        model.observe(draw)
    return cases, model


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(cases: Sequence[CaseResult]) -> Dict[str, object]:
    n = len(cases)
    if n == 0:
        return {"n": 0}

    top10_successes = sum(1 for case in cases if case.top10_coverage)
    any_prize_successes = sum(1 for case in cases if case.any_prize)
    tier_counts = {str(tier): 0 for tier in range(1, 7)}
    red_hits_distribution = {str(hits): 0 for hits in range(7)}
    for case in cases:
        if case.tier is not None:
            tier_counts[str(case.tier)] += 1
        red_hits_distribution[str(case.red_hits)] += 1

    total_cost = n * TICKET_COST
    total_payout = sum(case.payout for case in cases)
    coverage_ci = wilson_interval(top10_successes, n)
    any_prize_ci = wilson_interval(any_prize_successes, n)
    baselines = theoretical_baselines()

    return {
        "n": n,
        "exact_jackpot_count": tier_counts["1"],
        "exact_jackpot_rate": tier_counts["1"] / n,
        "any_prize_count": any_prize_successes,
        "any_prize_rate": any_prize_successes / n,
        "any_prize_wilson_95": list(any_prize_ci),
        "top10_coverage_count": top10_successes,
        "top10_coverage_rate": top10_successes / n,
        "top10_coverage_wilson_95": list(coverage_ci),
        "avg_top10_hits": _mean([case.top10_hits for case in cases]),
        "avg_red_hits": _mean([case.red_hits for case in cases]),
        "blue_hit_rate": _mean([1.0 if case.blue_hit else 0.0 for case in cases]),
        "mean_ndcg_at_10": _mean([case.ndcg10 for case in cases]),
        "mean_actual_rank": _mean([case.mean_rank for case in cases]),
        "brier_red": _mean([case.brier_red for case in cases]),
        "brier_blue": _mean([case.brier_blue for case in cases]),
        "logloss_red": _mean([case.logloss_red for case in cases]),
        "logloss_blue": _mean([case.logloss_blue for case in cases]),
        "tier_counts": tier_counts,
        "red_hits_distribution": red_hits_distribution,
        "total_cost": total_cost,
        "total_payout": total_payout,
        "net": total_payout - total_cost,
        "roi": (total_payout - total_cost) / total_cost if total_cost else 0.0,
        "baseline": baselines,
    }


def rolling_series(cases: Sequence[CaseResult], window: int = 50) -> List[Dict[str, object]]:
    series: List[Dict[str, object]] = []
    for index, case in enumerate(cases):
        start = max(0, index - window + 1)
        subset = cases[start : index + 1]
        series.append(
            {
                "issue": case.issue,
                "date": case.draw_date.isoformat(),
                "window": len(subset),
                "top10_coverage_rate": sum(1 for item in subset if item.top10_coverage) / len(subset),
                "any_prize_rate": sum(1 for item in subset if item.any_prize) / len(subset),
                "avg_red_hits": _mean([item.red_hits for item in subset]),
            }
        )
    return series


def yearly_summary(cases: Sequence[CaseResult]) -> List[Dict[str, object]]:
    groups: Dict[int, List[CaseResult]] = {}
    for case in cases:
        groups.setdefault(case.draw_date.year, []).append(case)
    return [
        {
            "year": year,
            "n": len(group),
            "top10_coverage_rate": sum(1 for case in group if case.top10_coverage) / len(group),
            "any_prize_rate": sum(1 for case in group if case.any_prize) / len(group),
            "avg_red_hits": _mean([case.red_hits for case in group]),
        }
        for year, group in sorted(groups.items())
    ]
