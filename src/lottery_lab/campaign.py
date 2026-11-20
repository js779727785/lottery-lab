"""Plan multi-draw ticket coverage without claiming predictive accuracy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .metrics import TOTAL_COMBINATIONS, theoretical_baselines
from .schema import TICKET_COST


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "campaign_plan.json"
WEB_REPORT = PROJECT_ROOT / "web" / "data" / "campaign.json"

_PRIZE_TARGETS = {
    "top3": {
        "label": "前三等奖及以上",
        "winning_outcomes": 178,
        "target_definition": "一等奖、二等奖或三等奖",
    },
    "jackpot": {
        "label": "头奖",
        "winning_outcomes": 1,
        "target_definition": "一等奖",
    },
}

# These are the fixed amounts used by the official records in the dataset.
# Tiers 1 and 2 are deliberately excluded because their amounts vary by draw.
_FIXED_PRIZE_OUTCOMES = {3: 162, 4: 7_695, 5: 137_475, 6: 1_921_140}
_FIXED_PRIZE_AMOUNTS = {3: 3_000, 4: 200, 5: 10, 6: 5}


def _validate_target(prize_target: str) -> Mapping[str, Any]:
    try:
        return _PRIZE_TARGETS[prize_target]
    except KeyError as exc:
        raise ValueError("prize_target must be 'top3' or 'jackpot'") from exc


def _validate_probability(value: float, name: str) -> float:
    probability = float(value)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return probability


def _distinct_hit_probability(winning_outcomes: int, total_outcomes: int, ticket_count: int) -> float:
    """Exact hit probability for uniformly selected distinct tickets."""

    if not 0 <= ticket_count <= total_outcomes:
        raise ValueError("ticket_count must be within the outcome space")
    if not 0 < winning_outcomes <= total_outcomes:
        raise ValueError("winning_outcomes must be within the outcome space")
    if ticket_count == 0:
        return 0.0
    if ticket_count > total_outcomes - winning_outcomes:
        return 1.0
    # C(T-W, n) / C(T, n) is also C(T-n, W) / C(T, W).  The latter
    # only iterates over the winning result count (at most 178 here),
    # avoiding catastrophic cancellation in lgamma(T + 1) differences.
    log_miss = sum(
        math.log1p(-ticket_count / (total_outcomes - offset))
        for offset in range(winning_outcomes)
    )
    return -math.expm1(log_miss)


def per_draw_probability(ticket_count: int, prize_target: str) -> float:
    """Return exact probability under uniform distinct-ticket sampling."""

    target = _validate_target(prize_target)
    return _distinct_hit_probability(
        int(target["winning_outcomes"]), TOTAL_COMBINATIONS, int(ticket_count)
    )


def campaign_probability(draw_count: int, ticket_count: int, prize_target: str) -> float:
    """Return probability of at least one hit over independent future draws."""

    if draw_count < 1:
        raise ValueError("draw_count must be positive")
    per_draw = per_draw_probability(ticket_count, prize_target)
    if per_draw >= 1.0:
        return 1.0
    return -math.expm1(draw_count * math.log1p(-per_draw))


def required_tickets_per_draw(
    draw_count: int,
    prize_target: str,
    target_probability: float = 0.99,
) -> int:
    """Find the minimum distinct tickets per draw for a campaign target."""

    if draw_count < 1:
        raise ValueError("draw_count must be positive")
    target = _validate_target(prize_target)
    desired_campaign_probability = _validate_probability(target_probability, "target_probability")
    required_per_draw = -math.expm1(
        math.log1p(-desired_campaign_probability) / draw_count
    )
    winning_outcomes = int(target["winning_outcomes"])
    low, high = 0, TOTAL_COMBINATIONS
    while low < high:
        middle = (low + high) // 2
        if _distinct_hit_probability(winning_outcomes, TOTAL_COMBINATIONS, middle) >= required_per_draw:
            high = middle
        else:
            low = middle + 1
    return low


def _fixed_expected_payout_per_ticket() -> float:
    return sum(
        _FIXED_PRIZE_OUTCOMES[tier] * _FIXED_PRIZE_AMOUNTS[tier]
        for tier in _FIXED_PRIZE_OUTCOMES
    ) / TOTAL_COMBINATIONS


def build_campaign_plan(
    draw_count: int,
    tickets_per_draw: int,
    prize_target: str = "top3",
    target_probability: float = 0.99,
    budget_yuan: int | None = None,
) -> Dict[str, Any]:
    """Build a budget-aware coverage plan for a fixed number of future draws."""

    if draw_count < 1:
        raise ValueError("draw_count must be positive")
    if tickets_per_draw < 0:
        raise ValueError("tickets_per_draw must be non-negative")
    if budget_yuan is not None and budget_yuan < 0:
        raise ValueError("budget_yuan must be non-negative")
    target = _validate_target(prize_target)
    target_probability = _validate_probability(target_probability, "target_probability")

    requested_tickets = min(int(tickets_per_draw), TOTAL_COMBINATIONS)
    affordable_tickets = (
        TOTAL_COMBINATIONS
        if budget_yuan is None
        else int(budget_yuan) // (draw_count * TICKET_COST)
    )
    applied_tickets = min(requested_tickets, affordable_tickets, TOTAL_COMBINATIONS)
    total_ticket_count = draw_count * applied_tickets
    cost_yuan = total_ticket_count * TICKET_COST
    fixed_payout = total_ticket_count * _fixed_expected_payout_per_ticket()
    campaign_rate = campaign_probability(draw_count, applied_tickets, prize_target)
    unused_budget = None if budget_yuan is None else int(budget_yuan) - cost_yuan

    return {
        "draw_count": draw_count,
        "prize_target": prize_target,
        "prize_target_label": target["label"],
        "requested_tickets_per_draw": requested_tickets,
        "tickets_per_draw": applied_tickets,
        "total_ticket_count": total_ticket_count,
        "cost_yuan": cost_yuan,
        "budget_yuan": budget_yuan,
        "unused_budget_yuan": unused_budget,
        "per_draw_probability": per_draw_probability(applied_tickets, prize_target),
        "campaign_probability": campaign_rate,
        "target_probability": target_probability,
        "target_met": campaign_rate >= target_probability,
        "expected_fixed_tier_payout_yuan": fixed_payout,
        "expected_loss_upper_bound_yuan": cost_yuan - fixed_payout,
        "fixed_prize_outcomes": dict(_FIXED_PRIZE_OUTCOMES),
        "fixed_prize_amounts_yuan": dict(_FIXED_PRIZE_AMOUNTS),
        "variable_prize_tiers_omitted": [1, 2],
        "coverage_only": True,
        "uniform_distinct_ticket_model": True,
        "prediction_accuracy_claim": False,
        "no_future_results_used": True,
        "independent_draw_assumption": True,
        "method": "每期在完整结果空间内按不重复彩票做精确覆盖；跨期按独立开奖复合。",
    }


def _requirement_row(
    draw_count: int,
    prize_target: str,
    target_probability: float,
) -> Dict[str, Any]:
    tickets = required_tickets_per_draw(draw_count, prize_target, target_probability)
    previous = max(0, tickets - 1)
    return {
        "draw_count": draw_count,
        "prize_target": prize_target,
        "target_probability": target_probability,
        "per_draw_target_probability": -math.expm1(
            math.log1p(-target_probability) / draw_count
        ),
        "tickets_per_draw": tickets,
        "previous_ticket_probability": campaign_probability(draw_count, previous, prize_target),
        "campaign_probability": campaign_probability(draw_count, tickets, prize_target),
        "total_ticket_count": draw_count * tickets,
        "total_cost_yuan": draw_count * tickets * TICKET_COST,
        "coverage_only": True,
        "prediction_accuracy_claim": False,
        "uniform_distinct_ticket_model": True,
    }


def build_campaign_report(
    draw_horizons: Iterable[int] = (10, 52, 100, 500, 1000),
    target_probability: float = 0.99,
    output_path: Path = DEFAULT_REPORT,
    web_path: Path = WEB_REPORT,
) -> Dict[str, Any]:
    """Write a reproducible X-draw coverage table for the dashboard and reports."""

    horizons = tuple(int(value) for value in draw_horizons)
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("draw_horizons must contain positive integers")
    target_probability = _validate_probability(target_probability, "target_probability")
    report: Dict[str, Any] = {
        "meta": {
            "experiment": "future X-draw coverage planner",
            "target_probability": target_probability,
            "draw_horizons": list(horizons),
            "total_combinations": TOTAL_COMBINATIONS,
            "ticket_cost_yuan": TICKET_COST,
            "no_future_results_used": True,
            "coverage_is_not_prediction": True,
            "uniform_distinct_ticket_model": True,
            "independent_draw_assumption": True,
        },
        "baselines": theoretical_baselines(),
        "requirements": {
            prize_target: [
                _requirement_row(draw_count, prize_target, target_probability)
                for draw_count in horizons
            ]
            for prize_target in ("top3", "jackpot")
        },
        "fixed_prize_assumptions": {
            "outcomes_by_tier": dict(_FIXED_PRIZE_OUTCOMES),
            "amounts_yuan_by_tier": dict(_FIXED_PRIZE_AMOUNTS),
            "omitted_variable_tiers": [1, 2],
        },
        "disclaimer": (
            "99% 仅表示在每期从完整票面空间均匀抽取指定数量的不重复组合、且开奖独立时至少覆盖一次目标结果的概率；"
            "它是组合覆盖和预算结论，不是单注预测准确率，也不证明模型识别了未来号码。"
        ),
    }
    for path in (output_path, web_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=0.99)
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--web-output", default=str(WEB_REPORT))
    parser.add_argument("--horizons", nargs="+", type=int, default=[10, 52, 100, 500, 1000])
    args = parser.parse_args()
    report = build_campaign_report(
        draw_horizons=args.horizons,
        target_probability=args.target,
        output_path=Path(args.output),
        web_path=Path(args.web_output),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
