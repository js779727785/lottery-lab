"""Audit the maximum honest jackpot/top-3 accuracy of a one-ticket predictor."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .backtest import CaseResult, run_model, summarize
from .data import DEFAULT_DATA_DIR, load_draws
from .metrics import (
    at_least_hits_probability,
    paired_bootstrap_difference,
    theoretical_baselines,
    wilson_interval,
)
from .models import model_registry
from .schema import BLUE_MAX, TICKET_COST, validate_draw_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "reports" / "extreme_audit.json"
WEB_JSON = PROJECT_ROOT / "web" / "data" / "extreme.json"
DEFAULT_TARGET = 0.99


def _log_combinations(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _iid_attempts_for_confidence(probability: float, target: float) -> int:
    if not 0.0 < probability < 1.0 or not 0.0 < target < 1.0:
        raise ValueError("probability and target must be in (0, 1)")
    return math.ceil(math.log(1.0 - target) / math.log(1.0 - probability))


def _distinct_tickets_for_confidence(
    winning_tickets: int,
    total_tickets: int,
    target: float,
) -> int:
    """Random distinct tickets without replacement, exact hypergeometric tail."""

    if winning_tickets <= 0 or winning_tickets > total_tickets:
        raise ValueError("winning_tickets out of range")
    low, high = 0, total_tickets
    while low < high:
        middle = (low + high) // 2
        log_miss = _log_combinations(total_tickets - winning_tickets, middle) - _log_combinations(
            total_tickets, middle
        )
        hit_probability = -math.expm1(log_miss)
        if hit_probability < target:
            low = middle + 1
        else:
            high = middle
    return low


def _candidate_top3_hit(case: CaseResult) -> bool:
    return len(set(case.top10_reds) & set(case.actual_reds)) >= 5 and case.actual_blue in case.top4_blues


def _model_summary(cases: Sequence[CaseResult]) -> Dict[str, Any]:
    summary = summarize(cases)
    tiers = summary["tier_counts"]
    top3_count = sum(int(tiers[str(tier)]) for tier in (1, 2, 3))
    candidate_count = sum(1 for case in cases if _candidate_top3_hit(case))
    n = len(cases)
    top3_ci = wilson_interval(top3_count, n)
    candidate_ci = wilson_interval(candidate_count, n)
    jackpot_ci = wilson_interval(summary["exact_jackpot_count"], n)
    return {
        "n": n,
        "jackpot_count": summary["exact_jackpot_count"],
        "jackpot_rate": summary["exact_jackpot_rate"],
        "jackpot_wilson_95": list(jackpot_ci),
        "top3_count": top3_count,
        "top3_rate": top3_count / n if n else 0.0,
        "top3_wilson_95": list(top3_ci),
        "top10_top4_candidate_count": candidate_count,
        "top10_top4_candidate_rate": candidate_count / n if n else 0.0,
        "top10_top4_candidate_wilson_95": list(candidate_ci),
        "any_prize_count": summary["any_prize_count"],
        "any_prize_rate": summary["any_prize_rate"],
        "top10_coverage_count": summary["top10_coverage_count"],
        "top10_coverage_rate": summary["top10_coverage_rate"],
        "avg_top10_hits": summary["avg_top10_hits"],
        "avg_red_hits": summary["avg_red_hits"],
        "mean_ndcg_at_10": summary["mean_ndcg_at_10"],
        "mean_joint_logloss": summary["logloss_red"] + summary["logloss_blue"],
        "tier_counts": tiers,
        "roi": summary["roi"],
    }


def _pool_requirement(target: float) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for red_pool in (10, 15, 20, 25, 28, 30, 31, 32, 33):
        red_rate = at_least_hits_probability(red_pool, 5)
        rows.append(
            {
                "red_pool": red_pool,
                "blue_pool": BLUE_MAX,
                "top3_candidate_rate": red_rate,
                "meets_target": red_rate >= target,
            }
        )
    minimum = next(row for row in rows if row["meets_target"])
    return {
        "target": target,
        "all_blues": True,
        "minimum_red_pool": minimum["red_pool"],
        "minimum_red_pool_rate": minimum["top3_candidate_rate"],
        "rows": rows,
    }


def run_audit(
    data_dir: Path = DEFAULT_DATA_DIR,
    eval_start: date = date(2008, 1, 1),
    target: float = DEFAULT_TARGET,
    json_path: Path = DEFAULT_JSON,
    web_json_path: Path = WEB_JSON,
) -> Dict[str, Any]:
    draws = load_draws(data_dir / "ssq_history.csv")
    validate_draw_sequence(draws)
    eval_end = draws[-1].draw_date
    warmup = [draw for draw in draws if draw.draw_date < eval_start]
    if not warmup:
        raise RuntimeError("extreme audit requires a warm-up period")

    baselines = theoretical_baselines()
    total_tickets = int(round(1.0 / baselines["exact_jackpot_rate"]))
    top3_winning_tickets = int(round(baselines["top3_prize_rate"] * total_tickets))
    iid_top3 = _iid_attempts_for_confidence(baselines["top3_prize_rate"], target)
    distinct_top3 = _distinct_tickets_for_confidence(
        top3_winning_tickets, total_tickets, target
    )
    iid_jackpot = _iid_attempts_for_confidence(baselines["exact_jackpot_rate"], target)
    distinct_jackpot = _distinct_tickets_for_confidence(1, total_tickets, target)

    model_rows: List[Dict[str, Any]] = []
    model_cases: Dict[str, List[CaseResult]] = {}
    for model in model_registry():
        cases, _ = run_model(model, draws, eval_start, eval_end)
        model_cases[model.key] = cases
        model_rows.append(
            {
                "key": model.key,
                "label": model.label,
                **_model_summary(cases),
            }
        )

    random_cases = model_cases["random"]
    comparisons_vs_random: Dict[str, Dict[str, Any]] = {}
    for key, cases in model_cases.items():
        if key == "random":
            continue
        ndcg = paired_bootstrap_difference(
            [case.ndcg10 for case in cases],
            [case.ndcg10 for case in random_cases],
        )
        joint_logloss = paired_bootstrap_difference(
            [case.logloss_red + case.logloss_blue for case in random_cases],
            [case.logloss_red + case.logloss_blue for case in cases],
        )
        comparisons_vs_random[key] = {
            "ndcg_at_10": {
                "difference": ndcg[0],
                "ci95_low": ndcg[1],
                "ci95_high": ndcg[2],
                "significant_improvement": ndcg[1] > 0.0,
            },
            "joint_logloss": {
                "difference": joint_logloss[0],
                "ci95_low": joint_logloss[1],
                "ci95_high": joint_logloss[2],
                "significant_improvement": joint_logloss[1] > 0.0,
            },
            "dual_metric_gate_passed": ndcg[1] > 0.0 and joint_logloss[1] > 0.0,
        }

    best_top3 = max(model_rows, key=lambda row: (row["top3_rate"], row["key"]))
    best_candidate = max(
        model_rows, key=lambda row: (row["top10_top4_candidate_rate"], row["key"])
    )
    payload: Dict[str, Any] = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "pure-prediction ceiling audit",
            "data_rows": len(draws),
            "data_first_issue": draws[0].issue,
            "data_last_issue": draws[-1].issue,
            "eval_start": eval_start.isoformat(),
            "eval_end": eval_end.isoformat(),
            "warmup_draws": len(warmup),
            "eval_draws": len(draws) - len(warmup),
            "no_lookahead": True,
            "history_replay_is_not_prediction": True,
            "predict_before_observe": True,
            "each_prediction_uses_all_earlier_draws": True,
            "observed_result_updates_future_only": True,
            "historical_results_cannot_promote": True,
            "target": target,
        },
        "baselines": baselines,
        "single_ticket_99pct": {
            "top3_or_better": {
                "per_draw_probability": baselines["top3_prize_rate"],
                "one_in": baselines["top3_prize_one_in"],
                "iid_ticket_attempts": iid_top3,
                "random_distinct_tickets": distinct_top3,
                "cost_yuan": distinct_top3 * TICKET_COST,
            },
            "jackpot": {
                "per_draw_probability": baselines["exact_jackpot_rate"],
                "one_in": baselines["jackpot_one_in"],
                "iid_ticket_attempts": iid_jackpot,
                "random_distinct_tickets": distinct_jackpot,
                "cost_yuan": distinct_jackpot * TICKET_COST,
            },
        },
        "candidate_pool_99pct": _pool_requirement(target),
        "models": model_rows,
        "comparisons_vs_random": comparisons_vs_random,
        "promotion": {
            "promoted_from_history": False,
            "action": "history_replay_can_reject_or_pre_register_but_cannot_promote",
            "required_future_evidence": (
                "候选必须在开奖前连续冻结至少 104 期，并同时通过 NDCG@10 与"
                "红蓝联合对数损失的预设门槛。"
            ),
        },
        "observed_ceiling": {
            "best_top3_model": best_top3["key"],
            "best_top3_rate": best_top3["top3_rate"],
            "best_top10_top4_candidate_model": best_candidate["key"],
            "best_top10_top4_candidate_rate": best_candidate["top10_top4_candidate_rate"],
            "best_jackpot_count": max(row["jackpot_count"] for row in model_rows),
        },
        "conclusion": (
            "在公平独立开奖假设下，单注每期前三等奖概率固定为 178/17,721,088；"
            "历史飞轮只能估计该概率，不能把单注每期准确率推到 99%。"
            "达到 99% 的单次命中保证需要多票组合、候选空间扩张，或使用未来信息；"
            "后者属于作弊，未纳入本审计。"
        ),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    web_json_path.parent.mkdir(parents=True, exist_ok=True)
    web_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    args = parser.parse_args()
    payload = run_audit(
        data_dir=Path(args.data_dir),
        eval_start=date.fromisoformat(args.start),
        target=args.target,
    )
    print(
        json.dumps(
            {
                "meta": payload["meta"],
                "baselines": {
                    key: payload["baselines"][key]
                    for key in (
                        "exact_jackpot_rate",
                        "top3_prize_rate",
                        "top3_prize_one_in",
                        "top10_top4_top3_candidate_rate",
                    )
                },
                "single_ticket_99pct": payload["single_ticket_99pct"],
                "candidate_pool_99pct": {
                    "minimum_red_pool": payload["candidate_pool_99pct"]["minimum_red_pool"],
                    "minimum_red_pool_rate": payload["candidate_pool_99pct"]["minimum_red_pool_rate"],
                },
                "observed_ceiling": payload["observed_ceiling"],
                "models": [
                    {
                        "key": row["key"],
                        "n": row["n"],
                        "jackpot_count": row["jackpot_count"],
                        "top3_count": row["top3_count"],
                        "top3_rate": row["top3_rate"],
                        "candidate_count": row["top10_top4_candidate_count"],
                        "candidate_rate": row["top10_top4_candidate_rate"],
                    }
                    for row in payload["models"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
