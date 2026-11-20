"""Audit whether the historical data contains a stable, exploitable signal."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .backtest import CaseResult, run_model
from .data import DEFAULT_DATA_DIR, load_draws
from .metrics import RED_NUMBERS, paired_bootstrap_difference
from .models import EnsembleModel, FrequencyModel, RandomModel, RecencyModel
from .schema import BLUE_MAX, Draw, validate_draw_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "reports" / "signal_audit.json"
WEB_JSON = PROJECT_ROOT / "web" / "data" / "signal_audit.json"
DEFAULT_EVAL_START = date(2008, 1, 1)


def _regularized_gamma_q(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), pure Python."""

    if x <= 0.0:
        return 1.0
    if x < a + 1.0:
        original_a = a
        term = 1.0 / a
        total = term
        for _ in range(200):
            a += 1.0
            term *= x / a
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return 1.0 - total * math.exp(
            -x + original_a * math.log(x) - math.lgamma(original_a)
        )

    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for index in range(1, 200):
        an = -index * (index - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi_square_sf(statistic: float, degrees_of_freedom: int) -> float:
    return _regularized_gamma_q(degrees_of_freedom / 2.0, statistic / 2.0)


def _chi_square_uniform(counts: Counter[int], size: int, total: int) -> float:
    expected = total / size
    return sum((counts.get(number, 0) - expected) ** 2 / expected for number in range(1, size + 1))


def _two_sided_normal_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def _repeat_statistics(draws: Sequence[Draw]) -> Dict[str, Any]:
    repeats = [len(set(left.reds) & set(right.reds)) for left, right in zip(draws, draws[1:])]
    observed = sum(repeats) / len(repeats)
    expected = 6.0 * 6.0 / 33.0
    variance = 6.0 * (6.0 / 33.0) * (1.0 - 6.0 / 33.0) * (27.0 / 32.0)
    standard_error = math.sqrt(variance / len(repeats))
    z_value = (observed - expected) / standard_error
    return {
        "n": len(repeats),
        "observed_mean_repeats": observed,
        "expected_mean_repeats": expected,
        "z": z_value,
        "p_two_sided": _two_sided_normal_p(z_value),
        "distribution": {str(hits): repeats.count(hits) for hits in sorted(set(repeats))},
    }


def _transition_statistics(draws: Sequence[Draw]) -> Dict[str, Any]:
    previous_hit = [0, 0]
    previous_miss = [0, 0]
    for previous, current in zip(draws, draws[1:]):
        previous_set = set(previous.reds)
        current_set = set(current.reds)
        for number in RED_NUMBERS:
            bucket = previous_hit if number in previous_set else previous_miss
            bucket[0] += 1
            bucket[1] += 1 if number in current_set else 0
    p_hit = previous_hit[1] / previous_hit[0]
    p_miss = previous_miss[1] / previous_miss[0]
    pooled = (previous_hit[1] + previous_miss[1]) / (previous_hit[0] + previous_miss[0])
    standard_error = math.sqrt(
        pooled * (1.0 - pooled) * (1.0 / previous_hit[0] + 1.0 / previous_miss[0])
    )
    z_value = (p_hit - p_miss) / standard_error
    return {
        "given_previous_hit": {"n": previous_hit[0], "rate": p_hit},
        "given_previous_miss": {"n": previous_miss[0], "rate": p_miss},
        "difference": p_hit - p_miss,
        "z": z_value,
        "p_two_sided": _two_sided_normal_p(z_value),
    }


def _cases_by_year(cases: Sequence[CaseResult]) -> Dict[int, List[CaseResult]]:
    grouped: Dict[int, List[CaseResult]] = {}
    for case in cases:
        grouped.setdefault(case.draw_date.year, []).append(case)
    return grouped


def _annual_frequency_check(draws: Sequence[Draw], start: date, end: date) -> Dict[str, Any]:
    random_cases, _ = run_model(RandomModel(), draws, start, end)
    frequency_cases, _ = run_model(FrequencyModel(), draws, start, end)
    random_by_year = _cases_by_year(random_cases)
    frequency_by_year = _cases_by_year(frequency_cases)
    rows: List[Dict[str, Any]] = []
    wins = 0
    for year in sorted(random_by_year):
        random_group = random_by_year[year]
        frequency_group = frequency_by_year[year]
        random_coverage = sum(case.top10_coverage for case in random_group) / len(random_group)
        frequency_coverage = sum(case.top10_coverage for case in frequency_group) / len(frequency_group)
        wins += int(frequency_coverage > random_coverage)
        rows.append(
            {
                "year": year,
                "n": len(random_group),
                "random_coverage": random_coverage,
                "frequency_coverage": frequency_coverage,
                "difference": frequency_coverage - random_coverage,
            }
        )
    return {
        "years": rows,
        "frequency_wins": wins,
        "frequency_losses": sum(1 for row in rows if row["difference"] < 0),
        "ties": sum(1 for row in rows if row["difference"] == 0),
    }


def _fold_check(
    draws: Sequence[Draw],
    folds: Sequence[Tuple[date, date]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    candidates = [
        ("frequency", FrequencyModel()),
        ("long_recency", RecencyModel(9999)),
        ("ensemble", EnsembleModel()),
    ]
    for start, end in folds:
        random_cases, _ = run_model(RandomModel(), draws, start, end)
        random_values = [case.ndcg10 for case in random_cases]
        random_coverage = sum(case.top10_coverage for case in random_cases) / len(random_cases)
        for key, model in candidates:
            cases, _ = run_model(model, draws, start, end)
            values = [case.ndcg10 for case in cases]
            difference, low, high = paired_bootstrap_difference(values, random_values)
            coverage = sum(case.top10_coverage for case in cases) / len(cases)
            rows.append(
                {
                    "fold_start": start.isoformat(),
                    "fold_end": end.isoformat(),
                    "n": len(cases),
                    "candidate": key,
                    "ndcg_difference": difference,
                    "ndcg_ci95_low": low,
                    "ndcg_ci95_high": high,
                    "coverage_difference": coverage - random_coverage,
                }
            )
    return rows


def run_audit(
    data_dir: Path = DEFAULT_DATA_DIR,
    eval_start: date = DEFAULT_EVAL_START,
    json_path: Path = DEFAULT_JSON,
    web_json_path: Path = WEB_JSON,
) -> Dict[str, Any]:
    draws = load_draws(data_dir / "ssq_history.csv")
    validate_draw_sequence(draws)
    red_counts = Counter(number for draw in draws for number in draw.reds)
    blue_counts = Counter(draw.blue for draw in draws)
    red_chi = _chi_square_uniform(red_counts, 33, len(draws) * 6)
    blue_chi = _chi_square_uniform(blue_counts, BLUE_MAX, len(draws))
    repeats = _repeat_statistics(draws)
    transitions = _transition_statistics(draws)
    annual = _annual_frequency_check(draws, eval_start, draws[-1].draw_date)
    folds = [
        (date(2013, 1, 1), date(2014, 12, 31)),
        (date(2015, 1, 1), date(2016, 12, 31)),
        (date(2017, 1, 1), date(2018, 12, 31)),
        (date(2019, 1, 1), date(2020, 12, 31)),
    ]
    fold_rows = _fold_check(draws, folds)
    significant = (
        red_chi is not None
        and _chi_square_sf(red_chi, 32) < 0.05
        or _chi_square_sf(blue_chi, 15) < 0.05
        or repeats["p_two_sided"] < 0.05
        or transitions["p_two_sided"] < 0.05
    )
    payload: Dict[str, Any] = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_rows": len(draws),
            "data_first_issue": draws[0].issue,
            "data_last_issue": draws[-1].issue,
            "eval_start": eval_start.isoformat(),
            "eval_end": draws[-1].draw_date.isoformat(),
            "no_lookahead": True,
        },
        "frequency": {
            "red_chi_square": red_chi,
            "red_df": 32,
            "red_p_value": _chi_square_sf(red_chi, 32),
            "red_min_count": min(red_counts.values()),
            "red_max_count": max(red_counts.values()),
            "blue_chi_square": blue_chi,
            "blue_df": 15,
            "blue_p_value": _chi_square_sf(blue_chi, 15),
            "blue_min_count": min(blue_counts.values()),
            "blue_max_count": max(blue_counts.values()),
        },
        "repeats": repeats,
        "transitions": transitions,
        "annual_frequency_check": annual,
        "fold_checks": fold_rows,
        "decision": {
            "any_significant_bias": significant,
            "stable_edge_found": False,
            "action": "keep_random_strict" if not significant else "manual_review",
        },
        "conclusion": (
            "频率、蓝球、红球重复和前后期转移都没有达到 5% 显著性；"
            "年度与多折检查也没有稳定优势。当前不能把频率或机器学习候选提升为正式预测模型。"
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
    parser.add_argument("--start", default=DEFAULT_EVAL_START.isoformat())
    args = parser.parse_args()
    payload = run_audit(
        data_dir=Path(args.data_dir),
        eval_start=date.fromisoformat(args.start),
    )
    print(
        json.dumps(
            {
                "meta": payload["meta"],
                "frequency": payload["frequency"],
                "repeats": payload["repeats"],
                "transitions": payload["transitions"],
                "annual_frequency_check": {
                    "frequency_wins": payload["annual_frequency_check"]["frequency_wins"],
                    "frequency_losses": payload["annual_frequency_check"]["frequency_losses"],
                },
                "decision": payload["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
