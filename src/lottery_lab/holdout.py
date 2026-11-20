"""Random-year time-out experiment with a fixed pre-test selection window."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .backtest import CaseResult, run_model, summarize
from .data import DEFAULT_DATA_DIR, load_draws
from .metrics import paired_bootstrap_difference, wilson_interval
from .models import model_registry
from .schema import Draw, validate_draw_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = PROJECT_ROOT / "reports" / "2015_holdout.json"
DEFAULT_CSV = PROJECT_ROOT / "reports" / "2015_random10.csv"
WEB_JSON = PROJECT_ROOT / "web" / "data" / "holdout2015.json"


def select_random_year(
    draws: Sequence[Draw], year: int, seed: int, count: int = 10
) -> List[str]:
    issues = [draw.issue for draw in draws if draw.draw_date.year == year]
    if len(issues) < count:
        raise RuntimeError(f"not enough {year} draws")
    selected = random.Random(seed).sample(issues, count)
    return sorted(selected)


def select_random_2015(draws: Sequence[Draw], seed: int, count: int = 10) -> List[str]:
    return select_random_year(draws, year=2015, seed=seed, count=count)


def _summary_for_issues(cases: Sequence[CaseResult], issues: set[str]) -> List[CaseResult]:
    return [case for case in cases if case.issue in issues]


def _compact_case(case: CaseResult) -> Dict[str, Any]:
    return {
        "issue": case.issue,
        "date": case.draw_date.isoformat(),
        "actual_reds": list(case.actual_reds),
        "actual_blue": case.actual_blue,
        "top10_reds": list(case.top10_reds),
        "top4_blues": list(case.top4_blues),
        "single_reds": list(case.red_ranking[:6]),
        "single_blue": case.blue_ranking[0],
        "top10_hits": case.top10_hits,
        "red_hits": case.red_hits,
        "blue_hit": case.blue_hit,
        "tier": case.tier,
        "any_prize": case.any_prize,
        "top10_coverage": case.top10_coverage,
        "top3_exact": case.tier is not None and case.tier <= 3,
        "top10_top4_candidate": _candidate_top3_hit(case),
        "ndcg10": case.ndcg10,
        "payout": case.payout,
    }


def _candidate_top3_hit(case: CaseResult) -> bool:
    return (
        len(set(case.top10_reds) & set(case.actual_reds)) >= 5
        and case.actual_blue in case.top4_blues
    )


def _with_confidence(
    summary: Dict[str, Any], cases: Optional[Sequence[CaseResult]] = None
) -> Dict[str, Any]:
    n = int(summary.get("n", 0))
    successes = int(summary.get("top10_coverage_count", 0))
    low, high = wilson_interval(successes, n)
    summary["top10_coverage_wilson_95"] = [low, high]
    summary["top10_coverage_lower_bound_95"] = low
    summary["meets_90pct_lower_bound"] = low >= 0.90
    if cases is not None:
        top3_exact_count = sum(1 for case in cases if case.tier is not None and case.tier <= 3)
        top10_top4_candidate_count = sum(1 for case in cases if _candidate_top3_hit(case))
        summary["top3_exact_count"] = top3_exact_count
        summary["top3_exact_rate"] = top3_exact_count / n if n else 0.0
        summary["top10_top4_candidate_count"] = top10_top4_candidate_count
        summary["top10_top4_candidate_rate"] = (
            top10_top4_candidate_count / n if n else 0.0
        )
        summary["top10_top4_candidate_wilson_95"] = list(
            wilson_interval(top10_top4_candidate_count, n)
        )
    return summary


def run_holdout(
    data_dir: Path = DEFAULT_DATA_DIR,
    seed: int = 20261120,
    count: int = 10,
    year: int = 2015,
    json_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    web_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if year < 2:
        raise ValueError("year must be at least 2")
    json_path = json_path or (
        DEFAULT_JSON if year == 2015 else PROJECT_ROOT / "reports" / f"{year}_holdout.json"
    )
    csv_path = csv_path or (
        DEFAULT_CSV if year == 2015 else PROJECT_ROOT / "reports" / f"{year}_random10.csv"
    )
    web_json_path = web_json_path or (
        WEB_JSON if year == 2015 else PROJECT_ROOT / "web" / "data" / f"holdout{year}.json"
    )
    draws = load_draws(data_dir / "ssq_history.csv")
    validate_draw_sequence(draws)
    selected = select_random_year(draws, year=year, seed=seed, count=count)

    validation_start = date(year - 2, 1, 1)
    validation_end = date(year - 1, 12, 31)
    test_start = date(year, 1, 1)
    test_end = date(year, 12, 31)

    validation_cases: Dict[str, List[CaseResult]] = {}
    validation_summaries: Dict[str, Dict[str, Any]] = {}
    for model in model_registry():
        cases, _ = run_model(model, draws, validation_start, validation_end)
        validation_cases[model.key] = cases
        validation_summaries[model.key] = summarize(cases)

    random_validation = validation_cases["random"]
    comparisons: Dict[str, Dict[str, Any]] = {}
    eligible: List[str] = []
    for model in model_registry():
        if model.key == "random":
            continue
        point, low, high = paired_bootstrap_difference(
            [case.ndcg10 for case in validation_cases[model.key]],
            [case.ndcg10 for case in random_validation],
        )
        comparisons[model.key] = {
            "metric": "mean_ndcg_at_10",
            "difference": point,
            "ci95_low": low,
            "ci95_high": high,
            "significant_improvement": low > 0.0,
        }
        if low > 0.0:
            eligible.append(model.key)

    if eligible:
        champion_key = max(
            eligible, key=lambda key: validation_summaries[key]["mean_ndcg_at_10"]
        )
        champion_reason = f"{year - 2}–{year - 1} 验证段显著优于随机基线。"
    else:
        champion_key = "random"
        champion_reason = (
            f"{year - 2}–{year - 1} 验证段没有任何候选模型显著优于随机基线；"
            "按同一门槛回退随机基线。"
        )

    test_cases: Dict[str, List[CaseResult]] = {}
    test_summaries: Dict[str, Dict[str, Any]] = {}
    for model in model_registry():
        cases, _ = run_model(model, draws, test_start, test_end)
        test_cases[model.key] = cases
        test_summaries[model.key] = summarize(cases)

    selected_set = set(selected)
    random10_summaries: Dict[str, Dict[str, Any]] = {}
    for key, cases in test_cases.items():
        random10_summaries[key] = _with_confidence(
            summarize(_summary_for_issues(cases, selected_set)),
            _summary_for_issues(cases, selected_set),
        )
    full_summaries = {
        key: _with_confidence(summary, test_cases[key])
        for key, summary in test_summaries.items()
    }

    by_issue: Dict[str, Dict[str, Any]] = {}
    for issue in selected:
        by_issue[issue] = {}
        for key in ("random", "ensemble", champion_key):
            case = next((item for item in test_cases[key] if item.issue == issue), None)
            if case is not None:
                by_issue[issue][key] = _compact_case(case)

    ensemble_cases = [case for case in test_cases["ensemble"] if case.issue in selected_set]
    random_cases = [case for case in test_cases["random"] if case.issue in selected_set]
    ensemble_diff = paired_bootstrap_difference(
        [case.ndcg10 for case in ensemble_cases],
        [case.ndcg10 for case in random_cases],
    )
    ensemble_coverage_diff = paired_bootstrap_difference(
        [1.0 if case.top10_coverage else 0.0 for case in ensemble_cases],
        [1.0 if case.top10_coverage else 0.0 for case in random_cases],
    )

    payload: Dict[str, Any] = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": f"{year} random {count}-draw time-out",
            "target_year": year,
            "seed": seed,
            "random_count": count,
            "selected_issues": selected,
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "champion_key": champion_key,
            "champion_label": next(
                model.label for model in model_registry() if model.key == champion_key
            ),
            "champion_reason": champion_reason,
            "data_rows": len(draws),
            "data_first_issue": draws[0].issue,
            "data_last_issue": draws[-1].issue,
            "no_lookahead": True,
            "history_replay_is_not_prediction": True,
            "selection_rule": (
                f"{year - 2}–{year - 1} NDCG@10 配对 bootstrap 95% 下界 > 0；"
                "否则回退随机"
            ),
        },
        "models_validation": {
            model.key: {
                "label": model.label,
                "summary": validation_summaries[model.key],
                "vs_random": comparisons.get(model.key),
                "selected": model.key == champion_key,
            }
            for model in model_registry()
        },
        "random10": {
            "summaries": random10_summaries,
            "cases": by_issue,
            "ensemble_vs_random_ndcg10": {
                "difference": ensemble_diff[0],
                "ci95_low": ensemble_diff[1],
                "ci95_high": ensemble_diff[2],
            },
            "ensemble_vs_random_coverage": {
                "difference": ensemble_coverage_diff[0],
                "ci95_low": ensemble_coverage_diff[1],
                "ci95_high": ensemble_coverage_diff[2],
            },
        },
        f"full_{year}": {
            "summaries": full_summaries,
            "n": len(test_cases["random"]),
        },
        "disclaimer": (
            "Top-10 覆盖率是候选池覆盖指标，不是单注中奖率。10 期样本的 95% 置信区间很宽；"
            "前三等奖精确命中与 Top-10 红球 + Top-4 蓝球候选也分别统计；"
            "即使 10/10 覆盖，也不能证明未来 90% 下限。"
        ),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    web_json_path.parent.mkdir(parents=True, exist_ok=True)
    web_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "issue",
                "date",
                "actual_reds",
                "actual_blue",
                "ensemble_top10",
                "ensemble_single",
                "ensemble_top10_hits",
                "ensemble_red_hits",
                "ensemble_blue_hit",
                "ensemble_tier",
                "ensemble_top3_exact",
                "ensemble_top10_coverage",
                "ensemble_top10_top4_candidate",
                "random_top10",
                "random_single",
                "random_top10_hits",
                "random_red_hits",
                "random_top10_coverage",
                "random_top3_exact",
                "random_top10_top4_candidate",
            ]
        )
        for issue in selected:
            item = by_issue[issue]
            ensemble = item.get("ensemble", {})
            baseline = item.get("random", {})
            writer.writerow(
                [
                    issue,
                    ensemble.get("date", ""),
                    " ".join(f"{n:02d}" for n in ensemble.get("actual_reds", [])),
                    f"{ensemble.get('actual_blue', 0):02d}",
                    " ".join(f"{n:02d}" for n in ensemble.get("top10_reds", [])),
                    " ".join(f"{n:02d}" for n in ensemble.get("single_reds", []))
                    + f" + {ensemble.get('single_blue', 0):02d}",
                    ensemble.get("top10_hits", ""),
                    ensemble.get("red_hits", ""),
                    int(bool(ensemble.get("blue_hit"))),
                    ensemble.get("tier", ""),
                    int(bool(ensemble.get("top3_exact"))),
                    int(bool(ensemble.get("top10_coverage"))),
                    int(bool(ensemble.get("top10_top4_candidate"))),
                    " ".join(f"{n:02d}" for n in baseline.get("top10_reds", [])),
                    " ".join(f"{n:02d}" for n in baseline.get("single_reds", []))
                    + f" + {baseline.get('single_blue', 0):02d}",
                    baseline.get("top10_hits", ""),
                    baseline.get("red_hits", ""),
                    int(bool(baseline.get("top10_coverage"))),
                    int(bool(baseline.get("top3_exact"))),
                    int(bool(baseline.get("top10_top4_candidate"))),
                ]
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    payload = run_holdout(
        data_dir=Path(args.data_dir),
        year=args.year,
        seed=args.seed,
        count=args.count,
    )
    print(
        json.dumps(
            {
                "meta": payload["meta"],
                "random10": {
                    key: payload["random10"]["summaries"][key]
                    for key in ("random", "ensemble", payload["meta"]["champion_key"])
                },
                f"full_{args.year}": {
                    key: payload[f"full_{args.year}"]["summaries"][key]
                    for key in ("random", "ensemble", payload["meta"]["champion_key"])
                },
                "selected_issues": payload["meta"]["selected_issues"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
