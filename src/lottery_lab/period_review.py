"""Strict walk-forward review for an arbitrary past draw period."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .backtest import CaseResult, run_model, summarize
from .data import DEFAULT_DATA_DIR, load_draws
from .metrics import paired_bootstrap_difference, wilson_interval
from .models import model_registry
from .schema import Draw, validate_draw_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_top3_hit(case: CaseResult) -> bool:
    return (
        len(set(case.top10_reds) & set(case.actual_reds)) >= 5
        and case.actual_blue in case.top4_blues
    )


def _summary(cases: Sequence[CaseResult]) -> Dict[str, Any]:
    summary = summarize(cases)
    n = len(cases)
    top3_count = sum(1 for case in cases if _candidate_top3_hit(case))
    top10_ci = wilson_interval(int(summary.get("top10_coverage_count", 0)), n)
    any_prize_ci = wilson_interval(int(summary.get("any_prize_count", 0)), n)
    top3_ci = wilson_interval(top3_count, n)
    return {
        "n": n,
        "jackpot_count": summary["exact_jackpot_count"],
        "jackpot_rate": summary["exact_jackpot_rate"],
        "top3_exact_count": sum(
            1 for case in cases if case.tier is not None and case.tier <= 3
        ),
        "any_prize_count": summary["any_prize_count"],
        "any_prize_rate": summary["any_prize_rate"],
        "any_prize_wilson_95": list(any_prize_ci),
        "top10_coverage_count": summary["top10_coverage_count"],
        "top10_coverage_rate": summary["top10_coverage_rate"],
        "top10_coverage_wilson_95": list(top10_ci),
        "top10_top4_candidate_count": top3_count,
        "top10_top4_candidate_rate": top3_count / n if n else 0.0,
        "top10_top4_candidate_wilson_95": list(top3_ci),
        "avg_red_hits": summary["avg_red_hits"],
        "avg_top10_hits": summary["avg_top10_hits"],
        "mean_ndcg_at_10": summary["mean_ndcg_at_10"],
        "tier_counts": summary["tier_counts"],
        "total_cost": summary["total_cost"],
        "total_payout": summary["total_payout"],
        "roi": summary["roi"],
    }


def _compact_case(
    case: CaseResult,
    draw_index: Dict[str, int],
    draws: Sequence[Draw],
) -> Dict[str, Any]:
    index = draw_index[case.issue]
    history_last_issue = draws[index - 1].issue if index > 0 else None
    return {
        "issue": case.issue,
        "date": case.draw_date.isoformat(),
        "history_rows": index,
        "history_last_issue": history_last_issue,
        "actual_reds": list(case.actual_reds),
        "actual_blue": case.actual_blue,
        "single_reds": list(case.red_ranking[:6]),
        "single_blue": case.blue_ranking[0],
        "top10_reds": list(case.top10_reds),
        "top4_blues": list(case.top4_blues),
        "top10_hits": case.top10_hits,
        "top10_coverage": case.top10_coverage,
        "red_hits": case.red_hits,
        "blue_hit": case.blue_hit,
        "tier": case.tier,
        "any_prize": case.any_prize,
        "top10_top4_candidate": _candidate_top3_hit(case),
        "payout": case.payout,
        "ndcg10": case.ndcg10,
    }


def _select_champion(
    draws: Sequence[Draw],
    validation_start: date,
    validation_end: date,
) -> Dict[str, Any]:
    validation_cases: Dict[str, List[CaseResult]] = {}
    validation_summaries: Dict[str, Dict[str, Any]] = {}
    for model in model_registry():
        cases, _ = run_model(model, draws, validation_start, validation_end)
        validation_cases[model.key] = cases
        validation_summaries[model.key] = _summary(cases)

    random_cases = validation_cases["random"]
    comparisons: Dict[str, Dict[str, Any]] = {}
    eligible: List[str] = []
    for model in model_registry():
        if model.key == "random":
            continue
        point, low, high = paired_bootstrap_difference(
            [case.ndcg10 for case in validation_cases[model.key]],
            [case.ndcg10 for case in random_cases],
        )
        significant = low > 0.0
        comparisons[model.key] = {
            "metric": "mean_ndcg_at_10",
            "difference": point,
            "ci95_low": low,
            "ci95_high": high,
            "significant_improvement": significant,
        }
        if significant:
            eligible.append(model.key)

    if eligible:
        champion_key = max(
            eligible,
            key=lambda key: validation_summaries[key]["mean_ndcg_at_10"],
        )
        reason = "验证窗存在显著优于随机基线的模型，选择验证 NDCG@10 最高者。"
    else:
        champion_key = "random"
        reason = "验证窗没有模型显著优于随机基线，按预设规则回退随机基线。"

    labels = {model.key: model.label for model in model_registry()}
    return {
        "champion_key": champion_key,
        "champion_label": labels[champion_key],
        "reason": reason,
        "comparisons": comparisons,
        "validation_summaries": validation_summaries,
    }


def run_period_review(
    start: date,
    end: date,
    validation_start: date,
    validation_end: date,
    data_dir: Path = DEFAULT_DATA_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> Dict[str, Any]:
    report_dir = Path(report_dir)
    if not report_dir.is_absolute():
        report_dir = (PROJECT_ROOT / report_dir).resolve()
    if start > end:
        raise ValueError("review start must not be after end")
    if validation_start > validation_end:
        raise ValueError("validation start must not be after validation end")
    if validation_end >= start:
        raise ValueError("validation window must end before the review window")

    csv_path = data_dir / "ssq_history.csv"
    draws = load_draws(csv_path)
    validate_draw_sequence(draws)
    if not any(draw.draw_date < start for draw in draws):
        raise RuntimeError("review requires a warm-up period before the review window")

    selection = _select_champion(draws, validation_start, validation_end)
    champion_key = selection["champion_key"]

    test_cases: Dict[str, List[CaseResult]] = {}
    test_summaries: Dict[str, Dict[str, Any]] = {}
    for model in model_registry():
        cases, _ = run_model(model, draws, start, end)
        test_cases[model.key] = cases
        test_summaries[model.key] = _summary(cases)

    draw_index = {draw.issue: index for index, draw in enumerate(draws)}
    case_by_issue = {
        key: {case.issue: case for case in cases}
        for key, cases in test_cases.items()
    }
    issues = [case.issue for case in test_cases["random"]]
    model_labels = {model.key: model.label for model in model_registry()}

    cases: List[Dict[str, Any]] = []
    for issue in issues:
        models = {
            key: _compact_case(case_by_issue[key][issue], draw_index, draws)
            for key in test_cases
        }
        actual = models[champion_key]
        cases.append(
            {
                "issue": issue,
                "date": actual["date"],
                "history_rows": actual["history_rows"],
                "history_last_issue": actual["history_last_issue"],
                "actual_reds": actual["actual_reds"],
                "actual_blue": actual["actual_blue"],
                "models": models,
            }
        )

    period_tag = (
        f"{start.year}_{start.month:02d}"
        if start.year == end.year and start.month == end.month
        else f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    )
    json_path = report_dir / f"period_{period_tag}.json"
    period_csv_path = report_dir / f"period_{period_tag}.csv"

    payload: Dict[str, Any] = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "strict walk-forward period review",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
            "data_rows": len(draws),
            "data_first_issue": draws[0].issue,
            "data_last_issue": draws[-1].issue,
            "csv_sha256": _sha256(csv_path),
            "review_draws": len(issues),
            "no_lookahead": True,
            "retrospective_replay": True,
            "selection_rule": "验证窗 NDCG@10 配对 bootstrap 95% 下界 > 0；否则回退随机",
            "champion_key": champion_key,
            "champion_label": selection["champion_label"],
            "champion_reason": selection["reason"],
        },
        "selection": selection,
        "summaries": test_summaries,
        "cases": cases,
        "comparisons": {
            "ensemble_vs_random": {
                "ndcg10": test_summaries["ensemble"]["mean_ndcg_at_10"]
                - test_summaries["random"]["mean_ndcg_at_10"],
                "top10_coverage": test_summaries["ensemble"]["top10_coverage_rate"]
                - test_summaries["random"]["top10_coverage_rate"],
                "any_prize": test_summaries["ensemble"]["any_prize_rate"]
                - test_summaries["random"]["any_prize_rate"],
            }
        },
        "disclaimer": (
            "这是事后执行的严格 walk-forward 回放，不是当时已发布并公证的预测。"
            "每期预测只使用该期之前的数据；Top-10 覆盖是候选池指标，不是单注准确率。"
        ),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with period_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "issue",
                "date",
                "history_last_issue",
                "actual_reds",
                "actual_blue",
                "strict_single",
                "strict_top10",
                "strict_red_hits",
                "strict_blue_hit",
                "strict_tier",
                "ensemble_single",
                "ensemble_top10",
                "ensemble_red_hits",
                "ensemble_blue_hit",
                "ensemble_top10_hits",
                "ensemble_tier",
                "ensemble_top10_top4_candidate",
            ]
        )
        for item in cases:
            strict = item["models"][champion_key]
            ensemble = item["models"]["ensemble"]
            writer.writerow(
                [
                    item["issue"],
                    item["date"],
                    item["history_last_issue"],
                    " ".join(f"{number:02d}" for number in item["actual_reds"]),
                    f"{item['actual_blue']:02d}",
                    " ".join(f"{number:02d}" for number in strict["single_reds"])
                    + f" + {strict['single_blue']:02d}",
                    " ".join(f"{number:02d}" for number in strict["top10_reds"]),
                    strict["red_hits"],
                    int(strict["blue_hit"]),
                    strict["tier"] or "",
                    " ".join(f"{number:02d}" for number in ensemble["single_reds"])
                    + f" + {ensemble['single_blue']:02d}",
                    " ".join(f"{number:02d}" for number in ensemble["top10_reds"]),
                    ensemble["red_hits"],
                    int(ensemble["blue_hit"]),
                    ensemble["top10_hits"],
                    ensemble["tier"] or "",
                    int(ensemble["top10_top4_candidate"]),
                ]
            )

    payload["artifacts"] = {
        "json": str(json_path.relative_to(PROJECT_ROOT)),
        "csv": str(period_csv_path.relative_to(PROJECT_ROOT)),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()
    payload = run_period_review(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        validation_start=date.fromisoformat(args.validation_start),
        validation_end=date.fromisoformat(args.validation_end),
        data_dir=Path(args.data_dir),
        report_dir=Path(args.report_dir),
    )
    champion = payload["meta"]["champion_key"]
    print(
        json.dumps(
            {
                "meta": payload["meta"],
                "strict": payload["summaries"][champion],
                "random": payload["summaries"]["random"],
                "ensemble": payload["summaries"]["ensemble"],
                "comparisons": payload["comparisons"],
                "artifacts": payload["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
