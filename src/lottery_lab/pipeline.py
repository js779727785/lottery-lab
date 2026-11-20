"""End-to-end: validate data, select a model without test leakage, and publish."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backtest import CaseResult, rolling_series, run_model, summarize, yearly_summary
from .challenger import DEFAULT_REPORT as DEFAULT_CHALLENGER_REPORT
from .challenger import challenger_model_by_key
from .data import DEFAULT_DATA_DIR, load_draws
from .metrics import paired_bootstrap_difference, theoretical_baselines
from .models import Model, Prediction, model_registry
from .portfolio import build_portfolio_payload
from .schema import Draw, validate_draw_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEB_DATA = PROJECT_ROOT / "web" / "data" / "dashboard.json"
DEFAULT_REPORT_SUMMARY = PROJECT_ROOT / "reports" / "backtest_summary.json"
DEFAULT_REPORT_CASES = PROJECT_ROOT / "reports" / "backtest_cases.csv"


def shift_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def next_issue_code(last_issue: str) -> str:
    year = int(last_issue[:4])
    number = int(last_issue[4:])
    if number >= 999:
        return f"{year + 1}001"
    return f"{year}{number + 1:03d}"


def _prediction_payload(prediction: Prediction, label: str) -> Dict[str, Any]:
    top10_reds = list(prediction.red_ranking[:10])
    top4_blues = list(prediction.blue_ranking[:4])
    single_reds = list(prediction.red_ranking[:6])
    single_blue = prediction.blue_ranking[0]
    return {
        "label": label,
        "top10_reds": top10_reds,
        "top4_blues": top4_blues,
        "single_ticket": {"reds": single_reds, "blue": single_blue},
        "red_ranking": list(prediction.red_ranking),
        "blue_ranking": list(prediction.blue_ranking),
        "red_probabilities": [
            round(float(prediction.red_probs[number]), 12) for number in range(1, 34)
        ],
        "blue_probabilities": [
            round(float(prediction.blue_probs[number]), 12) for number in range(1, 17)
        ],
        "top10_red_probs": [
            {"number": number, "probability": round(float(prediction.red_probs[number]), 6)}
            for number in top10_reds
        ],
        "top4_blue_probs": [
            {"number": number, "probability": round(float(prediction.blue_probs[number]), 6)}
            for number in top4_blues
        ],
    }


def _compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "n",
        "exact_jackpot_count",
        "exact_jackpot_rate",
        "any_prize_count",
        "any_prize_rate",
        "any_prize_wilson_95",
        "top10_coverage_count",
        "top10_coverage_rate",
        "top10_coverage_wilson_95",
        "avg_top10_hits",
        "avg_red_hits",
        "blue_hit_rate",
        "mean_ndcg_at_10",
        "mean_actual_rank",
        "brier_red",
        "brier_blue",
        "logloss_red",
        "logloss_blue",
        "tier_counts",
        "red_hits_distribution",
        "total_cost",
        "total_payout",
        "net",
        "roi",
    ]
    return {key: summary[key] for key in keys}


def _write_cases(path: Path, cases: Sequence[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "issue",
            "date",
            "actual_reds",
            "actual_blue",
            "top10_reds",
            "top4_blues",
            "top10_hits",
            "red_hits",
            "blue_hit",
            "tier",
            "payout",
            "ndcg10",
            "mean_rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "issue": case.issue,
                    "date": case.draw_date.isoformat(),
                    "actual_reds": " ".join(f"{number:02d}" for number in case.actual_reds),
                    "actual_blue": f"{case.actual_blue:02d}",
                    "top10_reds": " ".join(f"{number:02d}" for number in case.top10_reds),
                    "top4_blues": " ".join(f"{number:02d}" for number in case.top4_blues),
                    "top10_hits": case.top10_hits,
                    "red_hits": case.red_hits,
                    "blue_hit": int(case.blue_hit),
                    "tier": case.tier or "",
                    "payout": case.payout,
                    "ndcg10": f"{case.ndcg10:.6f}",
                    "mean_rank": f"{case.mean_rank:.6f}",
                }
            )


def _paired_comparison(
    candidate_cases: Sequence[CaseResult],
    random_cases: Sequence[CaseResult],
) -> Dict[str, Any]:
    ndcg_point, ndcg_low, ndcg_high = paired_bootstrap_difference(
        [case.ndcg10 for case in candidate_cases],
        [case.ndcg10 for case in random_cases],
    )
    loss_point, loss_low, loss_high = paired_bootstrap_difference(
        [case.logloss_red + case.logloss_blue for case in random_cases],
        [case.logloss_red + case.logloss_blue for case in candidate_cases],
    )
    return {
        "metric": "mean_ndcg_at_10_and_red_blue_joint_logloss",
        "difference": ndcg_point,
        "ci95_low": ndcg_low,
        "ci95_high": ndcg_high,
        "joint_logloss": {
            "metric": "red_blue_joint_logloss_improvement",
            "difference": loss_point,
            "ci95_low": loss_low,
            "ci95_high": loss_high,
            "significant_improvement": loss_low > 0.0,
        },
        "significant_improvement": ndcg_low > 0.0 and loss_low > 0.0,
    }


def build_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    validation_years: int = 2,
    test_years: int = 5,
    portfolio_ticket_count: int = 10,
    web_data_path: Path = DEFAULT_WEB_DATA,
    report_summary_path: Path = DEFAULT_REPORT_SUMMARY,
    report_cases_path: Path = DEFAULT_REPORT_CASES,
) -> Dict[str, Any]:
    draws = load_draws(data_dir / "ssq_history.csv")
    validate_draw_sequence(draws)
    if len(draws) < 500:
        raise RuntimeError("not enough historical draws for a meaningful backtest")

    last_date = draws[-1].draw_date
    test_start = shift_years(last_date, -test_years) + timedelta(days=1)
    validation_end = test_start - timedelta(days=1)
    validation_start = shift_years(validation_end, -validation_years) + timedelta(days=1)

    validation_cases: Dict[str, List[CaseResult]] = {}
    validation_summaries: Dict[str, Dict[str, Any]] = {}
    for model in model_registry():
        cases, _ = run_model(model, draws, validation_start, validation_end)
        validation_cases[model.key] = cases
        validation_summaries[model.key] = summarize(cases)

    random_validation = validation_cases["random"]
    validation_comparisons: Dict[str, Dict[str, Any]] = {}
    eligible: List[str] = []
    for model in model_registry():
        if model.key == "random":
            continue
        comparison = _paired_comparison(
            validation_cases[model.key], random_validation
        )
        validation_comparisons[model.key] = comparison
        if comparison["significant_improvement"]:
            eligible.append(model.key)

    if eligible:
        champion_key = max(
            eligible, key=lambda key: validation_summaries[key]["mean_ndcg_at_10"]
        )
        champion_reason = (
            "验证段 NDCG@10 与红蓝联合对数损失改善的配对 bootstrap 95% 下界"
            "均大于 0，表现优于随机基线。"
        )
    else:
        champion_key = "random"
        champion_reason = (
            "没有任何候选模型在验证段同时显著改善 NDCG@10 和红蓝联合对数损失；"
            "按预先设定的门槛回退为随机基线，避免把噪声当作预测能力。"
        )

    test_cases: Dict[str, List[CaseResult]] = {}
    test_summaries: Dict[str, Dict[str, Any]] = {}
    test_models: Dict[str, Model] = {}
    random_test: Optional[List[CaseResult]] = None
    for model in model_registry():
        cases, fitted = run_model(model, draws, test_start, last_date)
        test_cases[model.key] = cases
        test_summaries[model.key] = summarize(cases)
        test_models[model.key] = fitted
        if model.key == "random":
            random_test = cases

    assert random_test is not None
    test_comparisons: Dict[str, Dict[str, Any]] = {}
    for model in model_registry():
        if model.key == "random":
            continue
        test_comparisons[model.key] = _paired_comparison(
            test_cases[model.key], random_test
        )

    champion_cases = test_cases[champion_key]
    experimental_cases = test_cases["ensemble"]
    champion_model = test_models[champion_key]
    next_issue = next_issue_code(draws[-1].issue)
    strict_prediction = champion_model.predict(next_issue)
    latest_prediction = _prediction_payload(strict_prediction, f"{next_issue} · 严格模式")
    portfolio = build_portfolio_payload(
        strict_prediction, next_issue, ticket_count=portfolio_ticket_count
    )

    experimental = model_registry()
    experimental_model = next(model for model in experimental if model.key == "ensemble")
    for draw in draws:
        experimental_model.observe(draw)
    experimental_prediction = _prediction_payload(
        experimental_model.predict(next_issue), f"{next_issue} · 实验性多因子候选"
    )
    development_challenger = None
    exploratory_challengers = []
    if DEFAULT_CHALLENGER_REPORT.exists():
        challenger_report = json.loads(DEFAULT_CHALLENGER_REPORT.read_text(encoding="utf-8"))
        challenger_key = challenger_report["future_candidate"]["key"]
        if challenger_key:
            challenger_model = challenger_model_by_key(challenger_key)
            for draw in draws:
                challenger_model.observe(draw)
            challenger_prediction = _prediction_payload(
                challenger_model.predict(next_issue),
                f"{next_issue} · 开发集冻结挑战者",
            )
            development_challenger = {
                "key": challenger_key,
                "label": challenger_report["future_candidate"]["label"],
                "selection_data_end": challenger_report["meta"]["development_data_end"],
                "observed_2021_2026_used_for_selection": False,
                "promotion": challenger_report["promotion"],
                "latest_prediction": challenger_prediction,
            }
        for panel_item in challenger_report.get("exploratory_panel", []):
            panel_model = challenger_model_by_key(panel_item["key"])
            for draw in draws:
                panel_model.observe(draw)
            exploratory_challengers.append(
                {
                    **panel_item,
                    "selection_data_end": challenger_report["meta"]["development_data_end"],
                    "latest_prediction": _prediction_payload(
                        panel_model.predict(next_issue),
                        f"{next_issue} · 预注册探索模型 {panel_item['key']}",
                    ),
                }
            )

    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    quality = manifest.get("quality", {})

    models_payload: List[Dict[str, Any]] = []
    for model in model_registry():
        models_payload.append(
            {
                "key": model.key,
                "label": model.label,
                "selected": model.key == champion_key,
                "validation": _compact_summary(validation_summaries[model.key]),
                "test": _compact_summary(test_summaries[model.key]),
                "validation_vs_random": validation_comparisons.get(model.key),
                "test_vs_random": test_comparisons.get(model.key),
            }
        )

    dashboard: Dict[str, Any] = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": manifest.get("source", "https://www.cwl.gov.cn/"),
            "source_page": manifest.get("source_page"),
            "fetched_at_utc": manifest.get("fetched_at_utc"),
            "rows": len(draws),
            "first_issue": draws[0].issue,
            "first_date": draws[0].draw_date.isoformat(),
            "last_issue": draws[-1].issue,
            "last_date": draws[-1].draw_date.isoformat(),
            "csv_sha256": manifest.get("csv", {}).get("sha256"),
            "raw_sha256": manifest.get("raw_records", {}).get("sha256"),
            "pagination_crosscheck": manifest.get("pagination_crosscheck"),
            "quality": quality,
            "test_start": test_start.isoformat(),
            "test_end": last_date.isoformat(),
            "test_n": len(champion_cases),
            "validation_start": validation_start.isoformat(),
            "validation_end": validation_end.isoformat(),
            "validation_n": len(random_validation),
            "selection_rule": (
                "验证段 NDCG@10 与红蓝联合对数损失改善的配对 bootstrap 95% 下界"
                "均 > 0；否则回退随机基线"
            ),
            "champion_key": champion_key,
            "champion_label": next(model.label for model in model_registry() if model.key == champion_key),
            "champion_reason": champion_reason,
            "next_issue": next_issue,
            "no_lookahead": True,
        },
        "baseline": theoretical_baselines(),
        "champion": {
            "key": champion_key,
            "label": next(model.label for model in model_registry() if model.key == champion_key),
            "summary": _compact_summary(test_summaries[champion_key]),
            "validation_summary": _compact_summary(validation_summaries[champion_key]),
            "latest_prediction": latest_prediction,
        },
        "experimental": {
            "key": "ensemble",
            "label": "多因子集成（实验性）",
            "latest_prediction": experimental_prediction,
            "test_summary": _compact_summary(test_summaries["ensemble"]),
            "cases": [case.as_dict() for case in experimental_cases],
            "rolling": rolling_series(experimental_cases, window=50),
            "yearly": yearly_summary(experimental_cases),
        },
        "development_challenger": development_challenger,
        "exploratory_challengers": exploratory_challengers,
        "portfolio": portfolio,
        "models": models_payload,
        "cases": [case.as_dict() for case in champion_cases],
        "rolling": rolling_series(champion_cases, window=50),
        "yearly": yearly_summary(champion_cases),
        "disclaimer": (
            "本系统是概率与回测游戏，不构成投注建议。公平双色球每期近似独立随机；"
            "Top-10 红球覆盖率约 90.9% 是候选池覆盖属性，不是单注中奖率，也不能推出未来预测能力。"
        ),
    }

    web_data_path.parent.mkdir(parents=True, exist_ok=True)
    web_data_path.write_text(
        json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_summary = {
        "meta": dashboard["meta"],
        "baseline": dashboard["baseline"],
        "champion": dashboard["champion"],
        "experimental": {
            "key": dashboard["experimental"]["key"],
            "label": dashboard["experimental"]["label"],
            "latest_prediction": dashboard["experimental"]["latest_prediction"],
            "test_summary": dashboard["experimental"]["test_summary"],
            "yearly": dashboard["experimental"]["yearly"],
        },
        "development_challenger": dashboard["development_challenger"],
        "exploratory_challengers": dashboard["exploratory_challengers"],
        "portfolio": dashboard["portfolio"],
        "models": dashboard["models"],
        "yearly": dashboard["yearly"],
        "disclaimer": dashboard["disclaimer"],
    }
    report_summary_path.write_text(
        json.dumps(report_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_cases(report_cases_path, champion_cases)
    shutil.copyfile(data_dir / "ssq_history.csv", web_data_path.parent / "ssq_history.csv")
    return dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--test-years", type=int, default=5)
    parser.add_argument("--portfolio-tickets", type=int, default=10)
    parser.add_argument("--web-data", default=str(DEFAULT_WEB_DATA))
    parser.add_argument("--report-summary", default=str(DEFAULT_REPORT_SUMMARY))
    parser.add_argument("--report-cases", default=str(DEFAULT_REPORT_CASES))
    args = parser.parse_args()
    dashboard = build_pipeline(
        data_dir=Path(args.data_dir),
        validation_years=args.validation_years,
        test_years=args.test_years,
        portfolio_ticket_count=args.portfolio_tickets,
        web_data_path=Path(args.web_data),
        report_summary_path=Path(args.report_summary),
        report_cases_path=Path(args.report_cases),
    )
    print(
        json.dumps(
            {
                "meta": dashboard["meta"],
                "champion": {
                    "key": dashboard["champion"]["key"],
                    "summary": dashboard["champion"]["summary"],
                },
                "experimental": dashboard["experimental"]["test_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
