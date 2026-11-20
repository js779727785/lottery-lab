"""Aggregate only pre-draw frozen and post-draw settled future evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple

from .forecast import DEFAULT_FORECAST_DIR, verify_settlement
from .metrics import paired_bootstrap_difference


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "future_evidence.json"


def _mean(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scoring = [record for record in records if record["result"].get("scoring_available")]
    return {
        "n": len(records),
        "scoring_n": len(scoring),
        "avg_red_hits": _mean([float(record["result"]["red_hits"]) for record in records]),
        "blue_hit_rate": _mean(
            [1.0 if record["result"]["blue_hit"] else 0.0 for record in records]
        ),
        "any_prize_rate": _mean(
            [1.0 if record["result"]["tier"] is not None else 0.0 for record in records]
        ),
        "mean_ndcg_at_10": _mean(
            [float(record["result"]["ndcg_at_10"]) for record in scoring]
        ),
        "mean_combined_logloss": _mean(
            [float(record["result"]["combined_logloss"]) for record in scoring]
        ),
    }


def _comparison(
    candidate_records: List[Dict[str, Any]],
    strict_by_issue: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for candidate in candidate_records:
        strict = strict_by_issue.get(candidate["issue"])
        if (
            strict
            and candidate["result"].get("scoring_available")
            and strict["result"].get("scoring_available")
        ):
            matched.append((candidate, strict))
    if not matched:
        return {
            "matched_scoring_n": 0,
            "ndcg_difference_candidate_minus_strict": None,
            "combined_logloss_improvement_strict_minus_candidate": None,
        }
    ndcg = paired_bootstrap_difference(
        [float(candidate["result"]["ndcg_at_10"]) for candidate, _ in matched],
        [float(strict["result"]["ndcg_at_10"]) for _, strict in matched],
    )
    loss = paired_bootstrap_difference(
        [float(strict["result"]["combined_logloss"]) for _, strict in matched],
        [float(candidate["result"]["combined_logloss"]) for candidate, _ in matched],
    )
    return {
        "matched_scoring_n": len(matched),
        "ndcg_difference_candidate_minus_strict": {
            "point": ndcg[0],
            "ci95_low": ndcg[1],
            "ci95_high": ndcg[2],
        },
        "combined_logloss_improvement_strict_minus_candidate": {
            "point": loss[0],
            "ci95_low": loss[1],
            "ci95_high": loss[2],
        },
    }


def build_future_evidence_report(
    forecast_dir: Path = DEFAULT_FORECAST_DIR,
    output_path: Path = DEFAULT_REPORT,
    minimum_draws: int = 104,
) -> Dict[str, Any]:
    model_records: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    strict_records: List[Dict[str, Any]] = []
    settled_dir = forecast_dir / "settled"
    settlement_paths = sorted(settled_dir.glob("*.json"))
    for path in settlement_paths:
        verified = verify_settlement(path)
        settlement = verified["payload"]["settlement"]
        forecast = verified["forecast"]["forecast"]
        issue = settlement["target_issue"]
        strict_key = forecast["selection"]["strict_model_key"]
        strict_record = {
            "issue": issue,
            "role": "strict",
            "model_key": strict_key,
            "result": settlement["strict_result"],
        }
        strict_records.append(strict_record)
        model_records[f"strict:{strict_key}"].append(strict_record)

        challenger_key = forecast["selection"].get("challenger_model_key")
        if challenger_key and challenger_key != strict_key:
            model_records[challenger_key].append(
                {
                    "issue": issue,
                    "role": forecast["selection"].get(
                        "challenger_status", "legacy_challenger"
                    ),
                    "model_key": challenger_key,
                    "result": settlement["challenger_result"],
                }
            )
        for item in settlement.get("exploratory_results", []):
            model_records[item["key"]].append(
                {
                    "issue": issue,
                    "role": "exploratory",
                    "model_key": item["key"],
                    "result": item["result"],
                }
            )

    strict_by_issue = {record["issue"]: record for record in strict_records}
    models: Dict[str, Any] = {}
    eligible: List[str] = []
    for key, records in sorted(model_records.items()):
        comparison = None if key.startswith("strict:") else _comparison(records, strict_by_issue)
        models[key] = {
            "role": records[0]["role"],
            "summary": _summary(records),
            "vs_strict": comparison,
        }
        if comparison and comparison["matched_scoring_n"] >= minimum_draws:
            ndcg = comparison["ndcg_difference_candidate_minus_strict"]
            loss = comparison["combined_logloss_improvement_strict_minus_candidate"]
            if ndcg["ci95_low"] > 0.0 and loss["ci95_low"] > 0.0:
                eligible.append(key)

    report = {
        "meta": {
            "protocol": "future frozen forecasts only",
            "forecast_dir": str(forecast_dir),
            "settled_forecasts": len(settlement_paths),
            "minimum_draws": minimum_draws,
            "historical_replay_included": False,
            "tamper_checks_required": True,
        },
        "models": models,
        "promotion": {
            "eligible_for_manual_review": eligible,
            "promoted_automatically": False,
            "reason": (
                "即使达到预设门槛，也只进入人工复核；严格模型切换必须单独修改并留下新的开奖前冻结证据。"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dir", default=str(DEFAULT_FORECAST_DIR))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    parser.add_argument("--minimum-draws", type=int, default=104)
    args = parser.parse_args()
    report = build_future_evidence_report(
        Path(args.forecast_dir), Path(args.output), args.minimum_draws
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
