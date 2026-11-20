"""Development-only nested walk-forward selection for future challengers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from .backtest import CaseResult, run_model, summarize
from .data import DEFAULT_DATA_DIR, load_draws
from .metrics import paired_bootstrap_difference
from .models import (
    EnsembleModel,
    FrequencyModel,
    GapModel,
    MarkovModel,
    Model,
    OnlineExpertModel,
    OnlineLogisticModel,
    RandomModel,
    RecencyModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "challenger_protocol.json"


@dataclass(frozen=True)
class DevelopmentFold:
    key: str
    validation_start: date
    validation_end: date
    eval_start: date
    eval_end: date


@dataclass(frozen=True)
class CandidateSpec:
    key: str
    label: str
    factory: Callable[[], Model]


DEVELOPMENT_FOLDS = (
    DevelopmentFold(
        "fold_2011_2012",
        date(2009, 1, 1),
        date(2010, 12, 31),
        date(2011, 1, 1),
        date(2012, 12, 31),
    ),
    DevelopmentFold(
        "fold_2013_2014",
        date(2011, 1, 1),
        date(2012, 12, 31),
        date(2013, 1, 1),
        date(2014, 12, 31),
    ),
    DevelopmentFold(
        "fold_2015_2016",
        date(2013, 1, 1),
        date(2014, 12, 31),
        date(2015, 1, 1),
        date(2016, 12, 31),
    ),
    DevelopmentFold(
        "fold_2017_2018",
        date(2015, 1, 1),
        date(2016, 12, 31),
        date(2017, 1, 1),
        date(2018, 12, 31),
    ),
    DevelopmentFold(
        "fold_2019_2020",
        date(2017, 1, 1),
        date(2018, 12, 31),
        date(2019, 1, 1),
        date(2020, 12, 31),
    ),
)


def candidate_registry() -> List[CandidateSpec]:
    return [
        CandidateSpec("random", "随机基线/拒绝建模", RandomModel),
        CandidateSpec("frequency", "历史频率", FrequencyModel),
        CandidateSpec("recency_13", "近因 13 期半衰期", lambda: RecencyModel(13.0, "recency_13")),
        CandidateSpec("recency_26", "近因 26 期半衰期", lambda: RecencyModel(26.0, "recency_26")),
        CandidateSpec("recency_52", "近因 52 期半衰期", lambda: RecencyModel(52.0, "recency_52")),
        CandidateSpec("recency_104", "近因 104 期半衰期", lambda: RecencyModel(104.0, "recency_104")),
        CandidateSpec("recency_208", "近因 208 期半衰期", lambda: RecencyModel(208.0, "recency_208")),
        CandidateSpec("gap", "遗漏回补", GapModel),
        CandidateSpec("markov", "一阶转移", MarkovModel),
        CandidateSpec("ensemble", "多因子集成", EnsembleModel),
        CandidateSpec(
            "hedge_005",
            "在线专家加权（学习率 0.05）",
            lambda: OnlineExpertModel(0.05, "hedge_005"),
        ),
        CandidateSpec(
            "hedge_01",
            "在线专家加权（学习率 0.10）",
            lambda: OnlineExpertModel(0.10, "hedge_01"),
        ),
        CandidateSpec("hedge", "在线专家加权（学习率 0.25）", OnlineExpertModel),
        CandidateSpec(
            "logistic_slow",
            "在线逻辑回归慢速",
            lambda: OnlineLogisticModel("logistic_slow", 0.01, 0.001),
        ),
        CandidateSpec(
            "logistic",
            "在线逻辑回归",
            lambda: OnlineLogisticModel("logistic", 0.03, 0.0005),
        ),
        CandidateSpec(
            "logistic_fast",
            "在线逻辑回归快速",
            lambda: OnlineLogisticModel("logistic_fast", 0.08, 0.0005),
        ),
    ]


def challenger_model_by_key(key: str) -> Model:
    for candidate in candidate_registry():
        if candidate.key == key:
            return candidate.factory()
    raise KeyError(key)


def _combined_loss(summary: Dict[str, Any]) -> float:
    return float(summary["logloss_red"]) + float(summary["logloss_blue"])


def _case_loss(case: CaseResult) -> float:
    return case.logloss_red + case.logloss_blue


def build_challenger_report(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path = DEFAULT_REPORT,
) -> Dict[str, Any]:
    draws = load_draws(data_dir / "ssq_history.csv")
    candidates = candidate_registry()
    fold_payloads: List[Dict[str, Any]] = []
    selected_cases: List[CaseResult] = []
    random_cases: List[CaseResult] = []
    selection_counts: Counter[str] = Counter()
    validation_losses: Dict[str, List[float]] = {candidate.key: [] for candidate in candidates}

    for fold in DEVELOPMENT_FOLDS:
        validation_summaries: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            cases, _ = run_model(
                candidate.factory(), draws, fold.validation_start, fold.validation_end
            )
            summary = summarize(cases)
            validation_summaries[candidate.key] = summary
            validation_losses[candidate.key].append(_combined_loss(summary))
        selected = min(
            candidates,
            key=lambda candidate: (
                _combined_loss(validation_summaries[candidate.key]),
                -float(validation_summaries[candidate.key]["mean_ndcg_at_10"]),
                candidate.key,
            ),
        )
        selection_counts[selected.key] += 1
        outer_cases, _ = run_model(
            selected.factory(), draws, fold.eval_start, fold.eval_end
        )
        baseline_cases, _ = run_model(
            RandomModel(), draws, fold.eval_start, fold.eval_end
        )
        selected_cases.extend(outer_cases)
        random_cases.extend(baseline_cases)
        fold_payloads.append(
            {
                "key": fold.key,
                "validation_start": fold.validation_start.isoformat(),
                "validation_end": fold.validation_end.isoformat(),
                "eval_start": fold.eval_start.isoformat(),
                "eval_end": fold.eval_end.isoformat(),
                "selected_key": selected.key,
                "selected_label": selected.label,
                "selection_metric": "logloss_red + logloss_blue; NDCG@10 tie-break",
                "validation": {
                    key: {
                        "n": summary["n"],
                        "combined_logloss": _combined_loss(summary),
                        "mean_ndcg_at_10": summary["mean_ndcg_at_10"],
                    }
                    for key, summary in validation_summaries.items()
                },
                "outer_selected": summarize(outer_cases),
                "outer_random": summarize(baseline_cases),
            }
        )

    max_selections = max(selection_counts.values())
    finalists = [key for key, count in selection_counts.items() if count == max_selections]
    selected_development_key = min(
        finalists,
        key=lambda key: (sum(validation_losses[key]) / len(validation_losses[key]), key),
    )
    candidate_labels = {candidate.key: candidate.label for candidate in candidates}
    ndcg_difference = paired_bootstrap_difference(
        [case.ndcg10 for case in selected_cases],
        [case.ndcg10 for case in random_cases],
    )
    loss_improvement = paired_bootstrap_difference(
        [_case_loss(case) for case in random_cases],
        [_case_loss(case) for case in selected_cases],
    )
    historical_gate = ndcg_difference[1] > 0.0 and loss_improvement[1] > 0.0
    ranked_exploratory = sorted(
        (candidate for candidate in candidates if candidate.key != "random"),
        key=lambda candidate: (
            -selection_counts[candidate.key],
            sum(validation_losses[candidate.key]) / len(validation_losses[candidate.key]),
            candidate.key,
        ),
    )
    promoted_candidate_key = (
        selected_development_key
        if historical_gate and selected_development_key != "random"
        else None
    )
    report = {
        "meta": {
            "protocol": "nested walk-forward development selection",
            "data_rows": len(draws),
            "development_data_end": "2020-12-31",
            "observed_2021_2026_used_for_selection": False,
            "no_lookahead": True,
            "strict_model_unchanged": True,
            "historical_results_cannot_promote": True,
        },
        "candidate_count": len(candidates),
        "folds": fold_payloads,
        "selection_counts": dict(sorted(selection_counts.items())),
        "future_candidate": {
            "key": promoted_candidate_key,
            "label": candidate_labels[promoted_candidate_key]
            if promoted_candidate_key
            else "无：开发集未通过双指标门槛",
            "status": "challenger" if promoted_candidate_key else "abstain",
            "best_development_key": selected_development_key,
            "selection_basis": (
                "开发折叠允许随机基线胜出；仅当外层 NDCG 与联合对数损失均显著改善时，"
                "才产生正式未来挑战者。"
            ),
            "fit_for_next_forecast": "超参数固定后，可用截至预测时点的全部既往数据在线拟合。",
        },
        "exploratory_panel": [
            {
                "key": candidate.key,
                "label": candidate.label,
                "role": "precommitted_future_evidence_only",
                "selection_count": selection_counts[candidate.key],
                "mean_validation_combined_logloss": (
                    sum(validation_losses[candidate.key])
                    / len(validation_losses[candidate.key])
                ),
            }
            for candidate in ranked_exploratory[:3]
        ],
        "outer_estimate": {
            "selected_n": len(selected_cases),
            "selected_summary": summarize(selected_cases),
            "random_summary": summarize(random_cases),
            "ndcg_difference_selected_minus_random": {
                "point": ndcg_difference[0],
                "ci95_low": ndcg_difference[1],
                "ci95_high": ndcg_difference[2],
            },
            "combined_logloss_improvement_random_minus_selected": {
                "point": loss_improvement[0],
                "ci95_low": loss_improvement[1],
                "ci95_high": loss_improvement[2],
            },
            "historical_dual_metric_gate_passed": historical_gate,
        },
        "promotion": {
            "action": (
                "freeze_formal_challenger"
                if promoted_candidate_key
                else "abstain_and_freeze_exploratory_panel"
            ),
            "minimum_future_frozen_draws": 104,
            "required_conditions": [
                "开奖前冻结文件连续且哈希验证通过",
                "相对严格基线的 NDCG@10 配对 bootstrap 95% 下界大于 0",
                "相对严格基线的红蓝联合对数损失改善 95% 下界大于 0",
                "预先固定的单注或组合票指标不得在开奖后更改",
            ],
            "promoted_now": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    report = build_challenger_report(Path(args.data_dir), Path(args.output))
    print(
        json.dumps(
            {
                "future_candidate": report["future_candidate"],
                "outer_estimate": report["outer_estimate"],
                "promotion": report["promotion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
