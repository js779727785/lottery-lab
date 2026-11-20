from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from lottery_lab.backtest import run_model
from lottery_lab.challenger import DEVELOPMENT_FOLDS, candidate_registry
from lottery_lab.data import DEFAULT_DATA_DIR, build_provenance, load_draws
from lottery_lab.forecast import (
    ForecastIntegrityError,
    freeze_forecast,
    settle_forecast,
    verify_forecast,
)
from lottery_lab.future_audit import build_future_evidence_report
from lottery_lab.holdout import select_random_year
from lottery_lab.metrics import (
    TOTAL_COMBINATIONS,
    ndcg_at_k,
    paired_bootstrap_difference,
    theoretical_baselines,
    wilson_interval,
)
from lottery_lab.models import Model, OnlineExpertModel, OnlineLogisticModel, Prediction, RandomModel
from lottery_lab.schema import Draw, prize_tier, validate_draw_sequence


ROOT = Path(__file__).resolve().parents[1]


class TestSchema(unittest.TestCase):
    def test_prize_tiers(self) -> None:
        self.assertEqual(prize_tier(6, True), 1)
        self.assertEqual(prize_tier(6, False), 2)
        self.assertEqual(prize_tier(5, True), 3)
        self.assertEqual(prize_tier(5, False), 4)
        self.assertEqual(prize_tier(4, True), 4)
        self.assertEqual(prize_tier(4, False), 5)
        self.assertEqual(prize_tier(3, True), 5)
        self.assertEqual(prize_tier(2, True), 6)
        self.assertEqual(prize_tier(1, True), 6)
        self.assertEqual(prize_tier(0, True), 6)
        self.assertIsNone(prize_tier(3, False))
        self.assertIsNone(prize_tier(0, False))


class TestMetrics(unittest.TestCase):
    def test_exact_baselines(self) -> None:
        baseline = theoretical_baselines()
        self.assertEqual(baseline["total_combinations"], 17_721_088)
        self.assertAlmostEqual(baseline["top3_prize_rate"], 178 / 17_721_088, places=14)
        self.assertAlmostEqual(baseline["top3_prize_one_in"], 99_556.67415730337, places=6)
        self.assertAlmostEqual(baseline["top10_red_coverage_rate"], 0.9088570634, places=9)
        self.assertAlmostEqual(baseline["top10_top4_top3_candidate_rate"], 0.0013556730, places=10)
        self.assertAlmostEqual(baseline["any_prize_rate"], 0.0670945260, places=9)
        self.assertAlmostEqual(baseline["expected_red_hits"], 1.0909090909, places=9)

    def test_ndcg(self) -> None:
        relevant = {1, 2, 3, 4, 5, 6}
        perfect = tuple(range(1, 34))
        self.assertAlmostEqual(ndcg_at_k(perfect, relevant, 10), 1.0)
        self.assertAlmostEqual(ndcg_at_k(tuple(range(7, 34)) + tuple(range(1, 7)), relevant, 10), 0.0)

    def test_wilson_and_bootstrap(self) -> None:
        low, high = wilson_interval(900, 1000)
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)
        point, low, high = paired_bootstrap_difference([1.0] * 100, [0.0] * 100, n_boot=200)
        self.assertAlmostEqual(point, 1.0)
        self.assertGreater(low, 0.99)
        self.assertAlmostEqual(high, 1.0)


class TestOnlineLogisticModel(unittest.TestCase):
    def test_gradient_step_reduces_binary_log_loss(self) -> None:
        model = OnlineLogisticModel(learning_rate=0.1, l2=0.0)
        weights = [0.0]
        features = {1: (1.0,), 2: (-1.0,)}
        positives = {1}

        def loss() -> float:
            total = 0.0
            for number, vector in features.items():
                score = sum(weight * value for weight, value in zip(weights, vector))
                probability = 1.0 / (1.0 + math.exp(-score))
                label = 1.0 if number in positives else 0.0
                total -= label * math.log(probability) + (1.0 - label) * math.log(
                    1.0 - probability
                )
            return total / len(features)

        before = loss()
        model._update_weights(weights, features, positives, model.learning_rate)

        self.assertLess(loss(), before)
        self.assertGreater(weights[0], 0.0)


class TestOnlineExpertModel(unittest.TestCase):
    def test_mixes_only_past_experts_and_updates_after_observe(self) -> None:
        model = OnlineExpertModel()
        prediction = model.predict("2026001")

        self.assertEqual(len(model.experts), 5)
        self.assertAlmostEqual(sum(prediction.red_probs.values()), 6.0, places=10)
        self.assertAlmostEqual(sum(prediction.blue_probs.values()), 1.0, places=10)

        model.observe(
            Draw("2026001", date(2026, 1, 1), (1, 2, 3, 4, 5, 6), 1)
        )
        before = tuple(model.expert_weights)
        model.predict("2026002")
        model.observe(
            Draw("2026002", date(2026, 1, 3), (28, 29, 30, 31, 32, 33), 16)
        )

        self.assertAlmostEqual(sum(model.expert_weights), 1.0, places=10)
        self.assertNotEqual(tuple(model.expert_weights), before)
        self.assertIsNone(model._pending)


class TestCoveragePortfolio(unittest.TestCase):
    def test_portfolio_is_unique_deterministic_and_exactly_scored(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("lottery_lab.portfolio"))
        from lottery_lab.portfolio import evaluate_portfolio, generate_coverage_portfolio

        prediction = RandomModel().predict("2026134")
        tickets = generate_coverage_portfolio(prediction, "2026134", ticket_count=10)
        repeated = generate_coverage_portfolio(prediction, "2026134", ticket_count=10)
        metrics = evaluate_portfolio(tickets)

        self.assertEqual(tickets, repeated)
        self.assertEqual(len(tickets), 10)
        self.assertEqual(len(set(tickets)), 10)
        self.assertEqual(len({ticket.blue for ticket in tickets}), 10)
        self.assertEqual(metrics["unique_ticket_count"], 10)
        self.assertAlmostEqual(
            metrics["jackpot_probability"], 10 / TOTAL_COMBINATIONS, places=15
        )
        self.assertGreater(
            metrics["any_prize_probability"],
            metrics["random_independent_baseline"]["any_prize_probability"],
        )

    def test_sixteen_distinct_blues_guarantee_one_prize(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("lottery_lab.portfolio"))
        from lottery_lab.portfolio import evaluate_portfolio, generate_coverage_portfolio

        prediction = RandomModel().predict("2026134")
        tickets = generate_coverage_portfolio(prediction, "2026134", ticket_count=16)
        metrics = evaluate_portfolio(tickets)

        self.assertEqual(len({ticket.blue for ticket in tickets}), 16)
        self.assertEqual(metrics["any_prize_probability"], 1.0)


class TestCampaignPlanner(unittest.TestCase):
    def test_single_draw_requirements_match_extreme_audit(self) -> None:
        from lottery_lab.campaign import required_tickets_per_draw

        self.assertEqual(
            required_tickets_per_draw(1, "top3", target_probability=0.99),
            452_594,
        )
        self.assertEqual(
            required_tickets_per_draw(1, "jackpot", target_probability=0.99),
            17_543_878,
        )

    def test_campaign_probability_compounds_only_across_draws(self) -> None:
        from lottery_lab.campaign import campaign_probability, per_draw_probability

        per_draw = per_draw_probability(10, "jackpot")
        campaign = campaign_probability(52, 10, "jackpot")

        self.assertAlmostEqual(per_draw, 10 / TOTAL_COMBINATIONS, places=15)
        self.assertAlmostEqual(campaign, 1.0 - (1.0 - per_draw) ** 52, delta=1e-15)
        self.assertGreater(campaign, per_draw)
        self.assertLess(campaign, 0.001)

    def test_budget_and_expected_loss_are_labeled_as_coverage(self) -> None:
        from lottery_lab.campaign import build_campaign_plan

        plan = build_campaign_plan(
            draw_count=100,
            tickets_per_draw=10,
            prize_target="top3",
            target_probability=0.99,
            budget_yuan=1_500,
        )

        self.assertTrue(plan["coverage_only"])
        self.assertFalse(plan["prediction_accuracy_claim"])
        self.assertEqual(plan["tickets_per_draw"], 7)
        self.assertEqual(plan["total_ticket_count"], 700)
        self.assertEqual(plan["cost_yuan"], 1_400)
        self.assertEqual(plan["unused_budget_yuan"], 100)
        self.assertGreater(plan["expected_fixed_tier_payout_yuan"], 0.0)
        self.assertGreater(plan["expected_loss_upper_bound_yuan"], 0.0)
        self.assertEqual(plan["variable_prize_tiers_omitted"], [1, 2])

    def test_default_report_contains_requested_draw_horizons(self) -> None:
        from lottery_lab.campaign import build_campaign_report

        report = build_campaign_report(draw_horizons=(10, 52, 100, 500, 1000))

        self.assertEqual(
            [row["draw_count"] for row in report["requirements"]["top3"]],
            [10, 52, 100, 500, 1000],
        )
        self.assertEqual(
            [row["draw_count"] for row in report["requirements"]["jackpot"]],
            [10, 52, 100, 500, 1000],
        )
        for target in ("top3", "jackpot"):
            for row in report["requirements"][target]:
                self.assertGreaterEqual(row["campaign_probability"], 0.99)
                self.assertLess(row["previous_ticket_probability"], 0.99)
                self.assertEqual(
                    row["total_cost_yuan"],
                    row["draw_count"] * row["tickets_per_draw"] * 2,
                )


class TestOfficialData(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.draws = load_draws(DEFAULT_DATA_DIR / "ssq_history.csv")
        cls.manifest = json.loads((DEFAULT_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))

    def test_count_and_range(self) -> None:
        self.assertEqual(len(self.draws), 3166)
        self.assertEqual(self.draws[0].issue, "2006001")
        self.assertEqual(self.draws[0].draw_date, date(2006, 1, 1))
        self.assertEqual(self.draws[-1].issue, "2026133")
        self.assertEqual(self.draws[-1].draw_date, date(2026, 11, 19))

    def test_sample_issue(self) -> None:
        old_draw = next(item for item in self.draws if item.issue == "2006001")
        self.assertEqual(old_draw.reds, (1, 12, 15, 19, 21, 28))
        self.assertEqual(old_draw.blue, 3)
        self.assertEqual(old_draw.first_prize_count, 2)
        self.assertEqual(old_draw.first_prize_amount, 5_000_000)

        draw = next(item for item in self.draws if item.issue == "2018001")
        self.assertEqual(draw.reds, (1, 8, 11, 26, 28, 31))
        self.assertEqual(draw.blue, 4)
        self.assertEqual(draw.first_prize_count, 22)
        self.assertEqual(draw.first_prize_amount, 5_825_255)

    def test_no_missing_or_duplicate_issue(self) -> None:
        quality = validate_draw_sequence(self.draws)
        self.assertTrue(quality["ok"])
        self.assertEqual(quality["warnings"], [])
        self.assertEqual(quality["years"], list(range(2006, 2027)))

    def test_csv_hash_matches_manifest(self) -> None:
        path = DEFAULT_DATA_DIR / "ssq_history.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, self.manifest["csv"]["sha256"])
        self.assertEqual(self.manifest["csv"]["rows"], 3166)
        self.assertTrue(self.manifest["pagination_crosscheck"]["winning_numbers_match"])
        self.assertEqual(self.manifest["overlap_crosscheck"]["overlap_issues"], 2091)
        self.assertTrue(self.manifest["overlap_crosscheck"]["first_prize_match_where_cwl_available"])

    def test_every_draw_has_hashed_source_provenance(self) -> None:
        provenance_path = DEFAULT_DATA_DIR / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        records = provenance["records"]
        self.assertEqual(len(records), len(self.draws))
        self.assertEqual({record["issue"] for record in records}, {draw.issue for draw in self.draws})
        self.assertEqual(provenance["summary"]["dual_source_verified"], 2091)
        self.assertEqual(provenance["summary"]["single_source_structurally_validated"], 1075)
        self.assertEqual(self.manifest["provenance"]["sha256"], hashlib.sha256(provenance_path.read_bytes()).hexdigest())
        for record in records:
            self.assertEqual(len(record["canonical_draw_sha256"]), 64)
            self.assertTrue(record["sources"])
            for source in record["sources"]:
                self.assertEqual(len(source["raw_record_sha256"]), 64)
                self.assertTrue(source["numbers_match_canonical"])

    def test_selected_2015_issues_match_cwl_raw(self) -> None:
        holdout = json.loads((ROOT / "reports" / "2015_holdout.json").read_text(encoding="utf-8"))
        selected = holdout["meta"]["selected_issues"]
        self.assertEqual(len(selected), 10)
        self.assertEqual(len(set(selected)), 10)

        draw_map = {draw.issue: draw for draw in self.draws}
        cwl_records = json.loads(
            (DEFAULT_DATA_DIR / "raw" / "cwl_records.json").read_text(encoding="utf-8")
        )["records"]
        cwl_map = {record["code"]: record for record in cwl_records}

        for issue in selected:
            draw = draw_map[issue]
            official = cwl_map[issue]
            self.assertEqual(draw.draw_date.year, 2015)
            self.assertEqual(draw.reds, tuple(int(number) for number in official["red"].split(",")))
            self.assertEqual(draw.blue, int(official["blue"]))


class TestProvenanceBuilder(unittest.TestCase):
    def test_dual_and_single_source_records_are_labeled_without_overclaiming(self) -> None:
        zhcw_records = [
            {
                "issue": "2026001",
                "openTime": "2026-01-01",
                "frontWinningNum": "01 02 03 04 05 06",
                "backWinningNum": "07",
                "winnerDetails": [],
            },
            {
                "issue": "2026002",
                "openTime": "2026-01-04",
                "frontWinningNum": "02 03 04 05 06 07",
                "backWinningNum": "08",
                "winnerDetails": [],
            },
        ]
        cwl_records = [
            {
                "code": "2026002",
                "date": "2026-01-04(日)",
                "red": "02,03,04,05,06,07",
                "blue": "08",
                "prizegrades": [],
            }
        ]

        provenance = build_provenance(zhcw_records, cwl_records)

        by_issue = {record["issue"]: record for record in provenance["records"]}
        self.assertEqual(
            by_issue["2026001"]["confidence"],
            "single_source_structurally_validated",
        )
        self.assertEqual(by_issue["2026002"]["confidence"], "dual_source_verified")
        self.assertEqual(len(by_issue["2026002"]["sources"]), 2)
        self.assertEqual(provenance["summary"]["dual_source_verified"], 1)
        self.assertEqual(provenance["summary"]["single_source_structurally_validated"], 1)


class TestForecastLedger(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        data_dir = root / "data"
        data_dir.mkdir()
        csv_path = data_dir / "ssq_history.csv"
        csv_path.write_text(
            "issue,source,date,red1,red2,red3,red4,red5,red6,blue,first_prize_count,first_prize_amount,prize1,prize2,prize3,prize4,prize5,prize6,sales,poolmoney\n"
            "2026001,zhcw,2026-01-01,1,2,3,4,5,6,7,,,,,,,,,,\n",
            encoding="utf-8",
        )
        csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        provenance_path = data_dir / "provenance.json"
        provenance_path.write_text(
            json.dumps({"schema_version": 1, "records": [{"issue": "2026001"}]}),
            encoding="utf-8",
        )
        provenance_sha = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
        manifest = {
            "csv": {"path": "data/ssq_history.csv", "sha256": csv_sha, "rows": 1},
            "provenance": {
                "path": "data/provenance.json",
                "sha256": provenance_sha,
                "rows": 1,
            },
        }
        (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        dashboard_path = root / "dashboard.json"
        prediction = {
            "label": "2026002",
            "top10_reds": list(range(1, 11)),
            "top4_blues": [1, 2, 3, 4],
            "single_ticket": {"reds": [1, 2, 3, 4, 5, 6], "blue": 1},
            "red_ranking": list(range(1, 34)),
            "blue_ranking": list(range(1, 17)),
            "top10_red_probs": [],
            "top4_blue_probs": [],
        }
        dashboard_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "rows": 1,
                        "last_issue": "2026001",
                        "next_issue": "2026002",
                        "csv_sha256": csv_sha,
                        "champion_key": "random",
                        "champion_reason": "test",
                        "no_lookahead": True,
                    },
                    "champion": {
                        "key": "random",
                        "label": "random",
                        "latest_prediction": prediction,
                    },
                    "experimental": {
                        "key": "ensemble",
                        "label": "ensemble",
                        "latest_prediction": prediction,
                    },
                    "portfolio": {"tickets": [], "metrics": {}},
                }
            ),
            encoding="utf-8",
        )
        return data_dir, dashboard_path, root / "forecasts" / "frozen" / "2026002.json"

    def test_freeze_is_idempotent_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, dashboard_path, ledger_path = self._write_fixture(root)
            frozen_at = datetime(2026, 1, 2, tzinfo=timezone.utc)

            first = freeze_forecast(
                data_dir=data_dir,
                dashboard_path=dashboard_path,
                output_path=ledger_path,
                project_root=root,
                frozen_at=frozen_at,
            )
            second = freeze_forecast(
                data_dir=data_dir,
                dashboard_path=dashboard_path,
                output_path=ledger_path,
                project_root=root,
                frozen_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
            )

            self.assertEqual(first, second)
            self.assertTrue(verify_forecast(ledger_path)["valid"])
            tampered = json.loads(ledger_path.read_text(encoding="utf-8"))
            tampered["forecast"]["strict"]["single_ticket"]["blue"] = 16
            ledger_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ForecastIntegrityError):
                verify_forecast(ledger_path)

    def test_settlement_preserves_frozen_prediction_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir, dashboard_path, ledger_path = self._write_fixture(root)
            ledger = freeze_forecast(
                data_dir=data_dir,
                dashboard_path=dashboard_path,
                output_path=ledger_path,
                project_root=root,
                frozen_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
            original_hash = ledger["forecast_sha256"]
            original_bytes = ledger_path.read_bytes()
            csv_path = data_dir / "ssq_history.csv"
            with csv_path.open("a", encoding="utf-8") as handle:
                handle.write("2026002,zhcw,2026-01-04,1,2,3,4,5,6,1,,,,,,,,,,\n")
            manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["csv"]["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            manifest["csv"]["rows"] = 2
            (data_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            settled = settle_forecast(
                ledger_path,
                data_dir=data_dir,
                settled_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            )

            self.assertEqual(ledger_path.read_bytes(), original_bytes)
            self.assertEqual(settled["forecast_sha256"], original_hash)
            self.assertEqual(settled["target_issue"], "2026002")
            self.assertEqual(settled["actual"]["reds"], [1, 2, 3, 4, 5, 6])
            self.assertEqual(settled["strict_result"]["tier"], 1)
            self.assertTrue(verify_forecast(ledger_path)["valid"])


class TestChallengerProtocol(unittest.TestCase):
    def test_development_folds_never_use_the_observed_2021_2026_test_period(self) -> None:
        self.assertGreaterEqual(len(DEVELOPMENT_FOLDS), 4)
        previous_eval_end = None
        for fold in DEVELOPMENT_FOLDS:
            self.assertLess(fold.validation_end, fold.eval_start)
            self.assertLessEqual(fold.eval_end, date(2020, 12, 31))
            if previous_eval_end is not None:
                self.assertLess(previous_eval_end, fold.eval_start)
            previous_eval_end = fold.eval_end

    def test_candidate_keys_are_unique_and_rebuildable(self) -> None:
        candidates = candidate_registry()
        self.assertGreaterEqual(len(candidates), 10)
        self.assertIn("random", {candidate.key for candidate in candidates})
        self.assertEqual(len({candidate.key for candidate in candidates}), len(candidates))
        for candidate in candidates:
            self.assertEqual(candidate.factory().key, candidate.key)


class TestChallengerArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (ROOT / "reports" / "challenger_protocol.json").read_text(encoding="utf-8")
        )

    def test_protocol_abstains_without_development_edge(self) -> None:
        meta = self.report["meta"]
        self.assertEqual(meta["development_data_end"], "2020-12-31")
        self.assertFalse(meta["observed_2021_2026_used_for_selection"])
        self.assertTrue(meta["historical_results_cannot_promote"])
        self.assertEqual(self.report["future_candidate"]["status"], "abstain")
        self.assertIsNone(self.report["future_candidate"]["key"])
        self.assertFalse(self.report["promotion"]["promoted_now"])

    def test_exploratory_panel_is_precommitted(self) -> None:
        panel = self.report["exploratory_panel"]
        self.assertEqual(len(panel), 3)
        self.assertEqual(len({item["key"] for item in panel}), 3)
        self.assertTrue(
            all(item["role"] == "precommitted_future_evidence_only" for item in panel)
        )


class TestFrozenForecastArtifact(unittest.TestCase):
    def test_next_issue_is_frozen_before_result_exists(self) -> None:
        path = ROOT / "forecasts" / "frozen" / "2026134.json"
        verified = verify_forecast(path)
        forecast = verified["payload"]["forecast"]
        historical_issues = {
            draw.issue for draw in load_draws(DEFAULT_DATA_DIR / "ssq_history.csv")
        }
        self.assertEqual(forecast["target_issue"], "2026134")
        self.assertEqual(forecast["history_boundary"]["last_issue"], "2026133")
        self.assertNotIn(forecast["target_issue"], historical_issues)
        self.assertTrue(forecast["selection"]["no_lookahead"])
        self.assertEqual(len(forecast["exploratory_challengers"]), 3)
        self.assertFalse(
            (ROOT / "forecasts" / "settled" / "2026134.json").exists()
        )


class TestFutureEvidenceAudit(unittest.TestCase):
    def test_empty_future_window_never_promotes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = build_future_evidence_report(
                forecast_dir=root / "forecasts",
                output_path=root / "future.json",
            )

            self.assertEqual(report["meta"]["settled_forecasts"], 0)
            self.assertEqual(report["models"], {})
            self.assertEqual(report["promotion"]["eligible_for_manual_review"], [])
            self.assertFalse(report["promotion"]["promoted_automatically"])

    def test_current_artifact_contains_no_replayed_history(self) -> None:
        report = json.loads(
            (ROOT / "reports" / "future_evidence.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["meta"]["settled_forecasts"], 0)
        self.assertFalse(report["meta"]["historical_replay_included"])
        self.assertEqual(report["promotion"]["eligible_for_manual_review"], [])
        self.assertFalse(report["promotion"]["promoted_automatically"])

class SpyModel(Model):
    key = "spy"
    label = "spy"

    def __init__(self) -> None:
        self.seen = []
        self.snapshots = []

    def reset(self) -> None:
        self.seen = []
        self.snapshots = []

    def observe(self, draw: Draw) -> None:
        self.seen.append(draw.issue)

    def predict(self, target_issue: str) -> Prediction:
        self.snapshots.append((target_issue, tuple(self.seen)))
        scores_red = {number: float(34 - number) for number in range(1, 34)}
        scores_blue = {number: float(17 - number) for number in range(1, 17)}
        probs_red = {number: 6.0 / 33.0 for number in range(1, 34)}
        probs_blue = {number: 1.0 / 16.0 for number in range(1, 17)}
        return Prediction(
            red_scores=scores_red,
            blue_scores=scores_blue,
            red_probs=probs_red,
            blue_probs=probs_blue,
            red_ranking=tuple(range(1, 34)),
            blue_ranking=tuple(range(1, 17)),
        )


class TestNoLookahead(unittest.TestCase):
    def test_predict_precedes_observe(self) -> None:
        draws = [
            Draw("2026001", date(2026, 1, 1), (1, 2, 3, 4, 5, 6), 1),
            Draw("2026002", date(2026, 1, 3), (2, 3, 4, 5, 6, 7), 2),
            Draw("2026003", date(2026, 1, 5), (3, 4, 5, 6, 7, 8), 3),
            Draw("2026004", date(2026, 1, 7), (4, 5, 6, 7, 8, 9), 4),
        ]
        model = SpyModel()
        cases, fitted = run_model(model, draws, date(2026, 1, 5), date(2026, 1, 5))
        self.assertEqual(len(cases), 1)
        self.assertEqual(fitted.snapshots, [("2026003", ("2026001", "2026002"))])
        self.assertEqual(fitted.seen, ["2026001", "2026002", "2026003"])


class TestDashboardArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dashboard = json.loads((ROOT / "web" / "data" / "dashboard.json").read_text(encoding="utf-8"))

    def test_artifact_shape(self) -> None:
        data = self.dashboard
        self.assertEqual(data["meta"]["rows"], 3166)
        self.assertEqual(data["meta"]["test_n"], 754)
        self.assertEqual(len(data["cases"]), 754)
        self.assertEqual(len(data["experimental"]["cases"]), 754)
        self.assertEqual(data["champion"]["key"], "random")
        self.assertEqual(data["champion"]["summary"]["exact_jackpot_count"], 0)
        self.assertEqual(data["experimental"]["test_summary"]["exact_jackpot_count"], 0)
        self.assertGreater(data["experimental"]["test_summary"]["top10_coverage_rate"], 0.88)
        self.assertLess(data["experimental"]["test_summary"]["top10_coverage_rate"], 0.93)
        for model in data["models"]:
            if model["key"] != "random":
                self.assertFalse(model["validation_vs_random"]["significant_improvement"])

        self.assertIn("portfolio", data)
        portfolio = data["portfolio"]
        self.assertEqual(len(portfolio["tickets"]), 10)
        self.assertEqual(portfolio["metrics"]["unique_ticket_count"], 10)
        self.assertEqual(portfolio["metrics"]["distinct_blue_count"], 10)
        self.assertGreater(
            portfolio["metrics"]["any_prize_probability"],
            portfolio["metrics"]["random_independent_baseline"][
                "any_prize_probability"
            ],
        )


class Test2015HoldoutArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (ROOT / "reports" / "2015_holdout.json").read_text(encoding="utf-8")
        )

    def test_protocol_and_sample(self) -> None:
        meta = self.report["meta"]
        self.assertEqual(meta["validation_start"], "2013-01-01")
        self.assertEqual(meta["validation_end"], "2014-12-31")
        self.assertEqual(meta["test_start"], "2015-01-01")
        self.assertEqual(meta["test_end"], "2015-12-31")
        self.assertEqual(meta["seed"], 20261120)
        self.assertEqual(meta["random_count"], 10)
        self.assertEqual(meta["data_rows"], 3166)
        self.assertTrue(meta["no_lookahead"])
        self.assertTrue(meta["history_replay_is_not_prediction"])
        self.assertEqual(
            meta["selected_issues"],
            [
                "2015028",
                "2015036",
                "2015055",
                "2015076",
                "2015099",
                "2015107",
                "2015109",
                "2015114",
                "2015133",
                "2015150",
            ],
        )

    def test_random10_outcome_is_not_overstated(self) -> None:
        summaries = self.report["random10"]["summaries"]
        ensemble = summaries["ensemble"]
        baseline = summaries["random"]
        self.assertEqual(ensemble["n"], 10)
        self.assertEqual(ensemble["top10_coverage_count"], 10)
        self.assertEqual(ensemble["any_prize_count"], 1)
        self.assertEqual(ensemble["exact_jackpot_count"], 0)
        self.assertEqual(baseline["top10_coverage_count"], 9)
        self.assertEqual(baseline["any_prize_count"], 0)
        self.assertLess(ensemble["top10_coverage_lower_bound_95"], 0.90)
        self.assertFalse(ensemble["meets_90pct_lower_bound"])

    def test_full_2015_is_frozen_holdout(self) -> None:
        full = self.report["full_2015"]
        self.assertEqual(full["n"], 154)
        self.assertEqual(full["summaries"]["random"]["n"], 154)
        self.assertEqual(full["summaries"]["ensemble"]["exact_jackpot_count"], 0)


class Test2024HoldoutArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (ROOT / "reports" / "2024_holdout.json").read_text(encoding="utf-8")
        )
        cls.draws = load_draws(DEFAULT_DATA_DIR / "ssq_history.csv")

    def test_selection_is_deterministic_and_year_bounded(self) -> None:
        meta = self.report["meta"]
        self.assertEqual(
            meta["selected_issues"],
            select_random_year(self.draws, year=2024, seed=20261120, count=10),
        )
        self.assertTrue(all(issue.startswith("2024") for issue in meta["selected_issues"]))
        self.assertTrue(meta["no_lookahead"])
        self.assertTrue(meta["history_replay_is_not_prediction"])
        self.assertEqual(meta["validation_start"], "2022-01-01")
        self.assertEqual(meta["validation_end"], "2023-12-31")

    def test_2024_result_keeps_strict_baseline_and_separates_metrics(self) -> None:
        meta = self.report["meta"]
        random10 = self.report["random10"]["summaries"]["random"]
        full = self.report["full_2024"]
        self.assertEqual(meta["champion_key"], "random")
        self.assertEqual(random10["n"], 10)
        self.assertEqual(random10["any_prize_count"], 0)
        self.assertEqual(random10["top3_exact_count"], 0)
        self.assertEqual(random10["top10_top4_candidate_count"], 0)
        self.assertEqual(full["n"], 151)
        self.assertIn("top3_exact_rate", full["summaries"]["ensemble"])


class TestExtremeAuditArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (ROOT / "reports" / "extreme_audit.json").read_text(encoding="utf-8")
        )

    def test_protocol_is_pure_walk_forward(self) -> None:
        meta = self.report["meta"]
        self.assertTrue(meta["no_lookahead"])
        self.assertTrue(meta["history_replay_is_not_prediction"])
        self.assertEqual(meta["data_rows"], 3166)
        self.assertEqual(meta["eval_start"], "2008-01-01")
        self.assertEqual(meta["eval_draws"], 2859)
        self.assertEqual(meta["target"], 0.99)

    def test_single_ticket_ceiling(self) -> None:
        requirements = self.report["single_ticket_99pct"]
        self.assertEqual(requirements["top3_or_better"]["iid_ticket_attempts"], 458_474)
        self.assertEqual(requirements["top3_or_better"]["random_distinct_tickets"], 452_594)
        self.assertEqual(requirements["jackpot"]["random_distinct_tickets"], 17_543_878)
        self.assertEqual(self.report["candidate_pool_99pct"]["minimum_red_pool"], 32)

    def test_no_model_reaches_99pct(self) -> None:
        models = self.report["models"]
        self.assertGreaterEqual(len(models), 8)
        self.assertIn("hedge", {model["key"] for model in models})
        self.assertEqual(max(model["jackpot_count"] for model in models), 0)
        self.assertLess(max(model["top3_rate"] for model in models), 0.001)
        self.assertLess(max(model["top10_top4_candidate_rate"] for model in models), 0.01)
        best = self.report["observed_ceiling"]
        self.assertEqual(best["best_jackpot_count"], 0)
        self.assertLess(best["best_top3_rate"], 0.99)


class TestSignalAuditArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (ROOT / "reports" / "signal_audit.json").read_text(encoding="utf-8")
        )

    def test_no_significant_frequency_or_transition_signal(self) -> None:
        frequency = self.report["frequency"]
        self.assertAlmostEqual(frequency["red_p_value"], 0.085052, places=5)
        self.assertAlmostEqual(frequency["blue_p_value"], 0.905467, places=5)
        self.assertAlmostEqual(self.report["repeats"]["p_two_sided"], 0.612519, places=5)
        self.assertAlmostEqual(self.report["transitions"]["p_two_sided"], 0.607018, places=5)
        self.assertGreater(frequency["red_p_value"], 0.05)
        self.assertGreater(self.report["repeats"]["p_two_sided"], 0.05)

    def test_no_stable_edge_and_no_promotion(self) -> None:
        annual = self.report["annual_frequency_check"]
        self.assertEqual(annual["frequency_wins"], 10)
        self.assertEqual(annual["frequency_losses"], 9)
        self.assertFalse(self.report["decision"]["any_significant_bias"])
        self.assertFalse(self.report["decision"]["stable_edge_found"])
        self.assertEqual(self.report["decision"]["action"], "keep_random_strict")


class TestPeriodReviewArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(
            (ROOT / "reports" / "period_2026_01.json").read_text(encoding="utf-8")
        )

    def test_protocol_is_strict_walk_forward(self) -> None:
        meta = self.report["meta"]
        self.assertEqual(meta["start"], "2026-01-01")
        self.assertEqual(meta["end"], "2026-01-31")
        self.assertEqual(meta["validation_end"], "2025-12-31")
        self.assertEqual(meta["review_draws"], 13)
        self.assertTrue(meta["no_lookahead"])
        self.assertTrue(meta["retrospective_replay"])
        self.assertEqual(meta["champion_key"], "random")

    def test_each_prediction_uses_only_earlier_issues(self) -> None:
        cases = self.report["cases"]
        self.assertEqual(len(cases), 13)
        self.assertEqual(cases[0]["issue"], "2026001")
        self.assertEqual(cases[0]["history_last_issue"], "2025151")
        for index, case in enumerate(cases):
            self.assertLess(int(case["history_last_issue"]), int(case["issue"]))
            if index:
                self.assertEqual(case["history_last_issue"], cases[index - 1]["issue"])

    def test_outcome_is_not_overstated(self) -> None:
        strict = self.report["summaries"]["random"]
        ensemble = self.report["summaries"]["ensemble"]
        self.assertEqual(strict["top10_coverage_count"], 11)
        self.assertEqual(strict["any_prize_count"], 0)
        self.assertEqual(strict["jackpot_count"], 0)
        self.assertEqual(ensemble["top10_coverage_count"], 12)
        self.assertEqual(ensemble["any_prize_count"], 2)
        self.assertEqual(ensemble["top3_exact_count"], 0)
        self.assertEqual(ensemble["jackpot_count"], 0)
        self.assertEqual(ensemble["top10_top4_candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
