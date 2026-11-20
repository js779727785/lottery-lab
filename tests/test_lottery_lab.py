from __future__ import annotations

import hashlib
import json
import unittest
from datetime import date
from pathlib import Path

from lottery_lab.backtest import run_model
from lottery_lab.data import DEFAULT_DATA_DIR, load_draws
from lottery_lab.metrics import (
    ndcg_at_k,
    paired_bootstrap_difference,
    theoretical_baselines,
    wilson_interval,
)
from lottery_lab.models import Model, Prediction
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
        self.assertEqual(len(models), 8)
        self.assertEqual(max(model["jackpot_count"] for model in models), 0)
        self.assertLess(max(model["top3_rate"] for model in models), 0.001)
        self.assertLess(max(model["top10_top4_candidate_rate"] for model in models), 0.01)
        best = self.report["observed_ceiling"]
        self.assertEqual(best["best_jackpot_count"], 0)
        self.assertLess(best["best_top3_rate"], 0.99)


if __name__ == "__main__":
    unittest.main()
