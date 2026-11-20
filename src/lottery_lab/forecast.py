"""Tamper-evident pre-draw forecast freezing and post-draw settlement."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .data import DEFAULT_DATA_DIR, load_draws
from .metrics import (
    BLUE_NUMBERS,
    RED_NUMBERS,
    brier_score,
    log_loss,
    mean_actual_rank,
    ndcg_at_k,
)
from .pipeline import DEFAULT_WEB_DATA
from .schema import Draw, prize_tier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORECAST_DIR = PROJECT_ROOT / "forecasts"


class ForecastIntegrityError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_iso(value: Optional[datetime] = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat()


def _git_head(project_root: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_tree_sha256(project_root: Path) -> str:
    paths: Iterable[Path] = sorted((project_root / "src").rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _settlement_path(forecast_path: Path) -> Path:
    if forecast_path.parent.name == "frozen":
        return forecast_path.parent.parent / "settled" / forecast_path.name
    return forecast_path.with_name(f"{forecast_path.stem}.settlement.json")


def verify_forecast(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or "forecast" not in payload:
        raise ForecastIntegrityError(f"invalid forecast schema: {path}")
    expected = _value_sha256(payload["forecast"])
    if payload.get("forecast_sha256") != expected:
        raise ForecastIntegrityError(f"forecast hash mismatch: {path}")
    target = payload["forecast"].get("target_issue")
    if path.stem != target:
        raise ForecastIntegrityError(
            f"forecast filename {path.stem} does not match target {target}"
        )
    return {"valid": True, "forecast_sha256": expected, "payload": payload}


def verify_settlement(
    path: Path, forecast_path: Optional[Path] = None
) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    settlement = payload.get("settlement")
    if payload.get("schema_version") != 1 or not isinstance(settlement, dict):
        raise ForecastIntegrityError(f"invalid settlement schema: {path}")
    expected = _value_sha256(settlement)
    if payload.get("settlement_sha256") != expected:
        raise ForecastIntegrityError(f"settlement hash mismatch: {path}")
    frozen_path = forecast_path
    if frozen_path is None and path.parent.name == "settled":
        frozen_path = path.parent.parent / "frozen" / path.name
    if frozen_path is None:
        frozen_path_text = payload.get("forecast_path")
        frozen_path = Path(frozen_path_text) if frozen_path_text else None
    if frozen_path is None or not frozen_path.exists():
        raise ForecastIntegrityError(f"referenced frozen forecast not found: {path}")
    forecast_verified = verify_forecast(frozen_path)
    if payload.get("forecast_sha256") != forecast_verified["forecast_sha256"]:
        raise ForecastIntegrityError("settlement references a different forecast hash")
    if settlement.get("target_issue") != forecast_verified["payload"]["forecast"].get(
        "target_issue"
    ):
        raise ForecastIntegrityError("settlement target differs from frozen forecast")
    return {
        "valid": True,
        "settlement_sha256": expected,
        "payload": payload,
        "forecast": forecast_verified["payload"],
    }


def freeze_forecast(
    data_dir: Path = DEFAULT_DATA_DIR,
    dashboard_path: Path = DEFAULT_WEB_DATA,
    output_path: Optional[Path] = None,
    project_root: Path = PROJECT_ROOT,
    frozen_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    draws = load_draws(data_dir / "ssq_history.csv")
    issues = {draw.issue for draw in draws}
    target_issue = str(dashboard["meta"]["next_issue"])
    if target_issue in issues:
        raise ForecastIntegrityError(
            f"refusing to freeze {target_issue}: result already exists in historical data"
        )
    if not draws or dashboard["meta"].get("last_issue") != draws[-1].issue:
        raise ForecastIntegrityError("dashboard history boundary does not match current CSV")
    csv_path = data_dir / "ssq_history.csv"
    csv_sha = _file_sha256(csv_path)
    if csv_sha != manifest.get("csv", {}).get("sha256"):
        raise ForecastIntegrityError("CSV hash does not match manifest")
    if dashboard["meta"].get("csv_sha256") != csv_sha:
        raise ForecastIntegrityError("dashboard CSV hash does not match manifest")
    provenance = manifest.get("provenance", {})
    provenance_path = data_dir / "provenance.json"
    if provenance and _file_sha256(provenance_path) != provenance.get("sha256"):
        raise ForecastIntegrityError("provenance hash does not match manifest")

    destination = output_path or DEFAULT_FORECAST_DIR / "frozen" / f"{target_issue}.json"
    if destination.exists():
        existing = verify_forecast(destination)["payload"]
        if existing["forecast"]["target_issue"] != target_issue:
            raise ForecastIntegrityError(f"forecast path already belongs to another issue: {destination}")
        return existing

    formal_challenger = dashboard.get("development_challenger")
    challenger = formal_challenger or dashboard["champion"]
    forecast = {
        "target_issue": target_issue,
        "frozen_at_utc": _utc_iso(frozen_at),
        "history_boundary": {
            "last_issue": draws[-1].issue,
            "last_date": draws[-1].draw_date.isoformat(),
            "rows": len(draws),
        },
        "data_snapshot": {
            "manifest_sha256": _file_sha256(manifest_path),
            "csv_sha256": csv_sha,
            "provenance_sha256": provenance.get("sha256"),
        },
        "code_snapshot": {
            "git_head": _git_head(project_root),
            "source_tree_sha256": _source_tree_sha256(project_root),
        },
        "selection": {
            "strict_model_key": dashboard["champion"]["key"],
            "strict_model_label": dashboard["champion"]["label"],
            "champion_reason": dashboard["meta"].get("champion_reason"),
            "no_lookahead": dashboard["meta"].get("no_lookahead") is True,
            "challenger_model_key": challenger["key"],
            "challenger_model_label": challenger["label"],
            "challenger_status": "formal" if formal_challenger else "abstain",
        },
        "strict": dashboard["champion"]["latest_prediction"],
        "challenger": challenger["latest_prediction"],
        "exploratory_challengers": dashboard.get("exploratory_challengers", []),
        "portfolio": dashboard.get("portfolio"),
        "integrity_limit": (
            "本地 SHA-256 可发现意外或未同步篡改；只有提交到外部 Git 远端或可信时间戳后，"
            "才能对抗可重写全部本地文件的攻击者。"
        ),
    }
    payload = {
        "schema_version": 1,
        "forecast": forecast,
        "forecast_sha256": _value_sha256(forecast),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _ticket_result(prediction: Dict[str, Any], actual: Draw) -> Dict[str, Any]:
    ticket = prediction["single_ticket"]
    red_hits = len(set(ticket["reds"]) & set(actual.reds))
    blue_hit = int(ticket["blue"]) == actual.blue
    top10_hits = len(set(prediction.get("top10_reds", [])) & set(actual.reds))
    top4_blue_hit = actual.blue in set(prediction.get("top4_blues", []))
    result = {
        "red_hits": red_hits,
        "blue_hit": blue_hit,
        "tier": prize_tier(red_hits, blue_hit),
        "top10_red_hits": top10_hits,
        "top4_blue_hit": top4_blue_hit,
    }
    red_probabilities = prediction.get("red_probabilities")
    blue_probabilities = prediction.get("blue_probabilities")
    if (
        isinstance(red_probabilities, list)
        and len(red_probabilities) == len(RED_NUMBERS)
        and isinstance(blue_probabilities, list)
        and len(blue_probabilities) == len(BLUE_NUMBERS)
    ):
        red_labels = [1 if number in actual.red_set else 0 for number in RED_NUMBERS]
        blue_labels = [1 if number == actual.blue else 0 for number in BLUE_NUMBERS]
        result.update(
            {
                "scoring_available": True,
                "ndcg_at_10": ndcg_at_k(
                    prediction["red_ranking"], actual.reds, 10
                ),
                "mean_actual_rank": mean_actual_rank(
                    prediction["red_ranking"], actual.reds
                ),
                "brier_red": brier_score(red_probabilities, red_labels),
                "brier_blue": brier_score(blue_probabilities, blue_labels),
                "logloss_red": log_loss(red_probabilities, red_labels),
                "logloss_blue": log_loss(blue_probabilities, blue_labels),
                "combined_logloss": log_loss(red_probabilities, red_labels)
                + log_loss(blue_probabilities, blue_labels),
            }
        )
    else:
        result["scoring_available"] = False
    return result


def settle_forecast(
    forecast_path: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Optional[Path] = None,
    settled_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    verified = verify_forecast(forecast_path)
    frozen = verified["payload"]
    forecast = frozen["forecast"]
    target_issue = forecast["target_issue"]
    destination = output_path or _settlement_path(forecast_path)
    if destination.exists():
        return verify_settlement(destination, forecast_path)["payload"]["settlement"]

    actual = next(
        (draw for draw in load_draws(data_dir / "ssq_history.csv") if draw.issue == target_issue),
        None,
    )
    if actual is None:
        raise ForecastIntegrityError(f"cannot settle {target_issue}: official result not present")
    settlement = {
        "target_issue": target_issue,
        "settled_at_utc": _utc_iso(settled_at),
        "forecast_sha256": verified["forecast_sha256"],
        "actual": actual.as_dict(),
        "strict_result": _ticket_result(forecast["strict"], actual),
        "challenger_result": _ticket_result(forecast["challenger"], actual),
        "exploratory_results": [
            {
                "key": item["key"],
                "result": _ticket_result(item["latest_prediction"], actual),
            }
            for item in forecast.get("exploratory_challengers", [])
        ],
        "settlement_data_sha256": _file_sha256(data_dir / "ssq_history.csv"),
    }
    payload = {
        "schema_version": 1,
        "forecast_path": str(forecast_path),
        "forecast_sha256": verified["forecast_sha256"],
        "settlement": settlement,
        "settlement_sha256": _value_sha256(settlement),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return settlement


def sync_forecasts(
    data_dir: Path = DEFAULT_DATA_DIR,
    dashboard_path: Path = DEFAULT_WEB_DATA,
    forecast_dir: Path = DEFAULT_FORECAST_DIR,
) -> Dict[str, Any]:
    settled = []
    frozen_dir = forecast_dir / "frozen"
    for path in sorted(frozen_dir.glob("*.json")):
        target = verify_forecast(path)["payload"]["forecast"]["target_issue"]
        if any(draw.issue == target for draw in load_draws(data_dir / "ssq_history.csv")):
            settle_forecast(path, data_dir=data_dir)
            settled.append(target)
    frozen = freeze_forecast(
        data_dir=data_dir,
        dashboard_path=dashboard_path,
        output_path=frozen_dir
        / f"{json.loads(dashboard_path.read_text(encoding='utf-8'))['meta']['next_issue']}.json",
    )
    return {
        "settled": settled,
        "frozen_issue": frozen["forecast"]["target_issue"],
        "forecast_sha256": frozen["forecast_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "settle", "verify", "verify-settlement", "sync")
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dashboard", default=str(DEFAULT_WEB_DATA))
    parser.add_argument("--forecast-dir", default=str(DEFAULT_FORECAST_DIR))
    parser.add_argument("--forecast")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.command == "sync":
        result = sync_forecasts(data_dir, Path(args.dashboard), Path(args.forecast_dir))
    elif args.command == "freeze":
        result = freeze_forecast(data_dir, Path(args.dashboard))
    else:
        if not args.forecast:
            parser.error("--forecast is required for settle/verify")
        forecast_path = Path(args.forecast)
        if args.command == "settle":
            result = settle_forecast(forecast_path, data_dir)
        elif args.command == "verify-settlement":
            result = verify_settlement(forecast_path)
        else:
            result = verify_forecast(forecast_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
