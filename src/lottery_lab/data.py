"""Official cwl.gov.cn data collection, validation, and persistence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import (
    Draw,
    draw_from_official,
    draw_from_zhcw,
    draws_by_issue,
    validate_draw_sequence,
)


OFFICIAL_ENDPOINT = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
)
ZHCW_ENDPOINT = "https://jc.zhcw.com/port/client_json.php"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV = DEFAULT_DATA_DIR / "ssq_history.csv"
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "manifest.json"

CSV_FIELDS = [
    "issue",
    "source",
    "date",
    "red1",
    "red2",
    "red3",
    "red4",
    "red5",
    "red6",
    "blue",
    "first_prize_count",
    "first_prize_amount",
    "prize1",
    "prize2",
    "prize3",
    "prize4",
    "prize5",
    "prize6",
    "sales",
    "poolmoney",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_json(url: str, retries: int = 4, timeout: int = 30) -> Dict[str, Any]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Referer": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
        "Accept": "application/json, text/plain, */*",
    }
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            data = json.loads(payload)
            if data.get("state") != 0:
                raise RuntimeError(f"official API error: {data.get('message')!r}")
            return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch official data after {retries} attempts: {last_error}")


def _page_url(start: date, end: date, page: int, page_size: int) -> str:
    query = urllib.parse.urlencode(
        {
            "name": "ssq",
            "issueCount": "",
            "issueStart": "",
            "issueEnd": "",
            "dayStart": start.isoformat(),
            "dayEnd": end.isoformat(),
            "pageNo": page,
            "pageSize": page_size,
            "week": "",
            "systemType": "PC",
        }
    )
    return f"{OFFICIAL_ENDPOINT}?{query}"


def fetch_pages(
    start: date,
    end: date,
    page_size: int = 100,
    raw_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch every official page and optionally persist each raw response."""

    first = _request_json(_page_url(start, end, 1, page_size))
    total = int(first.get("total", 0))
    if total <= 0:
        raise RuntimeError("official API returned zero rows")
    page_count = math.ceil(total / page_size)
    pages: List[Dict[str, Any]] = [first]
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "page_0001.json").write_text(
            json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for page in range(2, page_count + 1):
        payload = _request_json(_page_url(start, end, page, page_size))
        pages.append(payload)
        if raw_dir is not None:
            (raw_dir / f"page_{page:04d}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    records: List[Dict[str, Any]] = []
    for payload in pages:
        records.extend(payload.get("result", []) or [])
    if len(records) != total:
        raise RuntimeError(f"pagination mismatch: expected {total}, collected {len(records)}")
    meta = {
        "source": OFFICIAL_ENDPOINT,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "page_size": page_size,
        "page_count": page_count,
        "reported_total": total,
        "collected": len(records),
    }
    return records, meta


def crosscheck_pagination(
    records: Sequence[Dict[str, Any]],
    start: date,
    end: date,
    page_size: int = 50,
) -> Dict[str, Any]:
    """Re-fetch with a different page size and compare every issue."""

    second_records, meta = fetch_pages(start, end, page_size=page_size)
    first_map = draws_by_issue(draw_from_official(record) for record in records)
    second_map = draws_by_issue(draw_from_official(record) for record in second_records)
    missing = sorted(set(first_map) - set(second_map))
    extra = sorted(set(second_map) - set(first_map))
    mismatched = sorted(
        issue
        for issue in set(first_map) & set(second_map)
        if (first_map[issue].reds, first_map[issue].blue)
        != (second_map[issue].reds, second_map[issue].blue)
    )
    if missing or extra or mismatched:
        raise RuntimeError(
            f"pagination cross-check failed: missing={missing[:10]}, "
            f"extra={extra[:10]}, mismatched={mismatched[:10]}"
        )
    return {
        "ok": True,
        "page_size": page_size,
        "rows": len(second_map),
        "issues_match": True,
        "winning_numbers_match": True,
        "pages": meta["page_count"],
    }


def write_csv(draws: Iterable[Draw], path: Path = DEFAULT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_draws = sorted(draws, key=lambda draw: int(draw.issue))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for draw in sorted_draws:
            prizes = draw.prize_amounts or {}
            row: Dict[str, Any] = {
                "issue": draw.issue,
                "source": draw.source,
                "date": draw.draw_date.isoformat(),
                "blue": draw.blue,
                "first_prize_count": draw.first_prize_count,
                "first_prize_amount": draw.first_prize_amount,
                "sales": draw.sales,
                "poolmoney": draw.poolmoney,
            }
            for index, number in enumerate(draw.reds, start=1):
                row[f"red{index}"] = number
            for tier in range(1, 7):
                row[f"prize{tier}"] = prizes.get(tier, "")
            writer.writerow(row)


def load_draws(path: Path = DEFAULT_CSV) -> List[Draw]:
    draws: List[Draw] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prize_amounts = {}
            for tier in range(1, 7):
                value = row.get(f"prize{tier}", "").strip()
                if value:
                    prize_amounts[tier] = int(value)
            draws.append(
                Draw(
                    issue=row["issue"],
                    draw_date=date.fromisoformat(row["date"]),
                    reds=tuple(int(row[f"red{i}"]) for i in range(1, 7)),  # type: ignore[arg-type]
                    blue=int(row["blue"]),
                    first_prize_count=int(row["first_prize_count"])
                    if row.get("first_prize_count")
                    else None,
                    first_prize_amount=int(row["first_prize_amount"])
                    if row.get("first_prize_amount")
                    else None,
                    prize_amounts=prize_amounts,
                    sales=int(row["sales"]) if row.get("sales") else None,
                    poolmoney=int(row["poolmoney"]) if row.get("poolmoney") else None,
                    source=row.get("source", "cwl"),
                )
            )
    return sorted(draws, key=lambda draw: int(draw.issue))


def collect_and_validate(
    start: date,
    end: date,
    data_dir: Path = DEFAULT_DATA_DIR,
    crosscheck: bool = True,
) -> Dict[str, Any]:
    """Full collection pipeline. Fails closed if any validation fails."""

    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    records, page_meta = fetch_pages(start, end, page_size=100, raw_dir=raw_dir)
    draws = [draw_from_official(record) for record in records]
    quality = validate_draw_sequence(draws, start=start, end=end)

    raw_combined = data_dir / "raw" / "official_records.json"
    raw_combined.write_text(
        json.dumps(
            {
                "meta": page_meta,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = data_dir / "ssq_history.csv"
    write_csv(draws, csv_path)
    manifest_path = data_dir / "manifest.json"
    manifest: Dict[str, Any] = {
        "source": OFFICIAL_ENDPOINT,
        "source_page": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "fetch": page_meta,
        "quality": quality,
        "csv": {
            "path": str(csv_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(csv_path),
            "rows": len(draws),
        },
        "raw_records": {
            "path": str(raw_combined.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(raw_combined),
        },
    }
    if crosscheck:
        manifest["pagination_crosscheck"] = crosscheck_pagination(records, start, end)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _parse_jsonp(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("(")
    end = stripped.rfind(")")
    if start < 0 or end <= start:
        raise RuntimeError("invalid JSONP response")
    return json.loads(stripped[start + 1 : end])


def _request_zhcw_json(
    start_issue: str,
    end_issue: str,
    page: int,
    page_size: int,
    retries: int = 4,
    timeout: int = 30,
) -> Dict[str, Any]:
    params = {
        "transactionType": "10001001",
        "lotteryId": "1",
        "issueCount": "0",
        "startIssue": start_issue,
        "endIssue": end_issue,
        "startDate": "",
        "endDate": "",
        "type": "1",
        "pageNum": str(page),
        "pageSize": str(page_size),
        "callback": "cb",
    }
    url = f"{ZHCW_ENDPOINT}?{urllib.parse.urlencode(params)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Referer": "https://www.zhcw.com/kjxx/ssq/",
        "Accept": "*/*",
    }
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            data = _parse_jsonp(payload)
            if data.get("resCode") and data.get("resCode") != "000000":
                raise RuntimeError(f"zhcw API error: {data.get('message')!r}")
            return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch zhcw data after {retries} attempts: {last_error}")


def fetch_zhcw_pages(
    start_issue: str,
    end_issue: str,
    page_size: int = 100,
    raw_dir: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    first = _request_zhcw_json(start_issue, end_issue, 1, page_size)
    total = int(first.get("total", 0))
    page_count = int(first.get("pages", 0))
    if total <= 0 or page_count <= 0:
        raise RuntimeError("zhcw API returned zero rows")
    pages: List[Dict[str, Any]] = [first]
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "page_0001.json").write_text(
            json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    for page in range(2, page_count + 1):
        payload = _request_zhcw_json(start_issue, end_issue, page, page_size)
        pages.append(payload)
        if raw_dir is not None:
            (raw_dir / f"page_{page:04d}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    records: List[Dict[str, Any]] = []
    for payload in pages:
        records.extend(payload.get("data", []) or [])
    if len(records) != total:
        raise RuntimeError(f"zhcw pagination mismatch: expected {total}, collected {len(records)}")
    return records, {
        "source": ZHCW_ENDPOINT,
        "start_issue": start_issue,
        "end_issue": end_issue,
        "page_size": page_size,
        "page_count": page_count,
        "reported_total": total,
        "collected": len(records),
    }


def crosscheck_sources(
    zhcw_draws: Sequence[Draw],
    cwl_draws: Sequence[Draw],
) -> Dict[str, Any]:
    zhcw_map = draws_by_issue(zhcw_draws)
    cwl_map = draws_by_issue(cwl_draws)
    missing = sorted(set(cwl_map) - set(zhcw_map))
    numbers_mismatch = sorted(
        issue
        for issue in set(zhcw_map) & set(cwl_map)
        if (zhcw_map[issue].reds, zhcw_map[issue].blue)
        != (cwl_map[issue].reds, cwl_map[issue].blue)
    )
    prize_mismatch = []
    for issue in set(zhcw_map) & set(cwl_map):
        left = zhcw_map[issue]
        right = cwl_map[issue]
        if right.first_prize_count is not None and left.first_prize_count != right.first_prize_count:
            prize_mismatch.append(issue)
        if right.first_prize_amount is not None and left.first_prize_amount != right.first_prize_amount:
            prize_mismatch.append(issue)
    if missing or numbers_mismatch or prize_mismatch:
        raise RuntimeError(
            f"source cross-check failed: missing={missing[:10]}, "
            f"numbers_mismatch={numbers_mismatch[:10]}, prize_mismatch={sorted(set(prize_mismatch))[:10]}"
        )
    return {
        "ok": True,
        "overlap_issues": len(set(zhcw_map) & set(cwl_map)),
        "zhcw_rows": len(zhcw_map),
        "cwl_rows": len(cwl_map),
        "issues_match": not missing,
        "winning_numbers_match": not numbers_mismatch,
        "first_prize_match_where_cwl_available": not prize_mismatch,
        "cwl_first_prize_missing": sum(
            1 for draw in cwl_draws if draw.first_prize_amount is None
        ),
    }


def collect_20y(
    start: date,
    end: date,
    data_dir: Path = DEFAULT_DATA_DIR,
    cwl_overlap_start: date = date(2013, 1, 1),
) -> Dict[str, Any]:
    """Collect 2006+ from 中彩网 and cross-check overlap against 福彩官网."""

    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    start_issue = f"{start.year}001"
    end_issue = f"{end.year}999"

    zhcw_records, zhcw_meta = fetch_zhcw_pages(
        start_issue, end_issue, page_size=100, raw_dir=raw_dir / "zhcw"
    )
    zhcw_draws = sorted(
        (draw_from_zhcw(record) for record in zhcw_records), key=lambda draw: int(draw.issue)
    )
    quality = validate_draw_sequence(zhcw_draws, start=start, end=end)

    cwl_start = max(start, cwl_overlap_start)
    cwl_records, cwl_meta = fetch_pages(
        cwl_start, end, page_size=50, raw_dir=raw_dir / "cwl"
    )
    cwl_draws = sorted(
        (draw_from_official(record) for record in cwl_records), key=lambda draw: int(draw.issue)
    )
    overlap = crosscheck_sources(zhcw_draws, cwl_draws)

    zhcw_raw = raw_dir / "zhcw_records.json"
    zhcw_raw.write_text(
        json.dumps(
            {
                "meta": zhcw_meta,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "records": zhcw_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    cwl_raw = raw_dir / "cwl_records.json"
    cwl_raw.write_text(
        json.dumps(
            {
                "meta": cwl_meta,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "records": cwl_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = data_dir / "ssq_history.csv"
    write_csv(zhcw_draws, csv_path)
    manifest = {
        "source": "中彩网 + 中国福彩官网",
        "source_page": "https://www.zhcw.com/kjxx/ssq/",
        "sources": {
            "zhcw": ZHCW_ENDPOINT,
            "cwl": OFFICIAL_ENDPOINT,
        },
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "zhcw_fetch": zhcw_meta,
        "cwl_overlap_fetch": cwl_meta,
        "quality": quality,
        "overlap_crosscheck": overlap,
        "pagination_crosscheck": {
            "ok": overlap["ok"],
            "rows": overlap["overlap_issues"],
            "issues_match": overlap["issues_match"],
            "winning_numbers_match": overlap["winning_numbers_match"],
            "method": "zhcw 100/page + cwl 50/page",
        },
        "csv": {
            "path": str(csv_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(csv_path),
            "rows": len(zhcw_draws),
        },
        "raw_records": {
            "path": str(zhcw_raw.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(zhcw_raw),
        },
        "cwl_raw_records": {
            "path": str(cwl_raw.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(cwl_raw),
        },
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
