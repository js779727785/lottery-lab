"""Core data types and deterministic win-tier rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RED_MIN, RED_MAX = 1, 33
BLUE_MIN, BLUE_MAX = 1, 16
RED_COUNT = 6
TICKET_COST = 2

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:\([^)]*\))?$")
ISSUE_RE = re.compile(r"^(\d{4})(\d{3})$")


class DataValidationError(ValueError):
    """Raised when an official record fails structural validation."""


@dataclass(frozen=True)
class Draw:
    issue: str
    draw_date: date
    reds: Tuple[int, int, int, int, int, int]
    blue: int
    first_prize_count: Optional[int] = None
    first_prize_amount: Optional[int] = None
    prize_amounts: Optional[Mapping[int, int]] = None
    sales: Optional[int] = None
    poolmoney: Optional[int] = None
    source: str = "cwl"

    @property
    def red_set(self) -> frozenset[int]:
        return frozenset(self.reds)

    @property
    def issue_year(self) -> int:
        return int(self.issue[:4])

    @property
    def issue_number(self) -> int:
        return int(self.issue[4:])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "issue": self.issue,
            "date": self.draw_date.isoformat(),
            "reds": list(self.reds),
            "blue": self.blue,
            "first_prize_count": self.first_prize_count,
            "first_prize_amount": self.first_prize_amount,
            "prize_amounts": dict(self.prize_amounts or {}),
            "sales": self.sales,
            "poolmoney": self.poolmoney,
            "source": self.source,
        }


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if not text.isdigit():
        return None
    return int(text)


def _parse_date(value: str) -> date:
    match = DATE_RE.match(value.strip())
    if not match:
        raise DataValidationError(f"invalid draw date: {value!r}")
    return date.fromisoformat(match.group(1))


def _parse_reds(value: str) -> Tuple[int, int, int, int, int, int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != RED_COUNT:
        raise DataValidationError(f"expected {RED_COUNT} red balls, got {len(parts)}: {value!r}")
    try:
        reds = tuple(sorted(int(part) for part in parts))
    except ValueError as exc:
        raise DataValidationError(f"non-integer red ball: {value!r}") from exc
    if len(set(reds)) != RED_COUNT:
        raise DataValidationError(f"duplicate red ball: {value!r}")
    if any(number < RED_MIN or number > RED_MAX for number in reds):
        raise DataValidationError(f"red ball out of range: {value!r}")
    return reds  # type: ignore[return-value]


def draw_from_official(record: Mapping[str, Any]) -> Draw:
    """Convert one cwl.gov.cn result object into a validated Draw."""

    issue = str(record.get("code", "")).strip()
    issue_match = ISSUE_RE.match(issue)
    if not issue_match:
        raise DataValidationError(f"invalid issue code: {issue!r}")

    draw_date = _parse_date(str(record.get("date", "")))
    if int(issue_match.group(1)) != draw_date.year:
        raise DataValidationError(
            f"issue year {issue_match.group(1)} does not match date year {draw_date.year} for {issue}"
        )

    reds = _parse_reds(str(record.get("red", "")))
    blue_text = str(record.get("blue", "")).strip()
    if not blue_text.isdigit():
        raise DataValidationError(f"invalid blue ball for {issue}: {blue_text!r}")
    blue = int(blue_text)
    if blue < BLUE_MIN or blue > BLUE_MAX:
        raise DataValidationError(f"blue ball out of range for {issue}: {blue}")

    prize_amounts: Dict[int, int] = {}
    first_prize_count: Optional[int] = None
    first_prize_amount: Optional[int] = None
    for grade in record.get("prizegrades", []) or []:
        grade_type = _parse_int(grade.get("type"))
        if grade_type is None or grade_type < 1 or grade_type > 6:
            continue
        amount = _parse_int(grade.get("typemoney"))
        if amount is not None:
            prize_amounts[grade_type] = amount
        if grade_type == 1:
            first_prize_count = _parse_int(grade.get("typenum"))
            first_prize_amount = amount

    return Draw(
        issue=issue,
        draw_date=draw_date,
        reds=reds,
        blue=blue,
        first_prize_count=first_prize_count,
        first_prize_amount=first_prize_amount,
        prize_amounts=prize_amounts,
        sales=_parse_int(record.get("sales")),
        poolmoney=_parse_int(record.get("poolmoney")),
        source="cwl",
    )


def _parse_money(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def draw_from_zhcw(record: Mapping[str, Any]) -> Draw:
    """Convert one 中彩网 result object into a validated Draw."""

    issue = str(record.get("issue", "")).strip()
    issue_match = ISSUE_RE.match(issue)
    if not issue_match:
        raise DataValidationError(f"invalid zhcw issue code: {issue!r}")

    draw_date = _parse_date(str(record.get("openTime", "")))
    if int(issue_match.group(1)) != draw_date.year:
        raise DataValidationError(
            f"zhcw issue year {issue_match.group(1)} does not match date year {draw_date.year} for {issue}"
        )

    reds = _parse_reds(str(record.get("frontWinningNum", "")).replace(" ", ","))
    blue_text = str(record.get("backWinningNum", "")).strip()
    if not blue_text.isdigit():
        raise DataValidationError(f"invalid zhcw blue ball for {issue}: {blue_text!r}")
    blue = int(blue_text)
    if blue < BLUE_MIN or blue > BLUE_MAX:
        raise DataValidationError(f"zhcw blue ball out of range for {issue}: {blue}")

    prize_amounts: Dict[int, int] = {}
    first_prize_count: Optional[int] = None
    first_prize_amount: Optional[int] = None
    for detail in record.get("winnerDetails", []) or []:
        grade_type = _parse_int(detail.get("awardEtc"))
        if grade_type is None or grade_type < 1 or grade_type > 6:
            continue
        base = detail.get("baseBetWinner") or {}
        count = _parse_int(base.get("awardNum"))
        amount = _parse_money(base.get("awardMoney"))
        if amount is not None:
            prize_amounts[grade_type] = amount
        if grade_type == 1:
            first_prize_count = count
            first_prize_amount = amount

    return Draw(
        issue=issue,
        draw_date=draw_date,
        reds=reds,
        blue=blue,
        first_prize_count=first_prize_count,
        first_prize_amount=first_prize_amount,
        prize_amounts=prize_amounts,
        sales=_parse_money(record.get("saleMoney")),
        poolmoney=_parse_money(record.get("prizePoolMoney")),
        source="zhcw",
    )


def prize_tier(red_hits: int, blue_hit: bool) -> Optional[int]:
    """Return the official 双色球 tier for one single ticket, or None."""

    if red_hits == 6 and blue_hit:
        return 1
    if red_hits == 6:
        return 2
    if red_hits == 5 and blue_hit:
        return 3
    if red_hits == 5 or (red_hits == 4 and blue_hit):
        return 4
    if red_hits == 4 or (red_hits == 3 and blue_hit):
        return 5
    if blue_hit:
        return 6
    return None


def validate_draw_sequence(
    draws: Sequence[Draw],
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> Dict[str, Any]:
    """Run strict invariants over a complete yearly sequence."""

    errors: List[str] = []
    warnings: List[str] = []
    if not draws:
        raise DataValidationError("draw sequence is empty")

    issues = [draw.issue for draw in draws]
    if len(set(issues)) != len(issues):
        errors.append("duplicate issue codes")

    sorted_draws = sorted(draws, key=lambda draw: (draw.draw_date, draw.issue))
    for previous, current in zip(sorted_draws, sorted_draws[1:]):
        if current.draw_date <= previous.draw_date:
            errors.append(
                f"non-increasing dates: {previous.issue} {previous.draw_date} -> "
                f"{current.issue} {current.draw_date}"
            )
        if int(current.issue) <= int(previous.issue):
            errors.append(f"non-increasing issues: {previous.issue} -> {current.issue}")

    by_year: Dict[int, List[Draw]] = {}
    for draw in draws:
        by_year.setdefault(draw.issue_year, []).append(draw)
        if start and draw.draw_date < start:
            errors.append(f"{draw.issue} is before requested start {start}")
        if end and draw.draw_date > end:
            errors.append(f"{draw.issue} is after requested end {end}")
        if draw.first_prize_count is None:
            warnings.append(f"{draw.issue} has no first-prize count")
        if draw.first_prize_amount is None:
            warnings.append(f"{draw.issue} has no first-prize amount")

    for year, year_draws in sorted(by_year.items()):
        numbers = sorted(draw.issue_number for draw in year_draws)
        expected = list(range(1, max(numbers) + 1))
        if numbers != expected:
            missing = sorted(set(expected) - set(numbers))
            duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
            errors.append(
                f"{year} issue sequence is not continuous; missing={missing[:20]}, "
                f"duplicates={duplicates[:20]}"
            )

    if errors:
        raise DataValidationError("; ".join(errors))

    return {
        "ok": True,
        "rows": len(draws),
        "first_issue": sorted_draws[0].issue,
        "first_date": sorted_draws[0].draw_date.isoformat(),
        "last_issue": sorted_draws[-1].issue,
        "last_date": sorted_draws[-1].draw_date.isoformat(),
        "years": sorted(by_year),
        "warnings": warnings,
    }


def draws_by_issue(draws: Iterable[Draw]) -> Dict[str, Draw]:
    return {draw.issue: draw for draw in draws}
