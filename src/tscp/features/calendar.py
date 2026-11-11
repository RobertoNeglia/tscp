"""Calendar fields of the origins' timestamps."""

import torch
from torch import Tensor

from tscp.data import ResidualArchive
from tscp.features.base import ENCODINGS, FeatureMap, encode_cycles
from tscp.registry import register

FIELDS = {"minute": 60, "hour": 24, "dayofweek": 7, "day": 31, "month": 12, "dayofyear": 366}
"""Calendar fields and the length of their cycle, for the cyclic encoding."""

_FROM_ONE = ("day", "month", "dayofyear")  # fields that count from 1
_NS_PER_SECOND = 1_000_000_000
_SECONDS_PER_DAY = 86_400
_MONTH_STARTS = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)  # in a common year


def _floordiv(a: Tensor, b: int) -> Tensor:
    return torch.div(a, b, rounding_mode="floor")


def calendar_fields(timestamps: Tensor) -> dict[str, Tensor]:
    """Calendar fields of int64 nanosecond timestamps, in UTC, by integer arithmetic.

    Dates follow the proleptic Gregorian calendar, by H. Hinnant's days-to-civil
    algorithm, so they agree with Python's :mod:`datetime` before and after 1970.

    Returns:
        ``minute`` (0-59), ``hour`` (0-23), ``dayofweek`` (0 is Monday), ``day`` (1-31),
        ``month`` (1-12) and ``dayofyear`` (1-366), each an int64 tensor shaped like
        ``timestamps``.
    """
    seconds = _floordiv(timestamps, _NS_PER_SECOND)
    days = _floordiv(seconds, _SECONDS_PER_DAY)
    second_of_day = seconds - days * _SECONDS_PER_DAY
    shifted = days + 719_468  # days since 0000-03-01
    era = _floordiv(shifted, 146_097)
    day_of_era = shifted - era * 146_097
    year_of_era = _floordiv(
        day_of_era
        - _floordiv(day_of_era, 1_460)
        + _floordiv(day_of_era, 36_524)
        - _floordiv(day_of_era, 146_096),
        365,
    )
    day_from_march = day_of_era - (
        365 * year_of_era + _floordiv(year_of_era, 4) - _floordiv(year_of_era, 100)
    )
    month_from_march = _floordiv(5 * day_from_march + 2, 153)
    day = day_from_march - _floordiv(153 * month_from_march + 2, 5) + 1
    month = torch.where(month_from_march < 10, month_from_march + 3, month_from_march - 9)
    year = year_of_era + era * 400 + (month <= 2).to(torch.int64)
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    starts = torch.tensor(_MONTH_STARTS, device=timestamps.device)
    dayofyear = starts[month - 1] + day + ((month > 2) & leap).to(torch.int64)
    return {
        "minute": _floordiv(second_of_day, 60) % 60,
        "hour": _floordiv(second_of_day, 3_600),
        "dayofweek": (days + 3) % 7,  # 1970-01-01 was a Thursday
        "day": day,
        "month": month,
        "dayofyear": dayofyear,
    }


@register("features", "calendar")
class Calendar(FeatureMap):
    """Calendar fields of each origin's timestamp: ``[T, F]``.

    Fields come from the archive's ``timestamps``, in UTC; pass local times as if they
    were UTC to get local fields. ``offset`` shifts the times first, for example by the
    forecast step to read the fields of the targets. The ``"integer"`` encoding gives one
    feature per field (see :func:`calendar_fields` for the ranges); ``"cyclic"`` gives the
    sine and cosine of each field in its cycle (see :data:`FIELDS`).

    Args:
        fields: Names from :data:`FIELDS`.
        encoding: ``"integer"`` or ``"cyclic"``.
        offset: Nanoseconds added to each timestamp.
    """

    def __init__(self, fields=("hour",), encoding: str = "integer", offset: int = 0):
        super().__init__()
        fields = list(fields)
        unknown = sorted(set(fields) - set(FIELDS))
        if not fields or unknown:
            raise ValueError(f"fields must name some of {sorted(FIELDS)}, not {unknown}")
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding must be one of {ENCODINGS}, not {encoding!r}")
        self.fields = fields
        self.encoding = encoding
        self.offset = int(offset)

    def forward(self, rows: ResidualArchive) -> Tensor:
        if rows.timestamps is None:
            raise ValueError("Calendar needs an archive with timestamps")
        values = calendar_fields(rows.timestamps + self.offset)
        shift = int(self.encoding == "cyclic")  # cycles start at 0
        columns = [values[f] - (shift if f in _FROM_ONE else 0) for f in self.fields]
        periods = torch.tensor([FIELDS[field] for field in self.fields])
        return encode_cycles(torch.stack(columns, dim=-1), periods, self.encoding)


__all__ = ["FIELDS", "Calendar", "calendar_fields"]
