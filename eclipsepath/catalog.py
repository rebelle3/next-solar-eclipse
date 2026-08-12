"""Top-level search: every solar eclipse in a date range, with its path."""

from __future__ import annotations

import datetime as dt

from . import eclipse as ec
from . import finder
from .ephemeris import Ephemeris

TYPE_NAMES = {'T': 'Total', 'A': 'Annular', 'H': 'Hybrid', 'P': 'Partial'}


def to_datetime(value):
    """Accept a date, a datetime or an ISO-8601 string."""
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day,
                           tzinfo=dt.timezone.utc)
    text = str(value).strip().replace('Z', '+00:00')
    parsed = dt.datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def add_years(moment, years):
    try:
        return moment.replace(year=moment.year + int(years))
    except ValueError:                      # 29 February
        return moment.replace(year=moment.year + int(years), day=28)


def search(start, end=None, years=10, threshold=1.0, samples=120,
           ephemeris=None, kernel=None, trace=True, progress=None):
    """Solar eclipses between ``start`` and ``end`` whose coverage reaches
    ``threshold`` (a fraction of the Sun's area, so 1.0 means totality).

    Eclipses below the threshold are still returned, flagged via
    ``Eclipse.meets_threshold``, so a caller can report what it filtered out.
    """
    ephem = ephemeris or Ephemeris(kernel)
    begin = to_datetime(start)
    finish = to_datetime(end) if end is not None else add_years(begin, years)
    if finish <= begin:
        raise ValueError('end of range must be after the start')

    ts = ephem.timescale
    tt_start = ts.from_datetime(begin).tt
    tt_end = ts.from_datetime(finish).tt
    low, high = ephem.coverage()
    if tt_start < low or tt_end > high:
        raise ValueError(
            'requested range %s..%s falls outside the ephemeris, which covers '
            'JD %.1f..%.1f' % (begin.date(), finish.date(), low, high))

    events = finder.find_events(ephem, tt_start, tt_end)
    results = []
    for index, event in enumerate(events):
        if progress:
            progress(index, len(events))
        results.append(ec.analyse(ephem, event, threshold=threshold,
                                  samples=samples, trace_limits=trace))
    return results
