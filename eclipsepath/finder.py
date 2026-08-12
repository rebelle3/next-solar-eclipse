"""Scanning a date range for solar eclipses.

Two passes.  A coarse pass walks the whole range and keeps the new-moon
neighbourhoods where the shadow axis comes anywhere near the Earth; a fine pass
then resolves, for each of those, whether the penumbra actually touches the
globe and when it first and last does so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as g

COARSE_STEP_DAYS = 3.0 / 24.0
COARSE_GAMMA_LIMIT = 4.0     # Earth radii; ~2.6 is the worst a 3 h grid can miss
FINE_STEP_DAYS = 1.0 / 1440.0
FINE_PAD_DAYS = 3.0 / 24.0
CHUNK_DAYS = 366.0


@dataclass
class RawEvent:
    """A bracketed eclipse before any detailed geometry is worked out."""
    tt_first_contact: float
    tt_last_contact: float
    tt_greatest: float


def axis_metrics(sun, moon, north):
    """Shadow-axis quantities: (miss distance km, signed gamma, axis unit vector)."""
    u = g.shadow_axis(sun, moon)
    c = moon - u * g.dot(moon, u)
    miss = g.norm(c)
    east = g.unit(np.cross(north, u, axis=0))
    up = np.cross(u, east, axis=0)
    gamma = np.sign(g.dot(c, up)) * miss / g.EARTH_A_KM
    return miss, gamma, u


def _penumbra_clearance(sun, moon):
    """Negative while the penumbral cone overlaps the globe."""
    return g.axis_earth_distance(sun, moon)


def _scan(ephem, tt):
    t = ephem.time(tt)
    sun, moon = ephem.sun_moon(t)
    return sun, moon


def coarse_candidates(ephem, tt_start, tt_end):
    """New-moon neighbourhoods worth examining closely."""
    hits = []
    start = tt_start
    while start < tt_end:
        stop = min(start + CHUNK_DAYS, tt_end)
        tt = np.arange(start, stop + COARSE_STEP_DAYS, COARSE_STEP_DAYS)
        sun, moon = _scan(ephem, tt)
        u = g.shadow_axis(sun, moon)
        along = -g.dot(moon, u)
        miss = g.norm(moon + u * along)
        near = (miss / g.EARTH_A_KM < COARSE_GAMMA_LIMIT) & (along > 0.0)
        hits.append(tt[near])
        start = stop
    if not hits:
        return []
    flagged = np.concatenate(hits)
    if flagged.size == 0:
        return []
    # Group samples belonging to the same lunation.
    breaks = np.nonzero(np.diff(flagged) > 2.0 * COARSE_STEP_DAYS)[0]
    groups = np.split(flagged, breaks + 1)
    return [(float(grp[0]) - FINE_PAD_DAYS, float(grp[-1]) + FINE_PAD_DAYS)
            for grp in groups]


def refine_event(ephem, tt_lo, tt_hi):
    """Resolve one candidate window into a :class:`RawEvent`, or ``None``."""
    tt = np.arange(tt_lo, tt_hi, FINE_STEP_DAYS)
    sun, moon = _scan(ephem, tt)
    clearance = _penumbra_clearance(sun, moon)
    inside = clearance < 0.0
    if not inside.any():
        return None
    first, last = int(np.argmax(inside)), len(inside) - 1 - int(np.argmax(inside[::-1]))

    def clearance_at(x):
        s, m = _scan(ephem, np.atleast_1d(x))
        return _penumbra_clearance(s, m)

    p1 = _bisect_scalar(clearance_at, tt[max(first - 1, 0)], tt[first])
    p4 = _bisect_scalar(clearance_at, tt[min(last + 1, len(tt) - 1)], tt[last])

    # Greatest eclipse: minimum axis-to-centre distance.
    def miss_at(x):
        s, m = _scan(ephem, np.atleast_1d(x))
        u = g.shadow_axis(s, m)
        return g.norm(m - u * g.dot(m, u))

    span = tt[first:last + 1]
    coarse = miss_at(span)
    k = int(np.argmin(coarse))
    lo = span[max(k - 1, 0)]
    hi = span[min(k + 1, len(span) - 1)]
    greatest = _golden_scalar(miss_at, lo, hi)
    return RawEvent(p1, p4, greatest)


def _bisect_scalar(func, lo, hi, iterations=60):
    f_lo = float(np.ravel(func(lo))[0])
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = float(np.ravel(func(mid))[0])
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _golden_scalar(func, lo, hi, iterations=60):
    phi = 0.6180339887498949
    x1, x2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
    scalar = lambda x: float(np.ravel(func(x))[0])
    f1, f2 = scalar(x1), scalar(x2)
    for _ in range(iterations):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - phi * (hi - lo)
            f1 = scalar(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + phi * (hi - lo)
            f2 = scalar(x2)
    return 0.5 * (lo + hi)


def find_events(ephem, tt_start, tt_end):
    """All solar eclipses whose global maximum falls inside the range."""
    events = []
    for lo, hi in coarse_candidates(ephem, tt_start, tt_end):
        event = refine_event(ephem, lo, hi)
        if event is not None and tt_start <= event.tt_greatest <= tt_end:
            events.append(event)
    return events
