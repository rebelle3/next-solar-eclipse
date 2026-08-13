#!/usr/bin/env python3
"""Cross-check coverage against a deliberately naive second implementation.

The library computes obscuration through Chebyshev-interpolated geocentric
vectors, its own rotation handling and refined peak-finding.  This recomputes
it the slow obvious way — Skyfield's own topocentric ``observe().apparent()``
for the Sun and Moon at every step of a fine time grid — and compares.  The two
share only the ephemeris file and the circle-overlap formula, so agreement is
evidence about the pipeline rather than a restatement of it.
"""

import math
import os
import sys

import numpy as np
from skyfield.api import wgs84

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import finder, geometry as g  # noqa: E402
from eclipsepath import eclipse as ec  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402
from eclipsepath.observer import circumstances_at  # noqa: E402

R_SUN_KM = 696000.0
STEP_SECONDS = 2.0


def brute_force(ephem, latitude, longitude, tt_from, tt_to):
    """Peak obscuration the slow way, straight from the kernel."""
    site = ephem.earth + wgs84.latlon(latitude, longitude)
    ts = ephem.timescale
    count = int((tt_to - tt_from) * 86400.0 / STEP_SECONDS) + 1
    best, best_tt = 0.0, float('nan')
    for chunk in range(0, count, 4000):
        tt = tt_from + (np.arange(chunk, min(chunk + 4000, count))
                        * STEP_SECONDS / 86400.0)
        t = ts.tt_jd(tt)
        here = site.at(t)
        sun = here.observe(ephem.sun).apparent()
        moon = here.observe(ephem.moon).apparent()
        altitude = sun.altaz()[0].degrees
        r_sun = np.arcsin(R_SUN_KM / sun.distance().km)
        r_moon = np.arcsin(g.R_MOON_KM / moon.distance().km)
        separation = sun.separation_from(moon).radians
        covered = g.obscuration(r_sun, r_moon, separation)
        covered = np.where(altitude >= g.HORIZON_ALTITUDE_DEG, covered, 0.0)
        index = int(np.argmax(covered))
        if covered[index] > best:
            best, best_tt = float(covered[index]), float(tt[index])
    return best, best_tt


def main():
    ephem = Ephemeris(os.environ.get('ECLIPSEPATH_KERNEL'))
    g.set_lunar_radius('espenak')
    ts = ephem.timescale
    rng = np.random.default_rng(20270802)

    windows = [((2026, 8, 1), (2026, 8, 31)), ((2027, 7, 15), (2027, 8, 15)),
               ((2026, 2, 1), (2026, 2, 28)), ((2029, 1, 1), (2029, 1, 31)),
               ((2031, 11, 1), (2031, 11, 30))]
    errors, timing, tested = [], [], 0
    for start, end in windows:
        event = finder.find_events(ephem, ts.utc(*start).tt, ts.utc(*end).tt)[0]
        result = ec.analyse(ephem, event, samples=40)
        date = ts.tt_jd(result.tt_greatest).utc_strftime('%Y-%m-%d')
        # Sites spread along and across the track, where coverage varies most.
        sites = []
        for point in result.path[::7]:
            for offset in (0.0, 3.0, -3.0, 9.0):
                sites.append((point.latitude + offset * rng.uniform(0.6, 1.4),
                              point.longitude + offset * rng.uniform(-1.4, 1.4)))
        sites = [(lat, lon) for lat, lon in sites if -89.0 < lat < 89.0][:14]
        for latitude, longitude in sites:
            seen = circumstances_at(result, latitude, longitude)
            mine = 0.0 if seen is None else seen['obscuration']
            theirs, when = brute_force(ephem, latitude, longitude,
                                       result.tt_first_contact,
                                       result.tt_last_contact)
            errors.append((abs(mine - theirs), date, latitude, longitude,
                           mine, theirs))
            # Skip the time comparison inside totality: coverage is flat at 1
            # right across it, so every instant ties and the two
            # implementations break that tie differently by design.
            if (seen is not None and 0.01 < theirs < 0.999 and when == when):
                timing.append((abs(seen['tt_maximum'] - when) * 86400.0, date))
            tested += 1
        print('  %s: %d sites' % (date, len(sites)), flush=True)

    worst, date, lat, lon, mine, theirs = max(errors)
    mean = sum(e[0] for e in errors) / len(errors)
    print('\n%d site/eclipse pairs against the brute-force implementation' % tested)
    print('  obscuration  mean %.3e  worst %.3e' % (mean, worst))
    print('               worst at %s %.2f,%.2f: %.6f vs %.6f'
          % (date, lat, lon, mine, theirs))
    if timing:
        t_worst, t_date = max(timing)
        print('  time of max  mean %.2f s  worst %.2f s (%s)'
              % (sum(t[0] for t in timing) / len(timing), t_worst, t_date))
    ok = worst < 0.002
    print('\n%s' % ('Two independent implementations agree.' if ok
                    else 'DISAGREEMENT beyond tolerance.'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
