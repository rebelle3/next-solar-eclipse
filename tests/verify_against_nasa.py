#!/usr/bin/env python3
"""Compare computed eclipse paths against NASA's published path tables.

Run as ``python3 tests/verify_against_nasa.py``.  The reference tables in
``tests/data`` were scraped from eclipse.gsfc.nasa.gov and are compared row by
row: central line position, northern and southern limits, path width, central
duration, Sun altitude and the Moon/Sun diameter ratio.

Two details matter when comparing:

* NASA's tables are stamped in UT and assume their own value of delta-T, which
  differs from the IERS value this package uses.  Times are therefore matched
  in TT, and longitudes are corrected for the residual difference in Earth
  rotation angle.
* The northern/southern limit entries on a given table row are points on the
  limit *curves* at that instant, not the ends of a perpendicular cut, so they
  are compared against the whole curve rather than point to point.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eclipsepath import circumstances as cc  # noqa: E402
from eclipsepath import eclipse as ec  # noqa: E402
from eclipsepath import finder, geometry as g  # noqa: E402
from eclipsepath.ephemeris import Ephemeris, EclipseWindow  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ROTATION_DEG_PER_SEC = 360.98564736629 / 86400.0

CASES = [
    {'file': 'nasa_path_2026.json', 'date': (2026, 8, 12), 'kind': 'T',
     'delta_t': 71.4, 'label': '2026 Aug 12 total'},
    {'file': 'nasa_path_2026feb.json', 'date': (2026, 2, 17), 'kind': 'A',
     'delta_t': 74.7, 'label': '2026 Feb 17 annular'},
    # A near-grazing total: the Sun is only about 11 degrees up along a path
    # over 750 km wide, which is where a tangent-plane width is least reliable.
    {'file': 'nasa_path_2033.json', 'date': (2033, 3, 30), 'kind': 'T',
     'delta_t': 78.9, 'label': '2033 Mar 30 total (grazing)'},
]


def julian_day(year, month, day, hour, minute):
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = (day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400
           - 32045)
    return jdn - 0.5 + (hour + minute / 60.0) / 24.0


def densify(lat, lon, spacing_km=0.25):
    """Resample a lat/lon polyline along great circles to a target spacing.

    Two errors are being avoided.  Interpolating linearly in latitude and
    longitude would cut a large chord wherever the line is strongly bent, which
    near a pole dwarfs anything measured against the result.  And because
    :func:`distance_to_polyline` only sees the sample points, a coarse spacing
    ``s`` inflates a true distance ``d`` to roughly ``hypot(d, s/2)``, so the
    spacing has to stay well below the distances of interest.
    """
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    if lat.size < 2:
        return lat, lon
    unit_xyz = g.unit(g.geodetic_to_itrf(lat, lon))
    out_lat, out_lon = [], []
    for i in range(lat.size - 1):
        a, b = unit_xyz[:, i], unit_xyz[:, i + 1]
        angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        steps = max(2, int(np.ceil(angle * g.EARTH_A_KM / spacing_km)) + 1)
        fraction = np.linspace(0.0, 1.0, steps)
        if angle < 1e-12:
            points = np.outer(a, np.ones(steps))
        else:
            points = (np.sin((1.0 - fraction) * angle) * a[:, None]
                      + np.sin(fraction * angle) * b[:, None]) / np.sin(angle)
        piece_lat, piece_lon, _ = g.itrf_to_geodetic(points * g.EARTH_A_KM)
        out_lat.append(piece_lat)
        out_lon.append(piece_lon)
    return np.concatenate(out_lat), np.concatenate(out_lon)


def distance_to_polyline(lat, lon, curve_lat, curve_lon):
    """Shortest distance from a point to an already densified polyline."""
    return float(g.geodesic_distance(lat, lon, curve_lat, curve_lon).min())


def polyline_distance_km(lat, lon, curve):
    """Shortest distance from a point to an already densified polyline."""
    return distance_to_polyline(lat, lon, curve[0], curve[1])


def run_case(ephem, case, verbose=True):
    rows = json.load(open(os.path.join(DATA, case['file'])))
    year, month, day = case['date']
    ts = ephem.timescale

    tt = np.array([julian_day(year, month, day, int(r['ut'][:2]), int(r['ut'][3:]))
                   + case['delta_t'] / 86400.0 for r in rows])
    delta_t_here = np.array([(ephem.time(x).tt - ephem.time(x).ut1) * 86400.0
                             for x in tt])
    lon_fix = (case['delta_t'] - delta_t_here) * ROTATION_DEG_PER_SEC

    event = finder.find_events(ephem, ts.utc(year, month, day - 1).tt,
                               ts.utc(year, month, day + 1).tt)[0]
    window = EclipseWindow(ephem, event.tt_first_contact - 0.01,
                           event.tt_last_contact + 0.01)
    coarse = np.linspace(event.tt_first_contact, event.tt_last_contact,
                         ec.PREDICATE_TIME_SAMPLES)

    point, hit = ec.shadow_point(window, tt)
    lat, lon, _ = g.itrf_to_geodetic(point)
    state = cc.state_at(window, point, tt)
    band = ec._trace_band(window, coarse, tt, lat, lon, cc.central_depth, 'central')
    by_index = {}
    inside = cc.reaches(window, coarse, lat, lon, cc.central_depth)
    for point_index, sample in zip(np.nonzero(inside)[0], band.points):
        by_index[int(point_index)] = sample

    # NASA samples its limit curves every couple of minutes; resampling along
    # great circles keeps a point-to-curve distance meaningful near a pole.
    ref_n = densify([r['n_lat'] for r in rows], [r['n_lon'] for r in rows])
    ref_s = densify([r['s_lat'] for r in rows], [r['s_lon'] for r in rows])
    ref_c = densify([r['c_lat'] for r in rows], [r['c_lon'] for r in rows])

    stats = {'central': [], 'north': [], 'south': [], 'width': [], 'duration': [],
             'ratio': [], 'altitude': []}
    if verbose:
        print(f"\n=== {case['label']} ===")
        print('  UT     centre   north   south |  width  table  vs.curve | '
              ' durat.   NASA  diff | ratio  alt')
    for i, row in enumerate(rows):
        if i not in by_index or not hit[i]:
            continue
        sample = by_index[i]
        polar = abs(row['c_lat']) > 84.0        # NASA's 2-min polyline is too
        edge = i < 2 or i > len(rows) - 3       # coarse near a pole or the ends
        d_c = polyline_distance_km(lat[i], lon[i] + lon_fix[i], ref_c)
        d_n = polyline_distance_km(sample.north_latitude,
                                   sample.north_longitude + lon_fix[i], ref_n)
        d_s = polyline_distance_km(sample.south_latitude,
                                   sample.south_longitude + lon_fix[i], ref_s)
        # Width is checked against the distance between NASA's own limit curves
        # rather than their tabulated figure: the tabulated width comes from a
        # tangent-plane Besselian formula that under-reports a very wide band
        # seen at low Sun altitude, by more than 20 km for the 2026 annular.
        crossing = polyline_distance_km(sample.north_latitude,
                                        sample.north_longitude + lon_fix[i], ref_s)
        d_w = sample.width_km - crossing
        d_d = sample.duration_seconds - row['dur']
        if not (polar or edge):
            stats['central'].append(d_c)
            stats['north'].append(d_n)
            stats['south'].append(d_s)
            stats['width'].append(d_w)
            stats['duration'].append(d_d)
        stats['ratio'].append(state['magnitude'][i] - row['ratio'])
        stats['altitude'].append(state['sun_altitude'][i] - row['alt'])
        if verbose:
            flag = ' (polar/edge: excluded)' if (polar or edge) else ''
            print('%s %8.2f %7.2f %7.2f | %6.1f %6d %+6.1f | %7.1f %6.1f %+5.1f |'
                  ' %+.4f %+5.1f%s'
                  % (row['ut'], d_c, d_n, d_s, sample.width_km, row['width'], d_w,
                     sample.duration_seconds, row['dur'], d_d,
                     state['magnitude'][i] - row['ratio'],
                     state['sun_altitude'][i] - row['alt'], flag))
    return stats


def main():
    kernel = sys.argv[1] if len(sys.argv) > 1 else None
    ephem = Ephemeris(kernel)
    g.R_MOON_KM = 0.2722810 * g.EARTH_A_KM
    failures = []
    for case in CASES:
        stats = run_case(ephem, case)
        print('\n  summary (polar and end rows excluded):')
        for key, limit, unit in (('central', 2.0, 'km'), ('north', 3.0, 'km'),
                                 ('south', 3.0, 'km'), ('width', 7.0, 'km'),
                                 ('duration', 1.0, 's'), ('ratio', 0.001, ''),
                                 ('altitude', 0.6, 'deg')):
            values = np.abs(np.array(stats[key]))
            if not values.size:
                continue
            worst = values.max()
            ok = worst <= limit
            print('    %-9s mean %8.3f  max %8.3f %-3s  limit %6.3f  %s'
                  % (key, values.mean(), worst, unit, limit, 'ok' if ok else 'FAIL'))
            if not ok:
                failures.append((case['label'], key, worst, limit))
    print()
    if failures:
        for label, key, worst, limit in failures:
            print('FAIL %s %s: %.3f > %.3f' % (label, key, worst, limit))
        return 1
    print('All NASA path-table comparisons within tolerance.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
