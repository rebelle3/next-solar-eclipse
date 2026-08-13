#!/usr/bin/env python3
"""Capture a fingerprint of everything the library computes.

A refactor is meant to move code without changing answers, so this dumps the
answers to a file that can be diffed across the change.  Run it before, run it
after, compare:

    python3 tests/fingerprint.py before.json
    ...refactor...
    python3 tests/fingerprint.py after.json
    python3 tests/fingerprint.py --compare before.json after.json
"""

import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import eclipse as ec  # noqa: E402
from eclipsepath import finder, geometry as g, output  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402
from eclipsepath.observer import circumstances_at  # noqa: E402

# One of each kind, plus sites on and off the track.
CASES = [((2026, 8, 1), (2026, 8, 31), 'total'),
         ((2026, 2, 1), (2026, 2, 28), 'annular'),
         ((2031, 11, 1), (2031, 11, 30), 'hybrid'),
         ((2029, 1, 1), (2029, 1, 31), 'partial')]
SITES = [('cardiff', 51.4816, -3.1791), ('luxor', 25.687, 32.640),
         ('reykjavik', 64.1466, -21.9426), ('sydney', -33.87, 151.21)]


def capture(kernel=None):
    ephem = Ephemeris(kernel)
    g.set_lunar_radius('espenak')
    g.set_refraction(34.0)
    ts = ephem.timescale
    out = {'constants': {'r_sun_km': g.R_SUN_KM, 'k_moon': g.K_MOON,
                         'earth_a_km': g.EARTH_A_KM, 'earth_f': g.EARTH_F,
                         'horizon_deg': g.HORIZON_ALTITUDE_DEG},
           'eclipses': []}
    for (start, end, name) in CASES:
        event = finder.find_events(ephem, ts.utc(*start).tt, ts.utc(*end).tt)[0]
        result = ec.analyse(ephem, event, threshold=1.0, samples=24)
        entry = {'name': name, 'detail': output.eclipse_detail(result, ts)}
        entry['raw'] = {
            'tt_greatest': result.tt_greatest,
            'tt_first_contact': result.tt_first_contact,
            'tt_last_contact': result.tt_last_contact,
            'gamma': result.gamma, 'magnitude': result.magnitude,
            'obscuration': result.obscuration,
            'peak_obscuration': result.peak_obscuration,
            'latitude': result.latitude, 'longitude': result.longitude,
            'sun_altitude': result.sun_altitude,
            'path_width_km': result.path_width_km,
            'central_duration_seconds': result.central_duration_seconds,
            'kind': result.kind, 'saros': result.saros,
            'lunation': result.lunation,
            'tt_central_start': result.tt_central_start,
            'tt_central_end': result.tt_central_end,
        }
        entry['track'] = [[p.tt, p.latitude, p.longitude, p.obscuration,
                           p.sun_altitude, p.central] for p in result.path]
        band = result.central_band
        entry['band'] = None if not band else [
            [p.tt, p.latitude, p.longitude, p.north_latitude, p.north_longitude,
             p.south_latitude, p.south_longitude, p.width_km,
             p.duration_seconds, p.tt_start, p.tt_end, p.transverse]
            for p in band.points]
        region = ec.coverage_region(result, 0.6, rays=60)
        entry['region_from_analyse'] = None if not result.coverage_region else [
            [p.azimuth, p.distance_km, p.latitude, p.longitude]
            for p in result.coverage_region.points]
        entry['region'] = None if region is None else {
            'centre': [region.centre_latitude, region.centre_longitude],
            'encloses_pole': region.encloses_pole,
            'points': [[p.azimuth, p.distance_km, p.latitude, p.longitude,
                        p.tt, p.sun_altitude] for p in region.points]}
        entry['observers'] = {}
        for site, lat, lon in SITES:
            seen = circumstances_at(result, lat, lon)
            entry['observers'][site] = None if seen is None else {
                key: seen[key] for key in sorted(seen)}
        out['eclipses'].append(entry)
    return out


def flatten(value, prefix=''):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flatten(value[key], '%s.%s' % (prefix, key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from flatten(item, '%s[%d]' % (prefix, i))
    else:
        yield prefix, value


def compare(before, after, tolerance=0.0):
    left = dict(flatten(json.load(open(before))))
    right = dict(flatten(json.load(open(after))))
    keys = set(left) | set(right)
    worst, differences = 0.0, []
    for key in sorted(keys):
        a, b = left.get(key, '<missing>'), right.get(key, '<missing>')
        if isinstance(a, float) and isinstance(b, float):
            if a != a and b != b:            # NaN on both sides
                continue
            delta = abs(a - b)
            scale = max(1.0, abs(a))
            if delta / scale > tolerance:
                differences.append((key, a, b, delta))
            worst = max(worst, delta / scale)
        elif a != b:
            differences.append((key, a, b, None))
    print('compared %d values' % len(keys))
    print('worst relative difference: %.3e' % worst)
    if not differences:
        print('IDENTICAL — the refactor changed no answers')
        return 0
    print('%d differences:' % len(differences))
    for key, a, b, delta in differences[:25]:
        print('   %-58s %s -> %s' % (key[:58], a, b))
    return 1


def main():
    if sys.argv[1:2] == ['--compare']:
        return compare(sys.argv[2], sys.argv[3],
                       float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
    target = sys.argv[1] if len(sys.argv) > 1 else 'fingerprint.json'
    data = capture(os.environ.get('ECLIPSEPATH_KERNEL'))
    with open(target, 'w') as handle:
        json.dump(data, handle, indent=1, sort_keys=True)
    count = sum(1 for _ in flatten(data))
    print('wrote %s — %d values across %d eclipses'
          % (target, count, len(data['eclipses'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
