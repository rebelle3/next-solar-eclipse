#!/usr/bin/env python3
"""Compare a century of computed eclipses against the NASA/Espenak catalogue.

Every solar eclipse for 100 years is found, analysed and checked field by
field against ``tests/data/nasa_catalog_2001_2200.json``, which was scraped
from the Five Millennium Catalog.  Run as::

    python3 tests/verify_century.py [years] [kernel]
"""

import json
import os
import re
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import eclipse as ec  # noqa: E402
from eclipsepath import finder, geometry as g  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402

TOLERANCE = {'time_s': 2.0, 'gamma': 0.001, 'magnitude': 0.001,
             'latitude': 0.7, 'longitude': 0.7, 'duration_s': 2.0,
             'width_km': 0.05}      # width as a fraction, see the note below


def parse_position(text):
    if not text:
        return None
    value = float(text[:-1])
    return -value if text[-1] in 'SW' else value


def parse_duration(text):
    if not text:
        return None
    match = re.fullmatch(r'(\d+)m(\d+)s', text)
    return int(match.group(1)) * 60 + int(match.group(2)) if match else None


def main():
    years = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    ephem = Ephemeris(sys.argv[2] if len(sys.argv) > 2 else
                      os.environ.get('ECLIPSEPATH_KERNEL'))
    g.set_lunar_radius('espenak')
    ts = ephem.timescale
    with open(os.path.join(HERE, 'data', 'nasa_catalog_2001_2200.json')) as handle:
        catalogue = {row['date']: row for row in json.load(handle)}

    start = ts.utc(2026, 8, 13)
    stop = ts.utc(2026 + int(years), 8, 13)
    began = time.time()
    events = finder.find_events(ephem, start.tt, stop.tt)
    print('found %d eclipses in %.0f years (%.1f s)'
          % (len(events), years, time.time() - began))

    expected = {d: r for d, r in catalogue.items()
                if '2026-08-13' <= d <= '%d-08-13' % (2026 + int(years))}
    print('catalogue lists %d over the same span\n' % len(expected))

    errors = {key: [] for key in
              ('time_s', 'gamma', 'magnitude', 'latitude', 'longitude',
               'duration_s', 'width_km')}
    kinds = {'match': 0, 'differ': []}
    missing, extra, checked = [], [], 0
    began = time.time()
    for index, event in enumerate(events):
        date = ts.tt_jd(event.tt_greatest).utc_strftime('%Y-%m-%d')
        row = expected.get(date)
        if row is None:                       # a day's slip either way is fine
            for offset in (-1, 1):
                other = ts.tt_jd(event.tt_greatest + offset).utc_strftime('%Y-%m-%d')
                if other in expected:
                    row, date = expected[other], other
                    break
        if row is None:
            extra.append(date)
            continue
        result = ec.analyse(ephem, event, samples=3)
        checked += 1
        if result.kind == row['type']:
            kinds['match'] += 1
        else:
            kinds['differ'].append((date, result.kind, row['type']))
        year, month, day = (int(v) for v in row['date'].split('-'))
        hour, minute, second = (int(v) for v in row['td'].split(':'))
        errors['time_s'].append(
            (abs(event.tt_greatest - ts.tt(year, month, day, hour, minute, second).tt)
             * 86400.0, date))
        errors['gamma'].append((abs(result.gamma - row['gamma']), date))
        errors['magnitude'].append((abs(result.magnitude - row['mag']), date))
        lat, lon = parse_position(row['lat']), parse_position(row['lon'])
        if lat is not None:
            errors['latitude'].append((abs(result.latitude - lat), date))
            errors['longitude'].append(
                (abs((result.longitude - lon + 180) % 360 - 180), date))
        duration = parse_duration(row['dur'])
        if duration:
            errors['duration_s'].append(
                (abs(result.central_duration_seconds - duration), date))
        if row['width'] and result.path_width_km == result.path_width_km:
            errors['width_km'].append(
                (abs(result.path_width_km - row['width']) / row['width'], date))
        if index % 25 == 0:
            print('  %3d/%d  %s' % (index, len(events), date), flush=True)
    for date in expected:
        if not any(abs(ts.tt_jd(e.tt_greatest).tt
                       - ts.utc(*(int(v) for v in date.split('-'))).tt) < 1.5
                   for e in events):
            missing.append(date)

    print('\nanalysed %d eclipses in %.0f s\n' % (checked, time.time() - began))
    print('%-14s %8s %10s %10s   %s' % ('quantity', 'n', 'mean', 'worst', 'worst case'))
    failures = []
    for key, limit in TOLERANCE.items():
        values = errors[key]
        if not values:
            continue
        worst, where = max(values)
        mean = sum(v for v, _ in values) / len(values)
        unit = {'time_s': 's', 'duration_s': 's', 'width_km': ' (fraction)'}.get(key, '')
        ok = worst <= limit
        print('%-14s %8d %10.5f %10.5f%s  %s  %s'
              % (key, len(values), mean, worst, unit, where, 'ok' if ok else 'OVER'))
        if not ok:
            failures.append((key, worst, limit, where))
    print('\ntype agreement: %d matched, %d differed %s'
          % (kinds['match'], len(kinds['differ']), kinds['differ'][:5]))
    print('in catalogue but not found: %s' % (missing or 'none'))
    print('found but not in catalogue: %s' % (extra or 'none'))
    if failures or missing or extra or kinds['differ']:
        print('\nDISCREPANCIES:')
        for key, worst, limit, where in failures:
            print('  %s worst %.5f > %.5f at %s' % (key, worst, limit, where))
        return 1
    print('\nA century of eclipses agrees with the catalogue on every field.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
