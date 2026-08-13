#!/usr/bin/env python3
"""Compare local circumstances against the US Naval Observatory.

Every other check in this directory compares against NASA's *catalogue* — the
global numbers for an eclipse — or against a second implementation written
here.  This one asks a third party what a specific place on Earth sees, which
is what the library is actually for.

The USNO Astronomical Applications API returns obscuration, magnitude, contact
times and Sun altitude for any site, computed from its own Besselian-element
code.  It shares nothing with this package but the underlying astronomy, and
it is US Government work, so quoting it here carries no licence conditions.

Its tables run out after 2026, which is why the eclipses checked are past ones.

    python3 tests/verify_usno.py --fetch    # query the API, fill the cache
    python3 tests/verify_usno.py            # compare, offline

Responses are cached in tests/data/usno_local_circumstances.json so the
comparison is reproducible without the network.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import finder, geometry as g  # noqa: E402
from eclipsepath import eclipse as ec  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402
from eclipsepath.observer import circumstances_at  # noqa: E402

CACHE = os.path.join(HERE, 'data', 'usno_local_circumstances.json')
API = 'https://aa.usno.navy.mil/api/eclipses/solar/date'
FIRST_YEAR, LAST_YEAR = 2015, 2026     # the span the API covers
SITES_PER_ECLIPSE = 14
GRID_DEG = 10.0
PAUSE_SECONDS = 0.25

# What USNO rounds to, and so the floor on any comparison against it.
OBSCURATION_STEP = 0.05        # they print 0.1%
MAGNITUDE_STEP = 0.0005        # they print 0.001
TIME_STEP = 0.05               # they print 0.1s

# Central duration is the one figure where the two authorities themselves
# disagree: USNO's totalities run about two seconds longer than NASA's and its
# annularities about two seconds shorter, the signature of a slightly larger
# lunar radius in the umbral cone.  The limit here is set wide enough to admit
# that gap; the tight limit is on the three-way check below, which pins this
# package to NASA rather than splitting the difference.
TOLERANCE = {'obscuration_pct': 0.25, 'magnitude': 0.003,
             'c1_seconds': 6.0, 'c4_seconds': 6.0, 'maximum_seconds': 30.0,
             'central_seconds': 6.0, 'altitude': 0.2}
THREE_WAY_TOLERANCE = 0.5      # seconds, this package against NASA


def fetch(date, latitude, longitude):
    url = ('%s?date=%s&coords=%.4f,%.4f&height=0'
           % (API, date, latitude, longitude))
    with urllib.request.urlopen(url, timeout=60) as handle:
        return json.loads(handle.read().decode())


def parse_clock(text):
    """"17:16:55.1" -> seconds since midnight.

    Sunrise and sunset are given to the minute, everything else to a tenth of
    a second.
    """
    parts = text.split(':')
    seconds = float(parts[2]) if len(parts) > 2 else 0.0
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + seconds


def parse_span(text):
    """"1h 08m 03.4s" or "1m 08.3s" -> seconds."""
    total = 0.0
    for value, unit in re.findall(r'([\d.]+)\s*([hms])', text):
        total += float(value) * {'h': 3600.0, 'm': 60.0, 's': 1.0}[unit]
    return total


def altitude_at(name, record):
    """The Sun's altitude USNO reports at a phenomenon, in degrees."""
    for entry in record['properties']['local_data']:
        if entry['phenomenon'].startswith(name) and 'altitude' in entry:
            return float(entry['altitude'])
    return None


def phenomena(record):
    """Phenomenon name -> (day, seconds since midnight UT)."""
    found = {}
    for entry in record['properties']['local_data']:
        found[entry['phenomenon']] = (int(entry['day']),
                                      parse_clock(entry['time']))
    return found


def pick(name, found):
    """The first phenomenon whose name starts with ``name``.

    USNO qualifies a phenomenon cut off by the horizon — "Eclipse Begins at
    Sunrise" and the like — so the comparison matches on the stem.
    """
    for key, value in found.items():
        if key.startswith(name):
            return key, value
    return None, None


def day_seconds(ut1_calendar, reference_day):
    """UT1 seconds since midnight, carried across a date boundary."""
    _, _, day, hour, minute, second = ut1_calendar
    return (day - reference_day) * 86400.0 + hour * 3600 + minute * 60 + second


def eclipses_in_span(ephem):
    ts = ephem.timescale
    events = finder.find_events(ephem, ts.utc(FIRST_YEAR, 1, 1).tt,
                                ts.utc(LAST_YEAR, 12, 31).tt)
    return events


def choose_sites(result):
    """Sites spanning the whole range of coverage this eclipse produces.

    A grid over the globe, kept where anything is visible, then sampled evenly
    by obscuration so the comparison is not all deep partials.
    """
    candidates = []
    for latitude in np.arange(-80.0, 80.1, GRID_DEG):
        for longitude in np.arange(-180.0, 180.0, GRID_DEG):
            seen = circumstances_at(result, float(latitude), float(longitude))
            if seen is not None and seen['obscuration'] > 0.02:
                candidates.append((seen['obscuration'], float(latitude),
                                   float(longitude)))
    if not candidates:
        return []
    candidates.sort()
    ranks = np.linspace(0, len(candidates) - 1, SITES_PER_ECLIPSE)
    sites = [candidates[int(round(r))][1:] for r in ranks]
    if result.path:
        # The grid rarely lands on a path only a couple of hundred km wide,
        # and the central phase is the part worth checking hardest.
        for point in (result.path[len(result.path) // 2],
                      result.path[len(result.path) // 4],
                      result.path[3 * len(result.path) // 4]):
            sites.append((round(point.latitude, 4), round(point.longitude, 4)))
    return list(dict.fromkeys((round(a, 4), round(b, 4)) for a, b in sites))


def key_for(date, latitude, longitude):
    return '%s|%.4f|%.4f' % (date, latitude, longitude)


def do_fetch(cache):
    ephem = Ephemeris(os.environ.get('ECLIPSEPATH_KERNEL'))
    g.set_lunar_radius('espenak')
    ts = ephem.timescale
    added = refused = 0
    for event in eclipses_in_span(ephem):
        result = ec.analyse(ephem, event, samples=40)
        date = ts.tt_jd(result.tt_greatest).utc_strftime('%Y-%m-%d')
        sites = choose_sites(result)
        for latitude, longitude in sites:
            key = key_for(date, latitude, longitude)
            if key in cache:
                continue
            try:
                record = fetch(date, latitude, longitude)
            except Exception as error:                 # noqa: BLE001
                print('  %s %s: %s' % (date, key, error))
                continue
            if 'error' in record:
                cache[key] = {'refused': record['error']}
                refused += 1
            else:
                cache[key] = record
                added += 1
            time.sleep(PAUSE_SECONDS)
        print('%s: %d sites (%d fetched, %d refused)'
              % (date, len(sites), added, refused), flush=True)
        with open(CACHE, 'w') as handle:
            json.dump(cache, handle, indent=0, sort_keys=True)
    print('cache holds %d responses' % len(cache))


def compare(cache, radius):
    g.set_lunar_radius(radius)
    by_date = {}
    for key, record in cache.items():
        if 'refused' in record:
            continue
        date, latitude, longitude = key.split('|')
        by_date.setdefault(date, []).append((float(latitude), float(longitude),
                                             record))

    rows, missing = [], 0
    for date in sorted(by_date):
        sites = by_date[date]
        delta_t = float(sites[0][2]['properties']['delta_t'].rstrip('s'))
        # Their delta-T, not ours: it decides how far the Earth had turned, and
        # comparing geometry means holding the timescale in common.
        ephem = Ephemeris(os.environ.get('ECLIPSEPATH_KERNEL'),
                          delta_t=delta_t)
        ts = ephem.timescale
        year, month, day = (int(v) for v in date.split('-'))
        event = finder.find_events(ephem, ts.utc(year, month, day - 1).tt,
                                   ts.utc(year, month, day + 1).tt)[0]
        result = ec.analyse(ephem, event, samples=40)

        for latitude, longitude, record in sites:
            mine = circumstances_at(result, latitude, longitude)
            if mine is None:
                missing += 1
                continue
            theirs = record['properties']
            found = phenomena(record)
            # Where the Sun rises or sets mid-eclipse the two are not
            # answering the same question at the ends: USNO stops at the
            # horizon and names the phenomenon, and the contact it reports is
            # the horizon crossing rather than the geometric contact.
            horizon = any(name.startswith(('Sunrise', 'Sunset'))
                          for name in found)
            row = {'date': date, 'latitude': latitude, 'longitude': longitude,
                   'delta_t': delta_t, 'horizon': horizon}
            row['obscuration_pct'] = (
                mine['obscuration'] * 100.0
                - float(theirs['obscuration'].rstrip('%')))
            central = mine['central_seconds'] > 0.0
            if not central:
                # Where the Moon covers the Sun entirely, "magnitude" is the
                # ratio of diameters by convention and the fraction of the
                # diameter covered by formula, which are different numbers.
                row['magnitude'] = mine['magnitude'] - float(
                    theirs['magnitude'])

            reference_day = min(d for d, _ in found.values())

            def offset(name, tt):
                _, entry = pick(name, found)
                if entry is None or tt != tt:
                    return None
                theirs_seconds = (entry[0] - reference_day) * 86400.0 + entry[1]
                ours = day_seconds(ts.tt_jd(tt).ut1_calendar(), reference_day)
                return ours - theirs_seconds

            if not horizon:
                row['c1_seconds'] = offset('Eclipse Begins',
                                           mine['tt_first_contact'])
                row['c4_seconds'] = offset('Eclipse Ends',
                                           mine['tt_last_contact'])
            if not central and not horizon:
                # Coverage is flat right through the central phase, annular as
                # much as total, so neither the instant of maximum nor the Sun
                # altitude at it is a quantity the two can be held to.
                row['maximum_seconds'] = offset('Maximum Eclipse',
                                                mine['tt_maximum'])
                altitude = altitude_at('Maximum Eclipse', record)
                if altitude is not None:
                    row['altitude'] = mine['sun_altitude'] - altitude
            for name in ('duration_of_totality', 'duration_of_annularity'):
                if name in theirs and central:
                    row['central_seconds'] = (mine['central_seconds']
                                              - parse_span(theirs[name]))
            rows.append(row)
    return rows, missing


def summarise(rows, field, floor, unit):
    values = [abs(r[field]) for r in rows if r.get(field) is not None]
    if not values:
        return None
    worst = max(values)
    where = max((r for r in rows if r.get(field) is not None),
                key=lambda r: abs(r[field]))
    return {'count': len(values), 'mean': sum(values) / len(values),
            'worst': worst, 'unit': unit, 'floor': floor,
            'where': '%s %.1f,%.1f' % (where['date'], where['latitude'],
                                       where['longitude'])}


def report(all_rows, missing, label):
    # Where the Sun rises or sets mid-eclipse the two report different things
    # on purpose: USNO quotes the eclipse's greatest magnitude whether or not
    # anyone could see it, and this package quotes the deepest phase actually
    # above the horizon.  Averaging the two together would say nothing about
    # either, so they are counted apart.
    rows = [r for r in all_rows if not r.get('horizon')]
    clipped = [r for r in all_rows if r.get('horizon')]
    print('\n%s' % label)
    print('  %d site/eclipse pairs over %d eclipses; %d more have the Sun '
          'rise or set mid-eclipse'
          % (len(rows), len({r['date'] for r in rows}), len(clipped)))
    if missing:
        print('  %d sites USNO reported but we found nothing at' % missing)
    lines = [('obscuration', 'obscuration_pct', OBSCURATION_STEP, '%'),
             ('magnitude', 'magnitude', MAGNITUDE_STEP, ''),
             ('Sun altitude', 'altitude', 0.05, ' deg'),
             ('first contact', 'c1_seconds', TIME_STEP, ' s'),
             ('maximum', 'maximum_seconds', TIME_STEP, ' s'),
             ('last contact', 'c4_seconds', TIME_STEP, ' s'),
             ('central duration', 'central_seconds', TIME_STEP, ' s')]
    worst_ratio = 0.0
    for name, field, floor, unit in lines:
        stats = summarise(rows, field, floor, unit)
        if stats is None:
            continue
        limit = TOLERANCE.get(field)
        print('  %-17s n=%-4d mean %7.3f%s  worst %7.3f%s  (%s)'
              % (name, stats['count'], stats['mean'], unit, stats['worst'],
                 unit, stats['where']))
        if limit:
            worst_ratio = max(worst_ratio, stats['worst'] / limit)
    if clipped:
        deficits = [r['obscuration_pct'] for r in clipped]
        below = sum(1 for d in deficits if d < -0.05)
        print('  of the %d Sun-rises-or-sets sites, %d see less than USNO '
              'quotes (deepest %.1f%% short): the eclipse goes on after '
              'sunset there' % (len(clipped), below, min(deficits)))
    return worst_ratio


def three_way(cache, rows):
    """Where the two authorities disagree, which one is this package with?

    USNO and NASA are independent of each other as well as of this package, so
    the central duration at a site can be asked of all three.  Reading the
    residual against USNO alone would put a two-second error at this package's
    door that belongs to the gap between the two sources.
    """
    elements_path = os.path.join(HERE, 'data', 'nasa_besselian_elements.json')
    if not os.path.exists(elements_path):
        return 0.0
    sys.path.insert(0, HERE)
    from verify_besselian import Elements, circumstances   # noqa: PLC0415
    with open(elements_path) as handle:
        catalogue = json.load(handle)

    against_nasa, usno_against_nasa = [], []
    for row in rows:
        if 'central_seconds' not in row:
            continue
        stamp = row['date'].replace('-', '')
        if stamp not in catalogue:
            continue
        nasa = circumstances(Elements(catalogue[stamp]), row['latitude'],
                             row['longitude'])
        if nasa is None or nasa['central_seconds'] <= 0.0:
            continue
        key = key_for(row['date'], row['latitude'], row['longitude'])
        theirs = cache[key]['properties']
        usno = next(parse_span(theirs[name]) for name in
                    ('duration_of_totality', 'duration_of_annularity')
                    if name in theirs)
        mine = usno + row['central_seconds']
        against_nasa.append(mine - nasa['central_seconds'])
        usno_against_nasa.append(usno - nasa['central_seconds'])

    if not against_nasa:
        return 0.0
    def mean(values):
        return sum(abs(v) for v in values) / len(values)
    print('\n  central duration, all three sources at %d central sites:'
          % len(against_nasa))
    print('    this package vs NASA   mean %.3f s  worst %.2f s'
          % (mean(against_nasa), max(abs(v) for v in against_nasa)))
    print('    USNO         vs NASA   mean %.3f s  worst %.2f s'
          % (mean(usno_against_nasa), max(abs(v) for v in usno_against_nasa)))
    return mean(against_nasa) / THREE_WAY_TOLERANCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true',
                        help='query the USNO API and fill the cache')
    args = parser.parse_args()

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as handle:
            cache = json.load(handle)
    if args.fetch:
        do_fetch(cache)
        return 0
    if not cache:
        print('no cache; run with --fetch first')
        return 1

    refused = sum(1 for r in cache.values() if 'refused' in r)
    print('%d cached USNO responses (%d sites it says see nothing)'
          % (len(cache), refused))

    rows, missing = compare(cache, 'espenak')
    ratio = report(rows, missing, 'Against USNO, k = 0.272281 (Espenak):')
    ratio = max(ratio, three_way(cache, rows))
    other, _ = compare(cache, 'iau')
    report(other, 0, 'Against USNO, k = 0.2725076 (IAU mean):')
    g.set_lunar_radius('espenak')

    ok = ratio <= 1.0
    print('\n%s' % ('Local circumstances agree with the USNO.' if ok else
                    'DISAGREEMENT with the USNO beyond tolerance.'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
