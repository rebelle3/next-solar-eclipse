#!/usr/bin/env python3
"""Compare local circumstances against NASA's published Besselian elements.

The USNO check covers past eclipses; its tables stop after 2026, and this
package is for future ones.  This closes that gap using the other independent
prediction available for arbitrary future dates: the polynomial Besselian
elements Fred Espenak published for the Five Millennium Canon, scraped by
``tests/scrape_besselian.py``.

The elements come from the VSOP87/ELP2000-82 ephemerides, not JPL DE440s, and
carry NASA's own delta-T and lunar radius constants.  Reducing them to local
circumstances is the classical algorithm of the Explanatory Supplement, which
has nothing in common with this package's direct vector geometry beyond the
plane-geometry formula for the overlap of two discs.  So agreement tests the
ephemeris handling, the light-time treatment, the rotation model and the
geodesy all at once, against a source that shares none of them.

    python3 tests/verify_besselian.py

Note that NASA uses two lunar radii — k = 0.272488 for penumbral contacts and
0.272281 for umbral ones — where this package uses one.  The report separates
the two so the cost of that choice is visible rather than buried in an average.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import circumstances as cc  # noqa: E402
from eclipsepath import finder, geometry as g  # noqa: E402
from eclipsepath import eclipse as ec  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402
from eclipsepath.observer import circumstances_at  # noqa: E402

ELEMENTS = os.path.join(HERE, 'data', 'nasa_besselian_elements.json')
FLATTENING_RATIO = 0.99664719      # b/a, the value the elements assume
ROTATION_DEG_PER_SEC = 360.985647 / 86400.0
GRID_DEG = 12.0
SITES_PER_ECLIPSE = 12
HOUR = 1.0 / 24.0

# The binding figure for the time of maximum is what it costs, not the clock
# difference: away from the central phase the coverage curve is still flat
# enough near its top that a twenty-second disagreement changes the depth by
# less than a ten-thousandth.  Both are reported; the loose limit on the clock
# is deliberate, and the tight one on the cost is what would catch a real
# regression.
TOLERANCE = {'obscuration_pct': 0.1, 'magnitude': 0.002,
             'maximum_seconds': 30.0, 'maximum_cost_pct': 0.02,
             'central_seconds': 2.0,
             'c1_seconds': 8.0, 'c4_seconds': 8.0}

# Below this the Sun is close enough to the horizon that the two are not
# answering the same question: this package reports the deepest eclipse
# actually *visible*, and stops at sunset, where the plain reduction of the
# elements carries on below the horizon.
HORIZON_LIMIT_DEG = 5.0

# NASA's own two constants, and the single one this package uses.
K_VALUES = [('0.272281 (this package, Espenak umbral)', 0.2722810),
            ('0.272488 (NASA penumbral)', 0.2724880),
            ('0.2725076 (IAU mean)', 0.2725076)]


def julian_day(year, month, day):
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = (day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400
           - 32045)
    return jdn - 0.5


class Elements:
    """NASA's polynomial elements, evaluated at any instant."""

    def __init__(self, record):
        self.record = record
        self.t0 = record['t0_hours']
        self.tan_f1 = record['tan_f1']
        self.tan_f2 = record['tan_f2']
        # mu is the *ephemeris* hour angle: the angle Greenwich would be at if
        # the Earth had turned by the ephemeris time rather than by the clock.
        # A place on the ground is that much further east in that frame, and
        # leaving the term out puts every site a couple of hundred kilometres
        # from where it belongs.
        self.ephemeris_shift = record['delta_t'] * ROTATION_DEG_PER_SEC
        stamp = record['stamp']
        self.jd0 = julian_day(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))

    def hours_of(self, tt):
        return (tt - self.jd0) * 24.0

    def at(self, hours):
        """Every element and its rate of change, per hour."""
        t = hours - self.t0
        out = {}
        for name in ('x', 'y', 'd', 'l1', 'l2', 'mu'):
            c = self.record[name]
            out[name] = c[0] + c[1] * t + c[2] * t * t + c[3] * t ** 3
            out[name + "'"] = c[1] + 2 * c[2] * t + 3 * c[3] * t * t
        return out


def observer_terms(elements, hours, latitude, longitude):
    """The observer's place in the fundamental plane, and its motion."""
    e = elements.at(hours)
    d, mu = math.radians(e['d']), math.radians(e['mu'])
    d_rate, mu_rate = math.radians(e["d'"]), math.radians(e["mu'"])

    phi = math.radians(latitude)
    u = math.atan(FLATTENING_RATIO * math.tan(phi))
    rho_sin = FLATTENING_RATIO * math.sin(u)
    rho_cos = math.cos(u)

    theta = mu + math.radians(longitude - elements.ephemeris_shift)
    xi = rho_cos * math.sin(theta)
    eta = rho_sin * math.cos(d) - rho_cos * math.cos(theta) * math.sin(d)
    zeta = rho_sin * math.sin(d) + rho_cos * math.cos(theta) * math.cos(d)

    xi_rate = mu_rate * rho_cos * math.cos(theta)
    eta_rate = mu_rate * xi * math.sin(d) - zeta * d_rate

    return {
        'u': e['x'] - xi, 'v': e['y'] - eta,
        'a': e["x'"] - xi_rate, 'b': e["y'"] - eta_rate,
        'l1': e['l1'] - zeta * elements.tan_f1,
        'l2': e['l2'] - zeta * elements.tan_f2,
        'altitude': math.degrees(math.asin(max(-1.0, min(1.0,
            math.sin(phi) * math.sin(d)
            + math.cos(phi) * math.cos(d) * math.cos(theta))))),
    }


def maximum(elements, hours, latitude, longitude):
    """Iterate to the instant of greatest eclipse at one site."""
    for _ in range(12):
        term = observer_terms(elements, hours, latitude, longitude)
        n2 = term['a'] ** 2 + term['b'] ** 2
        step = -(term['u'] * term['a'] + term['v'] * term['b']) / n2
        hours += step
        if abs(step) < 1e-9:
            break
    term = observer_terms(elements, hours, latitude, longitude)
    term['hours'] = hours
    term['m'] = math.hypot(term['u'], term['v'])
    return term


def contact(elements, hours, latitude, longitude, radius, sign):
    """Iterate to a contact: ``radius`` is 'l1' or 'l2', ``sign`` -1 or +1."""
    for _ in range(20):
        term = observer_terms(elements, hours, latitude, longitude)
        n = math.hypot(term['a'], term['b'])
        delta = (term['u'] * term['b'] - term['v'] * term['a']) / n
        limit = abs(term[radius])
        inside = limit ** 2 - delta ** 2
        if inside <= 0.0:
            return None
        step = (-(term['u'] * term['a'] + term['v'] * term['b']) / (n * n)
                + sign * math.sqrt(inside) / n)
        hours += step
        if abs(step) < 1e-9:
            break
    return hours


def circumstances(elements, latitude, longitude):
    """Obscuration, magnitude and contacts at a site, from the elements."""
    peak = maximum(elements, elements.t0, latitude, longitude)
    total_radius = peak['l1'] + peak['l2']
    if total_radius <= 0.0:
        return None
    magnitude = (peak['l1'] - peak['m']) / total_radius
    if magnitude <= 0.0 or peak['altitude'] < -34.0 / 60.0:
        return None
    # The two discs, in units of the Sun's radius.
    moon = (peak['l1'] - peak['l2']) / total_radius
    separation = 2.0 * peak['m'] / total_radius
    result = {
        'hours_maximum': peak['hours'],
        'magnitude': magnitude,
        'obscuration': float(g.obscuration(1.0, moon, separation)),
        'altitude': peak['altitude'],
        'hours_c1': contact(elements, peak['hours'], latitude, longitude,
                            'l1', -1),
        'hours_c4': contact(elements, peak['hours'], latitude, longitude,
                            'l1', +1),
        'central_seconds': 0.0,
    }
    if peak['m'] < abs(peak['l2']):
        c2 = contact(elements, peak['hours'], latitude, longitude, 'l2', -1)
        c3 = contact(elements, peak['hours'], latitude, longitude, 'l2', +1)
        if c2 is not None and c3 is not None:
            result['central_seconds'] = (c3 - c2) * 3600.0
    return result


def choose_sites(result):
    candidates = []
    for latitude in np.arange(-78.0, 78.1, GRID_DEG):
        for longitude in np.arange(-180.0, 180.0, GRID_DEG):
            seen = circumstances_at(result, float(latitude), float(longitude))
            if seen is not None and seen['obscuration'] > 0.03:
                candidates.append((seen['obscuration'], float(latitude),
                                   float(longitude)))
    if not candidates:
        return []
    candidates.sort()
    ranks = np.linspace(0, len(candidates) - 1, SITES_PER_ECLIPSE)
    sites = [candidates[int(round(r))][1:] for r in ranks]
    if result.path:
        for fraction in (0.25, 0.5, 0.75):
            point = result.path[int(fraction * (len(result.path) - 1))]
            sites.append((point.latitude, point.longitude))
    return list(dict.fromkeys((round(a, 4), round(b, 4)) for a, b in sites))


def obscuration_at(window, latitude, longitude, tt):
    """This package's coverage at an instant, for costing a time difference."""
    xyz = g.geodetic_to_itrf(latitude, longitude, 0.0).reshape(3, 1)
    state = cc.state_at(window, xyz, np.array([tt]))
    return float(g.obscuration(state['r_sun'], state['r_moon'],
                               state['separation'])[0])


def compare_site(mine, theirs, elements, stamp, kind, latitude, longitude,
                 window):
    central = mine['central_seconds'] > 0.0 or theirs['central_seconds'] > 0.0
    row = {'stamp': stamp, 'type': kind, 'latitude': latitude,
           'longitude': longitude,
           'obscuration_pct': (mine['obscuration']
                               - theirs['obscuration']) * 100.0,
           'grazing': min(mine['sun_altitude'],
                          theirs['altitude']) < HORIZON_LIMIT_DEG}
    if not central:
        # Where the Moon covers the Sun entirely, "magnitude" means the ratio
        # of diameters by convention and the fraction of the diameter covered
        # by formula; the two are different numbers, so only partial sites are
        # a like-for-like comparison.
        row['magnitude'] = mine['magnitude'] - theirs['magnitude']
    if theirs['central_seconds'] > 0.0 and mine['central_seconds'] > 0.0:
        row['central_seconds'] = (mine['central_seconds']
                                  - theirs['central_seconds'])
    if row['grazing']:
        return row
    # Through the central phase the coverage is flat — exactly the ratio of
    # the two discs while one is wholly inside the other — so every instant of
    # it ties and the time of maximum is not a quantity to compare.  That is
    # as true of the seven minutes of an annular eclipse as of totality.
    if not central:
        theirs_tt = elements.jd0 + theirs['hours_maximum'] / 24.0
        cost = mine['obscuration'] - obscuration_at(window, latitude,
                                                    longitude, theirs_tt)
        row['maximum_cost_pct'] = cost * 100.0
        row['maximum_seconds'] = (elements.hours_of(mine['tt_maximum'])
                                  - theirs['hours_maximum']) * 3600.0
    for name, key in (('c1', 'tt_first_contact'), ('c4', 'tt_last_contact')):
        if theirs['hours_' + name] is not None and mine[key] == mine[key]:
            row[name + '_seconds'] = ((elements.hours_of(mine[key])
                                       - theirs['hours_' + name]) * 3600.0)
    return row


def run(k_values):
    """Compare at each lunar radius in turn, reusing the expensive work."""
    with open(ELEMENTS) as handle:
        catalogue = json.load(handle)
    rows = {label: [] for label, _ in k_values}

    for stamp in sorted(catalogue):
        record = catalogue[stamp]
        elements = Elements(record)
        ephem = Ephemeris(os.environ.get('ECLIPSEPATH_KERNEL'),
                          delta_t=record['delta_t'])
        ts = ephem.timescale
        year, month, day = int(stamp[:4]), int(stamp[4:6]), int(stamp[6:])
        for label, k in k_values:
            g.set_lunar_radius(k)
            event = finder.find_events(ephem, ts.utc(year, month, day - 1).tt,
                                       ts.utc(year, month, day + 1).tt)[0]
            result = ec.analyse(ephem, event, samples=40)
            if label == k_values[0][0]:
                sites = choose_sites(result)
            for latitude, longitude in sites:
                mine = circumstances_at(result, latitude, longitude)
                theirs = circumstances(elements, latitude, longitude)
                if mine is None or theirs is None:
                    continue
                rows[label].append(compare_site(mine, theirs, elements, stamp,
                                                record['type'], latitude,
                                                longitude, result.window))
        print('  %s %s: %d sites' % (stamp, record['type'], len(sites)),
              flush=True)
    return rows


def summarise(rows, field):
    values = [(abs(r[field]), r) for r in rows if field in r]
    if not values:
        return None
    worst, where = max(values, key=lambda pair: pair[0])
    return (len(values), sum(v for v, _ in values) / len(values), worst,
            '%s %.0f,%.0f' % (where['stamp'], where['latitude'],
                              where['longitude']))


FIELDS = (('obscuration', 'obscuration_pct', '%'),
          ('magnitude', 'magnitude', ''),
          ('maximum', 'maximum_seconds', ' s'),
          ('cost of their max', 'maximum_cost_pct', '%'),
          ('central duration', 'central_seconds', ' s'),
          ('first contact', 'c1_seconds', ' s'),
          ('last contact', 'c4_seconds', ' s'))


def report(rows, label, tolerances=True):
    grazing = sum(1 for r in rows if r['grazing'])
    print('\n%s' % label)
    central = sum(1 for r in rows if 'maximum_seconds' not in r
                  and not r['grazing'])
    print('  %d sites over %d eclipses; %d too near the horizon to time, '
          '%d in the central phase' 
          % (len(rows), len({r['stamp'] for r in rows}), grazing, central))
    ratio = 0.0
    for name, field, unit in FIELDS:
        stats = summarise(rows, field)
        if stats is None:
            continue
        count, mean, worst, where = stats
        print('  %-17s n=%-4d mean %7.3f%s  worst %7.3f%s  (%s)'
              % (name, count, mean, unit, worst, unit, where))
        limit = TOLERANCE.get(field)
        if limit and tolerances:
            ratio = max(ratio, worst / limit)
    return ratio


def main():
    if not os.path.exists(ELEMENTS):
        print('no elements; run tests/scrape_besselian.py first')
        return 1
    rows = run(K_VALUES)
    g.set_lunar_radius('espenak')
    ratio = 0.0
    for index, (label, _) in enumerate(K_VALUES):
        ratio = max(ratio, report(rows[label], 'Lunar radius k = %s:' % label,
                                  tolerances=index == 0))
    ok = ratio <= 1.0
    print('\n%s' % ('Local circumstances agree with NASA.' if ok else
                    'DISAGREEMENT with NASA beyond tolerance.'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
