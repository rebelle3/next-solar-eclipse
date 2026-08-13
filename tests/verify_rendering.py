#!/usr/bin/env python3
"""Structural checks on what the renderer is handed, across awkward geometry.

Paths that cross the antimeridian, run over a pole, taper to nothing at
sunrise, or belong to annular, hybrid and partial eclipses all reach the
drawing code, and a map only has to be wrong once to mislead.  This walks a
spread of eclipses across the century, builds the GeoJSON for each, and checks
it mechanically rather than by looking at pictures.
"""

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import eclipse as ec  # noqa: E402
from eclipsepath import finder, geometry as g, output  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402


def points_of(geometry):
    kind = geometry['type']
    if kind == 'Point':
        return [geometry['coordinates']]
    if kind == 'Polygon':
        return geometry['coordinates'][0]
    return geometry['coordinates']


def check(collection, label, problems):
    def fail(message):
        problems.append('%s: %s' % (label, message))

    for feature in collection['features']:
        name = feature['properties']['feature']
        geometry = feature['geometry']
        coords = points_of(geometry)
        if not coords:
            fail('%s has no coordinates' % name)
            continue
        for lon, lat in coords:
            if not -90.0 <= lat <= 90.0:
                fail('%s latitude out of range: %s' % (name, lat))
            if not -190.0 <= lon <= 190.0:
                fail('%s longitude out of range: %s' % (name, lon))
        steps = [abs(coords[i][0] - coords[i - 1][0]) for i in range(1, len(coords))]
        if steps and max(steps) >= 180.0:
            fail('%s wraps the globe (step %.1f deg)' % (name, max(steps)))
        if geometry['type'] == 'Polygon':
            if coords[0] != coords[-1]:
                fail('%s ring not closed' % name)
            if len(coords) < 4:
                fail('%s ring has only %d points' % (name, len(coords)))
        if geometry['type'] == 'LineString':
            times = feature['properties'].get('times')
            if times is None or len(times) != len(coords):
                fail('%s times misaligned (%s vs %d coords)'
                     % (name, None if times is None else len(times), len(coords)))
            if any(t is None for t in times):
                fail('%s has a null timestamp' % name)


def check_band(result, label, problems):
    """Each cross-section's limits must sit either side of its centre line."""
    band = result.central_band
    if not band:
        return
    for point in band.points:
        if not point.transverse:
            continue
        to_north = g.geodesic_distance(point.latitude, point.longitude,
                                       point.north_latitude, point.north_longitude)
        to_south = g.geodesic_distance(point.latitude, point.longitude,
                                       point.south_latitude, point.south_longitude)
        across = g.geodesic_distance(point.north_latitude, point.north_longitude,
                                     point.south_latitude, point.south_longitude)
        if across < max(to_north, to_south) - 1e-6:
            problems.append('%s: limits on the same side of the centre line '
                            'at %.3f,%.3f' % (label, point.latitude, point.longitude))
            return


def main():
    ephem = Ephemeris(os.environ.get('ECLIPSEPATH_KERNEL'))
    g.set_lunar_radius('espenak')
    ts = ephem.timescale
    events = finder.find_events(ephem, ts.utc(2026, 8, 13).tt, ts.utc(2056, 8, 13).tt)
    print('checking %d eclipses over 30 years' % len(events))

    problems, tallies = [], {'antimeridian': 0, 'polar': 0, 'pole-enclosing': 0,
                             'kinds': {}}
    for index, event in enumerate(events):
        result = ec.analyse(ephem, event, threshold=1.0, samples=40)
        label = ts.tt_jd(result.tt_greatest).utc_strftime('%Y-%m-%d') + ' ' + result.kind
        tallies['kinds'][result.kind] = tallies['kinds'].get(result.kind, 0) + 1
        longitudes = [p.longitude for p in result.path]
        if any(abs(longitudes[i] - longitudes[i - 1]) > 180.0
               for i in range(1, len(longitudes))):
            tallies['antimeridian'] += 1
        if any(abs(p.latitude) > 80.0 for p in result.path):
            tallies['polar'] += 1
        check(output.to_geojson([result], ts), label, problems)
        check_band(result, label, problems)
        # and again at a threshold that produces a region rather than a band
        lower = ec.analyse(ephem, event, threshold=0.5, samples=40)
        if lower.coverage_region and lower.coverage_region.encloses_pole:
            tallies['pole-enclosing'] += 1
        check(output.to_geojson([lower], ts), label + ' @50%', problems)
        if index % 15 == 0:
            print('  %3d/%d %s' % (index, len(events), label), flush=True)

    print('\ngeometry exercised: %s' % tallies['kinds'])
    print('  crossing the antimeridian: %d' % tallies['antimeridian'])
    print('  reaching beyond 80 degrees: %d' % tallies['polar'])
    print('  regions enclosing a pole: %d' % tallies['pole-enclosing'])
    if problems:
        print('\n%d PROBLEMS:' % len(problems))
        for line in problems[:20]:
            print('  ' + line)
        return 1
    print('\nEvery geometry handed to the renderer is well formed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
