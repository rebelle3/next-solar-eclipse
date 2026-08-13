"""One eclipse, packed small enough for a browser to draw it.

The globe viewer does not re-derive any geometry.  It is handed the Sun's and
the Moon's positions in the Earth-fixed frame at a regular cadence, and works
out for itself, per pixel, how much of the Sun is covered at the point of the
Earth under that pixel -- using the same circle-overlap the rest of the package
uses.  So what a scene has to carry is small: two position tracks, a handful of
constants, and enough outline to recognise the world by.

Positions are Earth-fixed on purpose.  It leaves the globe and its coastlines
still while the Sun and Moon move around them, which is what a viewer wants
anyway, and it saves shipping an orientation matrix per sample.

Run as ``python3 -m eclipsepath.scene 2027-08-02 land.geojson -o globe.html``
to write a page that opens with no server and no network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

from . import eclipse as ec, finder, output
from . import geometry as g
from .catalog import TYPE_NAMES
from .ephemeris import Ephemeris

VIEWER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer.html')
PLACEHOLDER = '/*SCENE*/null/*SCENE*/'

# One sample a minute.  The viewer interpolates with a Catmull-Rom spline,
# whose error falls as the fourth power of the spacing, so a minute puts the
# Moon within a few centimetres of where the ephemeris has it -- tested in
# tests/test_scene.py rather than asserted here.
CADENCE_SECONDS = 60.0


def sample_times(eclipse, cadence_seconds=CADENCE_SECONDS):
    """Regular instants across the eclipse, one past each end for the spline.

    Catmull-Rom needs a neighbour on both sides of the interval it is
    interpolating, so the track has to start before first contact and finish
    after last contact or the first and last minutes cannot be drawn.
    """
    step = cadence_seconds / 86400.0
    count = int(np.ceil((eclipse.tt_last_contact - eclipse.tt_first_contact)
                        / step)) + 1
    start = eclipse.tt_first_contact - step
    return start + step * np.arange(count + 3), step


def tracks(window, times):
    """Sun and Moon in the Earth-fixed frame, kilometres, as flat lists."""
    sun, moon, rot = window.at(times)
    return (g.rotate_to_itrf(rot, sun), g.rotate_to_itrf(rot, moon))


def rings_from_geojson(path, places=3):
    """Closed outlines as flat [lat, lon, lat, lon, ...] lists.

    Three decimal places is about 100 m, which is finer than the coastline in
    a 1:110m dataset is drawn to in the first place.
    """
    with open(path) as handle:
        data = json.load(handle)
    rings = []
    for feature in data.get('features', [data]):
        geometry = feature['geometry']
        if geometry['type'] == 'Polygon':
            candidates = [geometry['coordinates'][0]]
        elif geometry['type'] == 'MultiPolygon':
            candidates = [polygon[0] for polygon in geometry['coordinates']]
        else:
            continue
        for ring in candidates:
            flat = []
            for lon, lat in ring:
                flat.append(round(float(lat), places))
                flat.append(round(float(lon), places))
            rings.append(flat)
    return rings


def build(eclipse, timescale, coastlines=None,
          cadence_seconds=CADENCE_SECONDS):
    """Everything the viewer needs, as plain JSON-able data."""
    times, step = sample_times(eclipse, cadence_seconds)
    sun, moon = tracks(eclipse.window, times)
    stamp = output.utc_iso(timescale, eclipse.tt_greatest)

    band = eclipse.central_band
    limits = {'north': [], 'south': []}
    if band is not None:
        for point in band.points:
            limits['north'].extend([round(point.north_latitude, 4),
                                    round(point.north_longitude, 4)])
            limits['south'].extend([round(point.south_latitude, 4),
                                    round(point.south_longitude, 4)])

    centre = []
    for point in eclipse.path:
        if point.central:
            centre.extend([round(point.latitude, 4), round(point.longitude, 4)])

    return {
        'eclipse': {
            'date': stamp[:10],
            'greatest_utc': stamp,
            'kind': eclipse.kind,
            'type_name': TYPE_NAMES[eclipse.kind],
            'gamma': round(eclipse.gamma, 4),
            'magnitude': round(eclipse.magnitude, 4),
            'obscuration': round(eclipse.obscuration, 6),
            'saros': eclipse.saros,
            'latitude': round(eclipse.latitude, 4),
            'longitude': round(eclipse.longitude, 4),
            'sun_altitude': round(eclipse.sun_altitude, 2),
            'central_duration_seconds': round(eclipse.central_duration_seconds, 2),
            'path_width_km': (None if eclipse.path_width_km != eclipse.path_width_km
                              else round(eclipse.path_width_km, 1)),
            'central': bool(eclipse.central),
        },
        'constants': {
            'r_sun_km': g.R_SUN_KM,
            'r_moon_km': round(g.R_MOON_KM, 6),
            'earth_a_km': g.EARTH_A_KM,
            'earth_b_km': round(g.EARTH_B_KM, 9),
            'horizon_altitude_deg': g.HORIZON_ALTITUDE_DEG,
            'lunar_radius_k': round(g.R_MOON_KM / g.EARTH_A_KM, 7),
        },
        'time': {
            'tt0': float(times[0]),
            'step_days': float(step),
            'count': int(times.size),
            'first_contact_tt': eclipse.tt_first_contact,
            'last_contact_tt': eclipse.tt_last_contact,
            'greatest_tt': eclipse.tt_greatest,
            # When the umbra is somewhere on the ground, for the band the
            # timeline draws.  NaN for a partial, which never has one.
            'central_start_tt': (None if eclipse.tt_central_start
                                 != eclipse.tt_central_start
                                 else eclipse.tt_central_start),
            'central_end_tt': (None if eclipse.tt_central_end
                               != eclipse.tt_central_end
                               else eclipse.tt_central_end),
            # A stamp for the first sample plus the step is all the viewer
            # needs to label a clock; it never has to know about leap seconds
            # because every instant it is asked about lies inside this window.
            'utc0': output.utc_iso(timescale, float(times[0])),
            'utc0_seconds': _seconds_of_day(timescale, float(times[0])),
        },
        'sun_itrf_km': [round(v, 0) for v in sun.T.ravel().tolist()],
        'moon_itrf_km': [round(v, 3) for v in moon.T.ravel().tolist()],
        'path': {'centre': centre, 'north': limits['north'],
                 'south': limits['south']},
        'coastlines': rings_from_geojson(coastlines) if coastlines else [],
        'generated_utc': dt.datetime.now(dt.timezone.utc)
                           .strftime('%Y-%m-%dT%H:%M:%SZ'),
    }


def _seconds_of_day(timescale, tt):
    stamp = output.utc_iso(timescale, tt)
    hour, minute, second = (int(part) for part in stamp[11:19].split(':'))
    return hour * 3600 + minute * 60 + second


def dumps(scene):
    """Compact JSON: no spaces, and the long arrays on one line each."""
    return json.dumps(scene, separators=(',', ':'))


def standalone(scene, viewer=VIEWER):
    """The viewer with the scene baked in, openable straight off disk.

    Not a separate .json fetched at load: a page opened from a file cannot
    fetch its neighbours, and the whole point is that this works with no
    server and no network.
    """
    with open(viewer) as handle:
        page = handle.read()
    if page.count(PLACEHOLDER) != 1:
        raise ValueError('viewer template has no single scene placeholder')
    return page.replace(PLACEHOLDER, dumps(scene))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='eclipsepath.scene',
        description='Write a self-contained 3D eclipse viewer.')
    parser.add_argument('date', help='date of the eclipse, YYYY-MM-DD')
    parser.add_argument('coastlines', nargs='?', default=None,
                        help='Natural Earth land GeoJSON')
    parser.add_argument('--out', '-o', default=None)
    parser.add_argument('--json', action='store_true',
                        help='write the scene data instead of a page')
    parser.add_argument('--cadence', type=float, default=CADENCE_SECONDS,
                        metavar='SECONDS')
    parser.add_argument('--samples', type=int, default=160)
    parser.add_argument('--kernel', default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args(argv)

    def note(message):
        if not args.quiet:
            print(message, file=sys.stderr)

    year, month, day = (int(part) for part in args.date.split('-'))
    note('loading ephemeris...')
    ephem = Ephemeris(args.kernel)
    ts = ephem.timescale
    note('finding the eclipse...')
    events = finder.find_events(ephem, ts.utc(year, month, day - 1).tt,
                                ts.utc(year, month, day + 1).tt)
    if not events:
        raise SystemExit('no solar eclipse on %s' % args.date)
    note('tracing the path...')
    eclipse = ec.analyse(ephem, events[0], threshold=1.0, samples=args.samples)
    scene = build(eclipse, ts, args.coastlines, args.cadence)

    out = args.out or ('eclipse_%s.%s' % (args.date.replace('-', ''),
                                          'json' if args.json else 'html'))
    text = dumps(scene) if args.json else standalone(scene)
    with open(out, 'w') as handle:
        handle.write(text)
    print('wrote %s (%.0f kB, %d position samples, %d coastline rings)'
          % (out, len(text) / 1024.0, scene['time']['count'],
             len(scene['coastlines'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
