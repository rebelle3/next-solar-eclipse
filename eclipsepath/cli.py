"""Command line interface."""

from __future__ import annotations

import argparse
import datetime as dt
import sys

from . import catalog, output
from . import geometry as g
from .ephemeris import Ephemeris
from .observer import circumstances_at

FORMATS = ('text', 'json', 'csv', 'geojson')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='eclipsepath',
        description='List solar eclipses over a time span and trace the ground '
                    'path where the Sun is covered by at least a given '
                    'percentage, with timings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='examples:\n'
               '  eclipsepath                             next 10 years of total eclipses\n'
               '  eclipsepath --coverage 90               anywhere reaching 90%% coverage\n'
               '  eclipsepath --years 3 --format geojson  paths as GeoJSON for a map\n'
               '  eclipsepath --at 51.48,-3.18            what Cardiff sees each time\n')
    when = parser.add_argument_group('time span')
    when.add_argument('--start', default=None, metavar='DATE',
                      help='ISO date to start from (default: today, UTC)')
    limit = when.add_mutually_exclusive_group()
    limit.add_argument('--years', type=float, default=10.0, metavar='N',
                       help='length of the span in years (default: 10)')
    limit.add_argument('--end', default=None, metavar='DATE',
                       help='ISO date to stop at, instead of --years')

    what = parser.add_argument_group('filtering')
    what.add_argument('--coverage', type=float, default=100.0, metavar='PCT',
                      help='minimum percentage of the Sun\'s area covered, '
                           'anywhere on Earth (default: 100, i.e. totality). '
                           'An annular eclipse never reaches 100%%.')
    what.add_argument('--all', action='store_true',
                      help='list every eclipse, including those below the '
                           'coverage threshold')

    how = parser.add_argument_group('output')
    how.add_argument('--format', choices=FORMATS, default='text')
    how.add_argument('--output', '-o', metavar='FILE',
                     help='write to a file instead of standard output')
    how.add_argument('--samples', type=int, default=120, metavar='N',
                     help='cross-sections traced along each path (default: 120)')
    how.add_argument('--rows', type=int, default=14, metavar='N',
                     help='path rows shown in text output; 0 for all')
    how.add_argument('--at', metavar='LAT,LON',
                     help='also report local circumstances at this location')
    how.add_argument('--quiet', '-q', action='store_true',
                     help='suppress progress messages on stderr')

    model = parser.add_argument_group('model')
    model.add_argument('--kernel', metavar='PATH',
                       help='JPL SPK kernel to use (default: de440s.bsp, '
                            'downloaded on first use)')
    model.add_argument('--lunar-radius', default='espenak',
                       choices=sorted(g.LUNAR_RADIUS_POLICIES),
                       help='k = lunar radius / Earth equatorial radius: '
                            '"espenak" = 0.2722810, matching published '
                            'catalogues (default); "iau" = 0.2725076')
    model.add_argument('--delta-t', type=float, metavar='SECONDS',
                       help='override TT - UT1; shifts paths in longitude')
    return parser


def parse_location(text):
    try:
        lat, lon = (float(part) for part in text.split(','))
    except ValueError:
        raise SystemExit('--at expects LAT,LON in degrees, e.g. 51.48,-3.18')
    if not -90.0 <= lat <= 90.0:
        raise SystemExit('latitude must be between -90 and 90')
    return lat, ((lon + 540.0) % 360.0) - 180.0


def local_report(eclipses, timescale, site):
    lat, lon = site
    lines = ['', 'Local circumstances at %.4f, %.4f' % (lat, lon),
             '  Date        Max cover  Magnitude   Starts    Maximum    Ends'
             '      Sun  Total/annular']
    for eclipse in eclipses:
        seen = circumstances_at(eclipse, lat, lon)
        date = output.utc_iso(timescale, eclipse.tt_greatest)[:10]
        if seen is None:
            lines.append('  %s        not visible' % date)
            continue
        lines.append('  %s   %6.2f%%     %6.4f   %8s  %8s  %8s  %3.0f  %s'
                     % (date, seen['obscuration'] * 100.0, seen['magnitude'],
                        output.utc_clock(timescale, seen['tt_first_contact']),
                        output.utc_clock(timescale, seen['tt_maximum']),
                        output.utc_clock(timescale, seen['tt_last_contact']),
                        seen['sun_altitude'],
                        output.format_duration(seen['central_seconds'])))
    return '\n'.join(lines)


def main(argv=None):
    args = build_parser().parse_args(argv)
    k = g.set_lunar_radius(args.lunar_radius)
    threshold = args.coverage / 100.0
    site = parse_location(args.at) if args.at else None
    start = args.start or dt.datetime.now(dt.timezone.utc).date().isoformat()

    def note(message):
        if not args.quiet:
            print(message, file=sys.stderr)

    note('loading ephemeris...')
    try:
        ephem = Ephemeris(args.kernel, delta_t=args.delta_t)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    def progress(index, total):
        note('  computing path %d/%d' % (index + 1, total))

    note('scanning for eclipses...')
    try:
        eclipses = catalog.search(start, end=args.end, years=args.years,
                                  threshold=threshold, samples=args.samples,
                                  ephemeris=ephem, progress=progress)
    except ValueError as exc:
        raise SystemExit(str(exc))

    ts = ephem.timescale
    selected = eclipses if args.all else [e for e in eclipses
                                          if e.meets_threshold]
    metadata = {
        'generated_utc': dt.datetime.now(dt.timezone.utc)
                           .strftime('%Y-%m-%dT%H:%M:%SZ'),
        'start': start, 'end': args.end, 'years': None if args.end else args.years,
        'coverage_threshold_percent': args.coverage,
        'ephemeris': ephem.path,
        'lunar_radius_k': k,
        'solar_radius_km': g.R_SUN_KM,
        'delta_t_seconds': round(ephem.delta_t_at(
            eclipses[0].tt_greatest), 3) if eclipses else None,
        'eclipse_count': len(eclipses),
        'reaching_threshold': sum(1 for e in eclipses if e.meets_threshold),
    }

    if args.format == 'text':
        text = output.format_text(eclipses, ts, threshold, path_rows=args.rows,
                                  show_all=args.all)
        if site:
            text += '\n' + local_report(selected or eclipses, ts, site)
    elif args.format == 'json':
        payload = output.to_json(selected, ts, metadata)
        if site:
            for entry, eclipse in zip(payload['eclipses'], selected):
                seen = circumstances_at(eclipse, *site)
                entry['observer'] = None if seen is None else {
                    'latitude': site[0], 'longitude': site[1],
                    'obscuration_percent': round(seen['obscuration'] * 100.0, 3),
                    'magnitude': round(seen['magnitude'], 4),
                    'sun_altitude': round(seen['sun_altitude'], 1),
                    'first_contact_utc': output.utc_iso(ts, seen['tt_first_contact']),
                    'maximum_utc': output.utc_iso(ts, seen['tt_maximum']),
                    'last_contact_utc': output.utc_iso(ts, seen['tt_last_contact']),
                    'central_seconds': round(seen['central_seconds'], 1)}
        text = output.dumps(payload)
    elif args.format == 'csv':
        text = output.to_csv(selected, ts)
    else:
        collection = output.to_geojson(selected, ts)
        collection['metadata'] = metadata
        text = output.dumps(collection)

    if args.output:
        with open(args.output, 'w') as handle:
            handle.write(text if text.endswith('\n') else text + '\n')
        note('wrote %s' % args.output)
    else:
        print(text)
    return 0
