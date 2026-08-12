"""Rendering eclipse results as text, JSON, CSV or GeoJSON."""

from __future__ import annotations

import csv
import io
import json

from .catalog import TYPE_NAMES


def utc_iso(timescale, tt, decimals=0):
    if tt != tt:                                   # NaN
        return None
    return timescale.tt_jd(tt).utc_strftime(
        '%Y-%m-%dT%H:%M:%SZ' if decimals == 0 else '%Y-%m-%dT%H:%M:%S.%fZ')


def utc_clock(timescale, tt):
    return None if tt != tt else timescale.tt_jd(tt).utc_strftime('%H:%M:%S')


def format_duration(seconds):
    if not seconds or seconds != seconds:
        return '-'
    minutes, rest = divmod(float(seconds), 60.0)
    return '%dm%04.1fs' % (minutes, rest)


def _signed(value, positive, negative, digits=4):
    return '%.*f%s' % (digits, abs(value), positive if value >= 0 else negative)


# --- antimeridian handling ---------------------------------------------------

def unwrap_to(reference, longitude):
    """Longitude expressed as the nearest equivalent to ``reference``.

    A band can straddle the antimeridian within a single cross-section, with
    its northern edge at +178 and its southern edge at -179.  Left alone that
    becomes a polygon stretching the wrong way round the globe, so the edges
    are expressed relative to the centre line, which may push a value slightly
    outside +/-180 - renderers handle that far better than a 357 degree jump.
    """
    return reference + ((longitude - reference + 540.0) % 360.0) - 180.0


def split_runs(*longitude_series):
    """Index runs over which none of the given longitude series wraps.

    GeoJSON consumers draw a line straight across the whole map when a segment
    steps from +179 to -179, so tracks are cut into pieces there.  A band is
    cut wherever *any* of its three lines wraps, otherwise one edge of the
    polygon would still stretch the wrong way round the globe.
    """
    length = len(longitude_series[0])
    runs, start = [], 0
    for i in range(1, length):
        if any(abs(series[i] - series[i - 1]) > 180.0
               for series in longitude_series):
            runs.append((start, i))
            start = i
    runs.append((start, length))
    return [(a, b) for a, b in runs if b - a >= 2]


# --- GeoJSON -----------------------------------------------------------------

def _line_feature(coords, times, properties):
    return {'type': 'Feature',
            'properties': dict(properties, times=times),
            'geometry': {'type': 'LineString', 'coordinates': coords}}


def eclipse_geojson_features(eclipse, timescale):
    base = eclipse_summary(eclipse, timescale)
    features = [{
        'type': 'Feature',
        'properties': dict(base, feature='greatest-eclipse'),
        'geometry': {'type': 'Point',
                     'coordinates': [round(eclipse.longitude, 6),
                                     round(eclipse.latitude, 6)]}}]

    track_lon = [p.longitude for p in eclipse.path]
    for a, b in split_runs(track_lon):
        features.append(_line_feature(
            [[round(p.longitude, 6), round(p.latitude, 6)]
             for p in eclipse.path[a:b]],
            [utc_iso(timescale, p.tt) for p in eclipse.path[a:b]],
            dict(base, feature='shadow-track',
                 description='locus of the deepest eclipse on the globe')))

    band = eclipse.coverage_band
    if band and band.points:
        label = band.label
        centre = [p.longitude for p in band.points]
        north = [unwrap_to(c, p.north_longitude)
                 for c, p in zip(centre, band.points)]
        south = [unwrap_to(c, p.south_longitude)
                 for c, p in zip(centre, band.points)]
        for a, b in split_runs(centre):
            chunk = list(zip(band.points[a:b], north[a:b], south[a:b]))
            times = [utc_iso(timescale, p.tt) for p, _, _ in chunk]
            features.append(_line_feature(
                [[round(p.longitude, 6), round(p.latitude, 6)]
                 for p, _, _ in chunk],
                times, dict(base, feature='centre-line', band=label)))
            features.append(_line_feature(
                [[round(lon, 6), round(p.north_latitude, 6)]
                 for p, lon, _ in chunk], times,
                dict(base, feature='northern-limit', band=label)))
            features.append(_line_feature(
                [[round(lon, 6), round(p.south_latitude, 6)]
                 for p, _, lon in chunk], times,
                dict(base, feature='southern-limit', band=label)))
            ring = ([[round(lon, 6), round(p.north_latitude, 6)]
                     for p, lon, _ in chunk]
                    + [[round(lon, 6), round(p.south_latitude, 6)]
                       for p, _, lon in reversed(chunk)])
            ring.append(ring[0])
            features.append({
                'type': 'Feature',
                'properties': dict(base, feature='band', band=label,
                                   start=times[0], end=times[-1]),
                'geometry': {'type': 'Polygon', 'coordinates': [ring]}})
    return features


def to_geojson(eclipses, timescale):
    features = []
    for eclipse in eclipses:
        features.extend(eclipse_geojson_features(eclipse, timescale))
    return {'type': 'FeatureCollection', 'features': features}


# --- structured summaries ----------------------------------------------------

def eclipse_summary(eclipse, timescale):
    return {
        'date': utc_iso(timescale, eclipse.tt_greatest)[:10],
        'type': eclipse.kind,
        'type_name': TYPE_NAMES[eclipse.kind],
        'greatest_eclipse_utc': utc_iso(timescale, eclipse.tt_greatest),
        'latitude': round(eclipse.latitude, 4),
        'longitude': round(eclipse.longitude, 4),
        'magnitude': round(eclipse.magnitude, 4),
        'peak_obscuration_percent': round(eclipse.peak_obscuration * 100.0, 3),
        'gamma': round(eclipse.gamma, 4),
        'sun_altitude': round(eclipse.sun_altitude, 1),
        'saros': eclipse.saros,
        'lunation': eclipse.lunation,
    }


def eclipse_detail(eclipse, timescale):
    detail = eclipse_summary(eclipse, timescale)
    detail.update({
        'first_contact_utc': utc_iso(timescale, eclipse.tt_first_contact),
        'last_contact_utc': utc_iso(timescale, eclipse.tt_last_contact),
        'path_width_km': (None if eclipse.path_width_km != eclipse.path_width_km
                          else round(eclipse.path_width_km, 1)),
        'central_phase_start_utc': utc_iso(timescale, eclipse.tt_central_start),
        'central_phase_end_utc': utc_iso(timescale, eclipse.tt_central_end),
        'central_duration_seconds': round(eclipse.central_duration_seconds, 1),
        'threshold_percent': round(eclipse.threshold * 100.0, 3),
        'meets_threshold': eclipse.meets_threshold,
    })
    band = eclipse.coverage_band
    detail['band'] = None if not band else {
        'label': band.label,
        'truncated': band.truncated,
        'points': [{
            'time_utc': utc_iso(timescale, p.tt),
            'latitude': round(p.latitude, 5),
            'longitude': round(p.longitude, 5),
            'north_latitude': round(p.north_latitude, 5),
            'north_longitude': round(p.north_longitude, 5),
            'south_latitude': round(p.south_latitude, 5),
            'south_longitude': round(p.south_longitude, 5),
            'width_km': None if p.width_km != p.width_km else round(p.width_km, 1),
            'width_is_transverse': p.transverse,
            'duration_seconds': round(p.duration_seconds, 1),
            'sun_altitude': round(p.sun_altitude, 1),
            'starts_utc': utc_iso(timescale, p.tt_start),
            'ends_utc': utc_iso(timescale, p.tt_end),
        } for p in band.points]}
    return detail


def to_json(eclipses, timescale, metadata=None):
    return {'metadata': metadata or {},
            'eclipses': [eclipse_detail(e, timescale) for e in eclipses]}


CSV_COLUMNS = ['date', 'type', 'band', 'time_utc', 'centre_latitude',
               'centre_longitude', 'north_latitude', 'north_longitude',
               'south_latitude', 'south_longitude', 'width_km',
               'duration_seconds', 'sun_altitude', 'starts_utc', 'ends_utc']


def to_csv(eclipses, timescale):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(CSV_COLUMNS)
    for eclipse in eclipses:
        band = eclipse.coverage_band
        if not band:
            continue
        date = utc_iso(timescale, eclipse.tt_greatest)[:10]
        for p in band.points:
            writer.writerow([
                date, eclipse.kind, band.label, utc_iso(timescale, p.tt),
                '%.5f' % p.latitude, '%.5f' % p.longitude,
                '%.5f' % p.north_latitude, '%.5f' % p.north_longitude,
                '%.5f' % p.south_latitude, '%.5f' % p.south_longitude,
                '' if p.width_km != p.width_km else '%.1f' % p.width_km,
                '%.1f' % p.duration_seconds,
                '%.1f' % p.sun_altitude,
                utc_iso(timescale, p.tt_start), utc_iso(timescale, p.tt_end)])
    return buffer.getvalue()


# --- human-readable text -----------------------------------------------------

def format_text(eclipses, timescale, threshold, path_rows=14, show_all=False):
    out = []
    qualifying = [e for e in eclipses if e.meets_threshold]
    out.append('Solar eclipses found: %d   (%d reach %g%% obscuration)'
               % (len(eclipses), len(qualifying), threshold * 100.0))
    out.append('')
    header = ('  Date         Greatest (UTC)  Type      Mag   Cover%   Gamma  '
              'Saros   Width      Duration   Greatest eclipse at')
    out.append(header)
    out.append('  ' + '-' * (len(header) - 2))
    for e in eclipses:
        if not show_all and not e.meets_threshold:
            continue
        width = ('%6.0f km' % e.path_width_km
                 if e.path_width_km == e.path_width_km else '       -')
        out.append('  %s  %s  %-8s %5.3f %6.2f%%  %+.3f   %3d  %s  %9s   %s, %s'
                   % (utc_iso(timescale, e.tt_greatest)[:10],
                      utc_clock(timescale, e.tt_greatest),
                      TYPE_NAMES[e.kind], e.magnitude,
                      e.peak_obscuration * 100.0, e.gamma, e.saros, width,
                      format_duration(e.central_duration_seconds),
                      _signed(e.latitude, 'N', 'S', 2),
                      _signed(e.longitude, 'E', 'W', 2)))
    if not show_all:
        skipped = [e for e in eclipses if not e.meets_threshold]
        if skipped:
            out.append('')
            out.append('  %d eclipse(s) below the %g%% threshold, not listed: %s'
                       % (len(skipped), threshold * 100.0,
                          ', '.join('%s (%s, %.1f%%)'
                                    % (utc_iso(timescale, e.tt_greatest)[:10],
                                       e.kind, e.peak_obscuration * 100.0)
                                    for e in skipped)))

    for e in qualifying:
        band = e.coverage_band
        if not band or not band.points:
            continue
        out.append('')
        out.append('=' * 100)
        out.append('%s  %s solar eclipse  -  path of %s'
                   % (utc_iso(timescale, e.tt_greatest)[:10],
                      TYPE_NAMES[e.kind], band.label))
        out.append('  eclipse begins %s UTC, ends %s UTC   saros %d, lunation %d'
                   % (utc_clock(timescale, e.tt_first_contact),
                      utc_clock(timescale, e.tt_last_contact),
                      e.saros, e.lunation))
        out.append('-' * 100)
        out.append('   Time UTC   Centre line          Northern limit       '
                   'Southern limit         Width   Duration  Sun')
        out.append('-' * 100)
        points = band.points
        step = max(1, len(points) // path_rows) if path_rows else 1
        shown = points[::step]
        if shown[-1] is not points[-1]:
            shown.append(points[-1])
        for p in shown:
            width = '%6.0f km' % p.width_km if p.width_km == p.width_km else '        -'
            out.append('   %s  %8s %9s  %8s %9s  %8s %9s  %s %9s %4.0f'
                       % (utc_clock(timescale, p.tt),
                          _signed(p.latitude, 'N', 'S', 3),
                          _signed(p.longitude, 'E', 'W', 3),
                          _signed(p.north_latitude, 'N', 'S', 3),
                          _signed(p.north_longitude, 'E', 'W', 3),
                          _signed(p.south_latitude, 'N', 'S', 3),
                          _signed(p.south_longitude, 'E', 'W', 3),
                          width, format_duration(p.duration_seconds),
                          p.sun_altitude))
        if band.truncated:
            out.append('   note: the band is wider than the %d km search limit '
                       'somewhere along the path' % 12000)
    return '\n'.join(out)


def dumps(data):
    return json.dumps(data, indent=1)
