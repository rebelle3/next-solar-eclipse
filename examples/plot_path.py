#!/usr/bin/env python3
"""Render an eclipse path as a map image.

An example of consuming ``eclipsepath --format json``.  Everything drawn, and
every word of the caption, comes out of that file.

    pip install matplotlib
    curl -sO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
    python3 -m eclipsepath --start 2027-01-01 --years 1 --samples 400 \
        --format json -o eclipse.json
    python3 examples/plot_path.py eclipse.json ne_110m_admin_0_countries.geojson
"""

import argparse
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Circle, Polygon as MplPolygon       # noqa: E402

# Roles from the reference data-visualisation palette, light mode.
SURFACE, INK, INK_2, GRIDLINE = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9'
SERIES_1, BAND_EDGE, CENTRE = '#2a78d6', '#1c5fae', '#0d3f78'
LAND, LAND_EDGE, SEA = '#e5e4dd', '#cfcec5', '#f3f4f4'

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']

POLAR_LATITUDE = 70.0    # beyond this a plate carree map is unreadable
GRATICULE = 15.0


def span(values):
    return max(values) - min(values)


def describe(eclipse, band_label):
    """Title and standfirst taken from the data, not written in.

    These used to be constants naming one particular eclipse, so every other
    one came out captioned as the total of 2 August 2027 — and an annular path
    labelled a path of totality is worse than no caption at all.
    """
    year, month, day = (int(v) for v in eclipse['date'].split('-'))
    title = '%s solar eclipse of %d %s %d' % (eclipse['type_name'], day,
                                              MONTHS[month - 1], year)
    minutes, seconds = divmod(round(eclipse['central_duration_seconds']), 60)
    lead = ('The Moon’s shadow crosses %d km wide between %s and %s UTC.'
            % (round(eclipse['path_width_km']),
               eclipse['central_phase_start_utc'][11:16],
               eclipse['central_phase_end_utc'][11:16]))
    note = ('Greatest eclipse is at %s UTC at %.1f, %.1f, where %s lasts '
            '%dm %02ds.'
            % (eclipse['greatest_eclipse_utc'][11:16], eclipse['latitude'],
               eclipse['longitude'], band_label, minutes, seconds))
    return title, lead, note


def near(longitude, reference):
    """``longitude`` moved whole turns to sit within 180 degrees of a point."""
    return longitude - 360.0 * round((longitude - reference) / 360.0)


def tracks(band):
    """Centre, north and south edges, longitudes made continuous.

    A path can run right round the world, so the raw longitudes in the file
    step from +180 to -180 partway along.  Drawn as they stand that step is a
    stripe straight back across the map; carrying each point to the turn
    nearest its neighbour removes it.
    """
    centre, north, south = [], [], []
    previous = band[0]['longitude']
    for cross in band:
        lon = near(cross['longitude'], previous)
        previous = lon
        centre.append((lon, cross['latitude']))
        north.append((near(cross['north_longitude'], lon),
                      cross['north_latitude']))
        south.append((near(cross['south_longitude'], lon),
                      cross['south_latitude']))
    return centre, north, south


class PlateCarree:
    """Longitude and latitude straight onto the axes."""

    def __init__(self, edges):
        lons = [p[0] for p in edges]
        lats = [p[1] for p in edges]
        self.xlim = (min(lons) - 5.0, max(lons) + 5.0)
        self.ylim = (min(lats) - 7.0, max(lats) + 7.0)
        # One degree of longitude is cos(latitude) as long on the ground, so
        # stretching the vertical by its reciprocal keeps shapes local-true.
        middle = min(60.0, abs(sum(lats) / len(lats)))
        self.aspect = 1.0 / math.cos(math.radians(middle))

    def point(self, lon, lat):
        return lon, lat

    def turns(self):
        """Whole turns of longitude the view spans, for repeating the land."""
        first = math.floor((self.xlim[0] + 180.0) / 360.0)
        last = math.floor((self.xlim[1] + 180.0) / 360.0)
        return [360.0 * k for k in range(first, last + 1)]

    def worth_drawing(self, ring, turn):
        return True

    def graticule(self):
        lon0, lon1 = self.xlim
        lat0, lat1 = self.ylim
        for k in range(int(math.floor(lon0 / GRATICULE)),
                       int(math.ceil(lon1 / GRATICULE)) + 1):
            yield [(k * GRATICULE, lat0), (k * GRATICULE, lat1)]
        for k in range(int(math.floor(lat0 / GRATICULE)),
                       int(math.ceil(lat1 / GRATICULE)) + 1):
            yield [(lon0, k * GRATICULE), (lon1, k * GRATICULE)]


class Polar:
    """Azimuthal equidistant about the pole the path leans on.

    Near the pole a plate carree map smears a compact path across the entire
    width of the sheet.  Distance from the pole as a radius keeps it a path.
    """

    def __init__(self, edges):
        self.sign = 1.0 if max(p[1] for p in edges) > 0 else -1.0
        self.clamp = 360.0
        self.aspect = 1.0
        # Fitted to the path rather than centred on the pole: the pole is only
        # the point the projection turns about, not necessarily in view.
        xy = [self.point(lon, lat) for lon, lat in edges]
        pad = 0.12 * max(4.0, max(span([p[i] for p in xy]) for i in (0, 1)))
        self.xlim = (min(p[0] for p in xy) - pad, max(p[0] for p in xy) + pad)
        self.ylim = (min(p[1] for p in xy) - pad, max(p[1] for p in xy) + pad)
        # Land running off the sheet is pinned beyond its corners rather than
        # folded back over the pole on the far side.
        self.clamp = 1.2 * max(abs(v) for v in self.xlim + self.ylim)

    def point(self, lon, lat):
        r = min(self.clamp, 90.0 - self.sign * lat)
        angle = math.radians(self.sign * lon)
        return r * math.sin(angle), r * math.cos(angle)

    def turns(self):
        return [0.0]

    def worth_drawing(self, ring, turn):
        """Is any of this country in view?

        Somewhere the projection cannot reach — the far side of the globe —
        every vertex pins to the same circle, and the country comes out as a
        disc laid over the whole map.  Such a country is simply not drawn.
        """
        return any(self.xlim[0] <= x <= self.xlim[1]
                   and self.ylim[0] <= y <= self.ylim[1]
                   for x, y in (self.point(lon, lat) for lon, lat in ring))

    def graticule(self):
        reach = max(math.hypot(x, y)
                    for x in self.xlim for y in self.ylim)
        for k in range(1, int(reach / GRATICULE) + 1):
            circle = self.sign * (90.0 - k * GRATICULE)
            yield [(lon, circle) for lon in range(-180, 181, 5)]
        for k in range(0, int(360 / GRATICULE)):
            yield [(k * GRATICULE, self.sign * 90.0),
                   (k * GRATICULE, self.sign * (90.0 - reach))]


def projection_for(edges):
    if max(abs(p[1]) for p in edges) > POLAR_LATITUDE:
        return Polar(edges)
    return PlateCarree(edges)


def rings(geometry):
    if geometry['type'] == 'Polygon':
        return [geometry['coordinates'][0]]
    if geometry['type'] == 'MultiPolygon':
        return [polygon[0] for polygon in geometry['coordinates']]
    return []


def inside(x, y, ring):
    """Ray-casting point-in-polygon."""
    hit, count = False, len(ring)
    for i in range(count):
        x1, y1 = ring[i][0], ring[i][1]
        x2, y2 = ring[(i + 1) % count][0], ring[(i + 1) % count][1]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) / (y2 - y1) * (x2 - x1):
            hit = not hit
    return hit


def render(eclipse, countries, out):
    strip = next((item for item in eclipse['geometries']
                  if item['kind'] == 'band'), None)
    if strip is None:
        # A partial eclipse has no central path, and one whose threshold is met
        # over a blob rather than a strip is given as a region instead.
        raise SystemExit('%s eclipse of %s has no path to draw; try '
                         'examples/plot_coverage.py'
                         % (eclipse['type_name'].lower(), eclipse['date']))
    band, band_label = strip['cross_sections'], strip['label']
    title, lead, note = describe(eclipse, band_label)
    centre, north, south = tracks(band)
    view = projection_for(north + south)

    def draw(points, **kw):
        xy = [view.point(lon, lat) for lon, lat in points]
        return ax.plot([p[0] for p in xy], [p[1] for p in xy], **kw)

    def offset_from(index, side, distance):
        """A point ``distance`` off the band on screen, square to it.

        Offsetting square to the band rather than straight up keeps a label
        clear of it however the track happens to be running.
        """
        cx, cy = view.point(*centre[index])
        ex, ey = view.point(*(north if side == 'north' else south)[index])
        dx, dy = ex - cx, ey - cy
        length = math.hypot(dx, dy) or 1.0
        return cx + dx / length * distance, cy + dy / length * distance

    polar = isinstance(view, Polar)
    fig, ax = plt.subplots(figsize=(13.8, 9.8) if polar else (17.5, 8.6),
                           dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SEA)

    on_centre, anywhere = {}, {}
    for feature in countries['features']:
        name = feature['properties']['NAME']
        for ring in rings(feature['geometry']):
            for turn in view.turns():
                if not view.worth_drawing(ring, turn):
                    continue
                ax.add_patch(MplPolygon(
                    [view.point(lon + turn, lat) for lon, lat in ring],
                    closed=True, facecolor=LAND, edgecolor=LAND_EDGE,
                    linewidth=0.6, zorder=1))
            # Which countries the band reaches is a question about the ground,
            # so it is asked of the raw coordinates, not the drawn ones.  A
            # country sees totality if any part of the band covers it, which
            # for some is only an edge and never the centre line.
            for i, cross in enumerate(band):
                for key, track in (('centre', ''), ('north', 'north_'),
                                   ('south', 'south_')):
                    if inside(cross[track + 'longitude'],
                              cross[track + 'latitude'], ring):
                        anywhere.setdefault(name, []).append(i)
                        if key == 'centre':
                            on_centre.setdefault(name, []).append(i)

    for line in view.graticule():
        draw(line, color=GRIDLINE, lw=0.5, zorder=0.5)

    ax.add_patch(MplPolygon([view.point(*p) for p in north + south[::-1]],
                            closed=True, facecolor=SERIES_1, alpha=0.82,
                            edgecolor='none', zorder=3))
    for edge in (north, south):
        draw(edge, color=BAND_EDGE, lw=0.7, alpha=0.9, zorder=4)
    draw(centre, color=CENTRE, lw=1.3, zorder=5)

    def place(entries, side, tick_from, tick_to, min_gap, **text_kw):
        """Leader-and-label pairs, dropping any that would crowd a neighbour."""
        placed = []
        for index, label in entries:
            tx, ty = offset_from(index, side, tick_to)
            if any(math.hypot(tx - px, ty - py) < min_gap for px, py in placed):
                continue
            placed.append((tx, ty))
            ax.plot(*zip(offset_from(index, side, tick_from), (tx, ty)),
                    color=INK_2, lw=0.9, zorder=6)
            ax.text(tx, ty + (0.9 if side == 'north' else -0.9), label,
                    ha='center', va='bottom' if side == 'north' else 'top',
                    zorder=7, bbox=dict(boxstyle='round,pad=0.28', fc=SURFACE,
                                        ec='none', alpha=0.88), **text_kw)

    times, seen = [], set()
    for i, cross in enumerate(band):
        hhmm = cross['time_utc'][11:16]
        if hhmm.endswith((':00', ':30')) and hhmm not in seen:
            seen.add(hhmm)
            times.append((i, hhmm))
    place(times, 'north', 1.2, 4.0, 7.0, fontsize=11, color=INK)
    place(sorted((idx[len(idx) // 2], name.upper())
                 for name, idx in on_centre.items()),
          'south', 1.2, 4.2, 8.0, fontsize=8.6, color=INK_2)

    ax.add_patch(Circle(view.point(eclipse['longitude'], eclipse['latitude']),
                        1.15, facecolor=INK, edgecolor=SURFACE, linewidth=1.8,
                        zorder=8))

    ax.set_xlim(*view.xlim)
    ax.set_ylim(*view.ylim)
    ax.set_aspect(view.aspect)
    for spine in ax.spines.values():
        spine.set_color(LAND_EDGE)
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.text(0.012, 0.955, title, fontsize=24, color=INK, va='top')
    fig.text(0.012, 0.905, lead, fontsize=13, color=INK_2, va='top')
    fig.text(0.012, 0.872, note, fontsize=13, color=INK_2, va='top')

    handles = [
        MplPolygon([(0, 0)], facecolor=SERIES_1, alpha=0.82,
                   label='Path of ' + band_label),
        Line2D([0], [0], color=CENTRE, lw=1.6, label='Centre line'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=INK,
               markeredgecolor=SURFACE, markeredgewidth=1.6, markersize=9,
               label='Greatest eclipse'),
    ]
    legend = ax.legend(handles=handles, loc='lower left', frameon=True,
                       fontsize=10.5, borderpad=0.8, labelspacing=0.75,
                       handlelength=1.7)
    legend.get_frame().set_facecolor(SURFACE)
    legend.get_frame().set_edgecolor(LAND_EDGE)
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(INK)

    west_to_east = sorted(anywhere, key=lambda name: min(anywhere[name]))
    if west_to_east:
        listed = (west_to_east[0] if len(west_to_east) == 1
                  else '%s and %s' % (', '.join(west_to_east[:-1]),
                                      west_to_east[-1]))
        summary = ('%s is visible from %s. Outside the band the eclipse is '
                   'partial.' % (band_label.capitalize(), listed))
    else:
        summary = ('No sampled point of the band falls on land. Elsewhere '
                   'the eclipse is partial.')
    fig.text(0.012, 0.028, summary, fontsize=10, color=INK_2, ha='left')
    fig.text(0.988, 0.028, 'Computed from JPL DE440s · times UTC · '
             'mean lunar limb k = 0.272281', fontsize=10, color=INK_2,
             ha='right')

    fig.subplots_adjust(left=0.012, right=0.988, top=0.845, bottom=0.075)
    fig.savefig(out, facecolor=SURFACE)
    return west_to_east


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eclipses', help='JSON written by eclipsepath')
    parser.add_argument('countries', help='Natural Earth countries GeoJSON')
    parser.add_argument('-o', '--out', default='eclipse_path.png')
    parser.add_argument('--index', type=int, default=0,
                        help='which eclipse in the file (default: the first)')
    args = parser.parse_args()

    with open(args.eclipses) as handle:
        eclipse = json.load(handle)['eclipses'][args.index]
    with open(args.countries) as handle:
        countries = json.load(handle)
    crossed = render(eclipse, countries, args.out)
    print('wrote %s — band crosses %s'
          % (args.out, ', '.join(crossed) if crossed else 'no land'))


if __name__ == '__main__':
    main()
