#!/usr/bin/env python3
"""Render an eclipse path as a map image.

An example of consuming ``eclipsepath --format json``.  Everything drawn comes
out of that file; only the headline prose is written by hand.

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

TITLE = 'Total solar eclipse of 2 August 2027'
NOTE = ('Greatest eclipse is at 10:06 UTC over southern Egypt near Luxor, where '
        'totality lasts 6m 22s — the longest of any total eclipse until June 2114.')


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
    band = next(item for item in eclipse['geometries']
                if item['kind'] == 'band')['cross_sections']
    centre = [(c['longitude'], c['latitude']) for c in band]
    north = [(c['north_longitude'], c['north_latitude']) for c in band]
    south = [(c['south_longitude'], c['south_latitude']) for c in band]

    lons = [p[0] for p in north + south]
    lats = [p[1] for p in north + south]
    lon0, lon1 = min(lons) - 5, max(lons) + 5
    lat0, lat1 = min(lats) - 7, max(lats) + 7

    def offset_from(index, side, distance):
        """A point ``distance`` degrees off the band, square to it.

        Offsetting square to the band rather than straight up keeps a label
        clear of it however the track happens to be running.
        """
        cx, cy = centre[index]
        ex, ey = (north if side == 'north' else south)[index]
        dx, dy = ex - cx, ey - cy
        length = math.hypot(dx, dy) or 1.0
        return cx + dx / length * distance, cy + dy / length * distance

    fig, ax = plt.subplots(figsize=(17.5, 8.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SEA)

    on_centre, anywhere = {}, {}
    for feature in countries['features']:
        name = feature['properties']['NAME']
        for ring in rings(feature['geometry']):
            ax.add_patch(MplPolygon(ring, closed=True, facecolor=LAND,
                                    edgecolor=LAND_EDGE, linewidth=0.6, zorder=1))
            # A country sees totality if any part of the band reaches it, which
            # for some is only an edge and never the centre line.
            for i in range(len(centre)):
                for track in (centre, north, south):
                    if inside(track[i][0], track[i][1], ring):
                        anywhere.setdefault(name, []).append(i)
                        if track is centre:
                            on_centre.setdefault(name, []).append(i)

    for lon in range(-180, 181, 15):
        ax.plot([lon, lon], [lat0, lat1], color=GRIDLINE, lw=0.5, zorder=0.5)
    for lat in range(-90, 91, 15):
        ax.plot([lon0, lon1], [lat, lat], color=GRIDLINE, lw=0.5, zorder=0.5)

    ax.add_patch(MplPolygon(north + south[::-1], closed=True, facecolor=SERIES_1,
                            alpha=0.82, edgecolor='none', zorder=3))
    for edge in (north, south):
        ax.plot([p[0] for p in edge], [p[1] for p in edge], color=BAND_EDGE,
                lw=0.7, alpha=0.9, zorder=4)
    ax.plot([p[0] for p in centre], [p[1] for p in centre], color=CENTRE,
            lw=1.3, zorder=5)

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

    ax.add_patch(Circle((eclipse['longitude'], eclipse['latitude']), 1.15,
                        facecolor=INK, edgecolor=SURFACE, linewidth=1.8, zorder=8))

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / math.cos(math.radians(20)))
    for spine in ax.spines.values():
        spine.set_color(LAND_EDGE)
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.text(0.012, 0.955, TITLE, fontsize=24, color=INK, va='top')
    fig.text(0.012, 0.905, 'The Moon’s shadow crosses %d km wide, from the '
             'Atlantic to the Indian Ocean, between %s and %s UTC.'
             % (round(eclipse['path_width_km']),
                eclipse['central_phase_start_utc'][11:16],
                eclipse['central_phase_end_utc'][11:16]),
             fontsize=13, color=INK_2, va='top')
    fig.text(0.012, 0.872, NOTE, fontsize=13, color=INK_2, va='top')

    handles = [
        MplPolygon([(0, 0)], facecolor=SERIES_1, alpha=0.82,
                   label='Path of totality'),
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
    fig.text(0.012, 0.028, 'Totality is visible from %s and %s. Outside the band '
             'the eclipse is partial.'
             % (', '.join(west_to_east[:-1]), west_to_east[-1]),
             fontsize=10, color=INK_2, ha='left')
    fig.text(0.988, 0.028, 'Computed from JPL DE440s · times UTC · '
             'mean lunar limb k = 0.272281', fontsize=10, color=INK_2, ha='right')

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
    print('wrote %s — totality visible from %s' % (args.out, ', '.join(crossed)))


if __name__ == '__main__':
    main()
