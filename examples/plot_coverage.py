#!/usr/bin/env python3
"""Render an eclipse as nested coverage bands over a map.

The path of totality with, around it, the areas seeing at least 80, 60, 40, 20
and 1 percent of the Sun covered.  Contours come from
``eclipsepath.coverage_region``.

    python3 examples/plot_coverage.py contours.json ne_110m_admin_0_countries.geojson
"""

import argparse
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Polygon as MplPolygon               # noqa: E402

SURFACE, INK, INK_2, GRIDLINE = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e0'
LAND, LAND_EDGE, SEA = '#e5e4dd', '#b9b8ae', '#f3f4f4'
# Sequential encoding: one hue, laid down once per level.  Because the areas
# nest, each layer darkens everything inside the one before it, so a single
# translucent hue produces the whole light-to-dark ramp — and the land stays
# visible underneath, which an opaque ramp would bury.
COVERAGE_HUE = '#17549f'
LAYER_ALPHA = 0.26
TOTALITY = '#0b2f5e'

TITLE = 'Total solar eclipse of 2 August 2027'
LEAD = ('Totality is a 258 km ribbon from the Atlantic to the Indian Ocean. Some '
        'part of the eclipse is visible from 23% of the Earth’s surface.')


def rings(geometry):
    if geometry['type'] == 'Polygon':
        return [geometry['coordinates'][0]]
    if geometry['type'] == 'MultiPolygon':
        return [polygon[0] for polygon in geometry['coordinates']]
    return []


def to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def over(foreground, alpha, background):
    """Composite ``foreground`` at ``alpha`` onto ``background``."""
    return tuple(alpha * f + (1 - alpha) * b
                 for f, b in zip(foreground, background))


def render(data, countries, out):
    eclipse = data['eclipse']
    contours = sorted(data['contours'], key=lambda c: c['threshold'])
    band = next(item for item in eclipse['geometries']
                if item['kind'] == 'band')['cross_sections']
    north = [(c['north_longitude'], c['north_latitude']) for c in band]
    south = [(c['south_longitude'], c['south_latitude']) for c in band]

    outer = contours[0]['boundary']
    lon0 = min(p[0] for p in outer) - 6
    lon1 = max(p[0] for p in outer) + 6
    lat0 = min(p[1] for p in outer) - 5
    lat1 = max(p[1] for p in outer) + 5

    fig, ax = plt.subplots(figsize=(16.4, 11.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SEA)

    for feature in countries['features']:
        for ring in rings(feature['geometry']):
            ax.add_patch(MplPolygon(ring, closed=True, facecolor=LAND,
                                    edgecolor='none', zorder=1))

    for lon in range(-180, 181, 30):
        ax.plot([lon, lon], [lat0, lat1], color=GRIDLINE, lw=0.6, zorder=1.5)
    for lat in range(-90, 91, 30):
        ax.plot([lon0, lon1], [lat, lat], color=GRIDLINE, lw=0.6, zorder=1.5)

    # Outermost first, so each nested area is laid over the last.
    for contour in contours:
        ax.add_patch(MplPolygon(contour['boundary'], closed=True,
                                facecolor=COVERAGE_HUE, alpha=LAYER_ALPHA,
                                edgecolor='none', zorder=2))
    ax.add_patch(MplPolygon(north + south[::-1], closed=True,
                            facecolor=TOTALITY, edgecolor='none', zorder=4))

    # Coastlines on top: the fills are translucent, so these still read.
    for feature in countries['features']:
        for ring in rings(feature['geometry']):
            ax.add_patch(MplPolygon(ring, closed=True, facecolor='none',
                                    edgecolor=LAND_EDGE, linewidth=0.55,
                                    alpha=0.85, zorder=3))

    # Label each contour where it runs furthest north.
    for contour in contours:
        top = max(contour['boundary'], key=lambda p: p[1])
        ax.text(top[0], top[1] + 1.2, '%g%%' % (contour['threshold'] * 100),
                ha='center', va='bottom', fontsize=10, color=INK, zorder=6,
                bbox=dict(boxstyle='round,pad=0.24', fc=SURFACE, ec='none',
                          alpha=0.86))

    ax.set_xlim(lon0, lon1)
    ax.set_ylim(lat0, lat1)
    ax.set_aspect(1.0 / math.cos(math.radians(20)))
    for spine in ax.spines.values():
        spine.set_color(LAND_EDGE)
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.text(0.012, 0.966, TITLE, fontsize=24, color=INK, va='top')
    fig.text(0.012, 0.928, LEAD, fontsize=13, color=INK_2, va='top')

    # Legend swatches are the composited colours, so they match the map.
    land = to_rgb(LAND)
    hue = to_rgb(COVERAGE_HUE)
    stack, colour = [], land
    for _ in contours:
        colour = over(hue, LAYER_ALPHA, colour)
        stack.append(colour)
    bounds = [c['threshold'] for c in contours] + [1.0]
    handles = [MplPolygon([(0, 0)], facecolor=TOTALITY,
                          label='100%  — totality')]
    for i in range(len(contours) - 1, -1, -1):
        handles.append(MplPolygon([(0, 0)], facecolor=stack[i],
                                  label='%g–%g%%' % (bounds[i] * 100,
                                                     bounds[i + 1] * 100)))
    handles.append(MplPolygon([(0, 0)], facecolor=LAND, edgecolor=LAND_EDGE,
                              linewidth=0.6, label='below 1%  — no eclipse'))
    legend = ax.legend(handles=handles, loc='lower left', frameon=True,
                       fontsize=10.5, borderpad=0.8, labelspacing=0.62,
                       handlelength=1.9, title='Sun covered at maximum')
    legend.get_frame().set_facecolor(SURFACE)
    legend.get_frame().set_edgecolor(LAND_EDGE)
    legend.get_frame().set_linewidth(0.8)
    legend.get_title().set_color(INK)
    legend.get_title().set_fontsize(10.5)
    for text in legend.get_texts():
        text.set_color(INK)

    fig.text(0.012, 0.022, 'Each contour is where the deepest moment of the '
             'eclipse reaches that coverage. The outer edge of the palest band '
             'is the limit of visibility.', fontsize=10, color=INK_2, ha='left')
    fig.text(0.988, 0.022, 'Computed from JPL DE440s · '
             'mean lunar limb k = 0.272281', fontsize=10, color=INK_2, ha='right')

    fig.subplots_adjust(left=0.012, right=0.988, top=0.905, bottom=0.055)
    fig.savefig(out, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('contours', help='JSON with eclipse + contour outlines')
    parser.add_argument('countries', help='Natural Earth countries GeoJSON')
    parser.add_argument('-o', '--out', default='eclipse_coverage.png')
    args = parser.parse_args()
    with open(args.contours) as handle:
        data = json.load(handle)
    with open(args.countries) as handle:
        countries = json.load(handle)
    render(data, countries, args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
