#!/usr/bin/env python3
"""Animate the Moon's shadow crossing the Earth.

Unlike ``plot_path.py``, this does not read ``eclipsepath --format json``.  It
cannot: that file describes the path over the whole eclipse, and an animation
needs the coverage everywhere at each instant, which only the ephemeris can
answer.  So this one drives the package directly.

    pip install matplotlib
    curl -sO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
    python3 examples/animate_shadow.py 2027-08-02 ne_110m_admin_0_countries.geojson

Every frame is drawn from :mod:`eclipsepath.shadow`, so the picture and the
path table are two views of one calculation.  The greyscale is the fraction of
the Sun's disc covered, the dark patch is the umbra at true size, and the
shadow stops at the sunrise line because past it nobody can see the Sun to
have it covered.
"""

import argparse
import datetime as dt
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter        # noqa: E402
from matplotlib.patches import Polygon as MplPolygon                # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eclipsepath import eclipse as ec, finder, output, shadow        # noqa: E402
from eclipsepath.catalog import TYPE_NAMES                            # noqa: E402
from eclipsepath import geometry as g                                # noqa: E402
from eclipsepath.ephemeris import Ephemeris                          # noqa: E402

# Roles from the reference data-visualisation palette, light mode.
SURFACE, INK, INK_2, GRIDLINE = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9'
LAND, LAND_EDGE, SEA = '#e5e4dd', '#cfcec5', '#f3f4f4'
UMBRA, UMBRA_EDGE, TRACK = '#0d3f78', '#2a78d6', '#1c5fae'
NIGHT = (0.08, 0.10, 0.18)

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']
CONTOUR_LEVELS = (0.2, 0.4, 0.6, 0.8)
GRATICULE = 15.0
TURNS = (-360.0, 0.0, 360.0)


# --- the data ---------------------------------------------------------------

def find_eclipse(ephem, date, samples=160):
    """The eclipse on ``date``, fully analysed."""
    year, month, day = (int(part) for part in date.split('-'))
    ts = ephem.timescale
    events = finder.find_events(ephem, ts.utc(year, month, day - 1).tt,
                                ts.utc(year, month, day + 1).tt)
    if not events:
        raise SystemExit('no solar eclipse on %s' % date)
    return ec.analyse(ephem, events[0], threshold=1.0, samples=samples)


def frames_for(eclipse, count, step, levels=CONTOUR_LEVELS, progress=None):
    """Everything each frame needs, computed once so the view can be fitted.

    Kept apart from the drawing on purpose: the numbers here are what the tests
    assert on, and a test that had to go through matplotlib to reach them would
    be checking the renderer rather than the geometry.
    """
    times = np.linspace(eclipse.tt_first_contact, eclipse.tt_last_contact, count)
    built = []
    for index, tt in enumerate(times):
        if progress:
            progress(index, count)
        grid = shadow.obscuration_grid(eclipse.window, tt, step=step)
        point, hit = ec.shadow_point(eclipse.window, np.atleast_1d(tt))
        lat, lon, _ = g.itrf_to_geodetic(point)
        built.append({
            'tt': float(tt),
            'grid': grid,
            'umbra': shadow.umbra_outline(eclipse.window, tt),
            'terminator': shadow.terminator(eclipse.window, tt),
            'contours': shadow.contours(eclipse.window, grid, levels),
            'centre': (float(lat[0]), float(lon[0])),
            'central': bool(hit[0]) and shadow.umbra_outline(
                eclipse.window, tt) is not None,
            'deepest': float(grid.obscuration.max()),
        })
    return built


def visible_extent(frames, pad=6.0):
    """A box holding every frame's shadow, so the view can stay still.

    A window that chased the shadow would make the Earth appear to slide about
    underneath it; fixing it once means the motion on screen is the shadow's.
    """
    lats, lons, reference = [], [], None
    for frame in frames:
        grid = frame['grid']
        rows, cols = np.nonzero(grid.obscuration > 0.0)
        if not rows.size:
            continue
        lats.extend(grid.latitudes[[rows.min(), rows.max()]])
        # Longitude needs care twice over.  A shadow straddling the seam
        # occupies both ends of the array, so the smallest and largest column
        # touched are a whole world apart; and unwrapping each frame about its
        # own centre is not enough either, because as the shadow crosses the
        # seam one frame comes back around +190 and the next around -170, and
        # the union of those spans everything.  One reference for all frames.
        centre_lon = frame['centre'][1]
        if reference is None:
            reference = centre_lon
        centre_lon -= 360.0 * round((centre_lon - reference) / 360.0)
        touched = grid.longitudes[cols]
        near = touched - 360.0 * np.round((touched - centre_lon) / 360.0)
        lons.extend([near.min(), near.max()])
    if not lats:
        return (-180.0, 180.0, -90.0, 90.0)
    west, east = min(lons) - pad, max(lons) + pad
    south, north = max(-90.0, min(lats) - pad), min(90.0, max(lats) + pad)
    if east - west >= 360.0:
        west, east = -180.0, 180.0
    return west, east, south, north


def hours_minutes(seconds):
    """A span of hours, which ``output.format_duration`` is not built for.

    That one exists to print the couple of minutes of totality and would call
    a three-hour eclipse 313m01.7s.
    """
    hours, rest = divmod(int(round(seconds)), 3600)
    return '%dh %02dm' % (hours, rest // 60) if hours else '%dm' % (rest // 60)


def describe(eclipse, timescale):
    """Title and standfirst, taken from the data rather than written in."""
    stamp = output.utc_iso(timescale, eclipse.tt_greatest)
    year, month, day = (int(v) for v in stamp[:10].split('-'))
    title = '%s solar eclipse of %d %s %d' % (TYPE_NAMES[eclipse.kind], day,
                                              MONTHS[month - 1], year)
    if eclipse.central:
        lead = ('The shadow crosses the Earth in %s; %s of %s at the greatest '
                'eclipse, in a band %.0f km wide.'
                % (hours_minutes(
                       (eclipse.tt_last_contact - eclipse.tt_first_contact) * 86400.0),
                   output.format_duration(eclipse.central_duration_seconds),
                   'totality' if eclipse.kind in 'TH' else 'annularity',
                   eclipse.path_width_km))
    else:
        lead = ('No part of the Earth sees the Sun wholly covered; at most '
                '%.1f%% of it is hidden.' % (eclipse.peak_obscuration * 100.0))
    return title, lead


# --- the drawing ------------------------------------------------------------

def land_rings(path):
    with open(path) as handle:
        import json
        data = json.load(handle)
    rings = []
    for feature in data.get('features', [data]):
        geometry = feature['geometry']
        if geometry['type'] == 'Polygon':
            rings.append(geometry['coordinates'][0])
        elif geometry['type'] == 'MultiPolygon':
            rings.extend(polygon[0] for polygon in geometry['coordinates'])
    return rings


def shade(grid, extent):
    """The coverage field as an image: white where the Sun is clear.

    Two separate darkenings, because they are two different things.  Night is
    where the Sun is below the horizon and has nothing to do with the eclipse;
    the shadow is the fraction of the Sun covered, and it stops at the sunrise
    line for the reason the outlines do.
    """
    covered = grid.obscuration
    down = grid.sun_altitude <= g.HORIZON_ALTITUDE_DEG
    # Coverage darkens the ground steeply near totality, which is what the eye
    # reports too: half the Sun gone is barely a dimming.
    darkness = np.clip(covered, 0.0, 1.0) ** 2.2
    rgba = np.zeros(covered.shape + (4,))
    rgba[..., 0:3] = 0.0
    rgba[..., 3] = 0.82 * darkness
    night = np.zeros_like(rgba)
    night[..., 0:3] = NIGHT
    night[..., 3] = np.where(down, 0.30, 0.0)
    # Three worlds side by side.  A view fitted to a shadow that crosses the
    # antimeridian runs past 180, and a single copy of the field would simply
    # stop there, leaving a hard edge across the picture where the data ended
    # rather than where the night did.
    return (np.concatenate([rgba] * 3, axis=1),
            np.concatenate([night] * 3, axis=1))


def render(eclipse, frames, timescale, rings, out, extent, fps, dpi):
    west, east, south, north = extent
    title, lead = describe(eclipse, timescale)

    # Size the figure to the map rather than fitting the map to a figure: at
    # one degree square, a whole-world frame and a narrow polar one want very
    # different shapes, and a fixed canvas letterboxes one of them.
    margin, header, footer = 0.55, 1.15, 0.95
    map_width = 9.6 - 2.0 * margin
    map_height = map_width * (north - south) / max(east - west, 1e-6)
    map_height = min(map_height, 7.5)
    figure_height = map_height + header + footer
    figure = plt.figure(figsize=(9.6, figure_height), facecolor=SURFACE)
    ax = figure.add_axes([margin / 9.6, footer / figure_height,
                          map_width / 9.6, map_height / figure_height])
    ax.set_facecolor(SEA)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    # Plate carree, degrees square.  Honest about being a projection with no
    # single true scale, rather than stretching to a parallel that means
    # nothing across a hundred degrees of latitude.
    ax.set_aspect(1.0)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=INK_2, labelsize=8)

    top = 1.0 - 0.22 / figure_height
    figure.text(margin / 9.6, top, title, fontsize=16, color=INK, va='top')
    figure.text(margin / 9.6, top - 0.40 / figure_height, lead, fontsize=10,
                color=INK_2, va='top')
    figure.text(margin / 9.6, 0.36 / figure_height,
                'Greyscale is the fraction of the Sun\'s disc hidden; contours '
                'at 20, 40, 60 and 80 per cent.  Gold is the sunrise line, '
                'past which the eclipse is below the horizon.',
                fontsize=7.5, color=INK_2, va='bottom')
    figure.text(margin / 9.6, 0.18 / figure_height,
                'Computed with eclipsepath from JPL DE440s.  Plate carree: no '
                'one scale is true across the frame.',
                fontsize=7.5, color=INK_2, va='bottom')

    # Land is drawn once and left alone; only the shadow moves.
    for turn in range(int(math.floor((west + 180.0) / 360.0)),
                      int(math.floor((east + 180.0) / 360.0)) + 1):
        offset = 360.0 * turn
        for ring in rings:
            xs = [p[0] + offset for p in ring]
            if max(xs) < west or min(xs) > east:
                continue
            ax.add_patch(MplPolygon([(x, p[1]) for x, p in zip(xs, ring)],
                                    closed=True, facecolor=LAND,
                                    edgecolor=LAND_EDGE, linewidth=0.4,
                                    zorder=1))
    for value in np.arange(-180.0, 180.1, GRATICULE):
        ax.axvline(value, color=GRIDLINE, linewidth=0.4, zorder=2)
    for value in np.arange(-90.0, 90.1, GRATICULE):
        ax.axhline(value, color=GRIDLINE, linewidth=0.4, zorder=2)

    grid = frames[0]['grid']
    # The samples are cell centres, so the image reaches half a cell beyond the
    # outermost of them.  Handing imshow the centres as if they were the edges
    # slides the whole field by half a step, which at half a degree is 28 km.
    half = 0.5 * grid.step
    image_extent = (float(grid.longitudes[0]) - half - 360.0,
                    float(grid.longitudes[-1]) + half + 360.0,
                    float(grid.latitudes[0]) - half,
                    float(grid.latitudes[-1]) + half)
    blank = np.zeros((grid.obscuration.shape[0],
                      grid.obscuration.shape[1] * 3, 4))
    night_layer = ax.imshow(blank, origin='lower', extent=image_extent,
                            zorder=3, interpolation='bilinear')
    shadow_layer = ax.imshow(blank, origin='lower', extent=image_extent,
                             zorder=4, interpolation='bilinear')
    # Everything drawn as a line has to be repeated a turn either side.  A
    # curve is handed over with its longitudes made continuous, which puts it
    # somewhere within one turn of wherever it started; the terminator's far
    # branch begins near the antimeridian and lands at +208, off the right of a
    # view that badly needs it on the left.
    class Repeated:
        """One polyline, drawn once per turn of longitude the view spans."""

        def __init__(self, count=1, **style):
            self.artists = [[ax.plot([], [], **style)[0] for _ in TURNS]
                            for _ in range(count)]

        def set(self, index, lat, lon):
            for artist, turn in zip(self.artists[index], TURNS):
                artist.set_data(np.asarray(lon) + turn, lat)

        def clear(self, index):
            for artist in self.artists[index]:
                artist.set_data([], [])

        def clear_from(self, index):
            for spare in range(index, len(self.artists)):
                self.clear(spare)

    track = Repeated(1, color=TRACK, linewidth=1.1, zorder=5,
                     linestyle=(0, (5, 3)))
    edge = Repeated(1, color='#d4a017', linewidth=1.0, zorder=6)
    room = max(sum(len(v) for v in f['contours'].values()) for f in frames)
    lines = Repeated(room, color=INK_2, linewidth=0.55, alpha=0.7, zorder=5)
    umbra_patches = []
    for _ in TURNS:
        patch = MplPolygon([(0.0, 0.0)], closed=True, facecolor=UMBRA,
                           edgecolor=UMBRA_EDGE, linewidth=0.9, zorder=7)
        patch.set_visible(False)
        ax.add_patch(patch)
        umbra_patches.append(patch)
    clock = ax.text(0.012, 0.965, '', transform=ax.transAxes, fontsize=11,
                    color=INK, va='top', family='monospace', zorder=9,
                    bbox=dict(facecolor=SURFACE, edgecolor=GRIDLINE,
                              boxstyle='round,pad=0.4', alpha=0.92))

    def wrap_near(lon, reference):
        return lon - 360.0 * np.round((lon - reference) / 360.0)

    def draw(index):
        frame = frames[index]
        centre_lon = frame['centre'][1]
        rgba, night = shade(frame['grid'], extent)
        shadow_layer.set_data(rgba)
        night_layer.set_data(night)

        history = [f['centre'] for f in frames[:index + 1] if f['central']]
        if history:
            track.set(0, [p[0] for p in history],
                      [wrap_near(p[1], centre_lon) for p in history])
        else:
            track.clear(0)

        for patch, turn in zip(umbra_patches, TURNS):
            if frame['umbra'] is None:
                patch.set_visible(False)
                continue
            lat, lon = frame['umbra']
            patch.set_xy(np.column_stack([wrap_near(lon, centre_lon) + turn,
                                          lat]))
            patch.set_visible(True)

        lat, lon = frame['terminator']
        edge.set(0, lat, lon)

        drawn = 0
        for _level, polylines in sorted(frame['contours'].items()):
            for lat, lon, _closed in polylines:
                lines.set(drawn, lat, lon)
                drawn += 1
        lines.clear_from(drawn)

        stamp = output.utc_iso(timescale, frame['tt'])
        clock.set_text('%s UTC   %5.1f%% covered'
                       % (stamp[11:19], 100.0 * frame['deepest']))

        return []

    animation = FuncAnimation(figure, draw, frames=len(frames), blit=False)
    writer = writer_for(out, fps)
    animation.save(out, writer=writer, dpi=dpi,
                   savefig_kwargs={'facecolor': SURFACE})
    plt.close(figure)
    return out


def writer_for(out, fps):
    """An MP4 writer if ffmpeg is about, otherwise Pillow's GIF writer."""
    if out.lower().endswith('.mp4'):
        from matplotlib.animation import FFMpegWriter
        for candidate in (matplotlib.rcParams['animation.ffmpeg_path'],
                          '/opt/pw-browsers/ffmpeg-1011/ffmpeg-linux',
                          'ffmpeg'):
            if candidate and (os.path.exists(candidate) or candidate == 'ffmpeg'):
                matplotlib.rcParams['animation.ffmpeg_path'] = candidate
                if FFMpegWriter.isAvailable():
                    return FFMpegWriter(fps=fps, bitrate=2400)
        raise SystemExit('no ffmpeg available for MP4; write a .gif instead')
    return PillowWriter(fps=fps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('date', help='date of the eclipse, YYYY-MM-DD')
    parser.add_argument('countries', help='Natural Earth GeoJSON')
    parser.add_argument('--out', default=None, help='.gif or .mp4 to write')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--step', type=float, default=0.5,
                        help='grid spacing in degrees (default: 0.5)')
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--dpi', type=int, default=110)
    parser.add_argument('--kernel', default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    def note(message):
        if not args.quiet:
            print(message, file=sys.stderr)

    note('loading ephemeris...')
    ephem = Ephemeris(args.kernel)
    note('finding the eclipse...')
    eclipse = find_eclipse(ephem, args.date)
    out = args.out or 'eclipse_%s_shadow.gif' % args.date.replace('-', '')

    def progress(index, total):
        if index % 10 == 0:
            note('  frame %d/%d' % (index + 1, total))

    note('computing %d frames...' % args.frames)
    frames = frames_for(eclipse, args.frames, args.step, progress=progress)
    note('drawing...')
    render(eclipse, frames, ephem.timescale, land_rings(args.countries), out,
           visible_extent(frames), args.fps, args.dpi)
    central = sum(1 for f in frames if f['central'])
    print('wrote %s: %d frames, %d with the umbra on the ground'
          % (out, len(frames), central))
    return 0


if __name__ == '__main__':
    sys.exit(main())
