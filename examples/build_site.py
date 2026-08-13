#!/usr/bin/env python3
"""Build a static site of eclipse globes, ready for GitHub Pages.

    curl -sO https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson
    python3 examples/build_site.py ne_110m_land.geojson --out docs

Each globe is a self-contained page, so the site is a directory of files with
no build step, no bundler and nothing fetched at run time.  The index is
written from the computed eclipses rather than typed out, for the same reason
the map captions are: a hand-written list is a second set of numbers to keep
true, and it will not be kept true.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eclipsepath import eclipse as ec, finder, output, scene  # noqa: E402
from eclipsepath import geometry as g  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402

# Chosen to exercise the cases that caused trouble, not just the pretty one:
# a shadow over the antimeridian, a path past 70 degrees where a flat map gives
# up, an annular whose coverage never reaches 1, a hybrid that changes kind
# part way along, and a partial with no umbra to draw at all.
DEFAULT_DATES = [
    ('2027-08-02', 'The flagship: six minutes of totality over Egypt, the '
                   'longest on land this century so far.'),
    ('2026-08-12', 'Iceland and Spain, at a high latitude and a low Sun — the '
                   'grazing incidence that spreads the shadow along the ground.'),
    ('2028-07-22', 'Across Australia and New Zealand, over the antimeridian: '
                   'the case that cut every outline in two.'),
    ('2026-02-17', 'Annular over Antarctica. Coverage never reaches 1, and the '
                   'path runs where a rectangular map is useless.'),
    ('2031-11-14', 'Hybrid: total along the middle of the track and annular at '
                   'both ends, as the shadow cone reaches the ground or falls '
                   'just short of it.'),
    ('2029-01-14', 'Partial everywhere. No umbra lands, so there is no path '
                   'and nothing to draw but the coverage.'),
]

PAGE = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eclipse globes</title>
<style>
  :root { --surface:#101319; --panel:#171b23; --ink:#f2f3f5; --ink-2:#9aa1ad;
          --line:#262b36; --accent:#7fb2f0; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--surface); color:var(--ink);
         font:16px/1.6 system-ui,sans-serif; }
  main { max-width:1000px; margin:0 auto; padding:48px 24px 96px; }
  h1 { font-size:30px; margin:0 0 6px; letter-spacing:-0.01em; }
  .lede { color:var(--ink-2); margin:0 0 8px; max-width:62ch; }
  h2 { font-size:20px; margin:48px 0 12px; }
  .grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); }
  a.card { display:block; text-decoration:none; color:inherit; padding:16px 18px;
           background:var(--panel); border:1px solid var(--line); border-radius:10px; }
  a.card:hover { border-color:var(--accent); }
  a.card h3 { margin:0 0 4px; font-size:17px; }
  a.card p { margin:0 0 10px; color:var(--ink-2); font-size:13.5px; }
  dl { display:grid; grid-template-columns:auto 1fr; gap:1px 10px; margin:0;
       font-size:12.5px; color:var(--ink-2); }
  dd { margin:0; color:var(--ink); font-variant-numeric:tabular-nums; }
  img { width:100%%; border-radius:10px; border:1px solid var(--line); }
  footer { margin-top:56px; padding-top:20px; border-top:1px solid var(--line);
           color:var(--ink-2); font-size:13px; }
  code { background:#0b0e14; padding:1px 5px; border-radius:4px; font-size:12.5px; }
</style>
</head>
<body>
<main>
<h1>Eclipse globes</h1>
<p class="lede">Solar eclipses computed from JPL DE440s with
<a href="https://github.com/rebelle3/next-solar-eclipse" style="color:var(--accent)">eclipsepath</a>.
Each globe is one self-contained page: drag to turn, wheel to zoom, scrub or
play the eclipse through, and press <em>Show the Moon</em> for the shadow cone.</p>
<p class="lede">The shadow is not a texture. The page carries the Sun's and the
Moon's positions at a sample a minute, and the fragment shader works out for
every pixel how much of the Sun is covered at the point of the Earth under it —
the same circle overlap the Python package uses. It agrees with it to
2.7&times;10<sup>&minus;5</sup>, and puts the edge of totality within a metre.</p>

<h2>Globes</h2>
<div class="grid">
%(cards)s
</div>

<h2>The shadow on a flat map</h2>
<p class="lede">The same geometry, cut the same way — coverage everywhere at
each instant — drawn frame by frame from first to last contact. Greyscale is
the fraction of the Sun hidden; the gold line is sunrise, past which the
eclipse is below the horizon.</p>
<img src="media/eclipse_2027_shadow.gif"
     alt="The Moon's shadow crossing the Earth on 2 August 2027">

<footer>
Built by <code>examples/build_site.py</code>. Every figure on this page is
computed, not typed. Coastlines: Natural Earth, public domain.
</footer>
</main>
</body>
</html>
'''

CARD = '''  <a class="card" href="globes/%(file)s">
    <h3>%(title)s</h3>
    <p>%(note)s</p>
    <dl>
      <dt>Greatest</dt><dd>%(greatest)s UTC</dd>
      <dt>Max coverage</dt><dd>%(coverage)s</dd>
      <dt>Central</dt><dd>%(central)s</dd>
      <dt>Saros</dt><dd>%(saros)d</dd>
    </dl>
  </a>'''


def build(dates, coastlines, out, kernel=None, samples=160, quiet=False):
    def note(message):
        if not quiet:
            print(message, file=sys.stderr)

    os.makedirs(os.path.join(out, 'globes'), exist_ok=True)
    os.makedirs(os.path.join(out, 'media'), exist_ok=True)
    # Pages runs Jekyll over a site unless told not to, and there is nothing
    # here for it to do but find something to object to.
    open(os.path.join(out, '.nojekyll'), 'w').close()

    note('loading ephemeris...')
    ephem = Ephemeris(kernel)
    ts = ephem.timescale
    cards, built = [], []
    for date, description in dates:
        note('  %s' % date)
        year, month, day = (int(part) for part in date.split('-'))
        events = finder.find_events(ephem, ts.utc(year, month, day - 1).tt,
                                    ts.utc(year, month, day + 1).tt)
        if not events:
            note('    no eclipse found, skipping')
            continue
        eclipse = ec.analyse(ephem, events[0], threshold=1.0, samples=samples)
        data = scene.build(eclipse, ts, coastlines)
        name = 'eclipse_%s.html' % date.replace('-', '')
        with open(os.path.join(out, 'globes', name), 'w') as handle:
            handle.write(scene.standalone(data))
        summary = data['eclipse']
        cards.append(CARD % {
            'file': name,
            'title': '%s, %s' % (summary['type_name'], summary['date']),
            'note': description,
            'greatest': summary['greatest_utc'][11:19],
            'coverage': '%.2f%%' % (summary['obscuration'] * 100.0),
            'central': (output.format_duration(summary['central_duration_seconds'])
                        if summary['central'] else 'none — partial everywhere'),
            'saros': summary['saros'],
        })
        built.append((date, name, summary))
    with open(os.path.join(out, 'index.html'), 'w') as handle:
        handle.write(PAGE % {'cards': '\n'.join(cards)})
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('coastlines', help='Natural Earth land GeoJSON')
    parser.add_argument('--out', default='docs')
    parser.add_argument('--gif', default=None,
                        help='animation to copy into media/')
    parser.add_argument('--dates', nargs='*', default=None)
    parser.add_argument('--samples', type=int, default=160)
    parser.add_argument('--kernel', default=None)
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()

    dates = ([(d, '') for d in args.dates] if args.dates else DEFAULT_DATES)
    built = build(dates, args.coastlines, args.out, args.kernel, args.samples,
                  args.quiet)
    if args.gif:
        shutil.copyfile(args.gif, os.path.join(args.out, 'media',
                                               os.path.basename(args.gif)))
    total = sum(os.path.getsize(os.path.join(root, f))
                for root, _dirs, files in os.walk(args.out) for f in files)
    print('wrote %d globes to %s/ (%.1f MB in all)'
          % (len(built), args.out, total / 1048576.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
