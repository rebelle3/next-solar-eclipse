#!/usr/bin/env python3
"""Scrape NASA's published Besselian elements into tests/data.

One page per eclipse from eclipse.gsfc.nasa.gov, holding the polynomial
elements Fred Espenak generated from VSOP87/ELP2000-82 for the Five Millennium
Canon, together with the delta-T and the two lunar radius constants his
predictions assume.  Reproduction is permitted with the acknowledgment
"Eclipse Predictions by Fred Espenak, NASA's GSFC", which tests/data/SOURCES
carries.
"""

import json
import os
import re
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'nasa_besselian_elements.json')
URL = 'https://eclipse.gsfc.nasa.gov/SEsearch/SEdata.php?Ecl=%s'

NUMBER = r'[-+]?\d+\.\d+'


def text_of(html):
    body = re.sub(r'<[^>]+>', ' ', html)
    body = body.replace('&nbsp;', ' ').replace('&Delta;', 'D')
    return re.sub(r'[ \t]+', ' ', body)


def scrape(stamp):
    with urllib.request.urlopen(URL % stamp, timeout=60) as handle:
        page = text_of(handle.read().decode('utf-8', 'replace'))

    def one(pattern, cast=float):
        found = re.search(pattern, page)
        if not found:
            raise ValueError('%s: no match for %s' % (stamp, pattern))
        return cast(found.group(1))

    # The elements table runs from the "n x y d l1 l2 mu" header to "tan f1";
    # PHP warnings are interleaved in some pages, so take the numbers in order
    # rather than trusting the row layout.
    block = page[page.index('Polynomial Besselian Elements'):page.index('tan f1')]
    block = re.sub(r'Warning:.*?on line \d+', ' ', block)
    numbers = re.findall(NUMBER, block)
    # The first number is t0, then four rows of six, the last row short by two
    # in the source but padded with zeros on the page.
    t0 = float(numbers[0])
    values = [float(v) for v in numbers[1:]]
    rows = []
    index = 0
    for order in range(4):
        # Each row starts with its own order number, which is an integer and so
        # not picked up by the decimal pattern above.
        rows.append(values[index:index + 6])
        index += 6
    if any(len(row) != 6 for row in rows):
        raise ValueError('%s: elements table did not parse: %r' % (stamp, rows))

    return {
        'stamp': stamp,
        'type': one(r'Eclipse Type = (\w)', str),
        'gamma': one(r'Gamma = (%s)' % NUMBER),
        'magnitude': one(r'Eclipse Magnitude = (%s)' % NUMBER),
        'delta_t': one(r'DT = (%s) s' % NUMBER),
        'k_penumbra': one(r'k1 = (%s)' % NUMBER),
        'k_umbra': one(r'k2 = (%s)' % NUMBER),
        't0_hours': t0,
        'x': [row[0] for row in rows],
        'y': [row[1] for row in rows],
        'd': [row[2] for row in rows],
        'l1': [row[3] for row in rows],
        'l2': [row[4] for row in rows],
        'mu': [row[5] for row in rows],
        'tan_f1': one(r'tan f1 = (%s)' % NUMBER),
        'tan_f2': one(r'tan f2 = (%s)' % NUMBER),
    }


def main():
    stamps = sys.argv[1:]
    if not stamps:
        print('usage: scrape_besselian.py YYYYMMDD [YYYYMMDD ...]')
        return 1
    saved = {}
    if os.path.exists(OUT):
        with open(OUT) as handle:
            saved = json.load(handle)
    for stamp in stamps:
        if stamp in saved:
            continue
        try:
            saved[stamp] = scrape(stamp)
            print('%s %s gamma %+.4f  dT %.1fs  k1 %.6f k2 %.6f'
                  % (stamp, saved[stamp]['type'], saved[stamp]['gamma'],
                     saved[stamp]['delta_t'], saved[stamp]['k_penumbra'],
                     saved[stamp]['k_umbra']))
        except Exception as error:                     # noqa: BLE001
            print('%s FAILED: %s' % (stamp, error))
        time.sleep(0.4)
    with open(OUT, 'w') as handle:
        json.dump(saved, handle, indent=1, sort_keys=True)
    print('%d eclipses in %s' % (len(saved), OUT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
