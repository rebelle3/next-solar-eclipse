# eclipsepath

Find every solar eclipse in a date range and trace, as latitude/longitude with
timings, the strip of the Earth's surface where the Sun is covered by at least
a given percentage. Output is plain text, JSON, CSV or GeoJSON, so a path can
be dropped straight onto a map.

By default it lists the next ten years of **total** eclipses — a 100% coverage
filter — and draws each path of totality.

```
$ eclipsepath --years 10

Solar eclipses found: 23   (8 reach 100% obscuration)

  Date         Greatest (UTC)  Type      Mag   Cover%   Gamma  Saros   Width      Duration   Greatest eclipse at
  --------------------------------------------------------------------------------------------------------------
  2026-08-12  17:45:57  Total    1.039 100.00%  +0.898   126     293 km    2m18.1s   65.22N, 25.24W
  2027-08-02  10:06:41  Total    1.079 100.00%  +0.142   136     258 km    6m22.4s   25.50N, 33.17E
  2028-07-22  02:55:30  Total    1.056 100.00%  -0.606   146     230 km    5m09.5s   15.58S, 126.69E
  ...

====================================================================================================
2027-08-02  Total solar eclipse  -  path of totality
  eclipse begins 07:30:10 UTC, ends 12:43:12 UTC   saros 136, lunation 341
----------------------------------------------------------------------------------------------------
   Time UTC   Centre line          Northern limit       Southern limit         Width   Duration  Sun
----------------------------------------------------------------------------------------------------
   09:02:14   35.345N    5.214E   36.436N    5.383E   34.255N    5.050E     244 km   5m26.8s   50
   09:44:20   29.891N   25.441E   30.900N   26.082E   28.881N   24.814E     255 km   6m19.0s   75
   10:26:25   20.993N   39.266E   21.873N   40.088E   20.111N   38.455E     259 km   6m13.2s   76
   ...
```

## Install

```
pip install skyfield numpy
```

The JPL ephemeris kernel (`de440s.bsp`, 32 MB, covering 1849–2150) is
downloaded to `~/.eclipsepath/` on first use, or point at an existing copy with
`--kernel`.

## Use

```
eclipsepath                                  # next 10 years of total eclipses
eclipsepath --years 3 --coverage 90          # 3 years, anywhere reaching 90%
eclipsepath --start 2030-01-01 --end 2035-01-01
eclipsepath --format geojson -o paths.geojson
eclipsepath --format csv -o paths.csv
eclipsepath --at 51.48,-3.18                 # what Cardiff sees each time
eclipsepath --all                            # include eclipses below the filter
```

`--coverage` is the percentage of the Sun's **disc area** hidden at the moment
of maximum eclipse (obscuration), not the fraction of its diameter (magnitude).
100% therefore selects total eclipses only: an annular eclipse never covers the
whole disc, so at its very deepest a few percent of the Sun is still showing as
a ring. Annular and hybrid eclipses still have their full antumbral/umbral path
computed — drop the filter to about 85% to bring them in.

As a library:

```python
from eclipsepath import search, circumstances_at

for eclipse in search('2026-08-12', years=10, threshold=1.0):
    if not eclipse.meets_threshold:
        continue
    print(eclipse.kind, eclipse.central_duration_seconds)
    for point in eclipse.coverage_band.points:
        print(point.latitude, point.longitude, point.tt_start, point.duration_seconds)
```

### GeoJSON

Each eclipse becomes a `greatest-eclipse` point, a `shadow-track` line, and —
for anything reaching the threshold — `centre-line`, `northern-limit` and
`southern-limit` lines plus a filled `band` polygon. Every line feature carries
a `times` array parallel to its coordinates. Geometries are cut where they
cross the antimeridian so nothing is drawn the wrong way round the globe.

## How it works

Rather than the classical Besselian elements, the geometry is solved directly
as vectors, which keeps every quantity defined by what an observer actually
sees.

1. **Positions.** Geocentric astrometric Sun and Moon vectors come from the JPL
   DE440s ephemeris via Skyfield — light-time corrected, so each body sits
   where it was when the light now arriving left it, which is what casting a
   shadow depends on. Aberration and deflection are deliberately not applied:
   they displace both bodies almost identically and cancel in the geometry.
2. **Finding eclipses.** A coarse pass over the range keeps the new moons where
   the shadow axis comes near the Earth; a fine pass then decides whether the
   penumbral cone actually touches the globe and brackets first and last
   contact. Greatest eclipse is the instant the axis passes closest to the
   Earth's centre.
3. **Local circumstances.** For a site, the apparent radii of the two discs and
   their separation give obscuration from the standard circle-overlap area, and
   magnitude from the covered fraction of the diameter. The deepest moment is
   found by golden-section search on the separation — or, for a site that only
   catches the eclipse around sunrise or sunset, at the instant the Sun's centre
   reaches the horizon.
4. **The track.** At each instant the darkest point on the globe is where the
   shadow axis pierces the WGS84 ellipsoid. When the axis misses the Earth
   entirely, as it does throughout a partial eclipse, a pattern search on the
   surface finds the deepest point instead, so the track stays defined.
5. **The band.** The region for a threshold X is `{p : max over t of
   obscuration(p, t) >= X}`. Its edges are found by bisecting outwards from the
   track along the perpendicular to the shadow's motion — the perpendicular
   comes from the shadow's velocity over a two-second baseline, since a track
   passing near a pole can swing tens of degrees in a couple of minutes. The
   path of totality is the same construction with the umbral criterion, "the
   Moon's disc lies wholly inside the Sun's, or wholly contains it", which is
   also what gives an annular eclipse its path.

Inside each eclipse the Sun, Moon and Earth-orientation are modelled by
Chebyshev series fitted to exact values, accurate to under a centimetre for the
Moon and under a millimetre of surface displacement for the rotation. That is
what makes the tens of thousands of evaluations behind the boundary searches
cheap enough to run: a ten-year scan with full paths takes about 50 seconds.

### Constants and conventions

| Quantity | Value | Note |
| --- | --- | --- |
| Ephemeris | JPL DE440s | 1849–2150 |
| Solar radius | 696 000 km | semi-diameter 959.63″ at 1 au |
| Lunar radius | k = 0.2722810 | `--lunar-radius iau` for 0.2725076 |
| Earth | WGS84 ellipsoid | a = 6378.137 km, f = 1/298.257223563 |
| Sun altitude | geometric | no refraction; visibility requires the centre above the horizon |

Two values of k are in use. The IAU (1982) mean lunar radius, k = 0.2725076,
averages over the peaks and valleys of the limb. Espenak adopts a smaller mean
minimum radius, k = 0.2722810, for umbral and antumbral contacts, because the
last of the photosphere shines through lunar valleys and the smaller figure
reproduces observed contact times and path limits better. The smaller value is
the default here, so results line up with published catalogues; it is worth
about 2% on path width and central duration.

## Accuracy

Verified against NASA/Espenak's published tables. `tests/verify_against_nasa.py`
compares full path tables row by row; `tests/test_eclipsepath.py` covers the
Five Millennium Catalog and the geometry itself.

All 25 solar eclipses from 2026 to 2036 are found, each classified correctly as
total, annular, hybrid or partial, with:

| Quantity | Agreement with NASA |
| --- | --- |
| Time of greatest eclipse | within 0.5 s (all 25) |
| Gamma | within 0.0005 |
| Eclipse magnitude | within 0.0006 |
| Position of greatest eclipse | within the catalogue's 1° rounding |
| Central duration | within 0.6 s |
| Saros and lunation number | exact |

Against the 2026 Aug 12 (total) and 2026 Feb 17 (annular) path tables, over the
body of each path:

| Quantity | Mean | Worst |
| --- | --- | --- |
| Centre line | 0.14 / 0.28 km | 2.0 km |
| Northern limit | 0.63 / 0.54 km | 1.2 km |
| Southern limit | 0.44 / 1.09 km | 1.9 km |
| Path width | 0.45 / 1.08 km | 3.0 km |
| Central duration | 0.08 / 0.07 s | 0.12 s |
| Moon/Sun diameter ratio | exact to the tabulated 3 decimals | |

(means given as total / annular)

The residual kilometre is about what the different ephemerides account for:
those NASA tables were computed with VSOP87/ELP2000-85 rather than DE440.
Independently, Wikipedia's figures for the 2027 Aug 2 eclipse (gamma 0.1421,
magnitude 1.079, 6m23s, 258 km, 25.5°N 33.2°E, saros 136) all reproduce, and
Cardiff's 93.2% at 19:13 BST on 2026 Aug 12 — quoted by timeanddate.com, and
the eclipse this project was written the day of — comes out as 93.24% at
18:13:48 UTC.

### Things worth knowing

* **Path width is a true geodesic distance** between the limits. Catalogue
  widths come from a tangent-plane Besselian formula that under-reports a very
  wide band seen at low Sun altitude: for the 2026 Feb 17 annular the distance
  between NASA's *own* published limits runs 15–26 km wider than the width
  NASA tabulates alongside them, matching the difference seen here.
* **Cross-sections that are not transverse report no width.** Where a track
  meets the terminator the perpendicular runs along the edge of the band rather
  than across it; those rows show `-` instead of a number several times too
  large. Their limit coordinates are still real points on the boundary.
* **Delta-T limits long-range accuracy.** The difference between Terrestrial
  Time and universal time cannot be predicted years ahead. It does not affect
  TT timings or latitudes, but it slides a path in longitude by about 0.46 km
  per second of error at the equator. Values here come from Skyfield's IERS
  data (69.1 s for 2026, against the 75 s assumed by the 2006-era NASA
  catalogue). Use `--delta-t` to impose a different assumption.
* **The lunar limb is treated as a circle.** Real umbral edges are ragged at
  the kilometre scale because of mountains and valleys, which is what the two
  values of k are trying to average over. Grazing-line observers should use
  limb-profile predictions.
* **No atmospheric refraction**, so a path's extreme sunrise/sunset ends are
  slightly conservative.

## Tests

```
python3 tests/test_eclipsepath.py        # 33 tests, also runs under pytest
python3 tests/verify_against_nasa.py     # row-by-row against NASA path tables
```

Set `ECLIPSEPATH_KERNEL` or pass the kernel path to reuse an existing download.

## Sources

* [NASA Five Millennium Catalog of Solar Eclipses](https://eclipse.gsfc.nasa.gov/SEcat5/SE2001-2100.html) — Espenak & Meeus
* [NASA path table, 2026 Aug 12](https://eclipse.gsfc.nasa.gov/SEpath/SEpath2001/SE2026Aug12Tpath.html)
* [NASA path table, 2026 Feb 17](https://eclipse.gsfc.nasa.gov/SEpath/SEpath2001/SE2026Feb17Apath.html)
* [Mean lunar radius and the two values of k](https://www.eclipsewise.com/solar/SEhelp/SEradius.html)
* [JPL DE440 ephemeris](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/)
