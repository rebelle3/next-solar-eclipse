# Two visualisers: a moving shadow, and a globe you can turn

Working plan and progress log.  Tick a box only when the thing is built *and*
a test asserts it is right.

## What is being built

**A. Animated 2D map.**  The Moon's shadow crossing the Earth, as a sequence of
frames on a world map: the umbra as a filled ellipse, the penumbra as its outer
limit, contours of partial coverage in between, and the day/night terminator.
Written out as an animated GIF (and MP4 where a writer exists).

**B. Interactive WebGL globe.**  A 3D Earth you can turn and zoom, with the Sun
and Moon in their real places, the Moon's shadow falling on the globe, and a
scrubber for time.  Self-contained: one HTML file plus a data file, no network,
no libraries — the shading is written straight against WebGL 2.

Both come off the same computation, so they cannot disagree.

## The shared piece that does not exist yet

The package can already say where the shadow's *centre* is, trace the edges of
the central band, and answer for any one site.  Neither visualiser wants any of
those.  Both want the same new thing:

> at one instant, what fraction of the Sun is covered, everywhere at once.

So the first task is `eclipsepath/shadow.py`, an instantaneous-footprint module:

* `obscuration_grid(window, tt, lat_step, lon_step)` — obscuration and Sun
  altitude on a lat/lon grid, masked where the Sun is down.  Vectorised through
  the existing `circumstances.state_at`; a 181x361 grid measures at 23 ms.
* `outlines(grid, levels)` — marching squares over the grid, returning closed
  polylines per level, longitudes unwrapped so nothing streaks across the map.
* `umbra_outline(window, tt)` — the true shadow edge from the cone tangent to
  the Sun and Moon, intersected with the ellipsoid.  Analytic, not contoured:
  the umbra is small and a grid would round its corners.
* `terminator(window, tt)` — the sunrise/sunset circle.

This is also the honest fix for the jagged region tips noted in the README:
contouring a grid does not have the radial-ray artefact.  Land that after the
visualisers are up, not before.

## Order of work

### Phase 1 — `eclipsepath/shadow.py`  [x]
- [x] `obscuration_grid`, `contours` (marching squares), `umbra_outline`, `terminator`
- [x] 17 tests in `tests/test_shadow.py`; the existing 46 still pass
  - [x] the umbra outline's centre equals `eclipse.shadow_point` to < 1 km
  - [x] the 0% contour encloses exactly those sites whose C1..C4 bracket `tt`
  - [x] every contour is closed and nested, and every vertex sits on its level
        to 1e-9 after refinement
  - [x] a contour crossing the antimeridian comes back as one continuous loop
  - [x] the terminator sits on the apparent horizon to 1e-6 degrees

Three things the building of it turned up, each now written into the code:

* **The band limits are not a cut across the umbra.**  `_trace_band` asks
  whether a place reaches totality *at any time*, so the band edges are the
  envelope of every umbra the eclipse casts.  The envelope touches each
  instantaneous outline without crossing it — within 100 m of it with the Sun
  overhead, several kilometres at grazing incidence near sunrise, where the
  shadow edge meets the ground almost tangentially.  The planned "width matches
  `BandPoint.width_km`" check was asserting these were the same curve, which
  they are not; the tests now assert the containment and the touch instead.
* **The umbra outline was drawing shadow on the night side.**  Near first
  contact the cone overruns the limb: a third of the outline was landing where
  the Sun was up to ten degrees below the horizon.  The footprint is now the
  cone cut by the lit hemisphere.
* **`Grid.peak` cannot locate anything during totality.**  The field is a flat
  1 right across the umbra, so the argmax is an arbitrary point inside it —
  the same flatness that makes the time of maximum undefined on the central
  line.  Documented rather than faked.

A fourth, checked and dropped: comparing an instantaneous contour against
`coverage_region` was in the plan, but that region is the maximum over the
whole eclipse, so there is no instant at which the two should agree.

### Phase 2 — `examples/animate_shadow.py`  [x]
- [x] Frames from first to last contact: coverage as a greyscale field, the
      umbra at true size, contours at 20/40/60/80%, terminator, night side
- [x] Land from the same Natural Earth file `plot_path.py` takes
- [x] GIF via Pillow; MP4 when an ffmpeg writer is present
- [x] Caption driven by the data, never hardcoded — the `plot_path.py` lesson
- [x] 8 tests in `tests/test_animation.py`, including a real render whose
      darkest umbra-coloured pixel is inverted back to degrees and compared
      with the computed centre

What it turned up:

* **`visible_extent` claimed the whole world for the 2028 total.**  Each frame's
  longitudes were unwrapped about that frame's own centre, so as the shadow
  crossed the antimeridian one frame came back around +190 and the next around
  -170, and the union of the two spanned 360 degrees.  One reference for all
  frames fixes it; the view is now 150 degrees wide.
* **Vector overlays needed repeating a turn either side.**  A curve is handed
  back with its longitudes made continuous, which puts the terminator's far
  branch at +208 — off the right of a view that badly needed it on the left.
  It was drawing as a hard diagonal cut with no gold line on it.
* **`imshow` was given sample centres as if they were array edges**, sliding
  the whole coverage field half a cell, 28 km at the default step.
* `output.format_duration` is built for the minutes and seconds of totality and
  called a five-hour eclipse `313m01.7s`; the example spells hours out itself
  rather than changing what the path tables print.

### Phase 3 — `eclipsepath/scene.py`, the web export  [x]
- [x] One eclipse to a compact JSON scene: Sun and Moon in the Earth-fixed
      frame at one sample a minute, plus the constants and the traced path
- [x] Coastlines from Natural Earth land at 1:110m — 127 rings, 5091 points,
      no decimation needed
- [x] Writes a standalone page with the scene baked in: a file opened from
      disk cannot fetch its neighbours, and the point is that it works with no
      server and no network
- [x] 127 kB for the whole 2027 page, scene included

### Phase 3 — `eclipsepath/scene.py`, the web export  [ ]
- [ ] One eclipse to a compact JSON scene: Sun and Moon geocentric positions and
      the Earth-rotation matrix, sampled across the window, plus the radii and
      constants the shader needs
- [ ] Coastlines decimated to a small line-segment array
- [ ] Test: JS-side interpolation of the samples reproduces Python's own
      `window.at(tt)` to < 1 km at instants between samples

### Phase 4 — `eclipsepath/viewer.html`  [x]
- [x] Raw WebGL 2: ellipsoid mesh, orbit camera, coastlines, graticule, path
- [x] Fragment shader computes obscuration per pixel from the same
      circle-overlap as `geometry.obscuration` — a real shadow, not a texture
- [x] Time scrubber, play/pause, speed, live readouts
- [x] Single self-contained page, no external requests
- [x] The Moon at its true distance and size, the umbral cone, and the Moon's
      track across the eclipse

The Sun is not drawn: at 23,000 Earth radii there is no frame that holds it and
the Earth together.  The wide view frames the Earth-Moon pair rather than
orbiting the Earth's centre -- pulling straight back until the Moon appears
wastes most of the picture on empty space and leaves the Earth two pixels
across.  Even framed, the Earth is small in that view, because it is.

Confirmed so far: the shader agrees with the same arithmetic in double
precision to 2e-5 of obscuration, so float32 in the fragment shader is not the
limit on anything.  Float colour targets have to be asked for by name --
without `EXT_color_buffer_float` the framebuffer comes back incomplete and
every draw is silently discarded, which reads exactly like a shader returning
zero.

### Phase 5 — automated tests for the viewer  [x]
- [x] `tests/test_webgl.py`, 6 tests in headless Chromium, skipping cleanly
      where there is no browser
- [x] The shader renders its answer into a floating-point target one pixel wide
      and it is read straight back: **2.7e-5** of obscuration against the
      package, and **under a metre** on the edge of totality
- [x] The scene's minute cadence, interpolated in the browser, is within a
      kilometre of the ephemeris — the Sun within three, its samples being
      rounded to the kilometre on the way out
- [x] Screenshot with the camera over the umbra: the darkest pixel lands within
      a tenth of a frame of the centre
- [x] One request while loading, the page itself

Worth recording: `RGBA32F` is not renderable until `EXT_color_buffer_float` is
asked for by name.  Without it the framebuffer comes back incomplete and every
draw is silently discarded, which reads exactly like a shader that computes
zero.

### Phase 6 — documentation  [x]
- [x] README: two sections, what each shows, how to run, what was checked
- [x] `examples/eclipse_2027_shadow.gif` (1.5 MB, requantised from 4.8) and
      `examples/eclipse_2027_globe.html` (135 kB) committed
- [x] `viewer.html` declared as package data so an install carries it

## Ground rules

* Every patch to an existing file goes through an assertion that the text it
  replaces occurs exactly once.
* Nothing is ticked above on the strength of it looking right in a picture.
* Commit at the end of each phase, on `claude/eclipse-visualisers-l6ndjr`.
* No pull request unless asked for.
