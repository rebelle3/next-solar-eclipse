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

### Phase 1 — `eclipsepath/shadow.py`  [ ]
- [ ] `obscuration_grid`, `outlines` (marching squares), `umbra_outline`, `terminator`
- [ ] Tests in `tests/test_shadow.py`, each an agreement with something already trusted:
  - [ ] the umbra outline's centre equals `eclipse.shadow_point` to < 1 km
  - [ ] the umbra outline's width across the track equals `BandPoint.width_km` to < 2 km
  - [ ] the 0% contour encloses exactly those sites whose C1..C4 bracket `tt`
  - [ ] a contour at the threshold agrees with `coverage_region` to < 10 km
  - [ ] every contour is closed, non-self-intersecting, and monotone-nested

### Phase 2 — `examples/animate_shadow.py`  [ ]
- [ ] Frames from first to last contact, umbra + penumbra + coverage contours + terminator
- [ ] Land from the same Natural Earth file `plot_path.py` takes
- [ ] GIF via Pillow; MP4 when an ffmpeg writer is present
- [ ] Caption driven by the data, never hardcoded — the `plot_path.py` lesson
- [ ] Test: frame count, monotone shadow motion, and a pixel check that the
      umbra centre lands on the right pixel in a known frame

### Phase 3 — `eclipsepath/scene.py`, the web export  [ ]
- [ ] One eclipse to a compact JSON scene: Sun and Moon geocentric positions and
      the Earth-rotation matrix, sampled across the window, plus the radii and
      constants the shader needs
- [ ] Coastlines decimated to a small line-segment array
- [ ] Test: JS-side interpolation of the samples reproduces Python's own
      `window.at(tt)` to < 1 km at instants between samples

### Phase 4 — `web/` viewer  [ ]
- [ ] Raw WebGL 2: sphere mesh, orbit camera, coastline overlay
- [ ] Fragment shader computes obscuration per pixel from the same formula as
      `geometry.obscuration` — a real shadow, not a texture pasted on
- [ ] Sun and Moon drawn in their real directions, at true angular size, with
      the Moon's orbit traced
- [ ] Time scrubber, play/pause, speed; reads the scene JSON
- [ ] Single self-contained page; strictly no external requests

### Phase 5 — automated tests for the viewer  [ ]
- [ ] `tests/test_webgl.py`: headless Chromium via Playwright
      (`/opt/pw-browsers/chromium-1194/chrome-linux/chrome`; WebGL 2 confirmed
      working, float textures and `readPixels` both available)
- [ ] Render obscuration to a float texture, read it back, compare against
      Python on the same points — the shader must match to < 0.001
- [ ] Screenshot at fixed camera and time; assert the umbra centre pixel
- [ ] Assert zero network requests while the page loads and runs

### Phase 6 — documentation  [ ]
- [ ] README section, kept short: what each shows, how to run, what was checked
- [ ] Both artefacts regenerated and committed

## Ground rules

* Every patch to an existing file goes through an assertion that the text it
  replaces occurs exactly once.
* Nothing is ticked above on the strength of it looking right in a picture.
* Commit at the end of each phase, on `claude/eclipse-visualisers-l6ndjr`.
* No pull request unless asked for.
