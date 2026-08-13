#!/usr/bin/env python3
"""Hold the browser viewer to the Python package, in a real browser.

Runs under pytest, or standalone with ``python3 tests/test_webgl.py``.  Needs
Playwright and a Chromium; without them every test here reports itself skipped
rather than failing, since the rest of the suite does not depend on a browser.

    pip install playwright && playwright install chromium

Three separate things get checked, because there are three places the viewer
could quietly disagree with the package:

* the scene's samples, interpolated in JavaScript, against the ephemeris;
* the viewer's own double-precision arithmetic, against ``state_at``;
* the fragment shader, in float32 on whatever the GPU is, against both.

The last is the one worth having.  A shadow that looks convincing and is a
hundred kilometres out looks exactly as convincing.
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from eclipsepath import circumstances as cc  # noqa: E402
from eclipsepath import eclipse as ec, finder, scene  # noqa: E402
from eclipsepath import geometry as g  # noqa: E402
from eclipsepath.ephemeris import Ephemeris  # noqa: E402

KERNEL = os.environ.get('ECLIPSEPATH_KERNEL')
COAST = os.environ.get('ECLIPSEPATH_COASTLINES')
CHROMIUM = os.environ.get(
    'ECLIPSEPATH_CHROMIUM',
    '/opt/pw-browsers/chromium-1194/chrome-linux/chrome')

# Sites spread across the penumbra and beyond it, so the comparison covers
# clear sky, the partial ranges, the edge of totality and the night side.
SITES = [(25.5, 33.2), (30.0, 20.0), (10.0, 50.0), (24.0, 34.0), (26.5, 32.0),
         (51.5, -0.1), (-20.0, 0.0), (40.0, -70.0), (0.0, 90.0), (60.0, 10.0),
         (-35.0, 25.0), (35.0, 60.0)]
_cache = {}


def available():
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(CHROMIUM)


def skip(reason):
    print('   (skipped: %s)' % reason)


def prepared():
    """One eclipse, its scene, and the page written out, built once."""
    if 'built' not in _cache:
        g.set_lunar_radius('espenak')
        ephem = Ephemeris(KERNEL)
        ts = ephem.timescale
        events = finder.find_events(ephem, ts.utc(2027, 8, 1).tt,
                                    ts.utc(2027, 8, 3).tt)
        eclipse = ec.analyse(ephem, events[0], threshold=1.0, samples=60)
        built = scene.build(eclipse, ts, COAST if COAST and
                            os.path.exists(COAST) else None)
        folder = tempfile.mkdtemp(prefix='eclipsepath-web-')
        page = os.path.join(folder, 'globe.html')
        with open(page, 'w') as handle:
            handle.write(scene.standalone(built))
        _cache['built'] = (eclipse, built, page)
    return _cache['built']


def in_browser(body, *args):
    """Run one snippet in a freshly loaded page and hand back its result."""
    from playwright.sync_api import sync_playwright
    _eclipse, _built, page_path = prepared()
    problems, requests = [], []
    with sync_playwright() as play:
        browser = play.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page(viewport={'width': 1000, 'height': 720})
        page.on('pageerror', lambda e: problems.append(str(e)))
        page.on('console',
                lambda m: problems.append(m.text) if m.type == 'error' else None)
        page.on('request', lambda r: requests.append(r.url))
        page.goto('file://' + page_path)
        page.wait_for_timeout(700)
        result = page.evaluate(body, *args) if args else page.evaluate(body)
        shot = None
        if body == 'null':
            shot = page.screenshot()
        browser.close()
    return result, problems, requests, shot


def test_the_page_loads_clean_and_asks_the_network_for_nothing():
    if not available():
        return skip('no playwright or chromium')
    _eclipse, built, page_path = prepared()
    ready, problems, requests, _shot = in_browser(
        '() => ({gl: typeof window.eclipseShaderProbe, '
        'rings: SCENE.coastlines.length})')
    assert not problems, problems
    assert ready['gl'] == 'function', ready
    outside = [url for url in requests if not url.startswith('file://')]
    assert not outside, outside
    assert len(requests) == 1, requests      # the page itself and nothing else


def test_javascript_interpolation_matches_the_ephemeris():
    """The scene carries a sample a minute; the viewer must land between them.

    Catmull-Rom rather than straight lines is what lets the cadence be that
    coarse, and this is the test that says so: the instants asked about are
    deliberately off-sample, a third and two thirds of the way along.
    """
    if not available():
        return skip('no playwright or chromium')
    eclipse, built, _page = prepared()
    step = built['time']['step_days']
    times = [built['time']['tt0'] + step * (index + fraction)
             for index in (10, 40, 90, 150, 200)
             for fraction in (0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0)]
    got, problems, _requests, _shot = in_browser(
        '(times) => times.map(t => window.eclipseState(t))', times)
    assert not problems, problems

    worst = {'sun': 0.0, 'moon': 0.0}
    for tt, answer in zip(times, got):
        sun, moon, rot = eclipse.window.at(np.atleast_1d(tt))
        want = {'sun': g.rotate_to_itrf(rot, sun)[:, 0],
                'moon': g.rotate_to_itrf(rot, moon)[:, 0]}
        for body in ('sun', 'moon'):
            offset = float(np.linalg.norm(np.array(answer[body]) - want[body]))
            worst[body] = max(worst[body], offset)
    # The Sun's own samples are rounded to the kilometre on the way out, which
    # is a part in 1.5e8 of its distance and nothing at all to the geometry.
    assert worst['moon'] < 1.0, worst
    assert worst['sun'] < 3.0, worst


def test_the_viewer_agrees_with_state_at():
    """The viewer's own arithmetic against the package's, site by site."""
    if not available():
        return skip('no playwright or chromium')
    eclipse, built, _page = prepared()
    times = [eclipse.tt_greatest,
             eclipse.tt_first_contact + 0.25 * (eclipse.tt_last_contact
                                                - eclipse.tt_first_contact),
             eclipse.tt_first_contact + 0.75 * (eclipse.tt_last_contact
                                                - eclipse.tt_first_contact)]
    got, problems, _requests, _shot = in_browser(
        '([sites, times]) => times.map(t => sites.map('
        's => window.eclipseProbe(s[0], s[1], t)))', [SITES, times])
    assert not problems, problems

    worst_obscuration = worst_altitude = 0.0
    for tt, row in zip(times, got):
        state = cc.state_at(
            eclipse.window,
            g.geodetic_to_itrf(np.array([s[0] for s in SITES]),
                               np.array([s[1] for s in SITES])), tt)
        up = state['sun_altitude'] > g.HORIZON_ALTITUDE_DEG
        want = np.where(up, state['obscuration'], 0.0)
        for index, answer in enumerate(row):
            worst_obscuration = max(worst_obscuration,
                                    abs(answer['obscuration'] - want[index]))
            worst_altitude = max(worst_altitude,
                                 abs(answer['sun_altitude']
                                     - state['sun_altitude'][index]))
    assert worst_obscuration < 1e-6, worst_obscuration
    assert worst_altitude < 1e-6, worst_altitude


def test_the_shader_agrees_with_the_package():
    """The one that matters: float32, on the GPU, per pixel.

    The shader is made to render its own answer into a floating-point target
    one pixel wide and that number is read straight back, so this is the
    arithmetic the picture is drawn from and not a re-implementation of it.
    """
    if not available():
        return skip('no playwright or chromium')
    eclipse, built, _page = prepared()
    span = eclipse.tt_last_contact - eclipse.tt_first_contact
    times = [eclipse.tt_first_contact + span * f for f in (0.2, 0.5, 0.8)]
    got, problems, _requests, _shot = in_browser(
        '([sites, times]) => times.map(t => sites.map('
        's => window.eclipseShaderProbe(s[0], s[1], t)))', [SITES, times])
    assert not problems, problems

    worst_obscuration = worst_altitude = 0.0
    for tt, row in zip(times, got):
        state = cc.state_at(
            eclipse.window,
            g.geodetic_to_itrf(np.array([s[0] for s in SITES]),
                               np.array([s[1] for s in SITES])), tt)
        up = state['sun_altitude'] > g.HORIZON_ALTITUDE_DEG
        want = np.where(up, state['obscuration'], 0.0)
        for index, answer in enumerate(row):
            worst_obscuration = max(worst_obscuration,
                                    abs(answer['obscuration'] - want[index]))
            worst_altitude = max(worst_altitude,
                                 abs(answer['sun_altitude']
                                     - state['sun_altitude'][index]))
    print('      shader vs package: %.2e obscuration, %.2e deg'
          % (worst_obscuration, worst_altitude))
    assert worst_obscuration < 1e-3, worst_obscuration
    assert worst_altitude < 1e-2, worst_altitude


def test_the_shader_finds_the_edge_of_totality_where_the_package_does():
    """Walk across the umbra's edge and compare where each says it is.

    A shader that got the Moon's apparent radius slightly wrong would still
    look like an eclipse; it would put the edge of totality somewhere else.
    """
    if not available():
        return skip('no playwright or chromium')
    eclipse, built, _page = prepared()
    tt = eclipse.tt_greatest
    # A line of sites crossing the northern limit of the band.
    latitudes = np.linspace(eclipse.latitude, eclipse.latitude + 3.0, 61)
    sites = [(float(lat), eclipse.longitude) for lat in latitudes]
    got, problems, _requests, _shot = in_browser(
        '([sites, tt]) => sites.map(s => window.eclipseShaderProbe(s[0], s[1], tt))',
        [sites, tt])
    assert not problems, problems

    state = cc.state_at(eclipse.window,
                        g.geodetic_to_itrf(latitudes,
                                           np.full(latitudes.size,
                                                   eclipse.longitude)), tt)
    want = state['obscuration']
    got_values = np.array([a['obscuration'] for a in got])
    edge_want = float(np.interp(0.999, -want[::-1], -latitudes[::-1]))
    edge_got = float(np.interp(0.999, -got_values[::-1], -latitudes[::-1]))
    offset = abs(edge_want - edge_got) * 111.32
    print('      edge of totality: %.3f km apart' % offset)
    assert offset < 1.0, (offset, edge_want, edge_got)


def test_a_screenshot_shows_a_globe_and_a_shadow():
    """The picture itself: lit, and darkest where the umbra is.

    Everything above compares numbers, which a viewer could get right while
    drawing nothing at all.  Here the camera is put directly over the umbra and
    close enough that the globe fills the frame, so the umbra should land in the
    middle of the picture.  That catches the whole chain from a latitude to a
    pixel -- the projection, the camera, the mesh -- which no amount of agreeing
    about numbers would.
    """
    if not available():
        return skip('no playwright or chromium')
    from playwright.sync_api import sync_playwright
    eclipse, built, page_path = prepared()
    problems = []
    with sync_playwright() as play:
        browser = play.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page(viewport={'width': 900, 'height': 700})
        page.on('pageerror', lambda e: problems.append(str(e)))
        page.goto('file://' + page_path)
        page.wait_for_timeout(500)
        # Look straight down on the umbra, close in, with nothing else drawn.
        page.evaluate('''([lat, lon, when]) => {
            camera.latitude = lat; camera.longitude = lon; camera.distance = 1.9;
            tt = when;
            document.getElementById('showCoast').checked = false;
            document.getElementById('showPath').checked = false;
            document.getElementById('showCone').checked = false;
        }''', [eclipse.latitude, eclipse.longitude, eclipse.tt_greatest])
        page.wait_for_timeout(400)
        raw = page.screenshot()
        browser.close()
    assert not problems, problems

    import io
    from PIL import Image
    image = np.asarray(Image.open(io.BytesIO(raw)).convert('L'), dtype=float)
    height, width = image.shape
    # The panel is a dark rectangle down the left and would win any search for
    # the darkest thing on the screen.  Take it out of the running rather than
    # cropping, which would move the centre of the frame.
    field = image.copy()
    field[:520, :350] = 1e9
    seen = image[:, 400:]
    assert seen.mean() > 12.0, seen.mean()      # a lit globe, not an empty page
    assert seen.max() > 40.0, seen.max()

    row, column = np.unravel_index(int(np.argmin(field)), field.shape)
    offset = float(np.hypot(row - height / 2.0, column - width / 2.0))
    # A tenth of the frame: the camera's latitude is used as a direction from
    # the centre of the Earth rather than as a geodetic one, which can put the
    # point under the eye a fifth of a degree from the point asked for.
    assert offset < 0.10 * height, (offset, (row, column), (height, width))


def test_the_panel_gets_out_of_the_way_on_a_phone():
    """On a small screen the controls start folded, and can be folded again.

    Open, the panel covers half a phone screen -- of the globe it is describing.
    A control you cannot put away is worse than one you have to fetch back.
    """
    if not available():
        return skip('no playwright or chromium')
    from playwright.sync_api import sync_playwright
    _eclipse, _built, page_path = prepared()
    screen = {'width': 390, 'height': 844}
    area = screen['width'] * screen['height']
    problems = []
    with sync_playwright() as play:
        browser = play.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page(viewport=screen, is_mobile=True, has_touch=True)
        page.on('pageerror', lambda e: problems.append(str(e)))
        page.goto('file://' + page_path)
        page.wait_for_timeout(600)

        shut = page.locator('#info').bounding_box()
        assert page.locator('#info').evaluate(
            "e => e.classList.contains('shut')"), 'info card did not start folded'
        assert shut['width'] * shut['height'] < 0.08 * area, shut
        assert page.locator('#infoToggle').inner_text() == 'Info'
        # Folding the card away must not take the transport with it: the clock
        # and the timeline are the controls there is no other way to reach.
        assert page.locator('#transport').is_visible()
        assert ':' in page.locator('#clock').inner_text()
        bar = page.locator('#transport').bounding_box()
        assert bar['height'] < 0.16 * screen['height'], bar

        page.click('#infoToggle')
        page.wait_for_timeout(200)
        opened = page.locator('#info').bounding_box()
        assert opened['height'] > 3 * shut['height'], (shut, opened)
        assert page.locator('#infoToggle').inner_text() == 'Hide'
        assert page.locator('#showPath').is_visible()

        page.click('#infoToggle')
        page.wait_for_timeout(200)
        assert page.locator('#info').evaluate(
            "e => e.classList.contains('shut')"), 'info card would not fold again'

        # Two fingers spreading apart must bring the globe closer.  A phone has
        # no wheel, so without this there is no way to zoom at all.
        before = page.evaluate('camera.distance')
        page.evaluate('''() => {
            const c = document.getElementById('gl');
            c.setPointerCapture = () => {};
            const send = (type, id, x) => c.dispatchEvent(new PointerEvent(type,
                {pointerId: id, clientX: x, clientY: 400, bubbles: true,
                 pointerType: 'touch'}));
            send('pointerdown', 1, 150); send('pointerdown', 2, 250);
            send('pointermove', 1, 100); send('pointermove', 2, 300);
            send('pointerup', 1, 100); send('pointerup', 2, 300);
        }''')
        after = page.evaluate('camera.distance')
        browser.close()
    assert not problems, problems
    # The gap doubled, so the camera should have come half the distance in.
    assert abs(after - before / 2.0) < 0.05 * before, (before, after)


def test_the_timeline_marks_the_central_phase_where_it_happens():
    """The gold rail is the stretch when the umbra is on the ground.

    Read back off the rendered element rather than recomputed, so a mistake in
    the arithmetic that places it shows up rather than being repeated.  The
    partial is the case that matters: it has no central phase at all, and a
    rail drawn anywhere would be a claim about a shadow that never lands.
    """
    if not available():
        return skip('no playwright or chromium')
    eclipse, built, _page = prepared()
    got, problems, _requests, _shot = in_browser(
        '''() => {
            const t = SCENE.time, rail = document.getElementById('band');
            return {left: rail.style.left, width: rail.style.width,
                    shown: getComputedStyle(rail).display !== 'none',
                    mark: document.getElementById('mark').style.left};
        }''')
    assert not problems, problems
    span = eclipse.tt_last_contact - eclipse.tt_first_contact
    want_left = 100.0 * (eclipse.tt_central_start - eclipse.tt_first_contact) / span
    want_width = 100.0 * (eclipse.tt_central_end
                          - eclipse.tt_central_start) / span
    assert got['shown'], got
    assert abs(float(got['left'].rstrip('%')) - want_left) < 0.01, (got, want_left)
    assert abs(float(got['width'].rstrip('%')) - want_width) < 0.01, (got,
                                                                     want_width)
    # And the mark sits at greatest eclipse, which for this one is the middle.
    assert '49.99' in got['mark'] or '50.0' in got['mark'], got

    partial = os.path.join(os.path.dirname(HERE), 'docs', 'globes',
                           'eclipse_20290114.html')
    if not os.path.exists(partial):
        return
    from playwright.sync_api import sync_playwright
    with sync_playwright() as play:
        browser = play.chromium.launch(executable_path=CHROMIUM)
        page = browser.new_page(viewport={'width': 900, 'height': 640})
        page.goto('file://' + os.path.abspath(partial))
        page.wait_for_timeout(600)
        shown = page.evaluate(
            "getComputedStyle(document.getElementById('band')).display")
        kind = page.evaluate('SCENE.eclipse.kind')
        browser.close()
    assert kind == 'P', kind
    assert shown == 'none', shown


def test_every_published_globe_loads_clean():
    """The pages in docs/ are what a visitor gets; check those, not a copy.

    A globe is generated, so it can go stale against the code that generates
    it, and the partial and the annular exercise branches the flagship total
    never reaches -- no umbra at all, and coverage that never gets to 1.
    """
    if not available():
        return skip('no playwright or chromium')
    import glob
    from playwright.sync_api import sync_playwright
    folder = os.path.join(os.path.dirname(HERE), 'docs', 'globes')
    pages = sorted(glob.glob(os.path.join(folder, '*.html')))
    if not pages:
        return skip('no built site in docs/globes')
    with sync_playwright() as play:
        browser = play.chromium.launch(executable_path=CHROMIUM)
        for path in pages:
            problems, requests = [], []
            page = browser.new_page(viewport={'width': 900, 'height': 640})
            page.on('pageerror', lambda e: problems.append(str(e)))
            page.on('console', lambda m: problems.append(m.text)
                    if m.type == 'error' else None)
            page.on('request', lambda r: requests.append(r.url))
            page.goto('file://' + os.path.abspath(path))
            page.wait_for_timeout(700)
            ready = page.evaluate(
                '() => ({probe: typeof window.eclipseShaderProbe, '
                'kind: SCENE.eclipse.kind, '
                'covered: document.getElementById("peak").textContent})')
            page.close()
            name = os.path.basename(path)
            assert not problems, (name, problems)
            assert [u for u in requests if not u.startswith('file://')] == [], name
            assert ready['probe'] == 'function', (name, ready)
            # A partial has no axis on the globe and must say so rather than
            # printing a coverage for a shadow that never lands.
            if ready['kind'] == 'P':
                assert ready['covered'] == '\u2014', (name, ready)
            else:
                assert ready['covered'].endswith('%'), (name, ready)
        browser.close()


def main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith('test_') and callable(obj)]
    failures = []
    for name, test in tests:
        try:
            test()
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print('FAIL %s: %s' % (name, exc))
        else:
            print('ok   %s' % name)
    print('\n%d passed, %d failed' % (len(tests) - len(failures), len(failures)))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
