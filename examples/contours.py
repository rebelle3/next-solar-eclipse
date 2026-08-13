import os, sys, json; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from eclipsepath import Ephemeris, analyse, coverage_region
from eclipsepath import finder, output
e = Ephemeris(); ts = e.timescale
ev = finder.find_events(e, ts.utc(2027,7,15).tt, ts.utc(2027,8,15).tt)[0]
E = analyse(e, ev, threshold=1.0, samples=400)

out = {'eclipse': output.eclipse_detail(E, ts), 'contours': []}
for th in (0.01, 0.20, 0.40, 0.60, 0.80):
    r = coverage_region(E, th, rays=540)
    out['contours'].append({
        'threshold': th,
        'centre': [r.centre_latitude, r.centre_longitude],
        'encloses_pole': r.encloses_pole,
        'boundary': [[p.longitude, p.latitude] for p in r.points],
        'sun_altitude': [p.sun_altitude for p in r.points],
    })
    lats = [p.latitude for p in r.points]; lons = [p.longitude for p in r.points]
    print('%3.0f%%  lat %6.1f..%5.1f   lon %7.1f..%6.1f' % (th*100, min(lats), max(lats), min(lons), max(lons)))
json.dump(out, open('contours.json','w'))
print('\nwrote contours.json')
