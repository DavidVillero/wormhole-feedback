"""Assemble the single-file page from live/*.js and the exported models.
    python3 live/build.py      # rewrites wormhole-control-room.html (scripts and data in place) and index.html
"""
import json, os, re
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
rd = lambda f: open(os.path.join(root, f)).read()
s = rd('wormhole-control-room.html')

def block(tag_open, body):
    global s
    i = s.index(tag_open) + len(tag_open); j = s.index('</script>', i)
    s = s[:i] + '\n' + body.strip() + '\n' + s[j:]

# data
i = s.index('const LIVE = ') + len('const LIVE = '); j = s.index('</script>', i)
live = json.loads(s[i:j].strip().rstrip(';'))
live['transit'] = json.loads(rd('viz_transit.json'))
s = s[:i] + json.dumps(live, separators=(',', ':')) + ';' + s[j:]
if 'id="solver-full-src"' not in s:
    k = s.index('<script type="text/plain" id="worker-src">')
    s = s[:k] + '<script type="text/plain" id="solver-full-src">\n</script>\n' + s[k:]
block('<script type="text/plain" id="solver-src">', rd('live/solver.js'))
block('<script type="text/plain" id="solver-full-src">', rd('live/solver_full.js'))
block('<script type="text/plain" id="worker-src">', rd('live/worker.js'))
k = s.index('</script>', s.index('id="worker-src"')); k = s.index('<script>', k)
block_at = k + len('<script>'); j = s.index('</script>', block_at)
s = s[:block_at] + '\n' + rd('live/app.js').strip() + '\n' + s[j:]
open(os.path.join(root, 'wormhole-control-room.html'), 'w').write(s)
old = rd('index.html')
head = old[:old.index('<body>') + len('<body>')]
open(os.path.join(root, 'index.html'), 'w').write(head + '\n' + s.rstrip() + '\n\n</body>\n</html>\n')
print('built', len(s), 'bytes')
