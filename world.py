"""Static map model: terrain height, solid prisms (for line of sight / shots), 2 m nav grid with A*."""
import json
import math
import os
import heapq
import zlib
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
NOPE = -32768


class World:
    def __init__(self, arena_dir):
        meta = json.load(open(os.path.join(HERE, "data", "mapmeta.json")))
        self.nx, self.ny = meta["nx"], meta["ny"]
        self.x0, self.y0 = meta["x0"], meta["y0"]
        self.NX2, self.NY2 = meta["NX2"], meta["NY2"]
        self.x02, self.y02 = meta["x02"], meta["y02"]
        n = self.nx * self.ny
        # terrain (prefer the mounted arena; fall back to nothing = flat)
        g = array("h")
        try:
            with open(os.path.join(arena_dir, "v2", "terrain.i16"), "rb") as f:
                g.frombytes(f.read())
        except OSError:
            g = array("h", bytes(2 * n))
        if len(g) != n:
            g = array("h", bytes(2 * n))
        self.g = g
        s = array("h")
        s.frombytes(zlib.decompress(open(os.path.join(HERE, "data", "solids.bin"), "rb").read()))
        self.b0 = s[0:n]
        self.t0 = s[n:2 * n]
        nav = array("h")
        nav.frombytes(zlib.decompress(open(os.path.join(HERE, "data", "nav.bin"), "rb").read()))
        self.nav = nav

    # ---------------------------------------------------------------- heights
    def ground(self, x, y):
        fx = x - self.x0
        fy = y - self.y0
        nx = self.nx
        if fx < 0: fx = 0.0
        if fy < 0: fy = 0.0
        if fx > nx - 1.001: fx = nx - 1.001
        if fy > self.ny - 1.001: fy = self.ny - 1.001
        i = int(fx); j = int(fy)
        ax = fx - i; ay = fy - j
        k = j * nx + i
        g = self.g
        h = (g[k] * (1 - ax) + g[k + 1] * ax) * (1 - ay) + (g[k + nx] * (1 - ax) + g[k + nx + 1] * ax) * ay
        return h * 0.01

    def solid_at(self, x, y, z):
        """True if (x, y, z) is inside a prism or below the terrain."""
        i = int(round(x - self.x0)); j = int(round(y - self.y0))
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return True
        k = j * self.nx + i
        zd = z * 10.0
        if self.b0[k] <= zd <= self.t0[k]:
            return True
        return z < self.g[k] * 0.01 - 0.3

    def los(self, a, b, step=1.5, skip_end=0.0):
        """Straight segment a->b clear of buildings and terrain (trees ignored: they don't stop shots)."""
        dx = b[0] - a[0]; dy = b[1] - a[1]; dz = b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < 1e-6:
            return True
        n = int((d - skip_end) / step)
        if n <= 0:
            return True
        sx = dx / d * step; sy = dy / d * step; sz = dz / d * step
        x, y, z = a[0], a[1], a[2]
        nx = self.nx; x0 = self.x0; y0 = self.y0
        b0 = self.b0; t0 = self.t0; g = self.g
        for _ in range(n):
            x += sx; y += sy; z += sz
            i = int(x - x0 + 0.5); j = int(y - y0 + 0.5)
            if i < 0 or j < 0 or i >= nx or j >= self.ny:
                return False
            k = j * nx + i
            zd = z * 10.0
            if b0[k] <= zd <= t0[k]:
                return False
            if zd < g[k] * 0.1 - 3.0:
                return False
        return True

    # ---------------------------------------------------------------- nav grid
    def cell(self, x, y):
        I = int((x - self.x02) / 2.0 + 0.5)
        J = int((y - self.y02) / 2.0 + 0.5)
        if I < 0: I = 0
        if J < 0: J = 0
        if I >= self.NX2: I = self.NX2 - 1
        if J >= self.NY2: J = self.NY2 - 1
        return J * self.NX2 + I

    def cell_xy(self, c):
        J, I = divmod(c, self.NX2)
        return (self.x02 + 2.0 * I, self.y02 + 2.0 * J)

    def walkable(self, x, y):
        return self.nav[self.cell(x, y)] != NOPE

    def surface(self, x, y):
        v = self.nav[self.cell(x, y)]
        return None if v == NOPE else v * 0.1

    def nearest_walkable(self, c, maxr=12):
        if self.nav[c] != NOPE:
            return c
        J, I = divmod(c, self.NX2)
        for r in range(1, maxr):
            for dj in range(-r, r + 1):
                for di in (-r, r) if abs(dj) != r else range(-r, r + 1):
                    i, j = I + di, J + dj
                    if 0 <= i < self.NX2 and 0 <= j < self.NY2:
                        k = j * self.NX2 + i
                        if self.nav[k] != NOPE:
                            return k
        return c

    def walk_line(self, a, b):
        """Can we walk straight from a to b on the nav grid (sampled every 1 m)?"""
        dx = b[0] - a[0]; dy = b[1] - a[1]
        d = math.hypot(dx, dy)
        n = int(d / 1.0)
        nav = self.nav
        prev = None
        for k in range(1, n + 1):
            t = k / max(n, 1)
            c = self.cell(a[0] + dx * t, a[1] + dy * t)
            v = nav[c]
            if v == NOPE:
                return False
            if prev is not None and v - prev > 22:
                return False
            prev = v
        return True

    def astar(self, start, goal, max_expand=60000, cost_fn=None):
        g = self.astar_gen(start, goal, max_expand, cost_fn)
        for r in g:
            if r is not None:
                return r
        return None

    def astar_gen(self, start, goal, max_expand=60000, cost_fn=None, chunk=400):
        """Resumable A*: yields None every `chunk` expansions, finally yields the path list."""
        nav = self.nav
        NX2 = self.NX2
        NY2 = self.NY2
        s = self.nearest_walkable(self.cell(*start))
        gcell = self.nearest_walkable(self.cell(*goal))
        if s == gcell:
            yield [goal]
            return
        gJ, gI = divmod(gcell, NX2)
        openh = [(0.0, s)]
        gs = {s: 0.0}
        parent = {s: -1}
        closed = set()
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
        expanded = 0
        best = s
        besth = 1e18
        while openh:
            f, c = heapq.heappop(openh)
            if c in closed:
                continue
            if c == gcell:
                best = c
                break
            closed.add(c)
            expanded += 1
            if expanded > max_expand:
                break
            if expanded % chunk == 0:
                yield None
            J, I = divmod(c, NX2)
            hc = nav[c]
            gc = gs[c]
            for di, dj, w in nbrs:
                i = I + di; j = J + dj
                if i < 0 or j < 0 or i >= NX2 or j >= NY2:
                    continue
                k = j * NX2 + i
                v = nav[k]
                if v == NOPE or k in closed:
                    continue
                if v - hc > (20 if w == 1.0 else 28):
                    continue
                if w != 1.0:
                    # no corner cutting
                    if nav[J * NX2 + i] == NOPE or nav[j * NX2 + I] == NOPE:
                        continue
                ng = gc + w + (cost_fn(k) if cost_fn else 0.0)
                if ng < gs.get(k, 1e18):
                    gs[k] = ng
                    parent[k] = c
                    ddx = abs(i - gI); ddy = abs(j - gJ)
                    h = (ddx + ddy) + (1.4142 - 2) * min(ddx, ddy)
                    if h < besth:
                        besth = h
                        best = k
                    heapq.heappush(openh, (ng + h * 1.001, k))
        # reconstruct to best (goal or closest reached)
        path = []
        c = best
        while c != -1:
            path.append(c)
            c = parent.get(c, -1)
        path.reverse()
        pts = [self.cell_xy(c) for c in path]
        # string-pull: keep points where straight walking fails
        if len(pts) > 2:
            out = [pts[0]]
            i = 0
            while i < len(pts) - 1:
                j = min(len(pts) - 1, i + 24)
                while j > i + 1 and not self.walk_line(pts[i], pts[j]):
                    j -= 1
                    yield None
                out.append(pts[j])
                i = j
                yield None
            pts = out
        if best == gcell:
            pts[-1] = (goal[0], goal[1]) if self.walkable(goal[0], goal[1]) else pts[-1]
        yield pts
