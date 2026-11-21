"""Mech Royale bot: survive, loot, and win 1-on-1 fights with good aim, cover and dodging."""
import gc
import math
import os
import random
import sys
import time

import mechfight as mf
from world import World

TAU = 2 * math.pi


def wrap(a):
    while a > math.pi:
        a -= TAU
    while a < -math.pi:
        a += TAU
    return a


def d2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def d3(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# Diagnostic switch: never shoot, only hide and evade (used to measure what pure survival scores).
HIDE = os.environ.get("MF_HIDE") == "1"
DUCK = False  # ducking while reloading tested worse (3.71); kept off

WEAPON_PREF = {"laser": 3, "lightning": 2, "mortar": -1, "gun": 0}
UPGRADE_ORDER = ["armor", "plating", "gun_damage", "gun_rate", "nanites", "laser_charge", "laser_damage",
                 "lightning_efficiency", "reactor", "capacitor", "lightning_chain", "lightning_range", "gun_mag",
                 "servos", "sensors", "gyros", "jets", "gun_velocity", "laser_pierce", "mortar_reload",
                 "mortar_velocity", "mortar_splash"]


class Planner:
    """Resumable A* run in small time slices from the main loop (no threads: 1 CPU, GIL)."""

    def __init__(self, world):
        self.world = world
        self.gen = None
        self.cur = None
        self.result = None  # (req_id, start, goal, path)
        self.req_id = 0

    def request(self, start, goal):
        self.req_id += 1
        self.cur = (self.req_id, start, goal)
        self.gen = self.world.astar_gen(start, goal, max_expand=30000, chunk=80)
        return self.req_id

    def work(self, budget_s):
        if self.gen is None:
            return
        end = time.perf_counter() + budget_s
        try:
            while time.perf_counter() < end:
                r = next(self.gen)
                if r is not None:
                    self.result = (self.cur[0], self.cur[1], self.cur[2], r)
                    self.gen = None
                    return
        except StopIteration:
            self.result = (self.cur[0], self.cur[1], self.cur[2], None)
            self.gen = None


class Track:
    __slots__ = ("id", "pos", "vel", "aim", "hp", "max_hp", "weapons", "charging", "seen_t", "cubes", "first_t")

    def __init__(self, m, t):
        self.id = m["id"]
        self.first_t = t
        self.update(m, t)

    def update(self, m, t):
        self.pos = m["pos"]
        self.vel = m.get("vel") or [0, 0, 0]
        self.aim = m.get("aim") or [0, 0]
        self.hp = m.get("hp", 1400)
        self.max_hp = m.get("max_hp", 1400)
        self.weapons = m.get("weapons") or []
        self.charging = m.get("charging", False)
        self.cubes = m.get("cubes", 0)
        self.seen_t = t


class Bot:
    name = "cyber-ai"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.slot = hello["slot"]
        self.rng = random.Random(hello.get("seed", 1))
        t0 = time.time()
        self.world = World(hello["map"]["dir"])
        mf.log(f"world loaded in {time.time() - t0:.2f}s")
        self.planner = Planner(self.world)
        gc.collect()
        gc.freeze()
        self.t_start = 0.0
        W = self.cfg["weapons"]
        self.gun_speed = W["gun"]["speed"]
        self.mech_cfg = self.cfg["mech"]
        self.tracks = {}
        self.path = None
        self.path_goal = None
        self.path_req = 0
        self.path_req_t = -99
        self.path_idx = 0
        self.last_pos_hist = []
        self.stuck_until = -1
        self.stuck_dir = None
        self.strafe_sign = 1
        self.strafe_until = 0
        self.target_id = None
        self.last_damage_t = -99
        self.last_damage_bearing = None
        self.attackers = {}  # id -> last time it hit us
        self.sound_bearing = None
        self.sound_t = -99
        self.scan_yaw = None
        self.scan_dir = 1
        self.laser_holding = False
        self.crate_target = None
        self.mode = "start"
        self.last_t = 0
        self.dodge_until = -1
        self.dodge_vec = None
        self.blocked_pickups = {}
        self.jump_cd = 0
        self.prev_hp = None
        self.cover_caches = {}
        self.npc_mem = {}
        self.crate_mem = {}
        self.explore_pt = None
        self.explore_until = -1
        # Test-only: extra action delay of the remote play server (unset in scored matches = lockstep).
        self.lat = float(os.environ.get("MF_LAT", "0") or 0)
        self.prev_cmd = None

    # ------------------------------------------------------------------ helpers
    def zone_at(self, z, t):
        """Current circle (center, radius) at time t, interpolated during a shrink."""
        if t <= z["shrink_start"]:
            return z["center"], z["radius"]
        if t >= z["shrink_end"]:
            return z["next_center"], z["next_radius"]
        f = (t - z["shrink_start"]) / max(1e-6, z["shrink_end"] - z["shrink_start"])
        c = [z["center"][0] + (z["next_center"][0] - z["center"][0]) * f,
             z["center"][1] + (z["next_center"][1] - z["center"][1]) * f]
        return c, z["radius"] + (z["next_radius"] - z["radius"]) * f

    def safe_circle(self, obs):
        """Circle we should be inside of now, given time to travel."""
        z = obs["zone"]
        t = obs["t"]
        me = obs["self"]["pos"]
        speed = self.me["stats"].get("walk_speed", 11.7)
        nc, nr = z["next_center"], z["next_radius"]
        dist_out = d2(me, nc) - nr * 0.8
        travel = max(0.0, dist_out) / speed
        if t + travel * 1.4 + 15 >= z["shrink_start"] or z["radius"] - z["next_radius"] < 1:
            return nc, max(nr, 1.0)
        return z["center"], z["radius"]

    def in_circle(self, p, c, r, margin=0.0):
        return d2(p, c) <= r - margin

    def find_cover(self, pos, threats, circ, t, radii=(10.0, 18.0, 28.0, 40.0), key="cover"):
        """A nearby walkable point hidden from the given threat positions. Cached for 0.5 s."""
        cc = self.cover_caches.get(key)
        if cc and t - cc[0] < 0.5:
            return cc[1]
        w = self.world
        best = None
        base = self.rng.uniform(0, TAU)
        threats = sorted(threats, key=lambda p: d2(p, pos))[:3]
        deadline = time.perf_counter() + 0.006
        for r in radii:
            if time.perf_counter() > deadline:
                break
            for k in range(10):
                a = base + k * TAU / 10
                x, y = pos[0] + math.cos(a) * r, pos[1] + math.sin(a) * r
                if not w.walkable(x, y):
                    continue
                if circ and not self.in_circle((x, y), circ[0], circ[1], 4):
                    continue
                z = w.surface(x, y)
                if z is None:
                    continue
                seen = 0
                for tp in threats:
                    e = (tp[0], tp[1], tp[2] + 8.0)
                    if w.los(e, (x, y, z + 7.5), step=2.0) or w.los(e, (x, y, z + 3.0), step=2.0):
                        seen += 1
                sc = seen * 100 + r + (0 if w.walk_line(pos, (x, y)) else 25)
                if best is None or sc < best[0]:
                    best = (sc, (x, y), seen)
            if best and best[2] == 0 and best[0] < r + 1:
                break
        res = best[1] if best and best[2] < len(threats) else None
        self.cover_caches[key] = (t, res)
        return res

    def farm_goal(self, pos, muzzle, sc, sr, t):
        """Go after remembered drones/crates (cubes = HP, damage, upgrades); explore if none known."""
        w = self.world
        best = None
        for mem, kind in ((self.npc_mem, "npc"), (self.crate_mem, "crate")):
            for k, (p, hp, ts) in mem.items():
                if not self.in_circle(p, sc, sr, 10):
                    continue
                if kind == "crate" and p[2] - w.ground(p[0], p[1]) > 3.0:
                    continue  # roof crate: its cube would land up there
                dd = d2(p, pos)
                if dd > 320:
                    continue
                score = dd - (45 if kind == "npc" else 0)
                if best is None or score < best[0]:
                    best = (score, p, kind, dd)
        if best:
            _, p, kind, dd = best
            standoff = 45 if kind == "npc" else 16
            if dd < standoff and w.los(muzzle, (p[0], p[1], p[2] + 1.5), step=2.0):
                return None  # in place: the shooting code takes it
            self.explore_pt = None
            return p
        # explore for loot
        if self.explore_pt is None or t > self.explore_until or d2(self.explore_pt, pos) < 15                 or not self.in_circle(self.explore_pt, sc, sr, 10):
            self.explore_pt = None
            for _ in range(25):
                a = self.rng.uniform(0, TAU)
                r = math.sqrt(self.rng.random()) * sr * 0.85
                x, y = sc[0] + math.cos(a) * r, sc[1] + math.sin(a) * r
                dd = math.hypot(x - pos[0], y - pos[1])
                if 60 < dd < 260 and w.walkable(x, y):
                    self.explore_pt = (x, y)
                    self.explore_until = t + 30
                    break
        return self.explore_pt

    def _want_duck(self, target, weapons, has, me):
        if not DUCK or HIDE or target is None or not target[2] or target[1] > 170:
            return False
        w0 = weapons[0]
        if not w0 or w0["kind"] != "gun":
            return False
        gun_busy = (w0.get("reload", 0) or 0) > 0.4 or w0.get("ammo", 1) == 0
        if not gun_busy:
            return False
        for w in weapons[1:]:
            if not w:
                continue
            if w["kind"] == "laser" and ((w.get("charge") or 0) > 0 or (w.get("cooldown", 0) or 0) < 0.5):
                return False
            if w["kind"] == "lightning" and target[1] < 45 and me["energy"] > 12:
                return False
        return True

    def _want_cover(self, visible_enemies, hp_frac, target, t):
        if HIDE:
            return True
        shooters = [m for m, dist, los in visible_enemies if los and dist < 150]
        if len(shooters) >= 2:
            return True
        if target is None:
            return False
        m = target[0]
        ehp = m.get("hp", 1400) / max(1.0, m.get("max_hp", 1400))
        if hp_frac < 0.4 and ehp > hp_frac + 0.2:
            return True
        return False

    # ------------------------------------------------------------------ navigation
    def nav_to(self, goal, t, pos, urgent=False):
        """Return a unit-ish move vector toward goal following A* paths."""
        w = self.world
        if d2(pos, goal) < 1.5:
            return (0.0, 0.0)
        direct = d2(pos, goal) < 60 and w.walk_line(pos, goal)
        if direct:
            self.path = None
            dx, dy = goal[0] - pos[0], goal[1] - pos[1]
            n = math.hypot(dx, dy)
            return (dx / n, dy / n)
        # need a path
        need = (self.path_goal is None or d2(self.path_goal, goal) > 10 or self.path is None
                or t - self.path_req_t > 6)
        if need and t - self.path_req_t > 0.5:
            self.path_req = self.planner.request((pos[0], pos[1]), (goal[0], goal[1]))
            self.path_req_t = t
            self.path_goal = (goal[0], goal[1])
        res = self.planner.result
        if res and res[0] == self.path_req and res[3]:
            if self.path is None or self.path_src != res[0]:
                self.path = list(res[3])
                self.path_src = res[0]
                self.path_idx = 0
        if self.path:
            # advance
            while self.path_idx < len(self.path) - 1 and d2(pos, self.path[self.path_idx]) < 3.0:
                self.path_idx += 1
            # skip ahead if a later waypoint is directly walkable
            for k in range(min(len(self.path) - 1, self.path_idx + 3), self.path_idx, -1):
                if d2(pos, self.path[k]) < 50 and w.walk_line(pos, self.path[k]):
                    self.path_idx = k
                    break
            wp = self.path[self.path_idx]
            dx, dy = wp[0] - pos[0], wp[1] - pos[1]
            n = math.hypot(dx, dy)
            if n > 0.01:
                return (dx / n, dy / n)
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        n = math.hypot(dx, dy)
        return (dx / n, dy / n)

    # ------------------------------------------------------------------ main
    def act(self, obs):
        self.t_start = time.perf_counter()
        a = self.act2(obs)
        # spend leftover time on path planning (keep total well under the 50 ms budget)
        left = 0.010 - (time.perf_counter() - self.t_start)
        if left > 0.002:
            self.planner.work(left)
        return a

    def act2(self, obs):
        t = obs["t"]
        me = obs["self"]
        self.me = me
        pos = me["pos"]
        act = {}
        dbg = []
        scale = 1 + self.cfg["cubes"].get("size_per_cube", 0.05) * me["cubes"]
        muzzle = [pos[0], pos[1], pos[2] + me["stats"].get("muzzle_height", 7)]
        eye = [pos[0], pos[1], pos[2] + me["stats"].get("eye_height", 8)]

        # ---------------- memory
        mechs = []
        npcs = []
        for m in obs["mechs"]:
            if m["kind"] == "mech":
                if m["id"] == me["id"]:
                    continue
                tr = self.tracks.get(m["id"])
                if tr is None:
                    tr = self.tracks[m["id"]] = Track(m, t)
                else:
                    tr.update(m, t)
                mechs.append(m)
            else:
                npcs.append(m)
        # remember drones and crates for farming
        for n in npcs:
            self.npc_mem[n["id"]] = (n["pos"], n.get("hp", 350), t)
        for c in obs["crates"]:
            self.crate_mem[c["id"]] = (c["pos"], c.get("hp", 300), t)
        if obs["tick"] % 4 == 0:
            seen_n = {n["id"] for n in npcs}
            seen_c = {c["id"] for c in obs["crates"]}
            for mem, seen, life in ((self.npc_mem, seen_n, 90), (self.crate_mem, seen_c, 1e9)):
                for k, (p_, hp_, ts_) in list(mem.items()):
                    if k in seen:
                        continue
                    if t - ts_ > life or (d2(p_, pos) < 32 and self.world.los(eye, (p_[0], p_[1], p_[2] + 1.5), step=2.0)):
                        del mem[k]
        took_damage = False
        for e in obs["events"]:
            et = e.get("type")
            if et == "damaged":
                took_damage = True
                if e.get("by") is not None:
                    self.attackers[e["by"]] = t
                if e.get("bearing") is not None:
                    self.last_damage_bearing = e["bearing"]
                    self.last_damage_t = t
            elif et == "take_failed":
                self.blocked_pickups[e["id"]] = t
            elif et == "kill":
                self.tracks.pop(e.get("target"), None)
        for s in obs["sounds"]:
            if s["kind"] in ("gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet") and s["dist"] != "far":
                self.sound_bearing = s["bearing"]
                self.sound_t = t

        # ---------------- offers
        if obs.get("offer"):
            ch = obs["offer"]["choices"]
            best, bi = 999, 0
            for i, c in enumerate(ch):
                r = UPGRADE_ORDER.index(c["id"]) if c["id"] in UPGRADE_ORDER else 50
                if r < best:
                    best, bi = r, i
            act["pick"] = bi

        weapons = me["weapons"]
        w0 = weapons[0]
        w1 = weapons[1] if len(weapons) > 1 else None
        kinds = [w["kind"] if w else None for w in weapons]
        has = {k: i for i, k in enumerate(kinds) if k}

        # ---------------- threat assessment
        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        zone = obs["zone"]
        cur_c, cur_r = self.zone_at(zone, t)
        safe_c, safe_r = self.safe_circle(obs)

        visible_enemies = []
        for m in mechs:
            dist = d3(m["pos"], pos)
            tc = [m["pos"][0], m["pos"][1], m["pos"][2] + 4.5 * (1 + 0.05 * m.get("cubes", 0))]
            los = dist < 260 and self.world.los(muzzle, tc, skip_end=2.0)
            visible_enemies.append((m, dist, los))

        # target choice: prefer LOS, close, low hp, attacking us, sticky
        target = None
        best = 1e9
        for m, dist, los in visible_enemies:
            sc = dist / 40.0 + 2.0 * (m.get("hp", 1400) / 1400.0)
            if not los:
                sc += 3.0
            if t - self.attackers.get(m["id"], -99) < 3:
                sc -= 1.5
            if m["id"] == self.target_id:
                sc -= 0.8
            if sc < best:
                best, target = sc, (m, dist, los)
        self.target_id = target[0]["id"] if target else None

        # ---------------- aim & fire
        aim = None
        self.gate = None
        fire = [False, False]
        reload = False
        shoot_target = None  # (pos, vel, height, kind)
        if target and (target[2] or target[1] < 200):
            m = target[0]
            shoot_target = (m["pos"], m.get("vel", [0, 0, 0]), 9 * (1 + 0.05 * m.get("cubes", 0)), "mech", target[2], target[1])
        elif not visible_enemies and (hp_frac > 0.9 or any(k >= 100 and t - v < 3 for k, v in self.attackers.items())):
            # NPCs and crates when calm -- but not while hurt: dealing damage blocks health regen
            cand = None
            bestd = 1e9
            for n in npcs:
                dist = d3(n["pos"], pos)
                if dist < (190 if t < 160 else 140) and dist < bestd and self.in_circle(n["pos"], cur_c, cur_r, 5):
                    tc = [n["pos"][0], n["pos"][1], n["pos"][2] + 1.75]
                    if self.world.los(muzzle, tc, skip_end=1.0):
                        bestd, cand = dist, (n["pos"], n.get("vel", [0, 0, 0]), 3.5, "npc", True, dist)
            if cand is None:
                for c in obs["crates"]:
                    dist = d3(c["pos"], pos)
                    if dist < (130 if t < 160 else 90) and dist < bestd:
                        tc = [c["pos"][0], c["pos"][1], c["pos"][2] + 1.2]
                        if self.world.los(muzzle, tc, skip_end=1.0):
                            bestd, cand = dist, (c["pos"], [0, 0, 0], 2.4, "crate", True, dist)
            shoot_target = cand

        if shoot_target:
            tp, tv, th, tkind, tlos, tdist = shoot_target
            center = [tp[0], tp[1], tp[2] + th * 0.5]
            mv_ = me.get("vel") or [0, 0, 0]
            L = 0.12 + self.lat  # decision latency: the torso gets there about one decision later
            mz = [muzzle[0] + mv_[0] * L, muzzle[1] + mv_[1] * L, muzzle[2]]

            g_ = self.mech_cfg.get("gravity", 22)
            floor_ = self.world.surface(tp[0], tp[1])
            if floor_ is None:
                floor_ = self.world.ground(tp[0], tp[1])
            airborne = abs(tv[2]) > 0.3 or tp[2] - floor_ > 1.0

            def lead(src, extra):
                p = center
                for _ in range(3):
                    tt = d3(p, src) / self.gun_speed + extra
                    pz = center[2] + tv[2] * tt
                    if airborne:
                        pz -= 0.5 * g_ * tt * tt
                    p = [center[0] + tv[0] * tt, center[1] + tv[1] * tt, pz]
                gz = max(self.world.ground(p[0], p[1]), floor_ if airborne else -1e9)
                if p[2] < gz + th * 0.5:
                    p[2] = gz + th * 0.5
                return p

            def yp(src, p):
                return (math.atan2(p[1] - src[1], p[0] - src[0]), math.atan2(p[2] - src[2], max(0.5, d2(p, src))))

            pL = lead(mz, L)          # where to command the torso
            p0 = lead(muzzle, 0.0)    # where a bullet fired right now should go
            yaw, pitch = yp(mz, pL)
            aim = [yaw, pitch]
            y0, pt0 = yp(muzzle, p0)
            ay, ap = me["aim"]
            hd = max(1.0, d2(p0, muzzle))
            tol_y = math.atan2(2.3 if tkind == "mech" else 1.2, hd)
            tol_p = math.atan2(th * 0.42, hd)
            on = abs(wrap(ay - y0)) < tol_y + 0.012 and abs(ap - pt0) < tol_p + 0.012
            self.gate = (math.degrees(wrap(ay - y0)), math.degrees(ap - pt0), math.degrees(tol_y), math.degrees(tol_p), tdist, tlos)
            # hitscan / instant (laser, lightning): current and soon-after bearing to the center
            yc0, pc0 = yp(muzzle, center)
            cL = [center[0] + tv[0] * 0.08, center[1] + tv[1] * 0.08, center[2] + tv[2] * 0.08]
            mz2 = [muzzle[0] + mv_[0] * 0.08, muzzle[1] + mv_[1] * 0.08, muzzle[2]]
            ycL, pcL = yp(mz2, cL)
            ey_inst = min(abs(wrap(ay - yc0)), abs(wrap(ay - ycL)))
            on_now = ey_inst < tol_y + 0.004 and min(abs(ap - pc0), abs(ap - pcL)) < tol_p + 0.01
            ey = ey_inst
            if self.lat > 0:
                # delayed server: our fire lands when the torso has reached this command
                steady = self.prev_cmd is not None and abs(wrap(self.prev_cmd[0] - yaw)) < 0.12
                on = on_now = steady
                ey = 0.0 if steady else 1.0
            for i, w in enumerate(weapons):
                if not w:
                    continue
                k = w["kind"]
                st = w.get("stats", {})
                if k == "gun":
                    ammo = w.get("ammo", 1)
                    if tkind == "crate" and ammo <= 4:
                        continue
                    if tlos and tdist < st.get("range", 230) * 0.98 and on and w.get("reload", 0) == 0 and ammo > 0:
                        fire[i] = True
                elif k == "lightning":
                    rng = st.get("range", 45)
                    half = math.radians(st.get("cone_deg", 30)) * 0.5
                    en_need = st.get("energy", 3) + (2 if tkind == "mech" else 30)
                    if tlos and tdist < rng * 0.96 and ey < half * 0.85 and me["energy"] > en_need:
                        fire[i] = True
                elif k == "laser":
                    charge = w.get("charge", 0) or 0
                    full = st.get("charge", 0.75)
                    if charge > 0:
                        # holding: release when charged and on target
                        if tlos and charge >= full * 0.97 and on_now:
                            fire[i] = False
                        elif charge >= full + st.get("hold_s", 2) - 0.3:
                            fire[i] = False if (tlos and on_now) else True
                        else:
                            fire[i] = True
                    elif w.get("cooldown", 0) <= 0.05 and tlos and tdist < st.get("range", 280) * 0.95 and tkind in ("mech", "npc"):
                        fire[i] = True
                elif k == "mortar":
                    pass
        if HIDE:
            fire = [False, False]
        # reload when no enemy is close and in sight
        if w0 and w0["kind"] == "gun" and not fire[0] and w0.get("ammo", 12) < w0.get("stats", {}).get("mag", 12) and w0.get("reload", 0) == 0:
            close = any(los and dist < 150 for _, dist, los in visible_enemies)
            if not close and (shoot_target is None or shoot_target[3] != "mech" or w0.get("ammo", 12) <= 3):
                if not (shoot_target and shoot_target[3] == "npc" and w0.get("ammo", 12) > 3):
                    reload = True
        if w0 and w0["kind"] == "gun" and w0.get("ammo", 1) == 0 and w0.get("reload", 0) == 0:
            reload = True

        # ---------------- movement goal
        move = (0.0, 0.0)
        boost = False
        jump = False
        walk_slow = False
        goal = None
        engaged = target is not None and (target[2] or target[1] < 80)
        threat_near = any(dist < 120 for _, dist, _ in visible_enemies)
        recently_hit = t - self.last_damage_t < 3

        # loot: weapon pickups
        take = None
        drop = None
        best_pick = None
        cur_pref = WEAPON_PREF.get(kinds[1], -1) if kinds[1] else -2
        for pk in obs["pickups"]:
            if t - self.blocked_pickups.get(pk["id"], -99) < 3:
                continue
            k = pk["kind"]
            if k in has:
                continue
            pref = WEAPON_PREF.get(k, -1)
            if pref <= cur_pref or pref < 0:
                continue
            dist = d2(pk["pos"], pos)
            if not self.in_circle(pk["pos"], cur_c, cur_r, 8):
                continue
            if abs(pk["pos"][2] - pos[2]) > 3.5 and dist < 4:
                pass
            if dist < 4.0 and abs(pk["pos"][2] - pos[2]) < 4.0:
                take = pk["id"]
                if kinds[1] is not None and kinds[0] is not None:
                    drop = 1
            sc = dist - pref * 30
            if best_pick is None or sc < best_pick[0]:
                best_pick = (sc, pk, dist)
        if take is not None:
            laser_busy = any(w and w["kind"] == "laser" and (w.get("charge") or 0) > 0 for w in weapons)
            if not laser_busy:
                act["take"] = take
                if drop is not None:
                    act["drop"] = drop

        # mortar shells: dodge
        danger = None
        for pr in obs["projectiles"]:
            if pr["kind"] != "mortar":
                continue
            # simulate shell to impact
            px, py, pz = pr["pos"]; vx, vy, vz = pr["vel"]
            g = self.cfg["weapons"]["mortar"].get("gravity", 22)
            for _ in range(60):
                dt = 0.05
                px += vx * dt; py += vy * dt; pz += vz * dt; vz -= g * dt
                if pz < self.world.ground(px, py) + 0.5:
                    break
            if math.hypot(px - pos[0], py - pos[1]) < 12:
                danger = (px, py)
        # laser telegraph dodge
        laser_threat = None
        for m, dist, los in visible_enemies:
            if m.get("charging") and los:
                byaw = math.atan2(pos[1] - m["pos"][1], pos[0] - m["pos"][0])
                if abs(wrap(m["aim"][0] - byaw)) < math.atan2(6, max(dist, 1)) + 0.08:
                    laser_threat = (m, byaw)

        self.mode = "calm"
        if not me["in_zone"]:
            goal = cur_c
            boost = me["energy"] > 10
            self.mode = "zone"
        elif engaged and self._want_duck(target, weapons, has, me) and                 self.find_cover(pos, [target[0]["pos"]], (cur_c, cur_r), t, radii=(8.0, 14.0, 20.0), key="duck") is not None:
            # weapons not ready: step out of its sight, come back loaded
            goal = self.find_cover(pos, [target[0]["pos"]], (cur_c, cur_r), t, radii=(8.0, 14.0, 20.0), key="duck")
            boost = me["energy"] > (60 if "lightning" in has else 35)
            self.mode = "duck"
        elif engaged and self._want_cover(visible_enemies, hp_frac, target, t):
            threats = [m["pos"] for m, dist, los in visible_enemies if dist < 160]
            cp = self.find_cover(pos, threats, (cur_c, cur_r), t)
            if cp is not None:
                goal = cp
                boost = me["energy"] > (60 if "lightning" in has else 30)
                self.mode = "cover"
            else:
                goal = None
        if goal is not None and self.mode in ("cover", "duck"):
            pass
        elif not me["in_zone"]:
            pass
        elif engaged:
            self.mode = "fight"
            m, dist, los = target
            tw = m.get("weapons") or []
            # preferred distance
            if "lightning" in has:
                pref_d = 28 if hp_frac > 0.35 else 60
            elif "laser" in has:
                pref_d = 70
            else:
                pref_d = 60
            if "lightning" in tw and "lightning" not in has:
                pref_d = max(pref_d, 60)
            rush = False
            if HIDE:
                pref_d = 250
            elif "laser" in tw and "lightning" in has and dist > 40 and hp_frac > 0.3:
                # a laser wins at range and loses up close (lightning shorts it, 1.65 s cooldown)
                pref_d = 25
                rush = True
            elif hp_frac < 0.3 and m.get("hp", 1400) / 1400 > hp_frac + 0.15:
                pref_d = 150  # back off
            bx, by = m["pos"][0] - pos[0], m["pos"][1] - pos[1]
            bn = math.hypot(bx, by) or 1
            bx, by = bx / bn, by / bn
            radial = 0.0
            if dist > pref_d + 10:
                radial = 1.0
            elif dist < pref_d - 10:
                radial = -1.0
            group = sum(1 for _m, _d, _l in visible_enemies if _d < 160)
            if not los:
                ehp = m.get("hp", 1400) / max(1.0, m.get("max_hp", 1400))
                if group >= 2:
                    # several hunters: wait at our corner, make them come around it one by one
                    goal = None
                elif hp_frac > 0.5 or ehp < hp_frac - 0.1:
                    # go find a line: move toward the target via path
                    goal = m["pos"]
                else:
                    # hurt: let it come to us and regenerate behind cover
                    goal = None
                    walk_slow = True
                radial = None
            if t > self.strafe_until:
                self.strafe_sign = self.rng.choice((-1, 1))
                self.strafe_until = t + self.rng.uniform(0.3, 0.7)
            if radial is not None:
                if group >= 2:
                    # outnumbered in the open: back away from the group while fighting the nearest
                    others = [_m["pos"] for _m, _d, _l in visible_enemies if _d < 160]
                    gx = sum(p_[0] for p_ in others) / len(others) - pos[0]
                    gy = sum(p_[1] for p_ in others) / len(others) - pos[1]
                    gn = math.hypot(gx, gy) or 1
                    bx, by = gx / gn, gy / gn
                    radial = -1.0
                sx, sy = -by * self.strafe_sign, bx * self.strafe_sign
                sw, rw = (0.35, 1.0) if rush and radial > 0 else (0.85, 0.7)
                mx, my = sx * sw + bx * radial * rw, sy * sw + by * radial * rw
                # keep inside zone
                if not self.in_circle((pos[0] + mx * 10, pos[1] + my * 10), cur_c, cur_r, 3):
                    cx, cy = cur_c[0] - pos[0], cur_c[1] - pos[1]
                    cn = math.hypot(cx, cy) or 1
                    mx, my = cx / cn, cy / cn
                # avoid walls: probe ahead
                ahead = (pos[0] + mx * 6, pos[1] + my * 6)
                if not self.world.walk_line(pos, ahead):
                    self.strafe_sign *= -1
                    sx, sy = -by * self.strafe_sign, bx * self.strafe_sign
                    mx, my = sx * 0.85 + bx * radial * 0.7, sy * 0.85 + by * radial * 0.7
                    ahead = (pos[0] + mx * 6, pos[1] + my * 6)
                    if not self.world.walk_line(pos, ahead):
                        mx, my = bx * radial if radial else bx, by * radial if radial else by
                move = (mx, my)
                charging = any(w and w["kind"] == "laser" and (w.get("charge") or 0) > 0 for w in weapons)
                if (laser_threat is not None or (rush and radial > 0)):
                    boost = me["energy"] > (55 if "lightning" in has else 30)
                else:
                    # boosted jinking makes leading gunners miss (sim: 45% -> 38% hit at 60 m)
                    boost = (not charging) and dist > 40 and me["energy"] > (70 if "lightning" in has else 45)
        else:
            # calm: supply drops (3 cubes + full HP/energy), loot, cubes, then hold a safe spot
            drop_goal = None
            if not HIDE:
                hot = any(los and dist < 120 for _m, dist, los in visible_enemies)
                for dr in obs.get("drops", []):
                    dd = d2(dr["pos"], pos)
                    holder = dr.get("holder")
                    if dd < 320 and not hot and (holder is None or holder == me["id"])                             and self.in_circle(dr["pos"], safe_c, safe_r, 0) and dr["lands_in"] < dd / 11.0 + 6:
                        drop_goal = dr["pos"]
                        break
            if drop_goal is not None:
                goal = drop_goal
            elif best_pick and best_pick[2] < 140:
                goal = best_pick[1]["pos"]
            else:
                cb = None
                for c in obs["cubes"]:
                    dist = d2(c["pos"], pos)
                    if dist < 120 and self.in_circle(c["pos"], cur_c, cur_r, 5) and (cb is None or dist < cb[0]):
                        cb = (dist, c)
                if cb:
                    goal = cb[1]["pos"]
                else:
                    # supply drop
                    for dr in obs.get("drops", []):
                        dd = d2(dr["pos"], pos)
                        if dd < 200 and hp_frac > 0.4 and not threat_near and self.in_circle(dr["pos"], safe_c, safe_r, 0):
                            if dr["lands_in"] < dd / 11.0 + 3:
                                goal = dr["pos"]
                                break
                    if goal is None and t < 210 and hp_frac > 0.45 and not HIDE:
                        goal = self.farm_goal(pos, muzzle, safe_c, safe_r, t)
                    if goal is None and shoot_target and shoot_target[3] in ("crate", "npc") and not HIDE:
                        tp = shoot_target[0]
                        if shoot_target[5] > 40:
                            goal = tp
                    if goal is None:
                        # hold position inside the safe circle
                        if not self.in_circle(pos, safe_c, safe_r, max(8.0, safe_r * 0.25)):
                            # point on the way to the center, inside the circle
                            cx, cy = safe_c
                            dx, dy = pos[0] - cx, pos[1] - cy
                            dn = math.hypot(dx, dy) or 1
                            rr = safe_r * 0.55
                            goal = (cx + dx / dn * rr, cy + dy / dn * rr)
                        else:
                            walk_slow = True
                            goal = None

        # final override: never leave the safe circle for loot
        if goal is not None and not engaged and me["in_zone"]:
            if not self.in_circle(goal, safe_c, safe_r, 3) and self.in_circle(pos, safe_c, safe_r, 3):
                goal = None
        # power cubes close by are worth a short detour (kills drop the victim's cubes + 2)
        if self.mode != "cover" and me["in_zone"]:
            hot = any(los and dist < 60 for _m, dist, los in visible_enemies)
            if not hot:
                cb = None
                for c in obs["cubes"]:
                    dc = d2(c["pos"], pos)
                    if dc < 25 and abs(c["pos"][2] - pos[2]) < 6 and (cb is None or dc < cb[0]):
                        cb = (dc, c["pos"])
                if cb:
                    goal = cb[1]
        if goal is not None:
            move = self.nav_to(goal, t, pos)
        elif move == (0.0, 0.0) and not self.in_circle(pos, safe_c, safe_r, 4):
            move = self.nav_to(safe_c, t, pos)

        # urgency: need to get inside the shrinking circle
        if not self.in_circle(pos, safe_c, safe_r, 0) and not engaged:
            z_left = zone["shrink_start"] - t
            dist_need = d2(pos, safe_c) - safe_r
            if z_left < dist_need / 11.0 + 10:
                boost = me["energy"] > 25

        # dodges override
        if danger is not None:
            ax, ay_ = pos[0] - danger[0], pos[1] - danger[1]
            an = math.hypot(ax, ay_) or 1
            move = (ax / an, ay_ / an)
            boost = me["energy"] > (35 if "lightning" in has else 10)
        elif laser_threat is not None:
            m, byaw = laser_threat
            if t > self.dodge_until:
                self.dodge_vec = self.rng.choice((-1, 1))
                self.dodge_until = t + 0.6
            px, py = -math.sin(byaw) * self.dodge_vec, math.cos(byaw) * self.dodge_vec
            move = (px * 0.9 + move[0] * 0.3, py * 0.9 + move[1] * 0.3)
            boost = me["energy"] > (60 if "lightning" in has else 20)

        # stuck handling
        self.last_pos_hist.append((t, pos[0], pos[1]))
        while self.last_pos_hist and t - self.last_pos_hist[0][0] > 1.5:
            self.last_pos_hist.pop(0)
        mv = math.hypot(*move)
        if mv > 0.5 and len(self.last_pos_hist) > 8:
            t0_, x0_, y0_ = self.last_pos_hist[0]
            if t - t0_ > 1.2 and math.hypot(pos[0] - x0_, pos[1] - y0_) < 2.0:
                self.stuck_until = t + 1.0
                self.stuck_dir = self.rng.uniform(-math.pi, math.pi)
                self.path = None
                self.path_req_t = -99
                self.last_pos_hist.clear()
        if t < self.stuck_until:
            jump = me["grounded"]
            boost = me["energy"] > 5
            if self.rng.random() < 0.5 and self.stuck_dir is not None:
                move = (move[0] * 0.5 + math.cos(self.stuck_dir) * 0.5, move[1] * 0.5 + math.sin(self.stuck_dir) * 0.5)

        if walk_slow:
            move = (move[0] * 0.38, move[1] * 0.38)

        # charging laser: no boost (we're slowed anyway)
        # ---------------- aim when no shooting target
        if aim is None:
            ty = None
            if t - self.last_damage_t < 4 and self.last_damage_bearing is not None:
                ty = self.last_damage_bearing
            elif visible_enemies:
                m = min(visible_enemies, key=lambda v: v[1])[0]
                ty = math.atan2(m["pos"][1] - pos[1], m["pos"][0] - pos[0])
            elif t - self.sound_t < 3 and self.sound_bearing is not None:
                ty = self.sound_bearing
            if ty is None:
                # scan: sweep around the movement direction
                base = math.atan2(move[1], move[0]) if math.hypot(*move) > 0.2 else me["aim"][0]
                if self.scan_yaw is None:
                    self.scan_yaw = base
                self.scan_yaw = wrap(self.scan_yaw + self.scan_dir * 0.25)
                if abs(wrap(self.scan_yaw - base)) > 1.3:
                    self.scan_dir *= -1
                ty = self.scan_yaw
            aim = [ty, 0.0]

        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["boost"] = bool(boost)
        act["jump"] = bool(jump)
        act["aim"] = [round(aim[0], 4), round(max(-0.69, min(1.29, aim[1])), 4)]
        self.prev_cmd = act["aim"]
        act["fire"] = fire
        if reload:
            act["reload"] = True
        mf.draw.label(f"{self.mode} hp{me['hp']:.0f} tgt{self.target_id}")
        self.last_t = t
        return act


mf.run(Bot)
