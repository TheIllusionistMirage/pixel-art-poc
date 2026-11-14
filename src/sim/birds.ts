// Rooks: a few flocks, each living in a rookery (2–3 big oaks with nests).
//
// At night the birds sit on the crowns of their rookery trees, now and then
// shuffling to another spot. Around sunrise each flock in turn (every flock has
// its own sun-height threshold) lifts off, a few birds at a time, and flies out
// to a field or an open meadow, where they land, peck and hop about. Every so
// often a flock moves on to another patch, sometimes circling over the village
// on the way. Over twilight they stream back, wheel noisily over the rookery,
// and drop onto their perches one by one.
//
// Bad weather (above a flock's hardiness) sends a flock into the nearest tree
// for cover; birds at the roost huddle into the crown. A nearby lightning strike
// flushes a flock: they burst out in all directions and regroup. Wind pushes
// flying birds sideways and makes the flock more ragged.
//
// Stored as struct-of-arrays like the villagers. Flying birds follow a loose
// boids model: each one steers towards the flock's virtual leader plus its own
// offset, keeps away from close neighbours, and is limited in speed and turn.

import { DETAIL } from '../assets/voxel';
import { clamp, lerp, mulberry32, range, type Rng } from '../engine/rng';
import type { BirdSites, ForagePatch, Rookery, RoostTree } from '../world/village';
import { facing } from '../world/village';
import type { Strike } from '../world/weather';

// Bird modes
const PERCH = 0;   // on a roost perch (sways with the tree)
const GROUND = 1;  // feeding on a forage patch
const FLY = 2;     // flying with the flock
const LAND = 3;    // gliding in to a perch, a ground spot or a cover tree
const HIDDEN = 4;  // tucked inside a crown (bad weather), not drawn

// Where a landing bird is headed (LAND).
const TO_GROUND = 0, TO_PERCH = 1, TO_COVER = 2;

// Delayed per-bird actions, so a flock never moves as one block.
const P_NONE = 0, P_TAKEOFF = 1, P_HIDE = 2, P_SHOW = 3, P_SETTLE = 4;

// Ground / perch activities.
const A_IDLE = 0, A_PECK = 1, A_HOP = 2;

// Flock modes
const F_ROOST = 0;   // in the rookery: perched, or hidden from the weather
const F_TRAVEL = 1;  // flying to a forage patch (maybe circling somewhere first)
const F_FORAGE = 2;  // on the ground at a patch
const F_HOME = 3;    // flying back to the rookery
const F_WHEEL = 4;   // circling over the rookery before settling
const F_SETTLE = 5;  // still circling, birds dropping onto their perches
const F_COVER = 6;   // heading into / hiding in a tree from bad weather
const F_SCATTER = 7; // flushed by lightning

/** Rook model frames (see Library.rooks): flap cycle over 0..2, 3 stand, 4 peck. */
const FLAP = [0, 1, 2, 1] as const;
const STAND = 3, PECK = 4;

/** Wind (≈0–1.3) → sideways push on anything in the air, world units / s. */
const WIND_PUSH = 7;
const TAU = Math.PI * 2;
const wrap = (a: number) => ((((a + Math.PI) % TAU) + TAU) % TAU) - Math.PI;

export interface BirdEnv {
  hour: number;
  /** sunHeight(hour): sine of the sun's elevation. */
  sun: number;
  /** Weather `shelter` (0 fine … 1 everyone takes cover). */
  shelter: number;
  wind: readonly [number, number];
  /** Seconds, same clock as the renderer's wind sway. */
  time: number;
  strikes: readonly Strike[];
}

export interface BirdSound {
  kind: 'caw' | 'flutter';
  x: number;
  y: number;
  z: number;
  /** caw: calls in the series; flutter: how many birds took off together. */
  n: number;
  /** 0..1, how urgent (alarm calls, wheeling) */
  urgency: number;
}

interface Perch {
  x: number;
  y: number;
  z: number;
  tree: RoostTree;
  /** Bird sitting here (or on its way), -1 if free. */
  bird: number;
}

class Flock {
  mode = F_ROOST;
  modeT = 0;
  started = false;
  day = false;
  sheltered = false;
  // Virtual leader the birds follow.
  lx = 0; ly = 0; lz = 0; lvz = 0;
  heading = 0;
  speed = 0;
  /** Leader height target. */
  lzT = 0;
  destX = 0; destY = 0;
  patch = -1;
  circleX = 0; circleY = 0; circleR = 30; circleT = 0; circleDir = 1;
  /** Random waypoints (x, y, height) to fly through before heading for destX/destY. */
  wps: [number, number, number][] = [];
  cover = -1;
  arrived = false;
  relocateT = 0;
  scatterT = 0;
  lastScatter = -1e9;
  shuffleT = 0;
  // Per-frame sound bookkeeping.
  flutterN = 0; flutterX = 0; flutterY = 0; flutterZ = 0;

  constructor(
    readonly first: number,
    readonly n: number,
    readonly rookery: Rookery,
    readonly perches: Perch[],
    /** Leaves the roost once the sun is above this (morning). */
    readonly dawnT: number,
    /** Heads home once the sun drops below this (evening). */
    readonly duskT: number,
    /** Stops wheeling and settles once the sun is below this. */
    readonly settleT: number,
    readonly hardiness: number,
    readonly cruise: number,
    readonly cruiseSpeed: number,
  ) {}
}

export class Birds {
  readonly count: number;
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly z: Float32Array;
  readonly angle: Float32Array;
  /** Index into Library.rooks. */
  readonly frame: Uint8Array;
  private readonly mode: Uint8Array;
  private readonly flockOf: Uint8Array;
  private readonly vx: Float32Array;
  private readonly vy: Float32Array;
  private readonly vz: Float32Array;
  /** Home perch (index into the flock's perches). */
  private readonly perch: Int16Array;
  private readonly pend: Uint8Array;
  private readonly delay: Float32Array;
  private readonly landTo: Uint8Array;
  private readonly landT: Float32Array;
  private readonly tx: Float32Array;
  private readonly ty: Float32Array;
  private readonly tz: Float32Array;
  /** Place in the flock relative to the leader. */
  private readonly ox: Float32Array;
  private readonly oy: Float32Array;
  private readonly oz: Float32Array;
  private readonly ph: Float32Array;
  private readonly flapT: Float32Array;
  private readonly glideT: Float32Array;
  private readonly gliding: Uint8Array;
  private readonly panic: Float32Array;
  private readonly act: Uint8Array;
  private readonly actT: Float32Array;
  private readonly hopT: Float32Array;
  private readonly hopX: Float32Array;
  private readonly hopY: Float32Array;
  private readonly flocks: Flock[] = [];
  private readonly rng: Rng;
  private sounds: BirdSound[] = [];
  private wind: readonly [number, number] = [0, 0];
  private time = 0;
  /** Live counts, for the HUD. */
  flying = 0;
  hiddenCount = 0;

  constructor(private readonly sites: BirdSites, private readonly center: readonly [number, number], seed = 1) {
    const rng = (this.rng = mulberry32(seed));
    const plan: { rk: Rookery; perches: Perch[]; n: number }[] = [];
    let total = 0;
    for (const rk of sites.rookeries) {
      const perches: Perch[] = rk.trees.flatMap((tree) => tree.perches.map(([x, y, z]) => ({ x, y, z, tree, bird: -1 })));
      // Shuffle so home perches spread over all the trees.
      for (let i = perches.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [perches[i], perches[j]] = [perches[j], perches[i]];
      }
      const n = Math.min(Math.floor(range(rng, 2, 6)), perches.length); // 2–5 birds
      if (n < 1 || !sites.forage.length) continue;
      plan.push({ rk, perches, n });
      total += n;
    }
    this.count = total;
    const F = () => new Float32Array(total), U = () => new Uint8Array(total);
    this.x = F(); this.y = F(); this.z = F(); this.angle = F(); this.frame = U();
    this.mode = U(); this.flockOf = U(); this.vx = F(); this.vy = F(); this.vz = F();
    this.perch = new Int16Array(total); this.pend = U(); this.delay = F();
    this.landTo = U(); this.landT = F(); this.tx = F(); this.ty = F(); this.tz = F();
    this.ox = F(); this.oy = F(); this.oz = F(); this.ph = F();
    this.flapT = F(); this.glideT = F(); this.gliding = U(); this.panic = F();
    this.act = U(); this.actT = F(); this.hopT = F(); this.hopX = F(); this.hopY = F();

    let first = 0;
    for (const { rk, perches, n } of plan) {
      const f = new Flock(
        first, n, rk, perches,
        range(rng, -0.1, 0.04), range(rng, 0.1, 0.24), range(rng, -0.12, -0.04),
        range(rng, 0.38, 0.6), range(rng, 64, 76), range(rng, 26, 34),
      );
      const fi = this.flocks.length;
      this.flocks.push(f);
      const spread = Math.sqrt(n) * 2.2;
      for (let k = 0; k < n; k++) {
        const i = first + k;
        this.flockOf[i] = fi;
        this.perch[i] = k;
        perches[k].bird = i;
        const a = rng() * TAU, r = spread * Math.sqrt(rng());
        this.ox[i] = Math.cos(a) * r * 1.3;
        this.oy[i] = Math.sin(a) * r;
        this.oz[i] = range(rng, -3, 3);
        this.ph[i] = rng() * 100;
        this.angle[i] = rng() * TAU;
        this.frame[i] = STAND;
        const p = perches[k];
        this.x[i] = p.x; this.y[i] = p.y; this.z[i] = p.z;
      }
      first += n;
    }
  }

  /** Is bird i drawn at all? */
  visible(i: number): boolean {
    return this.mode[i] !== HIDDEN;
  }

  /** In the air (drawn into the heightmap's airborne layer). */
  airborne(i: number): boolean {
    return this.mode[i] === FLY || this.mode[i] === LAND;
  }

  get flockCount(): number {
    return this.flocks.length;
  }

  /** Sound events since the last call. */
  takeSounds(): BirdSound[] {
    const s = this.sounds;
    this.sounds = [];
    return s;
  }

  update(dt: number, env: BirdEnv): void {
    this.wind = env.wind;
    this.time = env.time;
    for (const s of env.strikes) for (const f of this.flocks) this.maybeScatter(f, s);
    for (const f of this.flocks) {
      this.updateFlock(f, dt, env);
      f.flutterN = 0;
      f.flutterX = f.flutterY = f.flutterZ = 0;
    }
    let flying = 0, hidden = 0;
    for (let i = 0; i < this.count; i++) {
      this.updateBird(i, dt);
      if (this.airborne(i)) flying++;
      else if (this.mode[i] === HIDDEN) hidden++;
    }
    this.flying = flying;
    this.hiddenCount = hidden;
    for (const f of this.flocks)
      if (f.flutterN > 0)
        this.sounds.push({ kind: 'flutter', x: f.flutterX / f.flutterN, y: f.flutterY / f.flutterN, z: f.flutterZ / f.flutterN, n: f.flutterN, urgency: f.mode === F_SCATTER ? 1 : 0 });
  }

  // ── Flock level ──────────────────────────────────────────────────────

  private updateFlock(f: Flock, dt: number, env: BirdEnv): void {
    f.modeT += dt;
    f.day = env.sun > (env.hour < 12 ? f.dawnT : f.duskT);
    if (!f.sheltered && env.shelter > f.hardiness) f.sheltered = true;
    else if (f.sheltered && env.shelter < f.hardiness - 0.15) f.sheltered = false;

    if (!f.started) {
      f.started = true;
      this.startIn(f);
      return;
    }

    const out = f.day && !f.sheltered;
    switch (f.mode) {
      case F_ROOST:
        if (out) this.startTravel(f, 4, false);
        else this.roost(f, dt, env);
        break;
      case F_TRAVEL:
      case F_FORAGE:
        if (!f.day) this.goHome(f);
        else if (f.sheltered) this.goCover(f);
        break;
      case F_HOME:
      case F_WHEEL:
      case F_SETTLE:
        if (out) this.startTravel(f, 1, false);
        break;
      case F_COVER:
        if (!f.sheltered) {
          if (f.day) this.startTravel(f, 2.5, false);
          else this.goHome(f);
        }
        break;
      case F_SCATTER:
        f.scatterT -= dt;
        if (f.scatterT <= 0) this.decide(f);
        break;
    }

    // Leader motion.
    switch (f.mode) {
      case F_TRAVEL: {
        if (f.circleT > 0) {
          f.circleT -= dt;
          this.circle(f, dt, f.circleX, f.circleY, f.circleR);
          break;
        }
        if (f.wps.length) {
          const [wx, wy, wz] = f.wps[0];
          f.lzT = wz;
          this.steer(f, dt, wx - f.lx, wy - f.ly, f.cruiseSpeed);
          if (Math.hypot(wx - f.lx, wy - f.ly) < 18) f.wps.shift();
          break;
        }
        const d = Math.hypot(f.destX - f.lx, f.destY - f.ly);
        f.lzT = lerp(10, f.cruise, clamp(d / 140, 0, 1));
        this.steer(f, dt, f.destX - f.lx, f.destY - f.ly, Math.min(f.cruiseSpeed, 10 + d * 0.6));
        if (d < 16) this.arriveForage(f);
        break;
      }
      case F_FORAGE:
        f.relocateT -= dt;
        if (f.relocateT <= 0 && this.allIn(f, GROUND)) this.startTravel(f, 1.2, true);
        break;
      case F_HOME: {
        const d = Math.hypot(f.rookery.x - f.lx, f.rookery.y - f.ly);
        f.lzT = f.cruise - 8;
        this.steer(f, dt, f.rookery.x - f.lx, f.rookery.y - f.ly, f.cruiseSpeed);
        if (d < 45) {
          f.mode = F_WHEEL;
          f.modeT = 0;
          f.circleR = range(this.rng, 24, 34);
          f.circleDir = this.rng() < 0.5 ? -1 : 1;
        }
        break;
      }
      case F_WHEEL:
      case F_SETTLE: {
        f.lzT = f.cruise - 10;
        this.circle(f, dt, f.rookery.x, f.rookery.y, f.circleR);
        if (f.mode === F_WHEEL) {
          const minT = f.sheltered ? 1.5 : 5;
          if (f.modeT > minT && (env.sun < f.settleT || f.sheltered || f.modeT > 30)) {
            f.mode = F_SETTLE;
            f.modeT = 0;
            for (let i = f.first; i < f.first + f.n; i++)
              if (this.mode[i] === FLY) this.schedule(i, P_SETTLE, this.rng() * (f.sheltered ? 2 : 7));
          }
        } else if (this.allIn(f, PERCH, HIDDEN)) {
          f.mode = F_ROOST;
          f.modeT = 0;
          f.shuffleT = range(this.rng, 3, 8);
        }
        break;
      }
      case F_COVER:
        if (!f.arrived && f.cover >= 0) {
          const c = this.sites.cover[f.cover];
          const d = Math.hypot(c.x - f.lx, c.y - f.ly);
          f.lzT = c.z + 10;
          this.steer(f, dt, c.x - f.lx, c.y - f.ly, Math.min(f.cruiseSpeed * 1.2, 12 + d * 0.7));
          if (d < 14) {
            f.arrived = true;
            for (let i = f.first; i < f.first + f.n; i++) if (this.mode[i] === FLY) this.landInCover(i, f);
          }
        }
        break;
      case F_SCATTER:
        f.lzT = f.cruise - 6;
        this.steer(f, dt, Math.cos(f.heading), Math.sin(f.heading), 18);
        break;
    }

    this.calls(f, dt, env);
  }

  /** First frame: put each flock where it would be at this hour. */
  private startIn(f: Flock): void {
    if (f.day && !f.sheltered) {
      f.patch = this.pickPatch(f.rookery.x, f.rookery.y, -1);
      const p = this.sites.forage[f.patch];
      f.mode = F_FORAGE;
      f.relocateT = range(this.rng, 3, 30);
      for (let i = f.first; i < f.first + f.n; i++) {
        const [x, y] = this.patchPoint(p, range(this.rng, -0.7, 0.7), range(this.rng, -0.7, 0.7));
        this.x[i] = x; this.y[i] = y; this.z[i] = 0;
        this.mode[i] = GROUND;
        this.frame[i] = STAND;
      }
    } else {
      f.mode = F_ROOST;
      f.shuffleT = range(this.rng, 3, 8);
      for (let i = f.first; i < f.first + f.n; i++) this.mode[i] = f.sheltered ? HIDDEN : PERCH;
    }
  }

  /** At the roost: huddle into the crown in bad weather, come out after; shuffle about at night. */
  private roost(f: Flock, dt: number, env: BirdEnv): void {
    for (let i = f.first; i < f.first + f.n; i++) {
      if (this.pend[i]) continue;
      if (f.sheltered && this.mode[i] === PERCH) this.schedule(i, P_HIDE, this.rng() * 2.5);
      else if (!f.sheltered && this.mode[i] === HIDDEN) this.schedule(i, P_SHOW, this.rng() * 4);
      else if (this.mode[i] === FLY) this.landOnPerch(i, f);
    }
    if (f.sheltered) return;
    f.shuffleT -= dt;
    if (f.shuffleT > 0) return;
    // Busier just before dawn.
    f.shuffleT = env.sun > f.dawnT - 0.1 ? range(this.rng, 0.5, 2) : range(this.rng, 2.5, 8);
    const i = f.first + Math.floor(this.rng() * f.n);
    if (this.mode[i] !== PERCH) return;
    const own = f.perches[this.perch[i]];
    const free = f.perches.findIndex((p) => p.bird < 0 && p.tree === own.tree && Math.hypot(p.x - own.x, p.y - own.y) < 12);
    if (free >= 0 && this.rng() < 0.45) {
      own.bird = -1;
      f.perches[free].bird = i;
      this.perch[i] = free;
      this.landOnPerch(i, f);
      this.vz[i] = 3;
    } else this.angle[i] += range(this.rng, -1.4, 1.4);
  }

  private decide(f: Flock): void {
    if (f.sheltered && f.day) this.goCover(f);
    else if (!f.day) this.goHome(f);
    else this.startTravel(f, 0.5, false);
  }

  private flyingMode(f: Flock): boolean {
    return f.mode === F_TRAVEL || f.mode === F_HOME || f.mode === F_WHEEL || f.mode === F_SETTLE || f.mode === F_SCATTER || (f.mode === F_COVER && !f.arrived);
  }

  /** Put the leader at the flock's centre, heading for (tx, ty), unless it is already flying. */
  private initLeader(f: Flock, tx: number, ty: number): void {
    if (this.flyingMode(f)) return;
    let sx = 0, sy = 0, sz = 0;
    for (let i = f.first; i < f.first + f.n; i++) {
      sx += this.x[i]; sy += this.y[i]; sz += this.z[i];
    }
    f.lx = sx / f.n;
    f.ly = sy / f.n;
    f.lz = sz / f.n + 4;
    f.speed = 0;
    f.heading = Math.atan2(ty - f.ly, tx - f.lx);
  }

  private takeOffAll(f: Flock, stagger: number): void {
    for (let i = f.first; i < f.first + f.n; i++) {
      const m = this.mode[i];
      if (m === PERCH || m === GROUND || m === HIDDEN) this.schedule(i, P_TAKEOFF, this.rng() * stagger);
      else {
        this.pend[i] = P_NONE;
        this.mode[i] = FLY;
      }
    }
  }

  private startTravel(f: Flock, stagger: number, relocate: boolean): void {
    const [cx, cy] = f.mode === F_ROOST ? [f.rookery.x, f.rookery.y] : this.centroid(f);
    // Daytime flights: sometimes a loop that comes back to the same patch.
    const sally = relocate && this.rng() < 0.35;
    if (!sally) f.patch = this.pickPatch(cx, cy, relocate ? f.patch : -1);
    const p = this.sites.forage[f.patch];
    [f.destX, f.destY] = this.patchPoint(p, range(this.rng, -0.5, 0.5), range(this.rng, -0.5, 0.5));
    f.circleT = 0;
    f.wps = [];
    const r = this.rng();
    if (relocate && r < 0.3) {
      // Take a turn over the village on the way.
      f.circleX = this.center[0] + range(this.rng, -60, 60);
      f.circleY = this.center[1] + range(this.rng, -60, 60);
      f.circleR = range(this.rng, 30, 45);
      f.circleT = range(this.rng, 8, 14);
      f.circleDir = this.rng() < 0.5 ? -1 : 1;
    } else if (relocate && (sally || r < 0.85)) {
      // Wander: a few random waypoints around where the flock is, at varying heights.
      const n = 1 + Math.floor(this.rng() * 3), lim = this.center[0] * 2 - 40;
      let [x, y] = [cx, cy];
      for (let k = 0; k < n; k++) {
        const a = this.rng() * TAU, d = range(this.rng, 50, 160);
        x = clamp(x + Math.cos(a) * d, 40, lim);
        y = clamp(y + Math.sin(a) * d, 40, lim);
        f.wps.push([x, y, range(this.rng, 25, f.cruise)]);
      }
    }
    const [hx, hy] = f.circleT > 0 ? [f.circleX, f.circleY] : f.wps.length ? [f.wps[0][0], f.wps[0][1]] : [f.destX, f.destY];
    this.initLeader(f, hx, hy);
    f.lzT = f.cruise;
    this.takeOffAll(f, stagger);
    f.mode = F_TRAVEL;
    f.modeT = 0;
    this.caw(f, 2, 0.5);
  }

  private goHome(f: Flock): void {
    this.initLeader(f, f.rookery.x, f.rookery.y);
    this.takeOffAll(f, 1.5);
    f.mode = F_HOME;
    f.modeT = 0;
  }

  private goCover(f: Flock): void {
    const [cx, cy] = this.centroid(f);
    let best = -1, bd = Infinity;
    this.sites.cover.forEach((c, k) => {
      const d = (c.x - cx) ** 2 + (c.y - cy) ** 2;
      if (d < bd) {
        bd = d;
        best = k;
      }
    });
    if (best < 0) return this.goHome(f);
    f.cover = best;
    this.initLeader(f, this.sites.cover[best].x, this.sites.cover[best].y);
    this.takeOffAll(f, 1);
    f.arrived = false;
    f.mode = F_COVER;
    f.modeT = 0;
    this.caw(f, 2, 0.6);
  }

  private arriveForage(f: Flock): void {
    f.mode = F_FORAGE;
    f.modeT = 0;
    f.relocateT = range(this.rng, 12, 35);
    for (let i = f.first; i < f.first + f.n; i++) if (this.mode[i] === FLY) this.landOnGround(i, f);
  }

  private maybeScatter(f: Flock, s: Strike): void {
    if (this.time - f.lastScatter < 6 || !f.started) return;
    const [cx, cy] = this.centroid(f);
    if (Math.hypot(cx - s.x, cy - s.y) > 120 + 120 * s.strength) return;
    f.lastScatter = this.time;
    for (let i = f.first; i < f.first + f.n; i++) {
      const a = Math.atan2(this.y[i] - s.y, this.x[i] - s.x) + range(this.rng, -0.9, 0.9);
      const v = range(this.rng, 22, 40);
      if (this.mode[i] !== FLY && this.mode[i] !== LAND) this.flutter(f, i);
      this.mode[i] = FLY;
      this.pend[i] = P_NONE;
      this.vx[i] = Math.cos(a) * v;
      this.vy[i] = Math.sin(a) * v;
      this.vz[i] = range(this.rng, 6, 14);
      this.panic[i] = range(this.rng, 1.2, 2.6);
    }
    const [lx, ly] = this.centroid(f);
    f.lx = lx;
    f.ly = ly;
    f.lz = Math.max(f.lz, 20);
    f.heading = Math.atan2(ly - s.y, lx - s.x);
    f.speed = 15;
    f.mode = F_SCATTER;
    f.modeT = 0;
    f.arrived = false;
    f.scatterT = range(this.rng, 3.5, 5.5);
    for (let k = 0; k < 3; k++) this.caw(f, 3, 1);
  }

  /** Steer the leader towards direction (dx, dy) at `speed`, with limited turn rate and acceleration. */
  private steer(f: Flock, dt: number, dx: number, dy: number, speed: number): void {
    const want = Math.atan2(dy, dx);
    f.heading += clamp(wrap(want - f.heading), -1.2 * dt, 1.2 * dt);
    f.speed += clamp(speed - f.speed, -15 * dt, 15 * dt);
    f.lx += (Math.cos(f.heading) * f.speed + this.wind[0] * WIND_PUSH) * dt;
    f.ly += (Math.sin(f.heading) * f.speed + this.wind[1] * WIND_PUSH) * dt;
    const z0 = f.lz;
    f.lz += clamp(f.lzT - f.lz, -7 * dt, 7 * dt);
    f.lvz = (f.lz - z0) / Math.max(dt, 1e-4);
  }

  /** Steer the leader round a circle (cx, cy, r): tangent plus a pull back onto the circle. */
  private circle(f: Flock, dt: number, cx: number, cy: number, r: number): void {
    const rx = f.lx - cx, ry = f.ly - cy;
    const d = Math.hypot(rx, ry) || 1;
    const tx = (-ry / d) * f.circleDir, ty = (rx / d) * f.circleDir;
    const pull = clamp((d - r) / r, -1, 1.5);
    this.steer(f, dt, tx - (rx / d) * pull, ty - (ry / d) * pull, f.cruiseSpeed * 0.85);
  }

  private calls(f: Flock, dt: number, env: BirdEnv): void {
    let rate: number;
    switch (f.mode) {
      case F_ROOST: rate = f.sheltered ? 0.02 : !f.day && env.sun > f.dawnT - 0.1 ? 0.9 : env.sun < -0.15 ? 0.03 : 0.12; break;
      case F_FORAGE: rate = 0.15; break;
      case F_WHEEL: case F_SETTLE: rate = 2.2; break;
      case F_SCATTER: rate = 3; break;
      case F_COVER: rate = f.arrived ? 0.02 : 0.4; break;
      default: rate = 0.35;
    }
    // Rates are for a flock of ~25; smaller flocks call less.
    rate *= Math.sqrt(f.n / 25);
    if (this.rng() < rate * dt) this.caw(f, 1 + Math.floor(this.rng() * 3), f.mode === F_WHEEL || f.mode === F_SETTLE ? 0.6 : f.mode === F_SCATTER ? 1 : 0.2);
  }

  private caw(f: Flock, n: number, urgency: number): void {
    const i = f.first + Math.floor(this.rng() * f.n);
    // Birds tucked into a crown call less and sound muffled (fewer calls).
    if (this.mode[i] === HIDDEN && this.rng() < 0.7) return;
    this.sounds.push({ kind: 'caw', x: this.x[i], y: this.y[i], z: this.z[i], n, urgency });
  }

  private flutter(f: Flock, i: number): void {
    f.flutterN++;
    f.flutterX += this.x[i];
    f.flutterY += this.y[i];
    f.flutterZ += this.z[i];
  }

  private centroid(f: Flock): [number, number] {
    let sx = 0, sy = 0;
    for (let i = f.first; i < f.first + f.n; i++) {
      sx += this.x[i];
      sy += this.y[i];
    }
    return [sx / f.n, sy / f.n];
  }

  private allIn(f: Flock, a: number, b = a): boolean {
    for (let i = f.first; i < f.first + f.n; i++) if (this.mode[i] !== a && this.mode[i] !== b) return false;
    return true;
  }

  /** A patch near (x, y): one of the closest few, never `not`. */
  private pickPatch(x: number, y: number, not: number): number {
    const order = this.sites.forage
      .map((p, k) => ({ k, d: Math.hypot(p.x - x, p.y - y) + this.rng() * 60 }))
      .filter((e) => e.k !== not && e.d > 20)
      .sort((a, b) => a.d - b.d);
    if (!order.length) return Math.max(0, not);
    return order[Math.floor(this.rng() * Math.min(4, order.length))].k;
  }

  private patchPoint(p: ForagePatch, u: number, v: number): [number, number] {
    return [p.x + p.ca * u * p.hw - p.sa * v * p.hh, p.y + p.sa * u * p.hw + p.ca * v * p.hh];
  }

  private inPatch(p: ForagePatch, x: number, y: number): boolean {
    const dx = x - p.x, dy = y - p.y;
    return Math.abs(p.ca * dx + p.sa * dy) < p.hw && Math.abs(-p.sa * dx + p.ca * dy) < p.hh;
  }

  // ── Bird level ───────────────────────────────────────────────────────

  private schedule(i: number, what: number, delay: number): void {
    this.pend[i] = what;
    this.delay[i] = delay;
  }

  private landOnGround(i: number, f: Flock): void {
    const p = this.sites.forage[f.patch];
    // Near the leader's spot, roughly keeping the bird's place in the flock.
    let x = f.destX + this.ox[i] * 0.7 + range(this.rng, -2, 2), y = f.destY + this.oy[i] * 0.7 + range(this.rng, -2, 2);
    if (!this.inPatch(p, x, y)) [x, y] = this.patchPoint(p, range(this.rng, -0.8, 0.8), range(this.rng, -0.8, 0.8));
    this.land(i, x, y, 0, TO_GROUND);
  }

  private landOnPerch(i: number, f: Flock): void {
    const p = f.perches[this.perch[i]];
    this.land(i, p.x, p.y, p.z, TO_PERCH);
  }

  private landInCover(i: number, f: Flock): void {
    const c = this.sites.cover[f.cover];
    const a = this.rng() * TAU, r = this.rng() * 6;
    this.land(i, c.x + Math.cos(a) * r, c.y + Math.sin(a) * r, c.z + range(this.rng, -3, 2), TO_COVER);
  }

  private land(i: number, x: number, y: number, z: number, to: number): void {
    this.mode[i] = LAND;
    this.landTo[i] = to;
    this.landT[i] = 0;
    this.tx[i] = x;
    this.ty[i] = y;
    this.tz[i] = z;
  }

  /** Where a perch is right now, with the tree's wind sway (same formula as the vertex shader). */
  private perchNow(p: Perch): [number, number] {
    const t = p.tree, zv = p.z * DETAIL;
    const ph = t.x * 0.37 + t.y * 0.23;
    const wob = 0.6 + 0.4 * Math.sin(this.time * 2.1 + ph) + 0.15 * Math.sin(this.time * 5.3 + ph * 2.7);
    const k = t.sway * zv * zv * wob;
    return [p.x + this.wind[0] * k, p.y + this.wind[1] * k];
  }

  private updateBird(i: number, dt: number): void {
    const f = this.flocks[this.flockOf[i]];
    if (this.pend[i]) {
      this.delay[i] -= dt;
      if (this.delay[i] <= 0) this.runPending(i, f);
    }
    switch (this.mode[i]) {
      case PERCH: this.perched(i, f, dt); break;
      case GROUND: this.ground(i, f, dt); break;
      case FLY: this.fly(i, f, dt); break;
      case LAND: this.landing(i, f, dt); break;
    }
  }

  private runPending(i: number, f: Flock): void {
    const what = this.pend[i];
    this.pend[i] = P_NONE;
    const m = this.mode[i];
    if (what === P_TAKEOFF && (m === PERCH || m === GROUND || m === HIDDEN)) {
      this.mode[i] = FLY;
      this.vx[i] = range(this.rng, -3, 3);
      this.vy[i] = range(this.rng, -3, 3);
      this.vz[i] = range(this.rng, 5, 9);
      this.gliding[i] = 0;
      this.glideT[i] = range(this.rng, 1, 2);
      this.flutter(f, i);
    } else if (what === P_HIDE && m === PERCH) this.mode[i] = HIDDEN;
    else if (what === P_SHOW && m === HIDDEN) this.mode[i] = PERCH;
    else if (what === P_SETTLE && m === FLY) this.landOnPerch(i, f);
  }

  private perched(i: number, f: Flock, dt: number): void {
    const p = f.perches[this.perch[i]];
    const [x, y] = this.perchNow(p);
    this.x[i] = x;
    this.y[i] = y;
    this.z[i] = p.z;
    this.actT[i] -= dt;
    if (this.actT[i] <= 0) {
      const r = this.rng();
      const night = !f.day;
      if (r < 0.3) this.angle[i] += range(this.rng, -0.8, 0.8);
      this.act[i] = r > 0.8 ? A_PECK : A_IDLE; // preening looks like a peck
      this.actT[i] = this.act[i] === A_PECK ? 0.35 : range(this.rng, night ? 3 : 1, night ? 10 : 4);
    }
    this.frame[i] = this.act[i] === A_PECK ? PECK : STAND;
  }

  private ground(i: number, f: Flock, dt: number): void {
    const p = this.sites.forage[f.patch];
    if (this.act[i] === A_HOP) {
      this.hopT[i] += dt / 0.28;
      const h = Math.min(1, this.hopT[i]);
      this.x[i] += (this.hopX[i] * dt) / 0.28;
      this.y[i] += (this.hopY[i] * dt) / 0.28;
      this.z[i] = 2.8 * h * (1 - h) * 0.25 * 4;
      if (this.hopT[i] >= 1) {
        this.z[i] = 0;
        this.act[i] = A_IDLE;
      }
    }
    this.actT[i] -= dt;
    if (this.actT[i] <= 0 && this.act[i] !== A_HOP) {
      const r = this.rng();
      if (p && !this.inPatch(p, this.x[i], this.y[i])) {
        this.angle[i] = facing(p.x - this.x[i], p.y - this.y[i]);
        this.hop(i, 2);
      } else if (r < 0.035 && p) {
        this.flit(i, f, p);
      } else if (r < 0.45) {
        this.act[i] = A_PECK;
        this.actT[i] = range(this.rng, 0.15, 0.35);
      } else if (r < 0.7) {
        this.angle[i] += range(this.rng, -0.6, 0.6);
        this.hop(i, range(this.rng, 1, 2.2));
      } else if (r < 0.85) {
        this.angle[i] += range(this.rng, -1.6, 1.6);
        this.act[i] = A_IDLE;
        this.actT[i] = range(this.rng, 0.3, 1);
      } else {
        this.act[i] = A_IDLE;
        this.actT[i] = range(this.rng, 0.5, 2);
      }
    }
    this.frame[i] = this.act[i] === A_PECK ? PECK : STAND;
  }

  /** A short solo flight to another spot on the patch. */
  private flit(i: number, f: Flock, p: ForagePatch): void {
    for (let k = 0; k < 6; k++) {
      const a = this.rng() * TAU, d = range(this.rng, 6, 18);
      const x = this.x[i] + Math.cos(a) * d, y = this.y[i] + Math.sin(a) * d;
      if (!this.inPatch(p, x, y)) continue;
      this.flutter(f, i);
      this.vz[i] = range(this.rng, 5, 8);
      this.vx[i] = this.vy[i] = 0;
      this.land(i, x, y, 0, TO_GROUND);
      if (this.rng() < 0.3) this.caw(f, 1, 0.2);
      return;
    }
  }

  private hop(i: number, dist: number): void {
    // Models face +y: angle a turns the front to (-sin a, cos a).
    this.hopX[i] = -Math.sin(this.angle[i]) * dist;
    this.hopY[i] = Math.cos(this.angle[i]) * dist;
    this.hopT[i] = 0;
    this.act[i] = A_HOP;
    this.actT[i] = range(this.rng, 0.1, 0.5);
  }

  private fly(i: number, f: Flock, dt: number): void {
    // Stragglers catch up with what the flock is doing.
    if (!this.pend[i] && this.panic[i] <= 0) {
      if (f.mode === F_FORAGE) return this.landOnGround(i, f);
      if (f.mode === F_ROOST) return this.landOnPerch(i, f);
      if (f.mode === F_COVER && f.arrived) return this.landInCover(i, f);
      if (f.mode === F_SETTLE) this.schedule(i, P_SETTLE, this.rng() * 2);
    }
    const t = this.time, ph = this.ph[i];
    const wind = Math.hypot(this.wind[0], this.wind[1]);
    const spread = f.mode === F_WHEEL || f.mode === F_SETTLE ? 1.3 : 1;
    const rag = 2.5 * (1 + 0.6 * wind);
    let dvx: number, dvy: number, dvz: number;
    if (this.panic[i] > 0) {
      this.panic[i] -= dt;
      dvx = this.vx[i];
      dvy = this.vy[i];
      dvz = this.vz[i] * 0.9;
    } else {
      const gx = f.lx + this.ox[i] * spread + Math.sin(t * 0.6 + ph) * rag;
      const gy = f.ly + this.oy[i] * spread + Math.cos(t * 0.5 + ph * 1.3) * rag;
      const gz = f.lz + this.oz[i] + Math.sin(t * 0.9 + ph * 2) * 1.2;
      const lvx = Math.cos(f.heading) * f.speed, lvy = Math.sin(f.heading) * f.speed;
      dvx = lvx + (gx - this.x[i]) * 0.9;
      dvy = lvy + (gy - this.y[i]) * 0.9;
      dvz = f.lvz + (gz - this.z[i]) * 0.9;
    }
    // Separation from close flockmates.
    for (let j = f.first; j < f.first + f.n; j++) {
      if (j === i || this.mode[j] !== FLY) continue;
      const ex = this.x[i] - this.x[j], ey = this.y[i] - this.y[j], ez = this.z[i] - this.z[j];
      const d2 = ex * ex + ey * ey + ez * ez;
      if (d2 > 9 || d2 < 1e-4) continue;
      const k = 8 / d2;
      dvx += ex * k;
      dvy += ey * k;
      dvz += ez * k * 0.5;
    }
    this.integrate(i, dvx, dvy, clamp(dvz, -12, 14), 2.2, 35, 52, dt, 1);
    this.z[i] = Math.max(this.z[i], 2);
    this.animateFlight(i, dt, false);
  }

  private landing(i: number, f: Flock, dt: number): void {
    this.landT[i] += dt;
    let tx = this.tx[i], ty = this.ty[i];
    const tz = this.tz[i];
    if (this.landTo[i] === TO_PERCH) [tx, ty] = this.perchNow(f.perches[this.perch[i]]);
    const dx = tx - this.x[i], dy = ty - this.y[i], dz = tz - this.z[i];
    const dh = Math.hypot(dx, dy);
    if ((dh < 0.5 && Math.abs(dz) < 0.5) || this.landT[i] > 14) {
      this.x[i] = tx;
      this.y[i] = ty;
      this.z[i] = tz;
      this.vx[i] = this.vy[i] = this.vz[i] = 0;
      this.mode[i] = this.landTo[i] === TO_GROUND ? GROUND : this.landTo[i] === TO_PERCH ? PERCH : HIDDEN;
      this.act[i] = A_IDLE;
      this.actT[i] = range(this.rng, 0.2, 1);
      this.frame[i] = STAND;
      return;
    }
    // Come in from above: aim over the target, lower the closer we get.
    const sp = Math.min(40, dh * 1.6 + 1.5);
    const aimZ = tz + Math.min(dh * 0.4, 12);
    const dvx = dh > 1e-3 ? (dx / dh) * sp : 0, dvy = dh > 1e-3 ? (dy / dh) * sp : 0;
    const dvz = clamp((aimZ - this.z[i]) * 2, -12, 10);
    const near = clamp(dh / 20, 0, 1);
    this.integrate(i, dvx, dvy, dvz, 4, 60, 45, dt, near);
    // Never sink below the target until there.
    if (this.z[i] < tz && dh > 0.5) this.z[i] = tz;
    this.animateFlight(i, dt, true);
  }

  /** Accelerate towards a desired velocity (gain, max accel), clamp speed, move (plus wind × windK). */
  private integrate(i: number, dvx: number, dvy: number, dvz: number, gain: number, maxAcc: number, maxSpeed: number, dt: number, windK: number): void {
    let ax = (dvx - this.vx[i]) * gain, ay = (dvy - this.vy[i]) * gain, az = (dvz - this.vz[i]) * gain;
    const a = Math.hypot(ax, ay, az);
    if (a > maxAcc) {
      const k = maxAcc / a;
      ax *= k; ay *= k; az *= k;
    }
    this.vx[i] += ax * dt;
    this.vy[i] += ay * dt;
    this.vz[i] += az * dt;
    const s = Math.hypot(this.vx[i], this.vy[i]);
    if (s > maxSpeed) {
      this.vx[i] *= maxSpeed / s;
      this.vy[i] *= maxSpeed / s;
    }
    this.x[i] += (this.vx[i] + this.wind[0] * WIND_PUSH * windK) * dt;
    this.y[i] += (this.vy[i] + this.wind[1] * WIND_PUSH * windK) * dt;
    this.z[i] += this.vz[i] * dt;
    // Remember how hard we're working for the flap/glide choice.
    this.hopT[i] = a;
  }

  private animateFlight(i: number, dt: number, landing: boolean): void {
    const s = Math.hypot(this.vx[i], this.vy[i]);
    if (s > 2) this.angle[i] += wrap(facing(this.vx[i], this.vy[i]) - this.angle[i]) * Math.min(1, dt * 8);
    // Flap in bursts with glides between; always flap when climbing, slow or landing.
    const working = landing || this.vz[i] > 2 || this.hopT[i] > 14 || s < 16;
    this.glideT[i] -= dt;
    if (this.gliding[i]) {
      if (this.glideT[i] <= 0 || working) {
        this.gliding[i] = 0;
        this.glideT[i] = range(this.rng, 0.8, 2.2);
      }
    } else if (this.glideT[i] <= 0 && !working) {
      this.gliding[i] = 1;
      this.glideT[i] = range(this.rng, 0.4, 1.3);
    }
    if (this.gliding[i]) this.frame[i] = FLAP[1];
    else {
      this.flapT[i] += dt * (landing ? 5.5 : 4);
      this.frame[i] = FLAP[Math.floor(this.flapT[i] * 4) % 4];
    }
  }
}
