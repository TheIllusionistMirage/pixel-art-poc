// Villager simulation, stored as struct-of-arrays (typed arrays per field).
// This layout is what lets the simulation scale to many thousands of agents
// and later move into a Web Worker (or WASM) with shared memory, since the
// renderer only needs x / y / angle / look / frame / hidden.
//
// Normally villagers wander the road graph. When the weather turns, the less
// hardy ones head for the nearest house door (routed along the roads), step up
// to it and disappear inside; when it clears they come back out, a few at a time.

import { mulberry32, range, type Rng } from '../engine/rng';
import type { Door, RoadGraph } from '../world/village';
import { facing } from '../world/village';

const WALK_CYCLE = [0, 1, 0, 2] as const;

// Modes
const WALK = 0;    // wandering the roads
const ROUTE = 1;   // following the roads to the door's road edge
const ENTER = 2;   // straight from the road to the doorstep
const INSIDE = 3;  // hidden
const LEAVE = 4;   // straight from the doorstep back onto the road

/** Shortest-path tree towards one door: per node, the next node to walk to (-1: you're on the door's edge, -2: unreachable). */
interface Route {
  next: Int32Array;
  dist: Float32Array;
}

export class Villagers {
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly angle: Float32Array;
  readonly look: Uint8Array;
  readonly frame: Uint8Array;
  /** 1 while indoors: don't draw. */
  readonly hidden: Uint8Array;
  /**
   * Only the first `active` villagers are simulated and drawn (crowd density).
   * Spawn order is random, so any prefix is an even spread over the village;
   * the rest are frozen where they were and pick up again when brought back.
   */
  private activeCount: number;
  private readonly from: Int32Array;
  private readonly to: Int32Array;
  private readonly t: Float32Array;
  private readonly speed: Float32Array;
  private readonly lane: Float32Array;
  private readonly walk: Float32Array;
  private readonly wait: Float32Array;
  private readonly mode: Uint8Array;
  private readonly door: Int32Array;
  /** Straight-line walk target (ENTER / LEAVE). */
  private readonly tx: Float32Array;
  private readonly ty: Float32Array;
  /** Drawn offset from the road position, left over from corners (the lane offset
   *  turns with the road); eased away while walking so nobody teleports. */
  private readonly ox: Float32Array;
  private readonly oy: Float32Array;
  /** Goes indoors once the weather's `shelter` exceeds this. */
  private readonly hardiness: Float32Array;
  /** Seconds between the weather changing and this villager reacting to it. */
  private readonly patience: Float32Array;
  private readonly react: Float32Array;
  private readonly rng: Rng;
  private readonly routes = new Map<number, Route>();

  constructor(
    private readonly roads: RoadGraph,
    private readonly doors: Door[],
    readonly count: number,
    looks: number,
    seed = 3,
  ) {
    this.activeCount = count;
    this.rng = mulberry32(seed);
    const f32 = () => new Float32Array(count);
    this.x = f32(); this.y = f32(); this.angle = f32();
    this.t = f32(); this.speed = f32(); this.lane = f32(); this.walk = f32(); this.wait = f32();
    this.tx = f32(); this.ty = f32(); this.ox = f32(); this.oy = f32(); this.hardiness = f32(); this.patience = f32(); this.react = f32();
    this.look = new Uint8Array(count);
    this.frame = new Uint8Array(count);
    this.hidden = new Uint8Array(count);
    this.mode = new Uint8Array(count);
    this.from = new Int32Array(count);
    this.to = new Int32Array(count);
    this.door = new Int32Array(count).fill(-1);

    const { nodes, adj } = roads;
    const cx = 512, cy = 512;
    // Spawn spread along road edges, weighted by edge length, within the village.
    const edges: { a: number; b: number; len: number }[] = [];
    adj.forEach((ns, a) =>
      ns.forEach((b) => {
        if (b <= a) return;
        const mx = (nodes[a].x + nodes[b].x) / 2, my = (nodes[a].y + nodes[b].y) / 2;
        if (Math.hypot(mx - cx, my - cy) < 330) edges.push({ a, b, len: Math.hypot(nodes[b].x - nodes[a].x, nodes[b].y - nodes[a].y) });
      }),
    );
    const totalLen = edges.reduce((s, e) => s + e.len, 0);
    for (let i = 0; i < count; i++) {
      let r = this.rng() * totalLen, e = edges[0];
      for (e of edges) if ((r -= e.len) <= 0) break;
      const flip = this.rng() < 0.5;
      this.from[i] = flip ? e.b : e.a;
      this.to[i] = flip ? e.a : e.b;
      this.t[i] = this.rng();
      this.speed[i] = range(this.rng, 9, 15);
      this.lane[i] = range(this.rng, -3, 3);
      this.look[i] = Math.floor(this.rng() * looks);
      this.walk[i] = this.rng() * 10;
    }
    // Weather temperament from its own stream, so the crowd above doesn't change.
    const wr = mulberry32(seed + 99);
    for (let i = 0; i < count; i++) {
      this.hardiness[i] = wr();
      this.patience[i] = range(wr, 0.5, 6);
    }
    this.update(0, 0, 0);
  }

  get active(): number {
    return this.activeCount;
  }

  set active(n: number) {
    this.activeCount = Math.max(0, Math.min(this.count, Math.round(n)));
  }

  /** Active villagers currently indoors. */
  indoors(): number {
    let n = 0;
    for (let i = 0; i < this.activeCount; i++) n += this.hidden[i];
    return n;
  }

  private len(a: number, b: number): number {
    const A = this.roads.nodes[a], B = this.roads.nodes[b];
    return Math.max(1e-3, Math.hypot(B.x - A.x, B.y - A.y));
  }

  /** Dijkstra outwards from the door's spot on its road edge (cached per door). */
  private route(d: number): Route {
    const cached = this.routes.get(d);
    if (cached) return cached;
    const { adj } = this.roads;
    const n = adj.length;
    const dist = new Float32Array(n).fill(Infinity);
    const next = new Int32Array(n).fill(-2);
    const D = this.doors[d];
    const L = this.len(D.a, D.b);
    dist[D.a] = D.t * L;
    dist[D.b] = (1 - D.t) * L;
    next[D.a] = next[D.b] = -1;
    // Binary heap of [dist, node].
    const heap: [number, number][] = [[dist[D.a], D.a], [dist[D.b], D.b]];
    const push = (e: [number, number]) => {
      heap.push(e);
      for (let i = heap.length - 1; i > 0;) {
        const p = (i - 1) >> 1;
        if (heap[p][0] <= heap[i][0]) break;
        [heap[p], heap[i]] = [heap[i], heap[p]];
        i = p;
      }
    };
    const pop = (): [number, number] => {
      const top = heap[0], last = heap.pop()!;
      if (heap.length) {
        heap[0] = last;
        for (let i = 0; ;) {
          const l = i * 2 + 1, r = l + 1;
          let m = i;
          if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
          if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
          if (m === i) break;
          [heap[m], heap[i]] = [heap[i], heap[m]];
          i = m;
        }
      }
      return top;
    };
    if (heap[1][0] < heap[0][0]) heap.reverse();
    while (heap.length) {
      const [du, u] = pop();
      if (du > dist[u]) continue;
      for (const v of adj[u]) {
        const nd = du + this.len(u, v);
        if (nd < dist[v]) {
          dist[v] = nd;
          next[v] = u;
          push([nd, v]);
        }
      }
    }
    const r = { next, dist };
    this.routes.set(d, r);
    return r;
  }

  /** Parameter of door `d` along the edge the villager is on, or -1 if it's not on that edge. */
  private doorParam(i: number, d: number): number {
    const D = this.doors[d];
    if (this.from[i] === D.a && this.to[i] === D.b) return D.t;
    if (this.from[i] === D.b && this.to[i] === D.a) return 1 - D.t;
    return -1;
  }

  private turnAround(i: number): void {
    const f = this.from[i];
    this.from[i] = this.to[i];
    this.to[i] = f;
    this.t[i] = 1 - this.t[i];
    this.lane[i] = -this.lane[i]; // same side of the road, now seen from the other end
  }

  private goIn(i: number): boolean {
    if (!this.doors.length) return false;
    let best = -1, bd = Infinity;
    for (let d = 0; d < this.doors.length; d++) {
      const D = this.doors[d];
      const dd = (D.x - this.x[i]) ** 2 + (D.y - this.y[i]) ** 2;
      if (dd < bd) { bd = dd; best = d; }
    }
    const r = this.route(best);
    const td = this.doorParam(i, best);
    if (td >= 0) {
      if (this.t[i] > td) this.turnAround(i);
    } else {
      const L = this.len(this.from[i], this.to[i]);
      const ahead = (1 - this.t[i]) * L + r.dist[this.to[i]];
      const behind = this.t[i] * L + r.dist[this.from[i]];
      if (!Number.isFinite(Math.min(ahead, behind))) return false;
      if (behind < ahead) this.turnAround(i);
    }
    this.door[i] = best;
    this.mode[i] = ROUTE;
    this.wait[i] = 0;
    return true;
  }

  private goOut(i: number): void {
    const D = this.doors[this.door[i]];
    if (this.mode[i] === INSIDE) {
      this.x[i] = D.x;
      this.y[i] = D.y;
      this.hidden[i] = 0;
    }
    this.ox[i] = this.oy[i] = 0;
    const flip = this.rng() < 0.5;
    this.from[i] = flip ? D.b : D.a;
    this.to[i] = flip ? D.a : D.b;
    this.t[i] = flip ? 1 - D.t : D.t;
    const [px, py] = this.roadPos(i);
    this.tx[i] = px;
    this.ty[i] = py;
    this.mode[i] = LEAVE;
  }

  /** Position on the current edge at the current t, with the lane offset. */
  private roadPos(i: number): [number, number] {
    const A = this.roads.nodes[this.from[i]], B = this.roads.nodes[this.to[i]];
    const len = this.len(this.from[i], this.to[i]);
    const dx = (B.x - A.x) / len, dy = (B.y - A.y) / len;
    const t = this.t[i] * len;
    return [A.x + dx * t - dy * this.lane[i], A.y + dy * t + dx * this.lane[i]];
  }

  private face(i: number, dx: number, dy: number, dt: number): void {
    const target = facing(dx, dy);
    let diff = target - this.angle[i];
    diff = Math.atan2(Math.sin(diff), Math.cos(diff));
    // Settle exactly once close: an exponential approach never stops creeping,
    // and every tiny change re-rasterises the (nearest-sampled) rotated slices.
    if (Math.abs(diff) < 0.02) this.angle[i] = target;
    else this.angle[i] += diff * Math.min(1, dt * 10);
  }

  /** Walk straight towards (tx, ty); true on arrival. */
  private straight(i: number, dt: number, speed: number): boolean {
    const dx = this.tx[i] - this.x[i], dy = this.ty[i] - this.y[i];
    const d = Math.hypot(dx, dy);
    const step = speed * dt;
    if (d <= step || d < 1e-3) {
      this.x[i] = this.tx[i];
      this.y[i] = this.ty[i];
      return true;
    }
    this.x[i] += (dx / d) * step;
    this.y[i] += (dy / d) * step;
    this.face(i, dx, dy, dt);
    return false;
  }

  /**
   * @param shelter share of villagers that want to be indoors (weather)
   * @param hurry   0..1: how much faster those outside walk (rain)
   */
  update(dt: number, shelter: number, hurry: number): void {
    const { nodes, adj } = this.roads;
    for (let i = 0; i < this.activeCount; i++) {
      // React to the weather, after a personal delay. Coming out takes longer
      // than running in: nobody trusts the first break in the clouds.
      const mode = this.mode[i];
      const wantIn = this.hardiness[i] < shelter;
      const isIn = mode === ROUTE || mode === ENTER || mode === INSIDE;
      if (wantIn !== isIn) {
        this.react[i] += dt;
        if (this.react[i] > this.patience[i] * (wantIn ? 1 : 2)) {
          this.react[i] = 0;
          if (wantIn) {
            if (mode === LEAVE) {
              const D = this.doors[this.door[i]];
              this.tx[i] = D.x;
              this.ty[i] = D.y;
              this.mode[i] = ENTER;
            } else this.goIn(i);
          } else if (mode === ROUTE) this.mode[i] = WALK;
          else this.goOut(i);
        }
      } else this.react[i] = 0;

      const m = this.mode[i];
      if (m === INSIDE) continue;
      const speed = this.speed[i] * (m === WALK ? 1 + 0.35 * hurry : 1.35);
      this.walk[i] += speed * dt;
      this.frame[i] = WALK_CYCLE[Math.floor(this.walk[i] / 2.5) % 4];

      if (m === ENTER) {
        if (this.straight(i, dt, speed)) {
          this.mode[i] = INSIDE;
          this.hidden[i] = 1;
        }
        continue;
      }
      if (m === LEAVE) {
        if (this.straight(i, dt, speed)) this.mode[i] = WALK;
        continue;
      }

      if (this.wait[i] > 0) {
        this.wait[i] -= dt;
        this.frame[i] = 0;
        continue;
      }
      let len = this.len(this.from[i], this.to[i]);
      this.t[i] += (speed * dt) / len;
      if (m === ROUTE) {
        const td = this.doorParam(i, this.door[i]);
        if (td >= 0 && this.t[i] >= td) {
          // Reached the spot opposite the door: step off the road towards it.
          this.t[i] = td;
          this.ox[i] = this.oy[i] = 0; // x / y already include it; ENTER walks from there
          const D = this.doors[this.door[i]];
          this.tx[i] = D.x;
          this.ty[i] = D.y;
          this.mode[i] = ENTER;
          continue;
        }
      }
      if (this.t[i] >= 1) {
        const over = (this.t[i] - 1) * len;
        this.t[i] = 1;
        const [ax, ay] = this.roadPos(i);
        const prev = this.from[i];
        const here = this.to[i];
        let next = -2;
        if (m === ROUTE) {
          const D = this.doors[this.door[i]];
          const hop = this.route(this.door[i]).next[here];
          next = hop === -1 ? (here === D.a ? D.b : D.a) : hop;
          if (next < 0) this.mode[i] = WALK; // unreachable after all: carry on wandering
        }
        if (next < 0) {
          let options = adj[here].length > 1 ? adj[here].filter((n) => n !== prev) : adj[here];
          // Don't circle the plaza forever: usually take an exit road when there is one.
          const exits = options.filter((n) => n >= this.roads.ringCount);
          if (exits.length && this.rng() < 0.7) options = exits;
          next = options[Math.floor(this.rng() * options.length)];
          if (this.rng() < 0.08) this.wait[i] = range(this.rng, 1.5, 5);
        }
        this.from[i] = here;
        this.to[i] = next;
        this.t[i] = 0;
        // The lane offset is perpendicular to the road, so it swings round at the
        // corner: keep the drawn position continuous and ease the difference away.
        const [bx, by] = this.roadPos(i);
        this.ox[i] += ax - bx;
        this.oy[i] += ay - by;
        len = this.len(here, next);
        this.t[i] = Math.min(1, over / len);
      }
      const o = Math.hypot(this.ox[i], this.oy[i]);
      if (o > 0) {
        const k = Math.max(0, 1 - (0.6 * speed * dt) / o);
        this.ox[i] *= k;
        this.oy[i] *= k;
      }
      const A = nodes[this.from[i]], B = nodes[this.to[i]];
      const [px, py] = this.roadPos(i);
      this.x[i] = px + this.ox[i];
      this.y[i] = py + this.oy[i];
      this.face(i, (B.x - A.x) / len, (B.y - A.y) / len, dt);
    }
  }
}
