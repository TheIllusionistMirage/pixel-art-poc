// Villager simulation, stored as struct-of-arrays (typed arrays per field).
// This layout is what lets the simulation scale to many thousands of agents
// and later move into a Web Worker (or WASM) with shared memory, since the
// renderer only needs x / y / angle / look / frame.

import { mulberry32, range, type Rng } from '../engine/rng';
import type { RoadGraph } from '../world/village';
import { facing } from '../world/village';

const WALK_CYCLE = [0, 1, 0, 2] as const;

export class Villagers {
  readonly x: Float32Array;
  readonly y: Float32Array;
  readonly angle: Float32Array;
  readonly look: Uint8Array;
  readonly frame: Uint8Array;
  private readonly from: Int32Array;
  private readonly to: Int32Array;
  private readonly t: Float32Array;
  private readonly speed: Float32Array;
  private readonly lane: Float32Array;
  private readonly walk: Float32Array;
  private readonly wait: Float32Array;
  private readonly rng: Rng;

  constructor(
    private readonly roads: RoadGraph,
    readonly count: number,
    looks: number,
    seed = 3,
  ) {
    this.rng = mulberry32(seed);
    const f32 = () => new Float32Array(count);
    this.x = f32(); this.y = f32(); this.angle = f32();
    this.t = f32(); this.speed = f32(); this.lane = f32(); this.walk = f32(); this.wait = f32();
    this.look = new Uint8Array(count);
    this.frame = new Uint8Array(count);
    this.from = new Int32Array(count);
    this.to = new Int32Array(count);

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
    this.update(0);
  }

  update(dt: number): void {
    const { nodes, adj } = this.roads;
    for (let i = 0; i < this.count; i++) {
      let A = nodes[this.from[i]], B = nodes[this.to[i]];
      if (this.wait[i] > 0) {
        this.wait[i] -= dt;
        this.frame[i] = 0;
        continue;
      }
      let len = Math.max(1e-3, Math.hypot(B.x - A.x, B.y - A.y));
      this.t[i] += (this.speed[i] * dt) / len;
      if (this.t[i] >= 1) {
        const prev = this.from[i];
        const here = this.to[i];
        let options = adj[here].length > 1 ? adj[here].filter((n) => n !== prev) : adj[here];
        // Don't circle the plaza forever: usually take an exit road when there is one.
        const exits = options.filter((n) => n >= this.roads.ringCount);
        if (exits.length && this.rng() < 0.7) options = exits;
        this.from[i] = here;
        this.to[i] = options[Math.floor(this.rng() * options.length)];
        this.t[i] = 0;
        if (this.rng() < 0.08) this.wait[i] = range(this.rng, 1.5, 5);
        A = nodes[this.from[i]];
        B = nodes[this.to[i]];
        len = Math.max(1e-3, Math.hypot(B.x - A.x, B.y - A.y));
      }
      const dx = (B.x - A.x) / len, dy = (B.y - A.y) / len;
      const t = this.t[i] * len;
      this.x[i] = A.x + dx * t - dy * this.lane[i];
      this.y[i] = A.y + dy * t + dx * this.lane[i];
      // Smoothly turn towards the walking direction.
      const target = facing(dx, dy);
      let diff = target - this.angle[i];
      diff = Math.atan2(Math.sin(diff), Math.cos(diff));
      this.angle[i] += diff * Math.min(1, dt * 10);
      this.walk[i] += this.speed[i] * dt;
      this.frame[i] = WALK_CYCLE[Math.floor(this.walk[i] / 2.5) % 4];
    }
  }
}
