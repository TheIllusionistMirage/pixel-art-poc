// Procedural rook sounds, played through the weather audio's context and master
// bus (so the sound button mutes them too). No audio files.
//
//   caw      a sawtooth with a falling pitch, roughened by a low square-wave FM,
//            shaped by two formant band-passes, plus a band of noise for the
//            rasp. Each call is ~0.3 s; a caw event is a series of 1–3.
//   flutter  band-passed noise pulsed at wingbeat rate (~15 Hz); longer and
//            denser the more birds take off together.
//
// Each sound is attenuated by its distance from the camera and panned by where
// it is across the screen.

import type { View } from '../engine/renderer';
import type { BirdSound } from '../sim/birds';
import type { WeatherAudio } from './weatherAudio';

/** Distance (world units) at which a sound is at half volume. */
const HEAR = 140;
/** Screen-x offset (world units) that pans fully to one side. */
const PAN_WIDTH = 220;
/** Most caw series playing at once. */
const MAX_CAWS = 6;

export class BirdAudio {
  /** End times of the caw series still playing. */
  private cawEnds: number[] = [];

  constructor(private readonly audio: WeatherAudio) {}

  play(sounds: readonly BirdSound[], view: View): void {
    const ctx = this.audio.context;
    if (!ctx || this.audio.isMuted || !sounds.length || ctx.state !== 'running') return;
    const now = ctx.currentTime;
    this.cawEnds = this.cawEnds.filter((t) => t > now);
    const c = Math.cos(view.angle), s = Math.sin(view.angle);
    for (const e of sounds) {
      const dx = e.x - view.x, dy = e.y - view.y;
      const d = Math.hypot(dx, dy, e.z * 0.5);
      const att = 1 / (1 + (d / HEAR) ** 2);
      if (att < 0.04) continue;
      const pan = Math.max(-1, Math.min(1, (c * dx + s * dy) / PAN_WIDTH));
      if (e.kind === 'caw') {
        if (this.cawEnds.length >= MAX_CAWS) continue;
        this.cawEnds.push(this.caws(ctx, now + Math.random() * 0.15, e.n, att * (0.7 + 0.3 * e.urgency), pan, e.urgency));
      } else this.flutter(ctx, now, e.n, att, pan);
    }
  }

  /** Route through a stereo panner into the master bus. */
  private out(ctx: AudioContext, pan: number, until: number): AudioNode {
    const p = ctx.createStereoPanner();
    p.pan.value = pan;
    p.connect(this.audio.bus);
    // Disconnect once done so finished voices can be collected.
    setTimeout(() => p.disconnect(), (until - ctx.currentTime + 0.5) * 1000);
    return p;
  }

  /** A series of n caws from one bird. Returns when it ends. */
  private caws(ctx: AudioContext, t0: number, n: number, gain: number, pan: number, urgency: number): number {
    // One voice per bird: pitch and roughness stay the same through the series.
    const f0 = 330 + Math.random() * 140 + urgency * 40;
    const rough = 55 + Math.random() * 30;
    const gap = 0.32 + Math.random() * 0.18 - urgency * 0.08;
    let t = t0, end = t0;
    const calls: [number, number][] = [];
    for (let k = 0; k < n; k++) {
      const len = 0.22 + Math.random() * 0.13;
      calls.push([t, len]);
      end = t + len;
      t += len + gap * (0.85 + Math.random() * 0.3);
    }
    const dest = this.out(ctx, pan, end);
    for (const [ts, len] of calls) this.caw(ctx, dest, ts, len, f0 * (0.97 + Math.random() * 0.06), rough, gain);
    return end;
  }

  private caw(ctx: AudioContext, dest: AudioNode, t: number, len: number, f0: number, rough: number, gain: number): void {
    const osc = ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(f0 * 1.06, t);
    osc.frequency.linearRampToValueAtTime(f0, t + len * 0.2);
    osc.frequency.exponentialRampToValueAtTime(f0 * 0.8, t + len);
    // Harsh, throaty roughness: frequency-modulate with a low square wave.
    const lfo = ctx.createOscillator();
    lfo.type = 'square';
    lfo.frequency.value = rough;
    const depth = ctx.createGain();
    depth.gain.value = f0 * 0.12;
    lfo.connect(depth).connect(osc.frequency);

    const env = ctx.createGain();
    env.gain.setValueAtTime(0, t);
    env.gain.linearRampToValueAtTime(0.28 * gain, t + 0.025);
    env.gain.setTargetAtTime(0.18 * gain, t + 0.05, 0.06);
    env.gain.linearRampToValueAtTime(0, t + len);
    // Two formants give the nasal "aah".
    for (const [f, q, g] of [[1300, 3, 1], [2700, 4, 0.6]] as const) {
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = f;
      bp.Q.value = q;
      const bg = ctx.createGain();
      bg.gain.value = g * 2.2;
      osc.connect(bp).connect(bg).connect(env);
    }
    // Rasp: a little band-passed noise under the tone.
    const noise = ctx.createBufferSource();
    noise.buffer = this.audio.noise;
    const nb = ctx.createBiquadFilter();
    nb.type = 'bandpass';
    nb.frequency.value = 2200;
    nb.Q.value = 1.2;
    const ng = ctx.createGain();
    ng.gain.value = 0.35;
    noise.connect(nb).connect(ng).connect(env);
    env.connect(dest);

    for (const src of [osc, lfo, noise]) {
      src.start(t, src === noise ? Math.random() : 0);
      src.stop(t + len + 0.02);
    }
  }

  /** Wingbeats of n birds lifting off at once. */
  private flutter(ctx: AudioContext, t: number, n: number, gain: number, pan: number): void {
    const dur = Math.min(1.6, 0.45 + 0.12 * Math.sqrt(n));
    const density = Math.min(1, n / 10);
    const rate = 14 + Math.random() * 4;
    const steps = Math.ceil(dur * 120);
    const curve = new Float32Array(steps);
    const peak = 0.22 * Math.min(2.2, Math.sqrt(n)) * gain;
    for (let k = 0; k < steps; k++) {
      const x = k / (steps - 1), tt = x * dur;
      const env = Math.min(1, x * 8) * Math.pow(1 - x, 1.3);
      // One bird is a clean pulse train; many smear into a rush.
      const pulse = Math.pow(Math.max(0, Math.sin(Math.PI * 2 * rate * tt)), 2);
      curve[k] = peak * env * (pulse * (1 - 0.6 * density) + 0.6 * density * (0.6 + 0.4 * Math.random()));
    }
    curve[steps - 1] = 0;
    const src = ctx.createBufferSource();
    src.buffer = this.audio.noise;
    const bp = ctx.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 800 + Math.random() * 250;
    bp.Q.value = 0.8;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.setValueCurveAtTime(curve, t, dur);
    src.connect(bp).connect(g).connect(this.out(ctx, pan, t + dur));
    src.start(t, Math.random());
    src.stop(t + dur + 0.05);
  }
}
