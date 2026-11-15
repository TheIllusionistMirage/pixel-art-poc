// Procedural weather sound with WebAudio: no audio files, only filtered noise.
//
//   rain hiss   white noise → high-pass → low-pass        (∝ rain)
//   rain body   brown noise → low-pass                     (∝ rain, heavier rain only)
//   wind        brown noise → band-pass swept by the gusts (∝ wind)
//   thunder     a high-passed crack when close, then a long low rumble with a
//               ragged envelope, scheduled after the flash by distance.
//
// Browsers only allow audio after a user gesture, so nothing is created until
// `unlock()` is called from one.

import type { Strike } from '../world/weather';

export interface AudioWeather {
  rain: number;
  snow: number;
  wind: number;
  gust: number;
}

function noiseBuffer(ctx: AudioContext, seconds: number, brown: boolean): AudioBuffer {
  const buf = ctx.createBuffer(1, Math.floor(ctx.sampleRate * seconds), ctx.sampleRate);
  const d = buf.getChannelData(0);
  let last = 0;
  for (let i = 0; i < d.length; i++) {
    const w = Math.random() * 2 - 1;
    if (brown) {
      last = (last + 0.02 * w) / 1.02;
      d[i] = last * 3.5;
    } else d[i] = w;
  }
  return buf;
}

// iOS (every browser there is WebKit) plays Web Audio in the "ambient" audio session, which the
// ring/silent switch mutes — so on a phone set to silent there is no sound at all. Two ways out:
// the Audio Session API (iOS 17+, but not in every browser or on plain-http pages), and the
// long-standing trick of looping a silent <audio> element, which moves the page to the "playback"
// session that ignores the switch. Do both; each needs a user gesture to take effect.
const IOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
let silentLoop: HTMLAudioElement | null = null;

/** A short, silent 8 kHz 8-bit mono WAV. */
function silentWav(): string {
  const n = 4000, b = new DataView(new ArrayBuffer(44 + n));
  const str = (o: number, s: string) => [...s].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); b.setUint32(4, 36 + n, true); str(8, 'WAVEfmt ');
  b.setUint32(16, 16, true); b.setUint16(20, 1, true); b.setUint16(22, 1, true);
  b.setUint32(24, 8000, true); b.setUint32(28, 8000, true); b.setUint16(32, 1, true); b.setUint16(34, 8, true);
  str(36, 'data'); b.setUint32(40, n, true);
  for (let i = 0; i < n; i++) b.setUint8(44 + i, 128); // 8-bit PCM silence is the midpoint
  return URL.createObjectURL(new Blob([b.buffer], { type: 'audio/wav' }));
}

function playAsMedia(): void {
  const nav = navigator as Navigator & { audioSession?: { type: string } };
  try {
    if (nav.audioSession && nav.audioSession.type !== 'playback') nav.audioSession.type = 'playback';
  } catch { /* not allowed here; the <audio> fallback covers it */ }
  if (!silentLoop) {
    silentLoop = document.createElement('audio');
    silentLoop.src = silentWav();
    silentLoop.loop = true;
    silentLoop.setAttribute('playsinline', '');
    silentLoop.setAttribute('x-webkit-airplay', 'deny');
    // Don't hold the media session while the page is in the background.
    document.addEventListener('visibilitychange', () => document.hidden && silentLoop!.pause());
  }
  if (silentLoop.paused) silentLoop.play().catch(() => {});
}

export class WeatherAudio {
  private ctx: AudioContext | null = null;
  private master!: GainNode;
  private rainGain!: GainNode;
  private bodyGain!: GainNode;
  private windGain!: GainNode;
  private windFilter!: BiquadFilterNode;
  private white!: AudioBuffer;
  private brown!: AudioBuffer;
  private muted = false;

  get isMuted(): boolean {
    return this.muted;
  }

  /** The shared AudioContext, once unlocked (other sound sources play through it). */
  get context(): AudioContext | null {
    return this.ctx;
  }

  /** Master bus: follows the mute button and feeds the compressor. */
  get bus(): AudioNode {
    return this.master;
  }

  /** A 2 s loop of white noise. */
  get noise(): AudioBuffer {
    return this.white;
  }

  /** Call from a user gesture. Safe to call repeatedly. */
  unlock(): void {
    if (IOS) playAsMedia();
    if (this.ctx) {
      // iOS also uses a non-standard 'interrupted' state (calls, app switches), not just 'suspended'.
      if (this.ctx.state !== 'running') this.ctx.resume().catch(() => {});
      return;
    }
    const ctx = new AudioContext();
    this.ctx = ctx;
    this.white = noiseBuffer(ctx, 2, false);
    this.brown = noiseBuffer(ctx, 4, true);

    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.ratio.value = 4;
    comp.connect(ctx.destination);
    this.master = ctx.createGain();
    this.master.gain.value = this.muted ? 0 : 0.9;
    this.master.connect(comp);

    const loop = (buf: AudioBuffer, ...chain: AudioNode[]): void => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      src.loopStart = 0;
      src.start(0, Math.random() * buf.duration);
      let n: AudioNode = src;
      for (const c of chain) n = n.connect(c);
      n.connect(this.master);
    };
    const filter = (type: BiquadFilterType, freq: number, q = 0.7): BiquadFilterNode => {
      const f = ctx.createBiquadFilter();
      f.type = type;
      f.frequency.value = freq;
      f.Q.value = q;
      return f;
    };
    const gain = (): GainNode => {
      const g = ctx.createGain();
      g.gain.value = 0;
      return g;
    };

    this.rainGain = gain();
    loop(this.white, filter('highpass', 900), filter('lowpass', 7000), this.rainGain);
    this.bodyGain = gain();
    loop(this.brown, filter('lowpass', 500), this.bodyGain);
    this.windGain = gain();
    this.windFilter = filter('bandpass', 300, 1.6);
    loop(this.brown, this.windFilter, this.windGain);
  }

  setMuted(m: boolean): void {
    this.muted = m;
    if (this.ctx) this.master.gain.setTargetAtTime(m ? 0 : 0.9, this.ctx.currentTime, 0.15);
  }

  update(w: AudioWeather): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const t = ctx.currentTime;
    this.rainGain.gain.setTargetAtTime(0.22 * Math.pow(w.rain, 0.7), t, 0.4);
    this.bodyGain.gain.setTargetAtTime(0.35 * Math.pow(w.rain, 1.2), t, 0.4);
    const g = Math.max(0, w.gust);
    // Wind carrying snow howls a little louder.
    this.windGain.gain.setTargetAtTime(0.5 * Math.pow(w.wind, 1.4) * g * (1 + 0.3 * w.snow), t, 0.25);
    this.windFilter.frequency.setTargetAtTime(220 + 520 * w.wind * g, t, 0.3);
  }

  /** Schedule the thunder for a lightning strike that just flashed. */
  thunder(s: Strike): void {
    const ctx = this.ctx;
    if (!ctx || this.muted) return;
    const delay = Math.min(3, 0.15 + s.distance / 220);
    const t0 = ctx.currentTime + delay;
    const loud = s.strength;

    if (s.distance < 200) {
      // The crack: a short burst of bright noise.
      const src = ctx.createBufferSource();
      src.buffer = this.white;
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 1200;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, t0);
      g.gain.linearRampToValueAtTime(0.7 * loud, t0 + 0.01);
      g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.35);
      src.connect(hp).connect(g).connect(this.master);
      src.start(t0, Math.random());
      src.stop(t0 + 0.4);
    }

    // The rumble: low noise with a ragged, rolling envelope.
    const dur = 3.5 + Math.random() * 2.5;
    const src = ctx.createBufferSource();
    src.buffer = this.brown;
    src.loop = true;
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 380 - 200 * Math.min(1, s.distance / 600);
    lp.Q.value = 0.9;
    const g = ctx.createGain();
    const n = 64;
    const curve = new Float32Array(n);
    let roll = 1;
    for (let i = 0; i < n; i++) {
      const x = i / (n - 1);
      if (Math.random() < 0.12) roll = 0.5 + Math.random() * 0.8;
      const attack = Math.min(1, x * 14);
      curve[i] = Math.max(0, attack * Math.pow(1 - x, 1.6) * roll * (0.75 + 0.25 * Math.random()) * 1.4 * loud);
    }
    curve[n - 1] = 0;
    g.gain.setValueAtTime(0, t0);
    g.gain.setValueCurveAtTime(curve, t0, dur);
    src.connect(lp).connect(g).connect(this.master);
    src.start(t0, Math.random() * 2);
    src.stop(t0 + dur + 0.05);
  }
}
