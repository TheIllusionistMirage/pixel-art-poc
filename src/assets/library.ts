// Builds every model variant once at startup and packs them into the atlas.

import { mulberry32 } from '../engine/rng';
import { buildAtlas, type Atlas, type AtlasEntry } from './atlas';
import * as M from './models';
import { MAT, type RGB, type VoxelModel } from './voxel';

export interface Library {
  atlas: Atlas;
  houses: string[];
  oaks: string[];
  pines: string[];
  bushes: string[];
  rocks: string[];
  stalls: string[];
  reeds: string[];
  grasses: string[];
  /** villagers[look][frame] */
  villagers: string[][];
}

export function buildLibrary(seed = 1): Library {
  const rng = mulberry32(seed);
  const entries: AtlasEntry[] = [];
  const add = (name: string, model: VoxelModel, flex = 0) => (entries.push({ name, model, flex }), name);
  const many = (prefix: string, n: number, make: (s: number) => VoxelModel, flex = 0) =>
    Array.from({ length: n }, (_, i) => add(`${prefix}${i}`, make(seed * 1000 + i * 17 + prefix.length), flex));
  /** Characters are tagged so weather effects (snow, puddles) skip them. */
  const actor = (m: VoxelModel) => (m.retag(MAT.SOLID, MAT.ACTOR), m);

  // flex: how far the top bends in the wind (world units at full wind strength).
  const houses = many('house', 14, (s) => M.house(M.randomHouseOpts(rng, s)));
  const oaks = many('oak', 5, M.oak, 1.6);
  const pines = many('pine', 4, M.pine, 1.4);
  const bushes = many('bush', 3, M.bush, 0.5);
  const rocks = many('rock', 3, M.rock);
  const reeds = many('reed', 4, (s) => M.blades(s, 'reed'), 2.2);
  const grasses = many('tallgrass', 4, (s) => M.blades(s, 'grass'), 1.6);
  const cloths: RGB[] = [[158, 72, 62], [72, 98, 138], [186, 156, 84], [84, 124, 88], [122, 82, 126]];
  const stalls = cloths.map((c, i) => add(`stall${i}`, M.stall(seed + i, c)));
  add('church', M.church(seed));
  add('well', M.well(seed));
  add('barrel', M.barrel(seed));
  add('crate', M.crate(seed));
  add('haystack', M.haystack(seed));
  add('cart', M.cart(seed));
  add('fence', M.fence(seed));
  add('torch', M.torchPost(seed));
  add('campfire', M.campfire(seed));
  add('fisher', actor(M.fisher(seed)));
  add('lantern', M.lantern(seed));
  const villagers = M.VILLAGER_LOOKS.map((look, li) => ([0, 1, 2] as const).map((f) => add(`villager${li}_${f}`, actor(M.villager(look, f)))));

  return { atlas: buildAtlas(entries), houses, oaks, pines, bushes, rocks, stalls, reeds, grasses, villagers };
}
