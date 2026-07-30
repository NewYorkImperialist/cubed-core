// Cube state helpers for the tracker's "state-match" marker: replay a canonical
// move sequence to per-prefix states so GT vs decoded can be compared as cube
// TRANSFORMS (not move-by-move). Permutations come from analysis/cubemodel
// (cubePerms.data.ts), so this matches the decode's semantics exactly.
import CUBE from "./cubePerms.data";

/** States after 0,1,2,… moves (from solved). out[k] = state after the first k moves.
 * Comparing two trajectories from solved is exact iff the prefixes are the same
 * transform (the shared scramble cancels), which is what we want for "same state". */
export function cubeTrajectory(moves: string[]): Int8Array[] {
  let s = Int8Array.from(CUBE.solved);
  const out: Int8Array[] = [s];
  for (const m of moves) {
    const p = CUBE.perms[m];
    if (p) {
      const ns = new Int8Array(54);
      for (let i = 0; i < 54; i++) ns[i] = s[p[i]];
      s = ns;
    }
    out.push(s); // unknown token -> no-op, but keep 1:1 with `moves`
  }
  return out;
}
