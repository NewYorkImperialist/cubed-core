/** Canonicalise a timed face-turn sequence.
 *
 * Combine consecutive turns of the SAME face
 * by summing quarter-turns mod 4 (U U -> U2, U U U -> U', U U' -> dropped,
 * R' R' -> R2). A move is a leading face key (letter run: U D L R F B, wide or
 * lowercase, M E S, x y z) plus optional ' or 2. Single left-to-right stack
 * fold: merge into the top when the face matches (popping on net 0), else push.
 * A pop re-exposes the new top for the next move, so cancellations that open
 * fresh adjacencies collapse too. Grouping adjacent runs instead misses that
 * case and produces a longer sequence.
 */

/** Face key (leading letter run) + quarter-turns (plain=1, 2=2, '=3), or null if not a turn. */
function parseMove(m: string): { face: string; q: number } | null {
  const face = m.match(/^[A-Za-z]+/)?.[0];
  if (!face) return null; // not a face turn (skip rather than corrupt the fold)
  return { face, q: m.endsWith("2") ? 2 : m.endsWith("'") ? 3 : 1 };
}

const fmtMove = (face: string, q: number) =>
  face + (q === 2 ? "2" : q === 3 ? "'" : "");

/** Each raw move carries a frame; the surviving canonical move keeps the frame
 * of the LAST (most-recent) raw move folded into it (on a
 * merge keep the newer frame; drop on net-zero cancel). Returns the canonical moves with a
 * 1:1 parallel `frames` array, so an emitted canonical move maps back to where it happened. */
export function simplifyMovesWithFrames(
  raw: readonly { move: string; frame: number }[],
): { moves: string[]; frames: number[] } {
  const stack: { face: string; q: number; frame: number }[] = [];
  for (const { move, frame } of raw) {
    const p = parseMove(move);
    if (!p) continue;
    const top = stack[stack.length - 1];
    if (top && top.face === p.face) {
      const q = (top.q + p.q) % 4;
      if (q === 0) stack.pop();
      else {
        top.q = q;
        top.frame = frame; // newer contributing raw move owns the frame
      }
    } else stack.push({ ...p, frame });
  }
  return {
    moves: stack.map((s) => fmtMove(s.face, s.q)),
    frames: stack.map((s) => s.frame),
  };
}
