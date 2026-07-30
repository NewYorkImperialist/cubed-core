// The cube six (DESIGN.md palette) — keyed off the CSS vars so the render
// matches the rest of the identity.
const COLOR_MAP: Record<string, string> = {
  white: "var(--cube-white)",
  red: "var(--cube-red)",
  green: "var(--cube-green)",
  yellow: "var(--cube-yellow)",
  orange: "var(--cube-orange)",
  blue: "var(--cube-blue)",
};

interface CubeStateProps {
  state: Record<string, string[]> | null;
  /** Sticker keys ("<face>:<i>", e.g. "up:4") to ring red in comparisons. */
  highlight?: Set<string>;
  /** A face name ("up".."back") to outline — the face the current move turns. */
  faceHighlight?: string;
  /** Smaller stickers for compact diagnostic views. */
  compact?: boolean;
}

/**
 * 2D unfolded cube diagram.
 * Layout:
 *       [U]
 *   [L] [F] [R] [B]
 *       [D]
 */
export function CubeState({ state, highlight, faceHighlight, compact }: CubeStateProps) {
  if (!state) return null;
  const accessibleState = ["up", "left", "front", "right", "back", "down"]
    .flatMap((face) =>
      state[face]?.length ? [`${face}: ${state[face].join(", ")}`] : [],
    )
    .join(". ");

  const renderFace = (faceName: string, gridRow: number, gridCol: number) => {
    const stickers = state[faceName];
    if (!stickers) return null;

    return (
      <div
        className={"cube-face" + (faceHighlight === faceName ? " cube-face-turned" : "")}
        style={{ gridRow, gridColumn: gridCol }}
        key={faceName}
      >
        {/* A colour outside the cube six leaves the sticker unpainted. That
            blank surface is a fixed white, not a theme token, so it stays white
            against the black frame in both themes. */}
        {stickers.map((color, i) => (
          <div
            key={i}
            className={
              "cube-sticker" + (highlight?.has(`${faceName}:${i}`) ? " cube-sticker-diff" : "")
            }
            style={{ backgroundColor: COLOR_MAP[color] || "#fff" }}
            title={`${faceName}[${i}]: ${color}`}
          />
        ))}
      </div>
    );
  };

  // Diff mode dims matching stickers so the differing ones pop by contrast.
  const diffMode = !!highlight && highlight.size > 0;
  return (
    <div
      aria-label={`Cube state. ${accessibleState}.`}
      className={
        "cube-state" + (compact ? " cube-state-compact" : "") + (diffMode ? " cube-state-diffmode" : "")
      }
      role="img"
    >
      {renderFace("up", 1, 2)}
      {renderFace("left", 2, 1)}
      {renderFace("front", 2, 2)}
      {renderFace("right", 2, 3)}
      {renderFace("back", 2, 4)}
      {renderFace("down", 3, 2)}
    </div>
  );
}
