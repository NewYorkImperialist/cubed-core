export const CALIBRATION_COLORS = [
  "white",
  "green",
  "red",
  "blue",
  "orange",
  "yellow",
] as const;

export type CalibrationColor = (typeof CALIBRATION_COLORS)[number];

export interface ElementBounds {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface NativeVideoPoint {
  x: number;
  y: number;
}

export interface NativeCropBox {
  x: number;
  y: number;
  side: number;
}

export interface ContainedVideoRect {
  left: number;
  top: number;
  width: number;
  height: number;
  scale: number;
}

function positiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

export function containedVideoRect(
  bounds: ElementBounds,
  videoWidth: number,
  videoHeight: number,
): ContainedVideoRect | null {
  if (
    !positiveFinite(bounds.width) ||
    !positiveFinite(bounds.height) ||
    !positiveFinite(videoWidth) ||
    !positiveFinite(videoHeight)
  ) {
    return null;
  }
  const scale = Math.min(
    bounds.width / videoWidth,
    bounds.height / videoHeight,
  );
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  return {
    left: bounds.left + (bounds.width - width) / 2,
    top: bounds.top + (bounds.height - height) / 2,
    width,
    height,
    scale,
  };
}

/**
 * Map a pointer through CSS `object-fit: contain`. Points in the letterbox
 * return null instead of silently sampling the wrong native pixel.
 */
export function clientPointToNativeVideo(
  clientX: number,
  clientY: number,
  bounds: ElementBounds,
  videoWidth: number,
  videoHeight: number,
): NativeVideoPoint | null {
  const content = containedVideoRect(bounds, videoWidth, videoHeight);
  if (!content) return null;
  const localX = clientX - content.left;
  const localY = clientY - content.top;
  if (
    localX < 0 ||
    localY < 0 ||
    localX > content.width ||
    localY > content.height
  ) {
    return null;
  }
  return {
    x: Math.min(videoWidth - 1, Math.max(0, Math.floor(localX / content.scale))),
    y: Math.min(videoHeight - 1, Math.max(0, Math.floor(localY / content.scale))),
  };
}

/**
 * Use a small native source region: about 2.5% of the short edge, bounded
 * tightly enough to stay on one sticker.
 */
export function nativeCropBox(
  point: NativeVideoPoint,
  videoWidth: number,
  videoHeight: number,
): NativeCropBox {
  if (!positiveFinite(videoWidth) || !positiveFinite(videoHeight)) {
    throw new RangeError("Video dimensions must be positive.");
  }
  const maximumSide = Math.max(1, Math.floor(Math.min(videoWidth, videoHeight)));
  const desiredSide = Math.max(
    16,
    Math.min(96, Math.round(Math.min(videoWidth, videoHeight) * 0.025)),
  );
  const side = Math.min(maximumSide, desiredSide);
  const centerX = Math.min(videoWidth - 1, Math.max(0, point.x));
  const centerY = Math.min(videoHeight - 1, Math.max(0, point.y));
  return {
    x: Math.min(
      Math.max(0, Math.round(centerX - side / 2)),
      Math.max(0, Math.floor(videoWidth - side)),
    ),
    y: Math.min(
      Math.max(0, Math.round(centerY - side / 2)),
      Math.max(0, Math.floor(videoHeight - side)),
    ),
    side,
  };
}

export function moveNativePoint(
  point: NativeVideoPoint | null,
  direction: "left" | "right" | "up" | "down",
  videoWidth: number,
  videoHeight: number,
  coarse = false,
): NativeVideoPoint {
  if (!positiveFinite(videoWidth) || !positiveFinite(videoHeight)) {
    return { x: 0, y: 0 };
  }
  const current = point ?? {
    x: Math.floor(videoWidth / 2),
    y: Math.floor(videoHeight / 2),
  };
  const step = Math.max(
    1,
    Math.round(Math.min(videoWidth, videoHeight) * (coarse ? 0.02 : 0.005)),
  );
  const deltaX =
    direction === "left" ? -step : direction === "right" ? step : 0;
  const deltaY =
    direction === "up" ? -step : direction === "down" ? step : 0;
  return {
    x: Math.min(videoWidth - 1, Math.max(0, current.x + deltaX)),
    y: Math.min(videoHeight - 1, Math.max(0, current.y + deltaY)),
  };
}
