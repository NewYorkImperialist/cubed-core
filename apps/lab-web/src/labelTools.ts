import type {
  AlignmentLabel,
  AnnotationFace,
  AnnotationFrame,
  AnnotationSource,
  FrameAnnotationsDocument,
  Point,
} from "./localContracts";
import { bytesToHex } from "./hex.ts";

export type FrameDraft = Omit<AnnotationFrame, "frame_index" | "time_seconds">;
export type FrameDrafts = Record<string, FrameDraft>;
export type CameraMatrix3x3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];

export interface LabelAssistPin {
  xy: Point;
  vertex: number;
}

export interface LabelAssistRequest {
  labeled_corners: Point[];
  width: number;
  height: number;
  pins: LabelAssistPin[];
  K?: CameraMatrix3x3;
  shrink?: number;
}

/** Operator-selectable PnP solve setup.
 *
 * `camera` "auto" uses the capture receipt's intrinsics when the source has
 * one and otherwise omits K so the server applies its image-centered focal
 * prior; "generic" forces that prior even when a receipt exists; a number is
 * a custom focal length in pixels for footage from any other camera.
 * `shrink` scales the solved face corners toward the face centre (server
 * default 1.0, accepted range 0.1 to 1.5), for cubes where the operator
 * clicks sticker corners inset from the mechanical face outline.
 */
export interface LabelAssistSolveOptions {
  camera?: "auto" | "generic" | number;
  shrink?: number;
}

export interface NativeCameraGeometry {
  intrinsics: unknown;
  encodedWidth: unknown;
  encodedHeight: unknown;
  rotationDegrees: unknown;
}

export const EMPTY_LABEL_FRAME: FrameDraft = {
  aligned_label: null,
  faces: [],
  polygons: [],
  wireframe: null,
};

const LABEL_CAMERA_MATRIX_MAX_ABS = 1_000_000;
const RECEIPT_CAMERA_MATRIX_MAX_ABS = 10_000_000;
const MAX_CAMERA_DIMENSION = 100_000;
const CAMERA_INTRINSICS_KEYS = new Set(["matrix", "ref_w", "ref_h"]);

function isBoundedCameraNumber(value: unknown, maximumAbsoluteValue: number): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    Math.abs(value) <= maximumAbsoluteValue
  );
}

function isCameraDimension(value: unknown): value is number {
  return (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 1 &&
    value <= MAX_CAMERA_DIMENSION
  );
}

export function cameraMatrixForLabelFrame(
  geometry: NativeCameraGeometry | null | undefined,
  dimensions: { width: number; height: number },
): CameraMatrix3x3 | null {
  if (
    geometry == null ||
    !isCameraDimension(dimensions.width) ||
    !isCameraDimension(dimensions.height) ||
    !isCameraDimension(geometry.encodedWidth) ||
    !isCameraDimension(geometry.encodedHeight) ||
    !Number.isInteger(geometry.rotationDegrees) ||
    ![0, 90, 180, 270].includes(geometry.rotationDegrees as number) ||
    geometry.intrinsics === null ||
    typeof geometry.intrinsics !== "object" ||
    Array.isArray(geometry.intrinsics)
  ) {
    return null;
  }

  const intrinsics = geometry.intrinsics as Record<string, unknown>;
  if (
    Object.keys(intrinsics).length !== CAMERA_INTRINSICS_KEYS.size ||
    Object.keys(intrinsics).some((key) => !CAMERA_INTRINSICS_KEYS.has(key)) ||
    !isCameraDimension(intrinsics.ref_w) ||
    !isCameraDimension(intrinsics.ref_h)
  ) {
    return null;
  }
  const referenceWidth = intrinsics.ref_w as number;
  const referenceHeight = intrinsics.ref_h as number;

  if (
    !Array.isArray(intrinsics.matrix) ||
    intrinsics.matrix.length !== 3 ||
    intrinsics.matrix.some(
      (row) =>
        !Array.isArray(row) ||
        row.length !== 3 ||
        row.some(
          (entry) => !isBoundedCameraNumber(entry, RECEIPT_CAMERA_MATRIX_MAX_ABS),
        ),
    )
  ) {
    return null;
  }

  // Direct composition of Cubed's two existing camera-geometry helpers:
  // 1. camera_intrinsics.from_native scales preview-reference K to the raw
  //    encoded movie axis and keeps only the pinhole terms.
  // 2. cube_pose.camera_matrix_for_rotated_tracker applies the declared
  //    clockwise display rotation, then scales to the Label image.
  const matrix = intrinsics.matrix as CameraMatrix3x3;
  const nativeWidth = geometry.encodedWidth as number;
  const nativeHeight = geometry.encodedHeight as number;
  const nativeScaleX = nativeWidth / referenceWidth;
  const nativeScaleY = nativeHeight / referenceHeight;
  const fx = matrix[0][0] * nativeScaleX;
  const fy = matrix[1][1] * nativeScaleY;
  const cx = matrix[0][2] * nativeScaleX;
  const cy = matrix[1][2] * nativeScaleY;
  if (
    ![fx, fy, cx, cy].every((value) =>
      isBoundedCameraNumber(value, RECEIPT_CAMERA_MATRIX_MAX_ABS),
    ) ||
    fx <= 0 ||
    fy <= 0 ||
    cx < 0 ||
    cx > nativeWidth - 1 ||
    cy < 0 ||
    cy > nativeHeight - 1
  ) {
    return null;
  }

  const rotation = geometry.rotationDegrees as number;
  let rotatedWidth = nativeWidth;
  let rotatedHeight = nativeHeight;
  let rotatedFx = fx;
  let rotatedFy = fy;
  let rotatedCx = cx;
  let rotatedCy = cy;
  if (rotation === 90) {
    rotatedWidth = nativeHeight;
    rotatedHeight = nativeWidth;
    rotatedFx = fy;
    rotatedFy = fx;
    rotatedCx = nativeHeight - 1 - cy;
    rotatedCy = cx;
  } else if (rotation === 180) {
    rotatedCx = nativeWidth - 1 - cx;
    rotatedCy = nativeHeight - 1 - cy;
  } else if (rotation === 270) {
    rotatedWidth = nativeHeight;
    rotatedHeight = nativeWidth;
    rotatedFx = fy;
    rotatedFy = fx;
    rotatedCx = cy;
    rotatedCy = nativeWidth - 1 - cx;
  }

  const displayScaleX = dimensions.width / rotatedWidth;
  const displayScaleY = dimensions.height / rotatedHeight;
  const transformed: CameraMatrix3x3 = [
    [rotatedFx * displayScaleX, 0, rotatedCx * displayScaleX],
    [0, rotatedFy * displayScaleY, rotatedCy * displayScaleY],
    [0, 0, 1],
  ];
  if (
    transformed.some((row) =>
      row.some((entry) => !isBoundedCameraNumber(entry, LABEL_CAMERA_MATRIX_MAX_ABS)),
    ) ||
    transformed[0][0] <= 0 ||
    transformed[1][1] <= 0 ||
    transformed[0][2] < 0 ||
    transformed[0][2] > dimensions.width - 1 ||
    transformed[1][2] < 0 ||
    transformed[1][2] > dimensions.height - 1
  ) {
    return null;
  }
  return transformed;
}

export function buildLabelAssistRequest(
  labeledCorners: Point[],
  dimensions: { width: number; height: number },
  pins: LabelAssistPin[],
  cameraGeometry: NativeCameraGeometry | null | undefined,
  options?: LabelAssistSolveOptions,
): LabelAssistRequest {
  const camera = options?.camera ?? "auto";
  let cameraMatrix: CameraMatrix3x3 | null = null;
  if (camera === "auto") {
    cameraMatrix = cameraMatrixForLabelFrame(cameraGeometry, dimensions);
  } else if (typeof camera === "number" && Number.isFinite(camera) && camera > 0) {
    cameraMatrix = [
      [camera, 0, dimensions.width / 2],
      [0, camera, dimensions.height / 2],
      [0, 0, 1],
    ];
  }
  const shrink = options?.shrink;
  return {
    labeled_corners: labeledCorners,
    width: dimensions.width,
    height: dimensions.height,
    pins,
    ...(cameraMatrix ? { K: cameraMatrix } : {}),
    ...(shrink !== undefined && shrink !== 1 ? { shrink } : {}),
  };
}

export function cloneLabelDrafts(value: FrameDrafts): FrameDrafts {
  return JSON.parse(JSON.stringify(value)) as FrameDrafts;
}

export function frameHasAnnotation(frame: FrameDraft | undefined): boolean {
  return Boolean(
    frame &&
      (frame.faces.length > 0 ||
        frame.polygons.length > 0 ||
        frame.aligned_label !== null),
  );
}

export function buildFrameAnnotationsDocument(
  source: AnnotationSource,
  frames: FrameDrafts,
  dimensions: { width: number; height: number },
  fps: number,
  maxFrame: number,
): FrameAnnotationsDocument {
  return {
    schema: "cubed-core/frame-annotations",
    schema_version: 1,
    created_at: new Date().toISOString(),
    source: {
      ...source,
      fps,
      frame_count: source.frame_count ?? (maxFrame > 0 ? maxFrame + 1 : null),
    },
    image: {
      width: dimensions.width,
      height: dimensions.height,
      coordinate_space: "display-oriented-video-pixels",
    },
    frames: Object.entries(frames)
      .filter(([, frame]) => frameHasAnnotation(frame))
      .map(([key, frame]) => ({
        frame_index: Number(key),
        time_seconds: Number((Number(key) / fps).toFixed(9)),
        aligned_label: frame.aligned_label,
        faces: frame.faces,
        polygons: frame.polygons,
        wireframe: frame.wireframe,
      }))
      .sort((left, right) => left.frame_index - right.frame_index),
  };
}

export function draftsFromDocument(document: FrameAnnotationsDocument): FrameDrafts {
  const frames: FrameDrafts = {};
  for (const frame of document.frames) {
    frames[String(frame.frame_index)] = {
      aligned_label: frame.aligned_label,
      faces: frame.faces,
      polygons: frame.polygons,
      wireframe: frame.wireframe,
    };
  }
  return frames;
}

export function annotationSourceMismatchFields(
  imported: AnnotationSource,
  active: AnnotationSource,
): string[] {
  const mismatches: string[] = [];
  if (imported.filename !== active.filename) mismatches.push("filename");
  if (imported.sha256 !== active.sha256) mismatches.push("SHA-256 receipt");
  return mismatches;
}

/** Generate an annotation id without requiring a secure context.
 *
 * `crypto.randomUUID` exists only on HTTPS and localhost. Serving the workbench
 * on a plain-HTTP LAN address so a phone can reach it makes it undefined, and
 * every call site that minted a face id threw there.
 */
export function newAnnotationId(): string {
  const api = globalThis.crypto;
  if (api && typeof api.randomUUID === "function") {
    return api.randomUUID();
  }
  const bytes = new Uint8Array(16);
  if (api && typeof api.getRandomValues === "function") {
    api.getRandomValues(bytes);
  } else {
    for (let index = 0; index < bytes.length; index += 1) {
      bytes[index] = Math.floor(Math.random() * 256);
    }
  }
  // RFC 4122 version 4 layout, so ids stay indistinguishable from the native ones.
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = bytesToHex(bytes);
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join("-");
}

export function orderClockwise(points: Point[]): Point[] {
  if (points.length < 3) return points.map((point) => [...point] as Point);
  const center: Point = [
    points.reduce((sum, point) => sum + point[0], 0) / points.length,
    points.reduce((sum, point) => sum + point[1], 0) / points.length,
  ];
  const ordered = points
    .map((point) => [...point] as Point)
    .sort(
      (left, right) =>
        Math.atan2(left[1] - center[1], left[0] - center[0]) -
        Math.atan2(right[1] - center[1], right[0] - center[0]),
    );
  let start = 0;
  for (let index = 1; index < ordered.length; index += 1) {
    if (ordered[index][1] < ordered[start][1]) start = index;
  }
  return [...ordered.slice(start), ...ordered.slice(0, start)];
}

/** Build a face from four points in the order they are given.
 *
 * Hand-placed corners carry the operator's intent in their click order, so
 * re-sorting them by angle renumbers the keypoints underneath the operator and
 * reads as the pose running the wrong way around the face.
 */
export function faceFromOrderedPoints(
  points: Point[],
  id: string,
  origin: AnnotationFace["origin"] = "manual",
): AnnotationFace {
  if (points.length !== 4) {
    throw new Error("A face requires exactly four points.");
  }
  return {
    id,
    corners: points.map((point) => [...point] as Point) as [
      Point,
      Point,
      Point,
      Point,
    ],
    visible: [true, true, true, true],
    vertices: [null, null, null, null],
    pinned: [false, false, false, false],
    origin,
  };
}

/** Build a face from four unordered points, such as a converted polygon. */
export function faceFromFourPoints(
  points: Point[],
  id: string,
  origin: AnnotationFace["origin"] = "manual",
): AnnotationFace {
  if (points.length !== 4) {
    throw new Error("A face requires exactly four points.");
  }
  return faceFromOrderedPoints(orderClockwise(points), id, origin);
}

/**
 * Append the nearest earlier frame's raw label geometry, matching Cubed's
 * AnnotateView repeat behavior. Public-only PnP state is deliberately rebuilt
 * as manual geometry instead of carrying a stale solved pose across frames.
 */
export function appendRepeatedGeometry(
  current: FrameDraft,
  source: FrameDraft,
  createId: () => string,
): FrameDraft {
  const repeatedPolygons = source.polygons.map((polygon) => ({
    id: createId(),
    points: polygon.points.map(([x, y]) => [x, y] as Point),
  }));
  const repeatedFaces: AnnotationFace[] = source.faces.map((face) => ({
    id: createId(),
    corners: face.corners.map(
      ([x, y]) => [x, y] as Point,
    ) as AnnotationFace["corners"],
    visible: [...face.visible] as AnnotationFace["visible"],
    vertices: [null, null, null, null],
    pinned: [false, false, false, false],
    origin: "manual",
  }));
  return {
    ...current,
    faces: [...current.faces, ...repeatedFaces],
    polygons: [...current.polygons, ...repeatedPolygons],
  };
}

export function nextFilteredFrame(
  current: number,
  direction: -1 | 1,
  step: number,
  maxFrame: number,
  frames: FrameDrafts,
  filters: {
    annotated: boolean;
    aligned: boolean;
    modelAligned?: ReadonlySet<number> | null;
  },
): number {
  if (!filters.annotated && !filters.aligned && !filters.modelAligned) {
    return Math.max(0, Math.min(maxFrame, current + direction * step));
  }
  let remaining = step;
  for (
    let candidate = current + direction;
    candidate >= 0 && candidate <= maxFrame;
    candidate += direction
  ) {
    const frame = frames[String(candidate)];
    if (filters.annotated && !frameHasAnnotation(frame)) continue;
    if (filters.aligned && frame?.aligned_label !== "aligned") continue;
    if (filters.modelAligned && !filters.modelAligned.has(candidate)) continue;
    remaining -= 1;
    if (remaining === 0) return candidate;
  }
  return current;
}

export function nextUnclassifiedFrame(
  current: number,
  maxFrame: number,
  frames: FrameDrafts,
): number {
  if (maxFrame <= 0) return current;
  for (let offset = 1; offset <= maxFrame; offset += 1) {
    const candidate = (current + offset) % (maxFrame + 1);
    if ((frames[String(candidate)]?.aligned_label ?? null) === null) return candidate;
  }
  return current;
}

export function toggledAlignment(
  current: AlignmentLabel,
  requested: Exclude<AlignmentLabel, null>,
): AlignmentLabel {
  return current === requested ? null : requested;
}

export function translatePolygonWithinImage(
  original: Point[],
  start: Point,
  current: Point,
  dimensions: { width: number; height: number },
): Point[] {
  const minimumX = Math.min(...original.map((point) => point[0]));
  const maximumX = Math.max(...original.map((point) => point[0]));
  const minimumY = Math.min(...original.map((point) => point[1]));
  const maximumY = Math.max(...original.map((point) => point[1]));
  const deltaX = Math.max(
    -minimumX,
    Math.min(dimensions.width - maximumX, current[0] - start[0]),
  );
  const deltaY = Math.max(
    -minimumY,
    Math.min(dimensions.height - maximumY, current[1] - start[1]),
  );
  return original.map(
    (point) => [point[0] + deltaX, point[1] + deltaY] as Point,
  );
}

export function localAnnotationStorageKey(file: File): string {
  return [
    "cubed-core-label-autosave-v1",
    file.name,
    file.size,
    file.lastModified,
  ].join(":");
}
