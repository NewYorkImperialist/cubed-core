export type Point = [number, number];
export type AlignmentLabel = "aligned" | "unaligned" | null;

export interface AnnotationFace {
  id: string;
  corners: [Point, Point, Point, Point];
  visible: [boolean, boolean, boolean, boolean];
  vertices: [number | null, number | null, number | null, number | null];
  pinned: [boolean, boolean, boolean, boolean];
  origin: "manual" | "pnp-assist" | "model";
  confidence?: number;
  assist?: {
    face_name: "labeled" | "top" | "right" | "bottom" | "left" | "back";
    generated: boolean;
    intrinsics_source?: string;
  };
}

export interface AnnotationPolygon {
  id: string;
  points: Point[];
}

export interface AnnotationFrame {
  frame_index: number;
  time_seconds: number;
  aligned_label: AlignmentLabel;
  faces: AnnotationFace[];
  polygons: AnnotationPolygon[];
  wireframe: Point[] | null;
}

export interface AnnotationSource {
  kind: "workspace-capture" | "local-file";
  capture_id: string | null;
  filename: string;
  sha256: string | null;
  fps: number;
  frame_count: number | null;
}

export interface FrameAnnotationsDocument {
  schema: "cubed-core/frame-annotations";
  schema_version: 1;
  created_at: string;
  source: AnnotationSource;
  image: {
    width: number;
    height: number;
    coordinate_space: "display-oriented-video-pixels";
  };
  frames: AnnotationFrame[];
}

export interface TrackerFace {
  corners: [Point, Point, Point, Point];
  kpt_conf: [number, number, number, number];
  conf: number;
}

export type Nine<T> = [T, T, T, T, T, T, T, T, T];
export type LabVector = [number, number, number];

export interface TrackerFaceRead {
  slot: "up" | "front" | "right";
  lab: Nine<LabVector>;
  confidence: Nine<number>;
  valid_pixels: Nine<number>;
  total_pixels: Nine<number>;
  used_fallback: Nine<boolean>;
  relative_area: number;
  corners: [Point, Point, Point, Point];
}

/** Classified 3x3 read emitted by the tracker-dump producer.
 *
 * This is distinct from `TrackerFaceRead`: `reads` carries camera-space Lab
 * evidence for decoder inputs, while singular `read` carries the producer's
 * nearest-centroid diagnostic view.
 */
export interface TrackerClassifiedRead {
  slot: string;
  colors: Nine<string | null>;
  conf: Nine<number | null>;
  dist: Nine<number>;
}

export interface TrackerFrame {
  motion: number | null;
  aligned?: number;
  stk?: number | null;
  nfaces?: number | null;
  gated?: boolean;
  faces?: TrackerFace[];
  reads?: TrackerFaceRead[];
  read?: TrackerClassifiedRead[];
}

export interface TrackerSpan {
  f0: number;
  f1: number;
  event: number | null;
  top: { path: string; om: string[] | string; score: number }[];
}

export interface TrackerDump {
  schema: "cubed-core/tracker-dump-v1";
  schema_version?: 1;
  mode?: string;
  fps?: number;
  window: [number, number];
  width?: number;
  height?: number;
  gt: string[];
  emitted: string[];
  gt_canonical?: string[];
  emitted_canonical?: string[];
  anchors: [number, string, number][];
  events: number[];
  spans: TrackerSpan[];
  frames: Record<string, TrackerFrame>;
  bridge?: string[];
  reach?: boolean;
  acc?: number;
  states?: Record<string, string[]>[];
}

export class LocalContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LocalContractError";
  }
}

function objectAt(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new LocalContractError(`${path} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function stringAt(value: unknown, path: string, maxLength = 500): string {
  if (typeof value !== "string" || value.length === 0 || value.length > maxLength) {
    throw new LocalContractError(`${path} must be a non-empty string.`);
  }
  return value;
}

function boundedStringAt(value: unknown, path: string, maxLength: number): string {
  if (typeof value !== "string" || value.length > maxLength) {
    throw new LocalContractError(`${path} must be a string of at most ${maxLength} characters.`);
  }
  return value;
}

function numberAt(
  value: unknown,
  path: string,
  { min = -Infinity, max = Infinity }: { min?: number; max?: number } = {},
): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < min ||
    value > max
  ) {
    throw new LocalContractError(`${path} must be a finite number from ${min} to ${max}.`);
  }
  return value;
}

function integerAt(
  value: unknown,
  path: string,
  { min = 0, max = Number.MAX_SAFE_INTEGER }: { min?: number; max?: number } = {},
): number {
  const number = numberAt(value, path, { min, max });
  if (!Number.isInteger(number)) {
    throw new LocalContractError(`${path} must be an integer.`);
  }
  return number;
}

function arrayAt(value: unknown, path: string, maxLength: number): unknown[] {
  if (!Array.isArray(value) || value.length > maxLength) {
    throw new LocalContractError(`${path} must be an array with at most ${maxLength} items.`);
  }
  return value;
}

function pointAt(value: unknown, path: string): Point {
  const point = arrayAt(value, path, 2);
  if (point.length !== 2) {
    throw new LocalContractError(`${path} must contain x and y.`);
  }
  return [
    numberAt(point[0], `${path}[0]`, { min: 0, max: 100_000 }),
    numberAt(point[1], `${path}[1]`, { min: 0, max: 100_000 }),
  ];
}

function fourPointsAt(
  value: unknown,
  path: string,
): [Point, Point, Point, Point] {
  const points = arrayAt(value, path, 4);
  if (points.length !== 4) {
    throw new LocalContractError(`${path} must contain exactly four corners.`);
  }
  return [
    pointAt(points[0], `${path}[0]`),
    pointAt(points[1], `${path}[1]`),
    pointAt(points[2], `${path}[2]`),
    pointAt(points[3], `${path}[3]`),
  ];
}

function nineAt<T>(
  value: unknown,
  path: string,
  parse: (entry: unknown, index: number) => T,
): Nine<T> {
  const values = arrayAt(value, path, 9);
  if (values.length !== 9) {
    throw new LocalContractError(`${path} must contain exactly nine values.`);
  }
  return values.map(parse) as Nine<T>;
}

function booleanFourAt(
  value: unknown,
  path: string,
): [boolean, boolean, boolean, boolean] {
  const values = arrayAt(value, path, 4);
  if (values.length !== 4 || values.some((entry) => typeof entry !== "boolean")) {
    throw new LocalContractError(`${path} must contain four booleans.`);
  }
  return values as [boolean, boolean, boolean, boolean];
}

function vertexFourAt(
  value: unknown,
  path: string,
): [number | null, number | null, number | null, number | null] {
  const values = arrayAt(value, path, 4);
  if (values.length !== 4) {
    throw new LocalContractError(`${path} must contain four cube vertex IDs.`);
  }
  return values.map((entry, index) =>
    entry === null ? null : integerAt(entry, `${path}[${index}]`, { max: 7 }),
  ) as [number | null, number | null, number | null, number | null];
}

function stringListAt(value: unknown, path: string, maxLength = 2_000): string[] {
  return arrayAt(value, path, maxLength).map((entry, index) =>
    stringAt(entry, `${path}[${index}]`, 32),
  );
}

export async function readJsonFile(
  file: File,
  maxBytes: number,
  label: string,
): Promise<unknown> {
  if (file.size === 0) throw new LocalContractError(`${label} is empty.`);
  if (file.size > maxBytes) {
    throw new LocalContractError(
      `${label} is larger than ${Math.round(maxBytes / 1024 / 1024)} MB.`,
    );
  }
  try {
    return JSON.parse(await file.text()) as unknown;
  } catch {
    throw new LocalContractError(`${label} is not valid JSON.`);
  }
}

export function parseFrameAnnotations(value: unknown): FrameAnnotationsDocument {
  const root = objectAt(value, "annotation document");
  if (root.schema !== "cubed-core/frame-annotations" || root.schema_version !== 1) {
    throw new LocalContractError(
      "Annotations must declare cubed-core/frame-annotations schema version 1.",
    );
  }
  const sourceValue = objectAt(root.source, "source");
  if (sourceValue.kind !== "workspace-capture" && sourceValue.kind !== "local-file") {
    throw new LocalContractError("source.kind must be workspace-capture or local-file.");
  }
  const captureId =
    sourceValue.capture_id === null
      ? null
      : stringAt(sourceValue.capture_id, "source.capture_id", 64);
  if (captureId !== null && !/^[a-f0-9]{32}$/.test(captureId)) {
    throw new LocalContractError("source.capture_id must be a Cubed Core capture id.");
  }
  const sha256 =
    sourceValue.sha256 === null
      ? null
      : stringAt(sourceValue.sha256, "source.sha256", 64);
  if (sha256 !== null && !/^[a-f0-9]{64}$/.test(sha256)) {
    throw new LocalContractError("source.sha256 must be a lowercase SHA-256 digest.");
  }
  if (sourceValue.kind === "workspace-capture" && (captureId === null || sha256 === null)) {
    throw new LocalContractError(
      "A workspace-capture source must include capture_id and sha256 receipts.",
    );
  }
  if (sourceValue.kind === "local-file" && captureId !== null) {
    throw new LocalContractError("A local-file source must use a null capture_id.");
  }
  const frameCount =
    sourceValue.frame_count === null
      ? null
      : integerAt(sourceValue.frame_count, "source.frame_count", {
          min: 1,
          max: 10_000_000,
        });
  const source: AnnotationSource = {
    kind: sourceValue.kind,
    capture_id: captureId,
    filename: stringAt(sourceValue.filename, "source.filename", 500),
    sha256,
    fps: numberAt(sourceValue.fps, "source.fps", { min: 0.000_001, max: 1_000 }),
    frame_count: frameCount,
  };

  const imageValue = objectAt(root.image, "image");
  if (imageValue.coordinate_space !== "display-oriented-video-pixels") {
    throw new LocalContractError(
      "image.coordinate_space must be display-oriented-video-pixels.",
    );
  }
  const image = {
    width: integerAt(imageValue.width, "image.width", { min: 1, max: 100_000 }),
    height: integerAt(imageValue.height, "image.height", { min: 1, max: 100_000 }),
    coordinate_space: "display-oriented-video-pixels" as const,
  };

  const seenFrames = new Set<number>();
  const frames = arrayAt(root.frames, "frames", 200_000).map(
    (entry, framePosition): AnnotationFrame => {
      const frameValue = objectAt(entry, `frames[${framePosition}]`);
      const frameIndex = integerAt(
        frameValue.frame_index,
        `frames[${framePosition}].frame_index`,
        { max: 10_000_000 },
      );
      if (seenFrames.has(frameIndex)) {
        throw new LocalContractError(`Frame ${frameIndex} appears more than once.`);
      }
      seenFrames.add(frameIndex);
      const aligned = frameValue.aligned_label;
      if (aligned !== null && aligned !== "aligned" && aligned !== "unaligned") {
        throw new LocalContractError(
          `frames[${framePosition}].aligned_label must be aligned, unaligned, or null.`,
        );
      }
      const seenFaces = new Set<string>();
      const faces = arrayAt(
        frameValue.faces,
        `frames[${framePosition}].faces`,
        64,
      ).map((faceEntry, facePosition): AnnotationFace => {
        const face = objectAt(
          faceEntry,
          `frames[${framePosition}].faces[${facePosition}]`,
        );
        const id = stringAt(
          face.id,
          `frames[${framePosition}].faces[${facePosition}].id`,
          100,
        );
        if (seenFaces.has(id)) {
          throw new LocalContractError(`Frame ${frameIndex} contains duplicate face id ${id}.`);
        }
        seenFaces.add(id);
        const origin = face.origin ?? "manual";
        if (origin !== "manual" && origin !== "pnp-assist" && origin !== "model") {
          throw new LocalContractError(
            `frames[${framePosition}].faces[${facePosition}].origin is unsupported.`,
          );
        }
        let assist: AnnotationFace["assist"];
        if (face.assist !== undefined) {
          const assistValue = objectAt(
            face.assist,
            `frames[${framePosition}].faces[${facePosition}].assist`,
          );
          const faceName = assistValue.face_name;
          if (
            faceName !== "labeled" &&
            faceName !== "top" &&
            faceName !== "right" &&
            faceName !== "bottom" &&
            faceName !== "left" &&
            faceName !== "back"
          ) {
            throw new LocalContractError(
              `frames[${framePosition}].faces[${facePosition}].assist.face_name is unsupported.`,
            );
          }
          if (typeof assistValue.generated !== "boolean") {
            throw new LocalContractError(
              `frames[${framePosition}].faces[${facePosition}].assist.generated must be boolean.`,
            );
          }
          assist = {
            face_name: faceName,
            generated: assistValue.generated,
            ...(assistValue.intrinsics_source === undefined
              ? {}
              : {
                  intrinsics_source: stringAt(
                    assistValue.intrinsics_source,
                    `frames[${framePosition}].faces[${facePosition}].assist.intrinsics_source`,
                    100,
                  ),
                }),
          };
        }
        return {
          id,
          corners: fourPointsAt(
            face.corners,
            `frames[${framePosition}].faces[${facePosition}].corners`,
          ),
          visible: booleanFourAt(
            face.visible,
            `frames[${framePosition}].faces[${facePosition}].visible`,
          ),
          vertices:
            face.vertices === undefined
              ? [null, null, null, null]
              : vertexFourAt(
                  face.vertices,
                  `frames[${framePosition}].faces[${facePosition}].vertices`,
                ),
          pinned:
            face.pinned === undefined
              ? [false, false, false, false]
              : booleanFourAt(
                  face.pinned,
                  `frames[${framePosition}].faces[${facePosition}].pinned`,
                ),
          origin,
          ...(face.confidence === undefined
            ? {}
            : {
                confidence: numberAt(
                  face.confidence,
                  `frames[${framePosition}].faces[${facePosition}].confidence`,
                  { min: 0, max: 1 },
                ),
              }),
          ...(assist ? { assist } : {}),
        };
      });
      const polygons = arrayAt(
        frameValue.polygons ?? [],
        `frames[${framePosition}].polygons`,
        64,
      ).map((polygonEntry, polygonPosition): AnnotationPolygon => {
        const polygon = objectAt(
          polygonEntry,
          `frames[${framePosition}].polygons[${polygonPosition}]`,
        );
        const points = arrayAt(
          polygon.points,
          `frames[${framePosition}].polygons[${polygonPosition}].points`,
          128,
        );
        if (points.length < 3) {
          throw new LocalContractError(
            `frames[${framePosition}].polygons[${polygonPosition}].points needs at least three points.`,
          );
        }
        return {
          id: stringAt(
            polygon.id,
            `frames[${framePosition}].polygons[${polygonPosition}].id`,
            100,
          ),
          points: points.map((point, pointPosition) =>
            pointAt(
              point,
              `frames[${framePosition}].polygons[${polygonPosition}].points[${pointPosition}]`,
            ),
          ),
        };
      });
      const wireframeValue = frameValue.wireframe;
      let wireframe: Point[] | null = null;
      if (wireframeValue !== undefined) {
        const points = arrayAt(
          wireframeValue,
          `frames[${framePosition}].wireframe`,
          8,
        );
        if (points.length !== 8) {
          throw new LocalContractError(
            `frames[${framePosition}].wireframe must contain eight cube vertices.`,
          );
        }
        wireframe = points.map((point, pointPosition) =>
          pointAt(point, `frames[${framePosition}].wireframe[${pointPosition}]`),
        );
      }
      return {
        frame_index: frameIndex,
        time_seconds: numberAt(
          frameValue.time_seconds,
          `frames[${framePosition}].time_seconds`,
          { min: 0, max: 1_000_000 },
        ),
        aligned_label: aligned,
        faces,
        polygons,
        wireframe,
      };
    },
  );

  return {
    schema: "cubed-core/frame-annotations",
    schema_version: 1,
    created_at: stringAt(root.created_at, "created_at", 100),
    source,
    image,
    frames: frames.sort((a, b) => a.frame_index - b.frame_index),
  };
}

export function parseTrackerDump(value: unknown): TrackerDump {
  const root = objectAt(value, "tracker dump");
  if (root.schema !== "cubed-core/tracker-dump-v1") {
    throw new LocalContractError(
      "Tracker JSON must declare cubed-core/tracker-dump-v1.",
    );
  }
  if (
    root.schema_version !== undefined &&
    root.schema_version !== 1
  ) {
    throw new LocalContractError("Unsupported Cubed Core tracker dump version.");
  }
  const windowValue = arrayAt(root.window, "window", 2);
  if (windowValue.length !== 2) {
    throw new LocalContractError("window must contain the first and last frame.");
  }
  const window: [number, number] = [
    integerAt(windowValue[0], "window[0]", { max: 10_000_000 }),
    integerAt(windowValue[1], "window[1]", { max: 10_000_000 }),
  ];
  if (window[1] < window[0]) {
    throw new LocalContractError("window end must not be before its start.");
  }
  const fps =
    root.fps === undefined
      ? undefined
      : numberAt(root.fps, "fps", { min: 1, max: 1_000 });
  const width =
    root.width === undefined
      ? undefined
      : integerAt(root.width, "width", { min: 1, max: 100_000 });
  const height =
    root.height === undefined
      ? undefined
      : integerAt(root.height, "height", { min: 1, max: 100_000 });

  const anchors = arrayAt(root.anchors ?? [], "anchors", 10_000).map(
    (entry, index): [number, string, number] => {
      const anchor = arrayAt(entry, `anchors[${index}]`, 3);
      if (anchor.length !== 3) {
        throw new LocalContractError(`anchors[${index}] must have three values.`);
      }
      return [
        integerAt(anchor[0], `anchors[${index}][0]`, { max: 10_000_000 }),
        stringAt(anchor[1], `anchors[${index}][1]`, 2_000),
        numberAt(anchor[2], `anchors[${index}][2]`, {
          min: -1_000_000,
          max: 1_000_000,
        }),
      ];
    },
  );
  const events = arrayAt(root.events ?? [], "events", 100_000).map((entry, index) =>
    integerAt(entry, `events[${index}]`, { max: 10_000_000 }),
  );
  const spans = arrayAt(root.spans ?? [], "spans", 100_000).map(
    (entry, index): TrackerSpan => {
      const span = objectAt(entry, `spans[${index}]`);
      const f0 = integerAt(span.f0, `spans[${index}].f0`, { max: 10_000_000 });
      const f1 = integerAt(span.f1, `spans[${index}].f1`, { max: 10_000_000 });
      if (f1 < f0) throw new LocalContractError(`spans[${index}] has a reversed range.`);
      const event =
        span.event === null
          ? null
          : integerAt(span.event, `spans[${index}].event`, { max: 10_000_000 });
      const top = arrayAt(span.top ?? [], `spans[${index}].top`, 100).map(
        (candidateEntry, candidateIndex) => {
          const candidate = objectAt(
            candidateEntry,
            `spans[${index}].top[${candidateIndex}]`,
          );
          const om = Array.isArray(candidate.om)
            ? stringListAt(
                candidate.om,
                `spans[${index}].top[${candidateIndex}].om`,
                100,
              )
            : boundedStringAt(
                candidate.om,
                `spans[${index}].top[${candidateIndex}].om`,
                1_000,
              );
          return {
            path: stringAt(
              candidate.path,
              `spans[${index}].top[${candidateIndex}].path`,
              5_000,
            ),
            om,
            score: numberAt(
              candidate.score,
              `spans[${index}].top[${candidateIndex}].score`,
              { min: -1e100, max: 1e100 },
            ),
          };
        },
      );
      return { f0, f1, event, top };
    },
  );

  const rawFrames = objectAt(root.frames, "frames");
  const frameEntries = Object.entries(rawFrames);
  if (frameEntries.length > 250_000) {
    throw new LocalContractError("frames contains more than 250,000 records.");
  }
  const frames: Record<string, TrackerFrame> = {};
  for (const [key, entry] of frameEntries) {
    if (!/^(0|[1-9]\d*)$/.test(key) || Number(key) > 10_000_000) {
      throw new LocalContractError(`frames key ${key} is not a valid frame index.`);
    }
    const frame = objectAt(entry, `frames.${key}`);
    const motion =
      frame.motion === null
        ? null
        : numberAt(frame.motion, `frames.${key}.motion`, {
            min: -1e100,
            max: 1e100,
          });
    const faces =
      frame.faces === undefined
        ? undefined
        : arrayAt(frame.faces, `frames.${key}.faces`, 16).map(
            (faceEntry, faceIndex): TrackerFace => {
              const face = objectAt(faceEntry, `frames.${key}.faces[${faceIndex}]`);
              const confidence = arrayAt(
                face.kpt_conf,
                `frames.${key}.faces[${faceIndex}].kpt_conf`,
                4,
              );
              if (confidence.length !== 4) {
                throw new LocalContractError(
                  `frames.${key}.faces[${faceIndex}].kpt_conf needs four values.`,
                );
              }
              return {
                corners: fourPointsAt(
                  face.corners,
                  `frames.${key}.faces[${faceIndex}].corners`,
                ),
                kpt_conf: [
                  numberAt(confidence[0], "corner confidence", { min: 0, max: 1 }),
                  numberAt(confidence[1], "corner confidence", { min: 0, max: 1 }),
                  numberAt(confidence[2], "corner confidence", { min: 0, max: 1 }),
                  numberAt(confidence[3], "corner confidence", { min: 0, max: 1 }),
                ],
                conf: numberAt(
                  face.conf,
                  `frames.${key}.faces[${faceIndex}].conf`,
                  { min: 0, max: 1 },
                ),
              };
            },
          );
    const seenReadSlots = new Set<TrackerFaceRead["slot"]>();
    let reads: TrackerFaceRead[] | undefined;
    if (frame.reads !== undefined) {
      const readEntries = arrayAt(frame.reads, `frames.${key}.reads`, 3);
      if (readEntries.length === 0) {
        throw new LocalContractError(`frames.${key}.reads must not be empty when present.`);
      }
      reads = readEntries.map(
            (readEntry, readIndex): TrackerFaceRead => {
              const path = `frames.${key}.reads[${readIndex}]`;
              const read = objectAt(readEntry, path);
              if (read.slot !== "up" && read.slot !== "front" && read.slot !== "right") {
                throw new LocalContractError(`${path}.slot must be up, front, or right.`);
              }
              if (seenReadSlots.has(read.slot)) {
                throw new LocalContractError(
                  `frames.${key}.reads contains duplicate slot ${read.slot}.`,
                );
              }
              seenReadSlots.add(read.slot);
              const lab: Nine<LabVector> = nineAt(
                read.lab,
                `${path}.lab`,
                (vectorEntry, cellIndex): LabVector => {
                  const vector = arrayAt(vectorEntry, `${path}.lab[${cellIndex}]`, 3);
                  if (vector.length !== 3) {
                    throw new LocalContractError(
                      `${path}.lab[${cellIndex}] must contain L, a, and b.`,
                    );
                  }
                  return [
                    numberAt(vector[0], `${path}.lab[${cellIndex}][0]`, {
                      min: 0,
                      max: 255,
                    }),
                    numberAt(vector[1], `${path}.lab[${cellIndex}][1]`, {
                      min: 0,
                      max: 255,
                    }),
                    numberAt(vector[2], `${path}.lab[${cellIndex}][2]`, {
                      min: 0,
                      max: 255,
                    }),
                  ];
                },
              );
              const confidence = nineAt(
                read.confidence,
                `${path}.confidence`,
                (entry, index) =>
                  numberAt(entry, `${path}.confidence[${index}]`, {
                    min: 0,
                    max: 1,
                  }),
              );
              const validPixels = nineAt(
                read.valid_pixels,
                `${path}.valid_pixels`,
                (entry, index) =>
                  integerAt(entry, `${path}.valid_pixels[${index}]`, {
                    max: 16_777_216,
                  }),
              );
              const totalPixels = nineAt(
                read.total_pixels,
                `${path}.total_pixels`,
                (entry, index) =>
                  integerAt(entry, `${path}.total_pixels[${index}]`, {
                    min: 1,
                    max: 16_777_216,
                  }),
              );
              validPixels.forEach((count, index) => {
                if (count > totalPixels[index]) {
                  throw new LocalContractError(
                    `${path}.valid_pixels[${index}] must not exceed total_pixels.`,
                  );
                }
              });
              const usedFallback = nineAt(
                read.used_fallback,
                `${path}.used_fallback`,
                (entry, index) => {
                  if (typeof entry !== "boolean") {
                    throw new LocalContractError(
                      `${path}.used_fallback[${index}] must be boolean.`,
                    );
                  }
                  return entry;
                },
              );
              return {
                slot: read.slot,
                lab,
                confidence,
                valid_pixels: validPixels,
                total_pixels: totalPixels,
                used_fallback: usedFallback,
                relative_area: numberAt(read.relative_area, `${path}.relative_area`, {
                  min: 0,
                  max: 1,
                }),
                corners: fourPointsAt(read.corners, `${path}.corners`),
              };
            },
          );
    }
    let read: TrackerClassifiedRead[] | undefined;
    if (frame.read !== undefined) {
      const classifiedEntries = arrayAt(frame.read, `frames.${key}.read`, 16);
      if (classifiedEntries.length === 0) {
        throw new LocalContractError(`frames.${key}.read must not be empty when present.`);
      }
      read = classifiedEntries.map(
        (readEntry, readIndex): TrackerClassifiedRead => {
          const path = `frames.${key}.read[${readIndex}]`;
          const classified = objectAt(readEntry, path);
          return {
            slot: stringAt(classified.slot, `${path}.slot`, 32),
            colors: nineAt(
              classified.colors,
              `${path}.colors`,
              (entry, index) =>
                entry === null
                  ? null
                  : stringAt(entry, `${path}.colors[${index}]`, 32),
            ),
            conf: nineAt(
              classified.conf,
              `${path}.conf`,
              (entry, index) =>
                entry === null
                  ? null
                  : numberAt(entry, `${path}.conf[${index}]`, {
                      min: 0,
                      max: 1,
                    }),
            ),
            dist: nineAt(
              classified.dist,
              `${path}.dist`,
              (entry, index) =>
                numberAt(entry, `${path}.dist[${index}]`, {
                  min: 0,
                  max: 1e100,
                }),
            ),
          };
        },
      );
    }
    let gated: boolean | undefined;
    if (frame.gated !== undefined) {
      if (typeof frame.gated !== "boolean") {
        throw new LocalContractError(`frames.${key}.gated must be boolean.`);
      }
      gated = frame.gated;
    }
    frames[key] = {
      motion,
      ...(frame.aligned === undefined
        ? {}
        : {
            aligned: numberAt(frame.aligned, `frames.${key}.aligned`, {
              min: 0,
              max: 1,
            }),
          }),
      ...(frame.stk === undefined
        ? {}
        : {
            stk:
              frame.stk === null
                ? null
                : integerAt(frame.stk, `frames.${key}.stk`, { max: 10_000 }),
          }),
      ...(frame.nfaces === undefined
        ? {}
        : {
            nfaces:
              frame.nfaces === null
                ? null
                : integerAt(frame.nfaces, `frames.${key}.nfaces`, {
                    max: 100,
                  }),
          }),
      ...(gated === undefined ? {} : { gated }),
      ...(faces ? { faces } : {}),
      ...(reads ? { reads } : {}),
      ...(read ? { read } : {}),
    };
  }

  const cubeFaces = ["up", "right", "front", "down", "left", "back"] as const;
  let states: Record<string, string[]>[] | undefined;
  if (root.states !== undefined) {
    states = arrayAt(root.states, "states", 2_001).map((entry, stateIndex) => {
      const state = objectAt(entry, `states[${stateIndex}]`);
      const unsupported = Object.keys(state).find(
        (face) => !cubeFaces.includes(face as (typeof cubeFaces)[number]),
      );
      if (unsupported) {
        throw new LocalContractError(
          `states[${stateIndex}] contains unsupported face ${unsupported}.`,
        );
      }
      return Object.fromEntries(
        cubeFaces.map((face) => {
          const colors = arrayAt(
            state[face],
            `states[${stateIndex}].${face}`,
            9,
          );
          if (colors.length !== 9) {
            throw new LocalContractError(
              `states[${stateIndex}].${face} must contain exactly nine stickers.`,
            );
          }
          return [
            face,
            colors.map((color, colorIndex) =>
              stringAt(
                color,
                `states[${stateIndex}].${face}[${colorIndex}]`,
                32,
              ),
            ),
          ];
        }),
      );
    });
  }
  let reach: boolean | undefined;
  if (root.reach !== undefined) {
    if (typeof root.reach !== "boolean") {
      throw new LocalContractError("reach must be boolean.");
    }
    reach = root.reach;
  }

  return {
    schema: "cubed-core/tracker-dump-v1",
    schema_version: 1,
    ...(root.mode === undefined ? {} : { mode: stringAt(root.mode, "mode", 100) }),
    ...(fps === undefined ? {} : { fps }),
    window,
    ...(width === undefined ? {} : { width }),
    ...(height === undefined ? {} : { height }),
    gt: stringListAt(root.gt, "gt"),
    emitted: stringListAt(root.emitted, "emitted"),
    ...(root.gt_canonical === undefined
      ? {}
      : { gt_canonical: stringListAt(root.gt_canonical, "gt_canonical") }),
    ...(root.emitted_canonical === undefined
      ? {}
      : {
          emitted_canonical: stringListAt(
            root.emitted_canonical,
            "emitted_canonical",
          ),
        }),
    anchors,
    events,
    spans,
    frames,
    ...(root.bridge === undefined
      ? {}
      : { bridge: stringListAt(root.bridge, "bridge") }),
    ...(reach === undefined ? {} : { reach }),
    ...(root.acc === undefined
      ? {}
      : {
          acc: numberAt(root.acc, "acc", {
            min: -1e100,
            max: 1e100,
          }),
        }),
    ...(states ? { states } : {}),
  };
}
