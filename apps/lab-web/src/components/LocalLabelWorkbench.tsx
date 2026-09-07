import {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useSearchParams } from "react-router-dom";

import {
  RequestError,
  createCaptureMediaTicket,
  downloadLabelDatasetExport,
  extrapolateLabelFace,
  fetchCapabilities,
  fetchCaptureAnnotations,
  fetchCaptures,
  predictCaptureLabels,
  saveCaptureAnnotations,
  scanCaptureLabelAlignment,
} from "../api";
import { safeDownloadStem, triggerBlobDownload } from "../browserDownload";
import {
  EMPTY_LABEL_FRAME,
  FrameDraft,
  FrameDrafts,
  appendRepeatedGeometry,
  annotationSourceMismatchFields,
  buildLabelAssistRequest,
  buildFrameAnnotationsDocument,
  cloneLabelDrafts,
  draftsFromDocument,
  faceFromFourPoints,
  faceFromOrderedPoints,
  newAnnotationId,
  frameHasAnnotation,
  localAnnotationStorageKey,
  nextFilteredFrame,
  nextUnclassifiedFrame,
  toggledAlignment,
  translatePolygonWithinImage,
} from "../labelTools";
import {
  AlignmentLabel,
  AnnotationFace,
  AnnotationPolygon,
  AnnotationSource,
  FrameAnnotationsDocument,
  LocalContractError,
  Point,
  parseFrameAnnotations,
  readJsonFile,
} from "../localContracts";
import {
  CAPTURE_PARAM,
  captureIdFromParams,
  orderDecodeCaptures,
} from "../captureSelection";
import type { Capabilities, CaptureReceipt } from "../types";
import "../labTools.css";
import "./LocalLabelWorkbench.css";

type LabelTool = "select" | "draw" | "corners";
type Selection = { kind: "face" | "polygon"; id: string } | null;
type DragTarget =
  | { kind: "face-corner"; id: string; pointIndex: number }
  | { kind: "polygon-point"; id: string; pointIndex: number }
  | { kind: "polygon"; id: string; start: Point; original: Point[] };

const VIDEO_METADATA_TIMEOUT_MS = 8_000;

const CORNER_NAMES = ["1", "2", "3", "4"];
const WIREFRAME_EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
];

function annotationCount(frames: FrameDrafts): number {
  return Object.values(frames).reduce(
    (total, frame) => total + frame.faces.length + frame.polygons.length,
    0,
  );
}

function triggerJsonDownload(value: unknown, filename: string) {
  triggerBlobDownload(
    new Blob([`${JSON.stringify(value, null, 2)}\n`], {
      type: "application/json",
    }),
    filename,
  );
}

function workspaceSource(receipt: CaptureReceipt): AnnotationSource {
  const fps = receipt.video.actual_fps ?? receipt.probe.fps ?? 120;
  return {
    kind: "workspace-capture",
    capture_id: receipt.capture_id,
    filename: receipt.original_filename,
    sha256: receipt.video.sha256,
    fps,
    frame_count:
      receipt.video.frame_count ??
      (receipt.probe.duration_seconds
        ? Math.max(1, Math.ceil(receipt.probe.duration_seconds * fps))
        : null),
  };
}

function localSource(file: File): AnnotationSource {
  return {
    kind: "local-file",
    capture_id: null,
    filename: file.name,
    sha256: null,
    fps: 120,
    frame_count: null,
  };
}

function clampPoint(point: Point, dimensions: { width: number; height: number }): Point {
  return [
    Math.max(0, Math.min(dimensions.width, Math.round(point[0] * 10) / 10)),
    Math.max(0, Math.min(dimensions.height, Math.round(point[1] * 10) / 10)),
  ];
}

function distance(left: Point, right: Point): number {
  return Math.hypot(left[0] - right[0], left[1] - right[1]);
}

export function LocalLabelWorkbench() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [captures, setCaptures] = useState<CaptureReceipt[]>([]);
  const [capturesLoaded, setCapturesLoaded] = useState(false);
  const [captureListError, setCaptureListError] = useState("");
  const [capabilitiesError, setCapabilitiesError] = useState("");
  const [workspaceLoadAttempt, setWorkspaceLoadAttempt] = useState(0);
  const [captureId, setCaptureId] = useState("");
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [source, setSource] = useState<AnnotationSource | null>(null);
  const [sourceUrl, setSourceUrl] = useState("");
  const [localStorageKey, setLocalStorageKey] = useState("");
  const [fps, setFps] = useState(120);
  const [duration, setDuration] = useState(0);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [frameIndex, setFrameIndex] = useState(0);
  const [frames, setFrames] = useState<FrameDrafts>({});
  const [tool, setTool] = useState<LabelTool>("corners");
  const [draftPoints, setDraftPoints] = useState<Point[]>([]);
  const [selection, setSelection] = useState<Selection>(null);
  const [dragging, setDragging] = useState<DragTarget | null>(null);
  const [annotatedOnly, setAnnotatedOnly] = useState(false);
  const [alignedOnly, setAlignedOnly] = useState(false);
  const [modelAlignedOnly, setModelAlignedOnly] = useState(false);
  const [modelAlignedFrames, setModelAlignedFrames] = useState<number[]>([]);
  const [alignmentConfs, setAlignmentConfs] = useState<number[]>([]);
  const [alignmentScanStatus, setAlignmentScanStatus] = useState<
    "idle" | "scanning" | "ready" | "unavailable"
  >("idle");
  const [alignmentScanError, setAlignmentScanError] = useState("");
  const [classifyMode, setClassifyMode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [assistBusy, setAssistBusy] = useState(false);
  const [assistCamera, setAssistCamera] = useState<"auto" | "generic" | "custom">(
    "auto",
  );
  const [assistFocal, setAssistFocal] = useState("");
  const [assistShrink, setAssistShrink] = useState("1");
  /// Cursor position in image coordinates while placing points, for the
  /// dashed guide from the last placed point to the pointer.
  const [guidePoint, setGuidePoint] = useState<Point | null>(null);
  const [autosaveReady, setAutosaveReady] = useState(false);
  const [autosaveStatus, setAutosaveStatus] = useState("not started");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const objectUrlRef = useRef("");
  const alignmentRequestRef = useRef(0);
  const undoRef = useRef<FrameDrafts[]>([]);
  const framesRef = useRef<FrameDrafts>({});
  const sourceRef = useRef<AnnotationSource | null>(null);
  const dragChangedRef = useRef(false);
  const localRestoredKeyRef = useRef("");
  const solveTimerRef = useRef(0);
  const videoLoadTimeoutRef = useRef(0);

  useEffect(() => {
    framesRef.current = frames;
  }, [frames]);
  useEffect(() => {
    sourceRef.current = source;
  }, [source]);

  const current = frames[String(frameIndex)] ?? EMPTY_LABEL_FRAME;
  const selectedFace =
    selection?.kind === "face"
      ? current.faces.find((face) => face.id === selection.id) ?? null
      : null;
  const selectedPolygon =
    selection?.kind === "polygon"
      ? current.polygons.find((polygon) => polygon.id === selection.id) ?? null
      : null;
  const totalAnnotations = useMemo(() => annotationCount(frames), [frames]);
  const labeledFrames = useMemo(
    () => Object.values(frames).filter(frameHasAnnotation).length,
    [frames],
  );
  const maxFrame = useMemo(() => {
    if (source?.frame_count) return Math.max(0, source.frame_count - 1);
    if (duration > 0) return Math.max(0, Math.ceil(duration * fps) - 1);
    return 0;
  }, [duration, fps, source?.frame_count]);
  const frameRangeUnknown = useMemo(
    () => Boolean(source && !source.frame_count && duration <= 0),
    [duration, source],
  );
  const pnpCapability = capabilities?.label?.pnp_assist;
  const predictionCapability = capabilities?.label?.prediction;
  const predictionEnabled = Boolean(
    source?.kind === "workspace-capture" && predictionCapability?.enabled,
  );
  const modelAlignmentEnabled = Boolean(
    source?.kind === "workspace-capture" &&
      predictionCapability?.model_aligned_navigation,
  );
  const modelAlignedSet = useMemo(
    () => new Set(modelAlignedFrames),
    [modelAlignedFrames],
  );
  const currentModelAlignment = alignmentConfs[frameIndex];
  const captureGroups = useMemo(
    () => orderDecodeCaptures(captures),
    [captures],
  );
  const workspaceCapture =
    source?.kind === "workspace-capture"
      ? captures.find((capture) => capture.capture_id === source.capture_id) ?? null
      : null;
  const workspaceCameraGeometry = workspaceCapture
    ? {
        intrinsics: workspaceCapture.camera?.intrinsics ?? null,
        encodedWidth: workspaceCapture.video.encoded_width,
        encodedHeight: workspaceCapture.video.encoded_height,
        rotationDegrees: workspaceCapture.video.rotation_degrees,
      }
    : null;
  // The SVG overlay draws in image coordinates, so marker geometry must be
  // scaled to hold a constant ON-SCREEN size. Scaling by image width alone is
  // not enough: a portrait 4K frame fitted into the stage displays a few
  // hundred CSS pixels wide, and markers shrink with it. uiScale is image
  // pixels per displayed CSS pixel, measured from the rendered overlay.
  const [uiScale, setUiScale] = useState(1);
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg || !dimensions.width || !dimensions.height) return;
    const update = () => {
      const rect = svg.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      const contentWidth = Math.min(
        rect.width,
        rect.height * (dimensions.width / dimensions.height),
      );
      if (contentWidth > 0) setUiScale(dimensions.width / contentWidth);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(svg);
    return () => observer.disconnect();
  }, [dimensions.width, dimensions.height]);

  useEffect(() => {
    let active = true;
    setCapturesLoaded(false);
    setCaptureListError("");
    setCapabilitiesError("");
    setCaptures([]);
    setCapabilities(null);
    void fetchCaptures()
      .then((capturePayload) => {
        if (!active) return;
        setCaptures(capturePayload.captures);
        const orderedCaptures = orderDecodeCaptures(capturePayload.captures);
        setCaptureId(
          (currentId) =>
            currentId ||
            captureIdFromParams(searchParams, [
              ...orderedCaptures.yourRecordings,
              ...orderedCaptures.publishedDataset,
            ]),
        );
      })
      .catch((reason) => {
        if (active) {
          setCaptureListError(
            reason instanceof Error
              ? reason.message
              : "The capture workspace could not be read.",
          );
        }
      })
      .finally(() => {
        if (active) setCapturesLoaded(true);
      });
    void fetchCapabilities()
      .then((capabilityPayload) => {
        if (active) setCapabilities(capabilityPayload);
      })
      .catch((reason) => {
        if (active) {
          setCapabilitiesError(
            reason instanceof Error
              ? reason.message
              : "The local workbench capabilities could not be read.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, [workspaceLoadAttempt]);

  useEffect(
    () => () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      window.clearTimeout(solveTimerRef.current);
      window.clearTimeout(videoLoadTimeoutRef.current);
    },
    [],
  );

  useEffect(() => {
    let animationFrame = 0;
    const update = () => {
      const video = videoRef.current;
      if (video && !video.paused && Number.isFinite(video.currentTime)) {
        setFrameIndex(Math.min(maxFrame, Math.max(0, Math.floor(video.currentTime * fps))));
        animationFrame = window.requestAnimationFrame(update);
      }
    };
    const video = videoRef.current;
    const start = () => {
      window.cancelAnimationFrame(animationFrame);
      animationFrame = window.requestAnimationFrame(update);
    };
    video?.addEventListener("play", start);
    return () => {
      video?.removeEventListener("play", start);
      window.cancelAnimationFrame(animationFrame);
    };
  }, [fps, maxFrame, sourceUrl]);

  const pushUndo = useCallback((snapshot: FrameDrafts) => {
    undoRef.current.push(cloneLabelDrafts(snapshot));
    if (undoRef.current.length > 80) undoRef.current.shift();
  }, []);

  const resetForSource = useCallback(
    (nextSource: AnnotationSource, storageKey = "") => {
      alignmentRequestRef.current += 1;
      window.clearTimeout(videoLoadTimeoutRef.current);
      setSource(nextSource);
      setLocalStorageKey(storageKey);
      setFps(nextSource.fps);
      setDuration(0);
      setDimensions({ width: 0, height: 0 });
      setFrameIndex(0);
      setFrames({});
      setModelAlignedOnly(false);
      setModelAlignedFrames([]);
      setAlignmentConfs([]);
      setAlignmentScanStatus("idle");
      setAlignmentScanError("");
      setDraftPoints([]);
      setSelection(null);
      setTool("corners");
      setAutosaveReady(false);
      setAutosaveStatus("loading");
      localRestoredKeyRef.current = "";
      undoRef.current = [];
      setMessage(`Opened ${nextSource.filename}.`);
      setError("");
    },
    [],
  );

  const installLocalSource = useCallback(
    (file: File, nextSource: AnnotationSource) => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const nextUrl = URL.createObjectURL(file);
      objectUrlRef.current = nextUrl;
      setSourceUrl(nextUrl);
      resetForSource(nextSource, localAnnotationStorageKey(file));
      videoLoadTimeoutRef.current = window.setTimeout(() => {
        setError("The file could not be opened as a video. This browser could not decode it.");
        setAutosaveReady(true);
        setAutosaveStatus("ready");
      }, VIDEO_METADATA_TIMEOUT_MS);
    },
    [resetForSource],
  );

  const installWorkspaceSource = useCallback(
    async (nextSource: AnnotationSource, mediaUrl: string) => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = "";
      }
      setSourceUrl(mediaUrl);
      resetForSource(nextSource);
      videoLoadTimeoutRef.current = window.setTimeout(() => {
        setError(
          "The workspace video did not become ready. Open the capture again to issue a fresh read-only media ticket.",
        );
        setAutosaveReady(true);
        setAutosaveStatus("ready");
      }, VIDEO_METADATA_TIMEOUT_MS);
      try {
        const loaded = parseFrameAnnotations(
          await fetchCaptureAnnotations(nextSource.capture_id!),
        );
        setFrames(draftsFromDocument(loaded));
        setFps(loaded.source.fps);
        setDimensions({ width: loaded.image.width, height: loaded.image.height });
        setMessage(`Reloaded ${loaded.frames.length} autosaved frame(s).`);
      } catch (reason) {
        if (!(reason instanceof RequestError && reason.status === 404)) {
          setError(
            reason instanceof Error
              ? reason.message
              : "Saved annotations could not be reloaded.",
          );
        }
      } finally {
        setAutosaveReady(true);
        setAutosaveStatus("ready");
      }
    },
    [resetForSource],
  );

  useEffect(() => {
    if (
      source?.kind !== "local-file" ||
      !localStorageKey ||
      !dimensions.width ||
      localRestoredKeyRef.current === localStorageKey
    ) {
      return;
    }
    localRestoredKeyRef.current = localStorageKey;
    try {
      const stored = window.localStorage.getItem(localStorageKey);
      if (stored) {
        const document = parseFrameAnnotations(JSON.parse(stored) as unknown);
        if (document.source.filename === source.filename) {
          setFrames(draftsFromDocument(document));
          setFps(document.source.fps);
          setMessage(
            `Reloaded ${document.frames.length} browser-autosaved frame(s) for this file.`,
          );
        }
      }
    } catch (reason) {
      setError(
        reason instanceof Error
          ? `Browser autosave could not be restored: ${reason.message}`
          : "Browser autosave could not be restored.",
      );
    } finally {
      setAutosaveReady(true);
      setAutosaveStatus("ready");
    }
  }, [dimensions.width, localStorageKey, source]);

  const buildDocument = useCallback((): FrameAnnotationsDocument | null => {
    if (!source || !dimensions.width || !dimensions.height) return null;
    return buildFrameAnnotationsDocument(source, framesRef.current, dimensions, fps, maxFrame);
  }, [dimensions, fps, maxFrame, source]);

  const persistAnnotations = useCallback(
    async (announce: boolean) => {
      const document = buildDocument();
      const activeSource = sourceRef.current;
      if (!document || !activeSource) {
        if (announce) setError("Open a decoded video frame before saving annotations.");
        return false;
      }
      setAutosaveStatus("saving…");
      try {
        if (activeSource.kind === "workspace-capture" && activeSource.capture_id) {
          await saveCaptureAnnotations(activeSource.capture_id, document);
        } else {
          if (!localStorageKey) throw new Error("The local-file autosave key is unavailable.");
          window.localStorage.setItem(localStorageKey, JSON.stringify(document));
        }
        setAutosaveStatus(`saved ${new Date().toLocaleTimeString()}`);
        if (announce) setMessage(`Saved ${document.frames.length} labeled frame(s).`);
        return true;
      } catch (reason) {
        setAutosaveStatus("save failed");
        setError(
          reason instanceof Error ? reason.message : "Annotations could not be autosaved.",
        );
        return false;
      }
    },
    [buildDocument, localStorageKey],
  );

  useEffect(() => {
    if (!autosaveReady || !source || !dimensions.width || !dimensions.height) return;
    const timer = window.setTimeout(() => {
      void persistAnnotations(false);
    }, 650);
    return () => window.clearTimeout(timer);
  }, [autosaveReady, dimensions.height, dimensions.width, frames, persistAnnotations, source]);

  const updateCurrent = useCallback(
    (update: (value: FrameDraft) => FrameDraft, recordUndo = true) => {
      setFrames((previous) => {
        if (recordUndo) pushUndo(previous);
        const key = String(frameIndex);
        const nextFrame = update(previous[key] ?? EMPTY_LABEL_FRAME);
        const next = { ...previous };
        if (!frameHasAnnotation(nextFrame)) delete next[key];
        else next[key] = nextFrame;
        framesRef.current = next;
        return next;
      });
    },
    [frameIndex, pushUndo],
  );

  const moveAnnotation = useCallback(
    (target: Exclude<Selection, null>, deltaX: number, deltaY: number) => {
      if (dimensions.width <= 0 || dimensions.height <= 0) return;
      updateCurrent((frame) => {
        if (target.kind === "polygon") {
          return {
            ...frame,
            polygons: frame.polygons.map((polygon) =>
              polygon.id === target.id
                ? {
                    ...polygon,
                    points: translatePolygonWithinImage(
                      polygon.points,
                      [0, 0],
                      [deltaX, deltaY],
                      dimensions,
                    ),
                  }
                : polygon,
            ),
          };
        }

        return {
          ...frame,
          faces: frame.faces.map((face) => {
            if (face.id !== target.id) return face;
            const xs = face.corners.map(([x]) => x);
            const ys = face.corners.map(([, y]) => y);
            const boundedX = Math.max(
              -Math.min(...xs),
              Math.min(dimensions.width - Math.max(...xs), deltaX),
            );
            const boundedY = Math.max(
              -Math.min(...ys),
              Math.min(dimensions.height - Math.max(...ys), deltaY),
            );
            return {
              ...face,
              corners: face.corners.map(
                ([x, y]) => [x + boundedX, y + boundedY] as Point,
              ) as AnnotationFace["corners"],
            };
          }),
        };
      });
      setSelection(target);
      setTool("select");
      setMessage(
        `Moved ${target.kind} ${deltaX || deltaY}px on frame ${frameIndex}.`,
      );
    },
    [dimensions, frameIndex, updateCurrent],
  );

  const seekFrame = useCallback(
    (requested: number) => {
      const next = Math.max(0, Math.min(maxFrame, Math.round(requested)));
      const video = videoRef.current;
      video?.pause();
      if (video && Number.isFinite(video.duration)) {
        video.currentTime = Math.min(
          Math.max(0, video.duration - 0.000_001),
          (next + 0.5) / fps,
        );
      }
      setFrameIndex(next);
      setDraftPoints([]);
      setSelection(null);
    },
    [fps, maxFrame],
  );

  const seekFiltered = useCallback(
    (direction: -1 | 1, step: number) => {
      seekFrame(
        nextFilteredFrame(
          frameIndex,
          direction,
          step,
          maxFrame,
          framesRef.current,
          {
            annotated: annotatedOnly,
            aligned: alignedOnly,
            modelAligned: modelAlignedOnly ? modelAlignedSet : null,
          },
        ),
      );
    },
    [
      alignedOnly,
      annotatedOnly,
      frameIndex,
      maxFrame,
      modelAlignedOnly,
      modelAlignedSet,
      seekFrame,
    ],
  );

  const pointFromEvent = (
    event: ReactPointerEvent<SVGSVGElement | SVGCircleElement | SVGPolygonElement>,
  ): Point | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const matrix = svg.getScreenCTM();
    if (!matrix) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const transformed = point.matrixTransform(matrix.inverse());
    return clampPoint([transformed.x, transformed.y], dimensions);
  };

  const applyAssist = useCallback(
    (
      frame: FrameDraft,
      result: Awaited<ReturnType<typeof extrapolateLabelFace>>,
      baseId?: string,
    ): FrameDraft => {
      if (!result.ok || !result.faces || !result.wireframe) return frame;
      const pinnedByVertex = new Map<number, Point>();
      const visibleByVertex = new Map<number, boolean>();
      for (const face of frame.faces) {
        face.vertices.forEach((vertex, index) => {
          if (vertex === null) return;
          visibleByVertex.set(vertex, face.visible[index]);
          if (face.pinned[index]) pinnedByVertex.set(vertex, face.corners[index]);
        });
      }
      const existingByName = new Map(
        frame.faces
          .filter((face) => face.assist)
          .map((face) => [face.assist!.face_name, face]),
      );
      const unrelated = frame.faces.filter(
        (face) =>
          !face.assist?.generated &&
          face.id !== baseId &&
          face.assist?.face_name !== "labeled",
      );
      const solvedFaces = result.faces.map((solved): AnnotationFace => {
        const existing =
          solved.name === "labeled"
            ? frame.faces.find((face) => face.id === baseId) ??
              existingByName.get("labeled")
            : existingByName.get(solved.name);
        const corners = solved.corners.map((point, index) => {
          const pinned = pinnedByVertex.get(solved.vertices[index]);
          return pinned ? ([...pinned] as Point) : clampPoint(point, dimensions);
        }) as AnnotationFace["corners"];
        return {
          id: existing?.id ?? newAnnotationId(),
          corners,
          visible: solved.vertices.map(
            (vertex, index) => visibleByVertex.get(vertex) ?? existing?.visible[index] ?? true,
          ) as AnnotationFace["visible"],
          vertices: [...solved.vertices],
          pinned: solved.vertices.map((vertex) =>
            pinnedByVertex.has(vertex),
          ) as AnnotationFace["pinned"],
          origin: "pnp-assist",
          assist: {
            face_name: solved.name,
            generated: solved.name !== "labeled",
            intrinsics_source: result.intrinsics_source,
          },
        };
      });
      return {
        ...frame,
        faces: [...unrelated, ...solvedFaces],
        wireframe: result.wireframe.map((point) =>
          clampPoint(point, dimensions),
        ),
      };
    },
    [dimensions],
  );

  const runAssist = useCallback(
    async (points: Point[], baseId?: string, pins: { xy: Point; vertex: number }[] = []) => {
      if (!pnpCapability?.enabled) {
        setError(
          pnpCapability?.reason ??
            "PnP assist is disabled. Install the standard label dependencies.",
        );
        return;
      }
      if (!dimensions.width || !dimensions.height) return;
      setAssistBusy(true);
      setError("");
      try {
        const focal = Number(assistFocal);
        const shrink = Number(assistShrink);
        const result = await extrapolateLabelFace(
          buildLabelAssistRequest(
            points,
            dimensions,
            pins,
            workspaceCameraGeometry,
            {
              camera:
                assistCamera === "custom"
                  ? Number.isFinite(focal) && focal > 0
                    ? focal
                    : "generic"
                  : assistCamera,
              shrink:
                Number.isFinite(shrink) && shrink > 0 ? shrink : undefined,
            },
          ),
        );
        if (!result.ok) throw new Error(result.reason ?? "PnP could not solve this face.");
        updateCurrent((frame) => applyAssist(frame, result, baseId));
        const labeled = result.faces?.find((face) => face.name === "labeled");
        setDraftPoints([]);
        if (labeled) {
          window.setTimeout(() => {
            const updated = framesRef.current[String(frameIndex)];
            const face = updated?.faces.find(
              (candidate) => candidate.assist?.face_name === "labeled",
            );
            if (face) setSelection({ kind: "face", id: face.id });
          }, 0);
        }
        setMessage(
          `Rigid cube pose solved using ${result.intrinsics_source.replaceAll("-", " ")}.`,
        );
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "PnP assist failed.");
      } finally {
        setAssistBusy(false);
      }
    },
    [
      applyAssist,
      assistCamera,
      assistFocal,
      assistShrink,
      dimensions,
      frameIndex,
      pnpCapability,
      updateCurrent,
      workspaceCameraGeometry,
    ],
  );

  const resolveCurrentPose = useCallback(() => {
    const frame = framesRef.current[String(frameIndex)];
    if (!frame) return;
    const labeled =
      frame.faces.find((face) => face.assist?.face_name === "labeled") ??
      frame.faces.find((face) => !face.assist?.generated);
    if (!labeled) {
      setError("Select or draw a face before running rigid autofill.");
      return;
    }
    const pinsByVertex = new Map<number, Point>();
    for (const face of frame.faces) {
      face.vertices.forEach((vertex, index) => {
        if (vertex !== null && face.pinned[index]) {
          pinsByVertex.set(vertex, face.corners[index]);
        }
      });
    }
    void runAssist(
      labeled.corners,
      labeled.id,
      [...pinsByVertex].map(([vertex, xy]) => ({ vertex, xy })),
    );
  }, [frameIndex, runAssist]);

  const finishPolygon = useCallback(
    (points = draftPoints) => {
      if (points.length < 3) {
        setError("A polygon needs at least three points.");
        return;
      }
      const polygon: AnnotationPolygon = {
        id: newAnnotationId(),
        points: points.map((point) => [...point] as Point),
      };
      updateCurrent((frame) => ({
        ...frame,
        polygons: [...frame.polygons, polygon],
      }));
      setDraftPoints([]);
      setSelection({ kind: "polygon", id: polygon.id });
      setTool("select");
      setMessage(`Added polygon on frame ${frameIndex}.`);
    },
    [draftPoints, frameIndex, updateCurrent],
  );

  const addStagePoint = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (assistBusy) return;
    if (dragChangedRef.current) {
      dragChangedRef.current = false;
      return;
    }
    if (tool === "select") {
      setSelection(null);
      return;
    }
    videoRef.current?.pause();
    const point = pointFromEvent(event);
    if (!point) return;
    if (tool === "draw") {
      if (draftPoints.length >= 3 && distance(point, draftPoints[0]) <= 15) {
        finishPolygon();
      } else {
        setDraftPoints((previous) => [...previous, point]);
      }
      return;
    }
    const next = [...draftPoints, point];
    if (next.length === 3 && pnpCapability?.enabled) {
      setDraftPoints(next);
      void runAssist(next);
    } else if (next.length === 4) {
      // Keep the operator's click order. Angle-sorting hand-placed corners
      // renumbers them from the topmost point, which reads as the pose running
      // backwards around the face.
      const face = faceFromOrderedPoints(next, newAnnotationId());
      updateCurrent((frame) => ({ ...frame, faces: [...frame.faces, face] }));
      setDraftPoints([]);
      setSelection({ kind: "face", id: face.id });
      setTool("select");
      setMessage(`Added manual face on frame ${frameIndex}.`);
    } else {
      setDraftPoints(next);
      setSelection(null);
    }
  };

  const movePoint = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (!dragging) return;
    const point = pointFromEvent(event);
    if (!point) return;
    dragChangedRef.current = true;
    updateCurrent(
      (frame) => {
        if (dragging.kind === "polygon-point") {
          return {
            ...frame,
            polygons: frame.polygons.map((polygon) => {
              if (polygon.id !== dragging.id) return polygon;
              const points = polygon.points.map((candidate) => [...candidate] as Point);
              points[dragging.pointIndex] = point;
              return { ...polygon, points };
            }),
          };
        }
        if (dragging.kind === "polygon") {
          return {
            ...frame,
            polygons: frame.polygons.map((polygon) =>
              polygon.id === dragging.id
                ? {
                    ...polygon,
                    points: translatePolygonWithinImage(
                      dragging.original,
                      dragging.start,
                      point,
                      dimensions,
                    ),
                  }
                : polygon,
            ),
          };
        }
        const draggedFace = frame.faces.find((face) => face.id === dragging.id);
        const vertex = draggedFace?.vertices[dragging.pointIndex] ?? null;
        return {
          ...frame,
          faces: frame.faces.map((face) => {
            const corners = face.corners.map((candidate) => [...candidate] as Point) as
              AnnotationFace["corners"];
            const pinned = [...face.pinned] as AnnotationFace["pinned"];
            let changed = false;
            face.vertices.forEach((candidateVertex, index) => {
              if (
                (vertex !== null && candidateVertex === vertex) ||
                (vertex === null && face.id === dragging.id && index === dragging.pointIndex)
              ) {
                corners[index] = point;
                pinned[index] = true;
                changed = true;
              }
            });
            return changed ? { ...face, corners, pinned } : face;
          }),
        };
      },
      false,
    );
  };

  const finishDrag = () => {
    const previous = dragging;
    setDragging(null);
    if (previous?.kind === "face-corner" && dragChangedRef.current) {
      window.clearTimeout(solveTimerRef.current);
      solveTimerRef.current = window.setTimeout(resolveCurrentPose, 150);
    }
  };

  const deleteSelection = useCallback(() => {
    if (!selection) return;
    updateCurrent((frame) => ({
      ...frame,
      faces:
        selection.kind === "face"
          ? frame.faces.filter((face) => face.id !== selection.id)
          : frame.faces,
      polygons:
        selection.kind === "polygon"
          ? frame.polygons.filter((polygon) => polygon.id !== selection.id)
          : frame.polygons,
      wireframe:
        selection.kind === "face" &&
        frame.faces.find((face) => face.id === selection.id)?.assist
          ? null
          : frame.wireframe,
    }));
    setSelection(null);
  }, [selection, updateCurrent]);

  const convertSelectedPolygon = useCallback(() => {
    if (!selectedPolygon) {
      setTool("corners");
      setDraftPoints([]);
      return;
    }
    if (selectedPolygon.points.length !== 4) {
      setError("Only a four-point polygon can be converted to a pose face.");
      return;
    }
    const face = faceFromFourPoints(selectedPolygon.points, newAnnotationId());
    updateCurrent((frame) => ({
      ...frame,
      polygons: frame.polygons.filter((polygon) => polygon.id !== selectedPolygon.id),
      faces: [...frame.faces, face],
    }));
    setSelection({ kind: "face", id: face.id });
    setTool("select");
    setMessage("Converted the selected polygon to four pose keypoints.");
  }, [selectedPolygon, updateCurrent]);

  const clearAutofill = useCallback(() => {
    updateCurrent((frame) => ({
      ...frame,
      faces: frame.faces
        .filter((face) => !face.assist?.generated)
        .map((face) => ({
          ...face,
          origin: face.origin === "pnp-assist" ? "manual" : face.origin,
          pinned: [false, false, false, false],
          assist: undefined,
          vertices: face.assist ? [null, null, null, null] : face.vertices,
        })),
      wireframe: null,
    }));
    setMessage("Cleared generated faces, the wireframe, and all pins.");
  }, [updateCurrent]);

  const toggleVisibility = useCallback(
    (faceId: string, cornerIndex: number) => {
      updateCurrent((frame) => {
        const selected = frame.faces.find((face) => face.id === faceId);
        if (!selected) return frame;
        const vertex = selected.vertices[cornerIndex];
        const nextVisible = !selected.visible[cornerIndex];
        return {
          ...frame,
          faces: frame.faces.map((face) => {
            const visible = [...face.visible] as AnnotationFace["visible"];
            let changed = false;
            face.vertices.forEach((candidateVertex, index) => {
              if (
                (vertex !== null && candidateVertex === vertex) ||
                (vertex === null && face.id === faceId && index === cornerIndex)
              ) {
                visible[index] = nextVisible;
                changed = true;
              }
            });
            return changed ? { ...face, visible } : face;
          }),
        };
      });
    },
    [updateCurrent],
  );

  const copyPrevious = useCallback(() => {
    for (let index = frameIndex - 1; index >= 0; index -= 1) {
      const previous = framesRef.current[String(index)];
      if (previous && (previous.faces.length || previous.polygons.length)) {
        updateCurrent((frame) =>
          appendRepeatedGeometry(frame, previous, () => newAnnotationId()),
        );
        setMessage(
          `Repeated ${previous.faces.length} face(s) and ${previous.polygons.length} polygon(s) from frame ${index}.`,
        );
        return;
      }
    }
    setError("No earlier annotated frame is available to repeat.");
  }, [frameIndex, updateCurrent]);

  const setAlignment = useCallback(
    (alignedLabel: Exclude<AlignmentLabel, null>) => {
      const nextLabel = toggledAlignment(
        (framesRef.current[String(frameIndex)] ?? EMPTY_LABEL_FRAME).aligned_label,
        alignedLabel,
      );
      updateCurrent((frame) => ({ ...frame, aligned_label: nextLabel }));
      if (classifyMode && nextLabel !== null) {
        const next = nextUnclassifiedFrame(frameIndex, maxFrame, framesRef.current);
        if (next !== frameIndex) window.setTimeout(() => seekFrame(next), 0);
      }
    },
    [classifyMode, frameIndex, maxFrame, seekFrame, updateCurrent],
  );

  const scanModelAlignment = useCallback(async () => {
    if (
      !source ||
      source.kind !== "workspace-capture" ||
      !source.capture_id ||
      !predictionCapability?.model_aligned_navigation
    ) {
      setModelAlignedOnly(false);
      setAlignmentScanStatus("unavailable");
      setAlignmentScanError(
        predictionCapability?.reason ??
          "Model-aligned navigation requires a workspace capture and the native camera-tracker-v1 backend.",
      );
      return;
    }

    const request = ++alignmentRequestRef.current;
    setAlignmentScanStatus("scanning");
    setAlignmentScanError("");
    try {
      const result = await scanCaptureLabelAlignment(source.capture_id);
      if (request !== alignmentRequestRef.current) return;
      setModelAlignedFrames(result.aligned_frames);
      setAlignmentConfs(result.alignment_confs);
      setAlignmentScanStatus("ready");
      setModelAlignedOnly(true);
    } catch (reason) {
      if (request !== alignmentRequestRef.current) return;
      setModelAlignedOnly(false);
      setAlignmentScanStatus("unavailable");
      setAlignmentScanError(
        reason instanceof Error
          ? reason.message
          : "The model alignment scan failed.",
      );
    }
  }, [predictionCapability, source]);

  const runPrediction = useCallback(
    async (all: boolean) => {
      if (
        !source ||
        source.kind !== "workspace-capture" ||
        !source.capture_id ||
        !dimensions.width ||
        !dimensions.height
      ) {
        setError("Predictions require a workspace capture with a decoded frame.");
        return;
      }
      if (!predictionCapability?.enabled) {
        setError(
          predictionCapability?.reason ??
            "Prediction requires the native camera-tracker-v1 models or an explicit compatible runner.",
        );
        return;
      }
      if (
        all &&
        !window.confirm(
          "Predict all asks the configured local model to process the whole capture. Continue?",
        )
      ) {
        return;
      }
      setBusy(true);
      setError("");
      try {
        const result = await predictCaptureLabels(source.capture_id, {
          width: dimensions.width,
          height: dimensions.height,
          frame_indices: all ? null : [frameIndex],
          skip_frame_indices: all
            ? Object.entries(framesRef.current)
                .filter(([, frame]) => frameHasAnnotation(frame))
                .map(([index]) => Number(index))
            : undefined,
        });
        pushUndo(framesRef.current);
        setFrames((previous) => {
          const next = { ...previous };
          for (const predictedFrame of result.frames) {
            const key = String(predictedFrame.frame_index);
            const existing = next[key] ?? EMPTY_LABEL_FRAME;
            const predicted = predictedFrame.faces.map(
              (face, index): AnnotationFace => ({
                ...face,
                id: face.id || `model-${predictedFrame.frame_index}-${index}`,
                vertices: face.vertices ?? [null, null, null, null],
                pinned: face.pinned ?? [false, false, false, false],
                origin: "model",
              }),
            );
            next[key] = {
              ...existing,
              faces: [...existing.faces.filter((face) => face.origin !== "model"), ...predicted],
            };
          }
          framesRef.current = next;
          return next;
        });
        setMessage(
          `Loaded local-model predictions for ${result.frames.length} frame(s). Review before export.`,
        );
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "Prediction failed.");
      } finally {
        setBusy(false);
      }
    },
    [dimensions, frameIndex, predictionCapability, pushUndo, source],
  );

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const element = event.target as HTMLElement | null;
      if (element?.matches("input, select, textarea, button")) return;
      const key = event.key.toLowerCase();
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        seekFiltered(event.key === "ArrowRight" ? 1 : -1, event.shiftKey ? 10 : 1);
      } else if ((event.metaKey || event.ctrlKey) && key === "z") {
        event.preventDefault();
        const previous = undoRef.current.pop();
        if (previous) {
          framesRef.current = previous;
          setFrames(previous);
        }
      } else if ((event.metaKey || event.ctrlKey) && key === "s") {
        event.preventDefault();
        void persistAnnotations(true);
      } else if (key === "d") {
        setTool("draw");
        setDraftPoints([]);
        setSelection(null);
      } else if (key === "c") {
        if (selection?.kind === "polygon") convertSelectedPolygon();
        else {
          setTool("corners");
          setDraftPoints([]);
          setSelection(null);
        }
      } else if (key === "s" || key === "v") {
        setTool("select");
        setDraftPoints([]);
      } else if (key === "r") {
        copyPrevious();
      } else if (key === "p") {
        void runPrediction(false);
      } else if (key === "f") {
        if (draftPoints.length === 3) void runAssist(draftPoints);
        else resolveCurrentPose();
      } else if (key === "x") {
        clearAutofill();
      } else if (key === "enter" && tool === "draw") {
        event.preventDefault();
        finishPolygon();
      } else if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        deleteSelection();
      } else if (event.key === "Escape") {
        setDraftPoints([]);
        setSelection(null);
        setTool("select");
      } else if (/^[1-4]$/.test(key) && selectedFace) {
        event.preventDefault();
        toggleVisibility(selectedFace.id, Number(key) - 1);
      } else if ((key === "1" || key === "a") && !selectedFace) {
        setAlignment("aligned");
      } else if ((key === "2" || key === "u") && !selectedFace) {
        setAlignment("unaligned");
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [
    clearAutofill,
    convertSelectedPolygon,
    copyPrevious,
    deleteSelection,
    draftPoints,
    finishPolygon,
    persistAnnotations,
    resolveCurrentPose,
    runAssist,
    runPrediction,
    seekFiltered,
    selectedFace,
    selection,
    setAlignment,
    toggleVisibility,
    tool,
  ]);

  const mayReplaceSource = () =>
    totalAnnotations === 0 ||
    window.confirm(
      "Opening another video replaces the current in-memory view. Autosaved data remains recoverable. Continue?",
    );

  const selectCaptureId = (next: string) => {
    setCaptureId(next);
    if (searchParams.get(CAPTURE_PARAM) === next) return;
    const nextParams = new URLSearchParams(searchParams);
    if (next) nextParams.set(CAPTURE_PARAM, next);
    else nextParams.delete(CAPTURE_PARAM);
    setSearchParams(nextParams, { replace: true });
  };

  const openWorkspaceCapture = async () => {
    const receipt = captures.find((capture) => capture.capture_id === captureId);
    if (!receipt || !mayReplaceSource()) return;
    setBusy(true);
    setError("");
    try {
      const ticket = await createCaptureMediaTicket(receipt.capture_id);
      await installWorkspaceSource(workspaceSource(receipt), ticket.url);
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "The workspace video could not be opened.",
      );
    } finally {
      setBusy(false);
    }
  };

  const openLocalVideo = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !mayReplaceSource()) return;
    // A deliberately opened local file overrides the selected workspace
    // capture, so the dropdown cannot claim a video that is no longer shown.
    setCaptureId("");
    installLocalSource(file, localSource(file));
  };

  const importAnnotations = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (!source) {
      setError("Open the annotation's video before loading its JSON sidecar.");
      return;
    }
    setError("");
    try {
      const document = parseFrameAnnotations(
        await readJsonFile(file, 64 * 1024 * 1024, "Annotation file"),
      );
      const mismatchFields = annotationSourceMismatchFields(document.source, source);
      if (
        mismatchFields.length > 0 &&
        !window.confirm(
          `This annotation file's ${mismatchFields.join(
            " and ",
          )} does not match the open video. Loading it will replace the current in-memory annotations and autosave them under the open video. Continue?`,
        )
      ) {
        setMessage("Annotation import cancelled. Current annotations remain unchanged.");
        return;
      }
      setFrames(draftsFromDocument(document));
      setFps(document.source.fps);
      undoRef.current = [];
      setDraftPoints([]);
      setSelection(null);
      setMessage(
        `Loaded ${document.frames.length} labeled frame(s) from ${file.name}.${
          mismatchFields.length > 0
            ? " Confirmed source mismatch. Review every annotation before editing."
            : ""
        }`,
      );
    } catch (reason) {
      setError(
        reason instanceof LocalContractError
          ? reason.message
          : "The annotation file could not be read.",
      );
    }
  };

  const exportAnnotations = () => {
    const document = buildDocument();
    if (!document || !source) {
      setError("Open a video before exporting annotations.");
      return;
    }
    triggerJsonDownload(
      document,
      `${safeDownloadStem(source.filename, "capture")}.frame-annotations.json`,
    );
    setMessage(`Downloaded ${document.frames.length} labeled frame(s).`);
  };

  const exportDataset = async () => {
    if (source?.kind !== "workspace-capture" || !source.capture_id) {
      setError(
        "Dataset ZIP export requires exact workspace frames. Import this local file first.",
      );
      return;
    }
    setBusy(true);
    try {
      if (!(await persistAnnotations(false))) return;
      const archive = await downloadLabelDatasetExport(source.capture_id);
      triggerBlobDownload(
        archive,
        `${safeDownloadStem(source.filename, "capture")}.label-dataset.zip`,
      );
      setMessage(
        "Downloaded exact images with YOLO pose labels, COCO polygons, and alignment folders. The ZIP uses a convenient within-capture split. Create a separate held-out split for research evaluation.",
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "YOLO pose export failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page local-tool-page label-tool-page">
      <header className="page-header label-page-header">
        <div>
          <p className="eyebrow">Frame annotations</p>
          <h1>Label</h1>
          <p className="page-description">
            Step through a capture, mark visible cube geometry, and classify
            alignment. Changes autosave as you work.
          </p>
        </div>
        <details className="label-help">
          <summary>Keyboard shortcuts</summary>
          <div className="label-help-popover">
            <dl>
              <div>
                <dt><kbd>←</kbd> <kbd>→</kbd></dt>
                <dd>Step frames</dd>
              </div>
              <div>
                <dt><kbd>C</kbd> / <kbd>D</kbd> / <kbd>S</kbd></dt>
                <dd>Corners, polygon, select</dd>
              </div>
              <div>
                <dt><kbd>R</kbd> / <kbd>P</kbd> / <kbd>F</kbd></dt>
                <dd>Repeat, predict, autofill</dd>
              </div>
              <div>
                <dt><kbd>1</kbd> / <kbd>2</kbd></dt>
                <dd>Aligned or unaligned</dd>
              </div>
              <div>
                <dt><kbd>⌘S</kbd></dt>
                <dd>Save now</dd>
              </div>
            </dl>
            <p>
              Output contract: <code>frame-annotations / v1</code>
            </p>
          </div>
        </details>
      </header>

      <section className="label-source-panel" aria-labelledby="label-source-title">
        <div className="label-source-heading">
          <p className="station-code">Source</p>
          <h2 id="label-source-title">Workspace capture</h2>
        </div>
        <div className="label-source-main">
          <div className="label-source-control">
            <label className="sr-only" htmlFor="label-capture-select">
              Workspace capture
            </label>
            <select
              id="label-capture-select"
              value={captureId}
              onChange={(event) => selectCaptureId(event.target.value)}
            >
              {captures.length === 0 && (
                <option value="">
                  {captureListError
                    ? "Workspace captures unavailable"
                    : capturesLoaded
                      ? "No captures yet"
                      : "Loading captures…"}
                </option>
              )}
              {captureGroups.yourRecordings.length > 0 && (
                <optgroup label="Your recordings">
                  {captureGroups.yourRecordings.map((capture) => (
                    <option key={capture.capture_id} value={capture.capture_id}>
                      {capture.original_filename} · {capture.capture_id.slice(0, 8)}
                    </option>
                  ))}
                </optgroup>
              )}
              {captureGroups.publishedDataset.length > 0 && (
                <optgroup label="Published dataset">
                  {captureGroups.publishedDataset.map((capture) => (
                    <option key={capture.capture_id} value={capture.capture_id}>
                      {capture.original_filename} · {capture.capture_id.slice(0, 8)}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            <button
              className="button button-primary"
              type="button"
              disabled={!captureId || busy}
              onClick={() => void openWorkspaceCapture()}
            >
              {busy ? "Working…" : "Open capture"}
            </button>
          </div>
          {(captureListError || capabilitiesError) && (
            <div className="source-workspace-error">
              <p className="tool-message tool-message-error" role="alert">
                {captureListError
                  ? `Workspace captures could not be loaded: ${captureListError}`
                  : `Workbench capabilities could not be loaded: ${capabilitiesError}`}
              </p>
              <button
                className="text-button"
                type="button"
                onClick={() => setWorkspaceLoadAttempt((attempt) => attempt + 1)}
              >
                Retry workspace
              </button>
            </div>
          )}
        </div>
        <details className="label-secondary-source">
          <summary>Other sources</summary>
          <div className="label-secondary-actions">
            <label className="file-action">
              <input
                type="file"
                accept="video/*,.mov,.mp4,.m4v,.webm"
                onChange={openLocalVideo}
              />
              <strong>Open a local video</strong>
              <small>Approximate browser frame stepping</small>
            </label>
            <label className="file-action">
              <input
                type="file"
                accept=".json,application/json"
                onChange={importAnnotations}
              />
              <strong>Load annotation JSON</strong>
              <small>Requires the matching video first</small>
            </label>
          </div>
        </details>
      </section>

      {error && (
        <p className="tool-message tool-message-error" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="tool-message" role="status">
          {message}
        </p>
      )}

      {!source ? (
        <section className="tool-empty-state label-empty-state">
          <div>
            <p className="station-code">Ready for a source</p>
            <strong>No capture open</strong>
            <p>Choose a workspace capture above to label it on the source-video frame clock.</p>
            <Link className="text-button" to="/import">
              Add a video
            </Link>
          </div>
        </section>
      ) : (
        <>
          <section className="label-toolbar" aria-label="Label tools">
            <div className="label-tool-cluster">
              <span className="label-toolset-name">Mode</span>
              <div className="label-tool-group">
                <button
                  aria-keyshortcuts="S"
                  className={tool === "select" ? "tool-active" : ""}
                  type="button"
                  onClick={() => {
                    setTool("select");
                    setDraftPoints([]);
                  }}
                >
                  S · Select
                </button>
                <button
                  aria-keyshortcuts="D"
                  className={tool === "draw" ? "tool-active" : ""}
                  type="button"
                  onClick={() => {
                    setTool("draw");
                    setDraftPoints([]);
                    setSelection(null);
                  }}
                >
                  D · Polygon
                </button>
                <button
                  aria-keyshortcuts="C"
                  className={tool === "corners" ? "tool-active" : ""}
                  type="button"
                  onClick={() => {
                    setTool("corners");
                    setDraftPoints([]);
                    setSelection(null);
                  }}
                >
                  C · Corners
                </button>
              </div>
            </div>
            <div className="label-tool-cluster">
              <span className="label-toolset-name">Assist</span>
              <div className="label-tool-group">
                <button
                  aria-keyshortcuts="F"
                  type="button"
                  disabled={assistBusy}
                  onClick={resolveCurrentPose}
                >
                  F · {assistBusy ? "Solving…" : "Rigid autofill"}
                </button>
                <button aria-keyshortcuts="X" type="button" onClick={clearAutofill}>
                  X · Clear
                </button>
                <button aria-keyshortcuts="R" type="button" onClick={copyPrevious}>
                  R · Repeat
                </button>
              </div>
            </div>
            <div className="label-tool-cluster">
              <span className="label-toolset-name">Model</span>
              <div className="label-tool-group">
                <button
                  aria-keyshortcuts="P"
                  type="button"
                  disabled={
                    !predictionEnabled || busy || alignmentScanStatus === "scanning"
                  }
                  title={
                    predictionCapability?.reason ??
                    "Run the configured camera-tracker-v1 face-pose model"
                  }
                  onClick={() => void runPrediction(false)}
                >
                  P · Predict
                </button>
                <button
                  type="button"
                  disabled={
                    !predictionEnabled || busy || alignmentScanStatus === "scanning"
                  }
                  title={
                    predictionCapability?.reason ??
                    "Run the configured camera-tracker-v1 face-pose model"
                  }
                  onClick={() => void runPrediction(true)}
                >
                  Predict all
                </button>
              </div>
            </div>
            <details className="label-filter-menu">
              <summary>Frame filters</summary>
              <div className="label-filter-group">
                <label>
                  <input
                    type="checkbox"
                    checked={annotatedOnly}
                    onChange={(event) => setAnnotatedOnly(event.target.checked)}
                  />
                  Annotated only
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={alignedOnly}
                    onChange={(event) => setAlignedOnly(event.target.checked)}
                  />
                  Human-aligned only
                </label>
                <label
                  title={
                    alignmentScanError ||
                    "Runs the whole-video model alignment scan only when selected."
                  }
                >
                  <input
                    type="checkbox"
                    checked={modelAlignedOnly}
                    disabled={
                      !modelAlignmentEnabled || alignmentScanStatus === "scanning"
                    }
                    onChange={(event) => {
                      if (!event.target.checked) {
                        setModelAlignedOnly(false);
                      } else if (alignmentScanStatus === "ready") {
                        setModelAlignedOnly(true);
                      } else {
                        void scanModelAlignment();
                      }
                    }}
                  />
                  {alignmentScanStatus === "scanning"
                    ? "Scanning model alignment…"
                    : alignmentScanStatus === "ready"
                      ? `Model-aligned only (${modelAlignedFrames.length})`
                      : "Model-aligned only (scan on demand)"}
                </label>
              </div>
            </details>
          </section>

          <section
            className="label-workspace"
            aria-keyshortcuts="ArrowLeft ArrowRight Shift+ArrowLeft Shift+ArrowRight Control+S Meta+S"
          >
            <div className="label-stage-column">
              <div className="label-stage">
                <video
                  ref={videoRef}
                  src={sourceUrl}
                  muted
                  playsInline
                  preload="auto"
                  onLoadedMetadata={(event) => {
                    window.clearTimeout(videoLoadTimeoutRef.current);
                    const video = event.currentTarget;
                    setDuration(video.duration);
                    setDimensions({ width: video.videoWidth, height: video.videoHeight });
                    video.currentTime = Math.min(video.duration, 0.5 / fps);
                  }}
                  onError={() => {
                    window.clearTimeout(videoLoadTimeoutRef.current);
                    setError(
                      source.kind === "workspace-capture"
                        ? "Workspace video access expired or the stream became unavailable. Open the capture again to issue a fresh read-only media ticket."
                        : "The file could not be opened as a video. This browser could not decode it.",
                    );
                    setAutosaveReady(true);
                    setAutosaveStatus("ready");
                  }}
                />
                {dimensions.width > 0 && (
                  <svg
                    ref={svgRef}
                    className={`label-overlay label-tool-${tool}`}
                    viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    role="img"
                    aria-label={`Cube annotations on frame ${frameIndex}`}
                    onClick={addStagePoint}
                    onDoubleClick={(event) => {
                      if (tool === "draw" && draftPoints.length >= 3) {
                        event.preventDefault();
                        finishPolygon();
                      }
                    }}
                    onPointerMove={(event) => {
                      movePoint(event);
                      setGuidePoint(
                        tool === "select" ? null : pointFromEvent(event),
                      );
                    }}
                    onPointerLeave={() => setGuidePoint(null)}
                    onPointerUp={finishDrag}
                    onPointerCancel={finishDrag}
                  >
                    {current.wireframe &&
                      WIREFRAME_EDGES.map(([start, end]) => (
                        <line
                          key={`${start}-${end}`}
                          className="label-wireframe"
                          x1={current.wireframe![start][0]}
                          y1={current.wireframe![start][1]}
                          x2={current.wireframe![end][0]}
                          y2={current.wireframe![end][1]}
                        />
                      ))}
                    {current.polygons.map((polygon) => (
                      <g key={polygon.id}>
                        <polygon
                          points={polygon.points.map(([x, y]) => `${x},${y}`).join(" ")}
                          className={`label-polygon${
                            selection?.kind === "polygon" && selection.id === polygon.id
                              ? " label-polygon-selected"
                              : ""
                          }`}
                          onPointerDown={(event) => {
                            if (tool !== "select") return;
                            const start = pointFromEvent(event);
                            if (!start) return;
                            event.stopPropagation();
                            pushUndo(framesRef.current);
                            setSelection({ kind: "polygon", id: polygon.id });
                            setDragging({
                              kind: "polygon",
                              id: polygon.id,
                              start,
                              original: polygon.points.map(
                                (point) => [...point] as Point,
                              ),
                            });
                          }}
                          onClick={(event) => {
                            event.stopPropagation();
                            dragChangedRef.current = false;
                            setSelection({ kind: "polygon", id: polygon.id });
                            setTool("select");
                            setDraftPoints([]);
                          }}
                        />
                        {selection?.kind === "polygon" &&
                          selection.id === polygon.id &&
                          polygon.points.map(([x, y], pointIndex) => (
                            <circle
                              key={`${polygon.id}-${pointIndex}`}
                              cx={x}
                              cy={y}
                              r={9 * uiScale}
                              className="label-polygon-point"
                              onPointerDown={(event) => {
                                event.stopPropagation();
                                pushUndo(framesRef.current);
                                setDragging({
                                  kind: "polygon-point",
                                  id: polygon.id,
                                  pointIndex,
                                });
                              }}
                            />
                          ))}
                      </g>
                    ))}
                    {current.faces.map((face) => (
                      <g key={face.id}>
                        <polygon
                          points={face.corners.map(([x, y]) => `${x},${y}`).join(" ")}
                          className={`label-face${
                            selection?.kind === "face" && selection.id === face.id
                              ? " label-face-selected"
                              : ""
                          }${face.assist?.generated ? " label-face-generated" : ""}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            setSelection({ kind: "face", id: face.id });
                            setTool("select");
                            setDraftPoints([]);
                          }}
                        />
                        {face.corners.map(([x, y], cornerIndex) => (
                          <g key={`${face.id}-${cornerIndex}`}>
                            <circle
                              cx={x}
                              cy={y}
                              r={
                                (selection?.kind === "face" && selection.id === face.id
                                  ? 14
                                  : 12) * uiScale
                              }
                              className={`label-corner label-corner-${cornerIndex + 1}${
                                face.visible[cornerIndex] ? "" : " label-corner-hidden"
                              }${face.pinned[cornerIndex] ? " label-corner-pinned" : ""}`}
                              onPointerDown={(event) => {
                                event.stopPropagation();
                                pushUndo(framesRef.current);
                                setSelection({ kind: "face", id: face.id });
                                setDraftPoints([]);
                                setDragging({
                                  kind: "face-corner",
                                  id: face.id,
                                  pointIndex: cornerIndex,
                                });
                              }}
                              onContextMenu={(event) => {
                                event.preventDefault();
                                event.stopPropagation();
                                toggleVisibility(face.id, cornerIndex);
                              }}
                            />
                            <text
                              x={x}
                              y={y}
                              className="label-corner-number"
                              style={{
                                fontSize: `${14 * uiScale}px`,
                                strokeWidth: `${3 * uiScale}px`,
                              }}
                            >
                              {face.vertices[cornerIndex] === null
                                ? cornerIndex + 1
                                : `v${face.vertices[cornerIndex]}`}
                            </text>
                          </g>
                        ))}
                      </g>
                    ))}
                    {draftPoints.length > 0 && (
                      <g className="label-draft">
                        <polyline
                          points={draftPoints.map(([x, y]) => `${x},${y}`).join(" ")}
                        />
                        {guidePoint && (
                          <line
                            className="label-draft-guide"
                            x1={draftPoints[draftPoints.length - 1][0]}
                            y1={draftPoints[draftPoints.length - 1][1]}
                            x2={guidePoint[0]}
                            y2={guidePoint[1]}
                          />
                        )}
                        {draftPoints.map(([x, y], index) => (
                          <g key={`${x}-${y}-${index}`}>
                            <circle cx={x} cy={y} r={9 * uiScale} />
                            <text
                              x={x}
                              y={y}
                              style={{
                                fontSize: `${12 * uiScale}px`,
                                strokeWidth: `${3 * uiScale}px`,
                              }}
                            >
                              {index + 1}
                            </text>
                          </g>
                        ))}
                      </g>
                    )}
                  </svg>
                )}
                <span className="stage-frame-stamp">
                  F{String(frameIndex).padStart(6, "0")}
                </span>
                <span className="stage-source-stamp">
                  {source.kind === "workspace-capture"
                    ? "workspace source video"
                    : "browser-time preview"}
                </span>
                {current.aligned_label && (
                  <span className={`stage-alignment stage-${current.aligned_label}`}>
                    {current.aligned_label}
                  </span>
                )}
              </div>

              <div className="frame-transport">
                <button type="button" onClick={() => seekFiltered(-1, 10)}>
                  −10
                </button>
                <button type="button" onClick={() => seekFiltered(-1, 1)}>
                  −1
                </button>
                <button
                  className="transport-play"
                  type="button"
                  onClick={() => {
                    const video = videoRef.current;
                    if (!video) return;
                    if (video.paused) void video.play();
                    else video.pause();
                  }}
                >
                  Play / pause
                </button>
                <button type="button" onClick={() => seekFiltered(1, 1)}>
                  +1
                </button>
                <button type="button" onClick={() => seekFiltered(1, 10)}>
                  +10
                </button>
                <span>
                  frame <strong>{frameIndex}</strong> / {maxFrame}
                </span>
              </div>
              <input
                className="frame-scrubber"
                type="range"
                min={0}
                max={maxFrame}
                value={Math.min(frameIndex, maxFrame)}
                aria-label="Current video frame"
                onChange={(event) => seekFrame(Number(event.target.value))}
              />
              {frameRangeUnknown && (
                <p className="tool-message" role="status">
                  This capture does not report a frame count, so the scrubber
                  is limited to the first frame. Re-import the capture or
                  probe the video again to recover the full range.
                </p>
              )}
            </div>

            <aside className="label-inspector">
              <div className="inspector-section">
                <p className="station-code">Frame receipt</p>
                <dl className="tool-metrics">
                  <div>
                    <dt>Frame</dt>
                    <dd>{frameIndex}</dd>
                  </div>
                  <div>
                    <dt>Time</dt>
                    <dd>{(frameIndex / fps).toFixed(4)} s</dd>
                  </div>
                  <div>
                    <dt>Faces / polys</dt>
                    <dd>
                      {current.faces.length} / {current.polygons.length}
                    </dd>
                  </div>
                  <div>
                    <dt>Autosave</dt>
                    <dd>{autosaveStatus}</dd>
                  </div>
                </dl>
                <details className="label-inspector-disclosure">
                  <summary>Source and frame rate</summary>
                  <div>
                    <label className="fps-field" htmlFor="label-fps">
                      <span>Frame rate used for indexing</span>
                      <input
                        id="label-fps"
                        type="number"
                        min={1}
                        max={1000}
                        step="0.001"
                        value={fps}
                        onChange={(event) => {
                          const next = Number(event.target.value);
                          if (Number.isFinite(next) && next > 0) setFps(next);
                        }}
                      />
                    </label>
                    <p className="tool-fine-print">
                      {source.kind === "workspace-capture"
                        ? "The viewport seeks the source video on the receipt frame clock. Dataset export extracts the annotated frame indices exactly from the workspace source."
                        : "Browser-time stepping is approximate. Reopen this same file to restore browser autosave. Import it for exact YOLO frame export."}
                    </p>
                  </div>
                </details>
              </div>

              <div className="inspector-section">
                <p className="station-code">
                  {tool === "draw" ? "Polygon" : "Rigid face"}
                </p>
                <p className="instruction-copy">
                  {tool === "draw"
                    ? "Click around a face. Click the first point, double-click, or press Enter to close."
                    : "Three consecutive corner clicks trigger perspective completion and rigid cube autofill. If PnP is unavailable, click a fourth corner manually."}
                </p>
                {tool !== "draw" && (
                  <details className="label-inspector-disclosure">
                    <summary>Pose assist settings</summary>
                    <div>
                      <label className="fps-field" htmlFor="assist-camera">
                        <span>Camera model</span>
                        <select
                          id="assist-camera"
                          value={assistCamera}
                          onChange={(event) =>
                            setAssistCamera(
                              event.target.value as "auto" | "generic" | "custom",
                            )
                          }
                        >
                          <option value="auto">
                            {workspaceCameraGeometry?.intrinsics
                              ? "Capture receipt intrinsics"
                              : "Automatic generic prior"}
                          </option>
                          <option value="generic">
                            Generic prior (focal = image size)
                          </option>
                          <option value="custom">Custom focal length</option>
                        </select>
                      </label>
                      {assistCamera === "custom" && (
                        <label className="fps-field" htmlFor="assist-focal">
                          <span>Focal length in pixels</span>
                          <input
                            id="assist-focal"
                            type="number"
                            min={1}
                            step="1"
                            placeholder="e.g. 3000"
                            value={assistFocal}
                            onChange={(event) => setAssistFocal(event.target.value)}
                          />
                        </label>
                      )}
                      <label className="fps-field" htmlFor="assist-shrink">
                        <span>Corner inset</span>
                        <input
                          id="assist-shrink"
                          type="number"
                          min={0.1}
                          max={1.5}
                          step="0.01"
                          value={assistShrink}
                          onChange={(event) => setAssistShrink(event.target.value)}
                        />
                      </label>
                      <p className="tool-fine-print">
                        Receipt intrinsics are used for workspace captures.
                        Otherwise choose a generic prior or enter the camera focal
                        length. An inset below 1.0 fits sticker corners inside the
                        mechanical face edge.
                      </p>
                      <p className="tool-fine-print">
                        PnP:{" "}
                        {pnpCapability?.enabled ? "available on local CPU" : "disabled"}.
                        Prediction:{" "}
                        {predictionCapability?.enabled
                          ? predictionCapability.backend === "camera-tracker-v1"
                            ? "camera-tracker-v1"
                            : "external compatibility runner"
                          : "disabled"}.
                        Model alignment:{" "}
                        {alignmentScanStatus === "ready"
                          ? `${modelAlignedFrames.length} / ${alignmentConfs.length} frames`
                          : alignmentScanStatus}
                        {typeof currentModelAlignment === "number" &&
                        Number.isFinite(currentModelAlignment)
                          ? `. Current confidence: ${currentModelAlignment.toFixed(3)}`
                          : ""}
                      </p>
                    </div>
                  </details>
                )}
                <div className="corner-progress" aria-label="Point entry progress">
                  {CORNER_NAMES.map((name, index) => (
                    <span className={index < draftPoints.length ? "corner-done" : ""} key={name}>
                      {name}
                    </span>
                  ))}
                </div>
                <div className="inspector-actions">
                  <button
                    type="button"
                    disabled={draftPoints.length === 0}
                    onClick={() => setDraftPoints((points) => points.slice(0, -1))}
                  >
                    Undo point
                  </button>
                  <button
                    type="button"
                    disabled={!selectedPolygon}
                    onClick={convertSelectedPolygon}
                  >
                    Convert polygon
                  </button>
                </div>
              </div>

              <div className="inspector-section">
                <p className="station-code">Alignment class</p>
                <label className="classify-mode-toggle">
                  <input
                    type="checkbox"
                    checked={classifyMode}
                    onChange={(event) => setClassifyMode(event.target.checked)}
                  />
                  Classify mode. Advance after each selected label.
                </label>
                <div className="alignment-controls">
                  <button
                    aria-keyshortcuts="1 A"
                    className={current.aligned_label === "aligned" ? "choice-active" : ""}
                    type="button"
                    onClick={() => setAlignment("aligned")}
                  >
                    1 / A · Aligned
                  </button>
                  <button
                    aria-keyshortcuts="2 U"
                    className={current.aligned_label === "unaligned" ? "choice-active" : ""}
                    type="button"
                    onClick={() => setAlignment("unaligned")}
                  >
                    2 / U · Unaligned
                  </button>
                </div>
              </div>

              <div className="inspector-section">
                <p className="station-code">Annotations</p>
                {current.faces.length === 0 && current.polygons.length === 0 ? (
                  <p className="tool-fine-print">
                    No shapes on this frame.
                  </p>
                ) : (
                  <>
                    <p className="tool-fine-print">
                      Select a shape, then use the arrow keys to move it one
                      pixel. Hold Shift for ten pixels.
                    </p>
                    <ul className="label-annotation-list">
                      {current.faces.map((face, index) => (
                        <li key={face.id}>
                          <button
                            type="button"
                            aria-pressed={
                              selection?.kind === "face" &&
                              selection.id === face.id
                            }
                            onClick={() => {
                              setSelection({ kind: "face", id: face.id });
                              setTool("select");
                              setDraftPoints([]);
                            }}
                            onKeyDown={(event) => {
                              if (!event.key.startsWith("Arrow")) return;
                              event.preventDefault();
                              event.stopPropagation();
                              const step = event.shiftKey ? 10 : 1;
                              moveAnnotation(
                                { kind: "face", id: face.id },
                                event.key === "ArrowLeft"
                                  ? -step
                                  : event.key === "ArrowRight"
                                    ? step
                                    : 0,
                                event.key === "ArrowUp"
                                  ? -step
                                  : event.key === "ArrowDown"
                                    ? step
                                    : 0,
                              );
                            }}
                          >
                            Face {index + 1}
                            <span>
                              {face.assist?.generated ? "generated" : "manual"}
                            </span>
                          </button>
                        </li>
                      ))}
                      {current.polygons.map((polygon, index) => (
                        <li key={polygon.id}>
                          <button
                            type="button"
                            aria-pressed={
                              selection?.kind === "polygon" &&
                              selection.id === polygon.id
                            }
                            onClick={() => {
                              setSelection({ kind: "polygon", id: polygon.id });
                              setTool("select");
                              setDraftPoints([]);
                            }}
                            onKeyDown={(event) => {
                              if (!event.key.startsWith("Arrow")) return;
                              event.preventDefault();
                              event.stopPropagation();
                              const step = event.shiftKey ? 10 : 1;
                              moveAnnotation(
                                { kind: "polygon", id: polygon.id },
                                event.key === "ArrowLeft"
                                  ? -step
                                  : event.key === "ArrowRight"
                                    ? step
                                    : 0,
                                event.key === "ArrowUp"
                                  ? -step
                                  : event.key === "ArrowDown"
                                    ? step
                                    : 0,
                              );
                            }}
                          >
                            Polygon {index + 1}
                            <span>{polygon.points.length} points</span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>

              {(selectedFace || selectedPolygon) && (
                <div className="inspector-section selected-face-panel">
                  <p className="station-code">
                    Selected {selectedFace ? "face" : "polygon"}
                  </p>
                  {selectedFace && (
                    <>
                      <div className="visibility-grid">
                        {selectedFace.visible.map((visible, index) => (
                          <button
                            type="button"
                            className={visible ? "visibility-on" : ""}
                            key={index}
                            onClick={() => toggleVisibility(selectedFace.id, index)}
                          >
                            {index + 1} · {visible ? "visible" : "occluded"}
                          </button>
                        ))}
                      </div>
                      <p className="tool-fine-print">
                        Dragging a numbered cube vertex pins and moves every shared corner.
                        Release to re-solve the unpinned pose.
                      </p>
                    </>
                  )}
                  <button
                    className="danger-text-button"
                    type="button"
                    onClick={deleteSelection}
                  >
                    Delete selected {selectedFace ? "face" : "polygon"}
                  </button>
                </div>
              )}

              <div className="inspector-section export-panel">
                <p className="station-code">Portable outputs</p>
                <p>
                  Labeled frames: {labeledFrames}. Shapes: {totalAnnotations}.
                  JSON and autosave retain polygons, pose vertices, pins, and
                  visibility.
                </p>
                <div className="export-actions">
                  <button
                    aria-keyshortcuts="Control+S Meta+S"
                    type="button"
                    onClick={() => void persistAnnotations(true)}
                  >
                    Ctrl/Cmd+S · Save
                  </button>
                  <button type="button" onClick={exportAnnotations}>
                    Download JSON
                  </button>
                  <button
                    className="button button-primary"
                    type="button"
                    disabled={source.kind !== "workspace-capture" || busy}
                    onClick={() => void exportDataset()}
                  >
                    Download dataset ZIP
                  </button>
                </div>
              </div>
            </aside>
          </section>
        </>
      )}
    </div>
  );
}
