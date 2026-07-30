import assert from "node:assert/strict";
import test from "node:test";

import {
  EMPTY_LABEL_FRAME,
  appendRepeatedGeometry,
  annotationSourceMismatchFields,
  buildLabelAssistRequest,
  buildFrameAnnotationsDocument,
  cameraMatrixForLabelFrame,
  faceFromFourPoints,
  nextFilteredFrame,
  orderClockwise,
  toggledAlignment,
  translatePolygonWithinImage,
} from "../src/labelTools.ts";

test("workspace PnP composes native reference scaling with zero display rotation", () => {
  const intrinsics = {
    matrix: [
      [1200, 0, 960],
      [0, 1180, 540],
      [0, 0, 1],
    ],
    ref_w: 1920,
    ref_h: 1080,
  };
  const geometry = {
    intrinsics,
    encodedWidth: 1920,
    encodedHeight: 1080,
    rotationDegrees: 0,
  };
  const dimensions = { width: 960, height: 540 };
  assert.deepEqual(cameraMatrixForLabelFrame(geometry, dimensions), [
    [600, 0, 480],
    [0, 590, 270],
    [0, 0, 1],
  ]);
  assert.deepEqual(
    buildLabelAssistRequest([[1, 1], [2, 1], [2, 2]], dimensions, [], geometry),
    {
      labeled_corners: [[1, 1], [2, 1], [2, 2]],
      width: 960,
      height: 540,
      pins: [],
      K: [
        [600, 0, 480],
        [0, 590, 270],
        [0, 0, 1],
      ],
    },
  );
});

test("workspace PnP has exact golden camera matrices for every cardinal rotation", () => {
  const geometry = {
    intrinsics: {
      matrix: [
        [40, 0, 25],
        [0, 50, 20],
        [0, 0, 1],
      ],
      ref_w: 100,
      ref_h: 60,
    },
    encodedWidth: 200,
    encodedHeight: 120,
    rotationDegrees: 0,
  };
  const goldens = [
    {
      rotationDegrees: 0,
      dimensions: { width: 200, height: 120 },
      expected: [
        [80, 0, 50],
        [0, 100, 40],
        [0, 0, 1],
      ],
    },
    {
      rotationDegrees: 90,
      dimensions: { width: 120, height: 200 },
      expected: [
        [100, 0, 79],
        [0, 80, 50],
        [0, 0, 1],
      ],
    },
    {
      rotationDegrees: 180,
      dimensions: { width: 200, height: 120 },
      expected: [
        [80, 0, 149],
        [0, 100, 79],
        [0, 0, 1],
      ],
    },
    {
      rotationDegrees: 270,
      dimensions: { width: 120, height: 200 },
      expected: [
        [100, 0, 40],
        [0, 80, 149],
        [0, 0, 1],
      ],
    },
  ];

  for (const golden of goldens) {
    assert.deepEqual(
      cameraMatrixForLabelFrame(
        { ...geometry, rotationDegrees: golden.rotationDegrees },
        golden.dimensions,
      ),
      golden.expected,
    );
  }
});

test("workspace PnP camera mapping is differential against the reference camera helpers", () => {
  const privateComposition = (geometry, dimensions) => {
    const matrix = geometry.intrinsics.matrix;
    const nativeWidth = geometry.encodedWidth;
    const nativeHeight = geometry.encodedHeight;
    const sx = nativeWidth / geometry.intrinsics.ref_w;
    const sy = nativeHeight / geometry.intrinsics.ref_h;
    const fx = matrix[0][0] * sx;
    const fy = matrix[1][1] * sy;
    const cx = matrix[0][2] * sx;
    const cy = matrix[1][2] * sy;
    let rotatedWidth = nativeWidth;
    let rotatedHeight = nativeHeight;
    let rfx = fx;
    let rfy = fy;
    let rcx = cx;
    let rcy = cy;
    if (geometry.rotationDegrees === 90) {
      rotatedWidth = nativeHeight;
      rotatedHeight = nativeWidth;
      rfx = fy;
      rfy = fx;
      rcx = nativeHeight - 1 - cy;
      rcy = cx;
    } else if (geometry.rotationDegrees === 180) {
      rcx = nativeWidth - 1 - cx;
      rcy = nativeHeight - 1 - cy;
    } else if (geometry.rotationDegrees === 270) {
      rotatedWidth = nativeHeight;
      rotatedHeight = nativeWidth;
      rfx = fy;
      rfy = fx;
      rcx = cy;
      rcy = nativeWidth - 1 - cx;
    }
    const displayScaleX = dimensions.width / rotatedWidth;
    const displayScaleY = dimensions.height / rotatedHeight;
    return [
      [rfx * displayScaleX, 0, rcx * displayScaleX],
      [0, rfy * displayScaleY, rcy * displayScaleY],
      [0, 0, 1],
    ];
  };

  for (const rotationDegrees of [0, 90, 180, 270]) {
    for (let index = 1; index <= 25; index += 1) {
      const refWidth = 1000 + index * 7;
      const refHeight = 700 + index * 5;
      const encodedWidth = 1800 + index * 11;
      const encodedHeight = 1200 + index * 13;
      const geometry = {
        intrinsics: {
          matrix: [
            [850 + index * 3, 0, refWidth * (0.42 + index / 1000)],
            [0, 820 + index * 2, refHeight * (0.45 - index / 2000)],
            [0, 0, 1],
          ],
          ref_w: refWidth,
          ref_h: refHeight,
        },
        encodedWidth,
        encodedHeight,
        rotationDegrees,
      };
      const dimensions =
        rotationDegrees === 90 || rotationDegrees === 270
          ? { width: 600 + index, height: 900 + index * 2 }
          : { width: 900 + index * 2, height: 600 + index };
      assert.deepEqual(
        cameraMatrixForLabelFrame(geometry, dimensions),
        privateComposition(geometry, dimensions),
      );
    }
  }
});

test("workspace PnP requests preserve the centered-prior fallback", () => {
  const dimensions = { width: 1920, height: 1080 };
  const labeledCorners = [[1, 1], [2, 1], [2, 2]];
  const validIntrinsics = {
    matrix: [
      [1200, 0, 960],
      [0, 1180, 540],
      [0, 0, 1],
    ],
    ref_w: 1920,
    ref_h: 1080,
  };
  const invalidGeometry = [
    null,
    undefined,
    {
      intrinsics: { matrix: [[1]], ref_w: 1920, ref_h: 1080 },
      encodedWidth: 1920,
      encodedHeight: 1080,
      rotationDegrees: 0,
    },
    {
      intrinsics: validIntrinsics,
      encodedWidth: null,
      encodedHeight: 1080,
      rotationDegrees: 0,
    },
    {
      intrinsics: validIntrinsics,
      encodedWidth: 1920,
      encodedHeight: 1080,
      rotationDegrees: 45,
    },
    {
      intrinsics: {
        ...validIntrinsics,
        matrix: [
          [1200, 0, 2500],
          [0, 1180, 540],
          [0, 0, 1],
        ],
      },
      encodedWidth: 1920,
      encodedHeight: 1080,
      rotationDegrees: 0,
    },
    {
      intrinsics: { ...validIntrinsics, distortion: [] },
      encodedWidth: 1920,
      encodedHeight: 1080,
      rotationDegrees: 0,
    },
  ];
  for (const geometry of invalidGeometry) {
    const request = buildLabelAssistRequest(
      labeledCorners,
      dimensions,
      [],
      geometry,
    );
    assert.equal(Object.hasOwn(request, "K"), false);
  }
});

test("polygon conversion creates a clockwise four-corner face with editable metadata", () => {
  const face = faceFromFourPoints(
    [
      [90, 90],
      [10, 10],
      [10, 90],
      [90, 10],
    ],
    "face-1",
  );
  assert.deepEqual(face.corners[0], [10, 10]);
  assert.deepEqual(face.visible, [true, true, true, true]);
  assert.deepEqual(face.vertices, [null, null, null, null]);
  assert.deepEqual(face.pinned, [false, false, false, false]);
});

test("repeat matches the reference AnnotateView append behavior", () => {
  // Literal, shape-only adaptation of AnnotateView.tsx::handleRepeat from
  // the reference implementation.
  function privateRepeat(current, source, createId) {
    const copiedPolygons = source.polygons.map((polygon) => ({
      id: createId(),
      points: polygon.points.map(([x, y]) => [x, y]),
    }));
    const copiedKeypoints = source.keypoints.map((keypoints) => ({
      id: createId(),
      corners: keypoints.corners.map((corner) => ({ ...corner })),
    }));
    return {
      polygons: [...current.polygons, ...copiedPolygons],
      keypoints: [...current.keypoints, ...copiedKeypoints],
    };
  }

  const currentFace = faceFromFourPoints(
    [
      [1, 1],
      [2, 1],
      [2, 2],
      [1, 2],
    ],
    "current-face",
  );
  const sourceFace = {
    ...faceFromFourPoints(
      [
        [10, 10],
        [20, 10],
        [20, 20],
        [10, 20],
      ],
      "source-face",
      "pnp-assist",
    ),
    visible: [true, false, true, false],
    vertices: [0, 1, 2, 3],
    pinned: [true, false, true, false],
    confidence: 0.95,
    assist: { face_name: "labeled", generated: true },
  };
  const current = {
    aligned_label: "aligned",
    faces: [currentFace],
    polygons: [{ id: "current-polygon", points: [[3, 3], [4, 4]] }],
    wireframe: [[1, 1], [2, 2]],
  };
  const source = {
    aligned_label: "unaligned",
    faces: [sourceFace],
    polygons: [{ id: "source-polygon", points: [[30, 30], [40, 40]] }],
    wireframe: [[10, 10], [20, 20]],
  };
  const ids = ["repeat-polygon", "repeat-face"];
  const repeated = appendRepeatedGeometry(current, source, () => ids.shift());

  const privateIds = ["repeat-polygon", "repeat-face"];
  const privateResult = privateRepeat(
    {
      polygons: current.polygons,
      keypoints: current.faces.map((face) => ({
        id: face.id,
        corners: face.corners.map(([x, y], index) => ({
          x,
          y,
          visible: face.visible[index],
        })),
      })),
    },
    {
      polygons: source.polygons,
      keypoints: source.faces.map((face) => ({
        id: face.id,
        corners: face.corners.map(([x, y], index) => ({
          x,
          y,
          visible: face.visible[index],
        })),
      })),
    },
    () => privateIds.shift(),
  );
  const publicShapeProjection = {
    polygons: repeated.polygons,
    keypoints: repeated.faces.map((face) => ({
      id: face.id,
      corners: face.corners.map(([x, y], index) => ({
        x,
        y,
        visible: face.visible[index],
      })),
    })),
  };

  assert.deepEqual(publicShapeProjection, privateResult);
  assert.equal(repeated.aligned_label, "aligned");
  assert.deepEqual(repeated.wireframe, current.wireframe);
  assert.deepEqual(repeated.faces[1], {
    id: "repeat-face",
    corners: sourceFace.corners,
    visible: sourceFace.visible,
    vertices: [null, null, null, null],
    pinned: [false, false, false, false],
    origin: "manual",
  });
  assert.notEqual(repeated.faces[1].corners, sourceFace.corners);
  assert.notEqual(repeated.polygons[1].points, source.polygons[0].points);
});

test("rotated-quad ordering matches AnnotateView's literal topmost-start rule", () => {
  const points = [
    [80, 10],
    [120, 60],
    [50, 110],
    [5, 40],
  ];
  const center = [
    points.reduce((sum, point) => sum + point[0], 0) / points.length,
    points.reduce((sum, point) => sum + point[1], 0) / points.length,
  ];
  const privateOrdered = [...points].sort(
    (left, right) =>
      Math.atan2(left[1] - center[1], left[0] - center[0]) -
      Math.atan2(right[1] - center[1], right[0] - center[0]),
  );
  let privateStart = 0;
  for (let index = 1; index < privateOrdered.length; index += 1) {
    if (privateOrdered[index][1] < privateOrdered[privateStart][1]) {
      privateStart = index;
    }
  }
  const privateResult = [
    ...privateOrdered.slice(privateStart),
    ...privateOrdered.slice(0, privateStart),
  ];

  assert.deepEqual(orderClockwise(points), privateResult);
  assert.deepEqual(orderClockwise(points)[0], [80, 10]);
  assert.deepEqual(
    points.reduce(
      (best, point) =>
        point[0] + point[1] < best[0] + best[1] ? point : best,
      points[0],
    ),
    [5, 40],
  );
});

test("filtered arrow navigation skips frames outside the active filters", () => {
  const frames = {
    "2": { ...structuredClone(EMPTY_LABEL_FRAME), aligned_label: "unaligned" },
    "4": { ...structuredClone(EMPTY_LABEL_FRAME), aligned_label: "aligned" },
    "7": {
      ...structuredClone(EMPTY_LABEL_FRAME),
      aligned_label: "aligned",
      faces: [
        faceFromFourPoints(
          [
            [1, 1],
            [2, 1],
            [2, 2],
            [1, 2],
          ],
          "face",
        ),
      ],
    },
  };
  assert.equal(
    nextFilteredFrame(0, 1, 1, 9, frames, { annotated: false, aligned: true }),
    4,
  );
  assert.equal(
    nextFilteredFrame(0, 1, 1, 9, frames, { annotated: true, aligned: true }),
    4,
  );
  assert.equal(
    nextFilteredFrame(4, 1, 1, 9, frames, { annotated: true, aligned: true }),
    7,
  );
  assert.equal(
    nextFilteredFrame(0, 1, 1, 9, frames, {
      annotated: false,
      aligned: false,
      modelAligned: new Set([2, 7]),
    }),
    2,
  );
  assert.equal(
    nextFilteredFrame(0, 1, 1, 9, frames, {
      annotated: true,
      aligned: false,
      modelAligned: new Set([2, 7]),
    }),
    2,
  );
  assert.equal(
    nextFilteredFrame(2, 1, 1, 9, frames, {
      annotated: true,
      aligned: true,
      modelAligned: new Set([2, 7]),
    }),
    7,
  );
});

test("portable document and YOLO pose labels preserve visibility", () => {
  const face = faceFromFourPoints(
    [
      [10, 20],
      [30, 20],
      [30, 40],
      [10, 40],
    ],
    "face",
  );
  face.visible[2] = false;
  const frames = {
    "5": {
      ...structuredClone(EMPTY_LABEL_FRAME),
      faces: [face],
    },
  };
  const document = buildFrameAnnotationsDocument(
    {
      kind: "workspace-capture",
      capture_id: "a".repeat(32),
      filename: "solve.mp4",
      sha256: "b".repeat(64),
      fps: 120,
      frame_count: 10,
    },
    frames,
    { width: 100, height: 100 },
    120,
    9,
  );
  assert.equal(document.frames[0].frame_index, 5);
});

test("annotation import requires confirmation for filename or SHA-256 receipt drift", () => {
  const active = {
    kind: "workspace-capture",
    capture_id: "a".repeat(32),
    filename: "solve.mp4",
    sha256: "b".repeat(64),
    fps: 120,
    frame_count: 240,
  };

  assert.deepEqual(annotationSourceMismatchFields(structuredClone(active), active), []);
  assert.deepEqual(
    annotationSourceMismatchFields({ ...active, filename: "other.mp4" }, active),
    ["filename"],
  );
  assert.deepEqual(
    annotationSourceMismatchFields({ ...active, sha256: "c".repeat(64) }, active),
    ["SHA-256 receipt"],
  );
  assert.deepEqual(
    annotationSourceMismatchFields(
      { ...active, filename: "other.mp4", sha256: null },
      active,
    ),
    ["filename", "SHA-256 receipt"],
  );
});

test("classification toggles and polygon translation stays within the image", () => {
  assert.equal(toggledAlignment("aligned", "aligned"), null);
  assert.equal(toggledAlignment(null, "unaligned"), "unaligned");
  assert.deepEqual(
    translatePolygonWithinImage(
      [
        [10, 10],
        [30, 10],
        [30, 30],
      ],
      [20, 20],
      [-100, -100],
      { width: 100, height: 100 },
    ),
    [
      [0, 0],
      [20, 0],
      [20, 20],
    ],
  );
});
