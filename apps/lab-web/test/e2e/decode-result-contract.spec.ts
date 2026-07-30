import { expect, test } from "@playwright/test";

import { parseDecodeResultDocument } from "../../src/decodeRunContracts";
import { cubeTrajectory } from "../../src/lib/cubePerms";

const RECORDING_ID = "a".repeat(32);
const VIDEO_SHA = "b".repeat(64);
const CALIBRATION_SHA = "c".repeat(64);

const TEST_CUBE_COLORS = ["white", "yellow", "red", "orange", "blue", "green"];
const TEST_FACE_SLICES = [
  ["up", 0],
  ["right", 9],
  ["front", 18],
  ["down", 27],
  ["left", 36],
  ["back", 45],
] as const;

function fixtureCubeState(moves: string[]) {
  const state = cubeTrajectory(moves).at(-1)!;
  return Object.fromEntries(
    TEST_FACE_SLICES.map(([face, offset]) => [
      face,
      Array.from(
        state.slice(offset, offset + 9),
        (value) => TEST_CUBE_COLORS[value],
      ),
    ]),
  );
}

function validResult(): any {
  return {
    schema: "cubed-core/decode-result",
    schema_version: 1,
    recording_id: RECORDING_ID,
    status: "completed",
    profile: "local_camera_v1",
    config: {
      name: "cubed-core-local-camera-v1",
      cfg_hash: "1852738634",
      cfg_hash_algorithm: "posix-cksum",
      cfg_hash_input: "--profile local_camera_v1",
      extras_hash: null,
    },
    inputs: [
      { id: "video", sha256: VIDEO_SHA },
      { id: "color-calibration-v1", sha256: CALIBRATION_SHA },
    ],
    moves: ["R", "U'", "F2"],
    endpoint: { solved_reached: true },
    evaluation: null,
    provenance: {
      runtime_version: "cubed-core-test",
      implementation_id: "canonical-test-runner",
      implementation_sha256: "d".repeat(64),
      python_version: "3.12.4",
      numpy_version: "2.1.0",
      finished_at: "2026-07-29T12:00:00Z",
    },
    workstation: {
      schema: "cubed-core/decode-workstation-v1",
      schema_version: 1,
      video: {
        sha256: VIDEO_SHA,
        bytes: 1_024,
        fps: 120,
        frame_count: 120,
        width: 1_080,
        height: 1_920,
        encoded: {
          fps: 120,
          frame_count: 120,
          width: 1_920,
          height: 1_080,
        },
      },
      initialization: { scramble: "R U R'" },
      window: [0, 119],
      warnings: [
        {
          code: "shared-calibration",
          message: "Shared calibration is specific to its recording cohort.",
        },
      ],
      frames: {
        "0": {
          motion: 0.1,
          aligned: 0.9,
          aligned_streak: 3,
          face_count: 1,
          gated: false,
          faces: [
            {
              corners: [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
              ],
              confidence: 0.9,
              keypoint_confidence: [0.9, 0.9, 0.9, 0.9],
            },
          ],
          reads: [
            {
              slot: "up",
              lab: Array.from({ length: 9 }, () => [100, 120, 130]),
              confidence: Array(9).fill(0.8),
              valid_pixels: Array(9).fill(100),
              total_pixels: Array(9).fill(120),
              used_fallback: Array(9).fill(false),
              relative_area: 1,
              corners: [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
              ],
            },
          ],
        },
      },
      events: [10, 20, 30],
      trellis: {
        spans: [
          {
            f0: 8,
            f1: 12,
            event: 10,
            top: [{ path: "R", orientation: "UF", score: 1.5 }],
          },
        ],
        bridge: ["R", "U'", "F2"],
      },
      sequence: {
        moves: [
          { move: "R", frame: 10 },
          { move: "U'", frame: 20 },
          { move: "F2", frame: 30 },
        ],
        timing_basis: "canonical",
      },
      reconstruction: {
        states: [],
        solved_reached: true,
      },
    },
  };
}

function groundTruthDiagnostic(
  scope: "sequence-only" | "sequence-and-clip-frame-indexed" =
    "sequence-only",
): any {
  const frameIndexed = scope === "sequence-and-clip-frame-indexed";
  return {
    schema: "cubed-core/decode-ground-truth-diagnostic-v1",
    schema_version: 1,
    diagnostic_only: true,
    capture_id: RECORDING_ID,
    reference: {
      kind: "published-smart-cube-ble",
      scope,
      dataset_id: "cubed-core-gtd1",
      revision: "main",
      bootstrap_manifest_sha256: "1".repeat(64),
      download_receipt_sha256: "2".repeat(64),
      corpus_manifest_sha256: "3".repeat(64),
      video_sha256: VIDEO_SHA,
      scramble_sha256: "4".repeat(64),
      ble_sha256: "5".repeat(64),
      ...(frameIndexed
        ? { frame_ground_truth_sha256: "6".repeat(64) }
        : {}),
      video_link_status: "linked",
      video_recording_id_verified: true,
    },
    normalization: {
      raw_metric: "quarter-turn",
      comparison_metric: "half-turn",
      method: "adjacent-same-face-mod-4",
    },
    counts: {
      decoded_htm: 3,
      ble_raw_qtm: 4,
      ble_canonical_htm: 3,
    },
    comparison: {
      distance: 1,
      ops: [
        {
          op: "equal",
          decoded: "R",
          reference: "R",
          index_decoded: 0,
          index_reference: 0,
        },
        {
          op: "substitute",
          decoded: "U'",
          reference: "U2",
          index_decoded: 1,
          index_reference: 1,
        },
        {
          op: "equal",
          decoded: "F2",
          reference: "F2",
          index_decoded: 2,
          index_reference: 2,
        },
      ],
    },
    ...(frameIndexed
      ? {
          frame_timing: {
            available: true,
            basis: "clip-local",
            source_schema: "cubed-core/clip-ble-ground-truth-v1",
            source_sha256: "6".repeat(64),
          },
        }
      : {}),
  };
}

function expectRejected(mutator: (value: any) => void): void {
  const value = validResult();
  mutator(value);
  expect(() => parseDecodeResultDocument(value)).toThrow();
}

test("accepts the strict public local_camera_v1 completed and abstained branches", () => {
  const completed = parseDecodeResultDocument(validResult());
  expect(completed.recording_id).toBe(RECORDING_ID);
  expect(completed.workstation?.video.encoded.width).toBe(1_920);

  const abstained = validResult();
  abstained.status = "abstained";
  abstained.moves = [];
  abstained.endpoint.solved_reached = false;
  abstained.workstation.sequence.moves = [];
  abstained.workstation.reconstruction.solved_reached = false;
  expect(parseDecodeResultDocument(abstained).status).toBe("abstained");

  const abstainedWithoutEndpointEvidence = structuredClone(abstained);
  abstainedWithoutEndpointEvidence.endpoint.solved_reached = null;
  delete abstainedWithoutEndpointEvidence.workstation.reconstruction;
  expect(
    parseDecodeResultDocument(abstainedWithoutEndpointEvidence).endpoint
      .solved_reached,
  ).toBeNull();

  const withoutWorkstation = validResult();
  delete withoutWorkstation.workstation;
  expect(parseDecodeResultDocument(withoutWorkstation).workstation).toBeUndefined();
});

test("rejects identity, branch, config, input, endpoint, and provenance drift", () => {
  const invalidMutations: Array<(value: any) => void> = [
    (value) => {
      value.recording_id = "a".repeat(31);
    },
    (value) => {
      value.recording_id = "A".repeat(32);
    },
    (value) => {
      value.profile = "legacy_camera_v0";
    },
    (value) => {
      value.status = "failed";
    },
    (value) => {
      value.config.name = "";
    },
    (value) => {
      value.config.cfg_hash = "not-a-cksum";
    },
    (value) => {
      value.config.cfg_hash_algorithm = "sha256";
    },
    (value) => {
      delete value.config.cfg_hash_algorithm;
    },
    (value) => {
      value.inputs = [value.inputs[0]];
    },
    (value) => {
      value.inputs[0].id = "";
    },
    (value) => {
      value.inputs[0].sha256 = "A".repeat(64);
    },
    (value) => {
      value.endpoint.solved_reached = false;
    },
    (value) => {
      value.evaluation = {};
    },
    (value) => {
      delete value.provenance.runtime_version;
    },
    (value) => {
      delete value.provenance.finished_at;
    },
    (value) => {
      value.provenance.implementation_sha256 = "not-a-sha";
    },
  ];
  invalidMutations.forEach(expectRejected);
});

test("enforces canonical move and status semantics", () => {
  expectRejected((value) => {
    value.moves[1] = "u'";
  });
  expectRejected((value) => {
    value.moves = [];
    value.workstation.sequence.moves = [];
  });
  expectRejected((value) => {
    value.status = "abstained";
  });
  expectRejected((value) => {
    value.status = "abstained";
    value.moves = [];
    value.endpoint.solved_reached = true;
    value.workstation.sequence.moves = [];
  });
});

test("rejects unknown properties at every strict portable boundary", () => {
  const invalidMutations: Array<(value: any) => void> = [
    (value) => {
      value.provider_host = "private.example";
    },
    (value) => {
      value.config.command = "--unsafe";
    },
    (value) => {
      value.inputs[0].path = "/private/video.mp4";
    },
    (value) => {
      value.endpoint.detail = "solved";
    },
    (value) => {
      value.provenance.log_path = "/private/run.log";
    },
    (value) => {
      value.workstation.raw_npz = "intermediate.npz";
    },
    (value) => {
      value.workstation.video.path = "/private/video.mp4";
    },
    (value) => {
      value.workstation.video.encoded.codec = "h264";
    },
    (value) => {
      value.workstation.initialization.teacher = "ble.json";
    },
    (value) => {
      value.workstation.warnings[0].severity = "warning";
    },
    (value) => {
      value.workstation.frames["0"].hash = "diagnostic";
    },
    (value) => {
      value.workstation.sequence.moves[0].time = 0.1;
    },
    (value) => {
      value.workstation.reconstruction.final_state = "solved";
    },
  ];
  invalidMutations.forEach(expectRejected);
});

test("requires encoded video identity and canonical, matching diagnostics", () => {
  expectRejected((value) => {
    delete value.workstation.video.encoded;
  });
  expectRejected((value) => {
    value.workstation.video.encoded.frame_count = 0;
  });
  expectRejected((value) => {
    value.workstation.sequence.timing_basis = "backtracked";
  });
  expectRejected((value) => {
    value.workstation.sequence.moves[1].move = "D";
  });
  expectRejected((value) => {
    value.workstation.reconstruction.solved_reached = false;
  });
  expectRejected((value) => {
    delete value.workstation.reconstruction.states;
  });
});

test("accepts only ordered decoder-checkpoint reconstruction timelines", () => {
  const value = validResult();
  const scramble = ["R", "U", "R'"];
  const timelineMoves = ["R", "U'", "F2"];
  value.workstation.reconstruction = {
    states: [
      fixtureCubeState(scramble),
      fixtureCubeState([...scramble, timelineMoves[0]]),
      fixtureCubeState([...scramble, ...timelineMoves.slice(0, 2)]),
      fixtureCubeState([...scramble, ...timelineMoves]),
    ],
    solved_reached: true,
    timeline: {
      moves: [
        { move: "R", frame: 10 },
        { move: "U'", frame: 20 },
        { move: "F2", frame: 20 },
      ],
      timing_basis: "decoder-checkpoint",
    },
  };
  const parsedMoves =
    parseDecodeResultDocument(value).workstation?.reconstruction?.timeline
      ?.moves;
  expect(parsedMoves).toHaveLength(3);
  expect(parsedMoves?.map(({ frame }) => frame)).toEqual([10, 20, 20]);

  expectRejected((candidate) => {
    candidate.workstation.reconstruction = structuredClone(
      value.workstation.reconstruction,
    );
    candidate.workstation.reconstruction.timeline.timing_basis = "canonical";
  });
  expectRejected((candidate) => {
    candidate.workstation.reconstruction = structuredClone(
      value.workstation.reconstruction,
    );
    candidate.workstation.reconstruction.timeline.moves[0].frame = 0;
  });
  expectRejected((candidate) => {
    candidate.workstation.reconstruction = structuredClone(
      value.workstation.reconstruction,
    );
    candidate.workstation.reconstruction.timeline.moves[2].frame = 5;
  });
  expectRejected((candidate) => {
    candidate.workstation.reconstruction = structuredClone(
      value.workstation.reconstruction,
    );
    candidate.workstation.reconstruction.states.pop();
  });
  expectRejected((candidate) => {
    candidate.workstation.reconstruction = structuredClone(
      value.workstation.reconstruction,
    );
    candidate.workstation.reconstruction.timeline.moves[2].frame = 120;
  });
  expectRejected((candidate) => {
    candidate.workstation.reconstruction = structuredClone(
      value.workstation.reconstruction,
    );
    candidate.workstation.reconstruction.states[1] =
      candidate.workstation.reconstruction.states[0];
  });
  expectRejected((candidate) => {
    candidate.workstation.reconstruction = structuredClone(
      value.workstation.reconstruction,
    );
    candidate.workstation.reconstruction.timeline.moves[2].move = "B2";
    candidate.workstation.reconstruction.states[3] = fixtureCubeState([
      ...scramble,
      timelineMoves[0],
      timelineMoves[1],
      "B2",
    ]);
  });
  expectRejected((candidate) => {
    candidate.status = "abstained";
    candidate.moves = [];
    candidate.endpoint.solved_reached = false;
    candidate.workstation.sequence.moves = [];
    candidate.workstation.reconstruction = structuredClone(
      value.workstation.reconstruction,
    );
    candidate.workstation.reconstruction.solved_reached = false;
  });
});

test("accepts a bound post-hoc BLE diagnostic without requiring it", () => {
  const sequenceOnly = validResult();
  sequenceOnly.ground_truth_diagnostic = groundTruthDiagnostic();
  const parsed = parseDecodeResultDocument(sequenceOnly);
  expect(parsed.ground_truth_diagnostic?.comparison.distance).toBe(1);
  expect(parsed.ground_truth_diagnostic?.reference.scope).toBe(
    "sequence-only",
  );

  const frameIndexed = validResult();
  frameIndexed.ground_truth_diagnostic = groundTruthDiagnostic(
    "sequence-and-clip-frame-indexed",
  );
  expect(
    parseDecodeResultDocument(frameIndexed).ground_truth_diagnostic
      ?.frame_timing?.basis,
  ).toBe("clip-local");
});

test("rejects BLE diagnostics that drift from the run or their rendered tape", () => {
  const invalidMutations: Array<(value: any) => void> = [
    (value) => {
      value.ground_truth_diagnostic.capture_id = "f".repeat(32);
    },
    (value) => {
      value.ground_truth_diagnostic.reference.video_sha256 = "f".repeat(64);
    },
    (value) => {
      value.ground_truth_diagnostic.counts.decoded_htm = 2;
    },
    (value) => {
      value.ground_truth_diagnostic.counts.ble_canonical_htm = 2;
    },
    (value) => {
      value.ground_truth_diagnostic.comparison.distance = 0;
    },
    (value) => {
      value.ground_truth_diagnostic.comparison.ops[1].decoded = "D";
    },
    (value) => {
      value.ground_truth_diagnostic.comparison.ops[1].index_decoded = 2;
    },
    (value) => {
      value.ground_truth_diagnostic.comparison.ops[1].op = "equal";
    },
    (value) => {
      value.ground_truth_diagnostic.reference.internal_path = "/private/ble";
    },
    (value) => {
      value.ground_truth_diagnostic.frame_timing = {
        available: true,
        basis: "clip-local",
        source_schema: "cubed-core/clip-ble-ground-truth-v1",
        source_sha256: "7".repeat(64),
      };
    },
    (value) => {
      value.ground_truth_diagnostic.reference.scope =
        "sequence-and-clip-frame-indexed";
    },
  ];
  invalidMutations.forEach((mutator) => {
    const value = validResult();
    value.ground_truth_diagnostic = groundTruthDiagnostic();
    mutator(value);
    expect(() => parseDecodeResultDocument(value)).toThrow();
  });

  const mismatchedFrameReference = validResult();
  mismatchedFrameReference.ground_truth_diagnostic = groundTruthDiagnostic(
    "sequence-and-clip-frame-indexed",
  );
  mismatchedFrameReference.ground_truth_diagnostic.frame_timing.source_sha256 =
    "8".repeat(64);
  expect(() =>
    parseDecodeResultDocument(mismatchedFrameReference),
  ).toThrow();
});
