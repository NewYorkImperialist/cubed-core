export type ProbeStatus = "ok" | "unavailable" | "error";

export interface VideoProbe {
  status: ProbeStatus;
  error?: string;
  codec?: string | null;
  width?: number | null;
  height?: number | null;
  fps?: number | null;
  duration_seconds?: number | null;
  capture_class?:
    | "target"
    | "research-high-speed"
    | "borderline"
    | "unsupported"
    | "unknown";
  guidance?: string;
}

export interface CaptureReceipt {
  schema: string;
  schema_version?: number;
  recording_id?: string;
  capture_id: string;
  capture_session_id?: string;
  created_at: string;
  source: string;
  state?: "incomplete" | "sealed";
  sealed_at?: string | null;
  seal_purpose?: "label" | "decode" | null;
  original_filename: string;
  video: {
    path: string;
    bytes: number;
    sha256: string;
    container?: string | null;
    codec?: string | null;
    encoded_width?: number | null;
    encoded_height?: number | null;
    configured_fps?: number | null;
    actual_fps?: number | null;
    frame_count?: number | null;
    rotation_degrees?: number;
    mirrored?: boolean;
  };
  camera?: {
    facing: string;
    device_model?: string | null;
    camera_id?: string | null;
    intrinsics?: unknown;
  };
  solve?: {
    scramble?: string | null;
    end_condition?: string;
    start_frame?: number | null;
    end_frame?: number | null;
  };
  calibration?: SidecarReceipt | null;
  teacher?: SidecarReceipt | null;
  sensors?: {
    phone_imu?: SidecarReceipt | null;
    ble_raw?: SidecarReceipt | null;
  };
  probe: VideoProbe;
  notes: string;
  sidecars?: Record<string, string | null>;
  readiness?: {
    can_label: boolean;
    can_decode: boolean;
    missing_for_decode: string[];
    warnings: string[];
  };
}

export interface SidecarReceipt {
  path: string;
  sha256: string;
  kind?: string;
  schema_version?: number | null;
  display_name?: string;
}

export type SidecarKind = "calibration" | "teacher" | "ble-raw" | "phone-imu";

export interface CaptureList {
  captures: CaptureReceipt[];
  count: number;
}

export interface Health {
  status: string;
  version: string;
  publication_status?: string;
}

export interface ToolCapability {
  id: string;
  name: string;
  status: string;
  surface: string;
}

export interface GpuDeviceCapability {
  name: string;
  memory_mib: number | null;
}

export interface GpuCapability {
  available: boolean;
  devices: GpuDeviceCapability[];
  reason?: string;
}

export interface LabelCapability {
  pnp_assist: {
    enabled: boolean;
    status: "available" | "disabled";
    reason: string | null;
    execution_host: "api-host-cpu";
  };
  prediction: {
    enabled: boolean;
    status: "available" | "disabled" | "misconfigured";
    reason: string | null;
    executable: string | null;
    execution_host: "api-host";
    output_schema: "cubed-core/label-predictions-v1";
    backend: "camera-tracker-v1" | "external-command" | null;
    model_profile: "camera-tracker-v1" | null;
    model_aligned_navigation: boolean;
  };
  autosave: {
    workspace_capture: boolean;
    local_file: "browser-local-storage";
  };
  exports: string[];
}

export interface Capabilities {
  schema: string;
  workspace: string;
  upload_limits: {
    video_bytes: number;
  };
  gpu: GpuCapability;
  decode_jobs: DecodeJobCapability;
  label: LabelCapability;
  tools: ToolCapability[];
}

export type DecodeJobState =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "timed_out"
  | "cancelled";

export interface DecodeJobStageProgress {
  current: number;
  total: number;
}

export interface DecodeJobList {
  jobs: DecodeJobStatus[];
}

/**
 * One remote GPU host a Decode job can run against, as configured in
 * workspace/remote-hosts.json (see docs/CLOUD_GPU.md). Only the fields the
 * picker needs to show are public; per-host runner overrides (root, venvs,
 * NVDEC policy) stay server-side.
 */
export interface RemoteHost {
  id: string;
  label: string;
}

export interface RemoteHostsResponse {
  schema: "cubed-core/remote-hosts-v1";
  hosts: RemoteHost[];
  env_default_id: string | null;
}

export interface DecodeJobCapability {
  enabled: boolean;
  status: "available" | "configured-external" | "disabled" | "misconfigured";
  runner_kind: "native" | "external" | "disabled";
  runner_label?: string | null;
  reason: string | null;
  request_schema: string;
  output_schema: "cubed-core/decode-result-v1";
  executable?: string | null;
  profile?: string;
  execution_host?: string;
}

/**
 * Live Decode state plus the result identity and the server's replay check.
 */
export interface DecodeJobStatus {
  schema: "cubed-core/decode-job-status-v1";
  schema_version: 1;
  job_id: string;
  capture_id: string;
  status: DecodeJobState;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  return_code: number | null;
  log: string;
  log_truncated: boolean;
  error: string | null;
  result_available: boolean;
  result_url: string | null;
  result_sha256: string | null;
  result_bytes: number | null;
  replay_solved_reached: boolean | null;
  video_sha256?: string | null;
  outcome?: "completed" | "abstained" | "failed" | "cancelled" | null;
  failure?: {
    code: string;
    message: string;
    retryable: boolean;
  } | null;
  stage?: string | null;
  stages_seen?: string[];
  stage_progress?: DecodeJobStageProgress | null;
}

export interface DecodeResultConfig {
  name: string;
  cfg_hash: string;
  cfg_hash_algorithm: "posix-cksum";
  cfg_hash_input?: string;
  extras_hash?: string | null;
}

export interface DecodeResultInput {
  id: string;
  sha256: string;
}

export interface DecodeWorkstationVideo {
  sha256: string;
  bytes: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  encoded: {
    fps: number;
    frame_count: number;
    width: number;
    height: number;
  };
}

export interface DecodeWorkstationFace {
  corners: [
    [number, number],
    [number, number],
    [number, number],
    [number, number],
  ];
  confidence: number;
  keypoint_confidence: [number, number, number, number];
}

export interface DecodeWorkstationRead {
  slot: "up" | "front" | "right";
  lab: [
    [number, number, number],
    [number, number, number],
    [number, number, number],
    [number, number, number],
    [number, number, number],
    [number, number, number],
    [number, number, number],
    [number, number, number],
    [number, number, number],
  ];
  confidence: [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  valid_pixels?: [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  total_pixels?: [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  used_fallback?: [
    boolean,
    boolean,
    boolean,
    boolean,
    boolean,
    boolean,
    boolean,
    boolean,
    boolean,
  ];
  relative_area: number;
  corners: DecodeWorkstationFace["corners"];
}

export interface DecodeWorkstationFrame {
  motion: number | null;
  aligned?: number;
  aligned_streak?: number | null;
  face_count?: number | null;
  gated?: boolean;
  faces?: DecodeWorkstationFace[];
  reads?: DecodeWorkstationRead[];
}

export interface DecodeWorkstationSpan {
  f0: number;
  f1: number;
  event: number | null;
  top: Array<{
    path: string | string[];
    orientation: string | string[];
    score: number;
  }>;
}

export interface DecodeWorkstationSequence {
  moves: Array<{ move: string; frame: number }>;
  timing_basis: "canonical";
}

export interface DecodeWorkstationReconstruction {
  states: Array<Record<string, string[]>>;
  solved_reached: boolean;
  timeline?: {
    moves: Array<{ move: string; frame: number }>;
    timing_basis: "decoder-checkpoint";
  };
}

export interface DecodeWorkstation {
  schema: "cubed-core/decode-workstation-v1";
  schema_version: 1;
  initialization: {
    scramble: string;
  };
  video: DecodeWorkstationVideo;
  window: [number, number];
  warnings: Array<{ code: string; message: string }>;
  frames: Record<string, DecodeWorkstationFrame>;
  events?: number[];
  trellis?: {
    spans: DecodeWorkstationSpan[];
    bridge?: string[];
  };
  sequence?: DecodeWorkstationSequence;
  reconstruction?: DecodeWorkstationReconstruction;
}

export interface DecodeResultProvenance {
  runtime_version: string;
  implementation_id?: string;
  implementation_sha256?: string;
  python_version?: string;
  numpy_version?: string;
  finished_at: string;
}

export interface DecodeGroundTruthOp {
  op: "equal" | "substitute" | "insert" | "delete";
  decoded: string | null;
  reference: string | null;
  index_decoded: number | null;
  index_reference: number | null;
}

export interface DecodeGroundTruthDiagnostic {
  schema: "cubed-core/decode-ground-truth-diagnostic-v1";
  schema_version: 1;
  diagnostic_only: true;
  capture_id: string;
  reference: {
    kind: "published-smart-cube-ble";
    scope: "sequence-only" | "sequence-and-clip-frame-indexed";
    dataset_id: string;
    revision: string;
    bootstrap_manifest_sha256: string;
    download_receipt_sha256: string;
    corpus_manifest_sha256: string;
    video_sha256: string;
    scramble_sha256: string;
    ble_sha256: string;
    frame_ground_truth_sha256?: string;
    video_link_status: "linked" | "failed_timeout";
    video_recording_id_verified: true;
  };
  normalization: {
    raw_metric: "quarter-turn";
    comparison_metric: "half-turn";
    method: "adjacent-same-face-mod-4";
  };
  counts: {
    decoded_htm: number;
    ble_raw_qtm: number;
    ble_canonical_htm: number;
  };
  comparison: {
    distance: number;
    ops: DecodeGroundTruthOp[];
  };
  frame_timing?: {
    available: true;
    basis: "clip-local";
    source_schema: "cubed-core/clip-ble-ground-truth-v1";
    source_sha256: string;
  };
}

export interface DecodeResultDocument {
  schema: "cubed-core/decode-result";
  schema_version: 1;
  recording_id: string;
  // "completed" means the run reached the solved endpoint and carries its
  // moves. A run that terminated normally without reaching that endpoint is
  // "abstained" and carries no moves. A persisted result never says "failed";
  // a crash surfaces as a failed job instead.
  status: "completed" | "abstained";
  profile: "local_camera_v1";
  config: DecodeResultConfig;
  inputs: DecodeResultInput[];
  moves: string[];
  endpoint: { solved_reached: boolean | null };
  evaluation: null;
  provenance: DecodeResultProvenance;
  ground_truth_diagnostic?: DecodeGroundTruthDiagnostic;
  workstation?: DecodeWorkstation;
}
