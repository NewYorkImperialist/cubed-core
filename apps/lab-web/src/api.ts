import type {
  Capabilities,
  CaptureList,
  CaptureReceipt,
  Health,
  RemoteHostsResponse,
  SidecarKind,
} from "./types";
import type {
  AnnotationFace,
  FrameAnnotationsDocument,
  Point,
} from "./localContracts";
import {
  CALIBRATION_COLORS,
  type CalibrationColor,
} from "./calibrationSampling";

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

const ADMIN_TOKEN_KEY = "cubed-core-admin-token";
const ADMIN_TOKEN_HEADER = "X-Cubed-Admin-Token";
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
const LARGE_TRANSFER_TIMEOUT_MS = 5 * 60_000;

declare global {
  interface Window {
    __CUBED_CORE_ADMIN_TOKEN__?: string;
  }
}

function loadAdminToken(): string {
  const supplied = window.__CUBED_CORE_ADMIN_TOKEN__ ?? "";
  if (supplied) {
    return supplied;
  }
  try {
    return window.sessionStorage.getItem(ADMIN_TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

let adminToken = loadAdminToken();
let localSessionAttempt: Promise<boolean> | null = null;
let localSessionRefreshAttempt: Promise<boolean> | null = null;

export function setAdminToken(value: string): void {
  const trimmed = value.trim();
  window.__CUBED_CORE_ADMIN_TOKEN__ = trimmed;
  try {
    window.sessionStorage.setItem(ADMIN_TOKEN_KEY, trimmed);
  } catch {
    // The in-memory value still protects this page when storage is blocked.
  }
  adminToken = trimmed;
}

export function applyAdminToken(headers: Headers): void {
  if (adminToken) headers.set(ADMIN_TOKEN_HEADER, adminToken);
}

async function bootstrapLocalSession(force = false): Promise<boolean> {
  if (adminToken && !force) return true;
  let response: Response;
  try {
    response = await fetchWithTimeout(`${API_BASE}/api/local-session`, {
      method: "POST",
      cache: "no-store",
      credentials: "omit",
      headers: { Accept: "application/json" },
    });
  } catch {
    // The normal request below reports an unreachable service with its
    // established error contract. Let a later retry bootstrap after startup.
    return false;
  }
  if (!response.ok) return true;
  try {
    const payload = (await response.json()) as { admin_token?: unknown };
    if (typeof payload.admin_token === "string" && payload.admin_token) {
      setAdminToken(payload.admin_token);
    }
  } catch {
    // Fail closed. Protected requests remain protected by the server guard.
  }
  return true;
}

async function refreshLocalSessionAfterAuthFailure(): Promise<boolean> {
  if (!localSessionRefreshAttempt) {
    const staleToken = adminToken;
    const attempt = (async () => {
      await bootstrapLocalSession(true);
      return Boolean(adminToken && adminToken !== staleToken);
    })();
    localSessionRefreshAttempt = attempt;
    void attempt.finally(() => {
      if (localSessionRefreshAttempt === attempt) {
        localSessionRefreshAttempt = null;
      }
    });
  }
  return localSessionRefreshAttempt;
}

async function ensureAdminToken(): Promise<void> {
  if (adminToken) return;
  localSessionAttempt ??= bootstrapLocalSession();
  const attempt = localSessionAttempt;
  const conclusive = await attempt;
  if (!conclusive && localSessionAttempt === attempt) {
    localSessionAttempt = null;
  }
}

export class RequestError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "RequestError";
  }
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init?: RequestInit,
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const upstream = init?.signal;
  let timedOut = false;
  const abortFromUpstream = () => controller.abort();
  if (upstream?.aborted) abortFromUpstream();
  else upstream?.addEventListener("abort", abortFromUpstream, { once: true });
  const timeout = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (error) {
    if (upstream?.aborted) throw error;
    if (timedOut) {
      throw new Error(
        `The local Cubed Core service did not respond within ${Math.round(timeoutMs / 1000)} seconds.`,
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
    upstream?.removeEventListener("abort", abortFromUpstream);
  }
}

function rethrowTransportError(error: unknown): never {
  if (error instanceof Error && error.name === "AbortError") throw error;
  if (
    error instanceof Error &&
    error.message.startsWith("The local Cubed Core service did not respond")
  ) {
    throw error;
  }
  throw new Error("The local Cubed Core service is not reachable.");
}

async function retryWithFreshLocalSession(
  response: Response,
  path: string,
  init: RequestInit | undefined,
  timeoutMs: number,
): Promise<Response> {
  if (response.status !== 401 && response.status !== 403) return response;
  if (!(await refreshLocalSessionAfterAuthFailure())) return response;

  const headers = new Headers(init?.headers);
  applyAdminToken(headers);
  try {
    return await fetchWithTimeout(
      `${API_BASE}${path}`,
      { ...init, headers },
      timeoutMs,
    );
  } catch (error) {
    rethrowTransportError(error);
  }
}

export async function requestJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
): Promise<T> {
  let response: Response;
  await ensureAdminToken();
  const headers = new Headers(init?.headers);
  applyAdminToken(headers);
  try {
    response = await fetchWithTimeout(
      `${API_BASE}${path}`,
      { ...init, headers },
      timeoutMs,
    );
  } catch (error) {
    rethrowTransportError(error);
  }
  response = await retryWithFreshLocalSession(response, path, init, timeoutMs);

  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;
    try {
      const payload = (await response.json()) as {
        detail?: string | { message?: unknown };
      };
      if (typeof payload.detail === "string") {
        message = payload.detail;
      } else if (
        payload.detail &&
        typeof payload.detail.message === "string"
      ) {
        message = payload.detail.message;
      }
    } catch {
      // The status is still useful when an intermediary returns non-JSON.
    }
    throw new RequestError(message, response.status);
  }

  return (await response.json()) as T;
}

export async function requestBlob(
  path: string,
  init?: RequestInit,
  timeoutMs = LARGE_TRANSFER_TIMEOUT_MS,
): Promise<Blob> {
  let response: Response;
  await ensureAdminToken();
  const headers = new Headers(init?.headers);
  applyAdminToken(headers);
  try {
    response = await fetchWithTimeout(
      `${API_BASE}${path}`,
      { ...init, headers },
      timeoutMs,
    );
  } catch (error) {
    rethrowTransportError(error);
  }
  response = await retryWithFreshLocalSession(response, path, init, timeoutMs);
  if (!response.ok) {
    let message = `Request failed with status ${response.status}.`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Preserve the HTTP status when the failure body is not JSON.
    }
    throw new RequestError(message, response.status);
  }
  return response.blob();
}

export interface CaptureMediaTicket {
  capture_id: string;
  url: string;
  expires_at: number;
}

export async function createCaptureMediaTicket(
  captureId: string,
): Promise<CaptureMediaTicket> {
  const ticket = await requestJson<CaptureMediaTicket>(
    `/api/captures/${encodeURIComponent(captureId)}/media-ticket`,
    { method: "POST" },
  );
  return {
    ...ticket,
    url: `${API_BASE}${ticket.url}`,
  };
}

export interface LabelAssistFace {
  name: "labeled" | "top" | "right" | "bottom" | "left";
  vertices: [number, number, number, number];
  corners: [Point, Point, Point, Point];
}

export interface LabelAssistResponse {
  ok: boolean;
  reason?: string;
  intrinsics_source: string;
  inferred_corner: Point | null;
  rvec?: [number, number, number];
  tvec?: [number, number, number];
  wireframe?: Point[];
  faces?: LabelAssistFace[];
}

export function extrapolateLabelFace(body: {
  labeled_corners: Point[];
  width: number;
  height: number;
  K?: number[][];
  pins?: { xy: Point; vertex: number }[];
  shrink?: number;
}): Promise<LabelAssistResponse> {
  return requestJson<LabelAssistResponse>("/api/label/assist/extrapolate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchCaptureAnnotations(
  captureId: string,
): Promise<FrameAnnotationsDocument> {
  return requestJson<FrameAnnotationsDocument>(
    `/api/captures/${encodeURIComponent(captureId)}/annotations`,
  );
}

export function saveCaptureAnnotations(
  captureId: string,
  document: FrameAnnotationsDocument,
): Promise<{ status: string; capture_id: string; frames: number; bytes: number }> {
  return requestJson(
    `/api/captures/${encodeURIComponent(captureId)}/annotations`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(document),
    },
  );
}

export function downloadLabelDatasetExport(captureId: string): Promise<Blob> {
  return requestBlob(
    `/api/captures/${encodeURIComponent(captureId)}/annotations/dataset.zip`,
  );
}

export interface LabelPredictionResponse {
  schema: "cubed-core/label-predictions-v1";
  frames: { frame_index: number; faces: AnnotationFace[] }[];
}

export interface LabelAlignmentResponse {
  schema: "cubed-core/label-alignment-v1";
  threshold: number;
  model_profile: "camera-tracker-v1";
  width: number;
  height: number;
  frame_count: number;
  aligned_frames: number[];
  alignment_confs: number[];
}

export function scanCaptureLabelAlignment(
  captureId: string,
): Promise<LabelAlignmentResponse> {
  return requestJson<LabelAlignmentResponse>(
    `/api/captures/${encodeURIComponent(captureId)}/label-alignment`,
    { method: "POST" },
  );
}

export function predictCaptureLabels(
  captureId: string,
  body: {
    width: number;
    height: number;
    frame_indices: number[] | null;
    skip_frame_indices?: number[];
  },
): Promise<LabelPredictionResponse> {
  return requestJson<LabelPredictionResponse>(
    `/api/captures/${encodeURIComponent(captureId)}/label-predict`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

export function fetchHealth(signal?: AbortSignal): Promise<Health> {
  return requestJson<Health>("/api/health", { signal });
}

export function fetchCapabilities(signal?: AbortSignal): Promise<Capabilities> {
  return requestJson<Capabilities>("/api/capabilities", { signal });
}

export function fetchRemoteHosts(
  signal?: AbortSignal,
): Promise<RemoteHostsResponse> {
  return requestJson<RemoteHostsResponse>("/api/remote-hosts", { signal });
}

export function fetchCaptures(): Promise<CaptureList> {
  return requestJson<CaptureList>("/api/captures");
}

export interface CaptureDeleteReceipt {
  schema: "cubed-core/capture-delete-v1";
  schema_version: 1;
  capture_id: string;
  trashed: true;
  recoverable: true;
}

export function deleteCapture(
  captureId: string,
): Promise<CaptureDeleteReceipt> {
  return requestJson<CaptureDeleteReceipt>(
    `/api/captures/${encodeURIComponent(captureId)}`,
    { method: "DELETE" },
  );
}

export function importCapture(
  video: File,
  notes: string,
  scramble: string,
  captureSessionId: string,
): Promise<CaptureReceipt> {
  const data = new FormData();
  data.set("video", video);
  data.set("source", "import");
  data.set("notes", notes);
  data.set("scramble", scramble);
  data.set("capture_session_id", captureSessionId);
  return requestJson<CaptureReceipt>("/api/captures/import", {
    method: "POST",
    body: data,
  }, LARGE_TRANSFER_TIMEOUT_MS);
}

export function attachCaptureSidecar(
  captureId: string,
  kind: SidecarKind,
  sidecar: File,
): Promise<CaptureReceipt> {
  const data = new FormData();
  data.set("sidecar", sidecar);
  return requestJson<CaptureReceipt>(
    `/api/captures/${encodeURIComponent(captureId)}/sidecars/${kind}`,
    {
      method: "POST",
      body: data,
    },
  );
}

/**
 * Attaches a calibration sidecar copied server-side from an existing
 * source (the bundled GAN 12 release asset, or another capture's own
 * already-attached calibration) instead of an upload. The server funnels
 * either source through the exact same validation the upload route uses,
 * so this never carries its own accept/reject rule.
 */
export function attachReusedCalibration(
  captureId: string,
  source: "bundled" | "capture",
  sourceCaptureId?: string,
): Promise<CaptureReceipt> {
  const data = new FormData();
  data.set("source", source);
  if (sourceCaptureId) data.set("source_capture_id", sourceCaptureId);
  return requestJson<CaptureReceipt>(
    `/api/captures/${encodeURIComponent(captureId)}/sidecars/calibration/reuse`,
    {
      method: "POST",
      body: data,
    },
  );
}

export type CaptureCalibrationPatches = Record<CalibrationColor, File>;

/**
 * Create and immediately attach color-centroids-v1 from six bounded PNG
 * crops sampled from the capture's own video.
 */
export function createCaptureCalibration(
  captureId: string,
  patches: CaptureCalibrationPatches,
): Promise<CaptureReceipt> {
  const data = new FormData();
  for (const color of CALIBRATION_COLORS) {
    data.set(color, patches[color], `${color}.png`);
  }
  return requestJson<CaptureReceipt>(
    `/api/captures/${encodeURIComponent(captureId)}/calibration/from-crops`,
    {
      method: "POST",
      body: data,
    },
  );
}

export function sealCapture(
  captureId: string,
  purpose: "label" | "decode",
): Promise<CaptureReceipt> {
  const data = new FormData();
  data.set("purpose", purpose);
  return requestJson<CaptureReceipt>(
    `/api/captures/${encodeURIComponent(captureId)}/seal`,
    {
      method: "POST",
      body: data,
    },
  );
}
