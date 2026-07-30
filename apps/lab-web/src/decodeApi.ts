import { requestBlob, requestJson } from "./api";
import type {
  DecodeJobList,
  DecodeJobStatus,
  DecodeResultDocument,
} from "./types";

export interface DecodePreflightCheck {
  id: string;
  status: "pass" | "warning" | "fail";
  detail: string;
  path?: string;
}

export interface DecodePreflight {
  schema: "cubed-core/decode-preflight";
  schema_version: 1;
  recording_id: string | null;
  status: "ready" | "blocked";
  ready: boolean;
  execution_allowed: boolean;
  profile: "local_camera_v1";
  config: {
    name: string;
    cfg_hash: string;
    cfg_hash_algorithm: "posix-cksum";
    derived_from_cfg_hash: string;
    evidence_status: string;
  };
  checks: DecodePreflightCheck[];
  missing: string[];
  warnings: string[];
}

export interface DecodeRunDeleteReceipt {
  schema: "cubed-core/decode-run-delete-v1";
  schema_version: 1;
  job_id: string;
  capture_id: string;
  trashed: true;
  recoverable: true;
  trash_id: string;
}

export async function fetchDecodePreflight(
  recordingId: string,
  signal?: AbortSignal,
): Promise<DecodePreflight> {
  return requestJson<DecodePreflight>(
    `/api/captures/${encodeURIComponent(recordingId)}/decode-preflight`,
    { signal },
  );
}

/**
 * Decode job routes. The server gates job creation on decode preflight and
 * resolves every input path from its own workspace, so the page submits
 * against a capture id.
 */
export const decodeJobPaths = {
  submit(captureId: string, remoteHostId?: string): string {
    const path = `/api/captures/${encodeURIComponent(captureId)}/decode-jobs`;
    return remoteHostId
      ? `${path}?remote_host=${encodeURIComponent(remoteHostId)}`
      : path;
  },
  // Same route as submit(): a GET against the capture's job collection
  // instead of a POST that creates a new one.
  list(captureId: string): string {
    return `/api/captures/${encodeURIComponent(captureId)}/decode-jobs`;
  },
  listAll(query = "", limit = 100): string {
    const params = new URLSearchParams({ limit: String(limit) });
    if (query.trim()) params.set("q", query.trim());
    return `/api/decode/jobs?${params.toString()}`;
  },
  status(jobId: string): string {
    return `/api/decode/jobs/${encodeURIComponent(jobId)}`;
  },
  result(jobId: string): string {
    return `/api/decode/jobs/${encodeURIComponent(jobId)}/result`;
  },
  remove(jobId: string): string {
    return `/api/decode/jobs/${encodeURIComponent(jobId)}`;
  },
};

export function submitDecodeJob(
  captureId: string,
  remoteHostId?: string,
  signal?: AbortSignal,
): Promise<DecodeJobStatus> {
  return requestJson<DecodeJobStatus>(
    decodeJobPaths.submit(captureId, remoteHostId),
    {
      method: "POST",
      signal,
    },
  );
}

export function fetchDecodeJobStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<DecodeJobStatus> {
  return requestJson<DecodeJobStatus>(decodeJobPaths.status(jobId), { signal });
}

export function fetchDecodeJobResult(
  jobId: string,
  signal?: AbortSignal,
): Promise<DecodeResultDocument> {
  return requestJson<DecodeResultDocument>(decodeJobPaths.result(jobId), {
    signal,
  });
}

export function fetchDecodeJobResultBlob(
  jobId: string,
  signal?: AbortSignal,
): Promise<Blob> {
  return requestBlob(decodeJobPaths.result(jobId), { signal });
}

export function deleteDecodeRun(
  jobId: string,
  signal?: AbortSignal,
): Promise<DecodeRunDeleteReceipt> {
  return requestJson<DecodeRunDeleteReceipt>(decodeJobPaths.remove(jobId), {
    method: "DELETE",
    signal,
  });
}

// Servers that predate this route reply 404, and the Decode stage treats
// that identically to any other read failure here: it keeps its normal
// preflight-driven idle behavior instead of surfacing an error for a
// background probe.
export function fetchDecodeJobsForCapture(
  captureId: string,
  signal?: AbortSignal,
): Promise<DecodeJobList> {
  return requestJson<DecodeJobList>(decodeJobPaths.list(captureId), {
    signal,
  });
}

export function fetchDecodeJobs(
  query = "",
  limit = 100,
  signal?: AbortSignal,
): Promise<DecodeJobList> {
  return requestJson<DecodeJobList>(decodeJobPaths.listAll(query, limit), {
    signal,
  });
}
