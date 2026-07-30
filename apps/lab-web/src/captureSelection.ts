// Pure helpers for carrying the selected capture across Label and Decode
// through the ?capture= query param. No React or router imports here
// so this module stays directly importable from node --test.

export const CAPTURE_PARAM = "capture";

interface OrderedCapture {
  capture_id: string;
  capture_session_id?: string;
  created_at: string;
  original_filename: string;
}

const NATURAL_FILENAME = new Intl.Collator("en", {
  numeric: true,
  sensitivity: "base",
});

function createdTime(capture: OrderedCapture): number {
  const parsed = Date.parse(capture.created_at);
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY;
}

function compareCreatedNewest(
  left: OrderedCapture,
  right: OrderedCapture,
): number {
  return createdTime(right) - createdTime(left);
}

export function isPublishedDatasetCapture(capture: OrderedCapture): boolean {
  return capture.capture_session_id?.startsWith("public-dataset:") ?? false;
}

export function orderDecodeCaptures<T extends OrderedCapture>(
  captures: readonly T[],
): { yourRecordings: T[]; publishedDataset: T[] } {
  const yourRecordings = captures
    .filter((capture) => !isPublishedDatasetCapture(capture))
    .sort(
      (left, right) =>
        compareCreatedNewest(left, right) ||
        NATURAL_FILENAME.compare(
          left.original_filename,
          right.original_filename,
        ) ||
        left.capture_id.localeCompare(right.capture_id),
    );
  const publishedDataset = captures
    .filter(isPublishedDatasetCapture)
    .sort(
      (left, right) =>
        NATURAL_FILENAME.compare(
          left.original_filename,
          right.original_filename,
        ) ||
        compareCreatedNewest(left, right) ||
        left.capture_id.localeCompare(right.capture_id),
    );
  return { yourRecordings, publishedDataset };
}

export function captureIdFromParams(
  params: URLSearchParams,
  captures: { capture_id: string }[],
): string {
  const requested = params.get(CAPTURE_PARAM);
  if (requested && captures.some((capture) => capture.capture_id === requested)) {
    return requested;
  }
  return captures[0]?.capture_id ?? "";
}

export function explicitCaptureIdFromParams(
  params: URLSearchParams,
  captures: { capture_id: string }[],
): string {
  const requested = params.get(CAPTURE_PARAM);
  return requested &&
    captures.some((capture) => capture.capture_id === requested)
    ? requested
    : "";
}

export function captureHref(path: string, captureId: string): string {
  return captureId ? `${path}?${CAPTURE_PARAM}=${captureId}` : path;
}
