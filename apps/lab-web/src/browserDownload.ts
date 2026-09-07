// Small browser-only download helpers shared by the pages that let an
// operator save an already-fetched Blob (a run artifact, a dataset export)
// to disk without navigating away from the workbench.

/** Trigger a client-side download of an in-memory blob. */
export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Sanitize a filename into a short, safe download-name stem.
 *
 * Strips the extension and replaces anything outside the safe filename
 * charset, so the result is always a valid cross-platform filename stem.
 * Falls back to `fallback` when nothing safe remains.
 */
export function safeDownloadStem(filename: string, fallback: string): string {
  const stem = filename
    .replace(/\.[^.]+$/, "")
    .replace(/[^A-Za-z0-9._-]+/g, "_");
  return stem.slice(0, 100) || fallback;
}
