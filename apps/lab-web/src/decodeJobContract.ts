import type { DecodeJobState } from "./types";

const TERMINAL_DECODE_STATES: ReadonlySet<DecodeJobState> = new Set([
  "succeeded",
  "failed",
  "timed_out",
  "cancelled",
]);

export function isTerminalDecodeState(status: DecodeJobState): boolean {
  return TERMINAL_DECODE_STATES.has(status);
}

export function formatFinishedAt(finishedAt: string | null): string {
  if (!finishedAt) return "a previous run";
  const parsed = new Date(finishedAt);
  return Number.isNaN(parsed.getTime()) ? "a previous run" : parsed.toLocaleString();
}
