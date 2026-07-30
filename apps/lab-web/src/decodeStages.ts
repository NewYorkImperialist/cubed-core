const DECODE_STAGE_LABELS: Readonly<Record<string, string>> = {
  upload: "Uploading to runner",
  reads: "Reading frames",
  events: "Motion events",
  alignfeat: "Alignment features",
  decode: "Decode search",
  download: "Downloading result",
  validate: "Validating",
};

export type DecodeStageState = "done" | "current" | "pending";

export interface DecodeStageChip {
  token: string;
  label: string;
  state: DecodeStageState;
}

export interface DecodeStageProgress {
  current: number;
  total: number;
}

export function decodeStageLabel(token: string): string {
  const trimmed = token.trim();
  return DECODE_STAGE_LABELS[trimmed] ?? trimmed;
}

export function decodeStageChips(
  stagesSeen: readonly string[] | null | undefined,
  currentStage: string | null | undefined,
): DecodeStageChip[] {
  const ordered: string[] = [];
  for (const entry of stagesSeen ?? []) {
    if (typeof entry !== "string") continue;
    const token = entry.trim();
    if (!token || ordered.includes(token)) continue;
    ordered.push(token);
  }

  const current = typeof currentStage === "string" ? currentStage.trim() : "";
  if (current && !ordered.includes(current)) ordered.push(current);
  const currentIndex = current ? ordered.indexOf(current) : -1;

  return ordered.map((token, index) => ({
    token,
    label: decodeStageLabel(token),
    state:
      index === currentIndex
        ? "current"
        : currentIndex === -1 || index < currentIndex
          ? "done"
          : "pending",
  }));
}

export function decodeStageProgressLabel(
  progress: DecodeStageProgress | null | undefined,
): string {
  if (!progress) return "";
  const total = Math.round(progress.total);
  const current = Math.round(progress.current);
  if (!Number.isFinite(total) || !Number.isFinite(current) || total <= 0) {
    return "";
  }
  const bounded = Math.max(0, Math.min(total, current));
  return `${bounded} / ${total} frames`;
}
