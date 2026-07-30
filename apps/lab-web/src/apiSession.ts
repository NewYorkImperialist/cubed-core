export type ApiState = "checking" | "online" | "unauthorized" | "offline";

export function apiStateFromProbe(input: {
  healthOk: boolean;
  capabilitiesStatus: number | null;
}): ApiState {
  if (!input.healthOk) return "offline";
  if (input.capabilitiesStatus === null) return "online";
  if (input.capabilitiesStatus === 401 || input.capabilitiesStatus === 403) {
    return "unauthorized";
  }
  return "online";
}
