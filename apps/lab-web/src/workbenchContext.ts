import { useOutletContext } from "react-router-dom";

import type { ApiState } from "./apiSession";
import type { Capabilities } from "./types";

export interface WorkbenchContext {
  capabilities: Capabilities | null;
  apiState: ApiState;
}

export function useWorkbenchContext(): WorkbenchContext {
  return useOutletContext<WorkbenchContext>();
}
