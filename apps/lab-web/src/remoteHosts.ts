import { useCallback, useEffect, useRef, useState } from "react";

import { fetchRemoteHosts } from "./api";
import type { RemoteHost } from "./types";

export const REMOTE_HOST_STORAGE_KEY = "cubed-core-remote-host-id";

function readStoredHostId(): string {
  try {
    return window.localStorage.getItem(REMOTE_HOST_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

function writeStoredHostId(value: string): void {
  try {
    if (value) window.localStorage.setItem(REMOTE_HOST_STORAGE_KEY, value);
    else window.localStorage.removeItem(REMOTE_HOST_STORAGE_KEY);
  } catch {
    // The choice remains usable for this session when storage is blocked.
  }
}

export interface RemoteHostSelection {
  hosts: RemoteHost[];
  envDefaultId: string | null;
  selectedId: string;
  setSelectedId: (next: string) => void;
  loading: boolean;
  error: string;
  // Re-reads workspace/remote-hosts.json. A caller whose submit was
  // rejected because its remote_host id no longer exists there calls this
  // instead of leaving the stale id from localStorage failing on every
  // retry; once the list lands, the fallback effect below returns to local
  // CUDA rather than silently choosing a different remote machine.
  refresh: () => void;
}

/**
 * Fetches workspace/remote-hosts.json's public listing once, and owns which
 * host id is selected. The Decode page calls this once and passes the result
 * to its execution stage; a standalone stage can own the selection itself.
 * This keeps one fetch and one persisted selection.
 *
 * Pass `skip` when a parent already owns the selection so an embedded stage
 * does not issue a redundant fetch.
 */
export function useRemoteHostSelection(skip = false): RemoteHostSelection {
  const [hosts, setHosts] = useState<RemoteHost[]>([]);
  const [envDefaultId, setEnvDefaultId] = useState<string | null>(null);
  const [selectedId, setSelectedIdState] = useState(() => readStoredHostId());
  const [loading, setLoading] = useState(!skip);
  const [error, setError] = useState("");
  const generationRef = useRef(0);

  const load = useCallback(() => {
    if (skip) return;
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    setLoading(true);
    setError("");
    void fetchRemoteHosts()
      .then((payload) => {
        if (generationRef.current !== generation) return;
        setHosts(payload.hosts);
        setEnvDefaultId(payload.env_default_id);
      })
      .catch((reason) => {
        if (generationRef.current !== generation) return;
        setError(
          reason instanceof Error
            ? reason.message
            : "The remote host list could not be read.",
        );
      })
      .finally(() => {
        if (generationRef.current === generation) setLoading(false);
      });
  }, [skip]);

  useEffect(() => {
    load();
  }, [load]);

  const setSelectedId = useCallback((next: string) => {
    setSelectedIdState(next);
    writeStoredHostId(next);
  }, []);

  // Empty means Local CUDA and is the deliberate default. Preserve an
  // explicit stored remote choice while it still exists, but never
  // auto-select the environment default or first remote host. This also
  // covers a post-refresh() list that dropped the selected id.
  useEffect(() => {
    if (skip || loading || !selectedId) return;
    if (hosts.some((host) => host.id === selectedId)) return;
    setSelectedId("");
  }, [skip, loading, hosts, selectedId, setSelectedId]);

  return {
    hosts,
    envDefaultId,
    selectedId,
    setSelectedId,
    loading,
    error,
    refresh: load,
  };
}
