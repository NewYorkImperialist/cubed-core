import {
  FormEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  Link,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";

import { LocalLabelWorkbench } from "./components/LocalLabelWorkbench";
import { PublishedDemoPage } from "./components/PublishedDemoPage";
import { ReconstructWorkbench } from "./components/ReconstructWorkbench";
import { RunsWorkbench } from "./components/RunsWorkbench";
import { VideoImportWorkbench } from "./components/VideoImportWorkbench";
import {
  PUBLISHED_DEMO_ROUTE,
  publishedDemoRedirectFor,
  WORKBENCH_NAV_GROUPS,
  workbenchNavIdForPath,
} from "./workbenchRoutes";
import {
  fetchCapabilities,
  fetchHealth,
  RequestError,
  setAdminToken,
} from "./api";
import { apiStateFromProbe } from "./apiSession";
import type { ApiState } from "./apiSession";
import type { Capabilities } from "./types";

const SOURCE_REPOSITORY_URL = "https://github.com/KingBobJoeIV/cubed-core";

function StickerMark() {
  return (
    <span className="sticker-mark" aria-hidden="true">
      <i className="sticker sticker-blue" />
      <i className="sticker sticker-red" />
      <i className="sticker sticker-yellow" />
      <i className="sticker sticker-green" />
    </span>
  );
}

function SourceLink() {
  return (
    <a
      className="source-link"
      href={SOURCE_REPOSITORY_URL}
      target="_blank"
      rel="noopener noreferrer"
      aria-label="Open the Cubed Core repository on GitHub"
    >
      GitHub
      <span aria-hidden="true">↗</span>
    </a>
  );
}

function LabShell() {
  const location = useLocation();
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [apiState, setApiState] = useState<ApiState>("checking");
  const [authGeneration, setAuthGeneration] = useState(0);
  const probeActiveRef = useRef(true);

  const runProbe = useCallback(async () => {
    try {
      await fetchHealth(AbortSignal.timeout(8000));
    } catch {
      if (probeActiveRef.current) {
        setCapabilities(null);
        setApiState("offline");
      }
      return;
    }

    if (!probeActiveRef.current) return;

    try {
      const nextCapabilities = await fetchCapabilities(
        AbortSignal.timeout(8000),
      );
      if (!probeActiveRef.current) return;
      setCapabilities(nextCapabilities);
      setApiState(
        apiStateFromProbe({ healthOk: true, capabilitiesStatus: null }),
      );
    } catch (error) {
      if (!probeActiveRef.current) return;
      setCapabilities(null);
      if (error instanceof RequestError) {
        setApiState(
          apiStateFromProbe({
            healthOk: true,
            capabilitiesStatus: error.status,
          }),
        );
      } else {
        setApiState("offline");
      }
    }
  }, []);

  useEffect(() => {
    probeActiveRef.current = true;
    void runProbe();
    return () => {
      probeActiveRef.current = false;
    };
  }, [runProbe]);

  const handleAdminTokenSaved = useCallback(() => {
    setAuthGeneration((generation) => generation + 1);
    void runProbe();
  }, [runProbe]);

  const demoRoute =
    location.pathname === "/" || location.pathname === PUBLISHED_DEMO_ROUTE;

  return (
    <>
      <div className="desktop-only-notice">
        <StickerMark />
        <strong>Cubed Core is a desktop workbench.</strong>
        <p>Open it in a desktop-sized browser window to inspect video runs.</p>
      </div>
      <div className="lab-shell">
        <header className="lab-topbar">
          <Link className="brand" to="/demo" aria-label="Cubed Core demo">
            <StickerMark />
            <strong>Cubed Core</strong>
          </Link>

          <nav className="lab-nav" aria-label="Workbench pages">
            {WORKBENCH_NAV_GROUPS.map((group) => (
              <div
                className="nav-group"
                key={group.label ?? "overview"}
                role="group"
                aria-label={group.label ?? "Overview"}
              >
                {group.label && (
                  <span className="nav-group-label" aria-hidden="true">
                    {group.label}
                  </span>
                )}
                {group.items.map((item) => {
                  const active =
                    workbenchNavIdForPath(location.pathname) === item.id;
                  return (
                    <Link
                      key={item.id}
                      className={`nav-item${active ? " nav-item-active" : ""}`}
                      to={item.to}
                      aria-current={active ? "page" : undefined}
                      aria-label={
                        item.badgeDetail
                          ? `${item.label}. ${item.badgeDetail}`
                          : undefined
                      }
                      title={item.badgeDetail}
                    >
                      <span className="nav-name">{item.label}</span>
                      {item.badge && (
                        <span className="nav-badge">{item.badge}</span>
                      )}
                    </Link>
                  );
                })}
              </div>
            ))}
          </nav>

          <div className="topbar-utilities">
            {!demoRoute && apiState !== "online" && (
              <span className={`topbar-service topbar-service-${apiState}`}>
                {apiState === "unauthorized"
                  ? "Authentication required"
                  : apiState === "offline"
                    ? "Service offline"
                    : "Checking service"}
              </span>
            )}
            <SourceLink />
          </div>
        </header>

        <main className="lab-main" key={authGeneration}>
          {!demoRoute && apiState === "unauthorized" && (
            <div className="workbench-auth-notice">
              <div>
                <strong>This network workbench requires its admin token.</strong>
                <p>Use the authenticated URL printed by the server.</p>
              </div>
              <AdminTokenPanel onSaved={handleAdminTokenSaved} />
            </div>
          )}
          <Outlet context={{ capabilities, apiState }} />
        </main>
      </div>
    </>
  );
}

function AdminTokenPanel({ onSaved }: { onSaved: () => void }) {
  const [value, setValue] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAdminToken(value);
    onSaved();
  }

  return (
    <form className="admin-token-panel" onSubmit={handleSubmit}>
      <label className="field-label" htmlFor="admin-token-input">
        <span>Admin token</span>
      </label>
      <input
        className="text-input"
        id="admin-token-input"
        type="password"
        autoComplete="off"
        value={value}
        onChange={(event) => setValue(event.target.value)}
      />
      <button className="button button-quiet button-wide" type="submit">
        Save
      </button>
      <p>
        For network or public access, use the #admin= link printed by the
        server.
      </p>
    </form>
  );
}

function LegacyDecodeRoute() {
  const location = useLocation();
  const redirect = publishedDemoRedirectFor(location.pathname, location.hash);
  if (redirect) return <Navigate to={redirect} replace />;
  return <Navigate to={`/decode${location.search}`} replace />;
}

function App() {
  return (
    <Routes>
      <Route path="/phone" element={<Navigate to="/import" replace />} />
      <Route path="/capture/:code" element={<Navigate to="/import" replace />} />
      <Route element={<LabShell />}>
        <Route path="/" element={<Navigate to={PUBLISHED_DEMO_ROUTE} replace />} />
        <Route path="/import" element={<VideoImportWorkbench />} />
        <Route path="/capture" element={<Navigate to="/import" replace />} />
        <Route
          path="/capture/setup"
          element={<Navigate to="/import" replace />}
        />
        <Route
          path="/capture/record"
          element={<Navigate to="/import" replace />}
        />
        <Route path="/label" element={<LocalLabelWorkbench />} />
        <Route path={PUBLISHED_DEMO_ROUTE} element={<PublishedDemoPage />} />
        <Route path="/decode" element={<ReconstructWorkbench />} />
        <Route path="/runs" element={<RunsWorkbench />} />
        <Route path="/reconstruct" element={<LegacyDecodeRoute />} />
        <Route path="/analyze" element={<Navigate to="/runs" replace />} />
        <Route path="/analysis" element={<Navigate to="/runs" replace />} />
        <Route path="/research" element={<Navigate to="/demo" replace />} />
        <Route path="*" element={<Navigate to="/demo" replace />} />
      </Route>
    </Routes>
  );
}

export default App;
