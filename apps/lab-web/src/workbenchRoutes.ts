export type WorkbenchNavId =
  | "demo"
  | "decode"
  | "runs"
  | "import"
  | "label";

export interface WorkbenchNavItem {
  id: WorkbenchNavId;
  label: string;
  to: string;
  paths: readonly string[];
  /** Short state chip rendered beside the label. */
  badge?: string;
  /** Longer sentence behind the chip, for a tooltip and assistive text. */
  badgeDetail?: string;
}

export interface WorkbenchNavGroup {
  /** null keeps the first entry above the first divider. */
  label: string | null;
  items: readonly WorkbenchNavItem[];
}

/** The desktop workbench keeps execution separate from run inspection. */
export const WORKBENCH_NAV_GROUPS: readonly WorkbenchNavGroup[] = [
  {
    label: null,
    items: [
      {
        id: "demo",
        label: "Demo",
        to: "/demo",
        paths: ["/", "/demo"],
      },
      {
        id: "decode",
        label: "Decode",
        to: "/decode",
        paths: ["/decode", "/reconstruct"],
      },
      {
        id: "runs",
        label: "Runs",
        to: "/runs",
        paths: [
          "/runs",
          "/analyze",
          "/analysis",
        ],
      },
    ],
  },
  {
    label: "Data tools",
    items: [
      {
        id: "import",
        label: "Add video",
        to: "/import",
        paths: ["/capture", "/capture/setup", "/capture/record", "/import"],
      },
      {
        id: "label",
        label: "Label",
        to: "/label",
        paths: ["/label"],
      },
    ],
  },
];

export const WORKBENCH_NAV_ITEMS: readonly WorkbenchNavItem[] =
  WORKBENCH_NAV_GROUPS.flatMap((group) => group.items);

/** The published demo has its own page, reached from the rail and direct links. */
export const PUBLISHED_DEMO_ROUTE = "/demo";
export const LEGACY_PUBLISHED_DEMO_HASH = "#published-demo";

/**
 * Return the route a legacy published-demo deep link should land on.
 *
 * The demo used to live inside Reconstruct behind a fragment, so existing
 * links, documentation, and bookmarks still carry that fragment.
 */
export function publishedDemoRedirectFor(
  pathname: string,
  hash: string,
): string | null {
  if (hash !== LEGACY_PUBLISHED_DEMO_HASH) return null;
  const reconstructPaths = [
    "/runs",
    "/decode",
    "/reconstruct",
  ];
  return reconstructPaths.includes(pathname) ? PUBLISHED_DEMO_ROUTE : null;
}

export function workbenchNavIdForPath(
  pathname: string,
): WorkbenchNavId | null {
  return (
    WORKBENCH_NAV_ITEMS.find((item) => item.paths.includes(pathname))?.id ?? null
  );
}
