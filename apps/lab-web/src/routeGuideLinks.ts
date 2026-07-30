export const ROUTE_GUIDE_REPOSITORY_URL =
  "https://github.com/KingBobJoeIV/cubed-core";
export const ROUTE_GUIDE_LINK_TARGET = "_blank";
export const ROUTE_GUIDE_LINK_REL = "noopener noreferrer";

export function routeGuideDocumentHref(path: string): string {
  return `${ROUTE_GUIDE_REPOSITORY_URL}/blob/main/${path}`;
}
