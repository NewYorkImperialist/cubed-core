import { useMemo } from "react";

import type { TrackerDump, TrackerSpan } from "../localContracts";

const omText = (orientation: string[] | string) =>
  Array.isArray(orientation) ? orientation.join("/") : orientation;

type SpanKind = "move" | "reorient" | "rest";

const KIND_COLOR: Record<SpanKind, string> = {
  move: "#4caf50",
  reorient: "#ff9800",
  rest: "#9e9e9e",
};

const KIND_LABEL: Record<SpanKind, string> = {
  move: "move",
  reorient: "regrip / reorient",
  rest: "rest",
};

function spanKind(spans: TrackerSpan[], index: number): SpanKind {
  const path = spans[index].top[0]?.path ?? "";
  if (path && path !== "(stay)") return "move";
  if (
    index > 0 &&
    omText(spans[index].top[0]?.om ?? "") !==
      omText(spans[index - 1].top[0]?.om ?? "")
  ) {
    return "reorient";
  }
  return "rest";
}

/** Trellis candidates for the span containing the current video frame. */
export function TrellisBeam({
  dump,
  frameIdx,
  seekToFrame,
}: {
  dump: TrackerDump;
  frameIdx: number;
  seekToFrame: (frame: number) => void;
}) {
  const spans = dump.spans;
  const currentSpanIndex = useMemo(() => {
    let best = -1;
    for (let index = 0; index < spans.length; index += 1) {
      if (spans[index].f0 <= frameIdx) best = index;
      if (spans[index].f0 <= frameIdx && frameIdx <= spans[index].f1) {
        return index;
      }
    }
    return best;
  }, [spans, frameIdx]);
  const span =
    currentSpanIndex >= 0 ? spans[currentSpanIndex] : null;

  return (
    <div className="dash-beam">
      <h4>Trellis beam</h4>
      {spans.length === 0 && !dump.bridge?.length && (
        <div className="dash-muted">No trellis data in this artifact.</div>
      )}

      {span && (
        <>
          <div className="dash-beam-head">
            <button
              className="anchor-chip"
              disabled={currentSpanIndex <= 0}
              onClick={() =>
                seekToFrame(spans[currentSpanIndex - 1].f1)
              }
            >
              ‹
            </button>
            <span>
              span {currentSpanIndex + 1}/{spans.length} · f{span.f0}–{span.f1}{" "}
              <span
                className="dash-kind"
                style={{
                  color: KIND_COLOR[spanKind(spans, currentSpanIndex)],
                }}
              >
                {KIND_LABEL[spanKind(spans, currentSpanIndex)]}
              </span>
              {span.event != null && (
                <span className="dash-beam-event"> · event f{span.event}</span>
              )}
            </span>
            <button
              className="anchor-chip"
              disabled={currentSpanIndex >= spans.length - 1}
              onClick={() =>
                seekToFrame(spans[currentSpanIndex + 1].f1)
              }
            >
              ›
            </button>
          </div>
          <table className="dash-beam-table">
            <colgroup>
              <col className="dash-path-column" />
              <col className="dash-om-column" />
              <col className="dash-score-column" />
            </colgroup>
            <thead>
              <tr>
                <th>path</th>
                <th>om</th>
                <th className="dash-score">score</th>
              </tr>
            </thead>
            <tbody>
              {span.top.map((candidate, index) => (
                <tr
                  key={index}
                  className={index === 0 ? "cand-top" : ""}
                >
                  <td>
                    <code>{candidate.path}</code>
                  </td>
                  <td className="dash-om">{omText(candidate.om)}</td>
                  <td className="dash-score">
                    {candidate.score.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {dump.bridge && dump.bridge.length > 0 && (
        <div className="dash-bridge">
          <h4>Terminal bridge → solved</h4>
          <code>{dump.bridge.join(" ")}</code>
        </div>
      )}

      {spans.length > 0 && (
        <div className="dash-span-ribbon">
          {spans.map((candidateSpan, index) => (
            <button
              key={index}
              className={
                "dash-span-chip" +
                (index === currentSpanIndex ? " active" : "")
              }
              style={{
                borderColor: KIND_COLOR[spanKind(spans, index)],
              }}
              title={`f${candidateSpan.f0}–${candidateSpan.f1}  ${
                KIND_LABEL[spanKind(spans, index)]
              }  ${candidateSpan.top[0]?.path ?? ""}`}
              onClick={() => seekToFrame(candidateSpan.f1)}
            >
              {candidateSpan.top[0]?.path &&
              candidateSpan.top[0].path !== "(stay)"
                ? candidateSpan.top[0].path
                : "·"}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
