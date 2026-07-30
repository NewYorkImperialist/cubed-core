"""Minimal per-span projection used by the canonical scrub decoder."""

from __future__ import annotations

Z_SIGMA = 2.0


def build_scrub_span_view(
    *,
    moves,
    meta,
    n_spans,
    n_bridge,
):
    """Return only the span metadata the scrub search reads."""
    n_solve = len(moves) - max(0, int(n_bridge))

    fit = [
        (
            float(meta[span_index][2][0][2])
            if (
                span_index < len(meta)
                and len(meta[span_index]) > 2
                and meta[span_index][2]
            )
            else 0.0
        )
        for span_index in range(n_spans)
    ]

    def frame_bounds(span_index):
        return (
            int(meta[span_index][0]) if span_index < len(meta) else 0,
            int(meta[span_index][1]) if span_index < len(meta) else 0,
        )

    return {
        "n_sp": n_spans,
        "n_solve": n_solve,
        "fit": fit,
        "meta_f": frame_bounds,
    }
