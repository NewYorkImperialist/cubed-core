"""Diagnostic-only edit-distance comparison between a decode result and the
capture's smart-cube teacher moves.

This module never runs during a decode and never feeds a decode. It exists so
a completed job can be inspected against the recording's smart-cube teacher
sidecar, when one is attached, purely for human review. Nothing here is a
success metric. For a completed Decode result, the supported statements are
“The job completed on this input.” and, after the server replay check, “The
sequence replayed to solved.” Reach-LL belongs only to a separately named
closed evaluation. Full-sequence accuracy and edit distance remain diagnostic
only. Every document this module builds carries ``diagnostic_only: true`` so a
caller cannot mistake it for an evaluation block.

Move tokens on both sides are normalized through the same canonical outer-face
parser the decode job replay check uses (``cube.notation.parse_move``), so a
comparison never silently drifts between wide/slice/rotation conventions and
the plain ``U R F D L B`` vocabulary the sealed decode-result-v1 documents and
the CubeSession v2 teacher sidecar both already use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DIAGNOSTIC_SCHEMA = "cubed-core/decode-editdist-diagnostic-v1"
DIAGNOSTIC_SCHEMA_VERSION = 1

# Defensive cap: this is a diagnostic convenience, not a bulk-data endpoint,
# and the O(n*m) DP below should never be asked to run on unbounded input.
# A full solve plus reasonable slack is a few hundred moves.
MAX_SEQUENCE_MOVES = 400


class DecodeDiagnosticsError(Exception):
    """Raised when an edit-distance diagnostic cannot be built."""


def _normalize_moves(moves: Sequence[Any], *, field: str) -> list[str]:
    """Parse and re-render every token through the canonical move notation.

    This is the same parser ``decode_jobs.replay_decode_result`` uses to
    replay a result's moves, so both sequences compared here land in the
    identical vocabulary: the 18 fixed-center face turns, no wide moves, no
    slice moves, no whole-cube rotations.
    """

    try:
        from .cube.notation import parse_move
    except ModuleNotFoundError as exc:
        raise DecodeDiagnosticsError(
            "the decode diagnostics module requires the decode dependency extra"
        ) from exc

    normalized: list[str] = []
    for index, token in enumerate(moves):
        try:
            move = parse_move(token)
        except (TypeError, ValueError) as exc:
            raise DecodeDiagnosticsError(
                f"{field}[{index}] is not a canonical outer-face move token: {token!r}"
            ) from exc
        normalized.append(str(move))
    return normalized


def _levenshtein_ops(
    decoded: Sequence[str],
    reference: Sequence[str],
) -> tuple[int, list[dict[str, Any]]]:
    """Standard token-level Levenshtein DP, backtraced to an ordered ops list.

    Costs are unit insert/delete/substitute. "Insert" means decoded carries a
    token the reference does not (an extra move); "delete" means the
    reference carries a token decoded is missing (a dropped move). This is the
    conventional word-error-rate framing with decoded as the hypothesis and
    the teacher recording as the reference.
    """

    n = len(decoded)
    m = len(reference)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if decoded[i - 1] == reference[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    ops: list[dict[str, Any]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and decoded[i - 1] == reference[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(
                {
                    "op": "equal",
                    "decoded": decoded[i - 1],
                    "reference": reference[j - 1],
                    "index_decoded": i - 1,
                    "index_reference": j - 1,
                }
            )
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(
                {
                    "op": "substitute",
                    "decoded": decoded[i - 1],
                    "reference": reference[j - 1],
                    "index_decoded": i - 1,
                    "index_reference": j - 1,
                }
            )
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(
                {
                    "op": "insert",
                    "decoded": decoded[i - 1],
                    "reference": None,
                    "index_decoded": i - 1,
                    "index_reference": None,
                }
            )
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(
                {
                    "op": "delete",
                    "decoded": None,
                    "reference": reference[j - 1],
                    "index_decoded": None,
                    "index_reference": j - 1,
                }
            )
            j -= 1
        else:  # pragma: no cover - the DP recurrence makes this unreachable
            raise DecodeDiagnosticsError("edit-distance backtrace reached an invalid state")

    ops.reverse()
    return dp[n][m], ops


def build_editdist_diagnostic(
    decoded_moves: Sequence[Any],
    reference_moves: Sequence[Any],
    *,
    reference: str = "ble-teacher",
) -> dict[str, Any]:
    """Build a ``cubed-core/decode-editdist-diagnostic-v1`` document.

    ``decoded_moves`` is one decode result's ``moves`` array. ``reference_moves``
    is the capture's smart-cube teacher move tokens. Both are normalized
    through the canonical outer-face parser before alignment, so callers may
    pass either already-canonical strings or raw sidecar move fields.

    Raises ``DecodeDiagnosticsError`` for a malformed input or a sequence past
    ``MAX_SEQUENCE_MOVES``. Never raises for a legitimate zero-length sequence
    on either side.
    """

    if isinstance(decoded_moves, (str, bytes)) or not isinstance(decoded_moves, Sequence):
        raise DecodeDiagnosticsError("decoded moves must be a list of move tokens")
    if isinstance(reference_moves, (str, bytes)) or not isinstance(reference_moves, Sequence):
        raise DecodeDiagnosticsError("reference moves must be a list of move tokens")
    if len(decoded_moves) > MAX_SEQUENCE_MOVES:
        raise DecodeDiagnosticsError(
            f"decoded move sequence exceeds the {MAX_SEQUENCE_MOVES}-move diagnostic cap"
        )
    if len(reference_moves) > MAX_SEQUENCE_MOVES:
        raise DecodeDiagnosticsError(
            f"reference move sequence exceeds the {MAX_SEQUENCE_MOVES}-move diagnostic cap"
        )

    decoded = _normalize_moves(decoded_moves, field="decoded moves")
    reference_tokens = _normalize_moves(reference_moves, field="reference moves")
    distance, ops = _levenshtein_ops(decoded, reference_tokens)

    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_only": True,
        "reference": reference,
        "distance": distance,
        "decoded_length": len(decoded),
        "reference_length": len(reference_tokens),
        "ops": ops,
    }
