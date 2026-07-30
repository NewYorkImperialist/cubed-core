"""Device-resident stateful-OM dynamic programming primitives.

The public decoder still owns orientation reachability and candidate semantics.
This module owns only the dense tensor transaction after those small structural
tables have been derived: a resident ``T[read, orientation, state]`` score grid
enters, and resident per-candidate/final-orientation scores leave.  No candidate
or score tensor is copied to the host here.

Torch is imported lazily so CPU/default decoder imports stay unchanged.  The
implementation also accepts a CPU Torch tensor, which gives local tests an exact
oracle for the CUDA transaction without requiring a GPU.
"""

from __future__ import annotations

from importlib import import_module


# Bound the largest ``torch.where`` transition temporary. This is a memory
# safety envelope, not a decode knob: candidate rows are independent, so
# chunking cannot change arithmetic order or selection semantics.
_TRANSITION_TEMP_BYTES = 256 * 1024 * 1024


def _torch():
    return import_module("torch")


def dp_scores_stateful_om_device(
    cols,
    score_grid,
    *,
    first_reach,
    step_sources,
    tail_sources=None,
    end_pinned=False,
):
    """Run monotone state-prefix + orientation-path DP on one Torch device.

    Parameters mirror the dense core of
    :func:`detect.scrub_decode._dp_scores_stateful_om` after graph reachability
    has been factored into three small host-side tables:

    ``first_reach``
        orientation indices allowed for the first read;
    ``step_sources[w][dst]``
        source orientations that can reach ``dst`` between reads ``w-1`` and
        ``w`` (entry zero is ignored and may be ``None``);
    ``tail_sources[dst]``
        optional sources that can reach ``dst`` after the final read.

    Both returned tensors remain on ``score_grid.device``.  Grouping candidates
    by path length preserves the NumPy implementation's max/add/cummax order.
    """
    torch = _torch()
    if not isinstance(score_grid, torch.Tensor):
        raise TypeError("score_grid must be a Torch tensor")
    if score_grid.ndim != 3 or score_grid.dtype != torch.float64:
        raise ValueError("score_grid must be float64[reads, orientations, states]")
    n_reads, n_om, n_states = map(int, score_grid.shape)
    if n_reads < 1:
        raise ValueError("score_grid must contain at least one read")
    if len(step_sources) != n_reads:
        raise ValueError("step_sources must have one entry per read")

    n_cand = len(cols)
    scores = torch.full(
        (n_cand,), -torch.inf, dtype=torch.float64,
        device=score_grid.device)
    final_scores = torch.full(
        (n_cand, n_om), -torch.inf, dtype=torch.float64,
        device=score_grid.device)
    if n_cand == 0:
        return scores, final_scores

    first = sorted({int(oi) for oi in first_reach})
    if any(oi < 0 or oi >= n_om for oi in first):
        raise ValueError("first_reach contains an invalid orientation")

    by_length = {}
    for ci, state_cols in enumerate(cols):
        values = tuple(int(ix) for ix in state_cols)
        if not values:
            raise ValueError("candidate state paths must be non-empty")
        if any(ix < 0 or ix >= n_states for ix in values):
            raise ValueError("candidate state index is out of range")
        by_length.setdefault(len(values), []).append((ci, values))

    def source_mask(table):
        if table is None or len(table) != n_om:
            raise ValueError("invalid orientation source table")
        rows = []
        for dst in range(n_om):
            values = {int(src) for src in table[dst]}
            if any(src < 0 or src >= n_om for src in values):
                raise ValueError(
                    "orientation source table contains an invalid index")
            rows.append([src in values for src in range(n_om)])
        return torch.tensor(
            rows, dtype=torch.bool, device=score_grid.device)

    # One dense 24x24 relation per read replaces one kernel launch per
    # destination orientation.  The temporary [candidate,dst,src,state-pos]
    # block is small at scrub's bounded candidate/path dimensions and preserves
    # the exact max-plus recurrence.
    step_masks = [None] + [
        source_mask(step_sources[read_i])
        for read_i in range(1, n_reads)
    ]
    tail_mask = source_mask(tail_sources) if tail_sources is not None else None
    first_mask = torch.zeros(
        n_om, dtype=torch.bool, device=score_grid.device)
    if first:
        first_mask[torch.tensor(
            first, dtype=torch.int64, device=score_grid.device)] = True
    neg_inf = torch.tensor(
        -torch.inf, dtype=torch.float64, device=score_grid.device)

    def gather_read(read_i, C):
        # score_grid[read_i, :, C] is [orientation,batch,state-pos].
        return score_grid[read_i, :, C].permute(1, 0, 2)

    def transition(D, mask):
        return torch.amax(
            torch.where(mask[None, :, :, None],
                        D[:, None, :, :], neg_inf),
            dim=2,
        )

    for word_states, entries in by_length.items():
        # ``transition`` materializes [candidate,dst,src,state-pos] float64.
        # Chunk independently scored candidates so the valid structural
        # envelope cannot turn into a multi-gigabyte transient allocation.
        bytes_per_candidate = max(
            1, n_om * n_om * int(word_states) * 8)
        chunk = max(1, _TRANSITION_TEMP_BYTES // bytes_per_candidate)
        for c0 in range(0, len(entries), chunk):
            part = entries[c0:c0 + chunk]
            candidate_indices = torch.tensor(
                [ci for ci, _values in part], dtype=torch.int64,
                device=score_grid.device)
            C = torch.tensor(
                [values for _ci, values in part], dtype=torch.int64,
                device=score_grid.device)
            first_values = gather_read(0, C)
            D = torch.where(
                first_mask[None, :, None], first_values, neg_inf)

            for read_i in range(1, n_reads):
                previous = transition(D, step_masks[read_i])
                monotone = torch.cummax(previous, dim=2).values
                D = gather_read(read_i, C) + monotone

            if tail_mask is not None:
                D = transition(D, tail_mask)

            fs = D[:, :, -1] if end_pinned else torch.amax(D, dim=2)
            final_scores[candidate_indices] = fs
            scores[candidate_indices] = torch.amax(fs, dim=1)

    return scores, final_scores


def score_fixed_timing_programs_device(
    programs,
    score_grid,
    *,
    row_frames,
    rotation_frames,
    om_neighbors,
    start_ois,
):
    """Score fixed physical timings without copying the read grid to host.

    ``programs`` is an ordered sequence of ``(state_columns, timing)`` pairs.
    Each timing has exactly one frame per physical move and therefore one fewer
    entry than its state path.  Programs are independent, so memory chunking
    preserves candidate order and arithmetic order exactly.

    The recurrence is the literal tensor form of
    :func:`detect.scrub_decode._score_fixed_timing_candidate`: rotations happen
    before moves and reads at the same frame, reads are accumulated one at a
    time in float64, and equal-score orientation paths keep the
    lexicographically smallest trace while unioning every ambiguous move
    position.  The returned list contains the same per-final-OM dictionaries as
    the scalar oracle.  Only those bounded result dictionaries leave the
    device; the ``[read, orientation, state]`` grid remains resident.
    """
    torch = _torch()
    if not isinstance(score_grid, torch.Tensor):
        raise TypeError("score_grid must be a Torch tensor")
    if score_grid.ndim != 3 or score_grid.dtype != torch.float64:
        raise ValueError(
            "score_grid must be float64[reads, orientations, states]")
    n_reads, n_om, n_states = map(int, score_grid.shape)
    frames = tuple(int(frame) for frame in row_frames)
    if len(frames) != n_reads:
        raise ValueError("row_frames must have one entry per score-grid row")

    neighbors = []
    if len(om_neighbors) != n_om:
        raise ValueError("om_neighbors must have one entry per orientation")
    for src, raw in enumerate(om_neighbors):
        values = sorted({src} | {int(dst) for dst in raw})
        if any(dst < 0 or dst >= n_om for dst in values):
            raise ValueError("om_neighbors contains an invalid orientation")
        neighbors.append(tuple(values))
    inverse_sources = [
        tuple(src for src, destinations in enumerate(neighbors)
              if dst in destinations)
        for dst in range(n_om)
    ]
    starts = sorted({int(oi) for oi in start_ois})
    if any(oi < 0 or oi >= n_om for oi in starts):
        raise ValueError("start_ois contains an invalid orientation")

    normalized = []
    max_word = 0
    for state_columns, timing in programs:
        cols = tuple(int(value) for value in state_columns)
        times = tuple(int(value) for value in timing)
        if not cols:
            raise ValueError("fixed-timing state paths must be non-empty")
        if len(times) != len(cols) - 1:
            raise ValueError(
                "fixed timing does not cover every physical move")
        if times != tuple(sorted(times)):
            # The scalar oracle sorts its event timeline and would therefore
            # reinterpret trace positions for a malformed, nonchronological
            # timing.  Reject that foreign payload so the surrounding
            # transaction retries the literal scalar implementation.
            raise ValueError("fixed timing must be chronological")
        if any(value < 0 or value >= n_states for value in cols):
            raise ValueError("fixed-timing state column is out of range")
        normalized.append((cols, times))
        max_word = max(max_word, len(times))
    if not normalized:
        return []

    # The largest live tensors are score/active plus trace/ambiguity.  Reuse
    # the module's existing operational memory envelope; this changes only how
    # many independent program rows are resident at once, never their scores.
    bytes_per_program = max(
        1, n_om * (8 + 1 + max_word * (2 + 1)))
    chunk_size = max(1, _TRANSITION_TEMP_BYTES // bytes_per_program)
    output = []
    for chunk_start in range(0, len(normalized), chunk_size):
        chunk = normalized[chunk_start:chunk_start + chunk_size]
        output.extend(_score_fixed_timing_chunk(
            chunk,
            score_grid,
            row_frames=frames,
            rotation_frames=rotation_frames,
            inverse_sources=inverse_sources,
            start_ois=starts,
            max_word=max_word,
        ))
    return output


def _score_fixed_timing_chunk(
    programs,
    score_grid,
    *,
    row_frames,
    rotation_frames,
    inverse_sources,
    start_ois,
    max_word,
):
    """One independent fixed-timing program chunk on ``score_grid.device``."""
    torch = _torch()
    device = score_grid.device
    n_programs = len(programs)
    _n_reads, n_om, _n_states = map(int, score_grid.shape)
    neg_inf = torch.tensor(-torch.inf, dtype=torch.float64, device=device)

    scores = torch.full(
        (n_programs, n_om), -torch.inf,
        dtype=torch.float64, device=device)
    active = torch.zeros(
        (n_programs, n_om), dtype=torch.bool, device=device)
    if start_ois:
        start_t = torch.tensor(start_ois, dtype=torch.int64, device=device)
        scores[:, start_t] = 0.0
        active[:, start_t] = True
    traces = torch.full(
        (n_programs, n_om, max_word), -1,
        dtype=torch.int16, device=device)
    ambiguities = torch.zeros(
        (n_programs, n_om, max_word),
        dtype=torch.bool, device=device)

    # A read observes the state after every move at or before its frame because
    # the scalar event order is rotation(0), move(1), read(2).
    read_state_columns = []
    for read_frame in row_frames:
        row = []
        for cols, timing in programs:
            move_count = sum(frame <= read_frame for frame in timing)
            row.append(cols[move_count])
        read_state_columns.append(row)
    read_state_columns_t = (
        torch.tensor(read_state_columns, dtype=torch.int64, device=device)
        if read_state_columns else None
    )

    rotations_by_frame = {}
    for frame in rotation_frames:
        frame = int(frame)
        rotations_by_frame[frame] = rotations_by_frame.get(frame, 0) + 1
    moves_by_frame = {}
    for pi, (_cols, timing) in enumerate(programs):
        for move_i, frame in enumerate(timing):
            moves_by_frame.setdefault(int(frame), []).append((pi, move_i))
    reads_by_frame = {}
    for read_i, frame in enumerate(row_frames):
        reads_by_frame.setdefault(int(frame), []).append(read_i)
    all_frames = sorted(
        set(rotations_by_frame) | set(moves_by_frame) | set(reads_by_frame))

    def transition_once():
        nonlocal scores, active, traces, ambiguities
        next_scores = torch.full_like(scores, -torch.inf)
        next_active = torch.zeros_like(active)
        next_traces = torch.full_like(traces, -1)
        next_ambiguities = torch.zeros_like(ambiguities)
        for dst, sources in enumerate(inverse_sources):
            dst_score = next_scores[:, dst]
            dst_active = next_active[:, dst]
            dst_trace = next_traces[:, dst]
            dst_ambiguity = next_ambiguities[:, dst]
            for src in sources:
                candidate_active = active[:, src]
                candidate_score = scores[:, src]
                candidate_trace = traces[:, src]
                candidate_ambiguity = ambiguities[:, src]

                empty = candidate_active & ~dst_active
                both = candidate_active & dst_active
                greater = both & (candidate_score > dst_score)
                equal = both & (candidate_score == dst_score)
                replace = empty | greater
                # These updates are deliberately unconditional tensor ops.
                # A data-dependent Python ``if torch.any(...)`` would force a
                # device synchronization and scalar D2H transfer inside every
                # source/destination/rotation recurrence.
                dst_score = torch.where(
                    replace, candidate_score, dst_score)
                dst_trace = torch.where(
                    replace[:, None], candidate_trace, dst_trace)
                dst_ambiguity = torch.where(
                    replace[:, None], candidate_ambiguity, dst_ambiguity)
                dst_active = dst_active | empty

                if max_word:
                    trace_diff = dst_trace != candidate_trace
                    merged_ambiguity = (
                        dst_ambiguity | candidate_ambiguity | trace_diff)
                    has_diff = torch.any(trace_diff, dim=1)
                    first_diff = torch.argmax(
                        trace_diff.to(torch.int64), dim=1)
                    gather_index = first_diff[:, None]
                    candidate_first = torch.gather(
                        candidate_trace, 1, gather_index).squeeze(1)
                    current_first = torch.gather(
                        dst_trace, 1, gather_index).squeeze(1)
                    candidate_less = (
                        equal & has_diff
                        & (candidate_first < current_first))
                    dst_trace = torch.where(
                        candidate_less[:, None], candidate_trace, dst_trace)
                    dst_ambiguity = torch.where(
                        equal[:, None], merged_ambiguity, dst_ambiguity)

            next_scores[:, dst] = dst_score
            next_active[:, dst] = dst_active
            next_traces[:, dst] = dst_trace
            next_ambiguities[:, dst] = dst_ambiguity
        scores = next_scores
        active = next_active
        traces = next_traces
        ambiguities = next_ambiguities

    om_row = torch.arange(n_om, dtype=torch.int16, device=device)[None, :]
    for frame in all_frames:
        for _ in range(rotations_by_frame.get(frame, 0)):
            transition_once()

        move_events = moves_by_frame.get(frame, ())
        if move_events:
            program_index = torch.tensor(
                [row[0] for row in move_events],
                dtype=torch.int64, device=device)
            move_index = torch.tensor(
                [row[1] for row in move_events],
                dtype=torch.int64, device=device)
            traces[program_index, :, move_index] = om_row

        for read_i in reads_by_frame.get(frame, ()):
            state_columns = read_state_columns_t[read_i]
            # index_select returns [orientation, program]; transpose to the
            # live [program, orientation] frontier.
            emission = score_grid[read_i].index_select(
                1, state_columns).transpose(0, 1)
            finite = torch.isfinite(emission)
            active = active & finite
            scores = torch.where(
                active, scores + emission, neg_inf)

    score_host = scores.detach().cpu().numpy()
    active_host = active.detach().cpu().numpy()
    trace_host = traces.detach().cpu().numpy()
    ambiguity_host = ambiguities.detach().cpu().numpy()
    lengths = [len(timing) for _cols, timing in programs]
    output = []
    for pi, word_len in enumerate(lengths):
        values = {}
        for oi in range(n_om):
            if not active_host[pi, oi]:
                continue
            trace = tuple(
                int(value) for value in trace_host[pi, oi, :word_len])
            ambiguity = frozenset(
                int(value) for value in
                ambiguity_host[pi, oi, :word_len].nonzero()[0])
            values[oi] = (float(score_host[pi, oi]), trace, ambiguity)
        output.append(values)
    return output
