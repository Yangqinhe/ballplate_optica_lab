"""Shared bounded static-measurement sequence; no files, devices or NumPy.

Repeat count is the number of CCD samples per stationary command. It is NOT
the number of temporal input images in the four-image optical controller.
"""


def bracketed_states(frames_per_state=16):
    """Exact BG + Z-A-Z-B-Z / BA / AB + BG sequence, eight or sixteen repeats."""
    if type(frames_per_state) is not int or frames_per_state not in (8, 16):
        raise ValueError("static repeats must be the explicit integer 8 or 16")
    rows = [{"id": "BG_ZERO_BEFORE", "kind": "BG", "time_block": None, "repeat": frames_per_state}]
    for block, order in enumerate((("A", "B"), ("B", "A"), ("A", "B"))):
        for suffix, kind in (("Z0", "Z"), (order[0], order[0]), ("Z1", "Z"),
                             (order[1], order[1]), ("Z2", "Z")):
            rows.append({"id": f"TB{block}_{suffix}", "kind": kind,
                         "time_block": block, "repeat": frames_per_state})
    rows.append({"id": "BG_ZERO_AFTER", "kind": "BG", "time_block": None, "repeat": frames_per_state})
    return rows


def observability_states(axis_family):
    """Fixed-mask input contrasts, not the historical six SLM contrasts."""
    if axis_family not in ("POS", "VEL"):
        raise ValueError("observability axis family must be POS or VEL")
    axes=("X", "Y") if axis_family=="POS" else ("VX", "VY")
    names=[("BG0", "BG", "REF"), ("REF0", "INPUT", "REF"),
           (axes[0]+"_PLUS", "INPUT", axes[0]+"_PLUS"), ("REF1", "INPUT", "REF"),
           (axes[0]+"_MINUS", "INPUT", axes[0]+"_MINUS"), ("REF2", "INPUT", "REF"),
           (axes[1]+"_PLUS", "INPUT", axes[1]+"_PLUS"), ("REF3", "INPUT", "REF"),
           (axes[1]+"_MINUS", "INPUT", axes[1]+"_MINUS"), ("REF4", "INPUT", "REF"),
           ("BG1", "BG", "REF")]
    return [dict(id=name,kind=kind,input_key=key,time_block=None,repeat=8) for name,kind,key in names]
