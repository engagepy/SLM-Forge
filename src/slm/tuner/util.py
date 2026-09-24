"""Small helpers shared by the Tuner's tools and its specialists."""


def clip(v, n: int = 300):
    """Shorten what goes back to the model: long strings get an ellipsis, lists their first 12 items."""
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "…"
    if isinstance(v, list):
        return [clip(x, n) for x in v[:12]] + (["…"] if len(v) > 12 else [])
    if isinstance(v, dict):
        return {k: clip(x, n) for k, x in v.items()}
    return v
