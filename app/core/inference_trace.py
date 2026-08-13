from contextvars import ContextVar, Token


_TRACE: ContextVar[list[dict] | None] = ContextVar(
    "vlm_inference_trace",
    default=None,
)


def start_trace() -> Token:
    return _TRACE.set([])


def record_generation(measurement: dict) -> None:
    trace = _TRACE.get()
    if trace is not None:
        trace.append(measurement)


def finish_trace(token: Token) -> list[dict]:
    trace = list(_TRACE.get() or [])
    _TRACE.reset(token)
    return trace
