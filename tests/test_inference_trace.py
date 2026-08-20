from app.core.inference_trace import finish_trace, record_generation, start_trace


def test_trace_collects_only_measurements_in_its_context():
    token = start_trace()
    record_generation({"operation": "daily_generate", "generate_ms": 10})

    trace = finish_trace(token)
    record_generation({"operation": "daily_critic", "generate_ms": 20})

    assert trace == [
        {"operation": "daily_generate", "generate_ms": 10}
    ]
