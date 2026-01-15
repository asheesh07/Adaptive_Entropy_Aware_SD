def test_latency_speedup(
    speculative_engine,
    vanilla_decoder,
    prompt,
):
    """
    Speculative decoding should achieve lower latency per token
    than vanilla decoding (on average).
    """

    # --- Run vanilla decoding ---
    vanilla_output = vanilla_decoder.decode(prompt, max_tokens=50)
    vanilla_metrics = vanilla_decoder.performance_tracker.summary()

    # --- Run speculative decoding ---
    spec_output = speculative_engine.decode(prompt, max_tokens=50)
    spec_metrics = speculative_engine.performance_tracker.summary()

    # --- Sanity: outputs must match ---
    assert spec_output == vanilla_output

    vanilla_latency = vanilla_metrics["latency_per_token_ms"]
    spec_latency = spec_metrics["latency_per_token_ms"]

    # --- Core assertion ---
    assert spec_latency < vanilla_latency, (
        f"Expected speculative latency < vanilla latency "
        f"(spec={spec_latency}ms, vanilla={vanilla_latency}ms)"
    )
