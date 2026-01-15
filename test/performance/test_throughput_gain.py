def test_throughput_gain(
    speculative_engine,
    vanilla_decoder,
    prompt,
):
    """
    Speculative decoding should achieve higher throughput
    (tokens/sec) than vanilla decoding.
    """

    # --- Vanilla decoding ---
    vanilla_output = vanilla_decoder.decode(prompt, max_tokens=100)
    vanilla_metrics = vanilla_decoder.performance_tracker.summary()

    # --- Speculative decoding ---
    spec_output = speculative_engine.decode(prompt, max_tokens=100)
    spec_metrics = speculative_engine.performance_tracker.summary()

    # --- Sanity: outputs must match ---
    assert spec_output == vanilla_output

    vanilla_tps = vanilla_metrics["throughput_tokens_per_sec"]
    spec_tps = spec_metrics["throughput_tokens_per_sec"]

    # --- Core assertion ---
    assert spec_tps > vanilla_tps, (
        f"Expected speculative throughput > vanilla throughput "
        f"(spec={spec_tps}, vanilla={vanilla_tps})"
    )
