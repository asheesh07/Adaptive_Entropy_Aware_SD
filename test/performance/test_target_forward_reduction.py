def test_target_forward_calls_reduced(
    speculative_engine,
    vanilla_decoder,
    prompt,
):
    """
    Speculative decoding must use fewer target forward passes
    than vanilla decoding for the same output length.
    """

    # --- Run speculative decoding ---
    spec_output = speculative_engine.decode(prompt, max_tokens=50)
    spec_metrics = speculative_engine.performance_tracker.summary()

    # --- Run vanilla decoding ---
    vanilla_output = vanilla_decoder.decode(prompt, max_tokens=50)
    vanilla_metrics = vanilla_decoder.performance_tracker.summary()

    # --- Sanity: outputs must match ---
    assert spec_output == vanilla_output

    # --- Core assertion ---
    assert (
        spec_metrics["target_forward_calls"]
        < vanilla_metrics["target_forward_calls"]
    ), "Speculative decoding did not reduce target forward calls"
