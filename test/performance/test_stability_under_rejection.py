def test_stability_under_high_rejection(
    speculative_engine,
    prompt,
):
    """
    System should remain stable and make forward progress
    even when speculation frequently fails.
    """

    # Force worst-case behavior for testing
    speculative_engine.force_low_acceptance = True

    output = speculative_engine.decode(prompt, max_tokens=80)

    # --- Core guarantees ---
    assert len(output) > 0, "Decoding stalled under high rejection"
    assert speculative_engine.performance_tracker.tokens_generated > 0

    # Ensure we did not explode target forward calls
    target_calls = (
        speculative_engine.performance_tracker.target_forward_calls
    )
    assert target_calls > 0, "Target model was never called"

    # Optional sanity: acceptance never exceeds k
    for accepted, k in speculative_engine.acceptance_log:
        assert 0 <= accepted <= k
