def test_acceptance_bounds(speculative_engine, prompt):
    speculative_engine.decode(prompt, max_tokens=40)

    for accepted, k in speculative_engine.acceptance_log:
        assert 0 <= accepted <= k
