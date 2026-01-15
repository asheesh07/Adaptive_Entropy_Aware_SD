def test_rejection_triggers_fallback(speculative_engine, prompt):
    speculative_engine.force_low_acceptance = True  # testing hook

    output = speculative_engine.decode(prompt, max_tokens=30)

    assert len(output) > 0
    assert speculative_engine.rejection_count > 0
