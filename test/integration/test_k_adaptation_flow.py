def test_k_changes_over_time(speculative_engine, prompt):
    speculative_engine.decode(prompt, max_tokens=60)

    k_history = speculative_engine.k_history

    assert len(set(k_history)) > 1
