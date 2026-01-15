def test_speculative_matches_vanilla(
    speculative_engine,
    vanilla_decoder,
    prompt,
):
    spec_output = speculative_engine.decode(prompt, max_tokens=50)
    base_output = vanilla_decoder.decode(prompt, max_tokens=50)

    assert spec_output == base_output
