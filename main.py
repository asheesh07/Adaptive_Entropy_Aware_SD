import torch

from core.speculative_engine import SpeculativeEngine
from core.draft_model import DraftModel
from core.target_model import TargetModel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Load models
    # ----------------------------
    draft_model = DraftModel(
        model_name="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        device=device,
    )

    target_model = TargetModel(
        model_name="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        device=device,
    )

    # ----------------------------
    # Build speculative engine
    # ----------------------------
    engine = SpeculativeEngine(
        draft_model=draft_model,
        target_model=target_model,
        max_k=8,
        entropy_bins=[1.2, 2.2, 3.0],
        k_values=[8, 4, 2, 1],
        acceptance_alpha=0.1,
        acceptance_init=1.0,
    )

    # ----------------------------
    # Tokenize prompt
    # ----------------------------
    prompt = "The theory of evolution explains"
    input_ids = target_model.tokenizer(
        prompt, return_tensors="pt"
    ).input_ids.to(device)

    # ----------------------------
    # Run decoding
    # ----------------------------
    output_ids = engine.decode(
        input_ids=input_ids,
        max_tokens=50,
    )

    output_text = target_model.tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )

    print("\n=== OUTPUT ===")
    print(output_text)

    print("\n=== PERFORMANCE ===")
    print(engine.performance_tracker.summary())

    print("\n=== QUALITY ===")
    print(engine.quality_evaluator.summary())


if __name__ == "__main__":
    main()
