import torch

from core.speculative_engine import SpeculativeEngine
from core.draft_model import DraftModel
from core.target_model import TargetModel
from transformers import AutoTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------
    # Load models
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    draft_model = DraftModel(
        tokenizer=tokenizer,
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        dtype=torch.float16,
    )

    target_model = TargetModel(
        tokenizer=tokenizer,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        device=device,
        dtype=torch.float16,
    )

    # ----------------------------
    # Build speculative engine
    # ----------------------------
    engine = SpeculativeEngine(
        draft_model=draft_model,
        target_model=target_model,
        max_k=5,
        entropy_bins=[2.0, 4.0, 6.0],
        k_values=[5, 4, 3, 2],
        acceptance_alpha=0.1,
        acceptance_init=0.5,
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
