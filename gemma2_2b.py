import torch
from transformers import pipeline


def _chat(messages: list) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    outputs = pipe(messages, max_new_tokens=512)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response


def chat(question: str) -> str:
    messages = [
        {"role": "user", "content": question},
    ]
    return _chat(messages)


if __name__ == "__main__":
    messages = "這只是一個測試"
    print(chat(messages))
