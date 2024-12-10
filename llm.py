import torch
from transformers import pipeline, Pipeline

from config import MODEL_LIST


def create_pipeline(model: MODEL_LIST):
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    pipe = pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    return pipe


def _chat(messages: list, pipe: Pipeline) -> str:
    outputs = pipe(messages, max_new_tokens=512)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return assistant_response


def chat(messages: list, pipe: Pipeline) -> str:
    return _chat(messages, pipe)


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "這只是一個測試"},
    ]
    pipe = create_pipeline("google/gemma-2-2b-it")
    print(chat(messages, pipe))
