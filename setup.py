from transformers import pipeline


def init():
    # check model "google/gemma-2-2b-it"
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        device=0,
    )
    print("Model 'google/gemma-2-2b-it' is ready to use.")

    # check model "intfloat/multilingual-e5-small"
    pipe = pipeline(
        "text-generation",
        model="intfloat/multilingual-e5-small",
        device=0,
    )
    print("Model 'intfloat/multilingual-e5-small' is ready to use.")


if __name__ == "__main__":
    init()
