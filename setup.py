from transformers import AutoTokenizer, AutoModel

from config import MODEL_LIST, get_args


def init():
    for m in list(get_args(MODEL_LIST)):
        try:
            AutoModel.from_pretrained(m, local_files_only=True)
            AutoTokenizer.from_pretrained(m, local_files_only=True)
            print(f"Model '{m}' is ready to use.")
        except Exception as e:
            try:
                AutoModel.from_pretrained(m, local_files_only=False)
                AutoTokenizer.from_pretrained(m, local_files_only=False)
                print(f"Model '{m}' is ready to use.")
            except Exception as e:
                print(f"Model '{m}' is not available.")
                print(e)


if __name__ == "__main__":
    init()
