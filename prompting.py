from config import MODEL_LIST


def create_prompt(
    question: str, model_name: MODEL_LIST = None, examples: list[dict] = None
) -> list:
    prompt = "你是一個友善的聊天機器人。\n你只能從我提供的資料回答有關長庚大學裡各種社團的問題。\n如果不知道答案就不要回答。\n"
    messages = []
    if model_name is None or model_name == "google/gemma-2-2b-it":
        if examples:
            for e in examples:
                prompt += f"Question: {e['question']}\n{e['answer']}\n\n"
        prompt += f"Question: {question}\n"
        messages = [
            {"role": "user", "content": prompt},
        ]
    if model_name in [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]:
        messages = [
            {"role": "system", "content": prompt},
        ]
        if examples:
            for e in examples:
                messages.append(
                    {"role": "user", "content": f"Question: {e['question']}"}
                )
                messages.append({"role": "assistant", "content": e["answer"]})
        messages.append({"role": "user", "content": f"Question: {question}"})
    return messages


if __name__ == "__main__":
    e = [
        {
            "question": "咖啡社在哪裡？",
            "answer": "地點在活動中心 社團練習室一旁邊",
            "score": 0,
        },
        {
            "question": "咖啡社有沒有自己的網站？",
            "answer": "FB網址: https://zh-tw.facebook.com/cgucoffeeclub",
            "score": 0,
        },
        {
            "question": "攝影社在哪裡？",
            "answer": "地點在屈臣氏直走走到底",
            "score": 0,
        },
    ]
    # print(create_prompt("長庚大學的社團有哪些？", "google/gemma-2-2b-it", e))
    print(
        create_prompt("長庚大學的社團有哪些？", "meta-llama/Llama-3.2-1B-Instruct", e)
    )
