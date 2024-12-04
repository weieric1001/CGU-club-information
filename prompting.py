def create_prompt(question: str, examples: list[dict] = None) -> str:
    prompt = "你是一個友善的聊天機器人。n\你只能從我提供的資料回答有關長庚大學裡各種社團的問題。\n"
    if examples:
        for e in examples:
            prompt += f"Question: {e['question']}\n{e['answer']}\n\n"
    prompt += f"Question: {question}\n"
    return prompt


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
    print(create_prompt("長庚大學的社團有哪些？", e))
