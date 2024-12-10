import streamlit as st

from vector_search import vector_search
from prompting import create_prompt
from llm import chat, create_pipeline

from config import MODEL_LIST, get_args

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "type": "message",
            "role": "assistant",
            "content": "請在下方輸入有關長庚大學社團的問題！",
        }
    ]

with st.sidebar:
    LLM_MODEL = st.selectbox(
        "LLM Model",
        options=list(get_args(MODEL_LIST)),
    )

if "model_name" not in st.session_state:
    st.session_state["model_name"] = LLM_MODEL

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "message":
            st.write(msg["content"])

if prompt := st.chat_input(key="user_input"):
    with st.chat_message("user"):
        prompt_msg = {
            "type": "message",
            "role": "user",
            "content": f"{prompt}",
        }
        st.session_state.messages.append(prompt_msg)
        st.write(prompt_msg["content"])

    with st.chat_message("ai"):
        with st.status(
            label="Waiting for generate results...", state="running", expanded=True
        ) as status:
            st.write(f"1. Using faiss to search for similar QA...")
            example = vector_search(prompt)
            st.write(example)
            st.write(f"2. Prompting...")
            prompt = create_prompt(prompt, LLM_MODEL, example)
            st.write(f"3. Creating pipeline for {LLM_MODEL}...")
            if (
                "model_pipe" not in st.session_state
                or st.session_state["model_name"] != LLM_MODEL
            ):
                st.session_state["model_name"] = LLM_MODEL
                st.session_state["model_pipe"] = create_pipeline(LLM_MODEL)
            st.write(f"4. Waiting for response from {LLM_MODEL}...")
            response = chat(prompt, st.session_state["model_pipe"])
            st.write(f"5. Done!")
            status.update(label="Done!", state="complete", expanded=False)
        result = {
            "type": "message",
            "role": "ai",
            "content": f"「{LLM_MODEL}」  \n{response}",
        }
        st.write(result["content"])
        st.session_state.messages.append(result)
