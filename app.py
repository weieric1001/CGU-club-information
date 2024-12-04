import pandas as pd
import streamlit as st

from vector_search import vector_search
from prompting import create_prompt
from gemma2_2b import chat

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
        options=[
            "google/gemma-2-2b-it",
        ],
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "message":
            st.write(msg["content"])

if prompt := st.chat_input(key="user_input"):
    with st.chat_message("user"):
        st.session_state.messages.append(
            {"type": "message", "role": "user", "content": prompt}
        )
        st.write(prompt)

    with st.chat_message("ai"):
        with st.status(
            label="Waiting for generate results...", state="running", expanded=True
        ) as status:
            st.write(f"1. Using faiss to search for similar QA...")
            example = vector_search(prompt)
            st.write(example)
            st.write(f"2. Prompting...")
            prompt = create_prompt(prompt, example)
            st.write(f"3. Waiting for response from {LLM_MODEL}...")
            response = chat(prompt)
            st.write(f"4. Done!")
            status.update(label="Done!", state="complete", expanded=False)
        result = {"type": "message", "role": "ai", "content": f"{response}"}
        st.write(result["content"])
        st.session_state.messages.append(result)
