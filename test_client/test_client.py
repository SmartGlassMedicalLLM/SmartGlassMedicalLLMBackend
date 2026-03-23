import streamlit as st
import requests
from pydantic import BaseModel

class BasicPrompt(BaseModel):
    prompt: str

class ConvoPrompt(BaseModel):
    prompt: str
    new: bool

st.set_page_config(page_title="MedGemma Interface", layout="centered")

st.sidebar.header("Connection Settings")
server_ip = st.sidebar.text_input("Server IP/DNS", value="http://127.0.0.1")
server_port = st.sidebar.text_input("Port", value="8000")
endpoint = f"{server_ip}:{server_port}/convo"

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.success("History cleared!")

st.title("MedGemma Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ask input
if prompt := st.chat_input("Ask a medical question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    is_new = len(st.session_state.messages) <= 1
    payload = {
        "prompt": prompt,
        "new": is_new
    }

    # send request to API
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = requests.post(endpoint, json=payload, timeout=60)
                response.raise_for_status()
                
                try:
                    full_response = response.json().get("response")
                    st.markdown(full_response)
                except ValueError:
                    st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")