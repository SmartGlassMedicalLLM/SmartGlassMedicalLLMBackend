import streamlit as st
import requests

st.set_page_config(page_title="MedGemma Interface", layout="centered")

st.sidebar.header("Connection Settings")
server_ip = st.sidebar.text_input("Server IP/DNS", value="http://127.0.0.1")
server_port = st.sidebar.text_input("Port", value="8000")
endpoint = f"{server_ip}:{server_port}/convo"

if st.sidebar.button("Clear Chat History"):
    try:
        payload = {
            "reqRefId": "TEST",
            "resRefId": "TEST",
            "prompt": "",
            "new": True
        }
        response = requests.post(endpoint, json=payload, timeout=10)
        response.raise_for_status()
        st.text(response.json())

        st.session_state.messages = []
        st.success("History cleared!")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")

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

    # is_new = len(st.session_state.messages) <= 1
    payload = {
        "reqRefId": "TEST",
        "resRefId": "TEST",
        "prompt": prompt,
        "new": False
    }

    # send request to API
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = requests.post(endpoint, json=payload, timeout=180)
                response.raise_for_status()
                
                try:
                    full_response = response.json().get("answer")
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except (AttributeError, ValueError):
                    st.markdown(response.json())
                    st.session_state.messages.append({"role": "assistant", "content": response.json()})

                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")