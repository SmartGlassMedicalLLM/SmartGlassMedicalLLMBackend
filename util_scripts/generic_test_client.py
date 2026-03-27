import streamlit as st
import requests
import json

# Timeout for requests in seconds
TIMEOUT = 180

st.set_page_config(page_title="Generic Test Client", layout="wide")
st.title("Generic Test Client")

# Endpoint
st.header("Endpoint")
endpoint = st.text_input("URL", placeholder="http://localhost:8000/base", value="http://localhost:8000/base")

# JSON fields
st.header("JSON Fields")

if "fields" not in st.session_state:
    st.session_state.fields = [{"key": "", "value": ""}]

def add_field():
    st.session_state.fields.append({"key": "", "value": ""})

def remove_field(i):
    st.session_state.fields.pop(i)

for i, field in enumerate(st.session_state.fields):
    col1, col2, col3 = st.columns([2, 4, 0.5])
    with col1:
        st.session_state.fields[i]["key"] = st.text_input(
            "Key", value=field["key"], key=f"key_{i}", placeholder="e.g. prompt"
        )
    with col2:
        st.session_state.fields[i]["value"] = st.text_input(
            "Value", value=field["value"], key=f"val_{i}", placeholder="e.g. What are the side effects?"
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✕", key=f"del_{i}"):
            remove_field(i)
            st.rerun()

st.button("＋ Add Field", on_click=add_field)

# PDF upload
st.header("PDF Attachment (optional)")
pdf_field_name = st.text_input(
    "Form field name for PDF", value="pdf", placeholder="pdf"
)
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

# Request preview
st.header("Request Preview")
payload = {
    f["key"]: f["value"]
    for f in st.session_state.fields
    if f["key"].strip()
}
st.code(json.dumps(payload, indent=2), language="json")
if uploaded_pdf:
    st.info(f"PDF attached as `{pdf_field_name}`: {uploaded_pdf.name}")

# Send
st.divider()
if st.button("Send", type="primary", use_container_width=True):
    if not endpoint.strip():
        st.error("Please enter an endpoint URL.")
    else:
        try:
            if uploaded_pdf:
                # Send as multipart/form-data
                data = {k: v for k, v in payload.items()}
                files = {pdf_field_name: (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                response = requests.post(endpoint, data=data, files=files, timeout=TIMEOUT)
            else:
                # Send as plain JSON
                response = requests.post(endpoint, json=payload, timeout=TIMEOUT)

            st.subheader(f"Response — {response.status_code}")

            print(response.text)
            # Try to show JSON nicely, fallback to codeblock
            try:
                st.json(response.json())
            except Exception:
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Could not connect. Is the API running?")
        except requests.exceptions.Timeout:
            st.error(f"Request timed out after {TIMEOUT} seconds.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")