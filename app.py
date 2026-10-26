import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"


def display_message(message):
    st.write(message["content"])
    if message.get("sql") is not None and message.get("columns"):
        st.code(message["sql"], language="sql")
        st.dataframe(pd.DataFrame(message["rows"], columns=message["columns"]))


st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analyst Agent")
st.write("Upload a CSV file and ask questions about your data.")

# Initialize session state
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar: CSV Upload
with st.sidebar:
    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload your CSV",
        type=["csv"]
    )

    if st.button("Load Dataset", disabled=uploaded_file is None):

        try:
            with st.spinner("Uploading dataset..."):

                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/csv"
                        )
                    },
                    timeout=30
                )

                response.raise_for_status()
                data = response.json()

            st.session_state.dataset_id = data["dataset_id"]
            st.session_state.messages = []

            st.success("Dataset loaded!")

            st.write("Rows:", data["row_count"])
            st.write("Columns:", data["columns"])

        except requests.RequestException as e:
            st.error(f"Upload failed: {e}")


# Main: Chat Interface
if st.session_state.dataset_id is None:
    st.info("Please upload a CSV file to begin.")

else:

    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            display_message(message)

    # Chat input
    question = st.chat_input("Ask a question about your data...")

    if question:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        # Call FastAPI
        with st.chat_message("assistant"):

            try:
                with st.spinner("Analyzing..."):

                    response = requests.post(
                        f"{API_URL}/ask",
                        json={
                            "question": question,
                            "dataset_id": st.session_state.dataset_id
                        },
                        timeout=120
                    )

                    response.raise_for_status()
                    data = response.json()

                message = {
                    "role": "assistant",
                    "content": data["answer"],
                    "sql": data.get("sql"),
                    "columns": data.get("columns", []),
                    "rows": data.get("rows", []),
                }
                display_message(message)

                st.session_state.messages.append(message)

            except requests.RequestException as e:
                st.error(f"Analysis failed: {e}")
