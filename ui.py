import streamlit as st
from main import rag_system, query_rag, maintain_chat_history

# Get API keys from Streamlit secrets
import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

st.title("KDijagnostika Customer Support")
st.write("Welcome to the KDijagnostika Customer Support. How can I assist you today?")

# initialize the session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    

if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"], st.session_state["chat_history"] = rag_system("kdijagnostika-blogs-embeddings")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    st.session_state["messages"].append({"role":"user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    response = query_rag(user_input, st.session_state["rag_chain"], st.session_state["chat_history"])

    st.session_state["chat_history"] = maintain_chat_history(st.session_state["chat_history"], user_input, response)

    st.session_state["messages"].append({"role":"assistant", "content": response})
    with st.chat_message("assistant"):
        st.text(response)