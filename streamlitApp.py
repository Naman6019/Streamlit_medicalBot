import streamlit as st
import asyncio
from fastapi import HTTPException
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import torch
import os

# Paths
DB_FAISS_PATH = os.path.join(os.getcwd(), 'vectorstore', 'db_faiss')

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Custom prompt for QA
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Function to set custom prompt
def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Load Language Model
@st.cache_resource
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=128,  # Reduced token limit
        temperature=0.7,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return llm

# QA Bot function
@st.cache_resource
def qa_bot():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': device})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': qa_prompt}
    )
    return qa

# Initialize QA bot
qa = qa_bot()

# Async function for model inference
async def async_qa(query):
    result = await asyncio.to_thread(qa, {"query": query})
    return result

# Streamlit UI
st.title("Medical Chatbot")

user_input = st.text_input("You:", key="user_input")

if user_input:
    # Append user message to chat history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    
    # Use conversation history to construct input
    conversation_context = "\n".join(
        f"{entry['role']}: {entry['content']}" for entry in st.session_state["chat_history"]
    )

    # Use spinner to show loading
    with st.spinner('Processing...'):
        result = asyncio.run(async_qa(conversation_context))
        answer = result["result"]
        sources = result["source_documents"]

    # Append bot response to chat history
    st.session_state["chat_history"].append({"role": "bot", "content": answer})
    
    # Display chat history
    for entry in st.session_state["chat_history"]:
        if entry["role"] == "user":
            st.text(f"User: {entry['content']}")
        else:
            st.text(f"Bot: {entry['content']}")
    
    # Display sources if any
    if sources:
        st.markdown("**Sources:**")
        for doc in sources:
            source_name = doc.metadata.get('source', 'Unknown Source')
            page_number = doc.metadata.get('page', 'N/A')
            st.markdown(f"- {source_name} (Page {page_number})")
