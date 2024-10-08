import chainlit as cl
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import torch
import os

# Paths
DB_FAISS_PATH = os.path.join(os.getcwd(), 'vectorstore', 'db_faiss')

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
def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.7,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return llm

# QA Bot function
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

# Chainlit handlers
@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    await cl.Message(content="Hi, Welcome to Medical Bot. What is your query?").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    conversation_context = message.content
    
    try:
        # Generate response using the LLM
        result = chain({"query": conversation_context})
        answer = result.get("result", "Sorry, I couldn't find an answer.")
        sources = result.get("source_documents", [])

        # Format the sources
        formatted_sources = ""
        if sources:
            formatted_sources = "\n\n**Sources:**\n"
            for doc in sources:
                source_name = doc.metadata.get('source', 'Unknown Source')
                page_number = doc.metadata.get('page', 'N/A')
                formatted_sources += f"- {source_name} (Page {page_number})\n"
        else:
            formatted_sources = "\nNo sources found."

        # Combine the answer with formatted sources
        final_output = f"{answer}{formatted_sources}"

        # Send the response
        await cl.Message(content=final_output).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()