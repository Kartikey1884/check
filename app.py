import streamlit as st
from dotenv import load_dotenv
import os, time, csv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda

# --- Load env ---
load_dotenv()

# --- Streamlit UI ---
st.title("Conversational RAG chatbot")
st.sidebar.header("Configuration")

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
session_id = st.sidebar.text_input("Session ID", value="default_session")

if not api_key:
    st.warning("Please enter your Groq API Key.")
    st.stop()

# --- Initialize ---
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if "store" not in st.session_state:
    st.session_state.store = {}

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    docs = PyPDFLoader(temp_path).load()
    os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # --- Prompt ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions from the given context. "
                   "If you don't know, say 'I don’t know.'\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    # --- Helper: Retrieve context ---
    # def retrieve_context(input_dict):
    #     question = input_dict["question"]
    #     docs = retriever.get_relevant_documents(question)
    #     context = "\n\n".join([d.page_content for d in docs])
    #     input_dict["context"] = context
    #     return input_dict
    
    def retrieve_context(input_dict):
        question = input_dict["question"]
        # Use .invoke() instead of .get_relevant_documents()
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        input_dict["context"] = context
        return input_dict


    # --- Chain setup (LCEL) ---
    rag_chain = (
        RunnableLambda(retrieve_context)
        | RunnableMap({
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "context": lambda x: x["context"],
        })
        | prompt
        | llm
    )

    # --- Chat History ---
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    chat_history = get_session_history(session_id)

    user_query = st.text_input("Ask something about the PDF:")
    if user_query:
        start = time.perf_counter()
        inputs = {"question": user_query, "chat_history": chat_history.messages}
        response = rag_chain.invoke(inputs)
        latency = time.perf_counter() - start

        answer = response.content if hasattr(response, "content") else str(response)
        st.success(f"🤖 Assistant: {answer}")
        chat_history.add_user_message(user_query)
        chat_history.add_ai_message(answer)
        st.sidebar.write(f"Response time: {latency:.2f} sec")

else:
    st.info("Please upload a PDF file to start chatting.")
