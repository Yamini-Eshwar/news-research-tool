import os
import time
import streamlit as st
import langchain
import pandas as pd
import pickle as pkl
from langchain import OpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

# ------------------ Streamlit Page Config ------------------
st.set_page_config(
    page_title="News Research Tool📉",
    page_icon="📉",
    layout="wide",
)

# ------------------ Add Background Images ------------------
main_bg = "https://plus.unsplash.com/premium_photo-1661371627612-e48be62ce587?q=80&w=1169&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
sidebar_bg = ""

st.markdown(
    f"""
    <style>
    /* Main page background */
    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)), url("{main_bg}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
    }}

    /* Sidebar background */
    [data-testid="stSidebar"] {{
        background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("{sidebar_bg}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #ffffff;
    }}

    /* Optional: make input boxes darker for readability */
    .stTextInput>div>div>input {{
        background-color: rgba(0,0,0,0.5);
        color: #ffffff;
    }}

    .stButton>button {{
        background-color: #333333;
        color: #ffffff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ App Title & Sidebar ------------------
#st.title("News Research Tool📉")
st.markdown("""
<h1 style="
    background: -webkit-linear-gradient(#ff6a00, #ee0979);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align:center;
">
News Research Tool
</h1>
""", unsafe_allow_html=True)

st.sidebar.title("News Articles URL's")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placefolder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)  # only limits the output of the LLM
embeddings = OpenAIEmbeddings()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading....Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','], 
        chunk_size=1000
    )
    main_placefolder.text("Text Splitter....Started...✅✅✅")
    docs = text_splitter.split_documents(data) 
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embedding Vector Sarted Building...✅✅✅")
    # Save the FAISS index to a pickle file
    vectorstore_openai.save_local("faiss_store_openai")
    with open("faiss_store_metadata.pkl", "wb") as f:
        pkl.dump({"info": "OpenAI FAISS index"}, f)
    

query = main_placefolder.text_input("Question: ")
if query:
    vectorstore = FAISS.load_local("faiss_store_openai", embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    # {"answer": "", "sources": []}
    st.header("Answer")
    st.write(result["answer"])

    # Display sources if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)










