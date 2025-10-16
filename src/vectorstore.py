from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits raw extracted documents into overlapping text chunks to preserve semantic context.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vector_store(chunks: List[Document], api_key: str) -> FAISS:
    """
    Converts text chunks into dense vector representations and indexes them using FAISS.
    """
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key, 
        model="text-embedding-3-small"
    )
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore
