from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from utils.parameter import GEMINI_API_EMBEDDING_MODEL, GEMINI_API_KEY
def save_chunk(url:str,chunk_size: int = 1000, chunk_overlap: int = 200):
    # Load the webpage
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Split the content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GEMINI_API_KEY, model=GEMINI_API_EMBEDDING_MODEL)

    # Store into Chroma DB
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    return vectorstore

