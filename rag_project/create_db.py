from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os
import shutil


# load api key
load_dotenv()

print("API KEY:", os.getenv("GOOGLE_API_KEY")[:10])


# remove old db if exists
db_path = "./chroma_db"

if os.path.exists(db_path):
    shutil.rmtree(db_path)


# load pdf
loader = PyPDFLoader(
    "document_loader/SubmitEase Report.pdf"
)

documents = loader.load()

print("Pages loaded:", len(documents))


# split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# Gemini embeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# create vector db
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="submit_ease_collection",
    persist_directory=db_path
)


print("Vector DB created successfully")