from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import os

load_dotenv("../.env")

print("API Key Loaded:", bool(os.getenv("GOOGLE_API_KEY")))

docs = [
    Document(
        page_content="This is a test document.",
        metadata={"source": "test_document.txt"}
    ),
    Document(
        page_content="This is another test document. pandas is good",
        metadata={"source": "another_test_document.txt"}
    )
]

embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="test_collection",
    persist_directory="../chroma_db"
)

result=vector_store.similarity_search("test document", k=2)

for r in result:
    print(r.page_content)

retriever=vector_store.as_retriever()
docs=retriever.invoke("explain pandas")

for doc in docs:
    print(doc.page_content)


print("Vector DB created successfully")