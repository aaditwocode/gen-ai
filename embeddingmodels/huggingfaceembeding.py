from langchain_huggingface import HuggingFaceEmbeddings

HuggingFaceEmbeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts=[
    "Hello world",
    "How are you doing today?",
    "The quick brown fox jumps over the lazy dog."
    ]

embeddings = HuggingFaceEmbeddings.embed_documents(texts.dimensions(1))
print(embeddings)