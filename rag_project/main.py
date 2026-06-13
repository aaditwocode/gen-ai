from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# # -------- TEXT FILE --------
# data = TextLoader("test.txt")
# docs = data.load()

# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

# -------- PDF FILE --------
# pdf_loader = PyPDFLoader("document_loader/SubmitEase Report.pdf")
# pdf_docs = pdf_loader.load()

# print(pdf_docs[0].page_content)
# print(pdf_docs[0].metadata)

# pdf_text = "\n".join(doc.page_content for doc in pdf_docs)

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split_docs = splitter.split_documents(pdf_docs)



template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI that summarizes documents."),
        ("human", "{data}")
    ]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# prompt=template.format_messages(data=pdf_text)
# result = model.invoke(prompt)


# print("\nSUMMARY:\n")
# print(result.content)