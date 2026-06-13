from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_spliter import TokenTextSplitter,
from langchain.text_splitter import RecursiveCharacterTextSplitter


splitter = CharacterTextSplitter(
  separator=" ",
  chunk_size=100, chunk_overlap=0)

data=TextLoader('test.txt')
print(data)
docs=data.load()
split_docs=splitter.split_documents(docs)
print(len(split_docs))
print(split_docs[0].page_content)
print(split_docs[0].metadata)

data=PyPDFLoader('SubmitEase Report.pdf')
# splitter=TokenTextSplitter(chunk_size=100, chunk_overlap=10)
# split_docs=splitter.split_documents(data.load())
# print(len(split_docs))
# print(split_docs[0].page_content)
# print(split_docs[0].metadata)

splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
split_docs=splitter.split_documents(data.load())
# print(data)
# docs=data.load()
# split_docs=splitter.split_documents(docs)
# print(len(split_docs))
# print(split_docs[0].page_content)
# print(split_docs[0].metadata)

