from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "/home/sh/langchain_v1/rag/ARK_INNOVATION_ETF_ARKK_HOLDINGS (1).pdf"

loader = PDFPlumberLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
)

recursive_docs = text_splitter.split_documents(docs)
# print(recursive_docs)
print(len(recursive_docs))
print(recursive_docs[0].page_content)
print(len(recursive_docs[0].page_content))
