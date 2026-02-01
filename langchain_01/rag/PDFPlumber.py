from langchain_community.document_loaders import PDFPlumberLoader

file_path = "/home/sh/langchain_v1/rag/ARK_INNOVATION_ETF_ARKK_HOLDINGS (1).pdf"
loader = PDFPlumberLoader(file_path)
docs = loader.load()
# print(docs)
# print(len(docs))

import pprint

# pprint.pp(docs[0].metadata)

# print(docs[0].page_content[:100])
print(docs[0].page_content[:500])
