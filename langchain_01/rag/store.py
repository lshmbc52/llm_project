import common_utils as utils
from langchain_ollama import OllamaEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.vectorstores import InMemoryVectorStore
from load_split import recursive_docs
from langchain_chroma import Chroma
from embed import cached_embedder

CHROMA_PATH = "./chroma_db"

db = Chroma.from_documents(
    documents=recursive_docs,
    embedding=cached_embedder,
    persist_directory=CHROMA_PATH,
    collection_name="rag_collection",
)
query = "Tesla 비중이 얼마나 되나요?"

if __name__ == "__main__":

    result = db.similarity_search(query, k=1)
    # print(len(result))
    print(f"검색된 문서 내용:\n{result[0].page_content}")
