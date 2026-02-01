import common_utils as utils
from langchain_ollama import OllamaEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.vectorstores import InMemoryVectorStore
from load_split import recursive_docs


underlying_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model,
)

if __name__ == "__main__":

    vectorstore = InMemoryVectorStore.from_documents(
        recursive_docs,
        cached_embedder,
    )

    query = "Tesla 투자비용이 얼마나 되나요?"
    request = vectorstore.similarity_search(query)
    # print(request)
    print(f"검색된 문서내용 :\n{request[0].page_content}")
