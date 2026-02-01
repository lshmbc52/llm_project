from langchain_chroma import Chroma
from embed import cached_embedder
from store import CHROMA_PATH

db = Chroma(
    collection_name="rag_collection",
    embedding_function=cached_embedder,
    persist_directory=CHROMA_PATH,
    collection_metadata={"hnsw:space": "cosine"},
)

query = "Tesla 비중이 얼마나 되나요?"
result = db.similarity_search(query)
# print(f"검색된 문서내용:\n{result[0].page_content}")

retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2}
)

results = retriever.invoke("ARKK의 Tesla 투자비용은?")
print("----------------------------------------")
print(len(results))

if results:
    print(f"검색된 문서내용:\n{results[0].page_content}")
else:
    print("No relevant documents were found.")

retriever_mmr = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})
results = retriever.invoke("ARKK의 Tesla 투자비용은?")
print(len(results))
