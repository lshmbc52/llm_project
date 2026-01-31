from langchain_ollama import OllamaEmbeddings

# 1. 모델 초기화 (로컬 Ollama 서버와 통신)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 2. 문장 임베딩 테스트
text = "업스테이지 Solar 모델만큼 성능이 좋은 로컬 임베딩 모델을 테스트 중입니다."
query_result = embeddings.embed_query(text)

# 결과 확인
print(f"임베딩 벡터 차원: {len(query_result)}")  # mxbai-embed-large는 1024 차원입니다.
print(f"앞부분 5개 값: {query_result[:5]}")
