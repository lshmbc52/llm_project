# 테스트 텍스트
sample_text = """
인공지능(AI) 기술은 지난 몇 년간 급격하게 발전했습니다. 특히 OpenAI의 GPT 시리즈와 같은 대규모 언어 모델(LLM)은 자연어 처리 분야에서 혁신을 일으켰습니다.

하지만 LLM에게는 명확한 한계가 존재합니다.
첫째, 학습 데이터가 특정 시점에 고정되어 있어 최신 정보를 알지 못합니다.
둘째, 모델이 한 번에 처리할 수 있는 입력 토큰의 수(Context Window)에 제한이 있습니다.
이러한 문제를 해결하기 위해 등장한 것이 바로 RAG(검색 증강 생성) 기술입니다.

RAG 시스템의 핵심 프로세스는 다음과 같습니다:
1. 데이터 로드 (Loading)
2. 텍스트 분할 (Splitting)
3. 임베딩 및 저장 (Embedding & Storage)
4. 검색 및 답변 생성 (Retrieval & Generation)

텍스트 분할은 매우 중요합니다. RecursiveCharacterTextSplitter는문단이너무길경우줄바꿈으로자르고줄바꿈도없으면공백으로자르고공백도없으면결국글자단위로자르게됩니다.이문장은공백이거의없어서글자단위분할테스트에적합합니다.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

recursive_texts = text_splitter.split_text(sample_text)
print(f"총 청크 수:{len(recursive_texts)}\n")

for i, doc in enumerate(recursive_texts):
    print(f"---Chunk{i+1} ({len(doc)}자)------------)")
    print(doc)
    print("--------------------")
