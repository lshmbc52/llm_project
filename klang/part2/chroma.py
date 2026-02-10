import os
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import common_utils as utils

# 1. 설정 상수는 상단으로 모으기
FILE_PATH = "/home/sh/Downloads/law_markdown.docx"
DB_PATH = "./tax-original"
COLLECTION_NAME = "tax-original"


def get_vector_store():
    embedding = OpenAIEmbeddings(
        base_url="https://api.upstage.ai/v1/solar",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        model="embedding-passage",
        check_embedding_ctx_length=False,
    )

    # DB가 이미 존재하면 로드하고, 없으면 새로 생성
    if os.path.exists(DB_PATH):
        print("Existing DB found. Loading...")
        return Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding,
            collection_name=COLLECTION_NAME,
        )
    else:
        print("Creating new DB...")
        loader = Docx2txtLoader(FILE_PATH)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )
        docs = loader.load_and_split(text_splitter=text_splitter)
        return Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=COLLECTION_NAME,
            persist_directory=DB_PATH,
        )


def create_rag_chain(vector_store):
    model = utils.get_solar_model(model_name="solar-pro")

    # template = """언제나 법령 정보를 바탕으로 정중하게 답변하세요.
    # 제공된 컨텍스트(Context)를 사용하여 질문에 답하고, 만약 답을 모른다면 "해당 내용은 자료에 명시되어 있지 않습니다."라고 답하세요.

    template = """너는 유능한 세무 상담사야. 
제공된 법령 자료(Context)를 근거로 사용자의 질문에 답해줘.
만약 자료에 구체적인 계산 사례가 없더라도, 자료에 명시된 세율과 공제 기준을 활용해서 가능한 범위까지 계산 과정을 설명해줘.
단, 확실하지 않은 정보는 추측하지 말고 필요한 추가 정보를 요청해.
    
    Context: {context}

    Question: {input}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    combine_docs_chain = create_stuff_documents_chain(model, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# --- 메인 실행부 ---
if __name__ == "__main__":
    # 1. 저장소 가져오기
    vs = get_vector_store()

    # 2. 체인 구축
    chain = create_rag_chain(vs)

    # 3. 질문 실행
    # question = "김천에 8천만원 짜리 아파트 한채와 서울 신림동 8억짜리 다중주택 한채 등 두채를 가지고 있을 때 세금을 얼마나 내나요?"
    question = "10억짜리 집을 2채 가지고 있으면 세금을 얼마나 내나요?"
    response = chain.invoke({"input": question})

    print(f"\nQ: {question}\n")
    print(f"A: {response['answer']}")


# import os
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langsmith import Client
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_classic.chains import create_retrieval_chain
# from langchain_community.document_loaders import Docx2txtLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# import common_utils as utils

# model = utils.get_solar_model(model_name="solar-pro")
# client = Client()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1500,  # 각 청크의 최대 문자 수 (너무 크면 정보가 희석, 너무 작으면 맥락 손실)
#     chunk_overlap=200,  # 인접한 청크 간 중복되는 문자 수 (맥락 연결성 유지)
#     separators=[
#         "\n\n",  # 빈 줄 (문단 구분 우선)
#         "\n",  # 줄바꿈 (문장 구분)
#         " ",  # 공백 (단어 구분)
#         ".",  # 마침표 (문장 종료)
#         ",",  # 쉼표 (절 구분)
#         "\u200b",  # 폭 없는 공백 (웹에서 흔히 발견)
#         "\uff0c",  # 전각 쉼표 (한글 텍스트용)
#         "\u3001",  # 한중일 쉼표 (동아시아 언어용)
#         "\uff0e",  # 전각 마침표 (한글 텍스트용)
#         "\u3002",  # 한중일 마침표 (동아시아 언어용)
#         "",  # 마지막 구분자 (강제 분할)
#     ],
# )

# file_path = "/home/sh/Downloads/law_markdown.docx"
# loader = Docx2txtLoader(file_path)
# document_list = loader.load_and_split(text_splitter=text_splitter)

# embedding = OpenAIEmbeddings(
#     base_url="https://api.upstage.ai/v1/solar",
#     api_key=os.getenv("UPSTAGE_API_KEY"),
#     model="embedding-passage",
#     check_embedding_ctx_length=False,
# )

# vector_store = Chroma.from_documents(
#     documents=document_list,
#     embedding=embedding,
#     collection_name="tax-original",
#     persist_directory="./tax-original",
# )

# template = """언제나 법령 정보를 바탕으로 정중하게 답변하세요.
# 제공된 컨텍스트(Context)를 사용하여 질문에 답하고, 만약 답을 모른다면 "해당 내용은 자료에 명시되어 있지 않습니다."라고 답하세요.

# Context: {context}

# Question: {input}

# Answer:
# """
# retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(template)

# question = "김천에 8천만원 짜리 아파트 한채와 서울 신림동 8억짜리 다중주택  한채 등 두채를 가지고 있을 때 세금을 얼마나 내나요?"
# retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# # print(retriever.invoke(question))
# combine_docs_chain = create_stuff_documents_chain(model, retrieval_qa_chat_prompt)

# retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ai_message = retrieval_chain.invoke({"input": question})
# print(ai_message["answer"])
