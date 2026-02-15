import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import common_utils as utils
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
from langchain_core.prompts import PromptTemplate
from datetime import datetime
from langchain_tavily import TavilySearch

llm = utils.get_solar_model(model_name="solar-pro")
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        "\n\n",  # 빈 줄
        "\n",  # 줄바꿈
        " ",  # 공백
        ".",  # 마침표
        ",",  # 쉼표
        "\u200b",  # 폭 없는 공백
        "\uff0c",  # 전각 쉼표
        "\u3001",  # 한중일 쉼표
        "\uff0e",  # 전각 마침표
        "\u3002",  # 한중일 마침표
        "",
    ],
)

loader = Docx2txtLoader("/home/sh/Downloads/law_markdown.docx")
document_list = loader.load_and_split(text_splitter=text_splitter)

embedding = OpenAIEmbeddings(
    base_url="https://api.upstage.ai/v1/solar",
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-passage",
    check_embedding_ctx_length=False,
)

vector_store = PineconeVectorStore.from_documents(
    document_list,
    embedding,
    index_name="house-tax-index",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# print(retriever)


client = Client()
rag_prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


relevant_docs = retriever.invoke(
    "종합부동산세법 제8조 제9조 주택분 과세표준 기본 공제액 9억원 12억원"
)

context_text = format_docs(relevant_docs)
# print("******************************************")
# print(context_text)
tax_base_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

tax_base_question = "주택에 대한 종합부동산세 과세표준을 계산하는 방법을 수식으로 표현해서 수식만 반환해주세요. 부연설명을 하지 말아주세요"
tax_base_response = tax_base_chain.invoke(tax_base_question)
# print(tax_base_response)

question = "10억짜리 집 2채를 가지고 있을 때 종합부동산세는 얼마나 내나요?"

user_deduction_prompt = """아래 [Context]는 주택에 대한 종합부동산세의 공제액에 관한 내용입니다.
사용자의 질문에 답하기 위해'기본 공제액' 조항(보통 9억원 또는 12억원)을 찾아서 그 금액이 얼마인지만 답하세요.
#사용자의 질문을 통해서 가지고 있는 주택수에 대한 공제액이 얼마인지 금액만 반환해 주세요.
[Context]
{tax_deductible_response}

[Question]
질문:{question}
대답:
"""

user_deduction_prompt_template = PromptTemplate(
    template=user_deduction_prompt,
    input_variables=["tax_deductible_response", "question"],
)

user_deduction_chain = user_deduction_prompt_template | llm | StrOutputParser()

deductible_question = "주택에 대한 종합부동산세 과세표준의 공제액을 알려주세요"
tax_deductible_response = user_deduction_chain.invoke(
    {"tax_deductible_response": context_text, "question": question}
)

print("|||||||||||||||||||||||||||||")
# print(tax_deductible_response)

search = TavilySearch(include_answer=True)

market_value_rate_search = search.invoke(
    f"2026년 현재 시행중인 종합부동산세법상 공정시장가액비율 확정치"
)

# print("-----------------------")
# print(market_value_rate_search)
market_value_rate_search = market_value_rate_search["answer"]

market_value_rate_prompt = PromptTemplate.from_template(
    """아래 [Context]는 공정시장가액비율에 관한 내용입니다.
        현재 [Context]에서 '뉴스 전망' 혹은 '인상 게획'이 아닌 현재 법령으로 확정되어 시행중인 
            종합부동산세 공정시장가액비율을 찾아서 숫자만 추출하세요.
                        [Context]
                        {context}
                        
                        [Rule]
                        1. '실제 법령 문서'와 '인터넷 검색 결과'가 충돌할 경우, 반드시 **실제 법령 문서(시행령)**의 수치를 우선합니다.
                        2. 뉴스에서 언급되는 '전망', '인상 계획', '~할 것으로 보인다'는 수치는 무시하세요.
                        3. 오직 현재 법적으로 확정되어 시행 중인 수치만 선택합니다.
                        4.오직 숫자와 % 기호로만 출력하세요(예:60%)
                        -어떠한 부연설명이나 주의사항도 적지 마세요.
                        
                        
                        [Question]
                        질문:{question}
                        대답: """
)

market_value_rate_chain = market_value_rate_prompt | llm | StrOutputParser()

combined_context = f"--- 실제 법령 문서 ---\n{context_text}\n\n--- 최신 뉴스 및 인터넷 검색 결과 ---\n{market_value_rate_search}"

market_value_rate = market_value_rate_chain.invoke(
    {"context": combined_context, "question": question}
)

print(market_value_rate)
