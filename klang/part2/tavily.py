# from langsmith import Client
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import chroma
# import common_utils as utils
# from divide_conquer import (
#     embedding,
#     vector_store,
#     retriever,
#     get_tax_base_info,
#     format_docs,
# )
# from dotenv import load_dotenv
# from langchain_tavily import TavilySearch
# from datetime import datetime
# from lcel import tax_deductible_response

# llm = utils.get_solar_model(model_name="solar-pro")
# small_llm = utils.get_exaone_model(model_name="exaone3.5:2.4b")

# tax_base_response = get_tax_base_info()

# load_dotenv()

# search = TavilySearch(
#     include_answer=True,
# )
# datetime.now()

# market_value_rate_search = search.invoke(
#     f"{datetime.now().year}년도 공정시장가액비율은?"
# )

# market_value_rate_search = market_value_rate_search["answer"]
# print(market_value_rate_search)

# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# question = "10억짜리 집을 2채 가지고 있을 때 세금을 얼마나 내나요?"

# market_value_rate_prompt = PromptTemplate.from_template(
#     """
#     아래 [context]는 공정시장가액비율에 관한 내용입니다.
#     당신에게 주어진 공정시장가액비율에 관한 내용을 기반으로, 사용자의 상황애 대한 공정시장가액비율을 알려주세요.
#     별도의 설명없이 공정시장가액 비율만 알려주세요.
#     [Context]
#     {context}

#     [Question]
#     질문:{question}
#     답변:
# """
# )

# market_value_rate_chain = market_value_rate_prompt | llm | StrOutputParser()

# market_value_rate = None

# try:
#     market_value_rate = market_value_rate_chain.invoke(
#         {"context": market_value_rate_search, "question": question}
#     )

# except Exception as e:
#     if "BaseModel.__init__() in str(e)":
#         pass
#     else:
#         print(f"진짜 에러발생:{e}")
# if market_value_rate:
#     print("------------------")
#     print(f"추출된 공정시가액 비율:{market_value_rate}")
#     print("_______________")
# else:
#     print(f"\n 비율을 추출하지 못함. 검색결과를 다시 확인해 주세요")

# # chain 실행결과 종합으로 최종답변 생성

# from langchain_core.prompts import ChatPromptTemplate

# house_tax_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             f"""과세표준 계산방법:{tax_base_response}

#       공정시장가액비율:{market_value_rate}
#       공제액:{tax_deductible_response}

#       위의 공식과 아래 세율에 관한 정보를 활용해서 세금을 계산해 주세요.
#       세율:{{tax_rate}}
#       """,
#         ),
#         ("human", "{question}"),
#     ]
# )

# house_tax_chain = (
#     {"tax_rate": retriever | format_docs, "question": RunnablePassthrough()}
#     | house_tax_prompt
#     | llm
#     | StrOutputParser()
# )

# house_tax = house_tax_chain.invoke(question)
# print(house_tax)

import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

import common_utils as utils
from divide_conquer import retriever, get_tax_base_info, format_docs
from lcel import tax_deductible_response

# 1. 환경 설정 및 모델 초기화
load_dotenv()
llm = utils.get_solar_model(model_name="solar-pro")


def get_market_value_rate(user_question):
    """실시간 검색을 통해 올해의 공정시장가액비율을 가져오는 함수"""
    search = TavilySearch(include_answer=True)
    current_year = datetime.now().year

    # 실시간 웹 검색 실행
    search_result = search.invoke(f"{current_year}년도 공정시장가액비율은?")
    context = search_result.get("answer", "정보 없음")

    # 추출을 위한 프롬프트
    prompt = PromptTemplate.from_template(
        "다음 검색 결과에서 '{question}'에 필요한 공정시장가액 비율만 숫자(%)로 추출하세요.\n"
        "내용: {context}\n답변:"
    )

    chain = prompt | llm | StrOutputParser()

    try:
        return chain.invoke({"context": context, "question": user_question})
    except Exception as e:
        # Pydantic v2 충돌 에러 방어
        if "BaseModel.__init__" in str(e):
            return "60% (기본값 가정)"
        return "60% (조회 실패)"


def generate_final_tax_report(user_question, market_rate, tax_base_info):
    """모든 정보를 취합하여 최종 답변을 생성하는 함수"""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""당신은 전문 세무사입니다. 다음 데이터를 기반으로 사용자의 질문에 상세히 답하세요.
        
        [참고 정보]
        - 과세표준 계산법: {tax_base_info}
        - 공정시장가액비율: {market_rate}
        - 기본 공제액 정보: {tax_deductible_response}
        - 상세 세율: {{tax_rate}}
        """,
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        {"tax_rate": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(user_question)


# --- 실행부 ---
if __name__ == "__main__":
    user_query = "10억짜리 집을 2채 가지고 있을 때 세금을 얼마나 내나요?"

    print("1. 실시간 검색 및 비율 추출 중...")
    rate = get_market_value_rate(user_query)

    print("2. 내부 지식베이스(RAG) 조회 및 과세표준 분석 중...")
    base_info = get_tax_base_info()

    print("3. 최종 리포트 생성 중...")
    final_report = generate_final_tax_report(user_query, rate, base_info)

    print("\n" + "=" * 50)
    print(final_report)
