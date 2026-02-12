from langsmith import Client
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chroma
import common_utils as utils
from divide_conquer import (
    embedding,
    vector_store,
    retriever,
    get_tax_base_info,
    format_docs,
)
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from datetime import datetime
from lcel import tax_deductible_response

llm = utils.get_solar_model(model_name="solar-pro")
small_llm = utils.get_exaone_model(model_name="exaone3.5:2.4b")

tax_base_response = get_tax_base_info()

load_dotenv()

search = TavilySearch(
    include_answer=True,
)
datetime.now()

market_value_rate_search = search.invoke(
    f"{datetime.now().year}년도 공정시장가액비율은?"
)

market_value_rate_search = market_value_rate_search["answer"]
print(market_value_rate_search)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

question = "10억짜리 집을 2채 가지고 있을 때 세금을 얼마나 내나요?"

market_value_rate_prompt = PromptTemplate.from_template(
    """
    아래 [context]는 공정시장가액비율에 관한 내용입니다.
    당신에게 주어진 공정시장가액비율에 관한 내용을 기반으로, 사용자의 상황애 대한 공정시장가액비율을 알려주세요.
    별도의 설명없이 공정시장가액 비율만 알려주세요.
    [Context]
    {context}

    [Question]
    질문:{question}
    답변:
"""
)

market_value_rate_chain = market_value_rate_prompt | llm | StrOutputParser()

market_value_rate = None

try:
    market_value_rate = market_value_rate_chain.invoke(
        {"context": market_value_rate_search, "question": question}
    )

except Exception as e:
    if "BaseModel.__init__() in str(e)":
        pass
    else:
        print(f"진짜 에러발생:{e}")
if market_value_rate:
    print("------------------")
    print(f"추출된 공정시가액 비율:{market_value_rate}")
    print("_______________")
else:
    print(f"\n 비율을 추출하지 못함. 검색결과를 다시 확인해 주세요")

# chain 실행결과 종합으로 최종답변 생성

from langchain_core.prompts import ChatPromptTemplate

house_tax_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""과세표준 계산방법:{tax_base_response}
         
      공정시장가액비율:{market_value_rate}
      공제액:{tax_deductible_response}

      위의 공식과 아래 세율에 관한 정보를 활용해서 세금을 계산해 주세요.
      세율:{{tax_rate}}
      """,
        ),
        ("human", "{question}"),
    ]
)

house_tax_chain = (
    {"tax_rate": retriever | format_docs, "question": RunnablePassthrough()}
    | house_tax_prompt
    | llm
    | StrOutputParser()
)

house_tax = house_tax_chain.invoke(question)
print(house_tax)
