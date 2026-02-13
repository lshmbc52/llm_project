from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
import common_utils as utils
from dotenv import load_dotenv

load_dotenv()

small_llm = utils.get_gpt_model(model_name="gpt-4o-mini")

search = TavilySearch(include_answer=True)

question = "10억짜리 집을 2채 가지고 있을 때 세금을 얼마나 내나요?"
market_value_rate_search = search.invoke(
    f"{datetime.now().year}년도 공정시장갸액 비율은?"
)
market_value_rate_search = market_value_rate_search["answer"]
market_value_rate_prompt = PromptTemplate.from_template(
    """ 아래 [Context]는 공정시장가액 비율에 관한 내용입니다.
당신에게 주어진 공정시장가액 비율에 관한 내용을 기반으로, 사용자의 상황에 대한 공정시장가액 비율을 알려주세요.
별도의 설명없이 공정시장가액 비율만 반환해 주세요.
[Context]
{context}

[Question]
질문:{question}
답변:
"""
)

# market_value_rate_chain = market_value_rate_prompt | small_llm | StrOutputParser()

# market_value_rate = market_value_rate_chain.invoke(
#     {"context": market_value_rate_search, "question": question}
# )

# print(market_value_rate)

market_value_rate_chain = market_value_rate_prompt | small_llm
market__value_rate = market_value_rate_chain.invoke(
    {"context": market_value_rate_search, "question": question}
)

print(market__value_rate)
