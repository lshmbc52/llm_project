import common_utils as utils
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chroma

llm = utils.get_solar_model(model_name="solar-pro")
small_llm = utils.get_exaone_model(model_name="exaone3.5:2.4b")

embedding = OpenAIEmbeddings(
    base_url="https://api.upstage.ai/v1/solar",
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-passage",
    check_embedding_ctx_length=False,
)

vector_store = Chroma(
    embedding_function=embedding,
    collection_name=chroma.COLLECTION_NAME,
    persist_directory=chroma.DB_PATH,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# from langsmith import Client

# client = Client()
# rag_prompt = client.pull_prompt("rlm/rag-prompt", include_model=False)
# lcel.py 파일의 상단 프롬프트 부분을 아래와 같이 수정하세요.

from langchain_core.prompts import ChatPromptTemplate

# 1. 기존의 client.pull_prompt 코드를 주석 처리합니다.
# client = Client()
# rag_prompt = client.pull_prompt("rlm/rag-prompt")

# 2. 대신 아래와 같이 직접 프롬프트를 정의합니다.
rag_prompt = ChatPromptTemplate.from_template(
    "당신은 질문-답변 과업을 수행하는 보조자입니다. "
    "검색된 다음 문맥(Context)을 사용하여 질문에 답하세요. "
    "답을 모른다면 모른다고 답변하고, 답변은 세 문장 이내로 간결하게 작성하세요.\n\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "Answer:"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

tax_base_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

#tax_base_question = "주택에 대한 종합부동산세 과세표준을 계산하는 방법은 무엇인가요?"

tax_base_question = "주택에 대한 종합부동산세 과세표준을 계산하는 방법을 수식으로 표현해서 수식만 반환해주세요. 부연설명은 하지 말고" 
tax_base_response = tax_base_chain.invoke(tax_base_question)
print(tax_base_response)


# tax_base_prompt = PromptTemplate(
#     input_variables=["property_value"],
#     template="부동산 가액이 {property_value}원인 경우, 종합부동산세 과세표준 계산 방법을 법률적 근거와 함께 설명해주세요.",
# )

# fair_market_prompt = PromptTemplate(
#     input_variables=["year"],
#     template="{year}년도 종합부동산세 공정시장가액비율은 얼마인가요? 법률적 근거와 함께 설명해주세요.",
# )

# deduction_prompt = PromptTemplate(
#     input_types=["ownership_period"],
#     template="보유기간이 {ownership_period}년인 경우 적용 가능한 공제액을 모두 알려주세요.",
# )

# final_calculation_prompt = PromptTemplate(
#     input_variables=["tax_base", "fair_market_ratio_deductions"],
#     template="""
#     다음 정보를 바탕으로 최종 종합부동산세액을 계산해주세요:
#     과세표준: {tax_base}
#     공정시장가액비율: {fair_market_ratio}
#     적용 가능 공제액: {deductions}
#     """,
# )
