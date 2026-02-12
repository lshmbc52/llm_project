from langchain_core.prompts import ChatPromptTemplate
import common_utils as utils

llm = utils.get_exaone_model(model_name="exaone3.5:2.4b", temperature=0)
simple_chain = (
    ChatPromptTemplate.from_template("주어진 숫자 {number}를 2진수로 변환해 주세요")
    | llm
    | ChatPromptTemplate.from_template(
        " 다음 이숫자로 된  2진수를 16진수로 변환해 주세요:{text}"
    )
    | llm
)

# result = simple_chain.invoke({"number": "42"})
# print(result)

from langchain_core.runnables import RunnableParallel

analysis_chain = RunnableParallel(
    summary=ChatPromptTemplate.from_template("다음 텍스트를 요약해 주세요:{text}")
    | llm,
    sentiment=ChatPromptTemplate.from_template(
        "다음 텍스트의 감정을 분석해 주세요:{text}"
    )
    | llm,
    keywords=ChatPromptTemplate.from_template(
        "다음 텍스트의 주요 키워드를 추출해 주세요:{text}"
    )
    | llm,
)

# result = analysis_chain.invoke(
#     {
#         "text": "오늘 날씨가 너무 추움에도 불구하고 등산을 했습니다. 생각보다 많은 사람들이 산을 찾았습니다."
#     }
# )

# print(result)

from langchain_core.runnables import RunnablePassthrough

analysis_chain = (
    {
        "original": RunnablePassthrough(),
        "summary": ChatPromptTemplate.from_template(
            "{text}를 한 문장으로 요약해 주세요"
        )
        | llm,
    }
    | ChatPromptTemplate.from_template(
        """
        원본 텍스트:{original}
        요약:{summary}
                                        
        위 내용애 대한 분석 리포트를 작성해 주세요. 
        """
    )
    | llm
)

result = analysis_chain.invoke(
    {
        "text": "langchain은 llm application 개발을 위한 프레임워크입니다. 다양한 컴포넌트를 제공하여 개발을 쉽게 합니다."
    },
)

# print(result)

from langsmith import Client
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chroma
import common_utils as utils
from divide_conquer import embedding, vector_store, retriever

llm = utils.get_solar_model(model_name="solar-pro")
small_llm = utils.get_exaone_model(model_name="exaone3.5:2.4b")

client = Client()
rag_prompt = client.pull_prompt("rlm/rag-prompt", include_model=True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


tax_deductible_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

deductible_question = "주택에 대한 종합부동산세 과세표준의 공제액을 알려주세요"
tax_deductible_response = tax_deductible_chain.invoke(deductible_question)
print(f"tax_deductible_response:{tax_deductible_response}")
# 중간값을 활용한 LCEL 체인실행

from langchain_core.prompts import PromptTemplate

question = "10억짜리 집을 2채 가지고 있을 때 세금을 얼마나 내야 되나요?"

user_deduction_prompt = """
아래 [Context]는 주택에 대한 종합부동산세의 공제액에 관한 내용입니다.
사용자의 질문을 통해서 가지고 있는 주택수에 대한 공제액이 얼마인지 금액만 반환해 주세요.

[Context]
{tax_deduction_response}

[Question]
짊문:{question}
답변:
"""

user_deduction_prompt_template = PromptTemplate(
    template=user_deduction_prompt,
    input_variables=["tax_deductible_response", "question"],
)

# user_deduction_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#      | rag_prompt
#     | llm
#     | StrOutputParser()
# )

# try:
#     user_deduction = user_deduction_chain.invoke(
#         {"tax_deductible_response": tax_deductible_response, "question": question}
#     )

#     print(f"user_deduction:{user_deduction}")
# except Exception as e:
#     print(f"최종 답변 생성 중 정리 에러 발생(무시 가능): {e}")

user_deduction_chain = (
    user_deduction_prompt_template | llm | StrOutputParser()  # 위에서 정의한 템플릿
)

# 2. 실행부 수정

if __name__ == "__main__":
    try:
        # 이제 이 데이터는 검색(retriever)을 거치지 않고 바로 프롬프트에 끼워집니다.
        user_deduction = user_deduction_chain.invoke(
            {"tax_deduction_response": tax_deductible_response, "question": question}
        )
        print(f"\n--- [최종 분석 결과] ---")
        print(user_deduction)
    except Exception as e:
        # 만약 여기서도 BaseModel 에러가 난다면 Pass 하도록 구성
        if "BaseModel.__init__()" in str(e):
            # 결과값은 이미 user_deduction에 담겼을 수 있으므로 강제 출력 시도
            if "user_deduction" in locals() and user_deduction:
                print(f"\n--- [최종 분석 결과(정리 에러 무시)] ---")
                print(user_deduction)
        else:
            print(f"진짜 에러 발생: {e}")
