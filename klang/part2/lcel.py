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

print(result)
