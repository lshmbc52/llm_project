import common_utils_solar as utils
from langchain.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents.structured_output import ToolStrategy
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.search import GmailSearch

model = utils.get_solar_model(model_name="solar-pro")
toolkit = GmailToolkit()
search_tool = GmailSearch(api_resource=toolkit.api_resource)


# @tool
# def read_email_tool(query: str = "label:INBOX", limit: int = 5) -> str:
#     """
#     실제 Gmail에서 메일을 검색하고 내용을 읽어오는 도구입니다.
#     """
#     # 실제 gmail에서 메일검색(최근 5개)
#     emails = search_tool.run(f"{query} max_results ={limit}")
#     print(f"DEBUG:검색된 메일 데이터--> {emails}")

#     if not emails:
#         return "조회된 새 메일이 없습니다."

#     full_content = ""
#     for mail in emails:
#         full_content += f"\보낸 사람: {mail.get('sender')}\n제목: {mail.get('subject')}\n내용: {mail.get('body')}\n"

#     return full_content


@tool
def read_email_tool(query: str = "label:INBOX", limit: int = 5) -> str:
    """실제 Gmail에서 최근 메일을 검색합니다."""
    search_query = query if query else "label:INBOX"
    emails = search_tool.run(search_query)

    if not emails:
        return "조회된 메일이 없습니다."

    # 모델이 읽기 좋게 딱 필요한 정보만 정리해서 문자열로 넘깁니다.
    results = []
    for mail in emails[:limit]:
        results.append(
            f"보낸사람: {mail.get('sender')}\n제목: {mail.get('subject')}\n내용요약: {mail.get('snippet')}"
        )

    return "\n---\n".join(results)


class EmailAnalysis(BaseModel):
    """
    이메일 내용을 분석한 결과 구조
    """

    intent: Literal["complain", "inquiry", "confirmation", "other"] = Field(
        description="이메일의 주요 의도(예:complain=불만,inquiry = 문의,confirmation-확인,other= 기타)"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="이메일의 김정상태"
    )
    summary: str = Field(description="이메일 내용요약")
    next_action: str = Field(
        description="에이전트가 수행해야 할 다음 단계(예:회신,확인,무시 등)"
    )


agent = create_agent(
    model=model,
    tools=[read_email_tool],
    response_format=ToolStrategy(EmailAnalysis),
)

if __name__ == "__main__":

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "도구를 사용해서 내 Gmail의 최근 메일 10개을 읽어와줘. 그 다음, 읽어온 메일 가운데 가장 최근 것 5개를 분석해줘. 절대 예시 내용을 쓰지 말고 실제 메일 내용을 요약해.",
                },
            ]
        },
    )
    print(response)
    print("-----------------------")

    print(response["structured_response"])
    print(response["messages"][-1].content)
