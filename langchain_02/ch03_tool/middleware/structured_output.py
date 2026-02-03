import common_utils_solar as utils
from langchain.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import LLMToolEmulator

model = utils.get_solar_model(model_name="solar-pro")


@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """
    지정한 이메일 주소로 메일을 보내는 도구임
    """
    return f"이 메일이 성공적으로 전송되었음.\n수신자: {to}\n제목: {subject}\n내용: {body[:50]}..."


@tool
def read_email_tool(limit: int = 3) -> utils.List[utils.Dict[str, str]]:
    """
    고객이 온라인 쇼핑몰에 보낸 컴플레인, 문의, 혹은 확인관련 이메일 1개를 읽는 도구임
    """
    return f"이메일이 성공적으로 조회되었습니다."


# @tool
# def read_email_tool(
#     limit: int = 3,
# ) -> str:  # 반환 타입을 str로 단순화해서 테스트해 보세요.
#     """
#     고객이 온라인 쇼핑몰에 보낸 이메일을 읽는 도구입니다.
#     """
#     # 실제 환경이라면 DB나 API에서 가져오겠지만, 테스트를 위해 샘플 데이터를 넣습니다.
#     sample_email = """
#     수신: 쇼핑몰 고객센터
#     제목: 배송받은 상품이 파손되어 있습니다.
#     내용: 어제 받은 스마트워치 액정이 깨져서 왔어요. 너무 화가 나네요. 당장 환불해 주세요!
#     """
#     return sample_email


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
    tools=[send_email_tool, read_email_tool],
    response_format=ToolStrategy(EmailAnalysis),
    middleware=[LLMToolEmulator(model="solar-pro")],
)

if __name__ == "__main__":

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": " 최근 온 메일 확인하고 고개의 의도와 감정,요약, 어떤 행동이 필요한지 분석해 줘",
                }
            ]
        }
    )
    print(response["structured_response"])
