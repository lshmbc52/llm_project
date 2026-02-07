import common_utils_solar as utils
from after_agent_guardrail import answer_leakage_guardrail
from before_agent_guardrail import education_quardrail
from langchain.agents.middleware import before_agent
from langchain.agents import create_agent
import re

model = utils.get_solar_model(model_name="solar-pro")


@before_agent
def student_safety_middleware(state, runtime):
    """학생의 전화번호나 이메일이 감지되면 마스킹 처리해서 안전을 확보"""

    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if last_message.type != "human":
        return None

    content = last_message.content
    original_content = content

    phone_pattern = r"01[016789]-?[0-9]{3,4}-?[0-9]{4}"
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    is_redacted = False

    if re.search(phone_pattern, content):
        content = re.sub(phone_pattern, "<PHONE_REDACTED>", content)
        is_redacted = True

    if re.search(email_pattern, content):
        content = re.sub(email_pattern, "<EMAIL REDACTED", content)
        is_redacted = True

    if is_redacted:
        print(
            f"[학생보호] 개인정보가 감지되어 마스킹 처리함.\n원본:{original_content}\n수정:{content}"
        )

        last_message.content = content
    return None


ESCALATION_KEYWORDS = [
    "왕따",
    "괴롭힘",
    "우울해",
    "학교 폭력",
    "상담 선생님",
    "사람 불러줘",
]


@before_agent(can_jump_to="end")
def counseling_escalation_middleware(state, runtime):
    """
    [Layer 3] 심리적 위기 상황이나 상담요청이 감지되면 AI가 답변을 멈추고 인간 상담사에게 알림을 보냄.
    """

    if not state["messages"]:
        return None
    last_message = state["messages"][-1]

    for keyword in ESCALATION_KEYWORDS:
        if keyword in last_message.content:
            print(f"[상담이관] 심각한 고민/ 요청 감지:{keyword}")

            return {
                "messages": [
                    {
                        "role": "user",
                        "content": "학생, 많이 힘들었겠구나. 이 문제는 내가 답변하는 것보다 전문상담가\
선생님이 직접 듣고 도와 주는게 좋아.\n\n 지금 바로 상담선생님과 연결하니까 잠시만 기다려 (상담실 연결중)",
                    }
                ],
                "jump_to": "end",
            }
    return None


agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        education_quardrail,
        student_safety_middleware,
        counseling_escalation_middleware,
        answer_leakage_guardrail,
    ],
)

if __name__ == "__main__":
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": " 저는 공부하는 노인인데, 제 번호는 010-1234-2345입니다.",
                }
            ]
        },
    )

    print(response)
    print("-------------------------------")
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "요즘 학교에서 왕따당하는 것 같아. 너무 우울해",
                }
            ]
        },
    )
    print(response)
