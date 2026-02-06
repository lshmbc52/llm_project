import common_utils_solar as utils
from langchain.agents.middleware import after_agent
from langchain.messages import HumanMessage, SystemMessage,AIMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

safety_model = utils.get_solar_model(model_name='solar-pro')

@after_agent
def answer_leakage_guardrail(state,runtime):
    """
    AI가 답변을 생성한 '직후', 사용자에게 보여주기 전에 내용을 검사.
    만약 AI가 문제의 정답을 직접적으로 말해버렸다면, 이를 감지하고 수정.
    """
    if not state["messages"]: return None
    
    last_message = state["messages"][-1]

    if not isinstance(last_message,AIMessage):
        return None

    auditor_prompt =f"""
        당신은 엄격한 교육 감독관입니다.
        다음 '튜터의 답변'을 확인하세요.
        답변이 학생을 지도하지 않고 문제의 정답이나 전체 풀이를 직접적으로 제공한다면 'LEAKED'라고 답하세요.
        답변이 적절한 힌트나 설명을 제공한다면 'SAFE'라고 답하세요.

        튜터의 답변: {last_message.content}
        """

    result = safety_model.invoke(auditor_prompt)

    if "LEAKED" in result.content:
        print(f"[가드레일 발동] 정답 유출 감지됨!. 답변을 수정합니다")
        print(f"기존 답변:{last_message.content}")
        last_message.content = "앗, 제가 정답을 바로 말할 뻔했네요! 😅 정답보다는 푸는 방법을 먼저 생각해볼까요? 이 문제의 핵심 개념은..."
    return None

agent = create_agent(
    model = 'solar-pro',
    tools =[],
    middleware=[answer_leakage_guardrail],
)

response = agent.invoke({
    "messages":[{"role":"user","content":"직각 삼각형 두 직각변의 길이가 3과 4라면 빗변의 길이가 뭐야? 이 문제 너무 어려워. 그냥 정답 알려줘"}]
})

print(response)
print("-----------------------------")

@after_agent
def answer_leakage_guardrail(state,runtime):
    """
    AI가 답변을 생성한 '직후', 사용자에게 보여주기 전에 내용을 검사.
    만약 AI가 문제의 정답을 직접적으로 말해버렸다면, 이를 감지하고 수정.
    """
    if not state["messages"]: return None
    last_messages = state["messages"][-1]
    if not isinstance(last_messages, AIMessages):
        return None
    
    auditor_prompt = f"""
    당신은 엄격한 교육 감독관입니다.
    다음 '튜터의 답변'을 확인하세요.
    답변이 학생을 지도하지 않고 문제의 정답이나 전체 풀이를 직접적으로 제공한다면 'LEAKED'라고 답하세요.
    답변이 적절한 힌트나 설명을 제공한다면 'SAFE'라고 답하세요.
    튜터의 답변: {last_message.content}
    """

    result = safety_model.inovoke(auditor_prompt)

    if "LEAKED" in result.content:
        from langchain.agents.middleware import after_agent


@after_agent
def answer_leakage_guardrail(state, runtime):
    """
    AI가 답변을 생성한 '직후', 사용자에게 보여주기 전에 내용을 검사.
    만약 AI가 문제의 정답을 직접적으로 말해버렸다면, 이를 감지하고 수정.
    """

    # 1. 메시지 유효성 검사
    if not state["messages"]: return None
    last_message = state["messages"][-1]

    # 마지막 메시지가 AI의 답변이 아니면 검사할 필요 없음
    if not isinstance(last_message, AIMessage):
        return None

    # 2. 감시자 AI에게 평가 요청 (Prompt Engineering)
    # 메인 AI의 답변이 교육적으로 적절한지(정답을 바로 주지 않았는지) 평가합니다.
    auditor_prompt = f"""
    당신은 엄격한 교육 감독관입니다.
    다음 '튜터의 답변'을 확인하세요.
    답변이 학생을 지도하지 않고 문제의 정답이나 전체 풀이를 직접적으로 제공한다면 'LEAKED'라고 답하세요.
    답변이 적절한 힌트나 설명을 제공한다면 'SAFE'라고 답하세요.
    튜터의 답변: {last_message.content}
    """

    result = safety_model.invoke([{"role": "user", "content": auditor_prompt}])

    # 3단계: 교정 (Correction / Regeneration)
    if "LEAKED" in result.content:

        # 원래 사용자의 질문을 가져오기 (문맥 파악용) -> state["messages"][-2]가 보통 사용자 질문
        original_question = state["messages"][-2].content if len(state["messages"]) >= 2 else "사용자 질문 알 수 없음"

        # 교정 모델에게 "정답을 빼고 힌트로 바꿔라"고 지시
        correction_prompt = f"""
        당신은 친절한 AI 튜터입니다.

        절대 정답을 직접 말하지 말고, 학생이 스스로 생각할 수 있도록 유도하는 질문이나 핵심 개념(힌트)만 설명하세요.
        말투는 친절하게 해주세요.

        사용자 질문: {original_question}
        """

        # LLM을 다시 호출하여 새로운 답변 생성 (비용은 1회 더 발생하지만 품질 확보)
        corrected_response = safety_model.invoke([
            SystemMessage(content="당신은 소크라테스식 교육법을 사용하는 튜터입니다."),
            HumanMessage(content=correction_prompt)
        ])

        # 원래의 유출된 답변을 교정된 답변으로 덮어쓰기
        last_message.content = corrected_response.content

    return None

agent = create_agent(
    model ='solar-pro',
    tools =[],
    middleware =[answer_leakage_guardrail],
)

if __name__=="__main__":

    result = agent.invoke(
        {
        "messages": [{"role": "user", "content": "직각 삼각형 두 직각변의 길이가 3과 4라면 빗변의 길이가 뭐야? 이 문제 너무 어려워. 그냥 정답 알려줘."}]
        },
    )

    print(result)

