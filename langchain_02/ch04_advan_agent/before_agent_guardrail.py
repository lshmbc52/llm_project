import common_utils_solar as utils
from langchain.agents.middleware import before_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
model = utils.get_solar_model(model_name='solar-pro')

forbidden_topics = {
    "cheating": ["답지", "정답 알려줘", "숙제 대신", "써줘", "베끼기"], # 부정행위 관련
    "distraction": ["롤", "게임", "유튜브", "아이돌", "웹툰"],         # 학습 방해 요소
    "harmful": ["담배", "술", "폭력", "싸움", "바보"]                 # 유해 콘텐츠
}

@before_agent(can_jump_to="end")
def education_quardrail(state,runtime):
    """
    학생의 질문 의도를 파악하여 교육적이지 않거나 부정행위가 의심될 경우,
    LLM(AI)에게 질문을 넘기지 않고 교육적인 멘트로 즉시 교정합니다.
     """   

    if not state["messages"]:
        return None

    last_message = state["messages"][-1]

    if last_message.type !="human":
        return None

    user_text = last_message.content

    for keyword in forbidden_topics["cheating"]:
        if keyword in user_text:
            return {
                "messages":[{"role":"assistant",

                   "content": "🚫 스스로 고민해봐야 실력이 늘어요! 정답을 바로 알려드리는 대신, 힌트를 드릴까요? 어떤 부분이 가장 어려운지 말해주세요."
                }],
                "jump_to": "end"
            }

    # Case B: 학습 집중 유도 (Focus Management)
    # 공부 중에 게임이나 딴짓 이야기를 하면 다시 공부로 유도
    for keyword in forbidden_topics["distraction"]:
        if keyword in user_text:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "⏰ 지금은 공부에 집중할 시간이에요! 딴짓은 쉬는 시간에 하고, 지금 풀고 있는 문제에 집중해볼까요?"
                }],
                "jump_to": "end"
            }

    # Case C: 유해 콘텐츠 차단 (Safety)
    # 교육 서비스의 브랜드 안전성(Brand Safety)을 위한 기능
    for keyword in forbidden_topics["harmful"]:
        if keyword in user_text:
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "⚠️ 부적절한 대화 주제입니다. 바르고 고운 말을 사용해주세요."
                }],
                "jump_to": "end"
            }
    return None

agent = create_agent(
    model = model,
    tools =[],
    middleware=[education_quardrail],
)

if __name__ =="__main__":

    response = agent.invoke(
        {"messages":[{"role":"user","content":"양자얽힘이 뭔가요? 예를 들어서 쉽게 설명해 주세요"}]}
    )
    print(response)

    print("----------------------")

    response = agent.invoke(
        {"messages":[{"role":"user","content":" 독후감쓰기 귀찮은데 숙제 대신 해주세요"}]}
    )
    print(response)
    print("----------------------")

    response = agent.invoke(
        {"messages":[{"role":"user","content":"공부하기 싫다.놀기 좋은유트브를 찾아줘"}]}
    )
    print(response)
    print("----------------------")
    
