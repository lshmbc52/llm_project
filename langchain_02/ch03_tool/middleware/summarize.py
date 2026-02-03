import common_utils_solar as utils
from langchain.tools import tool
from langchain.agents.middleware import (
    LLMToolEmulator,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

model = utils.get_solar_model(model_name="solar-pro")


@tool
def save_customer_feedback(feedback: str) -> str:
    """고객 피드백을 저장하는 도구"""
    return f"고객 피드백 저장완료:{feedback}"


agent = create_agent(
    model=model,
    tools=[save_customer_feedback],
    middleware=[
        SummarizationMiddleware(
            model="solar-pro",
            trigger=("messages", 5),
            keep=("messages", 10),
            trim_tokens_to_summarize=4000,
        )
    ],
    checkpointer=InMemorySaver(),
)

if __name__ == "__main__":

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": " 너는 AI 공부하는 sh를 돕는 친절한 멘토야",
                },
                {"role": "user", "content": "안녕하세요. 저는 sh 입니다"},
            ]
        },
        {"configurable": {"thread_id": "4"}},
    )
    print(result["messages"][-1].content)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "저는 집에서 노는 나이많은 정년퇴직자 입니다.",
                }
            ]
        },
        {"configurable": {"thread_id": "4"}},
    )

    print(result["messages"][-1].content)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "저는 서울에서 AI 공부하고 있습니다.현재 langchain 가운데 middeleware를 배우는데 역시 어렵습니다.",
                }
            ]
        },
        {"configurable": {"thread_id": "4"}},
    )

    print(result["messages"][-1].content)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "내가 아까 내 상황에 대해 뭐라고 말했었지?",
                }
            ]
        },
        {"configurable": {"thread_id": "4"}},
    )

    print(result["messages"][-1].content)

    print("----------------------------")

    print(result["messages"])

    for i, msg in enumerate(result["messages"]):
        print(f"[{i}] {msg.__class__.__name__}: {msg.content[:50]}...")
