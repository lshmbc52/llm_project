import common_utils_solar as utils
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore
from typing import TypedDict
from langchain.tools import tool
import uuid
from langchain.agents import create_agent
from datetime import datetime
from typing_extensions import NotRequired

model = utils.get_solar_model(model_name="solar-pro")


@dataclass
class Context:
    user_id: str
    app_name: str


store = InMemoryStore()


class UserInfo(TypedDict):
    personal_info: str
    preference: str
    last_updated: NotRequired[str]


@tool
def get_user_info(runtime) -> str:
    """현재 사용자의 정보 조회(시스템 내부용 도구)"""
    user_id = runtime.context.user_id
    app_name = runtime.context.app_name
    memories = runtime.store.search((user_id, app_name))
    if not memories:
        return "기록된 정보 없음"

    results = []
    for item in memories:
        data = item.value
        info_piece = []
        if "personal_info" in data:
            info_piece.append(f"개인정보: {data['personal_info']}")
        if "preference" in data:
            info_piece.append(f"선호도: {data['preference']}")
        if "last_updated" in data:
            info_piece.append(f"(업데이트: {data['last_updated']})")  # ✅ 날짜 표시

        results.append(" - ".join(info_piece))

        if "personal_information" in data:
            results.append(f"-개인정보:{data['personal_information']}")
        if "preference" in data:
            results.append(f"-선호도:{data['preference']}")

    return "\n".join(results) if results else "데이터 형식 불일치로 읽을 수 없음"


@tool
def save_user_info(user_info: UserInfo, runtime) -> str:
    """
    사용자 정보를 저장하거나 업데이트
    사용자가 본인의 이름,취향, 신상정보등을 언급하면 '반드시" 이도구를 호출하여 저장해야 함
    """
    user_id = runtime.context.user_id
    app_name = runtime.context.app_name
    store = runtime.store

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    user_info["last_updated"] = current_time

    memory_key = str(uuid.uuid4())
    store.put((user_id, app_name), memory_key, user_info)

    return f"정보가 안전하게 저장되었음.(저장시간:{current_time})"


agent = create_agent(
    model=model,
    tools=[get_user_info, save_user_info],
    store=store,
    context_schema=Context,
)

if __name__ == "__main__":
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "내이름은 'yeena'야. 난 사탕보다 엄마가 좋아",
                },
            ]
        },
        context=Context(user_id="user_001", app_name="personal_assistant"),
    )
    # print(response)
    print(response["messages"][-1].content)
