from dataclasses import dataclass
import common_utils as utils
import gen_long_term


@dataclass
class Context:
    user_id: str
    app_name: str


store = gen_long_term.InMemoryStore()

from typing import TypedDict


class UserInfo(TypedDict):
    personal_info: str
    preference: str


from langchain_core.runnables import RunnableConfig
from langchain.tools import tool, ToolRuntime


@tool
def get_user_info(runtime) -> str:
    """현재 사용자의 정보조회(시스템 내부용 도구)"""

    user_id = runtime.context.user_id
    app = runtime.context.app_name

    memories = runtime.store.search((user_id, app))

    if not memories:
        return "기록된 정보없음"

    results = []

    for item in memories:
        data = item.value

        if "personal_information" in data:
            results.append(f"--개인정보:{data['personal_information']}")
        if "preference" in data:
            results.append(f"--선호도:{data['preference']}")

    return "\n".join(results) if results else "데이터 형식 불일치로 읽을 수 없음"


import uuid


@tool
def save_user_info(user_info: UserInfo, runtime):
    """사용자의 정보를 저장하거나 업데이트"""

    user_id = runtime.context.user_id
    app = runtime.context.app_name
    store = runtime.store

    memory_key = str(uuid.uuid4())
    store.put((user_id, app), memory_key, user_info)
    return f"정보가 안전하게 저장되었습니다.(ID:{memory_key})"


from langchain.agents import create_agent

if __name__ == "__main__":

    agent = create_agent(
        model="gpt-4o-mini",
        tools=[get_user_info, save_user_info],
        store=store,
        context_schema=Context,
    )

    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": "내이름은 'sh'야, 커피보단 차를 좋아해"}
            ]
        },
        context=Context(user_id="user_001", app_name="personal_assistant"),
    )
    print(response)

    print("-------------------------")

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "나에 대해 아는 정보를 말해줘"}]},
        context=Context(user_id="user_001", app_name="personal_assistant"),
    )
    print(response)
