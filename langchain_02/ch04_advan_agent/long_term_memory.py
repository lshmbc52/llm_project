import common_utils_solar as utils
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call
from langchain.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent

model = utils.get_solar_model(model_name="solar-pro")
store = InMemoryStore()


user_id = "user_001"
application_context = "personal_assistant"

namespace = (user_id, application_context)

store.put(
    namespace,
    "memory_001",
    {
        "facts": [
            "사용자는 커피보다 차를 선호함",
            "사용자는 매일 아침 6싱 기상함",
        ],
        "language": "Korean",
    },
)

item = store.get(namespace, "memory_001")
# print("저장된 메모리:", item.value)
# print(f"저장된 메모리:{item}")

# items = store.search(namespace)
# print(items)

store.put(
    namespace,
    "memory_002",
    {
        "facts": [
            "사용자는 그림을 좋아함",
            " 특히 반 고흐의 작품을 좋아함",
        ]
    },
)

items = store.search(namespace)
# print(items)


@dataclass
class Context:
    user_id: str
    app_name: str


@wrap_model_call
def inject_memory(request, handler):
    current_user = request.runtime.context.user_id
    current_app = request.runtime.context.app_name
    memories = request.runtime.store.search((current_user, current_app))

    memory_content = "기록된 정보 없음"

    if memories:
        extracted_facts = []
        for item in memories:
            if "facts" in item.value:
                extracted_facts.extend(item.value["facts"])
        memory_content = "\n- ".join(extracted_facts)

    system_message = f"사용자 관련 장기 메모리:{memory_content}"
    new_request = request.override(system_prompt=system_message)
    return handler(new_request)


agent = create_agent(
    model=model,
    store=store,
    context_schema=Context,
    middleware=[inject_memory],
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "나에 대해 알고 있는 정보를 알려줘"}]},
    context=Context(user_id=user_id, app_name=application_context),
)

print(response)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "나에 대해 알고 있는 정보를 알려줘"}]},
    context=Context(user_id="user_002", app_name=application_context),
)

print(response)
