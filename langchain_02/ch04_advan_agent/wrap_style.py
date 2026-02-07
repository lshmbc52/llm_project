import common_utils_solar as utils
from langchain.agents.middleware import wrap_model_call, after_model
from langchain.messages import HumanMessage, SystemMessage
from dataclasses import dataclass
from langchain.agents import create_agent

model = utils.get_solar_model(model_name="solar-pro")


@dataclass
class Context:
    user_name: str


@wrap_model_call
def inject_user_name(request, handler):
    print(f"Request:{request}")
    user_name = request.runtime.context.user_name
    if user_name:
        sys_prompt = f"사용자의 이름은 {user_name}입니다."
        request = request.override(system_prompt=sys_prompt)
    return handler(request)


@after_model
def log_after_model(state, runtime):
    print(f'모델 응답 완료 후:{state["messages"][-1].content}')
    return None


agent = create_agent(
    model=model,
    tools=[],
    middleware=[inject_user_name, log_after_model],
    context_schema=Context,
)
if __name__ == "__main__":
    response = agent.invoke(
        {"messages": [{"role": "user", "content": " 제 이름이 뭐죠?"}]},
        context=Context(user_name="SH Lee입니다."),
    )
    print(response)
