import common_utils_solar as utils
from dataclasses import dataclass
from langchain.agents.middleware import before_agent
from langchain.agents import create_agent

model = utils.get_solar_model(model_name="solar-pro")


@dataclass
class Context:
    user_name: str


@before_agent
def log_before_model(state, runtime):
    print(f"State:{state}")
    print(f"Runtime:{runtime}")
    print(f"사용자 이름:{runtime.context.user_name}")
    return None


agent = create_agent(
    model=model,
    tools=[],
    middleware=[log_before_model],
    context_schema=Context,
)

if __name__=="__main__":

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "제 이름이 뭐죠?"}]},
        context=Context(user_name="SH"),
    )

    print(response)
