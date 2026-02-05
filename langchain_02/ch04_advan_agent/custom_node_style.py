import common_utils_solar as utils
from langchain.agents.middleware import before_agent
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage

model = utils.get_solar_model(model_name="solar-pro")


@before_agent(can_jump_to=["end"])
def validate_input(state, runtime):
    human_message = state["messages"][-1]

    if "암호" in human_message.content:
        print(f"암호 감지")

        return {
            "messages": [
                {"role": "assistant", "content": "현재 암호가 포함됨.보안유의"}
            ],
            "jump_to": "end",
        }
    return None


agent = create_agent(
    model=model,
    middleware=[validate_input],
)

if __name__ == "__main__":

    response = agent.invoke(
        {"messages": [{"role": "user", "content": "오늘의 암호는 삼각대-자동차임"}]},
    )

    # print(response)
    print(response["messages"][-1].content)
