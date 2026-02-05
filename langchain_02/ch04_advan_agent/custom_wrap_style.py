import common_utils_solar as utils
from langchain.agents.middleware import before_agent, wrap_model_call
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage

model = utils.get_solar_model(model_name="solar-pro")


@wrap_model_call
def dynamic_model_selector(request, handler):
    last_msg = request.messages[-1].content if request.messages else ""
    msg_len = len(last_msg)

    if msg_len < 10:
        model_name = "gpt-5-nano"
    elif msg_len < 30:
        model_name = "gpt-5-mini"
    else:
        model_name = "gpt-5"

    print(f"메세지 길이:{msg_len},선택된 모델:{model_name}")

    new_model = init_chat_model(model_name)
    new_request = request.override(model=new_model)
    return handler(new_request)


agent = create_agent(
    model=model,
    tools=[],
    middleware=[dynamic_model_selector],
)

if __name__ == "__main__":
    response = agent.invoke({"messages": [{"role": "user", "content": " 안녕하세요?"}]})
    print(response)

    response_s = agent.invoke(
        {"messages": [{"role": "user", "content": " 안녕하세요?,저는 sh입니다."}]}
    )
    print(response_s)

    response_h = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": " 안녕하세요? 저는 hs입니다. 인공지능에 대해서\
            알고 싶어요",
                }
            ]
        }
    )

    print(response_h)
