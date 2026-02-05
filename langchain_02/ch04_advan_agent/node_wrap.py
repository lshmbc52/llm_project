import common_utils_solar as utils
from langchain.agents.middleware import wrap_model_call, after_model
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


model = utils.get_solar_model(model_name='solar-pro')

@dataclass
class Context:
    user_name:str
    is_premium:bool= False

@wrap_model_call
def dynamic_model_selector(request, handler):
    user_name = request.runtime.context.user_name
    is_premium = request.runtime.context.is_premium

    if is_premium:
        model_name = 'gpt-5'
        print(f"{user_name}사용자는 프리미엄 회원입니다. GPT-5를 호출합니다")
    else:
        model_name ='gpt-5-nano'
        print(f"{user_name}사용자는 일반 회원입니다. GPT-5-nano를 호출합니다")
 
    new_model = init_chat_model(model_name) 
    new_request = request.override(model= new_model)
    return handler(new_request)

agent = create_agent(
    model= model,
    tools =[],
    middleware =[dynamic_model_selector],
    context_schema = Context,
)

response = agent.invoke(
    {"messages":[{"role":"user","content":"안녕하세요"}]},
    context = Context(user_name="SH",is_premium=True)
)

print(response)

response_hs = agent.invoke(
    {"messages":[{"role":"user","content":"안녕하세요"}]},
    context = Context(user_name="Hs",is_premium=False)
)

print(response_hs)