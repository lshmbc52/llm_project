import common_utils_solar as utils
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

model = utils.get_solar_model(model_name="solar-pro")

tools =[]

agent = create_agent(
    model= model,
    tools =[],
    checkpointer = InMemorySaver(),
)

response = agent.invoke(
    {"messages":[{"role":"user","content":"안녕? 저는 sh 입니다."}]},
    {"configurable":{"thread_id":"thread_1"}}
)

#print(response["messages"][-1].content)
response = agent.invoke(
    {"messages":[{"role":"user","content":"사용자의 이름이 뭐라고 했죠"}]},
    {"configurable":{"thread_id":"thread_1"}},
)

#print(response["messages"][-1].content)

response = agent.invoke(
    {"messages":[{"role":"user","content":"지금까지 무슨 얘기를 나누었죠?"}]},
    {"configurable":{"thread_id":"thread_1"}}
)

# for i, msg in enumerate(response["messages"]):
#     print(f"-------Messages{i+1}---------")
#     print(msg.content)
#     print()
print(response)

