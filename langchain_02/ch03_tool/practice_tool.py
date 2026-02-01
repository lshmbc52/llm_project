from langchain.tools import tool
from langchain.agents import create_agent
import common_utils_solar as utils

model = utils.get_solar_model(model_name='solar-pro')

@tool
def get_weather(location:str)-> str:
    """특정지역의 날씨 정보 제공"""
    return f"{location} 날씨는 맑고 영하 5도입니다"

agent = create_agent(
    model = model,
    tools =[get_weather],
)
result = agent.invoke(
    {"messages":[{"role":"user","content":"김천의 날씨는 어때요?"}
    ]},
    )
print(result)
print("---------------")
print(result["messages"][-1].content)



