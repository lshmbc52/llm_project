import common_utils_solar as utils
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator

model = utils.get_solar_model(model_name="solar-pro")

@tool
def send_email_tool(to:str, subject:str,body:str)-> str:
    """
    지정한 이메일 주소로 메일을 보내는 도구입니다.
    """
    return f"이메일이 성공적으로 전송되었음.\n수신자:{to}\n제목:{subject}\n내용:{body[:50]}..."

@tool
def read_email_tool(limit:int=3)-> utils.List[utils.Dict[str,str]]:
    """
    최근 받은 이메일 3개를 읽는 도구임
    """
    return f"이 메일일 성공적으로 조회되었음"

agent = create_agent(
    model = model,
    tools =[send_email_tool, read_email_tool],
    middleware = [
        LLMToolEmulator(model = 'solar-pro')
    ], 
)

response = agent.invoke({
    "messages":[{"role":"user", "content":"최근에 온 이메일 3개만 확인하고 답장해줘"}]
},
)

print(response["messages"][-1].content)
print("____________")
print(response)
