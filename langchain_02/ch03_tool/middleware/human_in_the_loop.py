import common_utils_solar as utils
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator, HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

model = utils.get_solar_model(model_name="solar-pro")


@tool
def send_email_tool(to: str, subject: str, body: str) -> str:
    """
    이메일 주소로 메일을 보내는 도구입니다.
    """
    return f"이메일이 성공적으로 전송되었음.\n수신자:{to}\n제목:{subject}\n내용:{body[:50]}..."


@tool
def read_email_tool(limit: int = 3) -> utils.List[utils.Dict[str, str]]:
    """
    최근 받은 이메일 3개를 읽는 도구임
    """
    return f"이 메일일 성공적으로 조회되었음"


agent = create_agent(
    model=model,
    tools=[send_email_tool, read_email_tool],
    checkpointer=checkpointer,
    middleware=[
        LLMToolEmulator(model="solar-pro"),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "read_email_tool": False,
            }
        ),
    ],
)

if __name__ == "__main__":

    # prompt = "무슨 메일이 왔는지 확인해 줘"

    # response = agent.invoke(
    #     {"messages": [{"role": "user", "content": "무슨 메일이 왔는지 확인해 줘"}]},
    #     {"configurable": {"thread_id": "HID-a"}},
    # )

    # prompt = "교수님에게 내일 찾아뵙겠다는 메일 작성해서 보내 줘 "

    # response = agent.invoke(
    #     {"messages": [{"role": "user", "content": prompt}]},
    #     {"configurable": {"thread_id": "HId-b"}},
    # )

    prompt = "교수님 이메일:professor_kim@edu.com, 내이름:sh, 학과: 신문방송학과,장소: 교수님 연구실,내 이메일:sh@sh.com"

    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        {"configurable": {"thread_id": "HID-b"}},
    )
    print("================================")
    print(response["messages"][-1].content)
    print("____________")
    print(response)
