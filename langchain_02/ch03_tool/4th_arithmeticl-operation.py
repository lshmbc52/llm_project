import common_utils_solar as utils
from langchain.agents import create_agent
from langchain.tools import tool
model = utils.get_solar_model(model_name="solar-pro")

@tool
def add(a:int, b:int)-> int:
    """
    a와 b의 덧셈
    Args:
    a:First int
    b:Second int
    """
    return a + b
@tool
def multiply(a:int, b:int)-> int:
    """
    a와 b의 곱셈 
    Args:
    a:First int
    b:Second int
    """
    return a * b
@tool
def divide(a:int, b:int)-> float:
    """
    a와 b의 나눗셈
    Args:
    a:First int
    b:Second int
    """
    return a / b
@tool
def sub(a:int, b:int)-> int:
    """
    a와 b의 뺄셈
    Args:
    a:First int
    b:Second int
    """
    return a - b

if __name__ == "__main__":
    agent_1 = create_agent(
        model = model,
        tools =[add,multiply,divide,sub],
    )

    agent = create_agent(
        model = model,
        tools =[add, multiply,divide, sub],
        system_prompt = "당신은 유능한 수학선생님입니디.오직 주어진 도구(tool)을 이용해서 단계별로 계산만 하고 마지막에 최종 결과값만 알려주세요"
    )
    result = agent.invoke(
        {"messages":[{"role":"user","content":"78 -34 *456/12는 얼마인가요?"}]},
        config={"recursion_limit":15}
        )

    print(result["messages"][-1].content)


