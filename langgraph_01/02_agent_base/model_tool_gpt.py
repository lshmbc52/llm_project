from langchain.tools import tool
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage

model = init_chat_model("gpt-4o-mini")


@tool
def multiply(a: int, b: int) -> int:
    """Multipy 'a'and 'b'.
    Args:
    a:First int
    b:Second int
    """


@tool
def add(a: int, b: int) -> int:
    """Adds 'a' and  'b'.
    Args:
    a:First int
    b:Second int
    """


@tool
def divide(a: int, b: int) -> int:
    """Divide 'a'and 'b'.
    Args:
    a:First int
    b:Second int
    """


tools = [multiply, add, divide]

model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int


def llm_call(state):
    """LLM이 현재 상태를 보고 답변하거나,도구사용을 요청하는 Node"""
    response = model_with_tools.invoke(
        [
            SystemMessage(content="당신은 사칙연산하는 유능한 Agent입니다."),
        ]
        + state["messages"]
    )
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}


tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state):
    "LLM 도구 사용을 요청했을 때, 실제로 도구를 실행하는 단계"

    result = []

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_result = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    return {"messages": result}


agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("tool_node", "llm_call")


def should_continue(state: MessagesState):
    """LLM의 응답을 보고 다음 단계로 어디를 갈지 결정"""

    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"

    return END


agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])

agent = agent_builder.compile()

from IPython.display import Image

agent.get_graph().draw_mermaid_png(output_file_path="graph_image.png")
print("그래프 저장완료")

if __name__ == "__main__":

    messages = [HumanMessage(content="3과 4를 더해줘")]
    response = agent.invoke({"messages": messages})
    print(response)
