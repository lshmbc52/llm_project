import common_utils as utils
from langchain.tools import tool

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.messages import SystemMessage

model = utils.get_gemini_model(model_name="gemini-3-flash-preview")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b` .
    Args:
    a:First int
    b:Second int
    """


@tool
def add(a: int, b: int) -> int:
    """Adds 'a' and 'b'.
    Args:
    a:First int
    b:Seconde int
    """


@tool
def divide(a: int, b: int) -> int:
    """Divide 'a' and 'b'.
    Args:
    a:First int
    b:Second int
    """


tools = [multiply, add, divide]

model_with_tools = model.bind_tools(tools)


class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int


def llm_call(state):
    sys_msg = [SystemMessage(content="당신은 4칙연산에 유능한 Agent입니다")]
    response = model_with_tools.invoke(sys_msg + state["messages"])
    return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}


tools_by_name = {tool.name: tool for tool in tools}

from langchain.messages import ToolMessage


def tool_node(state):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_result = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
    return {"messages": result}


from langgraph.graph import StateGraph, START, END

agent_builder = StateGraph(MessageState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("tool_node", "llm_call")


def should_continue(state: MessageState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return END


agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    [END, "tool_node"],
)

agent = agent_builder.compile()

from IPython.display import Image

agent.get_graph().draw_mermaid_png(output_file_path="graph_image.png")
print("그래프 저장완료")

from langchain.messages import HumanMessage

if __name__ == "__main__":
    messages = [HumanMessage(content="3과 4를 더해 줘")]
    response = agent.invoke({"messages": messages})
    # print(response)
    print(response["messages"][-1].content[-1]["text"])
