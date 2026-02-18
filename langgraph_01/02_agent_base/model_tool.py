import common_utils as utils
from langchain.tools import tool

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

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
