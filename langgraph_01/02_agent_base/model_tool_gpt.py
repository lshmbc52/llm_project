from langchain.tools import tool
from langchain.messages import AnyMessage
from typing_extensions import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain.messages import SystemMessage
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini")

@tool
def multiply(a:int, b:int)-> int:
    """Multipy 'a'and 'b'.
    Args:
    a:First int
    b:Second int
    """
@tool
def add(a:int, b:int)-> int:
    """Adds 'a' and  'b'.
    Args:
    a:First int
    b:Second int
    """
@tool
def divide(a:int, b:int)-> int:
    """Divide 'a'and 'b'.
    Args:
    a:First int
    b:Second int
    """
tools =[multiply,add, divide]

model_with_tools = model.bind_tools(tools)

class MessageState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]
    llm_calls:int

def llm_call(state):
    """LLM이 현재 상태를 보고 답변하거나,도구사용을 요청하는 Node"""
    sys_msg =model_with_tools.invoke(
        [SystemMessage(content="당신은 사칙연산하는 유능한 Agent입니다."),
         ] + state["messages"] 
    )
    return {
        "messages":[sys_msg],
        'llm_calls':state.get('llm_calls',0) + 1
    }

 
 


