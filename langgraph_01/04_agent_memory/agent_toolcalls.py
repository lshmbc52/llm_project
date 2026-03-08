from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage,HumanMessage
import uuid
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
model = init_chat_model("gpt-5-nano")

class ChatState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

@tool
def save_profile(info:str):
    """
    사용자에 대한 중요한 정보(이름,취미,특징 등)를 저장할 때 사용함.
    단순한 대화나 인사는 저장하지말 것
    """
    print(f'저장한 정보:{info}')
    return "saved"

tools = [save_profile]

model_with_tools = model.bind_tools(tools)

def agent_node(state:ChatState,config,store:BaseStore):
    user_id = config['configurable']['user_id']
    namespace = (user_id,'profile')

    memories = store.search(namespace)
    if memories:
        info = "\n".join([f"- {m.value['data']}" for m in memories])
        system_msg =f"""
        당신은 사용자의 정보를 기억하는 비서입니다.\n[기억된 정보]\n{info}
        """
    else:    
        system_msg = "당신은 사용자의 기억을 담당하는 비서입니다"

    return{"messages":[model_with_tools.invoke([SystemMessage(content=system_msg) + state["messages"]])]}
    

