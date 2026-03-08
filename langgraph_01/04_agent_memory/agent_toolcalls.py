from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.store.base import BaseStore
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage, HumanMessage, ToolMessage
import uuid
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

model = init_chat_model("gpt-5-nano")


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@tool
def save_profile(info: str):
    """
    사용자에 대한 중요한 정보(이름,취미,특징 등)를 저장할 때 사용함.
    단순한 대화나 인사는 저장하지말 것
    """
    print(f"저장한 정보:{info}")
    return "saved"


tools = [save_profile]

model_with_tools = model.bind_tools(tools)


def agent_node(state: ChatState, config, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "profile")

    memories = store.search(namespace)
    if memories:
        info = "\n".join([f"- {m.value['data']}" for m in memories])
        system_msg = f"""
        당신은 사용자의 정보를 기억하는 비서입니다.\n[기억된 정보]\n{info}
        """
    else:
        system_msg = "당신은 사용자의 기억을 담당하는 비서입니다"

    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=system_msg)] + state["messages"]
            )
        ]
    }


def save_node(state: ChatState, config, store: BaseStore):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "profile")

    last_message = state["messages"][-1]

    tool_outputs = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "save_profile":
            info_to_save = tool_call["args"]["info"]

            print(f'\n [System] AI의 요청으로 정보를 저장합니다:"{info_to_save}"')

            memory_id = str(uuid.uuid4())
            store.put(namespace, memory_id, {"data": info_to_save})

            tool_outputs.append(
                ToolMessage(
                    content=f"정보 저장 완료:{info_to_save}",
                    tool_call_id=tool_call["id"],
                )
            )
    return {"messages": tool_outputs}


workflow = StateGraph(ChatState)

workflow.add_node("agent", agent_node)
workflow.add_node("save_node", save_node)
workflow.add_edge(START, "agent")


def should_continue(state: ChatState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "save_node"
    return END


workflow.add_conditional_edges("agent", should_continue, ["save_node", END])

workflow.add_edge("save_node", "agent")
workflow.add_edge("agent", END)

checkpointer = InMemorySaver()
store = InMemoryStore()

app = workflow.compile(checkpointer=checkpointer, store=store)

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_toolcalls.png")
# print("그래프 저장완료")

config = {"configurable": {"thread_id": "1", "user_id": "usr-영월캠프"}}

input_1 = {
    "messages": [HumanMessage(content="안녕, 나는 강원도 영월군에 있는 영월캠프야")]
}

resp_1 = app.invoke(input_1, config=config)
print(resp_1)
print("-----------------------")
print(resp_1["messages"][-1].content)

input_2 = {"messages": [HumanMessage(content="내 이름이 뭐고 어디 산다고 했지?")]}

config = {"configurable": {"thread_id": "2", "user_id": "usr-영월캠프"}}

resp_2 = app.invoke(input_2, config=config)
print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
print(resp_2)
print(resp_2["messages"][-1].content)
