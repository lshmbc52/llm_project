from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage,HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
import uuid
from langgraph.store.base import BaseStore
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

model = init_chat_model('gpt-5-nano')

class ChatState(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]

def memory_agent_node(state:ChatState,config,store:BaseStore):
    user_id = config['configurable']['user_id']
    namespace = (user_id,'profile')

    last_message = state["messages"][-1]

    if "remember" in last_message.content:
        memory_id = str(uuid.uuid4()) #파일생성
        store.put(
            namespace,
            memory_id,
            {
                "content":last_message.content
            }
        )
    
    memories = store.search(namespace)
    if memories:
        memory_text = "\n".join([f"- {m.value['content']}" for m in memories])
        system_msg =f"""
        당신은 사용자의 정보를 기억하는 비서입니다.
        [장기 기억 저장소]
        {memory_text}
        위 기억을 참고해서 답변하세요.
        """
    else:    
        system_msg = "당신은 도움이 되는 비서입니다. 아직 사용자에 대해  아는 것이 없습니다."

    prompt =  [SystemMessage(content=system_msg)] + state["messages"]
    
    return {"messages":[model.invoke(prompt)]}

workflow = StateGraph(ChatState)

workflow.add_node("agent", memory_agent_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent",END)

checkpointer = InMemorySaver()
store = InMemoryStore()

app = workflow.compile(checkpointer=checkpointer, store= store)

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_longterm-emory.png")
#print("그래프 저장완료")

if __name__ =="__main__":

    config_1 = {'configurable':{
        "thread_id":"store_demo",
        "user_id":"영월캠프",
    }
    }

    inputs_1 ={"messages":[HumanMessage(content="내이름은 영웙캠프야,나는 비오는 날씨를 싫어해. 꼭 기억해.")]}

    resp_1 = app.invoke(inputs_1,config_1)
    print(resp_1)
    
    print(resp_1["messages"][-1].content)

    config_2 ={'configurable':{
                'thread_id':"thread_2",
                "user_id":'영월캠프',
               }
            }

    inputs_2 ={"messages":[HumanMessage(content="비오는 날에는 어떻게 해야 돼? 나는 야외에 있는 캠핑장이야")]}
    resp_2 = app.invoke(inputs_2, config_2)
    print(resp_2)
    print(resp_2["messages"][-1].content)
