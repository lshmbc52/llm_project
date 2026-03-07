from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage,HumanMessage
from typing import TypedDict, Annotated 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver


model = init_chat_model(model= 'solar-pro',model_provider='upstage')

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def chatbot_node(state:ChatState):
    return{"messages":[model.invoke(state["messages"])]}

workflow = StateGraph(ChatState)

workflow.add_node("chatbot", chatbot_node)

workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

memory = InMemorySaver()
app = workflow.compile(checkpointer = memory)

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_shortmemory.png")
print("그래프 저장완료")

if __name__ =="__main__":
    config_1 = {'configurable':{"thread_id":"1"}}

    input_msg1 = {"messages":[HumanMessage(content="안녕? 내이름은 영월캠프야")]}
    response_1 = app.invoke(input_msg1,config = config_1)
    print("------------------------------")
    print(response_1)
    print(response_1["messages"][-1].content)

    config_2 = {'configurable':{"thread_id":"2"}}
    input_msg2 = {"messages":[HumanMessage(content=" 내이름이 뭐라고 했지?")]}
    response_2= app.invoke(input_msg2,config = config_2)
    #print(response_2)
    print("===========================================")
    print(response_2['messages'][-1].content)

    print("---------------------------")
    
    current_state = app.get_state(config_1)
    #print(current_state)
    print("*****************************************")
    print(current_state.values["messages"][-1].content)
    print(current_state.next)

    print(current_state.config)

    history = list(app.get_state_history(config_1))
    print(history)


