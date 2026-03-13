from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage

model = init_chat_model("gpt-5-nano")

class ChatState(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]
    select_camp:str

def recommender_node(state:ChatState):
    last_camp = state.get("select_camp")

    if last_camp == "영월캠프(20만원)":
        print(f"'[Recommender]' 그러면 실속형 '8만원'을 추천합니다.")
        return{"select_camp": "무릉캠프(8만원)"}

    else:
        return{"select_camp":"영월캠프(20만원)"}

def human_approved_node(state:ChatState):
    camp = state["select_camp"]

    approved = interrupt(f"{camp} 오케이 하시겠습니까?(yes/no)")

    if approved == 'yes':
        return Command(
        goto = 'booking',
        update = {"messages":["사용자가 예약을 승인함"]},
        )
    else: 
        return Command(
        goto='recommender',
        update = {"messages":["사용자가 거부함"]},
    )

def booking_node(state:ChatState):
    camp = state["select_camp"]
    return {"messages":[f" '{camp}'예약완료"]}

workflow = StateGraph(ChatState)

workflow.add_node("recommender", recommender_node)
workflow.add_node("approval", human_approved_node)
workflow.add_node("booking", booking_node)

workflow.add_edge(START,"recommender")
workflow.add_edge("recommender","approval")

checkpointer = InMemorySaver()

app = workflow.compile(checkpointer= checkpointer)

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_command_with_goto.png")
#print("그래프 저장완료")

if __name__ =="__main__":
    config = {'configurable':{"thread_id":"goto"}}
    response = app.invoke({"messages":[HumanMessage(content="예약해 주세요")]}, config = config)
    print(response)
    
    snapshot = app.get_state(config)
    print(f"{snapshot.tasks[0].interrupts[0].value}")

    result = app.invoke(Command(resume="no"), config =config)
    print(result)
    print(result["messages"][-1].content)
    result = app.invoke(Command(resume="yes"), config =config)
    print("---------------------") 
    print(result["messages"][-1].content)

    config ={"configurable":{"thread_id":"goto_2"}}
    print("=================================")
    result_2 = app.invoke(Command(resume="yes"),config=config)
    result_3 = app.invoke(Command(resume="yes"),config=config)
    print("\\\\\\\\\\\\\\\\\\\\\\\\")
    print(result_3)

    
    






