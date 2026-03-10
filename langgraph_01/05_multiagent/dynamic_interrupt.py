from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage,SystemMessage,HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

model = init_chat_model("gpt-5-nano")

class ChatState(TypedDict):
    messages:Annotated[list[AnyMessage], add_messages]
    context:dict

def agent_node(state:ChatState):
    ctx = state.get("context",{})
    food_category = ctx.get("food_category")

    if not food_category:
        print("[System]음식 종류 부재.사용자에게 질의(interrput) 시도...)")

        food_category = interrupt("당신은 어떤 종류의 음식을 먹고 싶나요?(예:한식,중식, 일식)")

    system_prompt = SystemMessage(content = f"""
        당신은 서울의 맛집 전문가입니다.
        사용자가 원하는 카테고리인 {food_category}에 맞춰서 실제로 유명햔 맛집 1 곳을 추천하고,
        추천이유를 2문장으로 설명해 주세요
        """)
    messages = [system_prompt] + state["messages"]
    response = model.invoke(messages)

    return{"messages":[response],
           "context":{"food_category":food_category}}

workflow = StateGraph(ChatState)
workflow.add_node("chatbot",agent_node)

workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot",END)

memory = InMemorySaver()

app = workflow.compile(checkpointer=memory)

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_dynamic_interrupty.png")
#print("그래프 저장완료")

config_1 = {"configurable":{"thread_id":"1"}}

input_msg1 = {"messages":[HumanMessage(content="신림역 근처 맛집 추천해 줘")],
"context":{}
 }

response1 = app.invoke(input_msg1,config = config_1)
print(response1)
 
snapshot = app.get_state(config_1)
#print(snapshot)

resume_command = Command(resume="한식")

result_2 = app.invoke(resume_command, config= config_1)
# print(result_2)
# print(result_2["messages"][-1].content)
# print("--- 최종 결과 확인 ---")
# if "messages" in result_2:
#     print(result_2["messages"][-1].content)
# else:
#     print("아직 메시지가 생성되지 않았습니다. 상태를 확인하세요:", result_2)

input_msg3 = {
    "messages":[HumanMessage(content="김천역 근처 맛집 추천해 줘")],
}

response3 = app.invoke(input_msg3,config=config_1)
print(response3)
