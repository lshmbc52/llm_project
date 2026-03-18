from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.types import interrupt
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage


model = init_chat_model("gpt-5-nano")


class ReservationState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    menu: str
    price: str
    status: str
    chef_name: str
    cooking_time: str


def check_ingredients(state: ReservationState):
    print(f"\n [주방/재료팀] '{state['menu']}'재료 재고 확인중...")
    return {"status": "ok"}


def assign_chef(state: ReservationState):
    menu = state["menu"]

    if "오마카세" in menu:
        chef = "Master Jiro"
        time = "40분"
    else:
        chef = "Chef Kim"
        time = "20분"

    print(f"[주방/인사팀] {chef} 쉐프 배정완료(예상소요시간{time})")
    return {"chef_name": chef, "cooking_time": time, "status": "cooking"}


kitchen_builder = StateGraph(ReservationState)
kitchen_builder.add_node("check_stock", check_ingredients)
kitchen_builder.add_node("assign_chef", assign_chef)


kitchen_builder.add_edge(START, "check_stock")
kitchen_builder.add_edge("check_stock", "assign_chef")
kitchen_builder.add_edge("assign_chef", END)
kitchen_subgraph = kitchen_builder.compile()

from IPython.display import Image

kitchen_subgraph.get_graph().draw_mermaid_png(
    output_file_path="graph_image_subgraph.png"
)
# print("그래프 저장완료")


def menu_recommender(state: ReservationState):
    last_msg = state["messages"][-1].content if state["messages"] else ""

    if "비싸" in last_msg or "부담" in last_msg:
        proposal = "B코스 (런치 스페셜)"
        price = 80000
        msg = f"부담없는 {proposal} ({price:,}원은 어떤신가요?)"
    else:
        proposal = "A코스 (쉐프 오마카세)"
        price = 150000
        msg = f"시그니쳐 '{proposal} ({price,})원을 추천합니다."
    return {"menu": proposal, "price": price, "messages": msg}


class UserIntent(BaseModel):
    action: Literal["confirm", "change", "cancel"] = Field(
        description="확정,변경,취소 판단"
    )


intent_analyzer = model.with_structured_output(UserIntent)


def customer_confirm_node(state: ReservationState):
    menu = state["menu"]
    price = state["price"]

    user_input = interrupt(f"'{menu}' ({price,})원으로 하시겠습니까?")

    prompt = f"""
    현재 제안된 메뉴: {menu} ({price}원)
    사용자의 답변: "{user_input}"
    
    위 답변을 바탕으로 다음 중 하나로 분류하세요:
    - confirm: 제안을 수락하고 진행하고자 할 때
    - change: 가격이 비싸다고 하거나, 다른 메뉴를 보고 싶어할 때
    - cancel: 아예 예약을 그만두거나 거절할 때
    """

    analysis = intent_analyzer.invoke(prompt)
    print(f"[AI Router]:의도='{analysis.action}")

    if analysis.action == "confirm":
        return Command(
            goto="kitchen", update={"messages": [HumanMessage(content=user_input)]}
        )

    elif analysis.action == "change":
        return Command(
            goto="recommender", update={"messages": [HumanMessage(content=user_input)]}
        )
    else:
        return Command(
            goto="cancel", update={"messages": [HumanMessage(content=user_input)]}
        )


def cancel_node(state: ReservationState):
    print(f"\n [카운터]에약종료")
    return {"state": "canceled"}


parent_builder = StateGraph(ReservationState)

parent_builder.add_node("recommender", menu_recommender)
parent_builder.add_node("confirm", customer_confirm_node)
parent_builder.add_node("cancel", cancel_node)

parent_builder.add_node("kitchen", kitchen_subgraph)

parent_builder.add_edge(START, "recommender")
parent_builder.add_edge("recommender", "confirm")
parent_builder.add_edge("kitchen", END)

checkpointer = InMemorySaver()

app = parent_builder.compile(checkpointer=checkpointer)

app.get_graph().draw_mermaid_png(output_file_path="graph_image_parent.png")
# print("그래프 저장완료")

config = {"configurable": {"thread_id": "subgraph_demo"}}

response = app.invoke({"messages": []}, config=config)
# print(response)

user_response = "좋아요. 그걸로 주세요."

result = app.invoke(Command(resume=user_response), config=config)
print(result)

config = {"configurable": {"thread_id": "subgraph-demo-2"}}
print("==================================")
result_2 = app.invoke({"messages": []}, config=config)
print(result_2)
user_response = "아니요, 너무 비싸요."
result_3 = app.invoke(Command(resume=user_response), config=config)
print("---------------")
print(result_3)

user_response = "그렇게 할게요."
final_result = app.invoke(Command(resume=user_response), config=config)
print(final_result)
print(f"메뉴:{final_result['menu']}")
print(f"상태:{final_result['status']}")
print(f"담당세프:{final_result.get('chef_name')}")
