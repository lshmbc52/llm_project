from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage

model = init_chat_model("gpt-5-nano")


class DeliveryState(TypedDict):
    address: str
    package_info: str
    delivery_status: str


def pickup_package(state: DeliveryState):
    print(f"\n [배달업체] 픽업완료 (물품:{state['package_info']})")
    return {"delivery_status": "picked_up"}


def deliver_to_customer(state: DeliveryState):
    print(f"\n [배달업체]{state['address']}로 배송중...배달완료")
    return {"delivery_status": "delivered"}


delivery_builder = StateGraph(DeliveryState)

delivery_builder.add_node("pickup", pickup_package)
delivery_builder.add_node("deliver", deliver_to_customer)

delivery_builder.add_edge(START, "pickup")
delivery_builder.add_edge("pickup", "deliver")
delivery_builder.add_edge("pickup", END)

delivery_graph = delivery_builder.compile()

from IPython.display import Image

delivery_graph.get_graph().draw_mermaid_png(
    output_file_path="graph_image_state_seperate.png"
)
# print("그래프 저장완료")


class RestaurantState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    menu: str
    customer_address: str
    final_status: str


def cook_food(state: RestaurantState):
    print(f"\n[주방]{state['menu']}조리완료! 포장했습니다")
    return {"messages": ["조리완료"]}


def call_delivery_service(state: RestaurantState):
    print(f"\n [매니저]배달 업체 호출중...")

    delivery_input = {
        "address": state["customer_address"],
        "package_info": state["menu"],
    }

    delivery_result = delivery_graph.invoke(delivery_input)
    final_status = delivery_result["delivery_status"]

    return {"final_status": final_status}
