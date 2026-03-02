from langchain.chat_models import init_chat_model
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph

model = init_chat_model(model="solar-pro", model_provider="upstage")


class SupportState(TypedDict):
    query: str
    category: str
    response: str


class RouteDecision(BaseModel):
    category: Literal["billing", "technical", "shipping", "general"] = Field(
        description="고객 문의 내용을 분석해서 적절한 부서를 선택하세요."
    )


router_llm = model.with_structured_output(RouteDecision)


def router_node(state: SupportState):
    query = state["query"]
    print(f"\n--- [Router] 문의 분석중:'{query}'....")

    decision = router_llm.invoke(query)
    print(f" --> 분류결과:{decision.category.upper()}부서로 연결합니다")

    return {"category": decision.category}


def billing_expert(state: SupportState):
    print(f"---[Billing Expert] 결제 전문가가 답변 작성중")
    prompt = f"당신은 결제 및 환불 전문가입니다. 다음 문의에 대해 정중하게 답변하세요: {state["query"]}"
    msg = model.invoke(prompt)
    return {"response": msg.content}


def technical_expert(state: SupportState):
    print(f"---[Tech Expert]기술 담당자가 답변 작성중")
    prompt = f"당신은 IT엔지니어입니다. 다음 IT관련 문의에 대해서 해결책을 제시해주세요.:{state['query']}"
    msg = model.invoke(prompt)
    return {"response": msg.content}


def shipping_expert(state: SupportState):
    print(f"---[Shipping Expert] 물류담당자가 지금 확인중")
    prompt = f"당신은 물류 배송 담당장 입니다. 다음 물류 문의에 대해서 답변해 주세요:{state['query']}"
    msg = model.invoke(prompt)
    return {"response": msg.content}


def general_expert(state: SupportState):
    print(f"---[General Expert] 일반적인 문제에 대해서 답변중")
    prompt = f"당신은 종합적인 문의에 대해서 답변해 주세요:{state["query"]}"
    msg = model.invoke(prompt)
    return {"response": msg.content}


workflow = StateGraph(SupportState)

workflow.add_node("router_node", router_node)
workflow.add_node("billing_expert", billing_expert)
workflow.add_node("technical_expert", technical_expert)
workflow.add_node("shipping_expert", shipping_expert)
workflow.add_node("general_expert", general_expert)

workflow.add_edge(START, "router_node")


def route_to_expert(state: SupportState):
    category = state["category"]

    if category == "billing":
        return "billing_expert"
    elif category == "technical":
        return "technical_expert"
    elif category == "shipping":
        return "shipping_expert"
    else:
        return "general_expert"


workflow.add_conditional_edges(
    "router_node",
    route_to_expert,
    ["billing_expert", "technical_expert", "shipping_expert", "general_expert"],
)

workflow.add_edge("billing_expert", END)
workflow.add_edge("technical_expert", END)
workflow.add_edge("shipping_expert", END)
workflow.add_edge("general_expert", END)

app = workflow.compile()

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_routing.png")
print("그래프 저장완료")

if __name__ == "__main__":
    result_bill = app.invoke(
        {"query": "지난 달 요금이 두 번 빠져 나갔어요. 확인해 주세요"}
    )
    print(result_bill)

    result_tech = app.invoke({"query": "API 연결할 때 404 에러발생해요"})
    print(result_tech)

    result_ship = app.invoke({"query": " 주문한 노트북 언제 도착하나요?"})
    print(result_ship)

    result_gen = app.invoke({"query": "그곳의 주소와 이메일 번호를 알려주세요"})
    print(result_gen)
