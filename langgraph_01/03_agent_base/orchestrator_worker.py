from langchain.chat_models import init_chat_model
from typing import List, Annotated, TypedDict
from pydantic import BaseModel, Field
import operator
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
import common_utils as utils

# model = utils.get_gemini_model(model_name="gemini-3-flash-preview")
# model = utils.get_gemini_model(model_name="gpt-4o-mini", model_provider="openai")

model = init_chat_model(model="solar-pro", model_provider="upstage")


class Section(BaseModel):
    name: str = Field(description="목차의 제목")
    description: str = Field(description="이 섹션에서 다룰 핵심내용")


class ReportState(TypedDict):
    topic: str
    sections: List[Section]
    completed_sections: Annotated[List[str], operator.add]
    final_report: str


class WorkState(TypedDict):
    section: Section


class Plan(BaseModel):
    sections: List[Section] = Field(description=" 보고서 작성을 위한 목차 리스트")


planner_llm = model.with_structured_output(Plan)


def orchestrator_node(state: ReportState):
    topic = state["topic"]
    print(f"\n---[Orchestrator]'{topic}보고서 계획수립중----")

    plan = planner_llm.invoke(
        f"'{topic}'에 대한 보고서 목차를 구성해주고, 3개 섹션 이내로 구성해 줘"
    )
    print(f"생성된 계획:{[s.name for s in plan.sections]}")
    return {"sections": plan.sections}


def worker_node(state: WorkState):
    section = state["section"]

    prompt = f"""
    댜음 섹션에 대한 내용을 짧게 작성해 줘.
    제목:{section.name}
    내용 가이드: {section.description}
    """
    res = model.invoke(prompt)
    content = f"##{section.name}\n {res.content}"
    return {"completed_sections": [content]}


def synthesizer_node(state: ReportState):
    topic = state["topic"]
    completed_docs = state["completed_sections"]
    print(f"\n--[Synthesizer] 모든 원고 취합 및 최종 편집---")

    raw_content = "\n\n".join(completed_docs)

    prompt = f"""
    당신은 전문 리포트 편집자 입니다.
    다음은 '{topic}'에 대해서 여러 전문가가 나누어 쓴 원고들입니다.
    
    이 초안들을 바탕으로 ** 하나의 자연스럽고 전문적인 보고서**로 다시 작성해 주세요.
    
    [지시사항]
    1.각 섹션의 연결이 메끄러워야 한다.
    2. 전체를 아우르는 '서론'과 '결론'을 추가해 주어야 한다.
    3. 마크다운(Markdown) 형식을 사용해서 가독성을 높여야 한다
    [원고 내용]
    {raw_content}
    """

    msg = model.invoke(prompt)
    return {"final_report": msg.content}

    # completed = state["completed_sections"]
    # final_report = "\n".join(completed)
    # return {"final_report": final_report}


def assign_workers(state: ReportState):
    sections = state["sections"]
    return [Send("worker_node", {"section": section}) for section in sections]


workflow = StateGraph(ReportState)

workflow.add_node("orchestrator_node", orchestrator_node)
workflow.add_node("worker_node", worker_node)
workflow.add_node("synthesizer_node", synthesizer_node)

workflow.add_edge(START, "orchestrator_node")

workflow.add_conditional_edges("orchestrator_node", assign_workers, ["worker_node"])

workflow.add_edge("worker_node", "synthesizer_node")
workflow.add_edge("synthesizer_node", END)

app = workflow.compile()

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_orch.png")
print("그래프 저장완료")

if __name__ == "__main__":

    inputs = {
        "topic": "강원도 영월군 무릉도원면에 있고 개인이 운영하는캠프장인 영월캠프의 향후 미래는?"
    }
    result = app.invoke(inputs)
    # print(result)

    print(result["final_report"])
