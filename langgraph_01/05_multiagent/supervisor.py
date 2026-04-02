from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.types import interrupt
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from typing import Optional

model = init_chat_model("gpt-5-nano")


class TeamState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class RouteResponse(BaseModel):
    next_step: Literal["researcher", "writer", "FINISH"] = Field(
        description="다음 작업을 수행할 팀원을 선택하거나, 모든 작업이 완료되었으면 FINISH를 선택"
    )
    reason: str = Field(description="선택한 이유")


supervisor_agent = model.with_structured_output(RouteResponse)


def supervisor_node(state: TeamState):
    print(f"\n [Supervisor]다음 작업자 결정중----")

    system_prompt = """
    당신은 블로그 작성팀의 관리자입니다.
    1.사용자 요청이 들어오면'researcher'에게 조사를 시키세요.
    2.조사 결과가 나오면 'writer'에게 글 작성을 시키세요.
    3.글 작성이 끝나면 'FINISH'를 선언하세요.
    """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    decision = supervisor_agent.invoke(messages)

    next_worker = decision.next_step
    reason = decision.reason

    print(f"다음 작업자:{decision.next_step}, 근거:{decision.reason}")

    if next_worker == "FINISH":
        return Command(goto=END, update={"next_step": "FINISH"})
    else:
        return Command(goto=next_worker, update={"next_step": "next_worker"})


def researcher_node(state: TeamState):
    print(f"\n [Researcher] LLM이 자료를 조사/생성중입니다.")

    system_msg = SystemMessage(
        content="""
    당신은 한국의 야영camp장 관련 전문 리서쳐입니다.
    사용자의 요청 주제에 대해 핵심 트랜드, 통계,야영장비,위치,설비 등을 포함하여 전문적인 조사 보고서를 3줄 요약 형태로 작성하세요.
    """
    )
    messages = [system_msg] + state["messages"]

    response = model.invoke(messages)

    return Command(
        update={
            "messages": [HumanMessage(content=response.content, name="Researcher")]
        },
        goto="supervisor",
    )


def writer_node(state: TeamState):
    print(f"\n [Writer] LLM이 블로그 포스팅을 작성중입니다...")

    system_msg = SystemMessage(
        content="""
    당신은  한국 야영camp장 관련 블로그 전문 작가입니다.
    위의 대화 내역에 있는 'Researcher'의 조사 결과를 바탕으로 매력적인 블로그 포스팅(제목+본문)을 작성하세요.
    이모지를 적절히 사용하여 가독성을 높이세요.
    """
    )

    messages = [system_msg] + state["messages"]
    response = model.invoke(messages)

    return Command(
        update={"messages": [HumanMessage(content=response.content, name="Writer")]},
        goto="supervisor",
    )


workflow = StateGraph(TeamState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "supervisor")

checkpointer = InMemorySaver()

app = workflow.compile(checkpointer=checkpointer)

app.get_graph().draw_mermaid_png(output_file_path="graph_image_supervisor.png")
# print("그래프 저장완료")

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "team-blog-1"}}
    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="한국 야영camp장의 최신 트랜드에 대해서 블로그 글을 써줘 "
                )
            ]
        },
        config=config,
    )
    print(result)
