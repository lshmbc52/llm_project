from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

model = init_chat_model("gpt-5-nano")


@tool
def ask_human(question: str) -> str:
    """사용자에게 추가정보를 물어볼 때 사용하는 도구입니다.
    사용자에게 답변을 받으려면 이도구를 호출하세요.
    """
    return "Human input required"


tools = [ask_human]


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def agent_node(state: ChatState):
    llm_with_model = model.bind_tools(tools)

    response = llm_with_model.invoke(state["messages"])
    return {"messages": [response]}


def human_node(state: ChatState):  # LLM이 사람에게 물어볼 내용을 담은 함수
    """
    AI가 'ask_human'도구를 호출했을 때 실행되는 노드입니다.
    실제로 도구를 실행하는 대신, interrupt를 걸어 사용자의 입력을 받습니다.
    """

    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    question_to_user = tool_call["args"]["question"]

    user_answer = interrupt(question_to_user)

    return {
        "messages": [ToolMessage(content=user_answer, tool_call_id=tool_call["id"])]
    }


workflow = StateGraph(ChatState)

workflow.add_node("agent", agent_node)
workflow.add_node("human_node", human_node)


def should_continue(state: ChatState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "human_node"
    return END


workflow.add_conditional_edges("agent", should_continue, ["human_node", END])

workflow.add_edge(START, "agent")
workflow.add_edge(
    "human_node",
    "agent",
)

memory = InMemorySaver()

app = workflow.compile(checkpointer=memory)

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_interrupt_in_tools.png")
# print("그래프 저장완료")

if __name__ == "__main__":

    thread_config = {"configurable": {"thread_id": "2"}}

    input_data = {
        "messages": [
            SystemMessage(
                content="""당신은 강원도 영월에 있는 캠핑장 추천 비서입니다, 
                1.사용자가 준 정보가 너무 부족할 때만 'ask_human'도구를 사용해 질문하세요.
                2.어느 정도 정보(인원,예산,환경 등)이 파악되었다면 더 이상 질문하지 말고 즉시 영월에 있는
                캠프장 2-4곳을 추천해 주세요
                3. 추천시에는 캠핑장 이름, 특징,선정이유를 명확히 밝히세요.
                """
            ),
            HumanMessage(content="강원도 영월에 있는 캠핑장 좀 추천해 줄래?"),
        ]
    }

    result_1 = app.invoke(input_data, config=thread_config)
    # print(result_1)

    natural_answer = " 3월 주말에 갈 예정임.텐트, 인원은 4명, 숲과 냇가가 있으면 좋고 1박에 6만원,아이들 안정성등을  고려해서 이제 그냥 2-4곳만 추천해줘 "

    resume_command = Command(resume=natural_answer)

    result_2 = app.invoke(resume_command, config=thread_config)
    print(result_2)
    print("--------------------------------")
    print(result_2["messages"][-1].content)

    natural_answer_1 = " 인터넷 검색 하세요"

    resume_command = Command(resume=natural_answer_1)

    result_3 = app.invoke(resume_command, config=thread_config)
    print("==================================")
    print(result_3)
    print(result_3["messages"][-1].content)

    natural_answer_3 = "영월캠프라는 캠프장은 어떤가요? 위의 조건들을 만족하는 곳인가요? 인터넷 찾아서 알려주세요"
    resume_command = Command(resume=natural_answer_3)
    result_4 = app.invoke(resume_command, config=thread_config)
    print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")
    print(result_4)
    print(result_4["messages"][-1].content)
