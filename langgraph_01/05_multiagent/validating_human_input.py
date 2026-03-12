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
def ask_human(question:str)-> str:
    """LLM이 사용자에게 추가 정보를 물어볼 때 사용하는 도구입니다.
       사용자에게 답변을 받으려면 이 도구를 사용하세요
    """
    return "Human input required"
tools = [ask_human]

class ChatState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

def human_validating_node(state:ChatState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    original_question = tool_call["args"]["question"]

    valid_categories = ["위치","대여료","인원수","날짜","성수기","캠핑장"]

    current_prompt = original_question #LLM이 하는 질문

    while True:
        user_input = interrupt(current_prompt) #사용자가 나중에 보내 줄 답변임

        if any(category in user_input for category in valid_categories):
            return {
                "messages":[
                    ToolMessage(
                        content = user_input,
                        tool_call_id = tool_call["id"]
                    )
                ]
            }
        else:
            current_prompt =(
                f"죄송합니다. 가능한 단어는 {valid_categories}입니다."
            )
    
def agent_node(state:ChatState):
    llm_with_tools = model.bind_tools(tools)

    response = llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}

workflow = StateGraph(ChatState)

workflow.add_node("agent",agent_node)
workflow.add_node("human_node",human_validating_node)

def should_continue(state:ChatState): 
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        if last_message.tool_calls[0]["name"] == "ask_human":
            return "human_node"
    return END

workflow.add_conditional_edges(
    "agent",
    should_continue,
    ["human_node",END]
)

workflow.add_edge(START,"agent")
workflow.add_edge("human_node","agent")

memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_validating_human_input.png")
#print("그래프 저장완료")

if __name__ =="__main__":

    thread_config ={"configurable":{"thread_id":"3"}}

    input_data = {
        "messages":[
            SystemMessage(content="""
                       당신은 강원도 영월군 캠핑장 추천 비서입니다.
                        1. 사용자가 추천을 요청하면 "알려드릴까요?"라고 되묻지 마세요.
                        2. **첫 번째 답변부터** 반드시 구체적인 [캠핑장 이름(상호명), 주소, 전화번호, 1박 요금, 특징]을 포함하여 리스트를 작성하세요.
                        3. 실시간 확인이 필요하다는 식의 변명은 생략하고, 당신이 알고 있는 최신 데이터를 즉시 출력하세요.
                        4. '무릉도원면' 내 무릉법흥로에 있는 캠핑장을 최우선으로 추천하세요. 
                        """),
            HumanMessage(content="영월군 무릉도원면에 있는 캠핑장을 추천해 주세요")
        ]
    }

    result_1 = app.invoke(input_data, config= thread_config)
    #print(result_1)

    result_2 = app.invoke(Command(resume="횟집"),config= thread_config)
    #print(result_2)
    user_final_answer ="인원수는 4명이고, 더 물어보지 말고 지금 즉시 영월군 무릉도원면 근처 캠핑장 3곳만 추천해줘."
    result_3 = app.invoke(Command(resume=user_final_answer),config = thread_config)
    print(result_3)
    print("-----------------")
    print(result_3["messages"][-1].content)

    # result_4: AI의 마지막 의심을 해소해줍니다.
    # valid_categories에 있는 '캠핑장' 단어를 포함시킵니다.
    result_4 = app.invoke(Command(resume="응, 영월군 맞으니까 그냥 캠핑장 빨리 추천해줘."), config=thread_config)

    print("================ 드디어 나온 추천 결과 ================")
    print(result_4["messages"][-1].content)    




