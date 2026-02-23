from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AnyMessage,SystemMessage,HumanMessage,ToolMessage,AIMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START,END

model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

from langchain.tools import tool

@tool
def search_manual(query:str)-> str:
    """
    고객의 질문에 답변하기 위해 참고할 만한 메뉴얼을 검색할 때 사용하는 도구임
    """
    if "비밀번호" in query:
        return "비밀번호변경은 마이페이지 -> 보안설정에 있음"
    elif '배송' in query:
        return '00택배에서 3일내 배송 예정임'
    else:
        return '해당 내용 관련 메뉴얼을 찾을 수 없습니다.'

tools =[search_manual]

model_with_tools = model.bind_tools(tools)
   
class AgentState(TypedDict):
    messages : Annotated[list[AnyMessage],add_messages]
    next_step:str

def classify_node(state: AgentState):
    print('f\n-- 1. 분류단계(LLM 판단)')
    last_message = state['messages'][-1]
    prompt ="""
    당신은 고객센터 관리자 입니다. 고객의 이메일을 분석해서 다음 단계를 결정하세요.
    1.  단순 문의나 정보요청은 --> 'consultant'반환.
    2.  환불요청,불만제기, 화난 고객이라면 --> 'escalate'반환.
    3.  답변은 오직 단어 하나만 하세요.
    """
    response = model.invoke([SystemMessage(content=prompt), last_message])
    raw_content = response.content
    
    if isinstance(raw_content,list):
        decision = "".join([block['text'] for block in raw_content if block.get('type')== 'text'])
    else:
        decision = str(raw_content)

    decision = decision.strip().lower()
    print(f'--> LLM 판단결과:{decision}')

    if "escalate" in decision:
        return {'next_step': 'escalate'}
    else:
        return {'next_step':'consultant'}

def consultant_node(state:AgentState):
    print(f'\n--- [2-A]상담 AI 답변 생성중')
    response = model_with_tools.invoke(state["messages"])
    return {"messages":[response]}

def escalate_node(state:AgentState):        
    return {"messages":[AIMessage(content="해당 메일은 전문 상담원에게 이관되었습니다.")]}

tools =[search_manual]

tools_by_name = {tool.name:tool for tool in tools}

def tool_node(state:AgentState):
    print(f"\n---[Tool Node]도구 직접실행----")

    result =[]
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        print(f"--> 실행중:{tool_call['name']}")
        tool_result = tool.invoke(tool_call['args'])

        result.append(ToolMessage(content=str(tool_result), tool_call_id = tool_call["id"]))

    return {"messages":result}

agent_builder = StateGraph(AgentState)

agent_builder.add_node('classify_node', classify_node)
agent_builder.add_node('consultant_node', consultant_node)
agent_builder.add_node('escalate_node', escalate_node)
agent_builder.add_node('tool_node', tool_node)

agent_builder.add_edge(START,'classify_node')

def route_after_classify(state:AgentState):
    if state['next_step']=='escalate':
        return 'escalate'
    else:
        return 'consultant'

def should_continue(state:AgentState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return 'tool_node'
    return END

agent_builder.add_conditional_edges(
    "classify_node",
    route_after_classify,
    { "escalate": 'escalate_node',
      "consultant":"consultant_node"
    },
)

agent_builder.add_conditional_edges(
    'consultant_node',
    should_continue,
    ['tool_node',END]
)

agent_builder.add_edge('tool_node','consultant_node')
agent_builder.add_edge('escalate_node',END)

agent = agent_builder.compile()

from IPython.display import Image

agent.get_graph().draw_mermaid_png(output_file_path="graph_image_2.png")
print("그래프 저장완료")

if __name__ =="__main__":
    inputs = {"messages":[HumanMessage(content="배송은 언제와요?")]}
    response = agent.invoke(inputs)
    #print(response)

    print(response["messages"][-1].content[-1]['text'])

    inputs ={"messages":[HumanMessage(content="당장 환불해 줘")]}
    response = agent.invoke(inputs)
    print(response)

