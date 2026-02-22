from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.messages import SystemMessage

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
    messages = Annotated[list[AnyMessage],add_messages]
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
    response = model.invoke([SystemMessage(content=prompt)], last_message)
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

        






