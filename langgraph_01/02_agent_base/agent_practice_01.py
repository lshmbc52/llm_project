from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

from typing_extensions import TypedDict

class AgentState(TypedDict):
    email_content:str
    category:str
    next_step:str
    response:str

def read_email(state:AgentState):
    return{"email_content":state["email_content"]}

def classify_intent(state:AgentState):
    email = state['email_content']
    if '환불' in email or '긴급' in email or '빨리' in email:
        category ='complaint'
        next_step = 'escalate_to_human'
    else:
        category ='inquiry'
        next_step = 'search_manual'
    return {'category':category,'next_step':next_step}

def search_manual(state:AgentState):
    print('3-A 진입... 메뉴얼을 검색합니다')
    return {}

def escalate_to_human(state:AgentState):
    print(f'3-B 진입... 상담원으로 이관함')
    return {'response':'불편을 드려 죄송합니다.상담원에게 이관했으니 잠시 기다려 주세요'}

def write_reply(state:AgentState):
    email = state['email_content']
    response = model.invoke(email)
    return {'response':response}

agent_builder = StateGraph(AgentState)

agent_builder.add_node('read_email', read_email)
agent_builder.add_node('classify_intent', classify_intent)
agent_builder.add_node('search_manual', search_manual)
agent_builder.add_node('escalate_to_human',escalate_to_human)
agent_builder.add_node('write_reply',write_reply)

agent_builder.add_edge(START, 'read_email')
agent_builder.add_edge('read_email', 'classify_intent')

def route_email(state:AgentState):
        if state['next_step'] == 'escalate_to_human':
            return 'escalate_to_human'
        else:
            return 'search_manual'
        


agent_builder.add_conditional_edges(
    'classify_intent',
    route_email,
    ['escalate_to_human','search_manual'],
)
agent_builder.add_edge('search_manual','write_reply')
agent_builder.add_edge('write_reply',END)
agent_builder.add_edge('escalate_to_human',END)

agent = agent_builder.compile()

from IPython.display import Image

agent.get_graph().draw_mermaid_png(output_file_path="graph_image_1.png")
print("그래프 저장완료")

if __name__=="__main__":
    inputs = {"email_content":"비밀번호 변경방법을 알려주세요"}
    response = agent.invoke(inputs)
    #print(response)

    inputs ={'email_content':'당장 환불해 주세요'}
    response = agent.invoke(inputs)
    print(response)

    inputs ={'email_content':'당장 환급해 주세요'}
    response = agent.invoke(inputs)
    print(response)







    




