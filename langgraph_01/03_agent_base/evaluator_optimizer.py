from langchain.chat_models import init_chat_model
from typing import TypedDict,Literal
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph,START,END

model = init_chat_model(model='solar-pro', model_provider='upstage',temperature=0.7)

class AdState(TypedDict):
    product_name:str
    ad_copy:str
    feedback:str
    status:str
    iteration_count:int

class EvaluationResult(BaseModel):
        status:Literal['pass','fail']=Field(description="기준 충족여부")
        feedback:str= Field(description='탈락 시 구체적인 수정사항')

evaluator_llm = model.with_structured_output(EvaluationResult)

def copywriter_node(state:AdState):
    product = state["product_name"]
    feedback= state.get("feedback")
    count = state.get('iteration_count',0)

    print(f"\n---[(Copywriter)]광고 문구 작성중(시도:{count + 1})----")

    if not feedback:
        prompt = f"'{product}'의 기능위주로 인스타그램 홍보문구를 건조하게 작성해줘. 홍보문구만 답변하고 반드시 20자 이하로 작성해"
    else:
        prompt =f"""
                '{product}' 인스타그램 홍보문구를 다시 작성해.
                
                <반드시 지켜야 할 수정사항>
                {feedback}
                </반드시 지겨야 할 수정사항>
                
                <작성시 반드시 지켜야 할 사항>
                홍보 문구만 답변하고 절대 50자 이하로 작성해.
                </작성시 반드 지켜야 할 사항>
                """
    res = model.invoke(prompt)
    return{"ad_copy":res.content, "iteration_count":count + 1}

def manager_node(state:AdState):
    ad_copy = state["ad_copy"]
    print(f"\n---[manager]문구 감수중")
    print(f"신입이 쓴 글:{ad_copy}")

    prompt = f"""
    당신은 깐깐한 마케팅 팀장입니다.신입사원이 작성한 다음 광고 문구를 평가하세요:
    "{ad_copy}"
    
    <평가기준>
    1.(정량) 해시태그(#)가 3개이상 있어야 함.
    2.(정량) "원시적 자연환경","전면 개조"라는 단어가 포함되어야 함.
    3.(정성- 중요!) **문구가 너무 설명문 같거나 딱딱하면 안됨됨.캠핑객들의 감성을 자극하는 '야성적이고, 활기찬 톤'이어야 함.
    </평가기준>
    
    위 3가지 기준중 하나라도 부족하면 fail을 준다.
    특히 3번(톤엔매너)이 부족하면 "좀 더 감성적으로 쓰세요" 같이 100자이내로 조언해야 함
    """
    res = evaluator_llm.invoke(prompt)
    print(f" 판정 :{res.status.upper()}")
    
    return {'status':res.status, 'feedback':res.feedback}

def route_submission(state:AdState):
    status = state["status"]
    iteration_count = state["iteration_count"]

    if status == 'pass':
        print("pass함")
        return END
    if iteration_count >= 3:
        print("3번 수정했음에도 불합격...")
        return END
    
    return 'copywriter_node'

workflow = StateGraph(AdState)
workflow.add_node('copywriter_node',copywriter_node)
workflow.add_node('manager_node',manager_node)

workflow.add_edge(START,'copywriter_node')
workflow.add_edge('copywriter_node', 'manager_node')

workflow.add_conditional_edges(
   'manager_node',
   route_submission,
   ['copywriter_node',END] 
)
app = workflow.compile()

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_eval.png")
print("그래프 저장완료")

inputs ={"product_name":"영월캠프"}
result = app.invoke(inputs)
print(result)

        