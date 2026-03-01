from langchain.chat_models import init_chat_model
from typing import TypedDict
from langgraph.graph import StateGraph, START,END

model = init_chat_model(model='solar-pro',
                        model_provider='upstage',
                        temperature= 1.0)

class WriteState(TypedDict):
    topic:str
    poem:str
    story:str
    joke:str
    final_report:str

def write_poem(state:WriteState):
    topic = state["topic"]
    print(f"[work AI] '{topic}' 주제로 시(Poem)작성 시작...")

    msg = model.invoke(f" '{topic}'에 대한 서정적인 시를 짧게  써줘")
    return {"poem": msg.content}

def write_story(state:WriteState):
    state = state['topic']
    print(f"[workAI] {'topic'}를 주제로 소설(story)작성 시작")
    msg = model.invoke(f"[workAI] {'topic'}를 주제로 짧고 감동적인 소설을 작성해 줘")
    return{"story":msg.content}

def write_joke(state:WriteState):
    topic= state['topic']
    print(f'[work AI] 조크(joke)작성 시작')
    msg = model.invoke(f"{topic}을 주제로 재미있는 조크를 짧게 만들어 줘")
    return {"joke":msg.content}

def aggregator(state:WriteState):
    print(f"\n-- [aggregator]모든원고 도착 최종편집중---")
    final_text = f"""
    1.시(Poem)
    =============
    {state['poem']}
    2.소설(Story)
    ================
    {state['story']}
    
`   3.조크(Joke)
    ________________
    {state['joke']}
    """
    return{"final_report": final_text}

workflow = StateGraph(WriteState)

workflow.add_node("write_poem", write_poem)
workflow.add_node("write_story", write_story)
workflow.add_node("write_joke", write_joke)
workflow.add_node("aggregator", aggregator)


workflow.add_edge(START,'write_poem')
workflow.add_edge(START,'write_story')
workflow.add_edge(START,'write_joke')

workflow.add_edge('write_poem','aggregator')
workflow.add_edge('write_story','aggregator')
workflow.add_edge('write_joke', 'aggregator')
workflow.add_edge('aggregator', END)

app =  workflow.compile()

from IPython.display import Image

app.get_graph().draw_mermaid_png(output_file_path="graph_image_para.png")
print("그래프 저장완료")

inputs = {"topic":"젊은 날의 봄"}
result = app.invoke(inputs)
print(result)

print(result["final_report"])
