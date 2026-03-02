from langchain.chat_models import init_chat_model
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import common_utils as utils

model = utils.get_solar_model(model_name="solar-pro", temperature=1.0)

# model = init_chat_model("solar-pro", temperature=1.0)


class MovieState(TypedDict):
    topic: str
    draft_comment: str
    improved_comment: str
    final_comment: str


def generate_movie_comment(state: MovieState):
    topic = state["topic"]
    print(f"\n---[1단계] '{topic}'주제로  영화 코멘트 초안 생성중 ")
    msg = model.invoke(
        f"'{topic}'에 대한 짧은 코멘트를 만들어주고. 다른 말은 하지말고 코멘트 하나면 돼"
    )
    return {"draft_comment": msg.content}


def critique_and_improve(state: MovieState):
    original_comment = state["draft_comment"]
    print(f"\n --[2단계] 좀 더 깊게 수정중")

    prompt = f"""
            다음 코멘트를 보고, 더 깊이있는 영화코멘트로 개선해주고, 다른 말은 하지말고 하나만 답변해 줘
            원문:{original_comment}
            """
    msg = model.invoke(prompt)
    return {"improved_comment": msg.content}


def polish_comment(state: MovieState):
    improved_comment = state["improved_comment"]
    print(f"\n---- [3단계] 이모지 추가및 마무리")
    prompt = f"""
            다음 영화에 적절한 이모지를 듬뿍 넣어서 SNS에 올리기 좋게 꾸며 줘.
            농담:{improved_comment}
            """
    msg = model.invoke(prompt)
    return {"final_comment": msg.content}


app = StateGraph(MovieState)
app.add_node("generate_movie_comment", generate_movie_comment)
app.add_node("critique_and_improve", critique_and_improve)
app.add_node("polish_comment", polish_comment)

app.add_edge(START, "generate_movie_comment")
app.add_edge("generate_movie_comment", "critique_and_improve")
app.add_edge("critique_and_improve", "polish_comment")
app.add_edge("polish_comment", END)

chain = app.compile()
from IPython.display import Image

chain.get_graph().draw_mermaid_png(output_file_path="graph_image_movie.png")
print("그래프 저장완료")

if __name__ == "__main__":

    inputs = {"topic": "그을린 사랑(Incendies)"}
    result = chain.invoke(inputs)
    print(result)

    print(f"주제:{result['topic']}")
    print(f"1차초안:{result['draft_comment']}")
    print(f"2차초안:{result['improved_comment']}")
    print("------------------")
    print(f"최종완성:\n{result['final_comment']}")

    # try:
    #     print("--- 통신 시도 중... ---")
    #     for chunk in chain.stream(inputs):
    #         print(f"현재 단계: {list(chunk.keys())[0]}")
    # except Exception as e:
    #     print(f"에러 발생: {e}")
