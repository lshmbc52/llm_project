from langchain.chat_models import init_chat_model
from retrieve import retriever_mmr

model = init_chat_model("solar-pro")

response = model.invoke("ARKK 펀드의 Tesla 투자 비중이 어떻게 돼?")
# print(response)

from langchain.tools import tool


@tool
def search_portfolio(query: str):
    """ARKK ETF의 포토폴리오 정보를 검색할 때 사용합니다.
    특정기업의 보유비중, 주식 수, 가치등을 찾을 때 이 도구를 호출하세요.
    """
    docs = retriever_mmr.invoke(query)
    return "\n".join([doc.page_content for doc in docs])


tools = [search_portfolio]

from langchain.agents import create_agent

agent = create_agent(model="solar-pro", tools=tools)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "ARKK 펀드의 Tesla 투자비용이 어떻게 돼?"}
        ]
    },
)
print("-------------------------")

# print(response["messages"][-1].content)

# print(response)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "ARKK의 펀드의 NVIDIA 투자비용이 어떻게 돼? "}
        ]
    }
)

print(response["messages"][-1].content)
print("----------------")
print(response)
