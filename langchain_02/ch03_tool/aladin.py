import common_utils_solar as utils
from langchain.tools import tool
from langchain.agents import create_agent
import requests

model = utils.get_solar_model(model_name="solar-pro")

@tool
def fetch_aladin_bestseller_top10()-> utils.List[utils.Dict[str, utils.Any]]:
    """
    알라딘 베스트셀러 목록을 조회하고 Top N개(기본 10)를 반환합니다
    """
    from langchain.tools import tool
from typing import List, Dict, Any

import requests

@tool
def fetch_aladin_bestseller_top10() -> List[Dict[str, Any]]:
    """
    알라딘 베스트셀러 목록을 조회하고 Top N개(기본 10)를 반환합니다.
    """
    url = "http://www.aladin.co.kr/ttb/api/ItemList.aspx"
    params = {
        "ttbkey": "ttblshmbc12106001",
        "QueryType": "Bestseller",
        "MaxResults": 10,
        "start": 1,
        "SearchTarget": "Book",
        "output": "js",
        "Version": "20131101"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("item", [])
    top10 = items[:10]
    return top10

agent = create_agent(
    model = model,
    tools = [fetch_aladin_bestseller_top10],
)

result = agent.invoke({
    "messages":[{"role":"user","content":"현재 aladin의 bestseller 10권을 알려줘"}]
},
)

print(result["messages"][-1].content)


