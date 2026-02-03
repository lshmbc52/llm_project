import common_utils_solar as utils
from langchain.tools import tool
from langchain.agents.middleware import LLMToolEmulator, PIIMiddleware
from langchain.agents import create_agent

model = utils.get_solar_model(model_name="solar-pro")


@tool
def save_customer_feedback(feedback: str) -> str:
    """고객 피드백을 저장하는 도구"""
    return f"고객 피드백 저장완료:{feedback}"


agent = create_agent(
    model,
    tools=[save_customer_feedback],
    middleware=[
        LLMToolEmulator(model="solar-pro"),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_output=True),
    ],
)

if __name__ == "__main__":
    # prompt = "저는 sh(sh@sh.com)입니다. 어제 스마트워치를 구매했는데 결재여부를 알려주세요"
    prompt = "저는 sh(sh@sh.com)입니다. 제카드 번호는 4234567843211234 입니다.어제 스마트워치를 구매했는데 결재여부를 알려주세요"

    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
    )
    print(response)
