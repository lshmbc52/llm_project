import common_utils_solar as utils
from langchain.tools import tool
from langchain.agents.middleware import LLMToolEmulator, PIIMiddleware
from langchain.agents import create_agent

model = utils.get_solar_model(model_name="solar-pro")


@tool
def save_customer_feedback(feedback: str) -> str:
    """고객 피드백을 저장하는 도구"""
    return f"고객 피드백 저장완료:{feedback}"


phone_number_detector_regex = r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}"

# phone_number_detector_regex = r"\b(010)[-\s]?(\d{3,4})[-\s]?(\d{4})\b"

phone_masking_middleware = PIIMiddleware(
    pii_type="phone_number",
    detector=phone_number_detector_regex,
    strategy="mask",
    apply_to_input=True,
)

agent = create_agent(
    model=model,
    tools=[save_customer_feedback],
    middleware=[phone_masking_middleware],
)

prompt = "안녕하세요. 제번호는 010-1234-5678입니다.등록해 주세요"
response = agent.invoke(
    {"messages": [{"role": "user", "content": prompt}]},
)

print(response)
