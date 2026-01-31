import common_utils_solar as utils

client = utils.ChatUpstage()
response = client.invoke(model="solar-pro", input="당신은 누구고,무슨 일을 하나요?")

# print(response.content)

prompt = """
앵무새의 털 색상이 여러개인 이유가 뭐야?
"""

response = client.invoke(
    model="solar-pro",
    # reasoning={"effort":"medium"},
    input=[{"role": "user", "content": prompt}],
)
print(response.content)
