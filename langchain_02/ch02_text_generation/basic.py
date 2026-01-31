from langchain.chat_models import init_chat_model

model = init_chat_model(
    "solar-pro",
    temperature=0.7,
    timeout=30,
    max_tokens=5000,
)

response = model.invoke("안녕하세요, 당신은 누구십니까?")
print(response)
print(response.usage_metadata)

for chunk in model.stream("당신은 누구십니까?"):
    print(chunk.text, end="")

inputs = ["양자얽힘이 뭔가요?", "숨은 변수 이론이 뭔가요?", "L2 loss가 뭔가요?`"]

responses = model.batch(inputs)

for response in responses:
    print(response, end="")
