from google import genai
import common_utils as utils

model = utils.get_gemini_model(model_name="gemini-3-flash-preview")

response = model.invoke("langchain과 langgraph의 차잇점을 알려주세요")
print(response.text)
