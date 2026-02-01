import common_utils_solar as utils
from langchain.chat_models import init_chat_model
#model = get_solar_model(model_name="solar-pro")
model = init_chat_model("solar-pro")
if __name__ =="__main__":
    response = model.invoke("현재 한국의 날씨는 어때?")
    print(response)


