import common_utils_solar as utils
from langchain.messages import HumanMessage, AIMessage, SystemMessage
model = utils.get_solar_model(model_name="solar-pro")

#sys_msg = SystemMessage("당신은 사용자와의 대화내용을 기억하고 친절하게 대답하는 유능한 비서야")
# human_msg =HumanMessage("궁금한게 있어")

# hm_1 = HumanMessage("안녕하세요.저는 sh입니다")
# aim_1 = AIMessage("안녕,sh.어떤 도움이 필요하나요?")
# hm_2 = HumanMessage("방금 대화에서 알려준 내이름이 뭐라고 했나요?")
# messages =[sys_msg,hm_1,aim_1,hm_2 ]

messages =[
    {"role":"system","content":"당신은 유능한 비서야"},
    {"role":"user","content":"안녕하세요, 저는 sh 입니다"},
    {"role":"ai","content":"어떤 도움이 필요한가요?"},
    {"role":"user","content":" 방금 대화에서 알려준 사용자의 이름이 뭐라고 했죠?"},
    ]
if __name__ =="__main__":
    response = model.invoke(messages)
    print(response.content)









