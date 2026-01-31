import os
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")
result = llm.invoke("지구의 자전주기는?")
print(result.content)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question:{input}")
#print(prompt)

chain = prompt | llm
#print(chain.invoke({"input":"지구의 자전주기는?"}))

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
#print(chain.invoke({"input":"지구의 자전주기는?"}))

prompt1 = ChatPromptTemplate.from_template("translate {korean_word} to English")
prompt2 = ChatPromptTemplate.from_template("explain {english_word} using oxford dictionry to me in Korean")

chain1 = prompt1 | llm | output_parser
#print(chain1.invoke({"korean_word":"미래"}))

chain2 = (
    {"english_word":chain1}
    |
    prompt2
    | llm
    | output_parser
)
#print(chain2.invoke({"korean_word":"미래"}))

from langchain_core.prompts import PromptTemplate   
template_text = "안녕하세요? 제이름은 {name}이고, 나이는 {age}살입니다."
prompt_template = PromptTemplate.from_template(template_text)
filled_prompt = prompt_template.format(name="홍길동", age=30)
print(filled_prompt)

combined_prompt = (
    prompt_template
    + PromptTemplate.from_template("\n\n 아버지를 아버지라 부를 수 없습니다")
    +"\n\n{language}로 번역해 주세요" 
)

#print(combined_prompt)

chain = combined_prompt | llm | output_parser
#print(chain.invoke({"name":"홍길동", "age":30, "language":"English"}))

chat_prompt = ChatPromptTemplate.from_messages([
    ("system","이시스템은 천문학 질문에 답변할 수 있습니다."),
    ("user""{user_input}"),
])

messages =chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은")
print(messages)

chain =chat_prompt | llm | output_parser
#print(chain.invoke({"user_input":"태양계에서 지구와 가장 먼 행성은?"}))

from langchain_core.prompts import SystemMessagePromptTemplate,HumanMessagePromptTemplate
ChatPromptTemplate.from_messages([

    SystemMessagePromptTemplate.from_template("이시스템은 천문학 질문에 답변할 수 있습니다."),
    HumanMessagePromptTemplate.from_template("{user_input}"),
])

messages =chat_prompt.format_messages(user_input ="태양계에서 가장 큰 행성은?")
#print(messages)

chain = chat_prompt | llm | output_parser
print(chain.invoke({"user_input":"태양계에서 지구와 가장 먼 행성은? "}))

# Model Parameter
# llm
from langchain_openai import OpenAI
llm = OpenAI()
print(llm.invoke("한국의 대표적인 관광지 3곳을 알려줘"))

#ChatModel

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat = OpenAI()

chat_prompt = ChatPromptTemplate.from_messages(
    [('system', "이시스템은 여행전문가입니다"),
    ('user',"{user_input}"),
    ])

chain = chat_prompt | chat 

print(chain.invoke({'user_input':'한국의 대표적인 관광지 3곳을 알려줘'}))


from langchain_openai import ChatOpenAI
params ={
    "temperature":0.7,
    "max_tokens":100,}

kwargs = {
        "frequency_penalty":0.5,
        "presence_penalty":0.5,
        "stop":["\n"] }

model = ChatOpenAI(model = 'gpt-3.5-turbo-0125', **params, model_kwargs=kwargs)
question = "태양계에서 가장 큰 행성은?"
response = model.invoke(input= question)
print(response)

params = {
    'temperature':0.7,
    'max_tokens':10,
}

response =model.invoke(input = question, **params)
print("--------------------")
print(response.content)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ('system', '이시스템은 천문학 질문에 답변할 수 있습니다.'),
    ('user', '{user_input}'),
])

model = ChatOpenAI(model='gpt-3.5-turbo-0125', max_tokens=100)
messages = prompt.format_messages(user_input='안드레메다성운은 지구에서 얼마나 떨어져 있나요?')
before_answer = model.invoke(messages)
print(before_answer.content)

chain = prompt | model.bind(max_tokens=20)
after_answer = chain.invoke({'user_input':'안드레메다성운은 지구에서 얼마나 떨어져 있나요?'})
print(after_answer.content)

# output parser

from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template = "List five {subject} \n{format_instructions}",
    input_variables =['subject'],
    partial_variables = {'format_instructions':format_instructions},
)

llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)
chain = prompt | llm | output_parser

print(chain.invoke({'subject':'popular Korean movies'}))

#json parser

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from pydantic import Field

class CusineRecipe(BaseModel):
    name: str = Field(description = "name of a cusine")
    recipe: str = Field(description = "recipe of a cusine")

output_parser = JsonOutputParser(pydantic_object = CusineRecipe)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

prompt = PromptTemplate(
   template ="Answer the user query. \n{format_instructions}\n{query}\n",
   input_variables =['query'],
   partial_variables ={'format_instructions': format_instructions},
)

print(prompt)

chain = prompt | llm | output_parser
print(chain.invoke({'query':'Let me know how to cook Bibimbap'}))













