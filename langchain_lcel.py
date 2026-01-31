from dotenv import load_dotenv
load_dotenv()

import os
print(os.getenv("LANGCHAIN_TRACING_V2"))

from langchain_ollama import ChatOllama

qwen = ChatOllama(model="qwen3:4b",temperature=0.3)
dsr1 = ChatOllama(model="deepseek-r1:14b",temperature=0.3)
# response = qwen.invoke("탄소의 원자 번호는?")
# print(response)
# print(response.content)
# print(response.response_metadata)

# from langchain_core.prompts import PromptTemplate

# template = """
# 당신은 {topic}분의의 전문가 입니다.{topic}에 관한 다음 질문에 답변해 주세요.
# 질문:{question}
# 답변:
# """
# prompt = PromptTemplate.from_template(template)

# formated_prompt = prompt.format(topic="양자역학",question=" 숨은 변수 이론을 알려주세요" )

# print("템플릿 변수:")
# print(f"-필수변수:{ prompt.input_variables}")
# print("\n생성된 프롬프트")
# print(formated_prompt)

# qwen_chain = prompt | qwen
# dsr1_chain = prompt | dsr1

# qwen_response = qwen_chain.invoke(
#     {"topic":"양자역학",
#     "question":"숨은 변수 이론을 알려주세요"}
# )

# print(qwen_response)

# dsr1_response = dsr1_chain.invoke(
#     {"topic":"양자역학",
#     "question":"숨은 변수이론을 알려주세요"}
# )

# print(dsr1_response)    

# from langchain_core.output_parsers import StrOutputParser

# output_parser =StrOutputParser()

# print(output_parser.invoke(response))

# chain = prompt | qwen | output_parser
# response = chain.invoke({"topic":"양자역학","question":"숨은 변수 이론을 알려주세요"})
#print(response)

#print(chain.input_schema.model_json_schema())

# from langchain_core.runnables import RunnableSequence
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama

# prompt = PromptTemplate.from_template("'{text}를 영어로 번역해 주세요.번역된 문장만을 출력해주세요")
# translator = ChatOllama(model="qwen3:4b",temperature=0.3)
# output_parser = StrOutputParser()

# translation_chain = RunnableSequence(
#     first = prompt,
#     middle = [translator],
#     last = output_parser
#     )

# translation_chain = prompt | translator | output_parser

# result = translation_chain.invoke({"text":"안녕하세요"})
# print(result)

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# output_parser = StrOutputParser()

# question_template = """
# 다음 카테고리 중 하나로 입력을 분류하세요.:
# -화확(Chemistry)
# -물리학(Physics)
# -생물학(Biology)
#  예시:
# 입력: 인간의 염색체는 모두 몇개인가요?
# 답변:생물학(Biology)

# 입력:{question}
# 답변:

# """
# question_prompt =ChatPromptTemplate.from_template(question_template)

# question_chain = question_prompt | qwen | output_parser

# result =question_chain.invoke({"question":"탄소의 원자번호는  몇번인가요?"})
# print(result)

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# output_parser = StrOutputParser()

# language_template = """
# 입력된 텍스트의 언어를 다음 카테고리 중 하나로 분류해 주세요.
# -영어(English)
# -한국어(Korean)
# -기타(Others)

# #예시:
# 입력: How many protons are in a carbon atom?
# 답변:영어(English)

# 입력: {question}
# 답변:
# """

# language_prompt = ChatPromptTemplate.from_template(language_template)

# language_chain = language_prompt | qwen | output_parser

# examples =[
#     "What is the atomic number of carbon?",
#     "탄소의 원자 번호는 몇번인가요?",
#     "¿Cuál es el número atómico del carbono?"
# ]

# for example in examples:
#     result = language_chain.invoke({"question":example})
#     print(f"입력: {example}")
#     print(f"분류 결과: {result}\n")

# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
# from langchain_core.output_parsers import StrOutputParser
# from operator import itemgetter

# answer_template ="""
# 당신은 {topic} 분야의 전문가입니다. {topic}에 관한 질문에 {language}로 답변해 주세요.
# 질문: {question}
# 답변:
# """

# answer_prompt = ChatPromptTemplate.from_template(answer_template)
# output_parser = StrOutputParser()

# answer_chain = RunnableParallel(
#   { 
#     "topic": question_chain,
#     "language": language_chain,
#     "question":itemgetter("question")
#   } 
# ) |answer_prompt |qwen | output_parser

# result = answer_chain.invoke({"question":"탄소의 원자번호는 몇번인가요?"})
# print("처리결과")
# print(f'답변: {result}')

# test_questions = [
#     "물의 분자식은 무엇인가요?",
#     "What is Newton's first law of motion",
#     "세포의 기본구조를 설명해 주세요"
# ]

# for question in test_questions:
#     print(f'\n질문:{question}')
#     result = answer_chain.invoke({"question":question})
#     print(f"답변: {result}")

from langchain_core.runnables import RunnablePassthrough
import re

runnable = RunnableParallel(
    passed = RunnablePassthrough(),
    modified =lambda x: int(re.search(r'\d+', x["query"]).group()),
)

#result = runnable.invoke({"query":"탄소의 원자번호는 6번입니다."})
#print(result)

runnable = RunnableParallel(
    passed = RunnablePassthrough(),
    modified = lambda x: int(re.search(r'\d+', x).group()),
)

rusult = runnable.invoke("탄소의 원자번호는 6번입니다.")
#print(rusult)

# from langchain_core.runnables import RunnableLambda,RunnablePassthrough

# def extract_number(query):
#     return int(re.search(r'\d+', query).group())

# runnable = RunnablePassthrough() | RunnableLambda(extract_number)

# result = runnable.invoke("탄소의 원자번호는 6번입니다.")
# print(result)

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

def preprocess_text(text:str) -> str:
    """
    입력 텍스트를 소문자로 변환하고 양쪽 공백을 제거합니다.
    """
    return text.lower().strip()

def postprocess_response(response:str) -> dict:
    """
    응답 텍스트를 대문자로 변환하고 길이를 계산합니다
    """
    response_text = response.content
    return {
        "processed_response": response_text.upper(),
        "length": len(response_text)
    }

prompt = ChatPromptTemplate.from_template("다음 주제애 대해 영어 한 문장으로 설명해주세요: {topic}")

chain = (
    RunnableLambda(preprocess_text) | 
    prompt |
    qwen |
    RunnableLambda(postprocess_response)
) 
result = chain.invoke("인공지능")
print(f"처리된 응답:{result['processed_response']}")
print(f"응답 길이:{result['length']}")








