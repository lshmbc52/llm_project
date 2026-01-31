import langchain
#print(langchain.__version__)
import os
os.environ["USER_AGENT"] = "langchain_rag.py"

from langchain_community.document_loaders import WebBaseLoader
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)
docs = loader.load()
# print(len(docs))
# print(len(docs[0].page_content))
# print(docs[0].page_content[5000:6000])

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 200)
splits = text_splitter.split_documents(docs)

# print(len(splits))
# print(splits[10].page_content)
#print(splits[10].metadata)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents = splits, embedding=OpenAIEmbeddings())
docs = vectorstore.similarity_search("격하과정에 대해서 설명해 주세요")
# print(len(docs))
# print(docs[0].page_content)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = '''Answer the Question based only on the following context:{context}
Question:{question}
'''
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model='gpt-3.5-turbo-0125',temperature= 0)

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

rag_chain =(
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    |prompt
    |model
    |StrOutputParser()
)

print(rag_chain.invoke("격하과정에 대해서 설명해 주세요"))




