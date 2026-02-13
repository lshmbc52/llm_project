import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import common_utils as utils
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

llm = utils.get_solar_model(model_name="solar-pro")
load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=[
        "\n\n",  # 빈 줄
        "\n",  # 줄바꿈
        " ",  # 공백
        ".",  # 마침표
        ",",  # 쉼표
        "\u200b",  # 폭 없는 공백
        "\uff0c",  # 전각 쉼표
        "\u3001",  # 한중일 쉼표
        "\uff0e",  # 전각 마침표
        "\u3002",  # 한중일 마침표
        "",
    ],
)

loader = Docx2txtLoader("/home/sh/Downloads/law_markdown.docx")
document_list = loader.load_and_split(text_splitter=text_splitter)

embedding = OpenAIEmbeddings(
    base_url="https://api.upstage.ai/v1/solar",
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model="embedding-passage",
    check_embedding_ctx_length=False,
)

vector_store = PineconeVectorStore.from_documents(
    document_list,
    embedding,
    index_name="house-tax-index",
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print(retriever)
