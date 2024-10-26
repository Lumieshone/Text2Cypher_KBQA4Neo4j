import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

MY_EMBEDDING = AzureOpenAIEmbeddings(
    model="embedding-ada02",
    azure_endpoint=os.getenv("MY_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("MY_AZURE_OPENAI_API_KEY"),
    api_version="2024-06-01",
)

MY_LLM = AzureChatOpenAI(
    azure_endpoint=os.getenv("MY_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("MY_AZURE_OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_deployment="gpt4o",
    temperature=0
)
ES_URL = "http://localhost:9200"
INIT_URL = "bolt://localhost:7687"
INIT_DATABASE = "jinyong"
INIT_USERNAME = "neo4j"
INIT_PASSWORD = "flx020709"

QUESTION_EXAMPLES = [
    "黄蓉的丈夫在哪出生？",
    "黄蓉父亲的徒弟都有谁？",
    "会降龙十八掌的有哪些人？",
    "谁参与了第三次华山论剑？",
    "洪七公和郭襄的父亲分别会什么武功？",
    "李萍她老公的儿媳妇是谁？",
    "郭靖是如何称呼黄药师的女儿的？"
]