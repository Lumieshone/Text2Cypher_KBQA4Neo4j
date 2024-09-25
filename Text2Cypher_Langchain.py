from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os
from uuid import uuid4
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f" [Text2Cypher] Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_API_KEY"] = "ls__5aea581c3a214e19ad211a4e8b1986a7"
URL = "bolt://10.230.34.215:7687"
DATABASE = "jinyong"

graph = Neo4jGraph(
    url=URL,
    database=DATABASE,
    username="neo4j",
    password="flx020709",
    enhanced_schema=True,
    # sanitize=True,
)
# DEMO_URL = "neo4j+s://demo.neo4jlabs.com"
# DATABASE = "recommendations"
#
# graph = Neo4jGraph(
#     url=DEMO_URL,
#     database=DATABASE,
#     username=DATABASE,
#     password=DATABASE,
#     enhanced_schema=True,
#     sanitize=True,
# )
llm = ChatOllama(model="Lumie/llama3-chinese-text2cypher")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定一个输入问题，将其转换为 Cypher 查询。无需前言。",
        ),
        (
            "human",
            (
                "根据下面的 Neo4j graph schema，编写一个 Cypher 查询来回答用户的问题： "
                "\n{schema} \n问题: {question} \nCypher 查询:"
            ),
        ),
    ]
)
chain = prompt | llm
# question = "How many movies did Tom Hanks play in?"
question = "黄蓉的丈夫的出生在哪？"
response = chain.invoke({"question": question, "schema": graph.schema})
print(response.content)
pass