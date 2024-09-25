import os
from uuid import uuid4

from langchain_community.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_community.chat_models.azure_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from Extract_objects import Entities, Relations
from prompts import ENTITY_PROMPT, RELATION_PROMPT, CYPHER_PROMPT, QA_PROMPT

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f" [Text2Cypher] Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_API_KEY"] = "ls__5aea581c3a214e19ad211a4e8b1986a7"

DEFAULT_CHAT_MODEL = "gpt-4o-2024-05-13"

question = "黄蓉的丈夫生于何处？"

llm = AzureChatOpenAI(
    azure_endpoint = "https://myembedding.openai.azure.com/",
    api_key="5e0b3816cabf4db69dbcbed119d672ed",
    api_version="2024-06-01",
    azure_deployment="gpt4o"
)

URL = "bolt://10.230.34.226:7687"
DATABASE = "jinyong"
graph = Neo4jGraph(
    url=URL,
    database=DATABASE,
    username="neo4j",
    password="flx020709",
    enhanced_schema=True
)
relations = graph.query("CALL db.relationshipTypes")
types = []
for relation_type in relations:
    types.append(relation_type["relationshipType"])
schema = graph.get_structured_schema
# Cypher validation tool for relationship directions
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in schema.get("relationships")
]
cypher_validation = CypherQueryCorrector(corrector_schema)

entity_parser = PydanticOutputParser(pydantic_object=Entities)
relation_parser = PydanticOutputParser(pydantic_object=Relations)

entity_prompt = ENTITY_PROMPT.partial(format_instructions=entity_parser.get_format_instructions())
relation_prompt = RELATION_PROMPT.partial(format_instructions=relation_parser.get_format_instructions())
cypher_prompt = CYPHER_PROMPT
qa_prompt = QA_PROMPT

entity_chain = entity_prompt | llm | entity_parser
relation_chain = relation_prompt | llm | relation_parser
cypher_chain = (
        RunnablePassthrough.assign(names=entity_chain) |
        RunnablePassthrough.assign(
            entities=lambda x: x["names"]) |
        RunnablePassthrough.assign(names=relation_chain) |
        RunnablePassthrough.assign(
            relations=lambda x: x["names"]) |
        cypher_prompt | llm | StrOutputParser()
)
qa_chain = RunnablePassthrough.assign(query=cypher_chain) | RunnablePassthrough.assign(
        list=lambda x: graph.query(cypher_validation(x["query"])),
    ) | qa_prompt | llm | StrOutputParser()

response = qa_chain.invoke({
    "question": question,
    "schema": schema,
    "nodes": schema.get("node_props"),
    "types": types
})
print(response)
# TODO 用elasticsearch完成前两步实体提取
pass