import os

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
from uuid import uuid4

from langchain_community.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from Extract_objects import Entities, Relations
from prompts import ENTITY_PROMPT, RELATION_PROMPT, CYPHER_PROMPT, QA_PROMPT, FIX_PROMPT

# 初始状态下不连接数据库
graph = None
relations = None
types = None
schema = None
corrector_schema = None
cypher_validation = None

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f" [Text2Cypher] Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_API_KEY"] = "ls__5aea581c3a214e19ad211a4e8b1986a7"
DEFAULT_CHAT_MODEL = "gpt-4o-2024-05-13"

# Initialize FastAPI
app = FastAPI()

llm = AzureChatOpenAI(
    azure_endpoint="https://myembedding.openai.azure.com/",
    api_key="5e0b3816cabf4db69dbcbed119d672ed",
    api_version="2024-06-01",
    azure_deployment="gpt4o",
    temperature=0
)

URL = "bolt://localhost:7687"
DATABASE = "jinyong"
USERNAME = "neo4j"
PASSWORD = "flx020709"

entity_parser = PydanticOutputParser(pydantic_object=Entities)
relation_parser = PydanticOutputParser(pydantic_object=Relations)

entity_prompt = ENTITY_PROMPT.partial(format_instructions=entity_parser.get_format_instructions())
relation_prompt = RELATION_PROMPT.partial(format_instructions=relation_parser.get_format_instructions())
cypher_prompt = CYPHER_PROMPT
qa_prompt = QA_PROMPT
fix_prompt = FIX_PROMPT

def create_graph(url, database, username, password):
    return Neo4jGraph(
        url=url,
        database=database,
        username=username,
        password=password,
        enhanced_schema=True
    )


def chat(que):
    r = qa_chain.invoke({"question": que, "schema": schema, "nodes": schema.get("node_props"), "types": types})
    print(r)
    return r


def chat_response(input_text, history):
    if graph is None:
        return "Please connect to the database first."
    return chat(input_text)


def on_connect(neo4j_url, neo4j_database, neo4j_username, neo4j_password):
    global graph, relations, types, schema, corrector_schema, cypher_validation
    try:
        graph = create_graph(neo4j_url, neo4j_database, neo4j_username, neo4j_password)
        graph.query("MATCH (n) RETURN n LIMIT 1")
    except Exception as e:
        return f"Connection failed: {str(e)}"
    relations = graph.query("CALL db.relationshipTypes")
    types = [relation_type["relationshipType"] for relation_type in relations]
    schema = graph.get_structured_schema
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in schema.get("relationships")
    ]
    cypher_validation = CypherQueryCorrector(corrector_schema)
    go_to_chat_button = gr.Button(value="进入聊天页面", interactive=True, link="/chat")
    return "Connection successful! You can now enter the chat page.", go_to_chat_button


def execute_cypher(cypher):
    print(f"Original Cypher: {cypher}")
    corrected_cypher = cypher_validation(cypher)
    print(f"Corrected Cypher: {corrected_cypher}")
    fixing_chain = (
            fix_prompt
            | llm
            | StrOutputParser()
    )
    # 我们尝试3次
    for _ in range(3):
        try:
            res = graph.query(corrected_cypher)
            print(f"Query Result: {res}")
            return res
        except Exception as e:
            # 出现异常后，用chain出来，尝试矫正这个cypher
            print(f"Query failed: {str(e)}")
            corrected_cypher = fixing_chain.invoke({"cypher": corrected_cypher, "error": e})
            print("LLM fix = {}".format(corrected_cypher))
    return None


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
    list=lambda x: execute_cypher(x["query"]),
) | qa_prompt | llm | StrOutputParser()

neo4j_url_input = gr.Textbox(label="Neo4j URL", placeholder="请输入 Neo4j URL", value=URL)
neo4j_database_input = gr.Textbox(label="Neo4j Database", placeholder="请输入 Neo4j Database", value=DATABASE)
neo4j_username_input = gr.Textbox(label="Neo4j Username", placeholder="请输入 Neo4j Username", value=USERNAME)
neo4j_password_input = gr.Textbox(label="Neo4j Password", placeholder="请输入 Neo4j Password", value=PASSWORD,
                                  type="password")

connect_interface = gr.Interface(fn=on_connect,
                                 inputs=[neo4j_url_input, neo4j_database_input, neo4j_username_input,
                                         neo4j_password_input],
                                 allow_flagging="never",
                                 outputs=[gr.Textbox(label="连接状态"),
                                          gr.Button(value="进入聊天页面",
                                                    interactive=False,
                                                    link="/chat")],
                                 clear_btn=None,
                                 submit_btn="测试连接",
                                 title="Test Neo4j Connection",
                                 description="Click the button to test your Neo4j connection.")
chat_interface = gr.ChatInterface(fn=chat_response,
                                  title="KBQA Chatbot",
                                  description="powered by Neo4j",
                                  theme="soft",
                                  chatbot=gr.Chatbot(height=500),
                                  undo_btn=None,
                                  retry_btn="🔄 重试",
                                  clear_btn="\U0001F5D1 清除对话",
                                  examples=["黄蓉的丈夫的在哪出生？",
                                            "黄蓉父亲的徒弟都有谁？",
                                            "会降龙十八掌的有哪些人？",
                                            "谁参与了第三次华山论剑？",
                                            "洪七公和郭襄的父亲分别会什么武功？",
                                            "李萍她老公的儿媳妇是谁？",
                                            "郭靖是如何称呼黄药师的女儿的？"
                                            ])

# interface.launch(share=True)
# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, connect_interface, path="/connect")
app = gr.mount_gradio_app(app, chat_interface, path="/chat")


@app.get("/")
async def redirect_to_connect():
    return RedirectResponse(url="/connect")


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
# TODO 用elasticsearch完成前两步实体提取
