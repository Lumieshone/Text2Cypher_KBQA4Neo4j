import os
from operator import itemgetter

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.memory import ConversationBufferMemory
from langchain_elasticsearch import ElasticsearchStore
from elastic_search_bm25 import ElasticSearchBM25Retriever

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
from uuid import uuid4

from langchain_community.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from Extract_objects import Entities, Relations
from prompts import ENTITY_PROMPT, RELATION_PROMPT, CYPHER_PROMPT, QA_PROMPT, FIX_PROMPT
from elasticsearch import Elasticsearch

# 初始化 Elasticsearch 客户端
es_client = Elasticsearch("http://localhost:9200")
# 初始状态下不连接数据库
graph = None
relations = None
types = None
schema = None
entity_names = None
corrector_schema = None
cypher_validation = None
vectorstore = None
entity_vector_retriever = None
relationship_vector_retriever = None
cypher_chain = None
qa_chain = None
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f" [Text2Cypher] Tracing Walkthrough - {unique_id}"
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_API_KEY"] = os.getenv("MY_LANGCHAIN_API_KEY")
DEFAULT_CHAT_MODEL = "gpt-4o-2024-05-13"
# 嵌入模型
embeddings = AzureOpenAIEmbeddings(
    model="embedding-ada02",
    azure_endpoint=os.getenv("MY_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("MY_AZURE_OPENAI_API_KEY"),
    api_version="2024-06-01",
)

# Initialize FastAPI
app = FastAPI()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("MY_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("MY_AZURE_OPENAI_API_KEY"),
    api_version="2024-06-01",
    azure_deployment="gpt4o",
    temperature=0
)

URL = "bolt://localhost:7687"
DATABASE = "jinyong"
USERNAME = "neo4j"
PASSWORD = "neo4j"

entity_parser = PydanticOutputParser(pydantic_object=Entities)
relation_parser = PydanticOutputParser(pydantic_object=Relations)
entity_filter_criteria = [{"term": {"metadata.type": 'node_props'}}]
relationship_filter_criteria = [{"term": {"metadata.type": 'relationships'}}]
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
    r = qa_chain.invoke({"question": que, "schema": schema, "nodes": entity_names, "types": types})
    print(r)
    return r


def chat_response(input_text, history):
    if graph is None:
        return "Please connect to the database first."
    return chat(input_text)


def on_connect(neo4j_url, neo4j_database, neo4j_username, neo4j_password):
    global graph, relations, types, schema, entity_names, vectorstore, corrector_schema, cypher_validation, \
        entity_vector_retriever, relationship_vector_retriever, cypher_chain, qa_chain
    try:
        graph = create_graph(neo4j_url, neo4j_database, neo4j_username, neo4j_password)
        graph.query("MATCH (n) RETURN n LIMIT 1")
    except Exception as e:
        return f"Connection failed: {str(e)}"
    relations = graph.query("CALL db.relationshipTypes")
    types = [relation_type["relationshipType"] for relation_type in relations]
    schema = graph.get_structured_schema
    prop_query = """
    MATCH (n)
    RETURN properties(n) AS properties;
    """
    properties = [x["properties"] for x in graph.query(prop_query)]
    first_key = list(properties[0].keys())[0]
    entity_names = [x[first_key] for x in properties]
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in schema.get("relationships")
    ]
    cypher_validation = CypherQueryCorrector(corrector_schema)
    # 构建索引名称
    index_name = f"index_{neo4j_database}_vectors"
    vectorstore = ElasticsearchStore(
        es_url="http://localhost:9200",
        index_name=index_name,
        embedding=embeddings,
        vector_query_field='question_vectors'
    )
    entity_vector_retriever = ElasticSearchBM25Retriever(client=vectorstore.client, index_name=index_name,
                                                         data_type="entity_names", k=3, score_threshold=0.72)
    relationship_vector_retriever = ElasticSearchBM25Retriever(client=vectorstore.client, index_name=index_name,
                                                               data_type="relationships", k=3, score_threshold=0.72)

    # 检查索引是否存在
    if not es_client.indices.exists(index=index_name):
        # 嵌入字符串列表 node_props 并添加到索引
        if entity_names:
            metadata_nodes = [{"type": "entity_names"} for _ in entity_names]
            vectorstore.add_texts(texts=entity_names, metadatas=metadata_nodes)
        # 嵌入字符串列表 types 并添加到索引
        if types:
            metadata_relationships = [{"type": "relationships"} for _ in types]
            vectorstore.add_texts(texts=types, metadatas=metadata_relationships)
        print(f"建立了新索引{index_name}")
    else:
        # 索引不为空，不做操作
        print(f"Index {index_name} already exists and is not empty.")
    cypher_chain = (
            {
                "entities": itemgetter("question") | entity_vector_retriever,
                "relations": itemgetter("question") | relationship_vector_retriever,
                "question": itemgetter("question"),
                "schema": itemgetter("schema")
            } |
            cypher_prompt | llm | StrOutputParser()
    )
    qa_chain = RunnablePassthrough.assign(query=cypher_chain) | RunnablePassthrough.assign(
        list=lambda x: execute_cypher(x["query"]),
    ) | qa_prompt | llm | StrOutputParser()
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
    return f"Query failed: {str(e)}"


# entity_chain = entity_prompt | llm | entity_parser
# relation_chain = relation_prompt | llm | relation_parser


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
                                  examples=["黄蓉的丈夫在哪出生？",
                                            "黄蓉父亲的徒弟都有谁？",
                                            "会降龙十八掌的有哪些人？",
                                            "谁参与了第三次华山论剑？",
                                            "洪七公和郭襄的父亲分别会什么武功？",
                                            "李萍她老公的儿媳妇是谁？",
                                            "郭靖是如何称呼黄药师的女儿的？"
                                            ])

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
