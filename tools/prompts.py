from langchain_core.prompts import ChatPromptTemplate

ENTITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个实体提取（Entity Linking）专家，正在提取文本中的Neo4j实体（Entity）\n{format_instructions}",
        ),
        (
            "human",
            "请你根据下面的Neo4j的节点属性，使用给定的格式从用户的输入文本中提取所有可能的实体（Entity）:"
            "\n{nodes} \n用户输入: {question}",
        ),
    ]
)

RELATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个知识图谱关系提取（Entity Linking）专家，正在提取文本中的Neo4j关系类型（Relationship types）\n{format_instructions}",
        ),
        (
            "human",
            "请你根据下面的Neo4j的关系类型列表，使用给定的格式从用户的输入文本中提取所有可能的关系类型，请尽可能多的提取，你认为有相近含义的也算:"
            "\n{types} \n用户输入: {question}",
        ),
    ]
)

CYPHER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定一个输入问题，将其转换为 Cypher 查询。请务必只返回Cypher的内容本身，不要用```cypher等回复！",
        ),
        (
            "human",
            "根据下面的 Neo4j graph schema，编写一个 Cypher 查询来回答用户的问题： "
            "\n{schema} \n问题中包含的实体：{entities} \n问题中包含的关系类型：{relations} \n问题: {question} \nCypher 查询:"
            ,
        ),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "给定一个输入问题，以及Cypher查询结果列表，将其转换为一个自然语言的回答，不要有前言",
        ),
        (
            "human",
            "根据下面的Cypher查询和Cypher查询结果列表，来回答用户的问题： "
            "Cypher查询：{query} \n查询结果列表：{list} \n问题: {question} \n你的回答:"
            ,
        ),
    ]
)

FIX_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个来自Neo4j公司的专家，你十分熟悉Cypher语法，现在你需要给你的新同事修改他写的Cypher语句，也就是给定一个出错的Cypher语句，"
            "你需要将其修正为一个可以运行的Cypher语句",
        ),
        (
            "human",
            "请根据报错信息纠正以下Cypher语句:"
            "```出错的Cypher语句:\n{cypher}\n```\n报错信息: {error}"
            "你的回复不应该有多余叙述, 请务必只返回修正后的Cypher语句本身，且不要用```cypher等回复！"
            ,
        ),
    ]
)