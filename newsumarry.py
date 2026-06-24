from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------------
# Search Tool
# -----------------------------

search_tool = TavilySearchResults(
    max_results=3
)


# -----------------------------
# LLM
# -----------------------------

llm = ChatMistralAI(
    model="mistral-small-2506"
)


# -----------------------------
# Prompt
# -----------------------------

prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Summarize the following news into clear bullet points.

News:
{news}
"""
)


# -----------------------------
# Runnable Chain
# prompt -> llm -> parser
# -----------------------------

chain = (
    prompt
    | llm
    | StrOutputParser()
)


# -----------------------------
# Get latest news
# -----------------------------

news_result = search_tool.invoke(
    "Latest AI news of 2026"
)


# -----------------------------
# Run chain
# -----------------------------

result = chain.invoke(
    {
        "news": news_result
    }
)


print(result)