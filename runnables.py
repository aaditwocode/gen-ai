from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)


load_dotenv()


# ---------------- MODEL ----------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


parser = StrOutputParser()


# ---------------- PROMPTS ----------------

short_prompt = ChatPromptTemplate.from_template(
    """
    Give a short explanation of {topic}
    """
)


long_prompt = ChatPromptTemplate.from_template(
    """
    Give a detailed explanation of {topic}
    """
)


code_prompt = ChatPromptTemplate.from_template(
    """
    Write python code for {topic}
    """
)


explain_prompt = ChatPromptTemplate.from_template(
    """
    Explain this code:

    {Code}
    """
)


# =================================================
# 1. RUNNABLE SEQUENCE
# Prompt → Model → Parser
# =================================================

sequence_chain = (
    short_prompt
    | model
    | parser
)


print("\n--- SEQUENCE ---")

result = sequence_chain.invoke(
    {
        "topic":"Machine Learning"
    }
)

print(result)



# =================================================
# 2. RUNNABLE LAMBDA
# Custom function inside chain
# =================================================


uppercase = RunnableLambda(
    lambda x: x.upper()
)


lambda_chain = (
    short_prompt
    | model
    | parser
    | uppercase
)


print("\n--- LAMBDA ---")

print(
    lambda_chain.invoke(
        {
        "topic":"AI"
        }
    )
)



# =================================================
# 3. RUNNABLE PARALLEL
# Run multiple chains together
# =================================================


parallel_chain = RunnableParallel(

    short =
        short_prompt
        | model
        | parser,


    long =
        long_prompt
        | model
        | parser

)


print("\n--- PARALLEL ---")


result = parallel_chain.invoke(
    {
        "topic":"Deep Learning"
    }
)


print(result)



# =================================================
# 4. RUNNABLE PASSTHROUGH
# Save intermediate result
# =================================================


code_chain = (
    code_prompt
    | model
    | parser
)



final_chain = (
    code_chain
    |
    RunnableParallel(

        Code = RunnablePassthrough(),


        Explain =
        explain_prompt
        | model
        | parser

    )
)



print("\n--- PASSTHROUGH ---")


result = final_chain.invoke(
    {
        "topic":"Binary Search"
    }
)


print(result)



# =================================================
# 5. BATCH
# Multiple inputs
# =================================================


print("\n--- BATCH ---")


result = sequence_chain.batch(
    [
        {"topic":"AI"},
        {"topic":"Blockchain"},
        {"topic":"Cloud"}
    ]
)


for r in result:
    print(r)



# =================================================
# 6. STREAM
# Get response token by token
# =================================================


print("\n--- STREAM ---")


for chunk in sequence_chain.stream(
    {
        "topic":"Generative AI"
    }
):
    print(chunk, end="")