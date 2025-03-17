from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from .tools import search_tool, wiki_tool, save_tool


load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


llm = ChatOpenAI(model="gpt-3.5-turbo")  # Opcja dla Chatgpt
# llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022") # Opcja dla Clode Ai
parser = PydanticOutputParser(ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a reasearch assistant that wil help generate a research paper.
            Answer the query and use neccessary tools.
            Wrap the output in this format and provide on othre text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can  help you research?")
raw_response = agent_executor.invoke({"query": query})


structured_response = parser.parse(raw_response.get("output")[0]["text"])
print(structured_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)

# response = llm2.invoke("What is the meaning of life?") #test question for GPT to see if APi works correctly
# print(response)
# response = llm2.invoke("What is the meaning of life?") #test question for Cloude to see if APi works correctly
# print(response)
