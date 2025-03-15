from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from  langchain_anthropic import ChatAnthropic


load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo" )
# llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022") # Opcja dla Clode Ai