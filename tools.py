from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime


def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_data = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_data)

    return f"Data succesfully saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves the output to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Searchesthe web fro information.",
)


api_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=256)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
