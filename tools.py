from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import re


def save_to_txt(data: str, filename: str = None):
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = re.sub(
            r'[^\w\s]', '', data[:20]
        ).strip().replace(' ', '_')
        filename = (
            f"research_{timestamp}_"
            f"{safe_topic}.txt"
        )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = (
        "--- Research Output ---\n"
        f"Timestamp: {timestamp}\n\n"
        f"{data}\n\n"
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data saved to {filename}"


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description=(
        "Saves structured research data to a text file. "
        "Creates a new file each time."
    ),
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
