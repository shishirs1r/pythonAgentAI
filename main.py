from datetime import datetime
from dotenv import load_dotenv  
load_dotenv()

import os  
import re
import json
from pydantic import BaseModel, ValidationError
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import PydanticOutputParser  
from langchain_core.messages import HumanMessage, SystemMessage  
from tools import search_tool, wiki_tool, save_tool  

class ResearchAnswer(BaseModel):
    topic: str  
    summary: str  
    sources: list[str] = [] 
    tools_used: list[str] = [] 

assistant = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),  
        temperature=0.2,  
        max_new_tokens=2000  
    )
)

answer_formatter = PydanticOutputParser(pydantic_object=ResearchAnswer)

chat_guide = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You're a helpful assistant that answers questions! You can:
            - Use tools like Wikipedia and web search
            - Use your own knowledge when appropriate
            - Always list tools used in 'tools_used'
            - List sources in 'sources' if available
            - Keep answers clear and friendly!
            - IMPORTANT: Your response MUST be in valid JSON format with all fields completed
            
            {format_instructions}
            """
        ),
        ("human", """
        Question: {query}
        Tool Results: {tool_results}
        """)
    ]
).partial(format_instructions=answer_formatter.get_format_instructions())

def gather_information(question):
    collected_info = ""

    try:
        wiki_info = wiki_tool.run(question)
        collected_info += f"Wikipedia says: {wiki_info}\n"
    except Exception as error:
        collected_info += f"Wikipedia didn't work: {str(error)}\n"
    
    try:
        search_info = search_tool.run(question)
        collected_info += f"Web search says: {search_info}\n"
    except Exception as error:
        collected_info += f"Web search didn't work: {str(error)}\n"
    
    return collected_info

def safe_parse_response(response_content):
    try:
        json_match = re.search(r'\{[\s\S]*\}', response_content)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            if "topic" not in data:
                data["topic"] = "Research Results"
            if "summary" not in data:
                data["summary"] = response_content[:500] if response_content else "Summary not available"
            if "sources" not in data:
                data["sources"] = []
            if "tools_used" not in data:
                data["tools_used"] = []
                
            return ResearchAnswer(**data)
    except (json.JSONDecodeError, TypeError, ValidationError) as e:
        pass
  
    return ResearchAnswer(
        topic="Research Results",
        summary=response_content,
        sources=[],
        tools_used=[]
    )

def answer_question(question):
    tool_info = gather_information(question)
    
    system_instructions = chat_guide.messages[0].prompt.format(
        format_instructions=answer_formatter.get_format_instructions()
    )
    
    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content=f"Question: {question}\nTool Results: {tool_info}")
    ]

    response = assistant.invoke(messages)
    
    final_answer = safe_parse_response(response.content)
    
    return final_answer

print("Welcome to the Research Assistant! Type 'exit' at any time to quit.")

while True:
    user_question = input("\nHi! What would you like to know today? ").strip()
    
    if user_question.lower() in ["exit", "quit", "bye"]:
        print("Goodbye! Thanks for using the Research Assistant.")
        break
        
    if not user_question:
        print("Please enter a question or type 'exit' to quit.")
        continue
        
    result = answer_question(user_question)

    print("\nHere's what I found:")
    print(f"Topic: {result.topic}")
    print(f"Summary: {result.summary}")
    
    if result.sources:
        print("\nSources:")
        for source in result.sources:
            print(f"- {source}")
    
    if result.tools_used:
        print(f"\nTools used: {', '.join(result.tools_used)}")
    
    if "save to a file" in user_question.lower():
        safe_topic = re.sub(r'[^\w\s]', '', result.topic[:30]).strip().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{timestamp}_{safe_topic}.txt"
        save_result = save_tool.run(str(result), filename)
        print(f"\n {save_result}")