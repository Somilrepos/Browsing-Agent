
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils import annotate, format_descriptions, parse, log_stage
from config import config

def load_prompt():
    prompt = hub.pull(config.get("prompt", "browser_agent_prompt"))
    return prompt

def create_agent(verbose=True):
    """Create the agent chain"""
    # Load the prompt
    prompt = load_prompt()
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model=config.get('agent', 'model'), 
        convert_system_message_to_human=True, 
        temperature=config.get('agent', 'temperature'), 
        top_p=config.get('agent', 'top_p')
    )
    
    # Create the agent chain
    agent = (
        annotate
        | RunnablePassthrough.assign(
            prediction=(
                format_descriptions
                | prompt
                | llm
                | StrOutputParser()
                | parse
            )
        )
    )
    
    if verbose:
        return agent.with_config(verbose=True)
    return agent