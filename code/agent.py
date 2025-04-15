
from langchain import hub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils import annotate, format_descriptions, parse, log_stage
from config import config

def load_prompt(prompt_name:str):
    prompt = hub.pull(config.get("prompt", prompt_name))
    return prompt

def create_planning_agent(verbose=True):
    prompt = load_prompt("planning_agent_prompt")

    llm = ChatGoogleGenerativeAI(
        model=config.get('agent', 'model'), 
        convert_system_message_to_human=True, 
        temperature=config.get('agent', 'temperature'), 
        top_p=config.get('agent', 'top_p')
    )
    
    agent = (
        annotate 
        | RunnablePassthrough.assign(
            steps = (
                prompt 
                |llm 
                |JsonOutputParser()
            )
        )
    )
    
    return agent
    
def create_browsing_agent(verbose=True):
    """Create the agent chain"""
    # Load the prompt
    prompt = load_prompt(prompt_name="browser_agent_prompt")
    
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