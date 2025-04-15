from langgraph.graph import END, START, StateGraph
#from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableLambda

from states import AgentState
from tools import tools_dict, update_scratchpad, update_steps
from utils import log_stage

def build_graph(planning_agent, browsing_agent):
    """Build the agent execution graph"""
    graph_builder = StateGraph(AgentState)
    
    graph_builder.add_node("planning-agent", planning_agent) #| (lambda steps: {"steps": steps}))# | log_stage("after planning agent calling.."))
    graph_builder.add_node("update_steps", update_steps)
    graph_builder.add_node("browsing-agent", browsing_agent)
    graph_builder.add_node("update_scratchpad", update_scratchpad)
    
    graph_builder.add_edge(START, "planning-agent")
    graph_builder.add_edge("planning-agent", "update_steps")
    graph_builder.add_edge("update_steps", "browsing-agent")
    
    # Add tool nodes
    for node_name, tool in tools_dict.items():
        graph_builder.add_node(
            node_name,
            log_stage(f"Now tool Calling..{node_name}") | RunnableLambda(tool) | (lambda observation: {"observation": observation}),
        )
        graph_builder.add_edge(node_name, "update_scratchpad")
    
    def select_tool(states: AgentState):
        print("Selecting next tool...")
        action = states["prediction"]["action"]
        if action == "ANSWER":
            return END
        if action == "retry":
            return "agent"
        return action
    
    graph_builder.add_conditional_edges("browsing-agent", select_tool)
    
    graph_builder.add_edge("update_scratchpad", "browsing-agent")
    
    #saver = InMemorySaver()
    return graph_builder.compile()#checkpointer=saver)