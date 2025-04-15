from typing import TypedDict, Optional, List, Dict
from langchain_core.messages import BaseMessage
from playwright.async_api import BrowserContext

class BBox(TypedDict):
    """Bounding box for an interactive web element"""
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str
    
class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    browserContext: BrowserContext
    page_id: int 
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: list[BaseMessage]
    observation: str
    steps: Dict[str, str]
    current_step_number: int