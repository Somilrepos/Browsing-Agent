from typing import TypedDict, Optional, List
from langchain_core.messages import BaseMessage
from playwright.async_api import Page

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
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: list[BaseMessage]
    observation: str