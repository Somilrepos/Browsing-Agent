import base64
import asyncio
from langchain_core.runnables import chain as chain_decorator

from config import config

def load_markup_script():
    """Load the JavaScript for page annotation"""
    with open("resources/mark_page.js") as f:
        return f.read()

mark_page_script = load_markup_script()

@chain_decorator
async def mark_page(page):
    """Mark elements on the page and take a screenshot"""
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            # May be loading...
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

async def annotate(state):
    """Annotate the page with bounding boxes and take a screenshot"""
    marked_page = await mark_page.with_retry().ainvoke(state["page"])
    return {**state, **marked_page}

def format_descriptions(state):
    """Format bounding box descriptions for the agent"""
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        labels.append(f'{i} (<{el_type}/>): "{text}"')
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}

def parse(text: str) -> dict:
    """Parse the LLM output to extract the action and arguments"""
    action_prefix = "Action: "
    if not text.strip().split("\n")[-1].startswith(action_prefix):
        return {"action": "retry", "args": f"Could not parse LLM Output: {text}"}
    action_block = text.strip().split("\n")[-1]

    action_str = action_block[len(action_prefix):]
    split_output = action_str.split(" ", 1)
    if len(split_output) == 1:
        action, action_input = split_output[0], None
    else:
        action, action_input = split_output
    action = action.strip()
    if action_input is not None:
        action_input = [
            inp.strip().strip("[]") for inp in action_input.strip().split(";")
        ]
    return {"action": action, "args": action_input}

def log_stage(name):
    """Utility for logging stages of execution"""
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(lambda x: (print(f"\n🔍 {name}:\n{x}"), x)[1])
