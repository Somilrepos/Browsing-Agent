import base64
import asyncio
from langchain_core.runnables import chain as chain_decorator
from urllib.parse import urlparse, urlunparse


def load_markup_script():
    """Load the JavaScript for page annotation"""
    with open("/var/home/oliver/Documents/MyCode/browser-use/resources/mark_page.js") as f:
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
    pages = state['browserContext'].pages
    page = pages[state['page_id']]
    marked_page = await mark_page.with_retry().ainvoke(page)
    return {**state, **marked_page, "page_list":[f"<Page_number={i}, url={p.url}>" for i, p in enumerate(pages)]}

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


def canonicalise(raw_url: str) -> str:
    original = raw_url.strip()

    if "://" not in original:
        parsed = urlparse(f"//{original}", scheme="")
    else:
        parsed = urlparse(original)

    host_port = parsed.netloc if parsed.netloc else parsed.path.split("/")[0]
    host = host_port.split(":")[0].lower().rstrip(".")

    if not host.startswith("www.") and host.count(".") == 1:
        host = f"www.{host}"

    # Reconstruct the canonical URL
    scheme = "https"
    netloc = host
    path = parsed.path if parsed.netloc else original[len(host_port):]
    query = parsed.query
    fragment = parsed.fragment

    return urlunparse((scheme, netloc, path or "", "", query, fragment))
