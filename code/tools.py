import asyncio
import platform
import re
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage

from states import AgentState
from utils import canonicalise

async def click(state: AgentState):
    page = state['browserContext'].pages[state['page_id']]
    click_args = state["prediction"]['args']
    if click_args is None or len(click_args) != 1:
        return f"Failed to click. Checkpoint 1"
    
    try:
        bbox_id = int(click_args[0])
        bbox = state['bboxes'][bbox_id]
    except:
        return f"No bbox found for {bbox_id}. Checkpoint 2"
    
    x, y = bbox['x'], bbox['y']
    print(x, y)
    await page.mouse.click(x, y)
    
    return f"Clicked {bbox_id}"

async def type(state: AgentState):
    page = state['browserContext'].pages[state['page_id']]
    type_args = state["prediction"]['args']
    if type_args is None or len(type_args) != 2:
        return f"Failed to Type. Checkpoint 1"
    
    try:
        bbox_id = int(type_args[0])
        bbox = state['bboxes'][bbox_id]
    except:
        return f"No bbox found for {bbox_id}. Checkpoint 2"

    x, y = bbox['x'], bbox['y']
    text = type_args[1]
    
    await page.mouse.click(x, y)
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text)
    await page.keyboard.press("Enter")
    return f"Typed '{text}' and submitted"

async def scroll(state: AgentState):
    page = state['browserContext'].pages[state['page_id']]
    scroll_args = state["prediction"]['args']
    if scroll_args is None or len(scroll_args) != 2:
        return f"Failed to scroll. Checkpoint 1"
    
    target, direction = scroll_args
    if target.upper() == "WINDOW":
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # Scrolling within a specific element
        scroll_amount = 200
        target_id = int(target)
        bbox = state["bboxes"][target_id]
        x, y = bbox["x"], bbox["y"]
        scroll_direction = (
            -scroll_amount if direction.lower() == "up" else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)

    return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s."

async def go_back(state: AgentState):
    page = state['browserContext'].pages[state['page_id']]
    await page.go_back()
    return f"Navigated back a page to {page.url}."

async def to_google(state: AgentState):
    page = state['browserContext'].pages[state['page_id']]
    await page.goto("https://www.google.com/")
    return "Navigated to google.com."

async def goto(state: AgentState):
    context = state['browserContext']
    # print(context)
    pages = context.pages  # FIX: await the coroutine
    # print(pages)
    page = pages[state['page_id']]
    # print(page)
    url = state["prediction"]['args'][0]
    clean_url = canonicalise(url)
    # print(page)
    await page.goto(clean_url)
    return f"Navigated to {clean_url}"

async def switch_page(state: AgentState):
    page_id =  int(state["prediction"]['args'])
    return f"Switched back to page number {page_id}"

async def new_page(state: AgentState):
    context = state['browserContext']
    page = await context.new_page()   
    url = state['prediction']['args'][0]  
    await page.goto(url)               

    pages = context.pages        
    page_id = len(pages) - 1          
    state["page_id"] = page_id
    return f"Opened and switched to the new page number {page_id}"


async def user_solve_captcha(state: AgentState):
    status = input("User is needed to solve the CAPTCHA!")
    if status == "SOLVED":
        return f"User solved the CAPTCHA."
    else: 
        return f"User failed to solve CAPTCHA because {status}"

async def ask_human(state: AgentState):
    question = state['prediction']['args']
    response = input(question)
    return f"Human said '{response}'."

def update_scratchpad(state: AgentState):
    old = state.get("scratchpad")
    print(old)
    if len(old)>1:
        txt = old[1].content
        last_line = txt.rsplit("\n", 1)[-1]
        step = int(re.match(r"\d+", last_line).group()) + 1
    else:
        txt = "Previous action observations:\n"
        step = 1
    txt += f"\n{step}. {state['observation']}"

    return {**state, "scratchpad": [old[0], SystemMessage(content=txt)]}

def update_steps(state: AgentState):
    txt = "Steps: \n"
    for num, descp in state['steps']['steps'].items():
        txt += f"{int(num)}. {descp}\n"
    return {**state, "scratchpad": [SystemMessage(content=txt)]}

tools_dict = {
    "Click": click,
    "Type": type,
    "Scroll": scroll,
    "Wait": wait,
    "Captcha": user_solve_captcha,
    "GoBack": go_back,
    # "Google": to_google,
    "GotoURL": goto,
    "SwitchPage": switch_page,   
    "AskHuman": ask_human, 
    "NewPage": new_page
}
