import warnings
warnings.filterwarnings("ignore")

import asyncio
from IPython.display import clear_output, display, Image
import base64
from playwright.async_api import async_playwright

from config import config
from agent import create_browsing_agent, create_planning_agent
from graph import build_graph

async def main():
    # Setup
    _ = config.setup_api_keys()
    config.setup_langchain_tracing()
    
    # Create the agent
    browsing_agent = create_browsing_agent()
    planning_agent = create_planning_agent()
    
    # Build the graph
    graph = build_graph(planning_agent, browsing_agent)
    
    # Start the browser
    browser = await async_playwright().start()
    browser = await browser.chromium.launch(headless=False)
    page = await browser.new_page()
    await page.goto("https://www.google.com")
    
    task = input("Enter the task: ")
    
    # Run the agent
    await run_agent(
        graph, 
        page, 
        task, 
        thread_id=1,
    )
    
    await browser.close()

async def run_agent(graph, page, task, thread_id=1, max_steps=50):
    """Run the agent for the given task"""
    event_stream = graph.astream(
        {
            "page": page,
            "input": task,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
            "configurable": {"thread_id": thread_id},
        },
    )

    final_answer = None
    steps = []

    async for event in event_stream:
        if "agent" not in event:
            continue

        data = event["agent"]

        ### Removed !!!
        # # Handle interruption (e.g., CAPTCHA)
        # if data.get("__type__") == "interrupt":
        #     clear_output(wait=False)
        #     print(data)
        #     print(f"🛑 INTERRUPT: {data.get('message', 'Human input required')}")
        #     if "screenshot" in data:
        #         display(Image(base64.b64decode(data["screenshot"])))

        #     input("👉 Press Enter after completing the required action (e.g. solving CAPTCHA)...")

        #     # Resume the graph
        #     await graph.ainvoke(
        #         {"resume": "User manually completed the task ✅"},
        #         config={"configurable": {"thread_id": thread_id}},
        #     )
        #     continue

        pred = data.get("prediction") or {}
        action = pred.get("action")
        action_input = pred.get("args")
        clear_output(wait=False)
        steps.append(f"{len(steps) + 1}. {action}: {action_input}")
        print("\n".join(steps))

        if "img" in data:
            display(Image(base64.b64decode(data["img"])))

        if action and "ANSWER" in action:
            final_answer = action_input[0] if isinstance(action_input, list) else action_input
            break

    return final_answer

if __name__ == "__main__":
    asyncio.run(main())
