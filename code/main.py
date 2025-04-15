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
    context = await browser.new_context()
    await context.new_page()
    # await page.goto("https://www.google.com")
    
    task = input("Enter the task: ")
    
    # Run the agent
    await run_agent(
        graph, 
        context, 
        task, 
        thread_id=1,
    )
    
    await browser.close()

async def run_agent(graph, context, task, thread_id=1, max_steps=50):
    """Run the agent for the given task"""
    event_stream = graph.astream(
        {
            "browserContext": context,
            "page_id": 0,
            "input": task,
            "scratchpad": [],
        },
        {
            "recursion_limit": max_steps,
            # "configurable": {"thread_id": thread_id},
        },
    )

    final_answer = None
    steps = []

    async for event in event_stream:
        if "agent" not in event:
            continue

        data = event["agent"]
        print(data.get("scratchpad"))
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
            print("Broke at point 1")
            break

    return final_answer

if __name__ == "__main__":
    asyncio.run(main())
