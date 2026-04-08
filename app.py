import gradio as gr
from env import StockTradingEnv, Action
import random

env = StockTradingEnv()
state = env.reset()

def reset_env():
    global state
    state = env.reset()
    return format_state(state)

def step_env(action_name):
    global state

    action_map = {"HOLD": 0, "BUY": 1, "SELL": 2}
    action = Action(action=action_map[action_name], quantity=1.0)

    result = env.step(action)
    state = result.state

    return format_state(state)

def format_state(state):
    return f"""
Step: {state.step}
Price: {state.price}
Cash: {state.cash}
Shares: {state.shares_held}
Net Worth: {state.net_worth}
Reward: {state.reward}
"""

with gr.Blocks() as demo:
    gr.Markdown("# 📈 Stock Trading Environment")

    output = gr.Textbox(label="Environment State")

    with gr.Row():
        gr.Button("Reset").click(reset_env, outputs=output)
        gr.Button("HOLD").click(lambda: step_env("HOLD"), outputs=output)
        gr.Button("BUY").click(lambda: step_env("BUY"), outputs=output)
        gr.Button("SELL").click(lambda: step_env("SELL"), outputs=output)

demo.launch()