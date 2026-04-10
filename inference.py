"""
OpenEnv Inference Script - Phase 2 Compliant 
Includes: LLM Proxy, Structured Logging, and Score Clamping.
"""

import os
import sys
from openai import OpenAI

# Ensure we can import your environment modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.trading_env import StockTradingEnv, Action

# 1. INITIALIZE THE CLIENT USING ENVIRONMENT VARIABLES
# The validator automatically provides these when grading. Do NOT hardcode a key.
client = OpenAI(
    base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ.get("API_KEY", "dummy-key")
)

def llama_3_strategy(state) -> Action:
    """
    Sends the trading state to Llama 3 via the Hackathon proxy and gets a decision.
    """
    prompt = (
        f"Context: Stock Trading.\n"
        f"Price: {state.price:.2f}, Cash: {state.cash:.2f}, Shares: {state.shares_held}.\n"
        f"Recent Prices: {state.price_history[-5:]}.\n"
        f"Instruction: Should I BUY, SELL, or HOLD? Reply with only one word."
    )

    try:
        # CALL THE API PROXY
        response = client.chat.completions.create(
            model="meta-llama/llama-3-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a professional financial trading agent. Reply with a single word: BUY, SELL, or HOLD."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.0 # Keeps the model's responses consistent
        )
        
        decision = response.choices[0].message.content.strip().upper()
        
        # TRANSLATE TO ACTION
        if "BUY" in decision and state.cash > state.price:
            return Action(action=1, quantity=0.5)  # Buy with 50% of available capacity
        elif "SELL" in decision and state.shares_held > 0:
            return Action(action=2, quantity=1.0)  # Sell 100% of holdings
            
    except Exception as e:
        # If the proxy times out or fails, log it to stderr and HOLD
        print(f"Proxy Error: {e}", file=sys.stderr)
    
    # Default fallback is HOLD
    return Action(action=0, quantity=0.0)


def run_inference(seed: int):
    # Task ID must not have spaces for the regex parser
    task_id = f"llama3_trading_seed_{seed}"
    
    # REQUIRED [START] BLOCK
    print(f"[START] task={task_id}", flush=True)

    env = StockTradingEnv(seed=seed)
    env.reset(seed=seed)
    done = False
    steps = 0

    while not done:
        state = env.state()
        
        # Get action from the LLM
        action = llama_3_strategy(state)
        
        # Step the environment
        result = env.step(action)
        
        steps += 1
        done = result.done

        # REQUIRED [STEP] BLOCK
        print(f"[STEP] step={steps} reward={result.reward:.4f}", flush=True)

    final = env.state()
    
    # --- THE NEW FIX: SCORE CLAMPING ---
    # Calculate the raw score based on net worth vs initial capital
    raw_score = final.net_worth / final.initial_capital
    
    # Squeeze the score so it is strictly > 0 and < 1 (e.g., 0.9999 instead of 1.0)
    clamped_score = max(0.0001, min(0.9999, raw_score))
    
    # REQUIRED [END] BLOCK
    print(f"[END] task={task_id} score={clamped_score:.4f} steps={steps} final_net_worth={final.net_worth:.2f}", flush=True)


if __name__ == "__main__":
    # The validator usually checks across multiple seeds for consistency
    SEEDS = [42, 123, 777]
    for s in SEEDS:
        run_inference(s)
