"""
Baseline Inference Script
Runs three deterministic agent strategies and prints reproducible scores.
Usage: python scripts/baseline_inference.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.trading_env import StockTradingEnv, Action

SEEDS = [42, 123, 777]


def run_strategy(name: str, strategy_fn, seed: int) -> dict:
    env = StockTradingEnv(seed=seed)
    env.reset(seed=seed)
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        s = env.state()
        action = strategy_fn(s)
        result = env.step(action)
        total_reward += result.reward
        done = result.done
        steps += 1

    final = env.state()
    profit_pct = (final.net_worth - final.initial_capital) / final.initial_capital * 100
    return {
        "strategy": name,
        "seed": seed,
        "net_worth": final.net_worth,
        "profit_pct": round(profit_pct, 2),
        "total_reward": round(total_reward, 4),
        "steps": steps,
    }


# ── Strategy definitions ──────────────────────────────────────────────────────

def hold_strategy(state) -> Action:
    """Never trade."""
    return Action(action=0, quantity=0.0)


def buy_and_hold_strategy(state) -> Action:
    """Buy 80% at step 0, hold forever."""
    if state.step == 0:
        return Action(action=1, quantity=0.8)
    return Action(action=0, quantity=0.0)


def ma_crossover_strategy(state) -> Action:
    """Short/long moving-average crossover."""
    h = state.price_history
    if len(h) < 5:
        return Action(action=0, quantity=0.0)
    short = sum(h[-3:]) / 3
    long_ = sum(h) / len(h)
    if short > long_ * 1.003 and state.cash > state.price:
        return Action(action=1, quantity=0.25)
    if short < long_ * 0.997 and state.shares_held > 0:
        return Action(action=2, quantity=0.5)
    return Action(action=0, quantity=0.0)


strategies = [
    ("Hold (baseline)",     hold_strategy),
    ("Buy & Hold",          buy_and_hold_strategy),
    ("MA Crossover",        ma_crossover_strategy),
]


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print(f"{'Strategy':<22} {'Seed':>6} {'Net Worth':>12} {'Profit%':>9} {'Reward':>10}")
    print("=" * 65)

    all_results = []
    for name, fn in strategies:
        for seed in SEEDS:
            r = run_strategy(name, fn, seed)
            all_results.append(r)
            print(f"{r['strategy']:<22} {r['seed']:>6} {r['net_worth']:>12.2f} "
                  f"{r['profit_pct']:>8.2f}% {r['total_reward']:>10.4f}")
        print("-" * 65)

    print("\nReproducibility check: run this script multiple times — scores must match.")
    print("All strategies use fixed seeds for deterministic results.\n")
