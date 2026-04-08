"""
Agent Graders — Easy / Medium / Hard
Each grader runs a full episode and returns a score in [0.0, 1.0].
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env.trading_env import StockTradingEnv, Action


# ── Grader Base ───────────────────────────────────────────────────────────────

class BaseGrader:
    name: str = "base"
    description: str = ""

    def run(self, seed: int = 42) -> dict:
        env = StockTradingEnv(seed=seed)
        env.reset(seed=seed)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = self.choose_action(env.state())
            result = env.step(action)
            total_reward += result.reward
            done = result.done
            steps += 1

        final_state = env.state()
        profit_pct = (final_state.net_worth - final_state.initial_capital) / final_state.initial_capital

        score = self.compute_score(profit_pct, total_reward, steps)
        return {
            "grader": self.name,
            "score": round(score, 4),
            "net_worth": final_state.net_worth,
            "profit_pct": round(profit_pct * 100, 2),
            "steps": steps,
            "description": self.description,
        }

    def choose_action(self, state) -> Action:
        raise NotImplementedError

    def compute_score(self, profit_pct: float, total_reward: float, steps: int) -> float:
        raise NotImplementedError


# ── Easy Grader: Did the agent avoid going bankrupt? ─────────────────────────

class EasyGrader(BaseGrader):
    """
    Task: Survive the episode without going bankrupt.
    Score: 1.0 if net_worth > 0 at end, else 0.0.
    Uses a simple BUY-and-HOLD strategy.
    """
    name = "easy"
    description = "Buy-and-hold: survive without going bankrupt."

    def choose_action(self, state) -> Action:
        # Buy once at step 0, hold forever
        if state.step == 0:
            return Action(action=1, quantity=0.5)
        return Action(action=0, quantity=0.0)

    def compute_score(self, profit_pct: float, total_reward: float, steps: int) -> float:
        return 1.0 if profit_pct > -1.0 else 0.0


# ── Medium Grader: Beat buy-and-hold ─────────────────────────────────────────

class MediumGrader(BaseGrader):
    """
    Task: Achieve positive profit (beat inflation baseline of 0%).
    Score: sigmoid-scaled profit percentage.
    Uses a simple moving-average crossover strategy.
    """
    name = "medium"
    description = "Moving-average crossover: must achieve positive returns."

    def choose_action(self, state) -> Action:
        history = state.price_history
        if len(history) < 5:
            return Action(action=0, quantity=0.0)

        short_ma = sum(history[-3:]) / 3
        long_ma = sum(history) / len(history)

        if short_ma > long_ma * 1.005 and state.cash > state.price * 2:
            return Action(action=1, quantity=0.3)
        elif short_ma < long_ma * 0.995 and state.shares_held > 0:
            return Action(action=2, quantity=0.5)
        return Action(action=0, quantity=0.0)

    def compute_score(self, profit_pct: float, total_reward: float, steps: int) -> float:
        if profit_pct <= 0:
            return max(0.0, 0.5 + profit_pct * 2)
        # Scale: 5% profit → 0.75, 15% profit → 1.0
        return min(1.0, 0.5 + profit_pct * 3.3)


# ── Hard Grader: Outperform buy-and-hold by a margin ─────────────────────────

class HardGrader(BaseGrader):
    """
    Task: Outperform a pure buy-and-hold baseline by at least 5%.
    Score: 0.0–1.0 based on alpha generated over the baseline.
    Uses an aggressive momentum strategy.
    """
    name = "hard"
    description = "Momentum strategy: must outperform buy-and-hold by ≥5%."

    def __init__(self):
        self._buy_hold_return = None

    def run(self, seed: int = 42) -> dict:
        # Compute buy-and-hold baseline
        env = StockTradingEnv(seed=seed)
        s = env.reset(seed=seed)
        start_price = s.price
        for _ in range(env.MAX_STEPS - 1):
            env.step(Action(action=0, quantity=0))
        end_state = env.state()
        # Simulate full buy at step 0
        shares = int(0.9 * env.INITIAL_CAPITAL / start_price)
        buy_hold_worth = shares * end_state.price + (env.INITIAL_CAPITAL - shares * start_price)
        self._buy_hold_return = (buy_hold_worth - env.INITIAL_CAPITAL) / env.INITIAL_CAPITAL
        return super().run(seed=seed)

    def choose_action(self, state) -> Action:
        history = state.price_history
        if len(history) < 6:
            return Action(action=1, quantity=0.4)

        momentum = (history[-1] - history[-6]) / history[-6]
        rsi = self._compute_rsi(history)

        if momentum > 0.01 and rsi < 70 and state.cash > state.price:
            return Action(action=1, quantity=0.4)
        elif momentum < -0.01 or rsi > 75:
            return Action(action=2, quantity=0.6)
        return Action(action=0, quantity=0.0)

    def _compute_rsi(self, prices: list, period: int = 6) -> float:
        if len(prices) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(-period, 0):
            diff = prices[i] - prices[i - 1]
            (gains if diff > 0 else losses).append(abs(diff))
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 1e-9
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def compute_score(self, profit_pct: float, total_reward: float, steps: int) -> float:
        baseline = self._buy_hold_return or 0.0
        alpha = profit_pct - baseline
        if alpha < -0.05:
            return 0.0
        elif alpha < 0.05:
            return 0.5 + alpha * 5
        else:
            return min(1.0, 0.75 + alpha * 5)


# ── Run all graders ───────────────────────────────────────────────────────────

def run_all_graders(seed: int = 42) -> list[dict]:
    results = []
    for GraderClass in [EasyGrader, MediumGrader, HardGrader]:
        g = GraderClass()
        results.append(g.run(seed=seed))
    return results


if __name__ == "__main__":
    for r in run_all_graders():
        print(r)
