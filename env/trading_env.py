"""
Stock Trading OpenEnv Environment
Implements step() / reset() / state() API
"""

import random
import math
from typing import Optional
from pydantic import BaseModel


# ── Typed Models ──────────────────────────────────────────────────────────────

class Action(BaseModel):
    action: int          # 0=HOLD, 1=BUY, 2=SELL
    quantity: float = 1.0  # fraction of max shares (0.0–1.0)

class StockState(BaseModel):
    step: int
    price: float
    price_history: list[float]
    portfolio_value: float
    cash: float
    shares_held: int
    max_shares: int
    net_worth: float
    initial_capital: float
    done: bool
    reward: float
    info: dict

class StepResult(BaseModel):
    state: StockState
    reward: float
    done: bool
    info: dict


# ── Price Simulation ──────────────────────────────────────────────────────────

def generate_price_series(steps: int, start_price: float = 100.0, seed: int = 42) -> list[float]:
    """Generate a realistic stock price series using GBM (Geometric Brownian Motion)."""
    rng = random.Random(seed)
    prices = [start_price]
    mu = 0.0002      # drift
    sigma = 0.015    # volatility

    for _ in range(steps):
        z = rng.gauss(0, 1)
        price = prices[-1] * math.exp((mu - 0.5 * sigma ** 2) + sigma * z)
        prices.append(round(max(price, 1.0), 2))

    return prices


# ── Environment ───────────────────────────────────────────────────────────────

class StockTradingEnv:
    """
    A real-world stock trading simulation environment.

    Observation space:
      - Current price, price history (last 10), portfolio stats

    Action space:
      - 0: HOLD
      - 1: BUY  (quantity × max_shares)
      - 2: SELL (quantity × shares_held)

    Reward:
      - Change in net worth per step, normalised by initial capital
    """

    MAX_STEPS = 200
    INITIAL_CAPITAL = 10_000.0
    MAX_SHARES = 50
    TRANSACTION_FEE = 0.001   # 0.1% per trade

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._prices: list[float] = []
        self._step = 0
        self._cash = self.INITIAL_CAPITAL
        self._shares = 0
        self._prev_net_worth = self.INITIAL_CAPITAL
        self._done = False
        self._last_reward = 0.0
        self.reset(seed=seed)

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> StockState:
        if seed is not None:
            self.seed = seed
        self._prices = generate_price_series(self.MAX_STEPS + 10, seed=self.seed)
        self._step = 0
        self._cash = self.INITIAL_CAPITAL
        self._shares = 0
        self._prev_net_worth = self.INITIAL_CAPITAL
        self._done = False
        self._last_reward = 0.0
        return self._get_state()

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise ValueError("Episode is done. Call reset() first.")

        price = self._current_price()
        act = action.action
        qty = max(0.0, min(1.0, action.quantity))

        info = {"action_taken": ["HOLD", "BUY", "SELL"][act], "price": price}

        # Execute action
        if act == 1:  # BUY
            shares_to_buy = int(qty * self.MAX_SHARES)
            cost = shares_to_buy * price * (1 + self.TRANSACTION_FEE)
            if shares_to_buy > 0 and cost <= self._cash:
                self._shares += shares_to_buy
                self._cash -= cost
                info["shares_bought"] = shares_to_buy
            else:
                info["note"] = "Insufficient cash or zero quantity"

        elif act == 2:  # SELL
            shares_to_sell = int(qty * self._shares)
            if shares_to_sell > 0:
                proceeds = shares_to_sell * price * (1 - self.TRANSACTION_FEE)
                self._shares -= shares_to_sell
                self._cash += proceeds
                info["shares_sold"] = shares_to_sell
            else:
                info["note"] = "No shares to sell"

        # Advance time
        self._step += 1
        new_net_worth = self._net_worth()
        reward = (new_net_worth - self._prev_net_worth) / self.INITIAL_CAPITAL
        self._prev_net_worth = new_net_worth
        self._last_reward = reward

        if self._step >= self.MAX_STEPS:
            self._done = True
            info["episode_end"] = "max_steps_reached"

        if new_net_worth <= 0:
            self._done = True
            reward -= 1.0
            info["episode_end"] = "bankrupt"

        state = self._get_state()
        return StepResult(state=state, reward=reward, done=self._done, info=info)

    def state(self) -> StockState:
        return self._get_state()

    # ── Internal ────────────────────────────────────────────────────────────

    def _current_price(self) -> float:
        return self._prices[self._step]

    def _net_worth(self) -> float:
        return round(self._cash + self._shares * self._current_price(), 2)

    def _get_state(self) -> StockState:
        price = self._current_price()
        history_start = max(0, self._step - 9)
        return StockState(
            step=self._step,
            price=price,
            price_history=self._prices[history_start: self._step + 1],
            portfolio_value=round(self._shares * price, 2),
            cash=round(self._cash, 2),
            shares_held=self._shares,
            max_shares=self.MAX_SHARES,
            net_worth=self._net_worth(),
            initial_capital=self.INITIAL_CAPITAL,
            done=self._done,
            reward=round(self._last_reward, 6),
            info={}
        )
