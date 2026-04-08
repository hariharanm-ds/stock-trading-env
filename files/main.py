"""
Stock Trading OpenEnv — FastAPI REST Server
Endpoints mirror the OpenEnv step() / reset() / state() spec.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uuid

from env.trading_env import StockTradingEnv, Action, StockState, StepResult
from graders.graders import run_all_graders

app = FastAPI(
    title="Stock Trading OpenEnv",
    description="A real-world stock trading RL environment with step/reset/state API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store  {session_id: StockTradingEnv}
_sessions: dict[str, StockTradingEnv] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = 42

class ResetResponse(BaseModel):
    session_id: str
    state: StockState

class StepRequest(BaseModel):
    session_id: str
    action: int        # 0=HOLD, 1=BUY, 2=SELL
    quantity: float = 1.0

class GradeRequest(BaseModel):
    seed: Optional[int] = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_env(session_id: str) -> StockTradingEnv:
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return env


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "env": "StockTradingEnv", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse, summary="Reset environment")
def reset(req: ResetRequest):
    """
    Start a new episode. Returns a session_id and the initial state.
    Pass `seed` for reproducibility.
    """
    session_id = str(uuid.uuid4())
    env = StockTradingEnv(seed=req.seed)
    state = env.reset(seed=req.seed)
    _sessions[session_id] = env
    return ResetResponse(session_id=session_id, state=state)


@app.post("/step", response_model=StepResult, summary="Take an action")
def step(req: StepRequest):
    """
    Take one step in the environment.
    - action: 0=HOLD, 1=BUY, 2=SELL
    - quantity: fraction [0.0–1.0] of max_shares to trade
    """
    env = _get_env(req.session_id)
    if env.state().done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset to start a new one.")
    action = Action(action=req.action, quantity=req.quantity)
    result = env.step(action)
    return result


@app.get("/state/{session_id}", response_model=StockState, summary="Get current state")
def state(session_id: str):
    """
    Get the current observation without taking an action.
    """
    return _get_env(session_id).state()


@app.post("/grade", summary="Run all agent graders")
def grade(req: GradeRequest):
    """
    Run Easy / Medium / Hard graders and return scores [0.0–1.0].
    """
    results = run_all_graders(seed=req.seed)
    return {"seed": req.seed, "results": results}


@app.get("/info", summary="Environment metadata")
def info():
    """
    Returns environment spec: action space, observation space, episode length.
    """
    return {
        "env_id": "StockTradingEnv-v1",
        "action_space": {
            "type": "Discrete+Continuous",
            "actions": {0: "HOLD", 1: "BUY", 2: "SELL"},
            "quantity": "float [0.0, 1.0]"
        },
        "observation_space": {
            "step": "int",
            "price": "float",
            "price_history": "list[float] (last 10 steps)",
            "cash": "float",
            "shares_held": "int",
            "portfolio_value": "float",
            "net_worth": "float",
            "done": "bool",
            "reward": "float"
        },
        "max_steps": StockTradingEnv.MAX_STEPS,
        "initial_capital": StockTradingEnv.INITIAL_CAPITAL,
        "transaction_fee": StockTradingEnv.TRANSACTION_FEE,
        "reward": "Δnet_worth / initial_capital per step",
        "graders": ["easy (survive)", "medium (positive returns)", "hard (beat buy-and-hold)"]
    }


@app.delete("/session/{session_id}", summary="Delete a session")
def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del _sessions[session_id]
    return {"deleted": session_id}
