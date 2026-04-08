# 📈 Stock Trading OpenEnv

A complete **real-world OpenEnv environment** for training AI agents to trade stocks.  
Implements the full `step()` / `reset()` / `state()` API spec with FastAPI REST endpoints.

---

## 🗂 Project Structure

```
stock_trading_env/
├── main.py                        # FastAPI app (REST API)
├── openenv.yaml                   # OpenEnv spec
├── Dockerfile                     # Deploy to HF Spaces
├── requirements.txt
├── env/
│   └── trading_env.py             # Core environment (step/reset/state)
├── graders/
│   └── graders.py                 # Easy / Medium / Hard graders
└── scripts/
    └── baseline_inference.py      # Reproducible baseline scores
```

---

## 🚀 Quick Start

### Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🔌 API Reference

### `POST /reset`
Start a new episode.
```json
{ "seed": 42 }
```
Returns `session_id` and initial `state`.

---

### `POST /step`
Take an action.
```json
{
  "session_id": "<uuid>",
  "action": 1,        // 0=HOLD, 1=BUY, 2=SELL
  "quantity": 0.5     // fraction of max_shares
}
```
Returns `state`, `reward`, `done`, `info`.

---

### `GET /state/{session_id}`
Get current observation without acting.

---

### `POST /grade`
Run all three graders and get scores.
```json
{ "seed": 42 }
```

---

### `GET /info`
Environment metadata, action/observation space.

---

## 🎯 Action Space

| Action | Code | Description |
|--------|------|-------------|
| HOLD   | 0    | Do nothing |
| BUY    | 1    | Buy `quantity × max_shares` shares |
| SELL   | 2    | Sell `quantity × shares_held` shares |

`quantity` is a float in `[0.0, 1.0]`.

---

## 👁 Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Current timestep (0–200) |
| `price` | float | Current stock price |
| `price_history` | list[float] | Last 10 prices |
| `cash` | float | Available cash |
| `shares_held` | int | Shares owned |
| `portfolio_value` | float | shares × price |
| `net_worth` | float | cash + portfolio |
| `done` | bool | Episode finished |
| `reward` | float | Last step reward |

---

## 🏆 Graders

| Level | Task | Passing Score |
|-------|------|---------------|
| **Easy** | Survive without going bankrupt | ≥ 0.5 |
| **Medium** | Achieve positive returns | ≥ 0.6 |
| **Hard** | Beat buy-and-hold by ≥ 5% | ≥ 0.75 |

Run graders:
```bash
python -c "from graders.graders import run_all_graders; [print(r) for r in run_all_graders()]"
```

---

## 📊 Reward Function

```
reward_t = (net_worth_t - net_worth_{t-1}) / initial_capital
```

- Dense signal at every step  
- Partial progress: agent gets rewarded for improving incrementally  
- Bankruptcy penalty: −1.0 bonus penalty

---

## 🔁 Reproducible Baseline Scores

```bash
python scripts/baseline_inference.py
```

Expected output (seed=42):

| Strategy | Net Worth | Profit % |
|----------|-----------|----------|
| Hold | 10,000.00 | 0.00% |
| Buy & Hold | ~10,200–10,800 | ~2–8% |
| MA Crossover | ~10,100–10,600 | ~1–6% |

*(Exact values depend on seed-determined price series)*

---

## 🐳 Deploy to Hugging Face Spaces

1. Create a new Space (Docker SDK)
2. Push this repo to the Space's git remote
3. The Dockerfile exposes port **7860** (required by HF Spaces)
4. Add this to your Space's README header:

```yaml
---
title: Stock Trading OpenEnv
sdk: docker
app_port: 7860
---
```

---

## ⚙️ Environment Details

| Parameter | Value |
|-----------|-------|
| Initial Capital | $10,000 |
| Max Steps | 200 |
| Max Shares | 50 |
| Transaction Fee | 0.1% |
| Price Model | Geometric Brownian Motion |
| Termination | Max steps OR Bankruptcy |
